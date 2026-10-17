"""固定官方 pySuStaIn 的训练、交叉验证、稳定性与冻结外部应用。

只包装官方接口，不修改 EM/MCMC。断点按输入/配置/源码哈希验证；
每折可并行，官方 startpoints 始终串行，避免上游默认占用全部核心。
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import math
import os
import pickle
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from .posterior import infer_posterior, sample_indices

UPSTREAM_COMMIT = "708fa22d89c9692dbeafcc2c31a1c8460ced5640"
UPSTREAM_HASHES = {
    "AbstractSustain": "5d0719ea48f04377215ee8c128b4483b7a055411a92d7f3989fa1e10a51b93a3",
    "ZscoreSustain": "4c510a7e10332eea7361f1fe7ff7c8d66961843bf6993850c22b0027eda9ef66",
}
META_COLUMNS = ("subject_key", "participant_id", "cohort", "age", "sex", "icv_ml")


# ---------- 原子输出、来源身份与进程互斥 ----------
def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def _table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temp, sep="\t", index=False)
    os.replace(temp, path)


def _npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temp, path)


@contextlib.contextmanager
def _lock(directory: Path) -> Iterator[None]:
    """Linux 文件锁随进程退出释放；保留锁文件不会阻碍中断后的恢复。"""
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"已有进程正在写入 {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _guard(directory: Path, identity: dict[str, Any]) -> None:
    path = directory / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != identity:
            raise ValueError(f"缓存身份与当前输入/参数不同，请使用新的输出目录：{directory}")
    else:
        if any(p.name != ".lock" for p in directory.iterdir()):
            raise ValueError(f"存在没有 manifest 的模型结果，拒绝复用：{directory}")
        _json(path, identity)


def _identity(cfg: dict, output: Path, files: list[Path], role: str) -> dict:
    required = [output / "inputs" / "feature_dictionary.tsv"]
    reference_model = output / "reference" / "model.json"
    reference = json.loads(reference_model.read_text())
    reference_hash = reference.get("reference_hash")
    if reference_hash is None:
        if cfg.get("runtime", {}).get("profile", "formal") == "formal":
            raise ValueError("正式 LS 参考模型缺少独立 reference_hash。")
        reference_hash = _hash(reference_model)  # 独立合成测试夹具兼容。
    identity = {
        "role": role, "schema_version": 1, "upstream_commit": UPSTREAM_COMMIT,
        "reference_hash": reference_hash,
        "upstream_source_sha256": _upstream_provenance(),
        "dependency_versions": {"python": ".".join(str(v) for v in sys.version_info[:3]),
                                **{name: version(name) for name in ("numpy", "scipy", "pandas", "scikit-learn", "pathos", "dill", "multiprocess", "threadpoolctl")}},
        "wrapper_source_sha256": {name: _hash(Path(__file__).with_name(name)) for name in ("model.py", "posterior.py")},
        "model_config": cfg["model"], "runtime_config": cfg.get("runtime", {}),
        "reference_config": cfg.get("reference", {}),
        "input_sha256": {str(p.relative_to(output)): _hash(p) for p in files + required},
    }
    # 训练身份不引用全三队列快照，GB 的变化不会改变 C3 训练身份。
    if role == "GB_frozen_validation":
        reference_manifest = output / "reference" / "manifest.json"
        manifest = json.loads(reference_manifest.read_text()) if reference_manifest.exists() else {}
        identity["transform_hash"] = manifest.get("transform_hash")
        identity["input_snapshot_hash"] = manifest.get("input_snapshot_hash")
    return identity


# ---------- 官方实现校验及输入合同 ----------
def _upstream_provenance() -> dict[str, str]:
    found = {}
    for module_name, expected in UPSTREAM_HASHES.items():
        module = importlib.import_module(f"pySuStaIn.{module_name}")
        actual = _hash(Path(module.__file__))
        if actual != expected:
            raise RuntimeError(f"官方 {module_name}.py 与冻结 commit 不符；请按锁定安装器重装。")
        found[module_name] = actual
    return found


def _features(output: Path) -> pd.DataFrame:
    frame = pd.read_csv(output / "inputs" / "feature_dictionary.tsv", sep="\t", dtype={"feature_id": str})
    required = {"feature_id", "label", "modality", "atlas_index"}
    if not required.issubset(frame.columns) or frame.feature_id.duplicated().any() or frame.empty:
        raise ValueError("feature_dictionary.tsv 字段不完整或特征 ID 重复。")
    return frame


def _read_z(path: Path, features: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path, sep="\t", dtype={"subject_key": str, "participant_id": str})
    needed = list(META_COLUMNS) + features.feature_id.tolist()
    missing = [name for name in needed if name not in frame]
    if missing:
        raise ValueError(f"{path.name} 缺少列：{missing}")
    if frame.empty or frame.subject_key.isna().any() or frame.subject_key.duplicated().any():
        raise ValueError(f"{path.name} 为空或 subject_key 无效/重复。")
    if frame[["cohort", "participant_id"]].duplicated().any():
        raise ValueError("同一队列存在重复患者；须在上游确定唯一记录以避免 CV 泄漏。")
    data = frame[features.feature_id].to_numpy(dtype=float)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path.name} 含非有限 z，禁止填补或静默删行。")
    return frame, data


def _settings(cfg: dict, n_features: int) -> dict:
    settings = dict(cfg["model"])
    settings["reference_age_min"] = float(cfg.get("reference", {}).get("age_min", 50))
    settings["reference_age_max"] = float(cfg.get("reference", {}).get("age_max", 75))
    for name in ("k_max", "folds", "startpoints", "cv_iterations", "fit_iterations", "stability_iterations", "posterior_samples", "workers"):
        if int(settings[name]) < 1:
            raise ValueError(f"model.{name} 必须为正整数。")
    thresholds = np.asarray(settings["z_vals"], dtype=float)
    if thresholds.ndim != 1 or not np.all(np.isfinite(thresholds)) or np.any(thresholds <= 0) or np.any(np.diff(thresholds) <= 0):
        raise ValueError("主分析 z_vals 必须是严格递增的有限正值。")
    if not np.isfinite(settings["z_max"]) or settings["z_max"] <= thresholds[-1]:
        raise ValueError("z_max 必须严格大于最大事件阈值。")
    if cfg.get("runtime", {}).get("profile", "formal") == "formal" and n_features != 40:
        raise ValueError("正式分析合同要求 20 WMH + 20 GM 共 40 个特征。")
    return settings


def _assert_primary(frame: pd.DataFrame, settings: dict, cohort: str) -> None:
    ages = pd.to_numeric(frame.age, errors="coerce")
    if not ages.between(settings["reference_age_min"], settings["reference_age_max"], inclusive="both").all():
        raise ValueError("主分析表包含参考年龄范围之外或缺失年龄的记录。")
    if set(frame.cohort) != {cohort}:
        raise ValueError(f"队列表身份不符，应为 {cohort}。")


def _engine(data: np.ndarray, features: pd.DataFrame, settings: dict, directory: Path,
            seed: int, iterations: int, n_subtypes: int):
    from pySuStaIn.ZscoreSustain import ZscoreSustain

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "pickle_files").mkdir(exist_ok=True)
    return ZscoreSustain(
        data, np.tile(settings["z_vals"], (data.shape[1], 1)),
        np.repeat(settings["z_max"], data.shape[1]), features.label.tolist(),
        int(settings["startpoints"]), int(n_subtypes), int(iterations),
        str(directory), "C3", False, seed=int(seed),
    )


def _model_seed(seed: int, fold: int, n_subtypes: int) -> int:
    """每个模型独立随机流；断点跳过先前 K 不会改变后续模型的随机流。"""
    return int(np.random.SeedSequence([seed, fold + 1, n_subtypes]).generate_state(1)[0])


# ---------- 资源限额：外层并行、内层串行，预先估计工作集 ----------
def _available_memory() -> int:
    mem = 0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                mem = int(line.split()[1]) * 1024
                break
    except OSError:
        mem = int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    for root, maximum, current in (("/sys/fs/cgroup", "memory.max", "memory.current"),
                                   ("/sys/fs/cgroup/memory", "memory.limit_in_bytes", "memory.usage_in_bytes")):
        try:
            limit = (Path(root) / maximum).read_text().strip()
            if limit != "max":
                remaining = int(limit) - int((Path(root) / current).read_text())
                if remaining > 0:
                    mem = min(mem, remaining)
        except (OSError, ValueError):
            continue
    return mem


def estimate_working_bytes(n_rows: int, n_features: int, n_events: int, n_subtypes: int,
                           iterations: int, safety_factor: float = 2.0) -> int:
    """涵盖官方候选位置张量、似然中间数组、MCMC 全链的保守估算。"""
    count = (2 * n_rows * (n_events + 1) * n_events
             + 4 * n_rows * n_features * (n_events + 1)
             + n_subtypes * n_events * iterations
             + 4 * n_rows * (n_events + 1) * n_subtypes)
    return math.ceil(8 * count * safety_factor)


def _resources(cfg: dict, directory: Path, n_rows: int, n_features: int,
               iterations: int, n_subtypes: int, jobs: int, copies: float) -> dict:
    runtime = cfg.get("runtime", {})
    fraction = float(runtime.get("max_memory_fraction", 0.5))
    factor = float(runtime.get("memory_safety_factor", 2.0))
    if not 0 < fraction <= 1 or factor < 1:
        raise ValueError("内存比例必须属于 (0,1]，安全倍数不能小于 1。")
    available = _available_memory()
    if runtime.get("memory_limit_gb") is not None:
        available = min(available, int(float(runtime["memory_limit_gb"]) * 1024 ** 3))
    budget = int(available * fraction)
    n_events = n_features * len(cfg["model"]["z_vals"])
    estimate = estimate_working_bytes(n_rows, n_features, n_events, n_subtypes, iterations, factor)
    workers = min(jobs, int(cfg["model"]["workers"]), max(1, os.cpu_count() or 1), budget // estimate)
    if workers < 1:
        raise MemoryError(f"单个模型预计工作集 {estimate / 1024**3:.2f} GiB 超过内存预算 {budget / 1024**3:.2f} GiB。")
    # 全候选模型均留链；重复种子及折数明确计入，而非只估计选定 K。
    disk = int(copies * 8 * iterations * n_events * n_subtypes * (n_subtypes + 1) / 2
               * float(runtime.get("disk_safety_factor", 1.2)))
    free = shutil.disk_usage(directory).free
    existing = sum(path.stat().st_size for path in directory.rglob("*.pickle"))
    additional = max(0, disk - existing)
    if free < additional:
        raise OSError(f"保留完整 MCMC 链预计还需要 {additional / 1024**3:.2f} GiB，输出盘剩余不足。")
    return {"workers": int(workers), "estimated_worker_bytes": estimate, "memory_budget_bytes": budget,
            "estimated_chain_disk_bytes": disk, "additional_chain_disk_bytes": additional, "disk_free_bytes": free,
            "parallel_startpoints": False, "blas_threads": int(runtime.get("blas_threads", 1))}


def _pickle(path: Path, n_subtypes: int, n_events: int, iterations: int) -> dict:
    with path.open("rb") as stream:
        value = pickle.load(stream)
    if np.asarray(value["samples_sequence"]).shape != (n_subtypes, n_events, iterations):
        raise ValueError(f"不完整或参数不符的 MCMC 缓存：{path}")
    if np.asarray(value["samples_f"]).shape != (n_subtypes, iterations):
        raise ValueError(f"亚型比例缓存维度无效：{path}")
    if not np.isfinite(value["samples_likelihood"]).all():
        raise ValueError(f"MCMC 似然非有限：{path}")
    return value


def _audit_pickles(directory: Path, settings: dict, n_features: int, iterations: int, prefix: str) -> None:
    """不完整文件保留为中断证据，官方仅重新计算未完成的那个模型。"""
    for k in range(1, int(settings["k_max"]) + 1):
        path = directory / "pickle_files" / f"{prefix}_subtype{k - 1}.pickle"
        if path.exists():
            try:
                _pickle(path, k, n_features * len(settings["z_vals"]), iterations)
            except (EOFError, pickle.UnpicklingError, KeyError, ValueError):
                path.rename(path.with_name(path.name + f".incomplete.{time.time_ns()}"))


# ---------- C3 十折选择：CVIC 按官方公式汇总，不读取 GB ----------
def _fold_worker(payload: dict) -> dict:
    output = Path(payload["output"])
    cfg, fold, splits = payload["cfg"], payload["fold"], payload["splits"]
    features = _features(output)
    _, data = _read_z(output / "reference" / "C3_primary_z.tsv", features)
    settings = _settings(cfg, len(features))
    directory = output / "model" / "cv" / f"fold_{fold}"
    with _lock(directory):
        identity = {"parent": payload["identity"], "fold": fold, "indices": splits[fold]}
        _guard(directory, identity)
        marker = directory / "complete.json"
        if marker.exists():
            try:
                for k in range(1, int(settings["k_max"]) + 1):
                    _pickle(directory / "pickle_files" / f"C3_fold{fold}_subtype{k - 1}.pickle", k,
                            len(features) * len(settings["z_vals"]), settings["cv_iterations"])
                return json.loads(marker.read_text())
            except (FileNotFoundError, EOFError, pickle.UnpicklingError, KeyError, ValueError):
                marker.rename(marker.with_name(marker.name + f".incomplete.{time.time_ns()}"))
        _audit_pickles(directory, settings, len(features), settings["cv_iterations"], f"C3_fold{fold}")
        with threadpool_limits(limits=int(cfg.get("runtime", {}).get("blas_threads", 1))):
            test_indices = [np.asarray(indices, dtype=np.int64) for indices in splits]
            for k in range(1, int(settings["k_max"]) + 1):
                engine = _engine(data, features, settings, directory,
                                 _model_seed(int(settings["seed"]), fold, k), settings["cv_iterations"], k)
                expected = directory / "pickle_files" / f"C3_fold{fold}_subtype{k - 1}.pickle"
                if not expected.exists():
                    engine.cross_validate_sustain_model(test_indices, select_fold=[fold], plot=False)
            from pySuStaIn.ZscoreSustain import ZScoreSustainData
            test_data = ZScoreSustainData(data[test_indices[fold]], data.shape[1] * len(settings["z_vals"]))
            rows = []
            for k in range(1, int(settings["k_max"]) + 1):
                model = _pickle(directory / "pickle_files" / f"C3_fold{fold}_subtype{k - 1}.pickle",
                                k, test_data.getNumStages(), settings["cv_iterations"])
                # 官方 partial-fold API 不返回汇总似然；从同一冻结链精确重算。
                likelihoods = engine._evaluate_likelihood_setofsamples(test_data, model["samples_sequence"], model["samples_f"])
                mean_likelihood = np.asarray(model["mean_likelihood_subj_test"])
                if not np.all(np.isfinite(mean_likelihood)) or np.any(mean_likelihood <= 0):
                    raise FloatingPointError("验证折出现零或非有限似然，不能产生有效 CVIC。")
                rows.append({"fold": fold, "n_subtypes": k,
                             "log_likelihood": float(np.mean(np.log(likelihoods + 1e-250).sum(axis=0))),
                             "cvic_component": float(-2 * np.log(mean_likelihood).sum()),
                             "n_test": len(test_indices[fold])})
        _table(directory / "metrics.tsv", pd.DataFrame(rows))
        result = {"fold": fold, "status": "complete", "metrics": rows}
        _json(marker, result)
        return result


def run_crossvalidate(cfg: dict, output: Path) -> dict:
    output = Path(output)
    features = _features(output)
    frame, data = _read_z(output / "reference" / "C3_primary_z.tsv", features)
    settings = _settings(cfg, len(features))
    _assert_primary(frame, settings, "C3")
    n_folds = int(settings["folds"])
    if n_folds < 2 or n_folds > len(data) or len(data) - math.ceil(len(data) / n_folds) < 2:
        raise ValueError("交叉验证必须至少两折，且每折训练样本不少于两例。")
    directory = output / "model" / "cv"
    with _lock(directory):
        identity = _identity(cfg, output, [output / "reference" / "C3_primary_z.tsv"], "C3_crossvalidation")
        _guard(directory, identity)
        resource = _resources(cfg, directory, len(data), len(features), settings["cv_iterations"], settings["k_max"], n_folds, n_folds)
        _json(directory / "resources.json", resource)
        splits = [test.tolist() for _, test in KFold(n_splits=n_folds, shuffle=True, random_state=int(settings["seed"])).split(data)]
        _json(directory / "folds.json", [{"fold": i, "indices": test,
                                         "subject_keys": frame.iloc[test].subject_key.tolist(),
                                         "model_seeds": {str(k): _model_seed(int(settings["seed"]), i, k)
                                                         for k in range(1, int(settings["k_max"]) + 1)}}
                                        for i, test in enumerate(splits)])
        jobs = [{"output": str(output), "cfg": cfg, "fold": fold, "splits": splits, "identity": identity} for fold in range(n_folds)]
        if resource["workers"] == 1:
            results = [_fold_worker(job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=resource["workers"]) as executor:
                results = list(executor.map(_fold_worker, jobs))
        metrics = pd.DataFrame([row for result in results for row in result["metrics"]])
        _table(directory / "fold_loglikelihood.tsv", metrics)
        cvic = metrics.groupby("n_subtypes", as_index=False).agg(cvic=("cvic_component", "sum"), mean_log_likelihood=("log_likelihood", "mean"))
        chosen = int(cvic.sort_values(["cvic", "n_subtypes"]).iloc[0].n_subtypes)
        cvic["selected"] = cvic.n_subtypes == chosen
        _table(directory / "cvic.tsv", cvic)
        selection = {"n_subtypes": chosen, "criterion": "minimum_CVIC", "tie_rule": "smaller_K",
                     "selection_status": "boundary_optimum" if chosen == settings["k_max"] else "interior_optimum",
                     "candidate_k": list(range(1, int(settings["k_max"]) + 1)), "n_folds": n_folds,
                     "scope": "C3 age 50-75 only; GB is never used for selection",
                     "clinical_validation": "not_in_scope"}
        _json(directory / "selection.json", selection)
        return selection


# ---------- 全数据模型、展示样本与个体推断 ----------
def _validated_selection(cfg: dict, output: Path) -> dict:
    """拟合前核对 CV 与当前科学输入，不能把旧 CV 选择套到新阈值。"""
    directory = output / "model" / "cv"
    saved = json.loads((directory / "manifest.json").read_text())
    current = _identity(cfg, output, [output / "reference" / "C3_primary_z.tsv"], "C3_crossvalidation")
    for key in ("role", "schema_version", "upstream_commit", "reference_hash", "upstream_source_sha256",
                "wrapper_source_sha256", "dependency_versions", "reference_config", "input_sha256"):
        if saved.get(key) != current.get(key):
            raise ValueError(f"当前拟合与已完成 CV 身份不符（{key}），必须重新交叉验证。")
    # 允许仅改变最终链、稳定性和后验展示参数；CV 实际使用的参数必须相同。
    for key in ("k_max", "folds", "startpoints", "cv_iterations", "seed", "z_vals", "z_max"):
        if saved["model_config"].get(key) != cfg["model"].get(key):
            raise ValueError(f"model.{key} 与已完成 CV 不同，不能复用旧选型。")
    selection = json.loads((directory / "selection.json").read_text())
    cvic = pd.read_csv(directory / "cvic.tsv", sep="\t")
    expected_k = list(range(1, int(cfg["model"]["k_max"]) + 1))
    if (sorted(cvic.n_subtypes.tolist()) != expected_k or not np.isfinite(cvic.cvic).all()
            or selection.get("candidate_k") != expected_k
            or selection.get("n_folds") != cfg["model"]["folds"]):
        raise ValueError("CVIC/选型记录不完整或与候选模型合同不符。")
    winner = int(cvic.sort_values(["cvic", "n_subtypes"]).iloc[0].n_subtypes)
    if int(selection["n_subtypes"]) != winner:
        raise ValueError("selection.json 与最小 CVIC 不一致，拒绝擅自改变选型。")
    for fold in range(int(cfg["model"]["folds"])):
        marker = directory / f"fold_{fold}" / "complete.json"
        selected_pickle = directory / f"fold_{fold}" / "pickle_files" / f"C3_fold{fold}_subtype{winner - 1}.pickle"
        if not marker.is_file() or not selected_pickle.is_file():
            raise ValueError(f"CV 第 {fold} 折未完成或缺少选定模型链。")
    return selection


def _inference(engine, frame: pd.DataFrame, data: np.ndarray, model: dict,
               settings: dict, destination: Path, tag: str) -> pd.DataFrame:
    posterior = infer_posterior(engine, data, model["samples_sequence"], model["samples_f"],
                                n_samples=int(settings["posterior_samples"]),
                                burn_in_fraction=float(settings.get("burn_in_fraction", 0.1)),
                                batch_size=int(settings.get("posterior_batch_size", 128)))
    _npz(destination / f"posterior_{tag}.npz", **posterior, subject_keys=frame.subject_key.to_numpy(dtype=str))
    columns = list(META_COLUMNS)
    assigned = frame[columns].copy()
    for name in ("subtype_raw_zero_based", "subtype", "stage", "stage_zero", "subtype_probability",
                 "stage_conditional_probability", "stage_zero_probability", "subtype_entropy", "stage_expected"):
        assigned[name] = posterior[name]
    assigned["n_subtypes"] = model["samples_sequence"].shape[0]
    assigned["subtype_name"] = np.where(assigned.stage_zero, "Undifferentiated", "Subtype " + assigned.subtype.astype(str))
    assigned["age_scope"] = np.where(assigned.age.between(settings["reference_age_min"], settings["reference_age_max"], inclusive="both"), "within_reference_range", "outside_reference_range")
    _table(destination / f"assignments_{tag}.tsv", assigned)
    return assigned


def _display(model: dict, settings: dict) -> dict[str, np.ndarray]:
    indices = sample_indices(model["samples_sequence"].shape[2], int(settings.get("display_samples", 1000)),
                             float(settings.get("burn_in_fraction", 0.1)))
    # 顺序与个体后验使用同一 posterior 抽样规则确定，而不受展示抽样数影响。
    posterior_indices = sample_indices(model["samples_sequence"].shape[2], int(settings["posterior_samples"]),
                                       float(settings.get("burn_in_fraction", 0.1)))
    order = np.argsort(model["samples_f"][:, posterior_indices].mean(axis=1))[::-1]
    return {"samples_sequence": model["samples_sequence"][:, :, indices].astype(np.int32),
            "samples_f": model["samples_f"][:, indices],
            "samples_likelihood": np.asarray(model["samples_likelihood"])[indices],
            "subtype_order": order, "sample_indices": indices,
            "posterior_sample_indices": posterior_indices}


def _match(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按事件平均位置的 Kendall tau 一对一匹配，不使用外部 GB。"""
    ref_positions = np.argsort(reference, axis=1).mean(axis=2)
    candidate_positions = np.argsort(candidate, axis=1).mean(axis=2)
    correlations = np.empty((len(reference), len(candidate)))
    for i in range(len(reference)):
        for j in range(len(candidate)):
            correlations[i, j] = kendalltau(ref_positions[i], candidate_positions[j]).statistic
    if not np.all(np.isfinite(correlations)):
        raise ValueError("无法计算事件顺序相关性，至少需要两个有效事件。")
    rows, columns = linear_sum_assignment(-correlations)
    order = columns[np.argsort(rows)]
    return order, correlations[np.arange(len(reference)), order]


def _cv_display(output: Path, primary: dict, settings: dict, n_subtypes: int) -> None:
    sequence_parts, fraction_parts, fold_parts, rows = [], [], [], []
    reference = _display(primary, settings)
    ref_order = reference["subtype_order"]
    reference_sequence = reference["samples_sequence"][ref_order]
    for fold in range(int(settings["folds"])):
        path = output / "model" / "cv" / f"fold_{fold}" / "pickle_files" / f"C3_fold{fold}_subtype{n_subtypes - 1}.pickle"
        model = _pickle(path, n_subtypes, primary["samples_sequence"].shape[1], settings["cv_iterations"])
        candidate = _display(model, settings)
        order, correlations = _match(reference_sequence, candidate["samples_sequence"])
        sequence_parts.append(candidate["samples_sequence"][order])
        fraction_parts.append(candidate["samples_f"][order])
        fold_parts.extend([fold] * candidate["samples_f"].shape[1])
        for subtype in range(n_subtypes):
            rows.append({"fold": fold, "subtype": subtype + 1, "matched_subtype_raw": int(order[subtype]),
                         "kendall_tau": float(correlations[subtype]),
                         "fraction": float(candidate["samples_f"][order[subtype]].mean())})
    destination = output / "model" / "fit"
    features = _features(output)
    _table(destination / "cv_stability.tsv", pd.DataFrame(rows))
    _npz(destination / "cv_display.npz", samples_sequence=np.concatenate(sequence_parts, axis=2),
         samples_f=np.concatenate(fraction_parts, axis=1), fold_ids=np.asarray(fold_parts),
         subtype_order=np.arange(n_subtypes), Z_vals=np.tile(settings["z_vals"], (len(features), 1)),
         Z_max=np.repeat(settings["z_max"], len(features)), feature_ids=features.feature_id.to_numpy(dtype=str),
         labels=features.label.to_numpy(dtype=str))


def run_fit(cfg: dict, output: Path) -> dict:
    output = Path(output)
    selection_path = output / "model" / "cv" / "selection.json"
    selection = _validated_selection(cfg, output)
    features = _features(output)
    frame, data = _read_z(output / "reference" / "C3_primary_z.tsv", features)
    settings = _settings(cfg, len(features))
    _assert_primary(frame, settings, "C3")
    chosen = int(selection["n_subtypes"])
    directory = output / "model" / "fit"
    with _lock(directory):
        identity = _identity(cfg, output, [selection_path, output / "reference" / "C3_primary_z.tsv",
                                          output / "reference" / "C3_z.tsv"], "C3_final_fit")
        _guard(directory, identity)
        seeds = [int(settings["seed"])] + [int(seed) for seed in settings.get("stability_seeds", [])]
        if len(set(seeds)) != len(seeds):
            raise ValueError("主随机种子和稳定性种子必须互不重复。")
        weighted_copies = 1 + (len(seeds) - 1) * settings["stability_iterations"] / settings["fit_iterations"]
        _json(directory / "resources.json", _resources(cfg, directory, len(data), len(features), settings["fit_iterations"], chosen, 1, weighted_copies))
        _json(directory / "model_seeds.json", {str(seed): {str(k): _model_seed(seed, -1, k)
                                                           for k in range(1, chosen + 1)} for seed in seeds})
        fitted = []
        with threadpool_limits(limits=int(cfg.get("runtime", {}).get("blas_threads", 1))):
            for i, seed in enumerate(seeds):
                location = directory / ("primary" if i == 0 else f"seed_{seed}")
                iterations = settings["fit_iterations"] if i == 0 else settings["stability_iterations"]
                with _lock(location):
                    _guard(location, {"parent": identity, "seed": seed, "iterations": iterations, "n_subtypes": chosen})
                    this_settings = {**settings, "k_max": chosen}
                    _audit_pickles(location, this_settings, len(features), iterations, "C3")
                    for k in range(1, chosen + 1):
                        engine = _engine(data, features, this_settings, location, _model_seed(seed, -1, k), iterations, k)
                        expected = location / "pickle_files" / f"C3_subtype{k - 1}.pickle"
                        if not expected.exists():
                            engine.run_sustain_algorithm(plot=False)
                    model = _pickle(location / "pickle_files" / f"C3_subtype{chosen - 1}.pickle", chosen,
                                    len(features) * len(settings["z_vals"]), iterations)
                    fitted.append((seed, model, engine))
            seed, primary, engine = fitted[0]
            display = _display(primary, settings)
            _npz(directory / "selected_model.npz", **display,
                 Z_vals=np.tile(settings["z_vals"], (len(features), 1)), Z_max=np.repeat(settings["z_max"], len(features)),
                 feature_ids=features.feature_id.to_numpy(dtype=str), labels=features.label.to_numpy(dtype=str))
            primary_assignments = _inference(engine, frame, data, primary, settings, directory, "C3_primary")
            all_frame, all_data = _read_z(output / "reference" / "C3_z.tsv", features)
            _inference(engine, all_frame, all_data, primary, settings, directory, "C3_all")
            rows = []
            reference_seq = display["samples_sequence"][display["subtype_order"]]
            reference_f = display["samples_f"][display["subtype_order"]].mean(axis=1)
            for other_seed, model, other_engine in fitted[1:]:
                candidate = _display(model, settings)
                order, correlations = _match(reference_seq, candidate["samples_sequence"])
                secondary = _inference(other_engine, frame, data, model, settings, directory / f"seed_{other_seed}", "C3_primary")
                ari = adjusted_rand_score(primary_assignments.subtype_raw_zero_based, secondary.subtype_raw_zero_based)
                for subtype in range(chosen):
                    fraction = float(candidate["samples_f"][order[subtype]].mean())
                    rows.append({"seed": other_seed, "subtype": subtype + 1, "matched_subtype": int(order[subtype]) + 1,
                                 "kendall_tau": float(correlations[subtype]), "fraction": fraction,
                                 "fraction_difference": fraction - float(reference_f[subtype]), "subtype_ari": float(ari)})
            _table(directory / "stability.tsv", pd.DataFrame(rows, columns=["seed", "subtype", "matched_subtype", "kendall_tau", "fraction", "fraction_difference", "subtype_ari"]))
            _cv_display(output, primary, settings, chosen)
        summary = {**selection, "status": "complete", "primary_seed_rule": "predeclared_seed_not_selected_by_GB",
                   "seed": seed, "stability_seeds": seeds[1:], "n_events": len(features) * len(settings["z_vals"]),
                   "n_features": len(features), "stage_max": len(features) * len(settings["z_vals"]), "n_subjects": len(data),
                   "iterations": int(settings["fit_iterations"]), "burn_in_fraction": settings.get("burn_in_fraction", 0.1),
                   "burn_in_scope": "derived posterior and display only; official CVIC preserves upstream sampling",
                   "single_subject_compatibility": "official inference temporarily repeats one row then retains only the original row; no training duplication",
                   "stability_interpretation": "descriptive; no arbitrary universal pass threshold",
                   "sensitivity": "frozen model application to all ages; no outside-reference refit",
                   "full_chains": "official pickle_files retained for every fitted K and seed"}
        _json(directory / "selection.json", selection)
        _json(directory / "summary.json", summary)
        return summary


# ---------- 冻结 GB 验证及全龄应用敏感性，不参与训练/选型 ----------
def run_validate(cfg: dict, output: Path) -> dict:
    output = Path(output)
    summary_path = output / "model" / "fit" / "summary.json"
    fit_summary = json.loads(summary_path.read_text())
    if fit_summary.get("status") != "complete":
        raise ValueError("主模型尚未完成，不能开始冻结外部应用。")
    features = _features(output)
    settings = _settings(cfg, len(features))
    chosen = int(fit_summary["n_subtypes"])
    fit_manifest = json.loads((output / "model" / "fit" / "manifest.json").read_text())
    if fit_manifest["model_config"] != cfg["model"]:
        raise ValueError("GB 应用参数必须与冻结 C3 模型完全相同。")
    reference_model_path = output / "reference" / "model.json"
    reference_hash = json.loads(reference_model_path.read_text()).get("reference_hash", _hash(reference_model_path))
    if fit_manifest["reference_hash"] != reference_hash:
        raise ValueError("LS 参考模型已改变，拒绝应用旧 C3 模型。")
    for source in ("inputs/feature_dictionary.tsv",):
        if fit_manifest["input_sha256"][source] != _hash(output / source):
            raise ValueError("参考模型或特征字典已改变，拒绝应用旧 C3 模型。")
    primary_path = output / "model" / "fit" / "primary" / "pickle_files" / f"C3_subtype{chosen - 1}.pickle"
    directory = output / "validation"
    with _lock(directory):
        identity = _identity(cfg, output, [summary_path, primary_path, output / "reference" / "GB_primary_z.tsv",
                                          output / "reference" / "GB_z.tsv"], "GB_frozen_validation")
        for key in ("upstream_commit", "upstream_source_sha256", "wrapper_source_sha256", "dependency_versions"):
            if fit_manifest.get(key) != identity.get(key):
                raise ValueError(f"当前验证实现与冻结主模型不同（{key}），请使用原锁定环境及源码。")
        _guard(directory, identity)
        model = _pickle(primary_path, chosen, len(features) * len(settings["z_vals"]), settings["fit_iterations"])
        primary_frame, primary_data = _read_z(output / "reference" / "GB_primary_z.tsv", features)
        _assert_primary(primary_frame, settings, "GB")
        # 创建计算器只需要冻结的事件合同；不会调用任何拟合函数。
        with threadpool_limits(limits=int(cfg.get("runtime", {}).get("blas_threads", 1))):
            engine = _engine(primary_data[:1], features, settings, directory / "calculator", settings["seed"], settings["fit_iterations"], chosen)
            primary_assignment = _inference(engine, primary_frame, primary_data, model, settings, directory, "GB_primary")
            all_frame, all_data = _read_z(output / "reference" / "GB_z.tsv", features)
            _inference(engine, all_frame, all_data, model, settings, directory, "GB_all")
        summary = {"status": "complete", "n_subtypes": chosen, "n_primary": len(primary_data), "n_all": len(all_data),
                   "n_stage_zero": int(primary_assignment.stage_zero.sum()), "model_frozen": True,
                   "reference_frozen": True, "gb_used_for_selection": False,
                   "scope": "external subtype/stage application and uncertainty; no clinical outcomes",
                   "sensitivity": "all ages use identical model; outside 50-75 flagged as extrapolative"}
        _json(directory / "summary.json", summary)
        return summary
