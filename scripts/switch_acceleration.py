#!/usr/bin/env python3
"""保留结果并迁移已核验的CV执行身份；精准停止旧协调器/worker后后台续跑。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
from datetime import datetime
import hashlib
import os
from pathlib import Path
import pickle
import subprocess
import sys
import uuid

import psutil
import yaml

from substain_sustain.config import load_config, validate_config
from substain_sustain.acceleration import execution_settings
import switch_parallel as parallel


# ===== 一、严格比较科学身份；仅放行本次指定的执行代码迁移 =====
def scientific_identity(value):
    value = copy.deepcopy(value)
    value.pop("wrapper_source_sha256", None)
    value.pop("acceleration_config", None)
    value.pop("startpoint_parallel_config", None)
    value.pop("numerical_evaluation_config", None)
    return value


def scientific_config(value):
    return {k: v for k, v in value.items() if not k.startswith("_") and k not in ("acceleration", "startpoint_parallel", "numerical_evaluation")}


def target_config(cfg, settings):
    target = copy.deepcopy(cfg)
    target["acceleration"] = copy.deepcopy(settings["acceleration"])
    if "startpoint_parallel" in settings:
        target["startpoint_parallel"] = copy.deepcopy(settings["startpoint_parallel"])
    if "numerical_evaluation" in settings:
        target["numerical_evaluation"] = copy.deepcopy(settings["numerical_evaluation"])
    validate_config(target)
    return target


def legacy_sources(settings):
    """迁移只接受配置中逐文件固定哈希的已发布版本。"""
    return [{"ref": settings["legacy_source_ref"], "hashes": settings["legacy_wrapper_source_sha256"]},
            *settings.get("additional_legacy_sources", [])]


def plan(target, settings):
    from sklearn.model_selection import KFold
    from substain_sustain.model import _identity, _features, _read_z, _model_seed
    output = Path(target["output_dir"])
    cv = output / "model/cv"
    if (output / "model/fit").exists() or (output / "validation").exists():
        raise ValueError("任务已进入fit/validation，本次CV迁移不适用；保留当前任务，未自动停止")
    for stage in ("fit", "validate", "report", "bundle"):
        if (output / "execution" / f"{stage}.json").exists():
            raise ValueError(f"已有{stage}记录，拒绝在未知阶段停止或迁移")
    old = parallel.read_json(cv / "manifest.json")
    new = _identity(target, output, [output / "reference/C3_primary_z.tsv"], "C3_crossvalidation")
    permitted = [item["hashes"] for item in legacy_sources(settings)] + [new["wrapper_source_sha256"]]
    if old.get("wrapper_source_sha256") not in permitted:
        raise ValueError("旧包装器不在已测试的迁移来源内；源码修改不能借本工具跳过检查")
    if scientific_identity(old) != scientific_identity(new):
        raise ValueError("输入、LS参考、官方库、依赖、模型/运行参数或特征方案改变，拒绝迁移")
    from substain_sustain.startpoints import parallel_settings
    if settings.get("preserve_acceleration") and old.get("acceleration_config") != new["acceleration_config"]:
        raise ValueError("此次起点并行迁移不允许同时改变分块似然或心跳参数")
    if settings.get("preserve_numerical_evaluation") and old.get("numerical_evaluation_config") != new["numerical_evaluation_config"]:
        raise ValueError("此次CPU调度迁移不允许同时改变冻结模型评估参数")
    if settings.get("require_existing_startpoint_parallel") and not old.get("startpoint_parallel_config", {}).get("enabled"):
        raise ValueError("本次SMT迁移要求已启用起点检查点的版本；未停止旧任务")
    if (settings.get("preserve_completed_startpoints") and old.get("startpoint_parallel_config", {}).get("enabled")
            and old["startpoint_parallel_config"]["checkpoint_dir"] != new["startpoint_parallel_config"]["checkpoint_dir"]):
        raise ValueError("保留完整起点的迁移不允许同时改检查点目录名称")
    if "startpoint_parallel" not in settings and old.get("startpoint_parallel_config", parallel_settings({})) != new["startpoint_parallel_config"]:
        raise ValueError("本次切换未授权更改起点并行配置")
    frame, _ = _read_z(output / "reference/C3_primary_z.tsv", _features(output, target))
    model = target["model"]
    splits = [t.tolist() for _, t in KFold(n_splits=model["folds"], shuffle=True, random_state=model["seed"]).split(frame)]
    expected = [{"fold": i, "indices": t, "subject_keys": frame.iloc[t].subject_key.tolist(),
                 "model_seeds": {str(k): _model_seed(model["seed"], i, k)
                                 for k in range(1, model["k_max"]+1)}} for i, t in enumerate(splits)]
    if parallel.read_json(cv / "folds.json") != expected:
        raise ValueError("分折或模型种子记录不一致")
    updates = {cv / "manifest.json": parallel.encoded(new)}
    for folder in cv.glob("fold_*"):
        fold = int(folder.name.removeprefix("fold_"))
        if not 0 <= fold < len(splits):
            raise ValueError("发现超出配置的折目录")
        before = {"parent": old, "fold": fold, "indices": splits[fold]}
        if parallel.read_json(folder / "manifest.json") != before:
            raise ValueError(f"第{fold}折身份不同")
        updates[folder / "manifest.json"] = parallel.encoded({**before, "parent": new})
    for stage in ("ingest", "reference"):
        path = output / "execution" / f"{stage}.json"
        record = parallel.read_json(path)
        previous_hash = record["config_sha256"]
        previous = parallel.read_json(output / "execution" / f"config_{previous_hash}.json")
        if (record["status"] != "complete" or parallel.config_hash(previous) != previous_hash
                or scientific_config(previous) != scientific_config(target)):
            raise ValueError(f"{stage}尚未完成或其配置快照不同")
        record.update(config_sha256=parallel.config_hash(target), reused_from_config_sha256=previous_hash,
                      reuse_reason="validated scheduling and stable frozen-model evaluation; stage not rerun")
        updates[path] = parallel.encoded(record)
    return old, new, updates


# ===== 二、全C3矩阵的逐元素检查；任何差异均在停止之前拒绝 =====
def equivalence(target, settings, directory):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from pySuStaIn.ZscoreSustain import ZScoreSustainData
    from substain_sustain.model import _features, _read_z, _settings, _engine
    from substain_sustain.acceleration import build_blocked_kernel
    output = Path(target["output_dir"])
    features = _features(output, target)
    if target["runtime"].get("profile", "formal") == "formal" and len(features) != settings["required_formal_features"]:
        raise ValueError("此次正式迁移要求38维；未停止旧任务")
    _, values = _read_z(output / "reference/C3_primary_z.tsv", features)
    model = _settings(target, len(features))
    model["_acceleration"] = execution_settings({})
    with threadpool_limits(limits=target["runtime"]["blas_threads"]):
        engine = _engine(values, features, model, directory, settings["equivalence_seed"], 1, 1)
        blocked = build_blocked_kernel(engine, target["acceleration"]["block_rows"])
        data = ZScoreSustainData(values, len(features)*len(model["z_vals"]))
        rng = np.random.default_rng(settings["equivalence_seed"])
        for _ in range(settings["equivalence_sequences"]):
            sequence = engine._initialise_sequence(data, rng)[0]
            expected = engine._calculate_likelihood_stage(data, sequence)
            actual = blocked(data, sequence)
            if not np.isfinite(expected).all() or not np.array_equal(expected, actual):
                raise ValueError("工作站C3的原始与分块似然未逐元素完全一致；旧任务未停止")
    return {"bitwise_equal": True, "subjects": len(values), "features": len(features),
            "sequences": settings["equivalence_sequences"], "seed": settings["equivalence_seed"]}


# ===== 三、旧后台续跑器也要停止；不能只终止其分析子进程 =====
def matching_processes(cfg):
    import monitor
    found = parallel.matching_processes(cfg)
    for process in psutil.process_iter():
        try:
            tokens = process.cmdline()
            if len(tokens) < 3 or "--resume" not in tokens:
                continue
            # nohup python -u script.py是已提供的续跑入口，不能漏掉其协调器。
            script_name = next((Path(token).name for token in tokens[1:] if not token.startswith("-")), None)
            if script_name not in ("switch_parallel.py", "switch_acceleration.py", "switch_startpoints.py"):
                continue
            cwd = Path(process.cwd()).resolve()
            config_arg = monitor.argument(tokens, "--config")
            if not config_arg:
                continue
            other = load_config(monitor.resolved(config_arg, str(cwd)))
            if (other["project_dir"], other["output_dir"]) != (cfg["project_dir"], cfg["output_dir"]):
                continue
            if (cwd != Path(cfg["project_dir"]) or process.uids().real != os.getuid()
                    or Path(process.exe()).resolve() != Path(sys.executable).resolve()):
                raise RuntimeError("相关续跑器的cwd/用户/解释器不同，拒绝自动停止")
            if all(p.pid != process.pid for p in found):
                found.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


# ===== 四、暂停后备份小文件并记录每个大缓存的哈希，不复制或删除大模型 =====
def prepare_backup(target, settings, backup):
    from substain_sustain.model import _hash, _pickle, _features
    old, new, updates = plan(target, settings)
    output = Path(target["output_dir"])
    local_path = Path(target["_config_file"])
    local = yaml.safe_load(local_path.read_text(encoding="utf-8"))
    local["acceleration"] = target["acceleration"]
    if "startpoint_parallel" in settings:
        local["startpoint_parallel"] = target["startpoint_parallel"]
    if "numerical_evaluation" in settings:
        local["numerical_evaluation"] = target["numerical_evaluation"]
    updates[local_path] = yaml.safe_dump(local, allow_unicode=True, sort_keys=False).encode()
    updates[output / "execution" / f"config_{parallel.config_hash(target)}.json"] = parallel.encoded(parallel.snapshot(target))
    originals = {path: path.read_bytes() if path.exists() else None for path in updates}
    inventory = []
    for i, (path, value) in enumerate(originals.items()):
        saved = backup / f"{i:03d}_{path.name}"
        if value is not None:
            saved.write_bytes(value)
        inventory.append({"path": str(path), "backup": str(saved) if value is not None else None,
                          "sha256_before": hashlib.sha256(value).hexdigest() if value is not None else None,
                          "sha256_after": hashlib.sha256(updates[path]).hexdigest()})
    for name in ("crossvalidate.json", "crossvalidate_error.txt"):
        path = output / "execution" / name
        if path.is_file():
            (backup / f"previous_{name}").write_bytes(path.read_bytes())
    # 数值评估更新后重算派生分数；旧派生表/完成标记另存，原始模型链不改。
    derived = list((output/"model/cv").glob("*.tsv")) + list((output/"model/cv").glob("fold_*/metrics.tsv"))
    derived += list((output/"model/cv").glob("fold_*/complete.json"))
    for path in derived:
        saved = backup/"previous_evaluation"/path.relative_to(output)
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(path.read_bytes())
    caches = []
    n_events = len(_features(output, target))*len(target["model"]["z_vals"])
    strict_caches = audit_cv_caches(target, settings) if "required_completed_k" in settings else None
    for path in sorted((output / "model/cv").glob("fold_*/pickle_files/*.pickle")):
        subtype = int(path.stem.rsplit("subtype", 1)[1])+1
        try:
            _pickle(path, subtype, n_events, target["model"]["cv_iterations"])
            valid, error = True, None
        except (EOFError, pickle.UnpicklingError, KeyError, ValueError) as exc:
            valid, error = False, str(exc)
        stat = path.stat()
        caches.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                       "sha256": _hash(path), "complete_shape": valid, "error": error})
    receipt = {"status": "prepared", "files": inventory, "caches": caches,
               "old_identity": old, "new_identity": new,
               "scientific_parameters_unchanged": True, "incomplete_chains": "preserved; will be rerun"}
    if strict_caches is not None:
        receipt["validated_caches"] = strict_caches
    if settings.get("preserve_completed_startpoints"):
        from startpoint_cache_migration import checkpoint_plan
        receipt["startpoint_checkpoints"] = checkpoint_plan(target, old, new)
    parallel.atomic_bytes(backup / "receipt.json", parallel.encoded(receipt))
    return updates, originals, receipt


def migrate(target, transaction, backup):
    from substain_sustain.model import _lock
    updates, originals, receipt = transaction
    output = Path(target["output_dir"])
    with ExitStack() as stack:
        stack.enter_context(_lock(output / "execution/parallel_resume"))
        stack.enter_context(_lock(output / "model/cv"))
        for folder in sorted((output / "model/cv").glob("fold_*")):
            stack.enter_context(_lock(folder))
        for path, original in originals.items():
            if (path.read_bytes() if path.exists() else None) != original:
                raise RuntimeError("备份后文件被其他进程修改；未迁移，请核查其他调度器")
        try:
            if "startpoint_checkpoints" in receipt:
                from startpoint_cache_migration import install_checkpoints
                receipt["startpoint_results_copied"] = install_checkpoints(receipt["startpoint_checkpoints"])
            for path, value in updates.items():
                parallel.atomic_bytes(path, value)
            if load_config(target["_config_file"]) != target:
                raise ValueError("写入后的有效配置与预检不符")
        except BaseException:
            for path, value in originals.items():
                if value is None:
                    path.unlink(missing_ok=True)
                else:
                    parallel.atomic_bytes(path, value)
            receipt["status"] = "rolled_back"
            parallel.atomic_bytes(backup / "receipt.json", parallel.encoded(receipt))
            raise
        receipt["status"] = "complete"
        parallel.atomic_bytes(backup / "receipt.json", parallel.encoded(receipt))


# ===== 五、一次性切换和可单独执行的安全续跑 =====
def audit_cv_caches(target, settings, folds_to_check=None):
    """分块核验完整链；不改写缓存，发现损坏时在停止旧任务前拒绝切换。"""
    import numpy as np
    from substain_sustain.model import _features, _pickle, _hash, _read_z
    from substain_sustain.log_evaluation import numerical_settings, evaluate_samples, check_raw_mean, z_diagnostics, training_floor_audit
    output = Path(target["output_dir"])
    cv, model = output/"model/cv", target["model"]
    if not 1 <= settings["required_completed_k"] <= model["k_max"]:
        raise ValueError("要求保留的K超出候选模型范围")
    features = _features(output, target)
    n_features = len(features)
    n_events = n_features * len(model["z_vals"])
    folds = parallel.read_json(cv/"folds.json")
    checked = []
    data = None
    evaluation_settings = numerical_settings(target)
    chunk = settings["cache_validation_chunk"]
    tolerance = settings["cache_probability_atol"]
    if type(chunk) is not int or chunk < 1 or not 0 < tolerance < 0.001:
        raise ValueError("缓存核验分块或概率容差无效")
    for fold in range(model["folds"]) if folds_to_check is None else folds_to_check:
        folder = cv/f"fold_{fold}"/"pickle_files"
        for k in range(1, min(model["k_max"], settings.get("audit_max_k", model["k_max"]))+1):
            path = folder/f"C3_fold{fold}_subtype{k-1}.pickle"
            if not path.exists():
                if k <= settings["required_completed_k"]:
                    raise ValueError(f"第{fold}折缺少要求保留的K={k}缓存；未停止旧任务")
                continue
            before = path.stat()
            cached = _pickle(path, k, n_events, model["cv_iterations"])
            def validate_sequence(values):
                if not np.isfinite(values).all():
                    raise ValueError(f"缓存序列非有限：{path}")
                expected = np.arange(n_events)[None, :, None]
                if not np.array_equal(np.sort(values, axis=1), np.broadcast_to(expected, values.shape)):
                    raise ValueError(f"缓存序列不是114事件的完整排列：{path}")
                positions = np.argsort(values, axis=1).reshape(k, len(model["z_vals"]), n_features, -1)
                if np.any(np.diff(positions, axis=1) <= 0):
                    raise ValueError(f"同一特征的阈值顺序无效：{path}")
            for start in range(0, model["cv_iterations"], chunk):
                validate_sequence(np.asarray(cached["samples_sequence"])[:, :, start:start+chunk])
            for name in ("ml_sequence_EM", "ml_sequence_prev_EM"):
                sequence = np.asarray(cached[name])
                if sequence.shape != (k, n_events):
                    raise ValueError(f"缓存{name}形状无效：{path}")
                validate_sequence(sequence[:, :, None])
            for name in ("samples_f", "ml_f_EM", "ml_f_prev_EM"):
                fractions = np.asarray(cached[name])
                if (not np.isfinite(fractions).all() or np.any(fractions < 0) or np.any(fractions > 1)
                        or fractions.shape[0] != k or not np.allclose(fractions.sum(axis=0), 1, atol=tolerance, rtol=0)):
                    raise ValueError(f"缓存{name}比例无效：{path}")
            likelihood = np.asarray(cached["samples_likelihood"])
            if likelihood.shape not in ((model["cv_iterations"],), (model["cv_iterations"], 1)):
                raise ValueError(f"缓存似然长度无效：{path}")
            test = np.asarray(cached["mean_likelihood_subj_test"])
            expected_n = len(folds[fold]["indices"])
            if test.shape not in ((expected_n,), (expected_n, 1)):
                raise ValueError(f"缓存留出似然形状{test.shape}与期望{expected_n}人不符：{path}")
            if not np.isfinite(test).all() or np.any(test < 0):
                raise ValueError(f"缓存留出似然负值={int((test<0).sum())}、非有限={int((~np.isfinite(test)).sum())}：{path}")
            zero = test.reshape(-1) == 0
            numerical = {"raw_zero_subjects": int(zero.sum()), "n_test": expected_n}
            from pySuStaIn.ZscoreSustain import ZscoreSustain, ZScoreSustainData
            if data is None:
                _, data = _read_z(output/"reference/C3_primary_z.tsv", features)
            heldout_values = data[np.asarray(folds[fold]["indices"])]
            calculator = ZscoreSustain(heldout_values, np.tile(model["z_vals"], (n_features, 1)),
                np.repeat(model["z_max"], n_features), features.label.tolist(), model["startpoints"], k,
                model["cv_iterations"], str(cv), "read_only_cache_check", False, seed=model["seed"])
            # 即使本折留出集没有零值，训练集也可能包含其他折的极端样本。
            train_mask = np.ones(len(data), dtype=bool)
            train_mask[folds[fold]["indices"]] = False
            numerical["training_at_saved_ml_em"] = training_floor_audit(calculator,
                ZScoreSustainData(data[train_mask], n_events), cached["ml_sequence_EM"], cached["ml_f_EM"], evaluation_settings)
            if zero.any():
                # 只对零值对应受试者重算冻结链，不拟合、不更改原缓存，也不填epsilon。
                values = heldout_values[zero]
                evaluated = evaluate_samples(calculator, ZScoreSustainData(values, n_events),
                    cached["samples_sequence"], cached["samples_f"], evaluation_settings)
                check_raw_mean(test.reshape(-1)[zero], evaluated, int(zero.sum()), evaluation_settings)
                diagnostic = z_diagnostics(values, features.feature_id.tolist(), evaluation_settings)
                reference = parallel.read_json(output/"reference/model.json")
                scales = dict(zip(reference.get("feature_ids", []), reference.get("residual_sd", [])))
                for feature in diagnostic["top_features"]:
                    feature["ls_residual_sd"] = scales.get(feature["feature_id"])
                numerical.update(verified_frozen_underflow=True, log_mean_min=float(evaluated["log_mean"].min()),
                                 log_mean_max=float(evaluated["log_mean"].max()), z_diagnostics=diagnostic)
                print(f"fold {fold:02d} K={k}：原始零似然{int(zero.sum())}/{expected_n}；"
                      f"冻结链复核一致，对数域有限，min logL={numerical['log_mean_min']:.6g}；保留缓存。", flush=True)
            digest = _hash(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError("核验期间缓存正在写入；请稍后重试，旧任务未停止")
            checked.append({"path": str(path), "fold": fold, "k": k, "sha256": digest,
                            "size": after.st_size, "mtime_ns": after.st_mtime_ns, "numerical_evaluation": numerical})
    return checked


def parallel_equivalence(target, settings, directory):
    """小型合成数据验证实际spawn、随机流、起点输出及后续MCMC；与正式数据分离。"""
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from pySuStaIn.ZscoreSustain import ZScoreSustainData
    from substain_sustain.model import _engine
    from substain_sustain.startpoints import cpu_topology, parallel_settings, worker_cpu_groups
    probe = settings["parallel_probe"]
    psettings = parallel_settings(target)
    topology = cpu_topology(psettings)
    values = np.random.default_rng(probe["seed"]).normal(1, 1.2, (probe["subjects"], probe["features"]))
    features = pd.DataFrame({"label": [f"synthetic_{i}" for i in range(probe["features"])]})
    plain = {"z_vals": probe["z_vals"], "z_max": probe["z_max"], "startpoints": probe["startpoints"],
             "_acceleration": {**execution_settings(target), "progress_enabled": False}}
    count = min(probe["startpoints"], psettings["workers_per_fold"])
    parallel_model = {**plain, "_acceleration": execution_settings(target), "_startpoint_parallel": psettings,
                      "_startpoint_runtime": {"workers": count, "core_cpu_groups": worker_cpu_groups(topology, psettings)[:count]}}
    with threadpool_limits(limits=1):
        serial = _engine(values, features, plain, directory/"serial", probe["seed"], probe["mcmc_iterations"], 1)
        concurrent = _engine(values, features, parallel_model, directory/"parallel", probe["seed"], probe["mcmc_iterations"], 1)
        data = ZScoreSustainData(values, probe["features"]*len(probe["z_vals"]))
        before, after = serial._find_ml(data), concurrent._find_ml(data)
        if any(not np.array_equal(a, b, equal_nan=True) for a, b in zip(before, after)):
            raise ValueError("工作站合成起点串并行不一致；旧任务未停止")
        if serial.global_rng.bit_generator.state != concurrent.global_rng.bit_generator.state:
            raise ValueError("起点并行改变了父进程随机流")
        before_chain = serial._perform_mcmc(data, before[0], before[1], probe["mcmc_iterations"], 1, 0.01)
        after_chain = concurrent._perform_mcmc(data, after[0], after[1], probe["mcmc_iterations"], 1, 0.01)
        if any(not np.array_equal(a, b, equal_nan=True) for a, b in zip(before_chain, after_chain)):
            raise ValueError("工作站起点并行后的MCMC不一致；旧任务未停止")
    return {"bitwise_equal": True, "parent_rng_equal": True, "parallel_workers": count, "synthetic": True}


def main(default_settings=None, entry_script=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--settings", type=Path, default=default_settings or Path(__file__).resolve().parents[1]/"config/acceleration_switch.yaml")
    parser.add_argument("--apply", action="store_true", help="完成预检后备份、停止旧任务、迁移并后台续跑")
    parser.add_argument("--resources-only", action="store_true", help="只读显示目标CPU/内存并发，不检查或迁移缓存、不停止任务")
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.resources_only and (args.apply or args.resume):
        parser.error("--resources-only不能与--apply或--resume同时使用")
    cfg = load_config(args.config)
    settings = yaml.safe_load(args.settings.read_text(encoding="utf-8"))
    execution_settings({"acceleration": settings["acceleration"]})
    for key in ("terminate_timeout_seconds", "kill_timeout_seconds", "equivalence_sequences"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"{key}必须为正整数")
    settings["workers"] = cfg["model"]["workers"]
    os.environ["PYTHONUNBUFFERED"] = "1"
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(cfg["runtime"]["blas_threads"])
    if args.resume:
        parallel.resume(cfg, settings)
        return
    root, config_path = Path(cfg["project_dir"]), Path(cfg["_config_file"])
    if Path.cwd().resolve() != root or config_path.parent != root/"config" or not config_path.name.startswith("local."):
        raise ValueError("请在项目根目录操作，并使用config/local.*.yaml配置")
    if args.resources_only:
        import json
        target = target_config(cfg, settings)
        resources = parallel.resource_check(target, {**settings, "require_requested_workers": False})
        print(json.dumps(resources, ensure_ascii=False, indent=2), flush=True)
        print("仅显示执行资源估算；未停止任务、未写入配置、未迁移缓存。", flush=True)
        return
    from substain_sustain.model import _lock
    with _lock(Path(cfg["output_dir"])/"execution/acceleration_switch"):
        target = target_config(cfg, settings)
        old, new, _ = plan(target, settings)
        processes = matching_processes(cfg)
        if old == new and cfg == target and processes:
            print("当前配置和缓存已完成此次切换，且任务正在运行；没有再次停止或启动。")
            return
        resources = parallel.resource_check(target, settings)
        if resources.get("startpoint_workers_per_model", 1) < settings.get("require_startpoint_workers", 1):
            raise MemoryError("内存不足以提供请求的每折起点并行数；未停止旧任务")
        cache_check = audit_cv_caches(target, settings) if "required_completed_k" in settings else None
        checkpoint_check = []
        if settings.get("preserve_completed_startpoints"):
            from startpoint_cache_migration import checkpoint_plan
            checkpoint_check = checkpoint_plan(target, old, new)
        backup = root/settings["backup_dir"]/(datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8])
        backup.mkdir(parents=True)
        # 保留允许来源的原代码，供审计和必要时人工恢复；缺失/哈希不符在停止前拒绝。
        legacy = next((item for item in legacy_sources(settings) if item["hashes"] == old["wrapper_source_sha256"]), legacy_sources(settings)[0])
        for name, expected in legacy["hashes"].items():
            data = subprocess.check_output(["git", "show", f"{legacy['ref']}:src/substain_sustain/{name}"], cwd=root)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("配置指定的旧代码与已验证哈希不同")
            (backup/f"legacy_{name}").write_bytes(data)
        check = equivalence(target, settings, backup/"equivalence")
        parallel_check = parallel_equivalence(target, settings, backup/"parallel_probe") if "parallel_probe" in settings else None
        parallel.atomic_bytes(backup/"preflight.json", parallel.encoded({"resources": resources, "equivalence": check,
            "parallel_equivalence": parallel_check, "validated_caches": cache_check,
            "startpoint_results_to_copy": sum(len(item["points"]) for item in checkpoint_check),
            "processes": [{"pid": p.pid, "created": p.create_time()} for p in processes]}))
        print(f"预检通过：{check['subjects']}例C3、{check['features']}维、逐元素一致；资源允许{resources['workers']}并行。", flush=True)
        if resources.get("parallel_startpoints"):
            print(f"每折{resources['startpoint_workers_per_model']}起点；同时计算上限{resources['max_compute_processes']}；"
                  "内存预算保持原配置，物理核/逻辑CPU预算为上限。", flush=True)
            unit = resources.get("worker_unit", "physical_core")
            print(f"调度单位：{unit}；预计工作集{resources['estimated_worker_bytes'] * resources['workers'] / 1024**3:.2f} GiB / "
                  f"预算{resources['memory_budget_bytes'] / 1024**3:.2f} GiB。", flush=True)
            if resources["max_compute_processes"] < resources.get("requested_compute_processes", 0):
                print("内存或CPU配额限制了实际并发；请求上限不等于操作系统running数量。", flush=True)
        if not args.apply:
            print(f"未停止或迁移任务。审计：{backup}；加--apply执行切换。")
            return
        transaction = []
        def prepare_while_paused():
            transaction.append(prepare_backup(target, settings, backup))
        parallel.stop_processes(processes, settings, verify=prepare_while_paused)
        if matching_processes(cfg):
            raise RuntimeError("仍有相关进程，未迁移；备份已保留")
        migrate(target, transaction[0], backup)
        log_dir = root/settings["log_dir"]
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir/f"{settings['log_prefix']}_{backup.name}.log"
        with log.open("ab") as stream:
            child = subprocess.Popen([sys.executable, str(entry_script or Path(__file__).resolve()), "--resume",
                "--config", str(config_path), "--settings", str(args.settings.resolve())], cwd=root,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        parallel.atomic_bytes(log_dir/f"{settings['log_prefix']}.latest_log", (str(log)+"\n").encode())
        print(f"已切换并后台续跑，PID={child.pid}；备份={backup}；日志={log}", flush=True)


if __name__ == "__main__":
    main()
