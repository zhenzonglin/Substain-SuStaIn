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
    return value


def scientific_config(value):
    return {k: v for k, v in value.items() if not k.startswith("_") and k != "acceleration"}


def target_config(cfg, settings):
    target = copy.deepcopy(cfg)
    target["acceleration"] = copy.deepcopy(settings["acceleration"])
    validate_config(target)
    return target


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
    if old.get("wrapper_source_sha256") not in (settings["legacy_wrapper_source_sha256"], new["wrapper_source_sha256"]):
        raise ValueError("旧包装器不在已测试的迁移来源内；源码修改不能借本工具跳过检查")
    if scientific_identity(old) != scientific_identity(new):
        raise ValueError("输入、LS参考、官方库、依赖、模型/运行参数或特征方案改变，拒绝迁移")
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
                      reuse_reason="bitwise-validated execution backend and progress only; stage not rerun")
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
            if Path(tokens[1]).name not in ("switch_parallel.py", "switch_acceleration.py"):
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
    caches = []
    n_events = len(_features(output, target))*len(target["model"]["z_vals"])
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
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--settings", type=Path, default=Path(__file__).resolve().parents[1]/"config/acceleration_switch.yaml")
    parser.add_argument("--apply", action="store_true", help="完成预检后备份、停止旧任务、迁移并后台续跑")
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
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
    from substain_sustain.model import _lock
    with _lock(Path(cfg["output_dir"])/"execution/acceleration_switch"):
        target = target_config(cfg, settings)
        old, new, _ = plan(target, settings)
        processes = matching_processes(cfg)
        if old == new and cfg == target and processes:
            print("当前配置和缓存已完成此次切换，且任务正在运行；没有再次停止或启动。")
            return
        resources = parallel.resource_check(target, settings)
        backup = root/settings["backup_dir"]/(datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8])
        backup.mkdir(parents=True)
        # 保留允许来源的原代码，供审计和必要时人工恢复；缺失/哈希不符在停止前拒绝。
        for name, expected in settings["legacy_wrapper_source_sha256"].items():
            data = subprocess.check_output(["git", "show", f"{settings['legacy_source_ref']}:src/substain_sustain/{name}"], cwd=root)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("配置指定的旧代码与已验证哈希不同")
            (backup/f"legacy_{name}").write_bytes(data)
        check = equivalence(target, settings, backup/"equivalence")
        parallel.atomic_bytes(backup/"preflight.json", parallel.encoded({"resources": resources, "equivalence": check,
            "processes": [{"pid": p.pid, "created": p.create_time()} for p in processes]}))
        print(f"预检通过：{check['subjects']}例C3、{check['features']}维、逐元素一致；资源允许{resources['workers']}并行。", flush=True)
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
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--resume",
                "--config", str(config_path), "--settings", str(args.settings.resolve())], cwd=root,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        parallel.atomic_bytes(log_dir/f"{settings['log_prefix']}.latest_log", (str(log)+"\n").encode())
        print(f"已切换并后台续跑，PID={child.pid}；备份={backup}；日志={log}", flush=True)


if __name__ == "__main__":
    main()
