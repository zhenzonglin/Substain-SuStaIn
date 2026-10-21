#!/usr/bin/env python3
"""将正在CV的同一分析安全切换并行数；仅迁移执行参数身份，不改变算法或链。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import psutil
import yaml

from substain_sustain.config import load_config, validate_config


# ===== 一、执行配置与严格身份比较 =====
def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_bytes(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_bytes(value)
    os.replace(temp, path)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode()


def snapshot(cfg):
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def config_hash(cfg):
    return hashlib.sha256(json.dumps(snapshot(cfg), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def scientific(value, identity=False):
    """唯一允许变化的是workers和内存预算比例；源码、数据、随机流仍逐字核验。"""
    result = copy.deepcopy(value)
    result.get("model_config" if identity else "model", {}).pop("workers", None)
    result.get("runtime_config" if identity else "runtime", {}).pop("max_memory_fraction", None)
    return result


def read_settings(path):
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("workers", "terminate_timeout_seconds", "kill_timeout_seconds"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"{key}必须为正整数")
    if not 0 < settings["max_memory_fraction"] <= 1:
        raise ValueError("max_memory_fraction必须在(0,1]")
    if settings["resume_stages"] != ["crossvalidate", "fit", "validate", "report", "bundle", "verify-bundle"]:
        raise ValueError("续跑阶段必须保持CV到展示包验收的完整顺序")
    return settings


def target_config(cfg, settings):
    result = copy.deepcopy(cfg)
    result["model"]["workers"] = settings["workers"]
    result["runtime"]["max_memory_fraction"] = settings["max_memory_fraction"]
    validate_config(result)
    return result


def resource_check(cfg, settings):
    from substain_sustain.model import _features, _read_z, _resources
    output = Path(cfg["output_dir"])
    features = _features(output, cfg)
    frame, _ = _read_z(output / "reference/C3_primary_z.tsv", features)
    model = cfg["model"]
    resource = _resources(cfg, output / "model/cv", len(frame), len(features),
                          model["cv_iterations"], model["k_max"], model["folds"], model["folds"])
    if settings["require_requested_workers"] and resource["workers"] != settings["workers"]:
        raise MemoryError(f"请求{settings['workers']}并行，当前CPU/内存/折数只允许{resource['workers']}；尚未停止旧进程。")
    return resource


def migration_plan(cfg):
    """只读预检；对每一折完整比较父身份和实际分折，不接受任何科学参数漂移。"""
    from substain_sustain.model import _identity, _features, _read_z, _model_seed
    from sklearn.model_selection import KFold
    output = Path(cfg["output_dir"])
    cv = output / "model/cv"
    for name in ("fit", "validation"):
        if (output / "model" / name).exists():
            raise ValueError("已进入CV后续阶段，本工具不停止或迁移主模型；请保留当前任务。")
    for name in ("fit", "validate", "report", "bundle"):
        if (output / "execution" / f"{name}.json").exists():
            raise ValueError(f"已有{name}执行记录；本工具仅用于尚未进入主拟合的CV任务。")
    old = read_json(cv / "manifest.json")
    new = _identity(cfg, output, [output / "reference/C3_primary_z.tsv"], "C3_crossvalidation")
    if scientific(old, True) != scientific(new, True):
        raise ValueError("输入、参考、源码、依赖或科学参数与CV缓存不一致，拒绝迁移。")
    features = _features(output, cfg)
    frame, _ = _read_z(output / "reference/C3_primary_z.tsv", features)
    model = cfg["model"]
    splits = [test.tolist() for _, test in KFold(n_splits=model["folds"], shuffle=True,
                                               random_state=model["seed"]).split(frame)]
    expected = [{"fold": i, "indices": indices, "subject_keys": frame.iloc[indices].subject_key.tolist(),
                 "model_seeds": {str(k): _model_seed(model["seed"], i, k)
                                 for k in range(1, model["k_max"] + 1)}} for i, indices in enumerate(splits)]
    if read_json(cv / "folds.json") != expected:
        raise ValueError("保存的分折或随机流不符，拒绝迁移。")
    updates = {cv / "manifest.json": encoded(new)}
    for folder in cv.glob("fold_*"):
        fold = int(folder.name.removeprefix("fold_"))
        if not 0 <= fold < len(splits):
            raise ValueError("存在额外折目录")
        path = folder / "manifest.json"
        before = {"parent": old, "fold": fold, "indices": splits[fold]}
        if read_json(path) != before:
            raise ValueError(f"第{fold}折身份不一致，拒绝迁移。")
        updates[path] = encoded({**before, "parent": new})
    # 只给经严格配置对照的已完成前置阶段增加复用记录；原记录及快照保留在备份中。
    for name in ("ingest", "reference"):
        path = output / "execution" / f"{name}.json"
        record = read_json(path)
        previous_hash = record["config_sha256"]
        previous = read_json(output / "execution" / f"config_{previous_hash}.json")
        if config_hash(previous) != previous_hash or scientific(previous) != scientific(snapshot(cfg)):
            raise ValueError(f"{name}配置快照不符，不能将旧阶段标记为可复用。")
        if record["status"] != "complete":
            raise ValueError(f"{name}尚未完成")
        record.update(config_sha256=config_hash(cfg), reused_from_config_sha256=previous_hash,
                      reuse_reason="only workers and max_memory_fraction changed; stage was not rerun")
        updates[path] = encoded(record)
    return updates


# ===== 二、精确识别当前配置的CV进程；不使用旧PID或广泛pkill =====
def matching_processes(cfg):
    import monitor
    bundle = Path(cfg["report"]["bundle_dir"]) / "sustain-display-bundle.tar.gz"
    rows, roots, _ = monitor.collect_processes(cfg, bundle)
    found = []
    for row in roots:
        process = psutil.Process(row["pid"])
        if process.create_time() != row["created"]:
            raise RuntimeError("预检期间PID已更换，请重新执行")
        if (row["command"] not in ("all", "crossvalidate") or
                Path(process.cwd()).resolve() != Path(cfg["project_dir"]) or
                process.uids().real != os.getuid() or
                Path(process.exe()).resolve() != Path(sys.executable).resolve()):
            raise RuntimeError("发现不属于本次CV入口的相关进程，拒绝自动停止")
        found.append(process)
    # 父进程及fork子进程可能有同一命令行；创建时间与父子关系均来自此次实时扫描。
    for row in rows:
        process = psutil.Process(row["pid"])
        if process.create_time() != row["created"] or process.uids().real != os.getuid():
            raise RuntimeError("进程身份发生变化")
        if all(p.pid != process.pid for p in found):
            found.append(process)
    return found


def stop_processes(processes, settings, verify=None):
    """先暂停协调器防止新建任务，再终止同一批进程；失败时恢复仍在暂停的进程。"""
    suspended = []
    try:
        ids = {p.pid for p in processes}
        ordered = sorted(processes, key=lambda p: p.ppid() in ids)
        for process in ordered:
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                process.suspend()
                suspended.append(process)
        # 暂停之后重新收集后代，防止扫描与暂停之间产生新worker。
        known = {p.pid for p in suspended}
        for process in list(suspended):
            for child in process.children(recursive=True):
                if child.pid not in known:
                    child.suspend()
                    suspended.append(child)
                    known.add(child.pid)
        if verify is not None:
            verify()  # 暂停后再检查尚未转入fit；不符则finally恢复原进程。
        for process in reversed(suspended):
            try:
                process.terminate()
                process.resume()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(suspended, timeout=settings["terminate_timeout_seconds"])
        for process in alive:
            process.kill()
        _, alive = psutil.wait_procs(alive, timeout=settings["kill_timeout_seconds"])
        if any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in alive):
            raise RuntimeError("仍有分析进程存活，未改动配置或缓存")
    finally:
        for process in suspended:
            try:
                if process.is_running() and process.status() == psutil.STATUS_STOPPED:
                    process.resume()
            except psutil.NoSuchProcess:
                pass


# ===== 三、加锁、备份、只迁移资源字段；不读写大型pickle =====
def migrate(cfg, settings, backup):
    from substain_sustain.model import _lock
    output = Path(cfg["output_dir"])
    with ExitStack() as stack:
        stack.enter_context(_lock(output / "model/cv"))
        for folder in sorted((output / "model/cv").glob("fold_*")):
            stack.enter_context(_lock(folder))
        updates = migration_plan(cfg)
        local_path = Path(cfg["_config_file"])
        local = yaml.safe_load(local_path.read_text(encoding="utf-8"))
        local.setdefault("model", {})["workers"] = settings["workers"]
        local.setdefault("runtime", {})["max_memory_fraction"] = settings["max_memory_fraction"]
        updates[local_path] = ("# 并行资源切换；原文件及身份记录保存在.cache/parallel_switches\n" +
                               yaml.safe_dump(local, allow_unicode=True, sort_keys=False)).encode()
        updates[output / "execution" / f"config_{config_hash(cfg)}.json"] = encoded(snapshot(cfg))
        originals = {path: path.read_bytes() if path.exists() else None for path in updates}
        backup.mkdir(parents=True, exist_ok=False)
        inventory = []
        for i, (path, value) in enumerate(originals.items()):
            saved = backup / f"{i:03d}_{path.name}"
            if value is not None:
                saved.write_bytes(value)
            inventory.append({"path": str(path), "backup": str(saved) if value is not None else None,
                              "sha256_before": hashlib.sha256(value).hexdigest() if value is not None else None,
                              "sha256_after": hashlib.sha256(updates[path]).hexdigest()})
        receipt = {"status": "prepared", "created_utc": datetime.now(timezone.utc).isoformat(),
                   "changed_fields": ["model.workers", "runtime.max_memory_fraction"], "files": inventory}
        atomic_bytes(backup / "receipt.json", encoded(receipt))
        try:
            for path, value in updates.items():
                atomic_bytes(path, value)
            if load_config(local_path) != cfg:
                raise ValueError("写入后配置与预检配置不一致")
        except BaseException:
            for path, value in originals.items():
                if value is None:
                    path.unlink(missing_ok=True)
                else:
                    atomic_bytes(path, value)
            receipt["status"] = "rolled_back"
            atomic_bytes(backup / "receipt.json", encoded(receipt))
            raise
        receipt["status"] = "complete"
        atomic_bytes(backup / "receipt.json", encoded(receipt))
        return receipt


# ===== 四、连续执行余下阶段；任何失败均停止，不重做导入和LS参考 =====
def resume(cfg, settings):
    from substain_sustain.model import _lock
    root, output = Path(cfg["project_dir"]), Path(cfg["output_dir"])
    with _lock(output / "execution/parallel_resume"):
        resource_check(cfg, settings)
        for stage in settings["resume_stages"]:
            command = [sys.executable, "-m", "substain_sustain", stage]
            if stage == "verify-bundle":
                command += ["--bundle", str(Path(cfg["report"]["bundle_dir"]) / settings["bundle_filename"])]
            else:
                command += ["--config", cfg["_config_file"]]
            subprocess.run(command, cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--settings", type=Path, default=Path(__file__).resolve().parents[1] / "config/parallel10.yaml")
    parser.add_argument("--apply", action="store_true", help="停止精确匹配的CV进程，迁移资源身份并后台续跑")
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = load_config(args.config)
    settings = read_settings(args.settings)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(cfg["runtime"]["blas_threads"])
    os.environ["MPLBACKEND"] = "Agg"
    if args.resume:
        resume(cfg, settings)
        return
    root = Path(cfg["project_dir"])
    config_path = Path(cfg["_config_file"])
    if config_path.parent != root / "config" or not config_path.name.startswith("local."):
        raise ValueError("只允许修改项目config/local.*.yaml；请使用config/local.38.yaml")
    if Path.cwd().resolve() != root:
        raise ValueError("请先cd到分析项目根目录")
    from substain_sustain.model import _lock
    with _lock(Path(cfg["output_dir"]) / "execution/parallel_switch"):
        # 已在运行的续跑器不会被误杀或重复启动。
        with _lock(Path(cfg["output_dir"]) / "execution/parallel_resume"):
            target = target_config(cfg, settings)
            migration_plan(target)
            resources = resource_check(target, settings)
            processes = matching_processes(cfg)
            print(json.dumps({"requested_workers": settings["workers"], "resources": resources,
                              "matched_pids": [p.pid for p in processes]}, ensure_ascii=False, indent=2), flush=True)
            if not args.apply:
                print("预检通过。加--apply执行停止、迁移和后台续跑；中断的MCMC链会重新计算。")
                return
            # 停止前检查备份位置可写，避免在明显配置错误时中断计算。
            backup_root = root / settings["backup_dir"]
            backup_root.mkdir(parents=True, exist_ok=True)
            token = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
            backup = backup_root / token
            stop_processes(processes, settings, verify=lambda: migration_plan(target))
            if matching_processes(cfg):
                raise RuntimeError("仍发现CV进程，未修改配置；请核查其他调度器")
            migrate(target, settings, backup)
        # 先释放续跑器锁再启动后台进程，避免子进程抢锁失败。
        log_dir = root / settings["log_dir"]
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / f"{settings['log_prefix']}_{token}.log"
        with log.open("ab") as stream:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--resume",
                                      "--config", str(config_path), "--settings", str(args.settings.resolve())],
                                     cwd=root, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        atomic_bytes(log_dir / f"{settings['log_prefix']}.latest_log", (str(log) + "\n").encode())
        print(f"切换完成；后台续跑PID={child.pid}；CV请求并行={settings['workers']}。", flush=True)
        print(f"审计备份：{backup}\n日志：{log}\n监控：.venv/bin/python scripts/monitor.py --config {config_path}")


if __name__ == "__main__":
    main()
