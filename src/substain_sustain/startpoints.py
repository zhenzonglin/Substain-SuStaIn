"""受限的官方EM起点调度：固定随机流、按原序汇总、逐起点保存检查点。"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time


# ===== 一、执行配置与Linux物理核拓扑；不在配置解析时加载NumPy =====
def parallel_settings(cfg: dict) -> dict:
    defaults = {"enabled": False, "workers_per_fold": 3, "fit_workers": 25,
                "physical_core_limit": 30, "logical_cpu_limit": 60,
                "worker_unit": "physical_core",
                "checkpoint_dir": "startpoints", "pool_progress_filename": "progress_pool.json",
                "worker_overhead_gb": 0.25}
    supplied = cfg.get("startpoint_parallel", {})
    if set(supplied) - set(defaults):
        raise ValueError("startpoint_parallel包含未知参数")
    settings = {**defaults, **supplied}
    if type(settings["enabled"]) is not bool:
        raise ValueError("startpoint_parallel.enabled必须为布尔值")
    for key in ("workers_per_fold", "fit_workers", "physical_core_limit", "logical_cpu_limit"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"startpoint_parallel.{key}必须为正整数")
    if settings["logical_cpu_limit"] < settings["physical_core_limit"]:
        raise ValueError("逻辑CPU预算不能小于物理核预算")
    if settings["worker_unit"] not in ("physical_core", "logical_cpu"):
        raise ValueError("worker_unit必须为physical_core或logical_cpu")
    slots = settings["physical_core_limit"] if settings["worker_unit"] == "physical_core" else settings["logical_cpu_limit"]
    if max(settings["workers_per_fold"], settings["fit_workers"]) > slots:
        raise ValueError("起点并行数超过计算槽位预算")
    import math
    overhead = settings["worker_overhead_gb"]
    if type(overhead) not in (int, float) or not math.isfinite(overhead) or overhead <= 0:
        raise ValueError("worker_overhead_gb必须为有限正数")
    for key, prefix in (("checkpoint_dir", "startpoints"), ("pool_progress_filename", "progress")):
        value = settings[key]
        if not isinstance(value, str) or Path(value).name != value or "\\" in value or not value.startswith(prefix):
            raise ValueError(f"startpoint_parallel.{key}必须为以{prefix}开头的单一名称")
    if settings["enabled"] and cfg.get("runtime", {}).get("blas_threads", 1) != 1:
        raise ValueError("起点进程并行要求BLAS线程为1，防止嵌套线程超配")
    if settings["enabled"] and cfg.get("acceleration", {}).get("backend") != "row_blocks":
        raise ValueError("起点并行的内存预算要求已验证的row_blocks后端")
    return settings


def worker_cpu_groups(topology: dict, settings: dict) -> list[list[int]]:
    """物理核模式每核一个起点；SMT模式每逻辑CPU一个起点，先铺开物理核。"""
    groups = topology["core_cpu_groups"]
    if settings.get("worker_unit", "physical_core") == "physical_core":
        return groups
    return [[group[level]] for level in range(max(map(len, groups)))
            for group in groups if level < len(group)]


def cpu_topology(settings: dict, *, allowed=None, sys_root=Path("/sys/devices/system/cpu"), quota=True) -> dict:
    """选取不同(socket,core)及其SMT兄弟；绝不把60个逻辑CPU称为60物理核。"""
    allowed = sorted(os.sched_getaffinity(0) if allowed is None else allowed)
    cores = {}
    for cpu in allowed:
        folder = sys_root / f"cpu{cpu}" / "topology"
        key = (int((folder / "physical_package_id").read_text()), int((folder / "core_id").read_text()))
        cores.setdefault(key, []).append(cpu)
    physical, logical = settings["physical_core_limit"], settings["logical_cpu_limit"]
    groups = [cpus for _, cpus in sorted(cores.items())][:physical]
    if len(groups) < physical:
        raise RuntimeError(f"请求{physical}物理核，当前CPU亲和性范围只含{len(groups)}个；未启动计算")
    # 先为每个物理核分配一个CPU，再按SMT层扩展，避免截断后丢失某个物理核。
    selected = [[group[0]] for group in groups]
    for level in range(1, max(map(len, groups))):
        for i, group in enumerate(groups):
            if sum(map(len, selected)) < logical and level < len(group):
                selected[i].append(group[level])
    if sum(map(len, selected)) < logical:
        raise RuntimeError(f"选定{physical}物理核只有{sum(map(len, selected))}逻辑CPU，不足请求的{logical}")
    # 亲和性不代表CPU时间配额；沿cgroup父目录读取可见的v2配额。
    quotas = []
    if quota:
        paths = [Path("/sys/fs/cgroup/cpu.max")]
        try:
            for line in Path("/proc/self/cgroup").read_text().splitlines():
                if line.startswith("0::"):
                    root = Path("/sys/fs/cgroup")
                    current = root / line[3:].lstrip("/")
                    while current.is_relative_to(root):
                        paths.append(current / "cpu.max")
                        if current == root:
                            break
                        current = current.parent
        except OSError:
            pass
        for path in set(paths):
            if path.is_file():
                maximum, period = path.read_text().split()
                if maximum != "max":
                    quotas.append(int(maximum) / int(period))
        legacy = Path("/sys/fs/cgroup/cpu")
        if (legacy / "cpu.cfs_quota_us").is_file():
            limit = int((legacy / "cpu.cfs_quota_us").read_text())
            if limit > 0:
                quotas.append(limit / int((legacy / "cpu.cfs_period_us").read_text()))
        requested = logical if settings.get("worker_unit") == "logical_cpu" else physical
        if quotas and min(quotas) < requested:
            raise RuntimeError(f"cgroup CPU时间配额{min(quotas):g}小于请求的{requested}个CPU时间单位")
    return {"physical_cores": physical, "logical_cpus": logical, "core_cpu_groups": selected,
            "cpu_ids": sorted(cpu for group in selected for cpu in group),
            "quota_cores": min(quotas) if quotas else None}


@contextmanager
def cpu_scope(cpu_ids):
    before = os.sched_getaffinity(0)
    os.sched_setaffinity(0, cpu_ids)
    try:
        yield
    finally:
        os.sched_setaffinity(0, before)


def within_cpu_budget(function):
    @wraps(function)
    def run(cfg, *args, **kwargs):
        settings = parallel_settings(cfg)
        if not settings["enabled"]:
            return function(cfg, *args, **kwargs)
        with cpu_scope(cpu_topology(settings)["cpu_ids"]):
            return function(cfg, *args, **kwargs)
    return run


_FOLD_RUNTIME = None


def initialize_fold(queue, parent_pid):
    global _FOLD_RUNTIME
    parent_death_guard(parent_pid)
    _FOLD_RUNTIME = queue.get()
    os.sched_setaffinity(0, [cpu for group in _FOLD_RUNTIME["core_cpu_groups"] for cpu in group])


def fold_runtime():
    if _FOLD_RUNTIME is None:
        raise RuntimeError("折进程缺少受限起点资源分配；禁止自动启动不限量进程池")
    return _FOLD_RUNTIME


def parent_death_guard(parent_pid):
    """Linux父进程死亡后退出，避免中断协调器留下脱离监控的计算子进程。"""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "无法设置父进程退出保护")
    if os.getppid() != parent_pid:
        raise RuntimeError("父进程在启动期间已退出")


# ===== 二、检查点身份包括数据、初始化、子种子和算法来源，不包含完成先后顺序 =====
def _normal(value):
    if isinstance(value, dict):
        return {str(k): _normal(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normal(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def array_identity(value):
    import numpy as np
    value = np.asarray(value)
    return {"shape": list(value.shape), "dtype": value.dtype.str,
            "fortran": bool(value.flags.f_contiguous and not value.flags.c_contiguous),
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}


def _validate_result(result, n_events):
    import numpy as np
    sequence, fractions, likelihood = map(np.asarray, result[:3])
    if (sequence.ndim != 2 or sequence.shape[1] != n_events or
            not np.array_equal(np.sort(sequence, axis=1), np.tile(np.arange(n_events), (len(sequence), 1)))):
        raise ValueError("起点检查点事件顺序不是完整排列")
    if (fractions.shape != (len(sequence),) or not np.isfinite(fractions).all() or
            np.any(fractions < 0) or not np.isclose(fractions.sum(), 1) or not np.isfinite(likelihood).all()):
        raise ValueError("起点检查点比例或似然无效")


def load_checkpoint(folder, identity, n_events):
    import numpy as np
    from .model import _hash
    receipt = folder / "complete.json"
    if not receipt.exists():
        return None
    record = json.loads(receipt.read_text())
    if record["identity"] != identity or record["sha256"] != _hash(folder / "result.npz"):
        raise ValueError(f"起点检查点身份或哈希不符，保留文件并停止：{folder}")
    with np.load(folder / "result.npz", allow_pickle=False) as values:
        result = tuple(values[f"value_{i}"] for i in range(record["result_count"]))
    _validate_result(result, n_events)
    return result


def _run_startpoint(payload):
    """子进程重新构造同一官方计算器，只调用官方的单起点函数。"""
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from pySuStaIn.ZscoreSustain import ZScoreSustainData
    from .model import _engine, _json, _hash
    os.sched_setaffinity(0, payload["cpu_ids"])
    spec, folder = payload["spec"], Path(payload["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    settings = {**spec["model"], "_acceleration": spec["acceleration"]}
    with threadpool_limits(limits=1):
        engine = _engine(payload["data"], pd.DataFrame({"label": spec["labels"]}), settings,
                         folder, spec["seed"], spec["iterations"], spec["target_k"])
        data = ZScoreSustainData(payload["data"], payload["n_events"])
        seed = np.random.SeedSequence(**payload["seed_state"])
        result = getattr(engine, payload["method"])(data, *payload["initialization"], seed)
    _validate_result(result, payload["n_events"])
    temp = folder / f"result.npz.tmp.{os.getpid()}"
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **{f"value_{i}": np.asarray(value) for i, value in enumerate(result)})
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(folder / "result.npz")
    _json(folder / "complete.json", {"identity": payload["identity"], "result_count": len(result),
                                    "sha256": _hash(folder / "result.npz"), "pid": os.getpid()})
    return result


def _abort_executor(executor):
    # Python3.10没有terminate_workers；仅终止本对象拥有的子进程，不按名称杀进程。
    processes = list((executor._processes or {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
    executor.shutdown(wait=True, cancel_futures=True)


# ===== 三、仅替换官方pool.map；随机数生成、EM、最优起点选择仍在官方实现 =====
class OrderedStartpointPool:
    METHODS = ("_find_ml_iteration", "_find_ml_split_iteration", "_find_ml_mixture_iteration")

    def __init__(self, engine, directory, settings, allocation, spec):
        self.engine, self.directory = engine, Path(directory)
        self.settings, self.allocation, self.spec = settings, allocation, spec

    def map(self, function, seeds):
        from importlib.metadata import version
        from .model import _guard, _json, _lock, _hash, _upstream_provenance
        method = next((name for name in self.METHODS if function.func == getattr(self.engine, name)), None)
        if method is None or function.keywords:
            raise ValueError("起点池只接受冻结官方版本的三个起点入口")
        seeds = list(seeds)
        data, *initialization = function.args
        n_events = data.getNumStages()
        identity = {"schema": 1, "method": method, "data": array_identity(data.data),
                    "initialization": [array_identity(x) for x in initialization], "spec": self.spec,
                    "seeds": [_normal(seed.state) for seed in seeds], "upstream": _upstream_provenance(),
                    "source": {name: _hash(Path(__file__).with_name(name))
                               for name in ("model.py", "acceleration.py", "progress.py", "startpoints.py")},
                    "dependencies": {name: version(name) for name in ("numpy", "scipy", "pathos")},
                    "python": sys.version}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        relative = Path(self.settings["checkpoint_dir"]) / f"k_{self.engine.N_S_max}" / digest
        folder = self.directory / relative
        results = [None] * len(seeds)
        state = {"pid": os.getpid(), "process_created": __import__("psutil").Process().create_time(),
                 "target_k": self.engine.N_S_max, "method": method, "batch": str(relative),
                 "total": len(seeds), "completed": [], "reused": [], "running": [], "status": "running"}
        tracker = getattr(self.engine, "_progress_tracker", None)

        def emit(status="running"):
            state.update(status=status, updated_unix=time.time())
            _json(self.directory / self.settings["pool_progress_filename"], state)
            if tracker:
                tracker.update(phase="parallel_startpoints", startpoints_completed=len(state["completed"]),
                               startpoints_reused=len(state["reused"]), startpoints_total=len(seeds))
                tracker.emit("startpoint_pool_updated")

        with _lock(folder):
            _guard(folder, identity)
            for i in range(len(seeds)):
                results[i] = load_checkpoint(folder / f"point_{i:03d}", {"batch": digest, "index": i}, n_events)
                if results[i] is not None:
                    state["completed"].append(i)
                    state["reused"].append(i)
            pending = [i for i, result in enumerate(results) if result is None]
            emit()
            if not pending:
                emit("complete")
                return results
            groups = self.allocation["core_cpu_groups"][:min(len(pending), self.allocation["workers"])]
            if not groups:
                raise RuntimeError("没有为起点分配CPU")
            # spawn避免从带心跳线程的父进程fork；子进程不复制RLock/线程或进度对象。
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                os.environ[key] = "1"
            executor = ProcessPoolExecutor(max_workers=len(groups), mp_context=mp.get_context("spawn"),
                                           initializer=parent_death_guard, initargs=(os.getpid(),))
            futures = {}

            def submit(index, slot):
                payload = {"spec": self.spec, "data": data.data, "initialization": initialization,
                           "n_events": n_events, "method": method, "seed_state": _normal(seeds[index].state),
                           "identity": {"batch": digest, "index": index},
                           "folder": str(folder / f"point_{index:03d}"), "cpu_ids": groups[slot]}
                futures[executor.submit(_run_startpoint, payload)] = (index, slot)

            try:
                for slot in range(len(groups)):
                    submit(pending.pop(0), slot)
                state["running"] = [i for i, _ in futures.values()]
                emit()
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        index, slot = futures.pop(future)
                        results[index] = future.result()
                        state["completed"].append(index)
                        if pending:
                            submit(pending.pop(0), slot)
                    state["running"] = [i for i, _ in futures.values()]
                    emit()
            except BaseException as exc:
                state.update(error_type=type(exc).__name__, error=str(exc))
                _abort_executor(executor)
                state["running"] = []
                emit("failed")
                raise
            else:
                executor.shutdown(wait=True)
                emit("complete")
        # 始终按原子种子索引返回，不让完成先后顺序改变argmax/并列处理。
        return results
