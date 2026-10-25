#!/usr/bin/env python3
"""独立、限时的官方K=1 EM计时；不代表已有worker的内部进度。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

import yaml
from substain_sustain.config import load_config


# ===== 一、诊断配置和输出隔离 =====
def read_settings(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("max_em_sweeps", "max_wall_seconds", "heartbeat_seconds", "termination_grace_seconds",
                "blas_threads", "max_memory_gb", "min_available_memory_gb", "memory_safety_factor"):
        if type(value[key]) not in (int, float) or value[key] <= 0:
            raise ValueError(f"{key}必须为正数")
    for key in ("fold", "startpoint", "nice_increment", "max_em_sweeps", "blas_threads"):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError(f"{key}必须为非负整数")
    if value["memory_safety_factor"] < 1:
        raise ValueError("内存安全倍数不得小于1")
    return value


def diagnostic_root(cfg: dict, settings: dict) -> Path:
    project = Path(cfg["project_dir"])
    root = (project / settings["output_dir"]).resolve()
    formal = Path(cfg["output_dir"]).resolve()
    if (not root.is_relative_to(project) or root == project or root.is_relative_to(formal)
            or formal.is_relative_to(root)):
        raise ValueError("诊断目录必须在本项目内且与正式结果目录分开")
    for spec in cfg["cohorts"].values():
        for key in ("derivatives", "source_root"):
            if spec.get(key) and root.is_relative_to(Path(spec[key]).resolve()):
                raise ValueError("诊断输出不能写入源数据目录")
    return root


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


# ===== 二、只读核对冻结身份，完整使用原折的训练样本 =====
def prepare(cfg: dict, settings: dict):
    from sklearn.model_selection import KFold
    from substain_sustain.model import (
        _identity, _features, _read_z, _settings, _assert_primary, _model_seed,
        _available_memory, estimate_working_bytes,
    )
    output = Path(cfg["output_dir"])
    cv = output / "model/cv"
    identity = _identity(cfg, output, [output / "reference/C3_primary_z.tsv"], "C3_crossvalidation")
    if json.loads((cv / "manifest.json").read_text()) != identity:
        raise ValueError("当前配置、输入或源码与CV身份不同，拒绝诊断")
    features = _features(output, cfg)
    frame, data = _read_z(output / "reference/C3_primary_z.tsv", features)
    model = _settings(cfg, len(features))
    _assert_primary(frame, model, "C3")
    if settings["fold"] >= model["folds"] or settings["startpoint"] >= model["startpoints"]:
        raise ValueError("探针折号或起点号超出正式配置")
    splits = list(KFold(n_splits=model["folds"], shuffle=True, random_state=model["seed"]).split(data))
    fold = settings["fold"]
    train, test = splits[fold]
    saved = json.loads((cv / "folds.json").read_text())[fold]
    seeds = {str(k): _model_seed(model["seed"], fold, k) for k in range(1, model["k_max"] + 1)}
    if saved != {"fold": fold, "indices": test.tolist(), "subject_keys": frame.iloc[test].subject_key.tolist(),
                 "model_seeds": seeds}:
        raise ValueError("保存的C3分折或随机流不一致")
    events = len(features) * len(model["z_vals"])
    estimate = estimate_working_bytes(len(train), len(features), events, 1, 1, settings["memory_safety_factor"])
    available = _available_memory()
    if (estimate > settings["max_memory_gb"] * 2**30 or
            available - estimate < settings["min_available_memory_gb"] * 2**30):
        raise MemoryError("当前内存不足以在保留余量的前提下运行额外一个诊断进程")
    metadata = {"scope": "independent probe; not live-worker telemetry or formal model",
                "fold": fold, "startpoint": settings["startpoint"], "n_c3": len(frame), "n_train": len(train),
                "n_features": len(features), "n_events": events, "k": 1, "model_seed": int(seeds["1"]),
                "estimated_working_gb": estimate / 2**30, "available_memory_gb": available / 2**30,
                "cv_identity": identity, "probe_settings": settings,
                "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return data[train], features, model, metadata


class ProbeLimit(Exception):
    """到达诊断预算；仅结束本进程内的探针。"""


# ===== 三、观测独立实例；每次计算仍调用原官方函数 =====
def observe(engine, data, seed_sequence, settings: dict, directory: Path) -> dict:
    import numpy as np
    started = time.monotonic()
    initial_times = os.times()
    initial_cpu = initial_times.user + initial_times.system
    lock, stop = threading.RLock(), threading.Event()
    state = {"pid": os.getpid(), "status": "running", "likelihood_calls_completed": 0,
             "em_sweeps_started": 0, "em_sweeps_completed": 0, "em_sweeps": []}
    kernel, optimise = engine._calculate_likelihood_stage, engine._optimise_parameters

    def check_time():
        if time.monotonic() - started >= settings["max_wall_seconds"]:
            raise ProbeLimit("time_limit")

    def emit():
        with lock:
            times = os.times()
            state.update(elapsed_seconds=time.monotonic() - started, cpu_seconds=times.user + times.system - initial_cpu,
                         updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(directory / "progress.json", state)
            event = state.get("current_event_rank", "-")
            event_total = state.get("events_in_sweep", "-")
            candidate = state.get("candidate_completed", "-")
            candidate_total = state.get("candidates_for_event", "-")
            print(f"[独立探针] {state['elapsed_seconds']:.1f}s | 似然计算完成 "
                  f"{state['likelihood_calls_completed']} 次 | EM轮完成 {state['em_sweeps_completed']} | "
                  f"本轮事件 {event}/{event_total} 候选 {candidate}/{candidate_total} | {state['status']}", flush=True)

    def heartbeat():
        try:
            while not stop.wait(settings["heartbeat_seconds"]):
                emit()
        except Exception as exc:
            with lock:
                state["heartbeat_error"] = str(exc)
            print(f"诊断心跳写入失败：{exc}", file=sys.stderr, flush=True)

    def wrapped_kernel(training_data, sequence):
        check_time()
        caller = sys._getframe(1)
        context = caller.f_locals if caller.f_code.co_name == "_optimise_parameters" else {}
        event_progress = {}
        if all(key in context for key in ("order_bio", "i", "index", "possible_positions")):
            order = np.asarray(context["order_bio"])
            event_progress = {"current_event_rank": int(np.flatnonzero(order == context["i"])[0]) + 1,
                              "events_in_sweep": len(order), "candidate_completed": int(context["index"]) + 1,
                              "candidates_for_event": len(context["possible_positions"])}
        began = time.monotonic()
        result = kernel(training_data, sequence)
        if not np.isfinite(result).all():
            raise FloatingPointError("官方阶段似然含非有限值")
        with lock:
            state["likelihood_calls_completed"] += 1
            state["last_likelihood_seconds"] = time.monotonic() - began
            state["last_sequence_sha256"] = hashlib.sha256(np.asarray(sequence).tobytes()).hexdigest()
            state.update(event_progress)
        return result

    def wrapped_optimise(training_data, sequence, fractions, rng):
        check_time()
        if state["em_sweeps_completed"] >= settings["max_em_sweeps"]:
            raise ProbeLimit("sweep_limit")
        # 只读取本进程官方EM调用帧中的标量，不输出患者或大数组。
        caller = sys._getframe(1)
        before = float(caller.f_locals["current_likelihood"])
        iteration = int(caller.f_locals["iteration"])
        with lock:
            state["em_sweeps_started"] += 1
            for key in ("current_event_rank", "events_in_sweep", "candidate_completed", "candidates_for_event"):
                state.pop(key, None)
            calls_before = state["likelihood_calls_completed"]
        emit()
        began = time.monotonic()
        result = optimise(training_data, sequence, fractions, rng)
        after = float(result[2])
        if not np.isfinite([before, after]).all():
            raise FloatingPointError("官方EM似然含非有限值")
        with lock:
            state["em_sweeps_completed"] += 1
            state["em_sweeps"].append({"iteration_zero_based": iteration, "seconds": time.monotonic() - began,
                "log_likelihood_before": before, "candidate_log_likelihood_after": after,
                "candidate_change": after - before, "sequence_changed": not np.array_equal(sequence, result[0]),
                "likelihood_calls": state["likelihood_calls_completed"] - calls_before})
        emit()
        return result

    engine._calculate_likelihood_stage = wrapped_kernel
    engine._optimise_parameters = wrapped_optimise
    thread = threading.Thread(target=heartbeat, daemon=True)
    try:
        emit()
        thread.start()
        result = engine._find_ml_iteration(data, seed_sequence)
        with lock:
            state.update(status="startpoint_finished", final_log_likelihood=float(result[2]),
                         final_sequence_sha256=hashlib.sha256(np.asarray(result[0]).tobytes()).hexdigest(),
                         final_fractions=np.asarray(result[1]).tolist())
    except ProbeLimit as exc:
        with lock:
            state.update(status=str(exc))
    except BaseException as exc:
        with lock:
            state.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        stop.set()
        if thread.ident is not None:
            thread.join()
        engine._calculate_likelihood_stage, engine._optimise_parameters = kernel, optimise
        emit()
        write_json(directory / "result.json", state)
    return state


# ===== 四、限时单进程；仅终止本工具创建的诊断子进程 =====
def child_run(cfg: dict, settings: dict, directory: Path) -> None:
    import numpy as np
    from threadpoolctl import threadpool_limits
    from pySuStaIn.ZscoreSustain import ZScoreSustainData
    from substain_sustain.model import _engine
    train, features, model, metadata = prepare(cfg, settings)
    from substain_sustain.acceleration import execution_settings
    model["_acceleration"] = execution_settings({})  # 诊断自己负责计时，避免叠加正式心跳包装器。
    write_json(directory / "input_identity.json", metadata)
    os.nice(settings["nice_increment"])
    print(f"独立计时：C3={metadata['n_c3']}，训练={metadata['n_train']}，特征={len(features)}，"
          f"事件={metadata['n_events']}，fold={settings['fold']}，startpoint={settings['startpoint']}；"
          f"最多{settings['max_wall_seconds']}秒。", flush=True)
    with threadpool_limits(limits=settings["blas_threads"]):
        engine = _engine(train, features, model, directory / "scratch", metadata["model_seed"],
                         model["cv_iterations"], 1)
        seeds = np.random.SeedSequence(engine.global_rng.integers(1e10)).spawn(model["startpoints"])
        result = observe(engine, ZScoreSustainData(train, metadata["n_events"]),
                         seeds[settings["startpoint"]], settings, directory)
    for sweep in result["em_sweeps"]:
        print(f"EM第{sweep['iteration_zero_based'] + 1}轮：{sweep['seconds']:.2f}秒；"
              f"logL {sweep['log_likelihood_before']:.6g} → {sweep['candidate_log_likelihood_after']:.6g}；"
              f"顺序变化={sweep['sequence_changed']}", flush=True)
    print("这是独立探针的证据，不能证明旧worker的迭代、收敛或剩余时间。", flush=True)


def stop_child(process, grace: float) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=grace)


def main(child_runner=child_run, default_probe_config=None, entry_script=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--probe-config", type=Path,
                        default=default_probe_config or Path(__file__).resolve().parents[1] / "config/cv_probe.yaml")
    parser.add_argument("--child-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    settings = read_settings(args.probe_config)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(settings["blas_threads"])
    os.environ["MPLBACKEND"] = "Agg"
    cfg = load_config(args.config)
    root = diagnostic_root(cfg, settings)
    if args.child_dir:
        directory = args.child_dir.resolve()
        if directory.parent != root:
            raise ValueError("子进程输出不在指定诊断目录")
        child_runner(cfg, settings, directory)
        return 0
    root.mkdir(parents=True, exist_ok=True)
    from substain_sustain.model import _lock
    with _lock(root):
        directory = root / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
        directory.mkdir()
        settings_path = directory / "settings.yaml"
        settings_path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")
        print(f"诊断记录：{directory}\n只额外运行一个限时诊断进程；现有正式任务继续运行。", flush=True)
        process = subprocess.Popen([sys.executable, str(entry_script or Path(__file__).resolve()), "--config", cfg["_config_file"],
                                    "--probe-config", str(settings_path), "--child-dir", str(directory)],
                                   cwd=cfg["project_dir"], start_new_session=True)
        receipt = {"pid": process.pid, "status": "running", "scope": "independent_probe_only"}
        write_json(directory / "launcher.json", receipt)
        try:
            code = process.wait(timeout=settings["max_wall_seconds"] + settings["termination_grace_seconds"])
            receipt.update(status="finished" if code == 0 else "failed", returncode=code)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            stop_child(process, settings["termination_grace_seconds"])
            receipt.update(status="hard_timeout" if isinstance(exc, subprocess.TimeoutExpired) else "interrupted")
            code = 2
        finally:
            write_json(directory / "launcher.json", receipt)
        print(f"诊断进程状态：{receipt['status']}；结果目录：{directory}", flush=True)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
