"""官方函数外围的实时进度；不修改优化、采样、随机流及返回值。"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import uuid
import numpy as np
import psutil


# ===== 一、独立心跳；活动时间不冒充算法推进 =====
class Progress:
    def __init__(self, engine, directory: Path, settings: dict):
        self.engine, self.directory, self.settings = engine, Path(directory), settings
        self.lock, self.depth = threading.RLock(), 0
        self.started = self.last_work = time.monotonic()
        self.state = {"schema": 1, "session_id": uuid.uuid4().hex, "pid": os.getpid(),
            "process_created": psutil.Process().create_time(), "target_k": int(engine.N_S_max),
            "seed": int(engine.seed), "backend": settings["backend"], "block_rows": settings["block_rows"],
            "status": "initialized", "likelihood_calls_completed": 0, "em_fits": 0,
            "em_sweeps_completed": 0, "optimization_pass": 0, "startpoints_completed": 0,
            "startpoints_total": int(engine.N_startpoints)}

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def emit(self, event=None):
        with self.lock:
            value = {**self.state, "updated_utc": datetime.now(timezone.utc).isoformat(),
                     "elapsed_seconds": time.monotonic()-self.started,
                     "work_age_seconds": time.monotonic()-self.last_work}
            path = self.directory / self.settings["progress_filename"]
            temp = path.with_name(path.name + f".tmp.{os.getpid()}")
            temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            temp.replace(path)
            if event:
                with (self.directory / self.settings["events_filename"]).open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({**value, "event": event}, ensure_ascii=False, allow_nan=False)+"\n")

    def heartbeat(self):
        while not self.stop.wait(self.settings["heartbeat_seconds"]):
            try:
                self.emit()
            except Exception as exc:
                self.error = exc
                print(f"SuStaIn进度记录失败：{exc}", file=sys.stderr, flush=True)
                return

    def healthy(self):
        if getattr(self, "error", None) is not None:
            raise RuntimeError("进度记录器失败，详见stderr") from self.error

    @contextmanager
    def active(self, phase):
        outer = self.depth == 0
        if outer:
            self.stop = threading.Event()
            self.update(status="running", phase=phase)
            self.emit("operation_started")
            thread = threading.Thread(target=self.heartbeat, daemon=True)
            thread.start()
        self.depth += 1
        try:
            yield
            self.healthy()
        except BaseException as exc:
            if outer:
                self.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise
        else:
            if outer:
                self.update(status="operation_complete", phase=phase)
        finally:
            self.depth -= 1
            if outer:
                self.stop.set()
                thread.join()
                self.emit("operation_finished")


# ===== 二、只读官方调用帧的计数，不复制或更改EM/MCMC =====
def install_progress(engine, directory: Path, settings: dict):
    tracker = Progress(engine, directory, settings)
    engine._progress_tracker = tracker
    original_kernel = engine._calculate_likelihood_stage

    def kernel(data, sequence):
        tracker.healthy()
        frame, values = sys._getframe(1), {}
        if frame.f_code.co_name == "_optimise_parameters":
            local = frame.f_locals
            if all(k in local for k in ("order_bio", "i", "index", "possible_positions")):
                values.update(event_rank=int(np.flatnonzero(np.asarray(local["order_bio"]) == local["i"])[0])+1,
                              event_total=len(local["order_bio"]), candidate=int(local["index"])+1,
                              candidate_total=len(local["possible_positions"]))
        elif frame.f_code.co_name == "_calculate_likelihood" and frame.f_back:
            parent = frame.f_back
            if parent.f_code.co_name == "_perform_mcmc":
                values["mcmc_iteration_current"] = int(parent.f_locals["i"])+1
        result = original_kernel(data, sequence)
        with tracker.lock:
            tracker.state.update(values)
            tracker.state["likelihood_calls_completed"] += 1
            tracker.last_work = time.monotonic()
        return result

    engine._calculate_likelihood_stage = kernel
    original_opt = engine._optimise_parameters

    def optimise(data, sequence, fractions, rng):
        frame = sys._getframe(1)
        before = float(frame.f_locals.get("current_likelihood", float("nan")))
        iteration = int(frame.f_locals.get("iteration", -1))+1
        tracker.update(phase="EM", em_iteration=iteration, log_likelihood_before=before if math.isfinite(before) else None,
                       event_rank=0, event_total=data.getNumStages(), candidate=0, candidate_total=0)
        tracker.emit("em_sweep_started")
        began = time.monotonic()
        result = original_opt(data, sequence, fractions, rng)
        after = float(result[2])
        with tracker.lock:
            tracker.state["em_sweeps_completed"] += 1
            tracker.state.update(candidate_log_likelihood=after if math.isfinite(after) else None,
                                 last_em_seconds=time.monotonic()-began)
            tracker.last_work = time.monotonic()
        tracker.emit("em_sweep_finished")
        return result

    engine._optimise_parameters = optimise

    def wrap(name, kind):
        original = getattr(engine, name)
        def call(*args, **kwargs):
            with tracker.active(kind):
                if kind == "optimization":
                    tracker.update(optimization_pass=tracker.state["optimization_pass"]+1,
                                   optimization_method=name, startpoint=0, startpoints_total=engine.N_startpoints)
                elif kind == "startpoint":
                    seed = args[-1] if args else kwargs["seed_seq"]
                    tracker.update(phase="startpoint", startpoint=int(seed.spawn_key[-1])+1 if seed.spawn_key else 1,
                                   startpoint_method=name, em_iteration=0)
                    tracker.emit("startpoint_started")
                elif kind == "EM":
                    tracker.update(phase="EM", em_fits=tracker.state["em_fits"]+1,
                                   fitting_subtypes=int(np.asarray(args[1]).shape[0]),
                                   training_rows=int(args[0].getNumSamples()), em_iteration=0)
                elif kind == "MCMC":
                    caller = sys._getframe(1)
                    while caller and caller.f_code.co_name not in ("_optimise_mcmc_settings", "_estimate_uncertainty_sustain_model"):
                        caller = caller.f_back
                    tuning = caller is not None and caller.f_code.co_name == "_optimise_mcmc_settings"
                    count = int(args[3] if len(args) > 3 else kwargs["n_iterations"])
                    tracker.update(phase="mcmc_tuning" if tuning else "mcmc", mcmc_iteration_current=0,
                                   mcmc_iterations=count, tuning_pass=int(caller.f_locals["i"])+1 if tuning else None)
                    tracker.emit("mcmc_started")
                elif kind in ("heldout_likelihood", "inference"):
                    tracker.update(phase=kind)
                result = original(*args, **kwargs)
                if kind == "startpoint":
                    tracker.update(startpoints_completed=tracker.state["startpoints_completed"]+1)
                    tracker.emit("startpoint_finished")
                elif kind == "MCMC":
                    tracker.update(mcmc_iteration_current=tracker.state["mcmc_iterations"])
                    tracker.emit("mcmc_finished")
                return result
        setattr(engine, name, call)

    for name in ("run_sustain_algorithm", "cross_validate_sustain_model"):
        wrap(name, "training")
    for name in ("_find_ml", "_find_ml_split", "_find_ml_mixture"):
        wrap(name, "optimization")
    for name in ("_find_ml_iteration", "_find_ml_split_iteration", "_find_ml_mixture_iteration"):
        wrap(name, "startpoint")
    wrap("_perform_em", "EM")
    wrap("_perform_mcmc", "MCMC")
    wrap("_evaluate_likelihood_setofsamples", "heldout_likelihood")
    wrap("subtype_and_stage_individuals_newData", "inference")
    return tracker
