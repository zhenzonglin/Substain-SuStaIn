#!/usr/bin/env python3
"""工作站独立核查：原始/分块似然一致性、速度、完整单轮EM及事件进度。"""
from __future__ import annotations

import hashlib
from pathlib import Path
import statistics
import time

import numpy as np

import probe_cv
from likelihood_blocks import build_blocked_kernel


# ===== 一、每组顺序均使用全体训练样本；逐元素不一致时立即拒绝 =====
def compare_kernels(engine, dataset, settings: dict) -> dict:
    for key in ("block_rows", "benchmark_sequences", "benchmark_repeats"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key}必须为正整数")
    rng = np.random.default_rng(settings["benchmark_seed"])
    blocked = build_blocked_kernel(engine, settings["block_rows"])
    timings = []
    for index in range(settings["benchmark_sequences"]):
        sequence = engine._initialise_sequence(dataset, rng)[0]
        official_times, blocked_times = [], []
        expected = engine._calculate_likelihood_stage(dataset, sequence)
        candidate = blocked(dataset, sequence)
        if not np.isfinite(expected).all() or not np.array_equal(expected, candidate):
            raise ValueError("原始/分块似然存在非有限值或未逐元素完全相等；停止诊断，不用于正式计算")
        # 交替执行顺序，减少固定先后顺序造成的缓存与系统负载偏差。
        for repeat in range(settings["benchmark_repeats"]):
            jobs = [(engine._calculate_likelihood_stage, official_times), (blocked, blocked_times)]
            for function, samples in (jobs if repeat % 2 == 0 else jobs[::-1]):
                began = time.perf_counter()
                result = function(dataset, sequence)
                samples.append(time.perf_counter() - began)
                if not np.array_equal(result, expected):
                    raise ValueError("重复计算未保持逐元素相同，停止诊断")
        row = {"sequence_index": index, "official_seconds": statistics.median(official_times),
               "blocked_seconds": statistics.median(blocked_times), "bitwise_equal": True}
        row["speedup"] = row["official_seconds"] / row["blocked_seconds"]
        timings.append(row)
        print(f"[同机对照 {index+1}] 原始 {row['official_seconds']:.4f}s | "
              f"分块 {row['blocked_seconds']:.4f}s | {row['speedup']:.2f}倍 | 逐元素完全相同", flush=True)
    return {"all_bitwise_equal": True, "sequences": timings,
            "official_seconds_median": statistics.median(r["official_seconds"] for r in timings),
            "blocked_seconds_median": statistics.median(r["blocked_seconds"] for r in timings),
            "warning": "kernel speedup is not whole-CV speedup; independent test does not observe old workers"}


# ===== 二、仅对独立实例使用分块，继续调用原官方EM =====
def audit_run(cfg: dict, settings: dict, directory: Path) -> None:
    import os
    from threadpoolctl import threadpool_limits
    from pySuStaIn.ZscoreSustain import ZScoreSustainData
    from substain_sustain.model import _engine
    train, features, model, metadata = probe_cv.prepare(cfg, settings)
    from substain_sustain.acceleration import execution_settings
    model["_acceleration"] = execution_settings({})
    metadata.update(diagnostic_backend="verified_official_subject_row_blocks",
                    audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    blocks_source_sha256=hashlib.sha256(Path(__file__).with_name("likelihood_blocks.py").read_bytes()).hexdigest())
    probe_cv.write_json(directory / "input_identity.json", metadata)
    os.nice(settings["nice_increment"])
    print(f"独立性能核查：C3={metadata['n_c3']}，训练={metadata['n_train']}，"
          f"特征={len(features)}，事件={metadata['n_events']}；正式任务保持运行。", flush=True)
    with threadpool_limits(limits=settings["blas_threads"]):
        dataset = ZScoreSustainData(train, metadata["n_events"])
        engine = _engine(train, features, model, directory / "scratch", metadata["model_seed"],
                         model["cv_iterations"], 1)
        comparison = compare_kernels(engine, dataset, settings)
        probe_cv.write_json(directory / "kernel_comparison.json", comparison)
        print("数值对照通过；仅在独立诊断实例上测量一轮EM，不更改正式worker。", flush=True)
        engine._calculate_likelihood_stage = build_blocked_kernel(engine, settings["block_rows"])
        seeds = np.random.SeedSequence(engine.global_rng.integers(1e10)).spawn(model["startpoints"])
        result = probe_cv.observe(engine, dataset, seeds[settings["startpoint"]], settings, directory)
    for sweep in result["em_sweeps"]:
        print(f"[EM完成] 耗时 {sweep['seconds']:.1f}s | 似然调用 {sweep['likelihood_calls']}次 | "
              f"logL {sweep['log_likelihood_before']:.8g} → {sweep['candidate_log_likelihood_after']:.8g} | "
              f"顺序变化={sweep['sequence_changed']}", flush=True)
        # 只给出同一候选数下的核函数耗时量级；不外推收敛轮数、K>1或剩余时间。
        kernel_seconds = sweep["likelihood_calls"] * comparison["official_seconds_median"]
        print(f"[量级参考] 相同调用数按本次原始核函数计时约 {kernel_seconds/3600:.2f}小时；"
              "不含EM其他操作，不是旧任务的进度或剩余时间。", flush=True)
    print(f"诊断结果：{directory}；旧worker内部状态仍未直接观测。", flush=True)


if __name__ == "__main__":
    raise SystemExit(probe_cv.main(child_runner=audit_run,
        default_probe_config=Path(__file__).resolve().parents[1] / "config/cv_performance.yaml",
        entry_script=Path(__file__).resolve()))
