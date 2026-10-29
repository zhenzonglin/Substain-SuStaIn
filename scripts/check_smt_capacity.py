#!/usr/bin/env python3
"""只读估算SMT并发；兼容7e38ac1，可从git show单独取出而不更新运行中的源码。"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import yaml


# ===== 一、配置输入；先设置BLAS限制，再加载数值库 =====
def capacity(cfg: dict, settings: dict) -> dict:
    from substain_sustain.model import _features, _read_z, _resources

    target = copy.deepcopy(cfg)
    requested = copy.deepcopy(settings["startpoint_parallel"])
    if requested.get("worker_unit") != "logical_cpu" or not requested["enabled"]:
        raise ValueError("容量检查要求logical_cpu模式并启用起点并行")
    # 旧版预算函数只认识物理核模式。借用其单折内存估算，独立计算逻辑CPU槽位上限。
    # 不修改cfg，不调用身份迁移，不将此估算写入正式resources.json。
    target["startpoint_parallel"] = {key: value for key, value in requested.items() if key != "worker_unit"}
    output = Path(target["output_dir"])
    features = _features(output, target)
    frame, _ = _read_z(output / "reference/C3_primary_z.tsv", features)
    model = target["model"]
    resource = _resources(target, output / "model/cv", len(frame), len(features),
                          model["cv_iterations"], model["k_max"], model["folds"], model["folds"])
    topology = resource["cpu_topology"]
    per_fold = resource["startpoint_workers_per_model"]
    requested_folds = min(model["workers"], model["folds"])
    logical = topology["logical_cpus"]
    quota = topology.get("quota_cores")
    quota_ok = quota is None or quota >= logical
    available_folds = min(requested_folds, logical // per_fold,
                          resource["memory_budget_bytes"] // resource["estimated_worker_bytes"])
    if not quota_ok:
        available_folds = min(available_folds, int(quota) // per_fold)
    allowed = (quota_ok and available_folds == requested_folds and per_fold == requested["workers_per_fold"])
    requested_worker_bytes = max(resource["estimated_worker_bytes"], resource["estimated_parent_bytes"]
                                 + requested["workers_per_fold"] * resource["estimated_em_worker_bytes"])
    gib = 1024**3
    return {"read_only": True, "config": cfg.get("_config_file"),
            "subjects": len(frame), "features": len(features),
            "physical_cores": topology["physical_cores"], "logical_cpus": logical,
            "requested_folds": requested_folds, "requested_startpoints_per_fold": requested["workers_per_fold"],
            "requested_compute_processes": requested_folds * requested["workers_per_fold"],
            "allowed_folds": int(available_folds), "allowed_startpoints_per_fold": per_fold,
            "allowed_compute_processes": int(available_folds) * per_fold,
            "requested_peak_estimate_gib": requested_worker_bytes * requested_folds / gib,
            "memory_budget_gib": resource["memory_budget_bytes"] / gib,
            "memory_safety_factor": cfg["runtime"]["memory_safety_factor"],
            "quota_cpu_units": quota, "full_request_allowed": bool(allowed),
            "meaning": "保守执行资源估算；不是实测峰值、速度测试或缓存完整性验收"}


# ===== 二、只打印结果；不停止任务，不写配置、模型身份或任何正式结果 =====
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    from substain_sustain.config import load_config
    cfg = load_config(args.config)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(cfg["runtime"]["blas_threads"])
    settings = yaml.safe_load(args.settings.read_text(encoding="utf-8"))
    result = capacity(cfg, settings)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(f"请求{result['requested_folds']}折×{result['requested_startpoints_per_fold']}起点"
          f"={result['requested_compute_processes']}；当前预算允许最多"
          f"{result['allowed_folds']}折×{result['allowed_startpoints_per_fold']}起点"
          f"={result['allowed_compute_processes']}。", flush=True)
    if not result["full_request_allowed"]:
        print("当前不满足请求上限；建议保留现有任务，不执行60进程切换。", flush=True)
    print("已完成只读检查：未停止任务、未更新运行源码、未改配置或缓存。", flush=True)


if __name__ == "__main__":
    main()
