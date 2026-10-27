#!/usr/bin/env python3
"""只读核验零似然、冻结链对数值、极端z与LS残差尺度；正式缓存不改写。"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import uuid
import yaml
from substain_sustain.config import load_config


# ===== 所有诊断路径与参数均来自配置 =====
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--settings", type=Path, default=Path(__file__).resolve().parents[1]/"config/likelihood_diagnostic.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    spec = yaml.safe_load(args.settings.read_text())
    root, formal = Path(cfg["project_dir"]), Path(cfg["output_dir"])
    destination = (root/spec["output_dir"]).resolve()
    if not destination.is_relative_to(root) or destination == root or destination.is_relative_to(formal) or formal.is_relative_to(destination):
        raise ValueError("诊断输出必须在项目内部，并与正式结果目录分开")
    for cohort in cfg["cohorts"].values():
        for key in ("derivatives", "source_root"):
            if cohort.get(key) and destination.is_relative_to(Path(cohort[key]).resolve()):
                raise ValueError("诊断目录不能位于队列源数据内")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(cfg["runtime"]["blas_threads"])
    import switch_acceleration as switch
    settings = yaml.safe_load((root/spec["migration_settings"]).read_text())
    if type(spec["audit_max_k"]) is not int or not 1 <= spec["audit_max_k"] <= cfg["model"]["k_max"]:
        raise ValueError("audit_max_k超出候选模型范围")
    settings["audit_max_k"] = spec["audit_max_k"]
    target = switch.target_config(cfg, settings)
    switch.plan(target, settings)
    records = switch.audit_cv_caches(target, settings)
    directory = destination/(datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    receipt = {"scope": "read_only_frozen_cache_diagnostic", "formal_files_modified": False, "records": records}
    switch.parallel.atomic_bytes(directory/"diagnosis.json", switch.parallel.encoded(receipt))
    for record in records:
        value = record["numerical_evaluation"]
        print(json.dumps({"fold": record["fold"], "K": record["k"], **value}, ensure_ascii=False), flush=True)
    print(f"只读诊断完成：{directory/'diagnosis.json'}；未停止任务、未改写配置或缓存。")


if __name__ == "__main__":
    main()
