"""中文分区的统一入口：配置、环境检查、逐阶段执行与审计日志。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback

from .config import load_config

# ===== 环境诊断：不启动分割或正式拟合 =====
def doctor(cfg: dict) -> dict:
    import importlib.metadata as metadata
    import psutil

    root = Path(cfg["project_dir"])
    checks = {"python": platform.python_version(), "platform": platform.platform(),
              "architecture": platform.machine(), "available_memory_gb": psutil.virtual_memory().available / 2**30,
              "profile": cfg["runtime"]["profile"], "sources": {}, "packages": {}, "errors": []}
    if sys.version_info[:2] != (3, 10):
        checks["errors"].append("正式环境要求Python 3.10")
    for name in ("numpy", "scipy", "pandas", "statsmodels", "patsy", "scikit-learn",
                 "nibabel", "matplotlib", "pySuStaIn", "awkde", "kde_ebm"):
        try:
            checks["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            checks["errors"].append(f"缺少依赖{name}")
    for cohort, spec in cfg["cohorts"].items():
        keys = ("input_bundle",) if spec.get("input_bundle") else ("derivatives", "participants", "metadata")
        checks["sources"][cohort] = {key: {"path": spec.get(key), "exists": Path(spec[key]).exists()}
                                    for key in keys if spec.get(key)}
    # 锁定依赖版本也属于推断身份；不仅显示版本，更检测用户环境漂移。
    lock = root / "requirements.lock"
    if lock.is_file():
        for line in lock.read_text(encoding="utf-8-sig").splitlines():
            if "==" not in line or line.startswith("#"):
                continue
            name, expected = line.strip().split("==", 1)
            try:
                actual = metadata.version(name)
                if actual != expected:
                    checks["errors"].append(f"依赖版本不符：{name}={actual}，要求{expected}")
            except metadata.PackageNotFoundError:
                checks["errors"].append(f"缺少锁定依赖：{name}")
    else:
        checks["errors"].append("缺少requirements.lock")
    manifest = root / "vendor" / "manifest.json"
    if manifest.is_file():
        pinned = json.loads(manifest.read_text())
        try:
            module = importlib.import_module("pySuStaIn")
            package_root = Path(module.__file__).parent
            for name, expected in pinned.get("pysustain_source_sha256", {}).items():
                got = hashlib.sha256((package_root / name).read_bytes()).hexdigest()
                if got != expected:
                    checks["errors"].append(f"官方源码校验失败：{name}")
            checks["upstream_commit"] = pinned["packages"]["pySuStaIn"]["commit"]
        except (ImportError, OSError) as exc:
            checks["errors"].append(str(exc))
    else:
        checks["errors"].append("缺少vendor/manifest.json")
    checks["software_ready"] = not checks["errors"]
    checks["cohort_paths_ready"] = all(
        (("input_bundle" in values) or {"derivatives", "participants"}.issubset(values))
        and all(v["exists"] for v in values.values()) for values in checks["sources"].values()
    )
    from .assets import validate_assets
    try:
        validate_assets(cfg)
        checks["display_assets_ready"] = True
    except (FileNotFoundError, ValueError, OSError) as exc:
        checks["display_assets_ready"] = False
        checks["display_asset_error"] = str(exc)
    checks["workflow_ready"] = checks["software_ready"] and checks["cohort_paths_ready"] and checks["display_assets_ready"]
    return checks


# ===== 稳定命令接口与配置快照 =====
STAGES = {
    "ingest": ("ingest", "run_ingest"),
    "reference": ("reference", "run_reference"),
    "crossvalidate": ("model", "run_crossvalidate"),
    "fit": ("model", "run_fit"),
    "validate": ("model", "run_validate"),
    "report": ("report", "run_report"),
}


def _execute(command: str, cfg: dict, output: Path) -> dict:
    module, name = STAGES[command]
    return getattr(importlib.import_module(f"substain_sustain.{module}"), name)(cfg, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LS–C3–GB WMH/GM SuStaIn独立下游分析")
    parser.add_argument("command", choices=["doctor", *STAGES, "all", "export-inputs", "bundle", "verify-bundle"])
    parser.add_argument("--config", default="config/analysis.yaml", help="包含路径与所有参数的YAML文件")
    parser.add_argument("--cohort", choices=["LS", "C3", "GB"], help="export-inputs队列")
    parser.add_argument("--destination", type=Path, help="导出数据包或展示包位置")
    parser.add_argument("--bundle", type=Path, help="verify-bundle待验收展示包")
    parser.add_argument("--software-only", action="store_true", help="doctor仅以软件环境决定退出码，仍显示数据/图谱状态")
    args = parser.parse_args(argv)
    if args.command == "verify-bundle":
        if args.bundle is None:
            parser.error("verify-bundle需要--bundle")
        from .bundle import verify_bundle
        print(json.dumps(verify_bundle(args.bundle), ensure_ascii=False, indent=2))
        return 0
    cfg = load_config(args.config)
    # 在加载numpy/BLAS前限定线程；实际进程数由模型模块控制。
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(cfg["runtime"]["blas_threads"])
    os.environ["MPLBACKEND"] = "Agg"
    if args.command == "doctor":
        result = doctor(cfg)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["software_ready" if args.software_only else "workflow_ready"] else 2
    if args.command == "all":
        readiness = doctor(cfg)
        if not readiness["workflow_ready"]:
            print(json.dumps(readiness, ensure_ascii=False, indent=2), file=sys.stderr)
            return 2
    output = Path(cfg["output_dir"])
    from .ingest import guard_output
    guard_output(cfg, output)
    output.mkdir(parents=True, exist_ok=True)
    commands = list(STAGES) if args.command == "all" else [args.command]
    for command in commands:
        # 每次执行单独记录配置，失败也保留原始错误，绝不标为完成。
        snapshot = {k: v for k, v in cfg.items() if not k.startswith("_")}
        encoded = json.dumps(snapshot, sort_keys=True, ensure_ascii=False).encode()
        identity = hashlib.sha256(encoded).hexdigest()
        records = output / "execution"
        records.mkdir(exist_ok=True)
        (records / f"config_{identity}.json").write_bytes(encoded)
        record = {"command": command, "config_sha256": identity,
                  "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
        status_path = records / f"{command}.json"
        status_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"开始：{command}", flush=True)
        try:
            if command == "export-inputs":
                if not args.cohort or args.destination is None:
                    raise ValueError("export-inputs需要--cohort及--destination")
                from .ingest import export_inputs
                result = export_inputs(cfg, args.cohort, args.destination)
            elif command == "bundle":
                from .bundle import create_bundle
                destination = args.destination or Path(cfg["report"]["bundle_dir"])
                result = create_bundle(cfg, output, destination)
            else:
                result = _execute(command, cfg, output)
            record.update(status="complete", result=result)
        except Exception as exc:
            record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            (records / f"{command}_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"{command}失败：{exc}", file=sys.stderr)
            return_code = 1
        else:
            return_code = 0
        finally:
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            status_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        if return_code:
            return return_code
        print(f"完成：{command}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
