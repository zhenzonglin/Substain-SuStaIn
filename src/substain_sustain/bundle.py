"""数值源驱动的可移动展示包：独立解包、校验、重新绘图，不依赖原影像目录。"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile

from .assets import ASSET_KEYS, file_sha256


def _save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _pixel_record(path: Path) -> dict:
    """PNG像素校验不依赖文件元数据，TIFF另外检查600DPI声明。"""
    import hashlib
    from PIL import Image
    with Image.open(path) as image:
        result = {"size": list(image.size), "mode": image.mode}
        if path.suffix.lower() == ".png":
            result["pixel_sha256"] = hashlib.sha256(image.convert("RGBA").tobytes()).hexdigest()
        if path.suffix.lower() in {".tiff", ".tif"}:
            result["dpi"] = [float(value) for value in image.info.get("dpi", (0, 0))]
        return result


def _eligible_data(output: Path, include_posterior: bool) -> list[Path]:
    """白名单只保留轻量输入、参考、评估与完整个人概率；不拷贝MCMC全链。"""
    selected = []
    if (output / "PROVENANCE.json").is_file():
        selected.append(output / "PROVENANCE.json")
    for relative in ("inputs", "reference", "model/cv", "model/fit", "validation", "execution"):
        folder = output / relative
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.is_symlink():
                continue
            # 压缩后缀不能作为数据白名单：NIfTI患者影像同样以.gz结尾。
            allowed_compressed_evidence = relative == "inputs" and path.name.endswith("_raw_records.jsonl.gz")
            if path.suffix not in {".tsv", ".json", ".npz"} and not allowed_compressed_evidence:
                continue
            if path.suffix == ".npz" and path.name.startswith("posterior_") and not include_posterior:
                continue
            selected.append(path)
    return selected


def _validate_probability_files(output: Path) -> list[str]:
    """主分析与全龄敏感性四份概率均需完整，且与对应分配表逐行同一身份。"""
    import numpy as np
    import pandas as pd
    required = []
    for cohort, relative in (("C3", "model/fit"), ("GB", "validation")):
        for scope in ("primary", "all"):
            posterior = output / relative / f"posterior_{cohort}_{scope}.npz"
            assignment = output / relative / f"assignments_{cohort}_{scope}.tsv"
            if not posterior.is_file() or not assignment.is_file():
                raise FileNotFoundError(f"Complete display delivery requires {cohort} {scope} probabilities and assignments")
            frame = pd.read_csv(assignment, sep="\t", dtype={"subject_key": str})
            if "subject_key" not in frame or frame.subject_key.duplicated().any():
                raise ValueError("Invalid subject identities in " + str(assignment))
            with np.load(posterior, allow_pickle=False) as source:
                fields = {"subject_keys", "posterior_joint", "posterior_subtype", "posterior_stage"}
                if not fields.issubset(source.files):
                    raise ValueError("Incomplete individual posterior fields: " + str(posterior))
                keys = np.asarray(source["subject_keys"]).astype(str)
                joint = np.asarray(source["posterior_joint"], dtype=float)
                if not np.array_equal(keys, frame.subject_key.to_numpy(dtype=str)):
                    raise ValueError("Posterior subject_keys differ from assignment rows: " + str(posterior))
                if joint.ndim != 3 or joint.shape[0] != len(frame) or not np.isfinite(joint).all() or np.any(joint < 0):
                    raise ValueError("Invalid full joint posterior: " + str(posterior))
                if not np.allclose(joint.sum(axis=(1, 2)), 1, atol=1e-8):
                    raise ValueError("Individual joint probabilities do not sum to one: " + str(posterior))
                for key, axis in (("posterior_subtype", 1), ("posterior_stage", 2)):
                    marginal = np.asarray(source[key], dtype=float)
                    if marginal.shape != joint.sum(axis=axis).shape or not np.allclose(marginal, joint.sum(axis=axis), atol=1e-8):
                        raise ValueError("Posterior marginal disagrees with joint: " + str(posterior))
            required.append(posterior.relative_to(output).as_posix())
    return required


def create_bundle(cfg: dict, output: Path, destination: Path) -> dict:
    """归档足以重建全部图表的源文件，包内无原始MRI、分割权重或已渲染图片。"""
    output, destination = Path(output), Path(destination)
    if destination.is_dir() or not destination.suffix:
        destination = destination / "sustain-display-bundle.tar.gz"
    if destination.exists():
        raise FileExistsError("Refusing to replace an existing display bundle: " + str(destination))
    bundle_cfg = cfg.get("bundle", {})
    if bundle_cfg.get("compression", "gz") != "gz":
        raise ValueError("bundle.compression currently supports gz only")
    if not bundle_cfg.get("include_full_posterior", True):
        raise ValueError("Complete display delivery requires full individual posterior probabilities")
    summary_path = output / "report/summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Run the report before creating a bundle")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") not in {"complete", "technical_demo_complete"}:
        raise ValueError("Cannot bundle an incomplete report")
    probability_files = _validate_probability_files(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_project = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="sustain-bundle-", dir=str(destination.parent)) as temporary:
        staging = Path(temporary) / "sustain-display"
        staging.mkdir()
        for source in _eligible_data(output, True):
            target = staging / "data" / source.relative_to(output)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        portable_cfg = {key: copy.deepcopy(cfg[key]) for key in ("runtime", "reference", "report", "bundle", "feature_selection") if key in cfg}
        portable_cfg["assets"] = {}
        for key in ASSET_KEYS:
            source = Path(cfg["assets"][key])
            filename = key + (".nii.gz" if source.name.endswith(".nii.gz") else source.suffix)
            target = staging / "assets" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            portable_cfg["assets"][key] = "assets/" + filename
        shutil.copyfile(Path(cfg["assets"]["manifest"]), staging / "assets/manifest.json")
        portable_cfg["assets"]["manifest"] = "assets/manifest.json"
        _save_json(staging / "config.json", portable_cfg)
        provenance = staging / "provenance"
        provenance.mkdir()
        if (source_project / "vendor/manifest.json").is_file():
            shutil.copyfile(source_project / "vendor/manifest.json", provenance / "upstream_manifest.json")
        mapping = cfg.get("paths", {}).get("mapping")
        if mapping and Path(mapping).is_file():
            shutil.copyfile(mapping, provenance / "macro_mapping.tsv")
        # 仅保留重画所需Python模块，确保运行不意外回到原始安装目录。
        code_dir = staging / "src/substain_sustain"
        code_dir.mkdir(parents=True)
        (code_dir / "__init__.py").write_text('"""Portable SuStaIn report reproduction code."""\n', encoding="utf-8")
        for name in ("report.py", "assets.py", "bundle.py", "feature_selection.py"):
            shutil.copyfile(Path(__file__).with_name(name), code_dir / name)
        dependency_found = False
        for name in ("requirements.lock", "requirements.txt", "requirements-report.txt", "pyproject.toml"):
            if (source_project / name).is_file():
                shutil.copyfile(source_project / name, staging / name)
                dependency_found = True
        if not dependency_found:
            (staging / "requirements-report.txt").write_text("numpy\npandas\nmatplotlib\nnibabel\nscipy\npillow\n", encoding="utf-8")
        versions = {"python": sys.version.split()[0]}
        from importlib.metadata import version
        for name in ("numpy", "pandas", "matplotlib", "nibabel", "scipy", "pillow"):
            versions[name] = version(name)
        _save_json(staging / "runtime_versions.json", versions)
        (staging / "requirements-report.txt").write_text(
            "\n".join(f"{name}=={value}" for name, value in versions.items() if name != "python") + "\n", encoding="utf-8")
        contract = {"figures": summary["figures"], "tables": {}, "images": {}, "artifacts": []}
        for artifact in summary["artifacts"]:
            relative = artifact["path"]
            path = output / "report" / relative
            if file_sha256(path) != artifact["sha256"]:
                raise ValueError("Report artifact changed since render: " + relative)
            contract["artifacts"].append(relative)
            if path.suffix == ".tsv":
                contract["tables"][relative] = artifact["sha256"]
            elif path.suffix in {".png", ".tiff"}:
                contract["images"][relative] = _pixel_record(path)
        _save_json(staging / "reproduction_contract.json", contract)
        script = '''#!/usr/bin/env python3
"""从本包数值源和图谱重新生成全部报告，不访问原患者目录。"""
import json
from pathlib import Path
import socket
import sys

root = Path(__file__).resolve().parent
# 仅约束当前重画Python进程；不修改工作站的网络配置。
class NetworkDisabled(RuntimeError):
    pass

def deny_network(*args, **kwargs):
    raise NetworkDisabled("Network access is disabled during detached report reproduction")

socket.getaddrinfo = deny_network
socket.create_connection = deny_network
socket.socket.connect = deny_network
socket.socket.connect_ex = deny_network
blocked = []
for name, action in (
    ("getaddrinfo", lambda: socket.getaddrinfo("example.invalid", 443)),
    ("create_connection", lambda: socket.create_connection(("example.invalid", 443))),
):
    try:
        action()
    except NetworkDisabled:
        blocked.append(name)
    else:
        raise RuntimeError("Network isolation self-test failed: " + name)
with socket.socket() as probe:
    for name in ("connect", "connect_ex"):
        try:
            getattr(probe, name)(("127.0.0.1", 9))
        except NetworkDisabled:
            blocked.append(name)
        else:
            raise RuntimeError("Network isolation self-test failed: " + name)
isolation = {"network_disabled": True, "blocked_self_tests": blocked,
             "scope": "Python socket DNS/connect APIs in the detached report process"}
(root / "NETWORK_ISOLATION.json").write_text(json.dumps(isolation, indent=2), encoding="utf-8")
sys.path.insert(0, str(root / "src"))
from substain_sustain.report import run_report
cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
for key, relative in cfg["assets"].items():
    cfg["assets"][key] = str(root / relative)
result = run_report(cfg, root / "data", bundle_only=True)
print(json.dumps({"status": result["status"], "figures": result["figures"], "network_disabled": True}))
'''
        (staging / "rebuild_report.py").write_text(script, encoding="utf-8")
        (staging / "README.txt").write_text(
            "SuStaIn portable display bundle\n"
            "Install the pinned display dependencies using pip install -r requirements-report.txt.\n"
            "Run python rebuild_report.py to regenerate data/report/index.html and every figure/table.\n"
            "No original MRI or original project directory is needed. The archive contains numeric sources,\n"
            "public display atlases, deterministic posterior display draws, and complete individual probabilities.\n"
            "Full MCMC chains and all fitted seed/fold models must remain in the original workstation archive.\n"
            "Do not interpret a synthetic demonstration as a cohort analysis.\n", encoding="utf-8")
        records = [{"path": path.relative_to(staging).as_posix(), "sha256": file_sha256(path), "bytes": path.stat().st_size}
                   for path in sorted(staging.rglob("*")) if path.is_file()]
        manifest = {"schema_version": 1, "report_status": summary["status"], "files": records,
                    "contains_rendered_figures": False, "contains_full_mcmc_chains": False,
                    "contains_complete_individual_probabilities": True, "probability_files": probability_files}
        _save_json(staging / "BUNDLE_MANIFEST.json", manifest)
        archive = Path(temporary) / "archive.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(staging, arcname="sustain-display", recursive=True)
        os.replace(archive, destination)
    receipt = {"status": "created", "path": str(destination), "sha256": file_sha256(destination),
               "bytes": destination.stat().st_size, "files": len(records), "report_status": summary["status"]}
    _save_json(destination.with_name(destination.name + ".json"), receipt)
    return receipt


def _safe_extract(archive: Path, destination: Path) -> Path:
    """拒绝越界、链接、重复成员与设备文件，随后逐项核验清单。"""
    with tarfile.open(archive, "r:gz") as tar:
        names = set()
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "sustain-display":
                raise ValueError("Unsafe archive member: " + member.name)
            if member.name in names or member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise ValueError("Unsupported or duplicate archive member: " + member.name)
            names.add(member.name)
            target = destination.joinpath(*path.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = tar.extractfile(member)
                if stream is None:
                    raise ValueError("Unreadable archive member: " + member.name)
                with stream, target.open("wb") as handle:
                    shutil.copyfileobj(stream, handle)
    return destination / "sustain-display"


def verify_bundle(bundle_path: Path) -> dict:
    """在临时独立目录中真实重画全部图，验证数值表、PNG像素及TIFF尺寸/DPI。"""
    bundle_path = Path(bundle_path).resolve()
    with tempfile.TemporaryDirectory(prefix="sustain-reproduce-") as temporary:
        root = _safe_extract(bundle_path, Path(temporary))
        manifest = json.loads((root / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
        expected_files = {record["path"] for record in manifest["files"]}
        actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()} - {"BUNDLE_MANIFEST.json"}
        if expected_files != actual_files:
            raise ValueError("Bundle manifest member mismatch")
        for record in manifest["files"]:
            if file_sha256(root / record["path"]) != record["sha256"]:
                raise ValueError("Bundle SHA256 mismatch: " + record["path"])
        _validate_probability_files(root / "data")
        if any((root / "data/report").glob("**/*")):
            raise ValueError("Reproduction must start without rendered report artifacts")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root / "src")
        env["PYTHONNOUSERSITE"] = "1"
        env["MPLCONFIGDIR"] = str(root / ".mplconfig")
        env["HF_HUB_OFFLINE"] = "1"
        # 在独立cwd运行包内脚本；报告只接收包内相对源数据和图谱。
        process = subprocess.run([sys.executable, str(root / "rebuild_report.py")], cwd=root, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if process.returncode:
            raise RuntimeError("Detached report regeneration failed:\n" + process.stderr[-6000:])
        isolation = json.loads((root / "NETWORK_ISOLATION.json").read_text(encoding="utf-8"))
        expected_blocks = {"getaddrinfo", "create_connection", "connect", "connect_ex"}
        if isolation.get("network_disabled") is not True or set(isolation.get("blocked_self_tests", [])) != expected_blocks:
            raise ValueError("Detached network isolation self-test did not pass")
        contract = json.loads((root / "reproduction_contract.json").read_text(encoding="utf-8"))
        report = root / "data/report"
        summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
        if summary["figures"] != contract["figures"]:
            raise ValueError("Detached figure count differs")
        for relative in contract["artifacts"]:
            if not (report / relative).is_file():
                raise FileNotFoundError("Regenerated report lacks " + relative)
        for relative, checksum in contract["tables"].items():
            if file_sha256(report / relative) != checksum:
                raise ValueError("Regenerated numeric table differs: " + relative)
        for relative, expected in contract["images"].items():
            observed = _pixel_record(report / relative)
            if observed != expected:
                raise ValueError("Regenerated image pixels/size/DPI differ: " + relative)
        receipt = {"status": "pass", "bundle_sha256": file_sha256(bundle_path), "files_verified": len(expected_files),
                   "figures_regenerated": summary["figures"], "tables_verified": len(contract["tables"]),
                   "images_verified": len(contract["images"]), "original_mri_required": False,
                   "original_project_required": False, "rendered_images_in_bundle": False,
                   "network_disabled": True, "network_guard_scope": isolation["scope"],
                   "network_self_tests": isolation["blocked_self_tests"],
                   "report_status": summary["status"]}
    _save_json(bundle_path.with_name(bundle_path.name + ".verification.json"), receipt)
    return receipt
