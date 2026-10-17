#!/usr/bin/env python3
"""只处理配置指定的公开模板，建立ROI展示资源；不扫描任何患者目录。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from substain_sustain.config import load_config

# ===== 显式配置与公开模板来源检查 =====
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", required=True)
args = parser.parse_args()
cfg = load_config(args.config)
assets = cfg["assets"]
available_kb = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:"))
if available_kb / 1024**2 < assets["template_min_free_memory_gb"]:
    raise SystemExit("公开模板生成前可用内存低于配置门槛")
if assets.get("public_template_kind") != "public_template":
    raise SystemExit("必须显式声明public_template_kind=public_template")
source = Path(assets["source_background"]).resolve()
work = Path(assets["template_work_dir"]).resolve()
work.mkdir(parents=True, exist_ok=True)
tool_output = work / "output"
tool_output.mkdir(exist_ok=True)
expected = Path(assets["source_gm_segmentation"])
digest = hashlib.sha256(source.read_bytes()).hexdigest()
record_path = work / "public_template_provenance.json"
if expected.is_file():
    if not record_path.is_file() or json.loads(record_path.read_text())["template_sha256"] != digest:
        raise SystemExit("已有图谱缺少匹配来源，不覆盖")
    print("公开模板分割已存在且来源匹配")
    raise SystemExit(0)
# 显式复制公开模板到缓存，避免上游工具触碰唯一源文件。
input_dir = work / "input"
input_dir.mkdir(exist_ok=True)
template = input_dir / "ch2better.nii.gz"
shutil.copy2(source, template)
env = os.environ.copy()
env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
env["OMP_NUM_THREADS"] = str(assets["template_threads"])
env["MKL_NUM_THREADS"] = str(assets["template_threads"])
env["PATH"] = str(Path(assets["template_python"]).parent) + os.pathsep + env.get("PATH", "")
# ===== 调用现有官方分割环境；仅一个公开模板 =====
entry = str(Path(assets["template_entry"]).resolve())
command = [entry, "-i", str(template), "-o", str(tool_output), "-d", assets["template_device"],
           *[str(x) for x in assets["template_cli_args"]]]
record = {"source_kind": "public_template", "template_sha256": digest,
          "source": str(source), "command": command,
          "entry_sha256": hashlib.sha256(Path(entry).read_bytes()).hexdigest()}
record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
with (work / "generation.log").open("w") as log:
    proc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
if proc.returncode or not expected.is_file():
    raise SystemExit(f"公开模板生成失败，exit={proc.returncode}；查看{work / 'generation.log'}")
record["segmentation_sha256"] = hashlib.sha256(expected.read_bytes()).hexdigest()
record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
print(f"公开模板分割完成：{expected}")
