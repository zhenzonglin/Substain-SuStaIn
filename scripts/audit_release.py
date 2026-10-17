#!/usr/bin/env python3
"""发布前仅审计Git暂存区；拒绝患者输出、权重、超大文件和明显凭据。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

# ===== 发布范围和限额全部从配置读取 =====
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config",default="config/release.json")
args=parser.parse_args()
path=Path(args.config).resolve()
cfg=json.loads(path.read_text())
root=(path.parent/cfg["project_dir"]).resolve()
names=subprocess.check_output(["git","-C",str(root),"diff","--cached","--name-only","--diff-filter=ACMR","-z"]).decode().split("\0")
records=[]
patterns=[re.compile(rb"ghp_[A-Za-z0-9]{36}"),re.compile(rb"github_pat_[A-Za-z0-9_]{40,}"),
          re.compile(rb"-----BEGIN (?:RSA|EC|OPENSSH) PRIVATE KEY-----")]
for name in filter(None,names):
    if any(name.startswith(value) for value in cfg["forbidden_prefixes"]) and name not in cfg["allowed_config_examples"]:
        raise SystemExit("禁止发布目录："+name)
    if name.endswith(".nii.gz") and name not in cfg["allowed_image_assets"]:
        raise SystemExit("禁止发布非白名单影像："+name)
    if any(name.endswith(value) for value in cfg["forbidden_suffixes"]):
        raise SystemExit("禁止发布模型或数据结果："+name)
    # 读取真实暂存blob，防止工作区内容与待推送内容不同。
    data=subprocess.check_output(["git","-C",str(root),"show",":"+name])
    if len(data)>cfg["maximum_file_bytes"]:
        raise SystemExit("文件超过配置大小限制："+name)
    if not name.endswith((".nii.gz",".whl")) and any(pattern.search(data) for pattern in patterns):
        raise SystemExit("检测到疑似凭据："+name)
    records.append({"path":name,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()})
result={"status":"pass","files":len(records),"total_bytes":sum(row["bytes"] for row in records),
        "largest_file_bytes":max((row["bytes"] for row in records),default=0),"records":records}
destination=root/cfg["receipt"]
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(result,indent=2))
print(json.dumps({key:result[key] for key in ("status","files","total_bytes","largest_file_bytes")},indent=2))
