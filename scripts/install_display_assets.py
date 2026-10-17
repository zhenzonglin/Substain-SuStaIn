#!/usr/bin/env python3
"""安装小型展示资源：复用工作站WMH图谱、加载仓库公开模板派生GM图谱。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from substain_sustain.config import load_config
from substain_sustain.assets import validate_assets

# ===== 路径与资源身份只从配置及冻结清单读取 =====
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config",required=True)
args=parser.parse_args()
cfg=load_config(args.config)
settings=cfg["assets"]
manifest=json.loads(Path(settings["bundled_manifest"]).read_text())
for key in ("gm_atlas","background","wmh_atlas"):
    record=manifest["assets"][key]
    target=Path(settings[key])
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_file():
        if hashlib.sha256(target.read_bytes()).hexdigest()!=record["sha256"]:
            raise SystemExit("已有展示文件与锁定版本不同，不覆盖："+str(target))
        continue
    if key=="wmh_atlas":
        source=Path(settings["source_wmh_atlas"])
        if source.is_file():
            data=source.read_bytes()
        else:
            # 缺少源副本时直接取上游GitHub固定版本，不在本仓库再分发该图谱。
            url=settings["wmh_download_url"]
            if not url.startswith("https://raw.githubusercontent.com/"):
                raise SystemExit("WMH下载入口必须为配置中固定的GitHub地址")
            with urllib.request.urlopen(url,timeout=settings["download_timeout_seconds"]) as response:
                data=response.read()
    else:
        data=(Path(settings["bundled_dir"])/record["filename"]).read_bytes()
    if hashlib.sha256(data).hexdigest()!=record["sha256"]:
        raise SystemExit("展示资源SHA256不匹配："+key)
    temporary=target.with_name(target.name+".partial")
    temporary.write_bytes(data)
    temporary.replace(target)
destination=Path(settings["manifest"])
destination.parent.mkdir(parents=True,exist_ok=True)
if destination.resolve()!=Path(settings["bundled_manifest"]).resolve():
    shutil.copyfile(settings["bundled_manifest"],destination)
print(json.dumps(validate_assets(cfg),indent=2,ensure_ascii=False))
