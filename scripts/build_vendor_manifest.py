"""保存小型第三方wheel的来源、哈希和源码身份，供安装及doctor验证。"""
import hashlib
import argparse
import json
from pathlib import Path
import urllib.request
import zipfile

# ===== 构建版本和所有路径从专用配置读取 =====
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="config/vendor_build.json")
args=parser.parse_args()
config_path=Path(args.config).resolve()
settings=json.loads(config_path.read_text())
root=(config_path.parent/settings["project_dir"]).resolve()
packages=settings["packages"]
manifest={"platform":settings["platform"],"python":settings["python"],"packages":{},"pysustain_source_sha256":{}}
licenses=root/settings["license_dir"]
licenses.mkdir(exist_ok=True)
for name,record in packages.items():
 repo,commit,pattern=record["repository"],record["commit"],record["wheel_pattern"]
 path=next((root/settings["wheel_dir"]).glob(pattern))
 manifest["packages"][name]={"repository":"https://github.com/"+repo,"commit":commit,"wheel":str(path.relative_to(root)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
 with zipfile.ZipFile(path) as wheel:
  candidates=[n for n in wheel.namelist() if "license" in n.lower() or "copying" in n.lower()]
  if candidates:
   for idx,member in enumerate(candidates):
    (licenses/(name+"_"+str(idx)+"_LICENSE.txt")).write_bytes(wheel.read(member))
  else:
   tree=json.load(urllib.request.urlopen("https://api.github.com/repos/"+repo+"/git/trees/"+commit, timeout=settings["download_timeout_seconds"]))
   names=[e["path"] for e in tree["tree"] if "license" in e["path"].lower() or "copying" in e["path"].lower()]
   if not names:
    raise RuntimeError("缺少许可证："+repo)
   for idx,member in enumerate(names):
    body=urllib.request.urlopen("https://raw.githubusercontent.com/"+repo+"/"+commit+"/"+member, timeout=settings["download_timeout_seconds"]).read()
    (licenses/(name+"_"+str(idx)+"_LICENSE.txt")).write_bytes(body)
  if name=="pySuStaIn":
   for member in record["verify_modules"]:
    manifest["pysustain_source_sha256"][member]=hashlib.sha256(wheel.read("pySuStaIn/"+member)).hexdigest()
# ===== 保存可复核的wheel及实际模块指纹 =====
(root/settings["manifest"]).write_text(json.dumps(manifest,indent=2),encoding="utf-8")
print(json.dumps(manifest,indent=2))
