#!/usr/bin/env python3
"""从配置安装独立环境；只从pip源和仓库内已校验wheel读取依赖。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

# ===== 安装配置与平台预检 =====
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="config/install.json")
parser.add_argument("--python", help="覆盖配置中的已有Python3.10解释器路径")
args = parser.parse_args()
config_path = Path(args.config).resolve()
cfg = json.loads(config_path.read_text(encoding="utf-8"))
root = (config_path.parent / cfg["project_dir"]).resolve()
base = args.python or cfg["base_python"]
if platform.system() != cfg["expected_system"] or platform.machine() != cfg["expected_machine"]:
    raise SystemExit("此wheel锁仅验证Linux x86_64；请使用相应工作站")
version = subprocess.check_output([base, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"], text=True).strip()
if version != ".".join(str(x) for x in cfg["expected_python"]):
    raise SystemExit("已有解释器不是配置指定的Python3.10")
venv = (root / cfg["venv_dir"]).resolve()
if venv == root or root not in venv.parents:
    raise SystemExit("独立venv必须位于本项目内")
if not (venv / "bin/python").is_file():
    subprocess.run([base, "-m", "venv", str(venv)], check=True)
python = str(venv / "bin/python")
env = os.environ.copy()
# 禁止用户全局pip配置引入额外软件源或静默升级。
env.update(PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1")
env.pop("PIP_EXTRA_INDEX_URL", None)
env["PIP_INDEX_URL"] = cfg["pip_index_url"]
pip = [python, "-m", "pip"]
subprocess.run(pip + ["install", "--index-url", cfg["pip_index_url"],
                     "pip==" + cfg["pip_version"], "setuptools==" + cfg["setuptools_version"],
                     "wheel==" + cfg["wheel_version"]], env=env, check=True)
subprocess.run(pip + ["install", "--only-binary=:all:", "--index-url", cfg["pip_index_url"],
                     "-r", str(root / cfg["requirements"])], env=env, check=True)

# ===== 本地wheel完整性与官方版本身份 =====
manifest = json.loads((root / cfg["wheel_manifest"]).read_text())
wheels = []
for record in manifest["packages"].values():
    path = root / record["wheel"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise SystemExit("wheel校验失败：" + str(path))
    wheels.append(str(path))
subprocess.run(pip + ["install", "--no-index", "--no-deps", *wheels], env=env, check=True)
subprocess.run(pip + ["install", "--no-deps", "--no-build-isolation", "-e", str(root)], env=env, check=True)
subprocess.run(pip + ["check"], env=env, check=True)
# 官方模块实际导入可及时发现编译扩展/ABI不兼容。
subprocess.run([python, "-c", "import pySuStaIn, awkde.backend, kde_ebm; print('官方工具导入通过')"],
               env=env, check=True)
print("安装完成：" + python)
