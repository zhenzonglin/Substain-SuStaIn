#!/usr/bin/env python3
"""安装独立环境：默认沿用工作站pip配置，也支持显式指定软件源或本地wheel目录。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from urllib.parse import urlsplit


# ===== 安装来源：只在用户明确指定时覆盖现有pip源 =====
def install_source(cfg: dict, index_override=None, wheelhouse_override=None) -> dict:
    wheelhouse = wheelhouse_override or (cfg.get("wheelhouse") if index_override is None else None)
    if wheelhouse:
        return {"mode": "wheelhouse", "wheelhouse": str(wheelhouse)}
    index = index_override if index_override is not None else cfg.get("pip_index_url")
    if index:
        if not isinstance(index, str) or urlsplit(index).scheme not in ("http", "https"):
            raise ValueError("pip_index_url必须为HTTP(S)软件源地址，或null以沿用工作站配置")
        return {"mode": "index", "index": index}
    return {"mode": "inherit"}


def pip_environment(source: dict, inherited=None) -> dict:
    env = dict(os.environ if inherited is None else inherited)
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    if source["mode"] == "wheelhouse":
        # 仅显式离线模式隔离网络配置；普通安装保留用户/系统源、代理和证书设置。
        env["PIP_CONFIG_FILE"] = os.devnull
        for key in ("PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_FIND_LINKS"):
            env.pop(key, None)
        env["PIP_NO_INDEX"] = "1"
    return env


def source_flags(source: dict, cfg: dict, root: Path) -> list:
    if source["mode"] == "wheelhouse":
        folder = Path(source["wheelhouse"]).expanduser()
        folder = (root / folder).resolve() if not folder.is_absolute() else folder.resolve()
        if not folder.is_dir():
            raise ValueError("本地wheel目录不存在：" + str(folder))
        return ["--no-index", "--find-links", str(folder)]
    flags = ["--index-url", source["index"]] if source["mode"] == "index" else []
    for key, option, minimum in (("pip_timeout_seconds", "--timeout", 1), ("pip_retries", "--retries", 0)):
        value = cfg.get(key)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(key + "参数无效")
            flags.extend([option, str(value)])
    return flags


# ===== 平台、虚拟环境和固定官方wheel的预检 =====
def python_version(executable: str) -> str:
    return subprocess.check_output(
        [str(executable), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"], text=True
    ).strip()


def validated_wheels(root: Path, cfg: dict) -> list:
    manifest = json.loads((root / cfg["wheel_manifest"]).read_text(encoding="utf-8"))
    wheels = []
    for record in manifest["packages"].values():
        path = (root / record["wheel"]).resolve()
        if root not in path.parents:
            raise ValueError("官方wheel路径越界")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("wheel校验失败：" + str(path))
        wheels.append(str(path))
    return wheels


def install(cfg: dict, root: Path, base_override=None, index_override=None, wheelhouse_override=None) -> Path:
    base = base_override or cfg["base_python"]
    if platform.system() != cfg["expected_system"] or platform.machine() != cfg["expected_machine"]:
        raise ValueError("此wheel锁仅验证Linux x86_64；请使用相应工作站")
    expected = ".".join(str(x) for x in cfg["expected_python"])
    if python_version(base) != expected:
        raise ValueError("已有基础解释器不是配置指定的Python " + expected)
    venv = (root / cfg["venv_dir"]).resolve()
    if venv == root or root not in venv.parents:
        raise ValueError("独立venv必须位于本项目内")
    python = str(venv / "bin/python")
    if Path(python).is_file() and python_version(python) != expected:
        raise ValueError("已有虚拟环境Python版本不符；保留该目录，请在config指定新的venv_dir")
    source = install_source(cfg, index_override, wheelhouse_override)
    flags = source_flags(source, cfg, root)
    wheels = validated_wheels(root, cfg)
    env = pip_environment(source)

    if source["mode"] == "inherit":
        print("安装源：沿用工作站现有pip配置和环境变量。", flush=True)
    elif source["mode"] == "index":
        # 只显示域名，不把软件源中的用户名、密码或查询参数写到终端。
        print("安装源：显式配置的 " + str(urlsplit(source["index"]).hostname), flush=True)
    else:
        print("安装源：本地wheel目录；不访问任何软件索引。", flush=True)
    if not Path(python).is_file():
        subprocess.run([base, "-m", "venv", str(venv)], check=True)
    pip = [python, "-m", "pip"]

    # ===== 依赖安装：版本锁保持不变，不删除已经创建的环境 =====
    subprocess.run(pip + ["install", *flags, "pip==" + cfg["pip_version"],
                         "setuptools==" + cfg["setuptools_version"], "wheel==" + cfg["wheel_version"]],
                   env=env, check=True)
    subprocess.run(pip + ["install", "--only-binary=:all:", *flags,
                         "-r", str(root / cfg["requirements"])], env=env, check=True)
    subprocess.run(pip + ["install", "--no-index", "--no-deps", *wheels], env=env, check=True)
    subprocess.run(pip + ["install", "--no-index", "--no-deps", "--no-build-isolation", "-e", str(root)],
                   env=env, check=True)
    subprocess.run(pip + ["check"], env=env, check=True)
    subprocess.run([python, "-c", "import pySuStaIn, awkde.backend, kde_ebm; print('官方工具导入通过')"],
                   env=env, check=True)
    print("安装完成：" + python, flush=True)
    return venv


# ===== 命令行与故障提示：保留pip原始错误，不输出含凭据的完整调用参数 =====
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/install.json")
    parser.add_argument("--python", help="覆盖基础Python3.10解释器位置")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--index-url", help="仅本次安装显式指定pip软件源；默认沿用工作站")
    sources.add_argument("--wheelhouse", help="仅从本地完整wheel目录安装")
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))
        root = (config_path.parent / cfg["project_dir"]).resolve()
        install(cfg, root, args.python, args.index_url, args.wheelhouse)
    except subprocess.CalledProcessError as exc:
        print("安装命令失败（退出码%s）；原始错误见上方。" % exc.returncode)
        print("若出现Network is unreachable/连接重试，说明当前软件源不可达，不代表锁定版本不存在。")
        print("默认沿用已有pip源；可用--index-url指定可达源，或--wheelhouse使用完整本地依赖。")
        print("已建虚拟环境予以保留；修正网络/软件源后可直接重试。")
        return 1
    except (OSError, ValueError, KeyError) as exc:
        print("安装预检失败：" + str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
