"""安装器必须保留工作站可用源，不能把网络故障当作版本缺失。"""
import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bootstrap", ROOT / "scripts/bootstrap.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def test_default_inherits_existing_pip_config():
    existing = {
        "PIP_CONFIG_FILE": "/etc/company-pip.conf",
        "PIP_INDEX_URL": "https://internal.example/simple",
        "PIP_EXTRA_INDEX_URL": "https://extras.example/simple",
        "HTTPS_PROXY": "http://proxy.example:8080",
        "PIP_CERT": "/etc/company.pem",
    }
    source = bootstrap.install_source({"pip_index_url": None})
    env = bootstrap.pip_environment(source, existing)
    assert source == {"mode": "inherit"}
    assert all(env[key] == value for key, value in existing.items())
    assert "--index-url" not in bootstrap.source_flags(source, {}, ROOT)
    assert existing.get("PIP_DISABLE_PIP_VERSION_CHECK") is None


def test_explicit_index_override_has_precedence():
    source = bootstrap.install_source({"pip_index_url": "https://configured.example/simple"},
                                      "https://chosen.example/simple")
    assert bootstrap.source_flags(source, {}, ROOT) == ["--index-url", "https://chosen.example/simple"]


def test_offline_wheelhouse_is_index_free(tmp_path):
    source = bootstrap.install_source({"pip_index_url": "https://pypi.org/simple"}, wheelhouse_override=tmp_path)
    env = bootstrap.pip_environment(source, {"PIP_INDEX_URL": "remote", "PIP_FIND_LINKS": "https://remote"})
    assert env["PIP_CONFIG_FILE"] == os.devnull
    assert env["PIP_NO_INDEX"] == "1"
    assert "PIP_INDEX_URL" not in env and "PIP_FIND_LINKS" not in env
    assert bootstrap.source_flags(source, {}, ROOT) == ["--no-index", "--find-links", str(tmp_path)]


def test_cli_index_overrides_configured_wheelhouse():
    assert bootstrap.install_source({"wheelhouse": "old"}, index_override="https://chosen.example/simple") == {
        "mode": "index", "index": "https://chosen.example/simple"}


def test_missing_offline_directory_rejected(tmp_path):
    with pytest.raises(ValueError, match="不存在"):
        bootstrap.source_flags({"mode": "wheelhouse", "wheelhouse": str(tmp_path / "missing")}, {}, ROOT)


def test_timeout_and_retry_config():
    assert bootstrap.source_flags({"mode": "inherit"}, {"pip_timeout_seconds": 30, "pip_retries": 0}, ROOT) == [
        "--timeout", "30", "--retries", "0"]
    with pytest.raises(ValueError):
        bootstrap.source_flags({"mode": "inherit"}, {"pip_retries": True}, ROOT)


def test_invalid_index_scheme_rejected():
    with pytest.raises(ValueError, match="HTTP"):
        bootstrap.install_source({"pip_index_url": "file:///not-an-index"})


def test_install_reuses_existing_environment_and_never_forces_pypi(tmp_path, monkeypatch):
    cfg = {"base_python": "base", "expected_system": "Linux", "expected_machine": "x86_64",
           "expected_python": [3, 10], "venv_dir": ".venv", "pip_index_url": None,
           "pip_version": "24.3.1", "setuptools_version": "75.6.0", "wheel_version": "0.45.1",
           "requirements": "requirements.lock"}
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(bootstrap.platform, "system", lambda: "Linux")
    monkeypatch.setattr(bootstrap.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(bootstrap, "python_version", lambda _: "3.10")
    monkeypatch.setattr(bootstrap, "validated_wheels", lambda *_: ["official.whl"])
    monkeypatch.setenv("PIP_CONFIG_FILE", "/workstation/pip.conf")
    calls = []
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    bootstrap.install(cfg, tmp_path)
    assert not any("venv" in command for command, _ in calls)
    assert all("--index-url" not in command for command, _ in calls)
    assert all(kw["env"]["PIP_CONFIG_FILE"] == "/workstation/pip.conf" for _, kw in calls)
    assert "--no-index" in calls[3][0]


def test_existing_wrong_python_is_not_overwritten(tmp_path, monkeypatch):
    cfg = {"base_python": "base", "expected_system": "Linux", "expected_machine": "x86_64",
           "expected_python": [3, 10], "venv_dir": ".venv"}
    path = tmp_path / ".venv/bin/python"
    path.parent.mkdir(parents=True)
    path.touch()
    monkeypatch.setattr(bootstrap.platform, "system", lambda: "Linux")
    monkeypatch.setattr(bootstrap.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(bootstrap, "python_version", lambda exe: "3.10" if exe == "base" else "3.8")
    with pytest.raises(ValueError, match="保留"):
        bootstrap.install(cfg, tmp_path)
    assert path.exists()
