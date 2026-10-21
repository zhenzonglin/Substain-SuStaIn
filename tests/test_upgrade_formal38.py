"""真实本地Git远端模拟配置冲突；确认备份、继承优先级与失败恢复。"""
import importlib.util
from pathlib import Path
import subprocess

import pytest
import yaml

from substain_sustain.config import load_config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("upgrade_formal38", ROOT / "scripts/upgrade_formal38.py")
upgrade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upgrade)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True).strip()


@pytest.fixture
def workstation(tmp_path):
    origin, publisher, station = [tmp_path / name for name in ("remote.git", "publisher", "station")]
    origin.mkdir()
    git(origin, "init", "--bare")
    publisher.mkdir()
    git(publisher, "init")
    git(publisher, "checkout", "-b", "main")
    git(publisher, "config", "user.name", "Synthetic test")
    git(publisher, "config", "user.email", "synthetic@example.invalid")
    (publisher / "config").mkdir()
    old = yaml.safe_load((ROOT / "config/analysis.yaml").read_text())
    old["feature_selection"] = {"expected_model_features": 40, "exclude": []}
    old["paths"].pop("model_mapping")
    (publisher / "config/analysis.yaml").write_text(yaml.safe_dump(old))
    (publisher / ".gitignore").write_text("config/local*.yaml\n.cache/\n")
    git(publisher, "add", ".")
    git(publisher, "commit", "-m", "old configuration")
    git(publisher, "remote", "add", "origin", str(origin))
    git(publisher, "push", "-u", "origin", "main")
    subprocess.run(["git", "clone", "-b", "main", str(origin), str(station)], check=True, capture_output=True)
    old["cohorts"]["LS"]["derivatives"] = "/actual/LS"
    old["cohorts"]["GB"]["participants"] = "/actual/GB.tsv"
    old["model"]["workers"] = 2
    (station / "config/analysis.yaml").write_text("# 用户手动修改，必须逐字保留备份\n" + yaml.safe_dump(old, allow_unicode=True))
    (station / "config/local.yaml").write_text(yaml.safe_dump({"extends": "analysis.yaml", "project_dir": "..",
        "cohorts": {"LS": {"derivatives": "/higher-priority/LS"}}, "model": {"posterior_batch_size": 9}}))
    for name in ("analysis.yaml", "formal38_migrated.example.yaml"):
        (publisher / "config" / name).write_bytes((ROOT / "config" / name).read_bytes())
    git(publisher, "add", ".")
    git(publisher, "commit", "-m", "final 38 configuration")
    git(publisher, "push")
    return station


def test_real_fast_forward_preserves_effective_paths_and_final_contract(workstation):
    original = (workstation / "config/analysis.yaml").read_bytes()
    local = (workstation / "config/local.yaml").read_bytes()
    result = upgrade.upgrade(workstation, Path("config/local.yaml"), Path("config/local.38.yaml"))
    assert result["status"] == "complete"
    assert (Path(result["backup_dir"]) / "analysis.yaml").read_bytes() == original
    assert (workstation / "config/local.yaml").read_bytes() == local
    cfg = load_config(result["target_config"])
    assert cfg["cohorts"]["LS"]["derivatives"] == "/higher-priority/LS"
    assert cfg["cohorts"]["GB"]["participants"] == "/actual/GB.tsv"
    assert cfg["model"]["workers"] == 2 and cfg["model"]["posterior_batch_size"] == 9
    assert cfg["feature_selection"]["expected_model_features"] == 38
    assert cfg["feature_selection"]["merge"][0]["source_features"] == ["gm_striatopallidal_ml", "gm_basal_forebrain_ml"]
    assert cfg["cohorts"]["C3"]["relocate_feature_pointers"] is True
    assert cfg["output_dir"].endswith("formal_LS_C3_GB_38_v1")
    assert git(workstation, "status", "--porcelain") == ""


def test_failed_pull_restores_original_analysis(workstation, monkeypatch):
    before = (workstation / "config/analysis.yaml").read_bytes()
    real_run = subprocess.run

    def fail_pull(command, **kwargs):
        if "pull" in command:
            raise subprocess.CalledProcessError(1, command)
        return real_run(command, **kwargs)

    monkeypatch.setattr(upgrade.subprocess, "run", fail_pull)
    with pytest.raises(subprocess.CalledProcessError):
        upgrade.upgrade(workstation, Path("config/local.yaml"), Path("config/local.38.yaml"))
    assert (workstation / "config/analysis.yaml").read_bytes() == before
    assert len(list((workstation / "config").glob("local.preserved_*.yaml"))) == 1
    assert not (workstation / "config/local.38.yaml").exists()


def test_existing_local38_is_backed_up_before_replacement(workstation):
    previous = b"# old target settings\n"
    (workstation / "config/local.38.yaml").write_bytes(previous)
    result = upgrade.upgrade(workstation, Path("config/local.yaml"), Path("config/local.38.yaml"))
    assert (Path(result["backup_dir"]) / "local.38.yaml").read_bytes() == previous


def test_staged_changes_are_not_discarded(workstation):
    git(workstation, "add", "config/analysis.yaml")
    before = git(workstation, "diff", "--cached")
    with pytest.raises(ValueError, match="已暂存"):
        upgrade.upgrade(workstation, Path("config/local.yaml"), Path("config/local.38.yaml"))
    assert git(workstation, "diff", "--cached") == before
