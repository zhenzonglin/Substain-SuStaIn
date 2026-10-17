"""配置路径、覆盖与研究参数约束的独立验收。"""
from pathlib import Path
import pytest
from substain_sustain.config import load_config, deep_merge


def test_deep_merge_preserves_unmodified_sections():
    assert deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"x": 3}}) == {"a": {"x": 3, "y": 2}}


def test_default_config_paths_and_formal_parameters():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config/analysis.yaml")
    assert cfg["project_dir"] == str(root)
    assert cfg["model"]["fit_iterations"] == 1_000_000
    assert Path(cfg["paths"]["mapping"]).is_file()
    assert cfg["reference"]["age_min"] == 50
    assert cfg["ingest"]["manual_qc_required"] is False


def test_extends_cycle_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("extends: bad.yaml")
    with pytest.raises(ValueError, match="循环"):
        load_config(path)
