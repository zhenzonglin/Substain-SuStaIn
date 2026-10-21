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


def test_migrated_38_example_preserves_local_ls_gb_and_selects_new_output(tmp_path):
    import yaml
    root = Path(__file__).resolve().parents[1]
    local = {"extends": str(root / "config/analysis.yaml"), "project_dir": str(tmp_path),
             "output_dir": "old_failed_run", "cohorts": {
                 "LS": {"derivatives": "/local/LS"}, "GB": {"derivatives": "/local/GB"}}}
    (tmp_path / "local.yaml").write_text(yaml.safe_dump(local))
    target = tmp_path / "local.38.yaml"
    target.write_text((root / "config/formal38_migrated.example.yaml").read_text())
    cfg = load_config(target)
    assert cfg["cohorts"]["LS"]["derivatives"] == "/local/LS"
    assert cfg["cohorts"]["GB"]["derivatives"] == "/local/GB"
    assert cfg["cohorts"]["C3"]["relocate_feature_pointers"] is True
    assert cfg["cohorts"]["C3"]["input_bundle"] is None
    assert cfg["output_dir"].endswith("runs/formal_LS_C3_GB_38_v1")
    assert cfg["feature_selection"]["expected_model_features"] == 38


def test_config_validation_does_not_load_blas_before_cli_thread_limits():
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    code = ("import sys; from substain_sustain.config import load_config; "
            "load_config(sys.argv[1]); assert 'numpy' not in sys.modules")
    subprocess.run([sys.executable, "-c", code, str(root / "config/analysis.yaml")], check=True)
