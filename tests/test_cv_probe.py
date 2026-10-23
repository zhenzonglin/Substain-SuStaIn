"""独立探针验证：官方数值等价、限时退出和正式文件隔离。"""
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.model_selection import KFold
from pySuStaIn.ZscoreSustain import ZScoreSustainData

from substain_sustain.config import load_config
from substain_sustain.model import _identity, _model_seed, _engine
from test_model import _synthetic_project

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("probe_cv", ROOT / "scripts/probe_cv.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
pytestmark = pytest.mark.filterwarnings(
    "ignore:Conversion of an array with ndim > 0 to a scalar:DeprecationWarning:pySuStaIn")


# ===== 一、模拟已开始CV但没有模型缓存的工作站 =====
def station(tmp_path, n_features=3):
    output = tmp_path / "formal"
    output.mkdir()
    synthetic = _synthetic_project(output)
    cfg = yaml.safe_load((ROOT / "config/analysis.yaml").read_text())
    cfg.update(synthetic)
    cfg.update(project_dir=str(tmp_path), output_dir=str(output))
    cfg["feature_selection"] = {"expected_model_features": 40, "exclude": [], "merge": []}
    if n_features != 3:
        features = pd.DataFrame({"feature_id": [f"f_{i:02d}" for i in range(n_features)],
            "label": [f"F{i}" for i in range(n_features)], "modality": ["WMH"] * 19 + ["GM"] * 19,
            "atlas_index": list(range(1, 20)) * 2})
        features.to_csv(output / "inputs/feature_dictionary.tsv", sep="\t", index=False)
        frame = pd.read_csv(output / "reference/C3_primary_z.tsv", sep="\t", dtype={"participant_id": str})
        frame = frame.drop(columns=["wmh_a", "gm_b", "gm_c"])
        rng = np.random.default_rng(42)
        for feature in features.feature_id:
            frame[feature] = rng.normal(1, 0.8, len(frame))
        frame.to_csv(output / "reference/C3_primary_z.tsv", sep="\t", index=False)
        cfg["model"].update(z_vals=[1, 2, 3], z_max=5)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    cfg = load_config(config_path)
    cv = output / "model/cv"
    cv.mkdir(parents=True)
    identity = _identity(cfg, output, [output / "reference/C3_primary_z.tsv"], "C3_crossvalidation")
    (cv / "manifest.json").write_text(json.dumps(identity))
    splits = [t.tolist() for _, t in KFold(n_splits=2, shuffle=True, random_state=2026).split(range(20))]
    (cv / "folds.json").write_text(json.dumps([
        {"fold": i, "indices": t, "subject_keys": [f"C3:{j:03d}" for j in t],
         "model_seeds": {"1": _model_seed(2026, i, 1)}} for i, t in enumerate(splits)]))
    settings = probe.read_settings(ROOT / "config/cv_probe.yaml")
    settings.update(nice_increment=0, min_available_memory_gb=0.001, max_wall_seconds=60, heartbeat_seconds=0.1)
    return cfg, settings


def snapshot(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


# ===== 二、完整起点数值与原官方调用严格一致 =====
def test_observer_preserves_official_sequence_likelihood_and_rng(tmp_path):
    cfg, settings = station(tmp_path)
    settings["max_em_sweeps"] = 101
    data, features, model, metadata = probe.prepare(cfg, settings)
    train = ZScoreSustainData(data, metadata["n_events"])
    plain = _engine(data, features, model, tmp_path / "plain", 123, 30, 1)
    expected = plain._find_ml_iteration(train, np.random.SeedSequence(987))
    directory = tmp_path / "observed"
    observed = _engine(data, features, model, directory, 123, 30, 1)
    original = observed._optimise_parameters
    result = probe.observe(observed, train, np.random.SeedSequence(987), settings, directory)
    assert result["status"] == "startpoint_finished"
    assert result["final_log_likelihood"] == float(expected[2])
    assert result["final_sequence_sha256"] == hashlib.sha256(np.asarray(expected[0]).tobytes()).hexdigest()
    np.testing.assert_array_equal(result["final_fractions"], expected[1])
    assert observed._optimise_parameters == original
    # 再直接运行同一随机流，确认观察器没有更改实例算法或 RNG 输入。
    actual = observed._find_ml_iteration(train, np.random.SeedSequence(987))
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)
    assert result["em_sweeps_completed"] >= 1 and result["likelihood_calls_completed"] > 1


@pytest.mark.parametrize("n_features", [3, 38])
def test_one_sweep_real_official_and_formal_files_untouched(tmp_path, n_features):
    cfg, settings = station(tmp_path, n_features)
    formal = Path(cfg["output_dir"])
    before = snapshot(formal)
    directory = probe.diagnostic_root(cfg, settings) / "test"
    directory.mkdir(parents=True)
    probe.child_run(cfg, settings, directory)
    result = json.loads((directory / "result.json").read_text())
    assert result["status"] in ("sweep_limit", "startpoint_finished")
    assert result["em_sweeps_completed"] == 1
    assert np.isfinite(result["em_sweeps"][0]["candidate_log_likelihood_after"])
    assert snapshot(formal) == before
    assert not list(directory.rglob("*.pickle"))
    meta = json.loads((directory / "input_identity.json").read_text())
    assert meta["n_features"] == n_features
    if n_features == 38:
        assert meta["n_events"] == 114


def test_time_budget_is_explicit_not_false_completion(tmp_path):
    cfg, settings = station(tmp_path)
    data, features, model, meta = probe.prepare(cfg, settings)
    directory = tmp_path / "timeout"
    engine = _engine(data, features, model, directory, 123, 30, 1)
    settings["max_wall_seconds"] = 1e-9
    result = probe.observe(engine, ZScoreSustainData(data, meta["n_events"]),
                           np.random.SeedSequence(42), settings, directory)
    assert result["status"] == "time_limit" and result["em_sweeps_completed"] == 0


def test_source_or_split_drift_rejected(tmp_path):
    cfg, settings = station(tmp_path)
    path = Path(cfg["output_dir"]) / "model/cv/manifest.json"
    saved = json.loads(path.read_text())
    saved["upstream_commit"] = "different"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="CV身份"):
        probe.prepare(cfg, settings)


def test_no_outputs_into_formal_or_source_directories(tmp_path):
    cfg, settings = station(tmp_path)
    for path in ("formal/probe", ".", str(tmp_path.parent)):
        with pytest.raises(ValueError):
            probe.diagnostic_root(cfg, dict(settings, output_dir=path))


def test_memory_guard_rejects_before_computation(tmp_path, monkeypatch):
    from substain_sustain import model
    cfg, settings = station(tmp_path)
    monkeypatch.setattr(model, "_available_memory", lambda: 1)
    with pytest.raises(MemoryError):
        probe.prepare(cfg, settings)


def test_real_command_runs_one_child_and_preserves_formal_data(tmp_path):
    cfg, settings = station(tmp_path)
    path = tmp_path / "probe.yaml"
    path.write_text(yaml.safe_dump(settings))
    formal = Path(cfg["output_dir"])
    before = snapshot(formal)
    result = subprocess.run([sys.executable, str(ROOT / "scripts/probe_cv.py"), "--config", cfg["_config_file"],
                             "--probe-config", str(path)], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert "独立探针" in result.stdout
    assert snapshot(formal) == before
    receipt = next((tmp_path / ".cache/cv_probe").glob("*/launcher.json"))
    assert json.loads(receipt.read_text())["status"] == "finished"


def test_hard_timeout_stops_only_probe_and_preserves_formal_data(tmp_path):
    cfg, settings = station(tmp_path)
    settings.update(max_wall_seconds=0.001, termination_grace_seconds=0.5)
    path = tmp_path / "probe.yaml"
    path.write_text(yaml.safe_dump(settings))
    formal = Path(cfg["output_dir"])
    before = snapshot(formal)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        result = subprocess.run([sys.executable, str(ROOT / "scripts/probe_cv.py"), "--config", cfg["_config_file"],
                                 "--probe-config", str(path)], capture_output=True, text=True, timeout=30)
        assert result.returncode == 2, result.stderr
        assert unrelated.poll() is None
        assert snapshot(formal) == before
        receipt = next((tmp_path / ".cache/cv_probe").glob("*/launcher.json"))
        assert json.loads(receipt.read_text())["status"] == "hard_timeout"
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=10)
