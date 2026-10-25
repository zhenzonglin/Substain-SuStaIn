"""分块只用于独立诊断；验证逐元素概率、原EM和原MCMC均不变。"""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml
from pySuStaIn.ZscoreSustain import ZscoreSustain, ZScoreSustainData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from likelihood_blocks import build_blocked_kernel  # noqa: E402
from audit_cv_performance import compare_kernels  # noqa: E402
from test_cv_probe import station, snapshot  # noqa: E402

pytestmark = pytest.mark.filterwarnings(
    "ignore:Conversion of an array with ndim > 0 to a scalar:DeprecationWarning:pySuStaIn")


def engine_data(tmp_path, rows, features, layout="C"):
    rng = np.random.default_rng(20261025)
    values = rng.normal(0.8, 2.5, (rows, features))
    if layout == "F":
        values = np.asfortranarray(values)
    elif layout == "sliced":
        values = np.repeat(values, 2, axis=0)[::2]
    engine = ZscoreSustain(values, np.tile([1., 2., 3.], (features, 1)), np.repeat(5., features),
                          [str(i) for i in range(features)], 1, 2, 75, str(tmp_path), "test", False, seed=42)
    data = ZScoreSustainData(values, features * 3)
    return engine, data


# ===== 一、大小边界、布局、负值与大样本均严格逐元素比较 =====
@pytest.mark.parametrize("rows,layout", [(1, "C"), (2, "C"), (31, "F"), (32, "C"),
                                        (33, "sliced"), (65, "C"), (65, "F"), (6112, "C"), (6112, "F")])
def test_probability_bitwise_equal(tmp_path, rows, layout):
    engine, data = engine_data(tmp_path, rows, 38, layout)
    rng = np.random.default_rng(100)
    blocked = build_blocked_kernel(engine, 32)
    for _ in range(2):
        sequence = engine._initialise_sequence(data, rng)[0]
        expected = engine._calculate_likelihood_stage(data, sequence)
        np.testing.assert_array_equal(blocked(data, sequence), expected)


def test_extreme_and_zero_values_preserve_underflow(tmp_path):
    engine, data = engine_data(tmp_path, 33, 6)
    data.data[0] = 0
    data.data[1] = 1000
    data.data[2] = -1000
    sequence = engine._initialise_sequence(data, np.random.default_rng(8))[0]
    original = engine._calculate_likelihood_stage(data, sequence)
    np.testing.assert_array_equal(build_blocked_kernel(engine, 32)(data, sequence), original)
    assert np.all(original[1:3] == 0)


# ===== 二、官方双亚型参数优化和MCMC采样逐项完全一致 =====
def test_official_optimisation_and_mcmc_unchanged(tmp_path):
    plain, data = engine_data(tmp_path, 12, 6)
    blocked, _ = engine_data(tmp_path, 12, 6)
    blocked._calculate_likelihood_stage = build_blocked_kernel(blocked, 4)
    rng = np.random.default_rng(73)
    sequences = np.concatenate([plain._initialise_sequence(data, rng) for _ in range(2)])
    fractions = np.array([0.4, 0.6])
    expected = plain._optimise_parameters(data, sequences, fractions, np.random.default_rng(44))
    actual = blocked._optimise_parameters(data, sequences, fractions, np.random.default_rng(44))
    for a, b in zip(expected, actual):
        np.testing.assert_array_equal(a, b)
    expected_chain = plain._perform_mcmc(data, expected[0], expected[1], 75, 1, 0.01)
    actual_chain = blocked._perform_mcmc(data, actual[0], actual[1], 75, 1, 0.01)
    for a, b in zip(expected_chain, actual_chain):
        np.testing.assert_array_equal(a, b)


def test_official_complete_startpoint_unchanged(tmp_path):
    plain, data = engine_data(tmp_path, 12, 6)
    blocked, _ = engine_data(tmp_path, 12, 6)
    blocked._calculate_likelihood_stage = build_blocked_kernel(blocked, 4)
    expected = plain._find_ml_iteration(data, np.random.SeedSequence(345))
    actual = blocked._find_ml_iteration(data, np.random.SeedSequence(345))
    for a, b in zip(expected, actual):
        np.testing.assert_array_equal(a, b)


def test_comparison_rejects_numeric_drift(tmp_path, monkeypatch):
    import audit_cv_performance as audit
    engine, data = engine_data(tmp_path, 5, 3)
    monkeypatch.setattr(audit, "build_blocked_kernel", lambda obj, rows:
                        lambda values, sequence: obj._calculate_likelihood_stage(values, sequence) * 1.001)
    settings = {"block_rows": 4, "benchmark_sequences": 1, "benchmark_repeats": 1, "benchmark_seed": 12}
    with pytest.raises(ValueError, match="逐元素"):
        compare_kernels(engine, data, settings)


def test_frozen_upstream_guard(tmp_path, monkeypatch):
    from substain_sustain import model
    engine, _ = engine_data(tmp_path, 5, 3)
    def reject():
        raise RuntimeError("upstream source changed")
    monkeypatch.setattr(model, "_upstream_provenance", reject)
    with pytest.raises(RuntimeError, match="source changed"):
        build_blocked_kernel(engine, 4)


# ===== 三、真实CLI整套核查后，正式文件内容及mtime保持不变 =====
def test_audit_cli_isolated_and_has_event_progress(tmp_path):
    cfg, _ = station(tmp_path)
    settings = yaml.safe_load((ROOT / "config/cv_performance.yaml").read_text())
    settings.update(nice_increment=0, min_available_memory_gb=0.001, max_wall_seconds=60,
                    benchmark_sequences=2, benchmark_repeats=1, heartbeat_seconds=0.1)
    path = tmp_path / "audit.yaml"
    path.write_text(yaml.safe_dump(settings))
    formal = Path(cfg["output_dir"])
    before = snapshot(formal)
    run = subprocess.run([sys.executable, str(ROOT / "scripts/audit_cv_performance.py"),
                          "--config", cfg["_config_file"], "--probe-config", str(path)],
                         capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stderr
    assert "逐元素完全相同" in run.stdout and "EM完成" in run.stdout
    assert snapshot(formal) == before
    folder = next((tmp_path / ".cache/cv_performance").glob("*/result.json")).parent
    comparison = json.loads((folder / "kernel_comparison.json").read_text())
    assert comparison["all_bitwise_equal"]
    result = json.loads((folder / "result.json").read_text())
    assert result["em_sweeps_completed"] == 1
    assert result["current_event_rank"] == result["events_in_sweep"] == 3
    assert result["candidate_completed"] == result["candidates_for_event"]
    assert result["em_sweeps"][0]["likelihood_calls"] > 1
