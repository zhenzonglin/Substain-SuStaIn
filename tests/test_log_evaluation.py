"""零似然的冻结链复核、稳定CVIC和后验；原始链及训练算法不改。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pySuStaIn.ZscoreSustain import ZscoreSustain, ZScoreSustainData
from pySuStaIn.AbstractSustain import AbstractSustain

from substain_sustain.log_evaluation import (evaluate_samples, check_raw_mean,
    numerical_settings, official_cv_indices, stable_subtyping, install_stable_subtyping, normalize_log_joint)
from substain_sustain.posterior import infer_posterior

pytestmark = pytest.mark.filterwarnings("ignore:Conversion of an array with ndim > 0 to a scalar:DeprecationWarning:pySuStaIn")


def calculator(tmp_path, data=None):
    if data is None:
        data = np.array([[0.1, 1.1, -0.3], [1.3, 2.1, 0.2], [50, 50, 50], [-50, -50, -50]])
    engine = ZscoreSustain(data, np.ones((3, 1)), np.repeat(3.0, 3), ["A", "B", "C"],
                          1, 2, 8, str(tmp_path), "synthetic", False, seed=13)
    sequences = np.repeat(np.array([[[0], [1], [2]], [[2], [1], [0]]]), 8, axis=2)
    fractions = np.repeat(np.array([[0.8], [0.2]]), 8, axis=1)
    return engine, ZScoreSustainData(data, 3), sequences, fractions


def test_raw_reproduction_and_log_evaluation_are_the_same_formula(tmp_path):
    engine, data, sequences, fractions = calculator(tmp_path)
    settings = numerical_settings({})
    result = evaluate_samples(engine, data, sequences, fractions, settings)
    raw = engine._evaluate_likelihood_setofsamples(data, sequences, fractions)
    np.testing.assert_array_equal(result["raw"], raw)
    assert np.all(result["raw_mean"][2:] == 0)
    assert np.isfinite(result["log_mean"]).all() and np.all(result["log_mean"][2:] < -1000)
    np.testing.assert_allclose(result["log_mean"][:2], np.log(raw.mean(axis=1)[:2]), rtol=0, atol=1e-12)
    check_raw_mean(raw.mean(axis=1), result, len(data.data), settings)
    assert np.isfinite(-2*result["log_mean"].sum())
    with np.errstate(divide="ignore"):
        assert np.isinf(-2*np.log(raw.mean(axis=1)).sum())
    # 独立高精度求和验证极小值；logsumexp不能简单改成log(mean+epsilon)。
    long_logs = result["logs"][2].astype(np.longdouble)
    expected = np.log(np.mean(np.exp(long_logs)))
    assert float(result["log_mean"][2]) == pytest.approx(float(expected), abs=1e-10)


@pytest.mark.parametrize("iterations", [9, 99999, 100000, 999999, 1000000])
def test_cv_indices_match_the_pinned_official_function(iterations):
    class Fake:
        def _calculate_likelihood(self, data, sequence, fractions):
            return 0, np.repeat(sequence[0, 0], data.getNumSamples()), None, None, None
    sequences = np.arange(iterations, dtype=float).reshape(1, 1, -1)
    samples = AbstractSustain._evaluate_likelihood_setofsamples(Fake(), ZScoreSustainData(np.zeros((2, 1)), 1),
                                                              sequences, np.ones((1, iterations)))
    np.testing.assert_array_equal(samples[0], official_cv_indices(iterations))


def test_zero_cache_is_not_blindly_accepted(tmp_path):
    engine, data, sequence, fractions = calculator(tmp_path)
    settings = numerical_settings({})
    result = evaluate_samples(engine, data, sequence, fractions, settings)
    forged = result["raw_mean"].copy()
    forged[0] = 0
    with pytest.raises(ValueError, match="不一致"):
        check_raw_mean(forged, result, len(forged), settings)
    for bad in (np.array([np.nan]*4), np.array([-1.0]*4), np.ones(3)):
        with pytest.raises(ValueError):
            check_raw_mean(bad, result, 4, settings)


def test_stable_subtyping_restores_training_method_and_rng(tmp_path):
    engine, data, sequence, fractions = calculator(tmp_path)
    original = engine._calculate_likelihood
    state = engine.global_rng.bit_generator.state
    values = stable_subtyping(engine, data, sequence, fractions, 8)
    assert all(np.isfinite(value).all() for value in values)
    np.testing.assert_allclose(values[4].sum(axis=1), 1)
    np.testing.assert_allclose(values[5].sum(axis=1), 1)
    assert engine._calculate_likelihood == original and engine.global_rng.bit_generator.state == state
    install_stable_subtyping(engine, numerical_settings({}))
    stable = engine.subtype_and_stage_individuals(data, sequence, fractions, 8)
    for a, b in zip(values, stable):
        np.testing.assert_array_equal(a, b)
    assert engine._calculate_likelihood == original


def test_extreme_posterior_raw_missing_is_explicit_and_batch_invariant(tmp_path):
    engine, data, sequence, fractions = calculator(tmp_path)
    combined = infer_posterior(engine, data.data, sequence, fractions, n_samples=8, burn_in_fraction=0, batch_size=4)
    batched = infer_posterior(engine, data.data, sequence, fractions, n_samples=8, burn_in_fraction=0, batch_size=1)
    np.testing.assert_array_equal(combined["official_raw_valid"], [True, True, False, False])
    assert np.isnan(combined["official_prob_subtype"][2:]).all()
    assert np.isfinite(combined["official_stable_prob_subtype"]).all()
    np.testing.assert_allclose(combined["posterior_joint"], batched["posterior_joint"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(combined["posterior_joint"].sum(axis=(1, 2)), 1)
    np.testing.assert_allclose(combined["posterior_subtype"], combined["official_stable_prob_subtype"])
    np.testing.assert_allclose(combined["posterior_stage"], combined["official_stable_prob_stage"])


def test_actual_k1_zero_cache_preserved_through_migration_and_formal_stages(tmp_path):
    from test_startpoint_migration import migration_station, create_caches, switch
    from substain_sustain.model import run_crossvalidate, run_fit, run_validate
    cfg, settings, _, folds = migration_station(tmp_path, k_max=1)
    # 合成的极端值，仅用于重现真实数值下溢，不改变任何真实队列。
    output = Path(cfg["output_dir"])
    for cohort in ("C3", "GB"):
        for scope in ("primary_z", "z"):
            path = output/f"reference/{cohort}_{scope}.tsv"
            frame = pd.read_csv(path, sep="\t", dtype={"participant_id": str})
            frame.loc[0, ["wmh_a", "gm_b", "gm_c"]] = [50.0, 50.0, 50.0]
            frame.to_csv(path, sep="\t", index=False)
    from substain_sustain.model import _identity
    old = _identity(cfg, output, [output/"reference/C3_primary_z.tsv"], "C3_crossvalidation")
    old["wrapper_source_sha256"] = settings["legacy_wrapper_source_sha256"]
    old.pop("startpoint_parallel_config")
    old.pop("numerical_evaluation_config")
    (output/"model/cv/manifest.json").write_text(json.dumps(old))
    create_caches(cfg, old, folds, through_k=1)
    preserved = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (output/"model/cv").glob("fold_*/pickle_files/*.pickle")}
    target = switch.target_config(cfg, settings)
    audits = switch.audit_cv_caches(target, settings)
    assert sum(row["numerical_evaluation"]["raw_zero_subjects"] for row in audits) == 1
    assert any(row["numerical_evaluation"].get("verified_frozen_underflow") for row in audits)
    assert sorted(row["numerical_evaluation"]["training_at_saved_ml_em"]["linear_zero_subjects"] for row in audits) == [0, 1]
    backup = tmp_path/"backup"
    backup.mkdir()
    switch.migrate(target, switch.prepare_backup(target, settings, backup), backup)
    run_crossvalidate(target, output)
    cvic = pd.read_csv(output/"model/cv/cvic.tsv", sep="\t")
    assert np.isfinite(cvic.cvic).all() and cvic.raw_zero_subjects.sum() == 1
    run_fit(target, output)
    assert run_validate(target, output)["status"] == "complete"
    assert preserved == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in preserved}
    for cohort, folder in (("C3", "model/fit"), ("GB", "validation")):
        with np.load(output/f"{folder}/posterior_{cohort}_primary.npz") as values:
            assert values["official_raw_valid"].sum() == len(values["subject_keys"])-1
            assert np.isfinite(values["posterior_joint"]).all()


def test_subnormal_raw_outputs_are_retained_but_marked_unreliable(tmp_path):
    engine, data, sequence, fractions = calculator(tmp_path, np.full((2, 3), 24.2))
    raw = engine._evaluate_likelihood_setofsamples(data, sequence, fractions)
    assert np.all(raw > 0) and np.all(raw < np.finfo(float).tiny)
    result = infer_posterior(engine, data.data, sequence, fractions, n_samples=8, burn_in_fraction=0)
    assert result["official_raw_available"].all() and not result["official_raw_valid"].any()
    assert np.isfinite(result["official_prob_subtype"]).all()
    assert np.isfinite(result["official_stable_prob_subtype"]).all()


def test_likelihood_diagnostic_is_read_only(tmp_path):
    from test_startpoint_migration import migration_station, create_caches
    from test_cv_probe import snapshot
    import subprocess
    import sys
    import yaml
    cfg, settings, old, folds = migration_station(tmp_path, k_max=1)
    create_caches(cfg, old, folds, through_k=1)
    migration = tmp_path/"migration.yaml"
    migration.write_text(yaml.safe_dump(settings))
    diagnostic = tmp_path/"diagnostic.yaml"
    diagnostic.write_text(yaml.safe_dump({"migration_settings": str(migration),
                                        "output_dir": ".cache/heldout_diagnostics", "audit_max_k": 1}))
    output = Path(cfg["output_dir"])
    before = snapshot(output)
    command = [sys.executable, str(Path(__file__).resolve().parents[1]/"scripts/diagnose_likelihood.py"),
               "--config", cfg["_config_file"], "--settings", str(diagnostic)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr
    assert before == snapshot(output)
    assert "未停止任务" in result.stdout


def test_large_common_log_offset_preserves_subtype_weights_and_normalization():
    stages = np.full((1, 2, 2), -1e20)
    result = normalize_log_joint(stages, np.array([0.9, 0.1]))
    np.testing.assert_allclose(result, [[[0.45, 0.05], [0.45, 0.05]]], rtol=0, atol=1e-15)
    np.testing.assert_allclose(result.sum(axis=(1, 2)), 1, rtol=0, atol=1e-15)
