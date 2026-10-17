"""模型合同、缓存身份及真实官方小规模全流程测试。"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from substain_sustain.model import (
    _guard, _identity, _lock, _match, _read_z, _upstream_provenance, _validated_selection,
    estimate_working_bytes, run_crossvalidate, run_fit, run_validate,
)

pytestmark = pytest.mark.filterwarnings("ignore:Conversion of an array with ndim > 0 to a scalar:DeprecationWarning:pySuStaIn")


def test_upstream_is_exact_frozen_code():
    pytest.importorskip("pySuStaIn")
    assert len(_upstream_provenance()) == 2


def test_cached_identity_rejects_changed_inputs(tmp_path):
    with _lock(tmp_path):
        _guard(tmp_path, {"input_hash": "a"})
    with _lock(tmp_path):
        _guard(tmp_path, {"input_hash": "a"})
        with pytest.raises(ValueError, match="缓存身份"):
            _guard(tmp_path, {"input_hash": "b"})


def test_sequence_matching_recovers_swapped_labels():
    sequence = np.repeat(np.array([[[0], [1], [2]], [[2], [1], [0]]]), 5, axis=2)
    order, tau = _match(sequence, sequence[::-1])
    np.testing.assert_array_equal(order, [1, 0])
    np.testing.assert_allclose(tau, 1)


def test_memory_estimate_increases_with_events_and_rows():
    base = estimate_working_bytes(100, 3, 9, 2, 100)
    assert estimate_working_bytes(200, 3, 9, 2, 100) > base
    assert estimate_working_bytes(100, 40, 120, 2, 100) > base


def test_input_order_comes_from_dictionary_and_duplicates_fail(tmp_path):
    features = pd.DataFrame({"feature_id": ["a", "b"]})
    frame = pd.DataFrame({"subject_key": ["C3:001"], "participant_id": ["001"], "cohort": ["C3"],
                          "age": [60], "sex": ["female"], "icv_ml": [1300], "b": [2], "a": [1]})
    path = tmp_path / "z.tsv"
    frame.to_csv(path, sep="\t", index=False)
    loaded, array = _read_z(path, features)
    assert loaded.participant_id.iloc[0] == "001"
    np.testing.assert_array_equal(array, [[1, 2]])
    pd.concat([frame, frame]).to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="重复"):
        _read_z(path, features)


def _synthetic_project(tmp_path):
    """只创建独立测试数据；不使用任何患者资料或降低官方调优轮数。"""
    (tmp_path / "inputs").mkdir()
    (tmp_path / "reference").mkdir()
    features = pd.DataFrame({"feature_id": ["wmh_a", "gm_b", "gm_c"], "label": ["A", "B", "C"],
                             "modality": ["WMH", "GM", "GM"], "atlas_index": [1, 2, 3]})
    features.to_csv(tmp_path / "inputs" / "feature_dictionary.tsv", sep="\t", index=False)
    rng = np.random.default_rng(2026)
    for cohort, n in (("C3", 20), ("GB", 5)):
        stage = rng.uniform(0, 2.5, size=(n, 1))
        values = stage * np.array([[1.0, 0.7, 0.4]]) + rng.normal(0, 0.5, size=(n, 3))
        frame = pd.DataFrame(values, columns=features.feature_id)
        frame.insert(0, "icv_ml", 1400)
        frame.insert(0, "sex", "female")
        frame.insert(0, "age", 60)
        frame.insert(0, "cohort", cohort)
        frame.insert(0, "participant_id", [f"{i:03d}" for i in range(n)])
        frame.insert(0, "subject_key", [f"{cohort}:{i:03d}" for i in range(n)])
        for suffix in ("_primary_z.tsv", "_z.tsv"):
            frame.to_csv(tmp_path / "reference" / f"{cohort}{suffix}", sep="\t", index=False)
    for name in ("model.json", "manifest.json"):
        (tmp_path / "reference" / name).write_text(json.dumps({"synthetic_test_only": True}))
    cfg = {"model": {"k_max": 1, "folds": 2, "startpoints": 1, "cv_iterations": 30,
                     "fit_iterations": 40, "stability_iterations": 30, "seed": 2026,
                     "stability_seeds": [2027], "z_vals": [1], "z_max": 3,
                     "posterior_samples": 5, "posterior_batch_size": 2, "workers": 2,
                     "display_samples": 5, "burn_in_fraction": 0.1},
           "runtime": {"profile": "synthetic_test", "max_memory_fraction": 0.5,
                       "memory_safety_factor": 2, "disk_safety_factor": 1.2, "blas_threads": 1}}
    return cfg


def test_c3_training_identity_does_not_depend_on_gb_snapshot(tmp_path):
    cfg = _synthetic_project(tmp_path)
    (tmp_path / "reference" / "model.json").write_text(json.dumps({"reference_hash": "LS-only-model"}))
    manifest = tmp_path / "reference" / "manifest.json"
    manifest.write_text(json.dumps({"input_snapshot_hash": "snapshot_a", "transform_hash": "transform_a"}))
    files = [tmp_path / "reference" / "C3_primary_z.tsv"]
    first_train = _identity(cfg, tmp_path, files, "C3_crossvalidation")
    first_validation = _identity(cfg, tmp_path, files, "GB_frozen_validation")
    manifest.write_text(json.dumps({"input_snapshot_hash": "snapshot_b", "transform_hash": "transform_b"}))
    assert _identity(cfg, tmp_path, files, "C3_crossvalidation") == first_train
    assert _identity(cfg, tmp_path, files, "GB_frozen_validation") != first_validation


def _selection_fixture(tmp_path, cfg):
    directory = tmp_path / "model" / "cv"
    directory.mkdir(parents=True)
    identity = _identity(cfg, tmp_path, [tmp_path / "reference" / "C3_primary_z.tsv"], "C3_crossvalidation")
    (directory / "manifest.json").write_text(json.dumps(identity))
    (directory / "selection.json").write_text(json.dumps({"n_subtypes": 1, "candidate_k": [1], "n_folds": 2}))
    pd.DataFrame({"n_subtypes": [1], "cvic": [100.0]}).to_csv(directory / "cvic.tsv", sep="\t", index=False)
    for fold in range(2):
        path = directory / f"fold_{fold}" / "pickle_files"
        path.mkdir(parents=True)
        (path.parent / "complete.json").write_text(json.dumps({"status": "complete"}))
        (path / f"C3_fold{fold}_subtype0.pickle").touch()


def test_fit_rejects_changed_event_endpoint_before_writing(tmp_path):
    cfg = _synthetic_project(tmp_path)
    _selection_fixture(tmp_path, cfg)
    cfg["model"]["z_max"] = 7
    with pytest.raises(ValueError, match="z_max.*CV"):
        run_fit(cfg, tmp_path)
    assert not (tmp_path / "model" / "fit").exists()


def test_final_iterations_can_change_without_changing_cv_selection(tmp_path):
    cfg = _synthetic_project(tmp_path)
    _selection_fixture(tmp_path, cfg)
    cfg["model"]["fit_iterations"] = 400
    cfg["model"]["stability_iterations"] = 300
    assert _validated_selection(cfg, tmp_path)["n_subtypes"] == 1


@pytest.mark.skipif(os.environ.get("SUBSTAIN_RUN_OFFICIAL_INTEGRATION") != "1",
                    reason="显式运行真实官方 EM/MCMC，含每模型 3×10000 调优")
def test_real_official_end_to_end_and_resume(tmp_path):
    cfg = _synthetic_project(tmp_path)
    selected = run_crossvalidate(cfg, tmp_path)
    assert selected["n_subtypes"] == 1
    summary = run_fit(cfg, tmp_path)
    assert summary["status"] == "complete"
    validation = run_validate(cfg, tmp_path)
    assert validation["n_primary"] == 5
    assert validation["gb_used_for_selection"] is False
    posterior = np.load(tmp_path / "validation" / "posterior_GB_primary.npz")
    np.testing.assert_allclose(posterior["posterior_joint"].sum(axis=(1, 2)), 1)
    chain = tmp_path / "model" / "cv" / "fold_0" / "pickle_files" / "C3_fold0_subtype0.pickle"
    timestamp = chain.stat().st_mtime_ns
    run_crossvalidate(cfg, tmp_path)
    assert chain.stat().st_mtime_ns == timestamp
    cfg["model"]["seed"] += 1
    with pytest.raises(ValueError, match="缓存身份"):
        run_crossvalidate(cfg, tmp_path)
