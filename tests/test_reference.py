"""LS常模测试：冻结中心样条、OOF无泄漏、疾病方向与失败门槛。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from substain_sustain.ingest import atomic_json, atomic_tsv, load_mapping, read_tsv, sha256_file
from substain_sustain.reference import (
    _design, _settings, fit_reference, load_reference, run_reference, transform_frame,
)

MAPPING = Path(__file__).resolve().parents[1] / "resources/mappings/muse_gm_macro20_v1.tsv"


# ---------- 合成生物标志物；不作为真实临床证据 ----------
def synthetic(n=120, cohort="LS", seed=13):
    _, dictionary = load_mapping(MAPPING)
    rng = np.random.default_rng(seed)
    age = rng.uniform(50.1, 74.9, n)
    sex = np.array(["female", "male"] * (n // 2) + (["female"] if n % 2 else []))
    icv = np.exp(7.2 + 0.05 * (sex == "male") + rng.normal(0, 0.1, n))
    frame = pd.DataFrame({"participant_id": [f"{index:05d}" for index in range(n)],
                          "subject_key": [f"{cohort}:{index:05d}" for index in range(n)], "cohort": cohort,
                          "age": age, "sex": sex, "icv_ml": icv, "manual_qc_state": "unreviewed"})
    for index, row in dictionary.iterrows():
        latent = 1.5 + 0.007 * (age - 62) + 0.03 * (sex == "male") + 0.5 * (np.log(icv) - 7.2)
        latent += rng.normal(0, 0.08 + index * 0.001, n)
        frame[row.feature_id] = np.expm1(latent) if row.modality == "WMH" else np.exp(latent)
    return frame, dictionary


def make_run(tmp_path):
    output = tmp_path / "run"
    inputs = output / "inputs"
    inputs.mkdir(parents=True)
    ls, dictionary = synthetic()
    c3, _ = synthetic(20, "C3", seed=16)
    gb, _ = synthetic(20, "GB", seed=17)
    c3.loc[0, "age"], gb.loc[0, "age"] = 42.0, 81.0
    for cohort, frame in (("LS", ls), ("C3", c3), ("GB", gb)):
        atomic_tsv(inputs / f"{cohort}.tsv", frame)
    atomic_tsv(inputs / "feature_dictionary.tsv", dictionary)
    hashes = {path.name: sha256_file(path) for path in inputs.iterdir()}
    atomic_json(inputs / "manifest.json", {"input_hash": "synthetic_inputs", "files": hashes})
    cfg = {"paths": {"mapping": str(MAPPING)}, "reference": {"folds": 5, "seed": 42, "min_reference_n": 40},
           "cohorts": {cohort: {"derivatives": str(tmp_path / "source" / cohort),
                                "participants": str(tmp_path / "source" / cohort / "participants.tsv")}
                       for cohort in ("LS", "C3", "GB")}}
    return cfg, output, ls, dictionary


# ---------- 冻结样条可重载，并保留截距与中心约束 ----------
def test_json_reload_reproduces_centered_full_rank_design(tmp_path):
    frame, dictionary = synthetic()
    model = fit_reference(frame, dictionary, _settings({}))
    design = _design(frame, model["design"])
    assert design.shape == (120, 7)
    assert np.linalg.matrix_rank(design) == 7
    np.testing.assert_allclose(design[:, 1:5].mean(axis=0), 0, atol=1e-14)
    atomic_json(tmp_path / "model.json", model)
    loaded = load_reference(tmp_path / "model.json")
    np.testing.assert_allclose(_design(frame, loaded["design"]), design, atol=0, rtol=0)
    before, after = transform_frame(frame, model), transform_frame(frame, loaded)
    np.testing.assert_allclose(before[dictionary.feature_id], after[dictionary.feature_id], atol=0, rtol=0)


def test_coefficients_and_residual_scale_match_statsmodels_ols():
    frame, dictionary = synthetic()
    model = fit_reference(frame, dictionary, _settings({}))
    design = _design(frame, model["design"])
    for index in (0, 20):
        raw = frame[dictionary.feature_id.iloc[index]].to_numpy()
        values = np.log1p(raw) if index == 0 else np.log(raw)
        expected = sm.OLS(values, design, hasconst=True, missing="raise").fit(method="pinv")
        np.testing.assert_allclose(np.asarray(model["coefficients"])[:, index], expected.params, rtol=0, atol=0)
        assert model["residual_sd"][index] == np.sqrt(np.sum(expected.resid ** 2) / expected.df_resid)
    assert model["estimator"] == "statsmodels.api.OLS"


def test_natural_spline_extrapolation_is_linear_and_flagged():
    frame, dictionary = synthetic()
    model = fit_reference(frame, dictionary, _settings({}))
    extrapolate = pd.concat([frame.iloc[[0]].copy() for _ in range(3)], ignore_index=True)
    extrapolate["subject_key"] = ["C3:a", "C3:b", "C3:c"]
    extrapolate["cohort"] = "C3"
    extrapolate["age"] = [80.0, 85.0, 90.0]
    design = _design(extrapolate, model["design"])
    prediction = design @ np.asarray(model["coefficients"])
    np.testing.assert_allclose(prediction[1] - prediction[0], prediction[2] - prediction[1], atol=1e-12)
    assert not transform_frame(extrapolate, model).age_in_support.any()


def test_disease_direction_increases_with_wmh_and_gm_loss():
    frame, dictionary = synthetic()
    model = fit_reference(frame, dictionary, _settings({}))
    baseline = frame.iloc[[0]].copy()
    abnormal = baseline.copy()
    wmh = dictionary.loc[dictionary.modality == "WMH", "feature_id"].iloc[0]
    gm = dictionary.loc[dictionary.modality == "GM", "feature_id"].iloc[0]
    abnormal[wmh] *= 2
    abnormal[gm] *= 0.7
    before, after = transform_frame(baseline, model), transform_frame(abnormal, model)
    assert after[wmh].iloc[0] > before[wmh].iloc[0]
    assert after[gm].iloc[0] > before[gm].iloc[0]


# ---------- OOF分折和正式产物的身份保护 ----------
def test_reference_pipeline_preserves_oof_exclusion_and_primary_age_range(tmp_path):
    cfg, output, ls, dictionary = make_run(tmp_path)
    manifest = run_reference(cfg, output)
    assert manifest["training_n"] == 120
    assert manifest["cohorts"]["C3"] == {"all_ages": 20, "primary": 19, "age_extrapolated": 1}
    assert manifest["cohorts"]["GB"]["primary"] == 19
    folds = read_tsv(output / "reference/fold_assignments.tsv")
    fold_models = json.loads((output / "reference/fold_models.json").read_text())["models"]
    for index, model in enumerate(fold_models):
        held_out = set(folds.loc[folds.fold == str(index), "subject_key"])
        assert held_out.isdisjoint(model["training_subject_keys"])
        assert len(held_out) == 24
    diagnostics = read_tsv(output / "reference/diagnostics.tsv")
    assert set(diagnostics.split) == {"LS_fit", "LS_oof", "fold_0", "fold_1", "fold_2", "fold_3", "fold_4"}
    assert len(diagnostics) == 40 * 7
    assert len(read_tsv(output / "reference/LS_oof_z.tsv")) == 120
    assert run_reference(cfg, output)["reference_hash"] == manifest["reference_hash"]
    cfg["reference"]["seed"] += 1
    with pytest.raises(ValueError, match="不同输入或参数"):
        run_reference(cfg, output)


def test_only_ls_trains_reference_and_rank_failures_are_explicit():
    frame, dictionary = synthetic()
    frame["cohort"] = "GB"
    with pytest.raises(ValueError, match="只能在LS"):
        fit_reference(frame, dictionary, _settings({}))
    frame["cohort"] = "LS"
    frame["icv_ml"] = 1500
    with pytest.raises(ValueError, match="不满秩"):
        fit_reference(frame, dictionary, _settings({}))


def test_constant_residual_feature_blocks_reference():
    frame, dictionary = synthetic()
    frame[dictionary.feature_id.iloc[0]] = 1
    with pytest.raises(ValueError, match="残差SD"):
        fit_reference(frame, dictionary, _settings({}))


def test_input_modification_cannot_silently_reuse_reference(tmp_path):
    cfg, output, _, _ = make_run(tmp_path)
    run_reference(cfg, output)
    path = output / "inputs/C3.tsv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="哈希改变"):
        run_reference(cfg, output)


def update_snapshot(output, cohort, frame):
    """模拟合法的新输入快照，更新表hash但不改任何LS参考数据。"""
    table = output / "inputs" / f"{cohort}.tsv"
    atomic_tsv(table, frame)
    path = output / "inputs/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][f"{cohort}.tsv"] = sha256_file(table)
    manifest["input_hash"] = "synthetic_inputs_gb_changed"
    atomic_json(path, manifest)


def test_gb_change_never_changes_reference_identity_or_model_json(tmp_path):
    cfg1, output1, _, _ = make_run(tmp_path / "first")
    cfg2, output2, _, dictionary = make_run(tmp_path / "second")
    gb = pd.read_csv(output2 / "inputs/GB.tsv", sep="\t", dtype={"participant_id": str})
    gb[dictionary.feature_id.iloc[0]] *= 3.0
    update_snapshot(output2, "GB", gb)
    first, second = run_reference(cfg1, output1), run_reference(cfg2, output2)
    assert first["reference_hash"] == second["reference_hash"]
    assert first["training_data_sha256"] == second["training_data_sha256"]
    assert first["input_snapshot_hash"] != second["input_snapshot_hash"]
    assert first["transform_hash"] != second["transform_hash"]
    assert (output1 / "reference/model.json").read_bytes() == (output2 / "reference/model.json").read_bytes()
    assert (output1 / "reference/GB_z.tsv").read_bytes() != (output2 / "reference/GB_z.tsv").read_bytes()
    assert (output1 / "reference/C3_z.tsv").read_bytes() == (output2 / "reference/C3_z.tsv").read_bytes()
    assert json.loads((output1 / "reference/model.json").read_text())["reference_hash"] == first["reference_hash"]


def test_changed_gb_snapshot_cannot_overwrite_existing_transforms(tmp_path):
    cfg, output, _, dictionary = make_run(tmp_path)
    run_reference(cfg, output)
    previous_model = (output / "reference/model.json").read_bytes()
    gb = pd.read_csv(output / "inputs/GB.tsv", sep="\t", dtype={"participant_id": str})
    gb[dictionary.feature_id.iloc[0]] *= 2
    update_snapshot(output, "GB", gb)
    with pytest.raises(ValueError, match="三队列转换输入已改变"):
        run_reference(cfg, output)
    assert (output / "reference/model.json").read_bytes() == previous_model
