"""38维模型：原始40维保留、LS独立排除证据和冻结字典身份。"""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from substain_sustain.config import load_config
from substain_sustain.feature_selection import frozen_dictionary, selection_policy, model_frame, select_for_reference
from substain_sustain.ingest import atomic_tsv
from substain_sustain.model import _features, _read_z, _settings as model_settings
from substain_sustain.reference import load_reference, run_reference
from test_reference import make_run, update_snapshot

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = "wmh_basal_ganglia_layer4_ml"
MERGED = "gm_striatopallidal_basal_forebrain_ml"
SOURCES = ["gm_striatopallidal_ml", "gm_basal_forebrain_ml"]
pytestmark = pytest.mark.filterwarnings("ignore:Conversion of an array with ndim > 0 to a scalar:DeprecationWarning:pySuStaIn")


def make_38(tmp_path):
    cfg, output, ls, dictionary = make_run(tmp_path)
    cfg["feature_selection"] = load_config(ROOT / "config/analysis.yaml")["feature_selection"]
    cfg["paths"]["model_mapping"] = load_config(ROOT / "config/analysis.yaml")["paths"]["model_mapping"]
    ls[EXCLUDED] = 0.0
    update_snapshot(output, "LS", ls)
    return cfg, output, ls, dictionary


def test_38_preserves_raw40_and_atlas_indices_and_omits_excluded_from_z(tmp_path):
    cfg, output, _, raw_dictionary = make_38(tmp_path)
    before = (output / "inputs/LS.tsv").read_bytes()
    run_reference(cfg, output)
    assert (output / "inputs/LS.tsv").read_bytes() == before
    frozen = load_reference(output / "reference/model.json")
    assert len(frozen["feature_ids"]) == 38 and EXCLUDED not in frozen["feature_ids"]
    selected = frozen_dictionary(output, cfg)
    assert selected.loc[selected.modality == "WMH", "atlas_index"].tolist() == [1, 2, 3, *range(5, 21)]
    assert len(pd.read_csv(output / "inputs/feature_dictionary.tsv", sep="\t")) == 40
    for name in ("LS_z.tsv", "LS_oof_z.tsv", "C3_z.tsv", "GB_z.tsv"):
        frame = pd.read_csv(output / "reference" / name, sep="\t")
        assert EXCLUDED not in frame and len(set(frame.columns) & set(raw_dictionary.feature_id)) == 37
    _, values = _read_z(output / "reference/C3_primary_z.tsv", _features(output, cfg))
    assert values.shape == (19, 38)
    assert frozen["feature_selection"]["excluded"][0]["ls_primary_n"] == 120
    assert pd.read_csv(output / "reference/feature_exclusions.tsv", sep="\t").feature_id.tolist() == [EXCLUDED]
    gm = selected.loc[selected.modality == "GM"].set_index("atlas_index")
    assert gm.index.tolist() == list(range(1, 20))
    assert gm.at[16, "feature_id"] == "gm_thalamus_ml"
    assert gm.at[17, "feature_id"] == MERGED
    assert json.loads(gm.at[17, "source_atlas_indices"]) == [17, 19]
    assert gm.at[18, "feature_id"] == "gm_hippocampus_amygdala_ml"
    assert gm.at[19, "feature_id"] == "gm_cerebellar_cortex_vermis_ml"


def test_merge_is_raw_volume_sum_then_log_and_ls_ols(tmp_path):
    """独立用statsmodels验证合并区的系数与z；不能平均旧z或丢失基底前脑。"""
    import statsmodels.api as sm
    from substain_sustain.reference import _design, transform_frame
    cfg, output, ls, _ = make_38(tmp_path)
    run_reference(cfg, output)
    model = load_reference(output / "reference/model.json")
    raw = pd.read_csv(output / "reference/LS_model_raw.tsv", sep="\t")
    np.testing.assert_allclose(raw[MERGED], ls[SOURCES].sum(axis=1))
    design = _design(ls, model["design"])
    response = np.log(ls[SOURCES].to_numpy(float).sum(axis=1))
    expected = sm.OLS(response, design, hasconst=True).fit()
    index = model["feature_ids"].index(MERGED)
    np.testing.assert_allclose(np.array(model["coefficients"])[:, index], expected.params, atol=1e-12)
    actual = transform_frame(ls, model)
    sd = np.sqrt(np.sum(expected.resid ** 2) / expected.df_resid)
    np.testing.assert_allclose(actual[MERGED], -expected.resid / sd, atol=1e-10)
    abnormal = ls.iloc[[0]].copy()
    abnormal[SOURCES] *= .8
    assert transform_frame(abnormal, model)[MERGED].iloc[0] > actual[MERGED].iloc[0]
    raw_gm = [c for c in ls if c.startswith("gm_")]
    active_gm = [c for c in raw if c.startswith("gm_")]
    np.testing.assert_allclose(raw[active_gm].sum(axis=1), ls[raw_gm].sum(axis=1))


def test_gm19_mapping_and_model_space_preserve_all_native_labels(tmp_path):
    from substain_sustain.assets import project_model_atlas
    cfg, output, _, _ = make_38(tmp_path)
    run_reference(cfg, output)
    mapping = pd.read_csv(output / "reference/gm_model_mapping.tsv", sep="\t")
    assert len(mapping) == mapping.native_label.nunique() == 119
    assert set(mapping.macro_index) == set(range(1, 20))
    assert set(mapping.loc[mapping.macro_index == 17, "native_label"]) == {23, 30, 36, 37, 55, 56, 57, 58, 75, 76}
    assert set(mapping.loc[mapping.macro_index == 16, "native_label"]) == {59, 60}
    assert not set(mapping.native_label).intersection({35, 61, 62})
    dictionary = frozen_dictionary(output, cfg)
    atlas = np.arange(21).reshape(3, 7, 1)
    projected = project_model_atlas(atlas, dictionary, "GM").ravel()
    assert projected.tolist() == [*range(19), 17, 19]
    wmh = project_model_atlas(atlas, dictionary, "WMH").ravel()
    assert wmh[4] == 0 and wmh[5] == 5


@pytest.mark.parametrize("problem", ["missing", "nan", "zero"])
def test_missing_or_invalid_merge_source_cannot_be_summed_as_zero(tmp_path, problem):
    cfg, _, ls, dictionary = make_38(tmp_path)
    _, selection = select_for_reference(dictionary, ls, cfg)
    if problem == "missing":
        ls = ls.drop(columns=SOURCES[1])
    else:
        ls.loc[0, SOURCES[1]] = np.nan if problem == "nan" else 0
    with pytest.raises(ValueError, match="GM合并"):
        model_frame(ls, selection)


def test_previous_thalamus_merge_cannot_reuse_final_merge_cache(tmp_path):
    cfg, output, _, _ = make_38(tmp_path)
    run_reference(cfg, output)
    old = copy.deepcopy(cfg)
    old["feature_selection"]["merge"][0]["source_features"] = ["gm_thalamus_ml", "gm_striatopallidal_ml"]
    with pytest.raises(ValueError, match="配置.*冻结LS"):
        frozen_dictionary(output, old)
    old["paths"]["model_mapping"] = cfg["paths"]["model_mapping"]
    with pytest.raises(ValueError, match="GM19映射"):
        run_reference(old, output)


def test_nonzero_ls_does_not_satisfy_exclusion_criterion(tmp_path):
    cfg, output, ls, _ = make_38(tmp_path)
    ls.loc[0, EXCLUDED] = .001
    update_snapshot(output, "LS", ls)
    with pytest.raises(ValueError, match="排除依据不成立"):
        run_reference(cfg, output)


def test_other_zero_feature_still_blocks_without_automatic_exclusion(tmp_path):
    cfg, output, ls, dictionary = make_38(tmp_path)
    ls[dictionary.feature_id.iloc[0]] = 0
    update_snapshot(output, "LS", ls)
    with pytest.raises(ValueError, match="残差SD"):
        run_reference(cfg, output)


def test_changed_policy_or_dictionary_cannot_reuse_reference(tmp_path):
    cfg, output, _, _ = make_38(tmp_path)
    run_reference(cfg, output)
    other = copy.deepcopy(cfg)
    other["feature_selection"] = {"expected_model_features": 40, "exclude": []}
    with pytest.raises(ValueError, match="配置.*冻结LS"):
        frozen_dictionary(output, other)
    path = output / "reference/feature_dictionary.tsv"
    frame = pd.read_csv(path, sep="\t")
    frame.at[3, "atlas_index"] = 4
    atomic_tsv(path, frame)
    with pytest.raises(ValueError, match="atlas标签"):
        frozen_dictionary(output, cfg)


def test_gb_excluded_raw_values_do_not_affect_reference_or_selection(tmp_path):
    cfg1, out1, _, _ = make_38(tmp_path / "a")
    cfg2, out2, _, _ = make_38(tmp_path / "b")
    gb = pd.read_csv(out2 / "inputs/GB.tsv", sep="\t", dtype={"participant_id": str})
    gb[EXCLUDED] = 999
    gb[SOURCES] *= 2.0
    update_snapshot(out2, "GB", gb)
    a, b = run_reference(cfg1, out1), run_reference(cfg2, out2)
    assert a["reference_hash"] == b["reference_hash"]
    assert (out1 / "reference/model.json").read_bytes() == (out2 / "reference/model.json").read_bytes()
    assert (out1 / "reference/feature_selection.json").read_bytes() == (out2 / "reference/feature_selection.json").read_bytes()
    assert a["transform_hash"] != b["transform_hash"]


def test_empty_c3_stops_before_reference_writes(tmp_path):
    cfg, output, _, _ = make_38(tmp_path)
    frame = pd.read_csv(output / "inputs/C3.tsv", sep="\t").iloc[:0]
    update_snapshot(output, "C3", frame)
    with pytest.raises(ValueError, match="C3主年龄范围内导入人数为0"):
        run_reference(cfg, output)
    assert not (output / "reference").exists()


def test_old40_reference_cannot_be_used_under_38_config(tmp_path):
    cfg, output, _, _ = make_run(tmp_path)
    run_reference(cfg, output)
    cfg["feature_selection"] = load_config(ROOT / "config/analysis.yaml")["feature_selection"]
    with pytest.raises(ValueError, match="特征排除配置"):
        frozen_dictionary(output, cfg)


def test_formal_event_contract_is_38_features_114_events():
    cfg = load_config(ROOT / "config/analysis.yaml")
    settings = model_settings(cfg, 38)
    assert 38 * len(settings["z_vals"]) == 114
    with pytest.raises(ValueError, match="38维"):
        model_settings(cfg, 40)
    cfg["feature_selection"]["expected_model_features"] = 40
    with pytest.raises(ValueError, match="减去"):
        selection_policy(cfg)


def test_model_reload_requires_selection_evidence_for_38(tmp_path):
    cfg, output, _, _ = make_38(tmp_path)
    run_reference(cfg, output)
    path = output / "reference/model.json"
    model = json.loads(path.read_text())
    model.pop("feature_selection")
    path.write_text(json.dumps(model))
    with pytest.raises(ValueError, match="维数"):
        load_reference(path)


@pytest.mark.skipif(os.environ.get("SUBSTAIN_RUN_38_INTEGRATION") != "1",
                    reason="38维真实官方EM/MCMC，含上游固定3×10000次提议调优")
def test_real_official_38_feature_reference_cv_fit_and_validation(tmp_path):
    """完整38维合成技术验证；不是正式队列发现，短链不证明收敛。"""
    from substain_sustain.model import run_crossvalidate, run_fit, run_validate
    cfg, output, _, _ = make_38(tmp_path)
    defaults = load_config(ROOT / "config/analysis.yaml")
    cfg["model"] = {**defaults["model"], "k_max": 1, "folds": 2, "startpoints": 1,
                    "cv_iterations": 30, "fit_iterations": 30, "stability_iterations": 30,
                    "stability_seeds": [], "posterior_samples": 5, "display_samples": 5,
                    "posterior_batch_size": 4, "workers": 1}
    # 测试路径和数据为合成；formal用于严格执行生产38维合同。
    cfg["runtime"] = {**defaults["runtime"], "profile": "formal"}
    run_reference(cfg, output)
    assert run_crossvalidate(cfg, output)["n_subtypes"] == 1
    result = run_fit(cfg, output)
    assert result["n_features"] == 38 and result["n_events"] == 114
    assert result["stage_max"] == 114
    assert run_validate(cfg, output)["n_primary"] == 19
    with np.load(output / "validation/posterior_GB_primary.npz") as posterior:
        assert posterior["posterior_joint"].shape == (19, 115, 1)
        np.testing.assert_allclose(posterior["posterior_joint"].sum(axis=(1, 2)), 1)
    # 仅检查按已完成模型复用，不宣称支持MCMC链内续跑。
    chain = output / "model/cv/fold_0/pickle_files/C3_fold0_subtype0.pickle"
    before = chain.stat().st_mtime_ns
    run_crossvalidate(cfg, output)
    assert chain.stat().st_mtime_ns == before
