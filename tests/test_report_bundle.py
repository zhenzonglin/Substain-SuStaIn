"""展示交付回归：缺失诚实失败、脱离原项目重画、资源篡改可检出。"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tarfile

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from substain_sustain.assets import validate_assets, write_asset_manifest
from substain_sustain.bundle import create_bundle, verify_bundle
from substain_sustain.report import DEFAULTS, REQUIRED_FILES, ReportWriter, _plot_posterior_trace, _trajectory_table, run_report


def write_tsv(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, sep="\t", index=False)


@pytest.fixture
def report_case(tmp_path):
    """小型全流程数值夹具，atlas也明确为合成；不冒充真实数据。"""
    output = tmp_path / "analysis"
    assets = tmp_path / "public_display"
    assets.mkdir()
    cfg = {"runtime": {"profile": "synthetic"}, "reference": {"age_min": 50, "age_max": 75},
           "report": {"formats": ["png", "svg", "tiff"], "preview_dpi": 40, "raster_dpi": 60,
                      "figure_width": 5, "figure_height": 3, "font_size": 7, "stage_points": [.25, .75],
                      "slice_fractions": [.3, .6], "max_posterior_draws": 4},
           "bundle": {"include_full_posterior": True}, "assets": {}}
    atlas = np.zeros((12, 12, 12), dtype=np.int16)
    atlas[2:6, 2:10, 2:10] = 1
    atlas[6:10, 2:10, 2:10] = 2
    for key in ("wmh_atlas", "gm_atlas", "background"):
        path = assets / (key + ".nii.gz")
        nib.save(nib.Nifti1Image(atlas.astype(np.float32) if key == "background" else atlas, np.eye(4)), path)
        cfg["assets"][key] = str(path)
    cfg["assets"]["manifest"] = str(assets / "manifest.json")
    write_asset_manifest(cfg["assets"], assets / "manifest.json", "synthetic test fixture")
    dictionary = pd.DataFrame({"feature_id": ["wmh_a", "wmh_b", "gm_a", "gm_b"],
                               "label": ["WMH A", "WMH B", "GM A", "GM B"],
                               "modality": ["WMH", "WMH", "GM", "GM"], "atlas_index": [1, 2, 1, 2]})
    write_tsv(output / "inputs/feature_dictionary.tsv", dictionary)
    rng = np.random.default_rng(20)
    for cohort in ("LS", "C3", "GB"):
        frame = pd.DataFrame({"subject_key": [cohort + ":" + str(i) for i in range(8)],
                              "participant_id": [f"P{i:03d}" for i in range(8)], "cohort": cohort,
                              "age": [45, 55, 60, 61, 65, 68, 72, 80], "sex": ["male", "female"] * 4,
                              "icv_ml": [1450] * 8})
        for feature in dictionary.feature_id:
            frame[feature] = rng.normal(size=8)
        write_tsv(output / "inputs" / (cohort + ".tsv"), frame)
        write_tsv(output / "reference" / (cohort + "_z.tsv"), frame)
        write_tsv(output / "reference" / (cohort + "_primary_z.tsv"), frame.iloc[1:7])
        if cohort == "LS":
            write_tsv(output / "reference/LS_oof_z.tsv", frame)
            continue
        frame["subtype"] = [0, 1, 1, 2, 2, 1, 2, 1]
        frame["stage"] = [0, 1, 2, 4, 5, 7, 9, 10]
        frame["subtype_probability"] = [.5, .8, .7, .9, .8, .6, .8, .9]
        frame["subtype_entropy"] = [.6] * 8
        frame["stage_zero_probability"] = [.8, .1, .1, .1, .1, .1, .1, .1]
        frame["stage_expected"] = frame.stage + .2
        directory = output / ("model/fit" if cohort == "C3" else "validation")
        write_tsv(directory / f"assignments_{cohort}_all.tsv", frame)
        write_tsv(directory / f"assignments_{cohort}_primary.tsv", frame.iloc[1:7])
        np.savez_compressed(directory / f"posterior_{cohort}_all.npz", posterior_joint=np.ones((8, 13, 2)) / 26,
                            posterior_subtype=np.ones((8, 2)) / 2, posterior_stage=np.ones((8, 13)) / 13,
                            subject_keys=frame.subject_key.to_numpy(dtype=str))
        np.savez_compressed(directory / f"posterior_{cohort}_primary.npz", posterior_joint=np.ones((6, 13, 2)) / 26,
                            posterior_subtype=np.ones((6, 2)) / 2, posterior_stage=np.ones((6, 13)) / 13,
                            subject_keys=frame.subject_key.iloc[1:7].to_numpy(dtype=str))
    write_tsv(output / "reference/diagnostics.tsv", {"feature_id": dictionary.feature_id, "split": ["LS_oof"] * 4,
              "mean_z": [0] * 4, "sd_z": [1] * 4, "coverage_95": [.95] * 4,
              "correlation_age": [0] * 4, "correlation_sex": [0] * 4, "correlation_log_icv": [0] * 4})
    write_tsv(output / "model/cv/cvic.tsv", {"n_subtypes": [1, 2], "cvic": [200, 150], "selected": [False, True]})
    write_tsv(output / "model/cv/fold_loglikelihood.tsv", {"fold": [0, 0, 1, 1], "n_subtypes": [1, 2, 1, 2],
                                                        "log_likelihood": [-50, -40, -55, -42]})
    write_tsv(output / "model/fit/stability.tsv", {"seed": [21, 21], "subtype": [1, 2], "kendall_tau": [.8, .9]})
    write_tsv(output / "model/cv/cv_stability.tsv", {"fold": [0, 0], "subtype": [1, 2], "kendall_tau": [.7, .8]})
    for relative in REQUIRED_FILES:
        path = output / relative
        if path.suffix == ".json":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"settings": cfg["reference"], "n_subtypes": 2}), encoding="utf-8")
    sequence = np.arange(12).reshape(3, 4)
    samples = np.empty((2, 12, 4), dtype=int)
    for subtype in range(2):
        for draw in range(4):
            samples[subtype, :, draw] = np.roll(sequence, subtype + draw % 2, axis=1).ravel()
    np.savez_compressed(output / "model/fit/selected_model.npz", samples_sequence=samples,
                        samples_f=np.ones((2, 4)) / 2, samples_likelihood=np.ones((4, 1)), subtype_order=np.arange(2),
                        Z_vals=np.tile([1, 2, 3], (4, 1)), Z_max=np.ones(4) * 5,
                        feature_ids=dictionary.feature_id.to_numpy(dtype=str), labels=dictionary.label.to_numpy(dtype=str))
    shutil.copyfile(output / "model/fit/selected_model.npz", output / "model/fit/cv_display.npz")
    return cfg, output


def test_formal_missing_result_is_explicitly_incomplete(tmp_path):
    with pytest.raises(FileNotFoundError, match="incomplete"):
        run_report({"runtime": {"profile": "formal"}}, tmp_path)
    status = json.loads((tmp_path / "report/summary.json").read_text())
    assert status["status"] == "incomplete"
    assert "FORMAL REPORT INCOMPLETE" in (tmp_path / "report/index.html").read_text()


def test_tampered_display_asset_is_rejected(report_case):
    cfg, output = report_case
    path = Path(cfg["assets"]["wmh_atlas"])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        validate_assets(cfg)


def test_bundle_regenerates_without_original_project_or_figures(report_case, tmp_path):
    cfg, output = report_case
    report = run_report(cfg, output)
    assert report["status"] == "technical_demo_complete"
    assert report["figures"] == 17
    page = (output / "report/index.html").read_text()
    assert "SYNTHETIC DATA" in page
    assert "<script" not in page and "https://" not in page
    bundle = tmp_path / "delivery.tar.gz"
    create_bundle(cfg, output, bundle)
    with tarfile.open(bundle) as archive:
        names = archive.getnames()
    assert not any(name.endswith((".png", ".tiff", ".svg")) for name in names)
    assert not any("pickle_files" in name for name in names)
    # 关键验收：移除原数值源与图谱，重画仅能使用归档本身。
    shutil.rmtree(output)
    shutil.rmtree(Path(cfg["assets"]["manifest"]).parent)
    receipt = verify_bundle(bundle)
    assert receipt["status"] == "pass"
    assert receipt["figures_regenerated"] == report["figures"]
    assert receipt["original_project_required"] is False
    assert receipt["network_disabled"] is True
    assert set(receipt["network_self_tests"]) == {"getaddrinfo", "create_connection", "connect", "connect_ex"}


def test_missing_crossfold_evidence_does_not_claim_complete(report_case):
    cfg, output = report_case
    (output / "model/cv/cv_stability.tsv").unlink()
    with pytest.raises(FileNotFoundError, match="Cross-fold"):
        run_report(cfg, output)
    assert json.loads((output / "report/summary.json").read_text())["status"] == "incomplete"


def test_38_feature_report_and_offline_bundle_preserve_excluded_raw_values(report_case, tmp_path):
    """38维展示夹具验证114事件与离线重画；数值为合成，不声称正式拟合结果。"""
    from substain_sustain.config import load_config
    from substain_sustain.feature_selection import select_for_reference, model_frame
    from substain_sustain.ingest import atomic_json, atomic_tsv, load_mapping, sha256_file
    cfg, output = report_case
    cfg["report"].update(raster_dpi=600, preview_dpi=100, figure_width=10, figure_height=6, font_size=8)
    root = Path(__file__).resolve().parents[1]
    _, raw = load_mapping(root / "resources/mappings/muse_gm_macro20_v1.tsv")
    cfg["feature_selection"] = load_config(root / "config/analysis.yaml")["feature_selection"]
    excluded = "wmh_basal_ganglia_layer4_ml"
    for cohort in ("LS", "C3", "GB"):
        frame = pd.read_csv(output / f"inputs/{cohort}.tsv", sep="\t", dtype={"participant_id": str})
        frame = frame.drop(columns=["wmh_a", "wmh_b", "gm_a", "gm_b"])
        for i, name in enumerate(raw.feature_id):
            frame[name] = np.arange(len(frame)) * .1 + i * .03 + .1
        if cohort == "LS":
            frame[excluded] = 0.0
            selected, selection = select_for_reference(raw, frame.loc[frame.age.between(50, 75)], cfg)
        atomic_tsv(output / f"inputs/{cohort}.tsv", frame)
        z = model_frame(frame, selection)
        atomic_tsv(output / f"reference/{cohort}_z.tsv", z)
        atomic_tsv(output / f"reference/{cohort}_primary_z.tsv", z.loc[z.age.between(50, 75)])
        if cohort == "LS":
            atomic_tsv(output / "reference/LS_oof_z.tsv", z)
        else:
            directory = output / ("model/fit" if cohort == "C3" else "validation")
            for scope in ("primary", "all"):
                assignments = pd.read_csv(directory / f"assignments_{cohort}_{scope}.tsv", sep="\t")
                n = len(assignments)
                np.savez_compressed(directory / f"posterior_{cohort}_{scope}.npz",
                                    posterior_joint=np.ones((n, 115, 2)) / 230,
                                    posterior_subtype=np.ones((n, 2)) / 2,
                                    posterior_stage=np.ones((n, 115)) / 115,
                                    subject_keys=assignments.subject_key.to_numpy(dtype=str))
    atomic_tsv(output / "inputs/feature_dictionary.tsv", raw)
    atomic_tsv(output / "reference/feature_dictionary.tsv", selected)
    atomic_json(output / "reference/feature_selection.json", selection)
    atomic_json(output / "reference/model.json", {"settings": cfg["reference"],
                "feature_ids": selected.feature_id.tolist(), "feature_selection": selection})
    atomic_json(output / "reference/manifest.json", {"files": {
        name: sha256_file(output / "reference" / name)
        for name in ("feature_selection.json", "feature_dictionary.tsv")}})
    atomic_tsv(output / "reference/diagnostics.tsv", pd.DataFrame({
        "feature_id": selected.feature_id, "split": "LS_oof", "mean_z": 0, "sd_z": 1,
        "coverage_95": .95, "correlation_age": 0, "correlation_sex": 0, "correlation_log_icv": 0}))
    sequence = np.arange(114).reshape(3, 38)
    samples = np.empty((2, 114, 4), dtype=int)
    for subtype in range(2):
        for draw in range(4):
            samples[subtype, :, draw] = np.roll(sequence, subtype + draw % 2, axis=1).ravel()
    np.savez_compressed(output / "model/fit/selected_model.npz", samples_sequence=samples,
                        samples_f=np.ones((2, 4)) / 2, samples_likelihood=np.ones((4, 1)),
                        subtype_order=np.arange(2), Z_vals=np.tile([1, 2, 3], (38, 1)), Z_max=np.ones(38) * 5,
                        feature_ids=selected.feature_id.to_numpy(dtype=str), labels=selected.label.to_numpy(dtype=str))
    shutil.copyfile(output / "model/fit/selected_model.npz", output / "model/fit/cv_display.npz")
    atlas = (np.arange(1728).reshape(12, 12, 12) % 21).astype(np.int16)
    for key in ("wmh_atlas", "gm_atlas"):
        nib.save(nib.Nifti1Image(atlas, np.eye(4)), cfg["assets"][key])
    write_asset_manifest(cfg["assets"], Path(cfg["assets"]["manifest"]), "synthetic 38-feature display fixture")
    report = run_report(cfg, output)
    assert report["figures"] == 17
    html = (output / "report/index.html").read_text()
    assert "model uses 38 features" in html and "normal z=0" in html
    descriptions = pd.read_csv(output / "report/tables/00_excluded_raw_features.tsv", sep="\t")
    assert len(descriptions) == 6
    assert descriptions.loc[descriptions.cohort == "LS", "n_nonzero"].eq(0).all()
    assert descriptions.loc[descriptions.cohort == "GB", "n_nonzero"].gt(0).all()
    spatial = pd.read_csv(output / "report/tables/05_spatial_wmh_subtype_01.tsv", sep="\t")
    assert spatial.feature_id.nunique() == 19 and excluded not in set(spatial.feature_id)
    bundle = tmp_path / "38-feature-display.tar.gz"
    create_bundle(cfg, output, bundle)
    shutil.rmtree(output)
    shutil.rmtree(Path(cfg["assets"]["manifest"]).parent)
    receipt = verify_bundle(bundle)
    assert receipt["status"] == "pass" and receipt["network_disabled"] is True
    assert receipt["tables_verified"] == 18


def test_display_trajectory_uses_official_midpoint_stage_convention():
    model = {"samples_sequence": np.arange(3).reshape(1, 3, 1), "Z_vals": np.ones((3, 1)),
             "Z_max": np.ones(3) * 3, "feature_ids": np.array(["a", "b", "c"])}
    table = _trajectory_table(model, {"max_posterior_draws": 1, "stage_points": [0, 1/3, 2/3, 1]})
    np.testing.assert_allclose(table[table.feature_id == "a"].mean_z, [.5, 4/3, 2, 8/3])


def test_verify_bundle_rejects_path_traversal_before_extraction(tmp_path):
    import io
    archive = tmp_path / "invalid.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("sustain-display/../../escape")
        member.size = 3
        tar.addfile(member, io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="Unsafe archive member"):
        verify_bundle(archive)
    assert not (tmp_path / "escape").exists()


def test_trace_uses_display_indices_when_inference_sample_count_differs(tmp_path):
    settings = dict(DEFAULTS, formats=["png"], preview_dpi=20, figure_width=4, figure_height=3)
    writer = ReportWriter(tmp_path, settings)
    model = {"samples_f": np.ones((1, 3)), "samples_likelihood": np.arange(3).reshape(3, 1),
             "sample_indices": np.array([10, 20, 30]), "posterior_sample_indices": np.array([10, 15, 20, 25, 30])}
    _plot_posterior_trace(writer, model)
    table = pd.read_csv(tmp_path / "tables/03b_posterior_trace.tsv", sep="\t")
    np.testing.assert_array_equal(table.draw_index, [10, 20, 30])
    model["samples_likelihood"] = np.arange(2)
    with pytest.raises(ValueError, match="indices differ"):
        _plot_posterior_trace(writer, model)


def test_report_rejects_changed_reference_age_but_accepts_style_only(report_case):
    from substain_sustain.report import _reference_settings
    cfg, output = report_case
    cfg["report"]["font_size"] = 9
    assert _reference_settings(cfg, output)["age_min"] == 50
    cfg["reference"]["age_min"] = 40
    with pytest.raises(ValueError, match="scientific configuration"):
        run_report(cfg, output)
    assert json.loads((output / "report/summary.json").read_text())["status"] == "incomplete"


def test_complete_probability_contract_requires_all_age_files_and_exact_ids(report_case):
    from substain_sustain.bundle import _validate_probability_files
    _, output = report_case
    assert len(_validate_probability_files(output)) == 4
    path = output / "validation/posterior_GB_all.npz"
    with np.load(path, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    payload["subject_keys"] = payload["subject_keys"][::-1]
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="subject_keys"):
        _validate_probability_files(output)
    path.unlink()
    with pytest.raises(FileNotFoundError, match="GB all"):
        _validate_probability_files(output)


def test_display_bundle_excludes_accidental_patient_images(report_case):
    from substain_sustain.bundle import _eligible_data
    _, output = report_case
    (output / "inputs/patient.nii.gz").write_bytes(b"not a real NIfTI; must never enter the package")
    (output / "inputs/patient.nii").write_bytes(b"image")
    (output / "inputs/arbitrary.gz").write_bytes(b"unknown compressed payload")
    (output / "inputs/LS_raw_records.jsonl.gz").write_bytes(b"fixture compressed provenance")
    names = {path.name for path in _eligible_data(output, True)}
    assert not {"patient.nii.gz", "patient.nii", "arbitrary.gz"} & names
    assert "LS_raw_records.jsonl.gz" in names
