"""验证残差诊断，不改变正式LS参数或以疾病队列估计参考尺度。"""
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from substain_sustain.ingest import atomic_tsv, sha256_file
from substain_sustain.reference import _settings, fit_reference
from test_reference import make_run, synthetic

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("diagnose_reference", ROOT / "scripts/diagnose_reference.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)
FEATURE = "wmh_basal_ganglia_layer4_ml"
OPTIONS = {"focus_features": [FEATURE], "verify_ls_raw_snapshots": True,
           "comparison_rtol": 1e-12, "comparison_atol": 1e-12}


def test_diagnostic_scale_matches_production():
    frame, dictionary = synthetic()
    settings = _settings({})
    expected = fit_reference(frame, dictionary, settings)
    rows = diagnostic.residual_summary(frame, dictionary, settings, "LS_full")
    np.testing.assert_array_equal([row["residual_sd"] for row in rows], expected["residual_sd"])
    assert all(row["status"] == "pass" for row in rows)


def test_zero_and_nonzero_constant_are_distinguished():
    frame, dictionary = synthetic()
    frame[FEATURE] = 0.0
    frame[dictionary.feature_id.iloc[0]] = 1.0
    rows = diagnostic.residual_summary(frame, dictionary, _settings({}), "LS_full")
    by_id = {row["feature_id"]: row for row in rows}
    assert by_id[FEATURE]["status"] == "all_zero"
    assert by_id[FEATURE]["residual_sd"] == 0
    assert by_id[dictionary.feature_id.iloc[0]]["status"] == "constant_nonzero"


def prepare_inputs(tmp_path, rare_nonzero=False):
    cfg, output, ls, dictionary = make_run(tmp_path)
    cfg["output_dir"] = str(output)
    ls[FEATURE] = 0.0
    if rare_nonzero:
        ls.loc[0, FEATURE] = 0.01
    source = output / "inputs"
    atomic_tsv(source / "LS.tsv", ls)
    with gzip.open(source / "LS_raw_records.jsonl.gz", "wt", encoding="utf-8") as stream:
        for _, row in ls.iterrows():
            stream.write(json.dumps({"included": True, "subject_key": row.subject_key,
                                     "features": {"wmh": {"raw_ml": {FEATURE: row[FEATURE]}}}}) + "\n")
    path = source / "manifest.json"
    manifest = json.loads(path.read_text())
    for name in ("LS.tsv", "LS_raw_records.jsonl.gz"):
        manifest["files"][name] = sha256_file(source / name)
    path.write_text(json.dumps(manifest))
    return cfg, output


def test_full_zero_produces_aggregate_evidence_and_is_read_only(tmp_path):
    cfg, output = prepare_inputs(tmp_path)
    before = {str(p): sha256_file(p) for p in output.rglob("*") if p.is_file()}
    result = diagnostic.diagnose(cfg, OPTIONS)
    assert result["flagged_features"] == [FEATURE]
    assert len(result["residuals"]) == 6
    assert all(row["status"] == "all_zero" for row in result["residuals"])
    assert result["ls_raw_snapshot_check"]["features"][FEATURE] == {
        "compared": 120, "mismatch": 0, "missing_or_invalid": 0}
    assert before == {str(p): sha256_file(p) for p in output.rglob("*") if p.is_file()}
    encoded = json.dumps(result)
    assert "LS:00000" not in encoded and "subject_key" not in encoded


def test_fold_only_failure_is_distinguished_from_full_ls(tmp_path):
    cfg, _ = prepare_inputs(tmp_path, rare_nonzero=True)
    result = diagnostic.diagnose(cfg, OPTIONS)
    full = next(row for row in result["residuals"] if row["split"] == "LS_full")
    failed = [row for row in result["residuals"] if row["status"] != "pass"]
    assert full["status"] == "pass" and full["n_positive"] == 1
    assert len(failed) == 1 and failed[0]["status"] == "all_zero"
    assert failed[0]["split"].startswith("LS_train_fold_")


def test_input_hash_change_stops_diagnosis(tmp_path):
    cfg, output = prepare_inputs(tmp_path)
    path = output / "inputs/LS.tsv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="哈希改变"):
        diagnostic.diagnose(cfg, OPTIONS)


def test_raw_snapshot_mismatch_is_counted(tmp_path):
    frame, _ = synthetic()
    path = tmp_path / "raw.jsonl.gz"
    subset = frame.iloc[:1].copy()
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps({"included": True, "subject_key": subset.subject_key.iloc[0],
                                 "features": {"wmh": {"raw_ml": {FEATURE: 0}}}}) + "\n")
    result = diagnostic.compare_snapshots(path, subset, [FEATURE], OPTIONS)
    assert result["features"][FEATURE]["mismatch"] == 1


def test_input_audit_finds_zero_c3_without_exposing_original_id(tmp_path):
    import pandas as pd
    from substain_sustain.ingest import atomic_tsv
    cfg, output = prepare_inputs(tmp_path)
    atomic_tsv(output / "inputs/C3_audit.tsv", pd.DataFrame([
        {"participant_id": "P001", "included": False, "reason": "file_missing",
         "detail": "/source/sub-P001/status/t1.json"}]))
    result = diagnostic.input_audit(cfg)
    assert result["C3"]["included_n"] == 0
    assert result["C3"]["reasons"] == {"file_missing": 1}
    assert "P001" not in json.dumps(result)
