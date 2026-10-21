"""输入契约测试：使用小型合成JSON，不访问患者影像或历史队列。"""
import copy
import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from substain_sustain.ingest import (
    REQUIRED_STAGES, WMH_FEATURES, export_inputs, guard_output, load_mapping,
    read_tsv, run_ingest, sha256_file,
)


# ---------- 合成上游分割结果 ----------
MAPPING = Path(__file__).resolve().parents[1] / "resources/mappings/muse_gm_macro20_v1.tsv"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_source(root, cohort, people=("0001",)):
    mapping, _ = load_mapping(MAPPING)
    source = root / cohort
    derivatives = source / "derivatives"
    config = source / "config"
    config.mkdir(parents=True)
    roster = pd.DataFrame([{"participant_id": person, "age": 60, "sex": "female"} for person in people])
    roster.to_csv(config / "participants.tsv", sep="\t", index=False)
    for person in people:
        subject = derivatives / f"sub-{person}"
        wmh_path = subject / "wmh/wmh_features.json"
        t1_path = subject / "t1/t1_features.json"
        write_json(wmh_path, {"participant_id": person, "raw_ml": {name: index / 10 for index, name in enumerate(WMH_FEATURES)},
                              "z_chung": {"SHOULD_NEVER_BE_READ": "invalid"}})
        native = {str(label): float(index + 1) for index, label in enumerate(mapping.native_label)}
        write_json(t1_path, {"participant_id": person, "gm119_ml": native,
                            "native145_ml": dict(native, **{"35": 100000.0, "61": 100000.0, "62": 100000.0}),
                            "macro20_atrophy_z": {"SHOULD_NEVER_BE_READ": "invalid"},
                            "dlicv_icv": {"available": True, "icv_label702_ml": 1500.0, "icv_label702_mm3": 1500000.0}})
        for stage in REQUIRED_STAGES:
            details = {}
            if stage == "wmh":
                details["feature_json"] = str(wmh_path)
            if stage == "t1":
                details["feature_json"] = str(t1_path)
            write_json(subject / "status" / f"{stage}.json",
                       {"participant_id": person, "stage": stage, "status": "pass", "details": details})
    return {"derivatives": str(derivatives), "participants": str(config / "participants.tsv")}


def config_for(root, people=("0001",)):
    return {"paths": {"mapping": str(MAPPING)},
            "cohorts": {cohort: make_source(root, cohort, people) for cohort in ("LS", "C3", "GB")}}


def mutate_json(path, edit):
    value = json.loads(path.read_text())
    edit(value)
    write_json(path, value)


# ---------- 导入、纯GM重聚合与人工QC旁注 ----------
def test_three_cohorts_use_raw_values_and_pure_gm_and_keep_manual_fail(tmp_path):
    cfg = config_for(tmp_path / "source")
    tables = Path(cfg["cohorts"]["LS"]["derivatives"]) / "tables"
    tables.mkdir()
    pd.DataFrame([{"participant_id": "0001", "review_state": "fail"}]).to_csv(tables / "qc_reviews_LS.tsv", sep="\t", index=False)
    manifest = run_ingest(cfg, tmp_path / "run")
    assert manifest["cohorts"]["LS"]["included"] == 1
    frame = read_tsv(tmp_path / "run/inputs/LS.tsv")
    assert frame.participant_id.tolist() == ["0001"]
    assert frame.subject_key.tolist() == ["LS:0001"]
    assert frame.manual_qc_state.tolist() == ["fail"]
    mapping, dictionary = load_mapping(MAPPING)
    t1 = json.loads((Path(cfg["cohorts"]["LS"]["derivatives"]) / "sub-0001/t1/t1_features.json").read_text())
    expected = sum(t1["gm119_ml"][str(label)] for label in mapping.loc[mapping.macro_id == "basal_forebrain", "native_label"])
    assert float(frame.gm_basal_forebrain_ml.iloc[0]) == expected
    assert len(dictionary) == 40
    assert float(frame.icv_ml.iloc[0]) == 1500
    assert manifest["old_normative_z_used"] is False
    assert run_ingest(cfg, tmp_path / "run")["input_hash"] == manifest["input_hash"]


def test_failed_stage_missing_icv_and_missing_gm_are_audited(tmp_path):
    cfg = config_for(tmp_path / "source", ("0001", "0002", "0003", "0004"))
    root = Path(cfg["cohorts"]["C3"]["derivatives"])
    mutate_json(root / "sub-0002/status/skullstrip.json", lambda value: value.update(status="fail"))
    mutate_json(root / "sub-0003/t1/t1_features.json", lambda value: value.update(dlicv_icv={"available": False}))
    mutate_json(root / "sub-0004/t1/t1_features.json", lambda value: value["gm119_ml"].pop(next(iter(value["gm119_ml"]))))
    manifest = run_ingest(cfg, tmp_path / "run")
    assert manifest["cohorts"]["C3"] == {"included": 1, "roster": 4, "excluded": 3}
    audit = read_tsv(tmp_path / "run/inputs/C3_audit.tsv").set_index("participant_id")
    assert audit.loc["0002", "reason"] == "stage_not_pass"
    assert audit.loc["0003", "reason"] == "missing_true_icv"
    assert audit.loc["0004", "reason"] == "missing_gm_labels"
    assert read_tsv(tmp_path / "run/inputs/C3.tsv").participant_id.tolist() == ["0001"]


def test_zero_native_gm_allowed_but_zero_macro_rejected(tmp_path):
    cfg = config_for(tmp_path / "source", ("0001", "0002"))
    mapping, _ = load_mapping(MAPPING)
    labels = mapping.loc[mapping.macro_index == 1, "native_label"].astype(str).tolist()
    root = Path(cfg["cohorts"]["GB"]["derivatives"])
    mutate_json(root / "sub-0001/t1/t1_features.json", lambda value: value["gm119_ml"].update({labels[0]: 0}))
    mutate_json(root / "sub-0002/t1/t1_features.json", lambda value: value["gm119_ml"].update({label: 0 for label in labels}))
    run_ingest(cfg, tmp_path / "run")
    assert read_tsv(tmp_path / "run/inputs/GB.tsv").participant_id.tolist() == ["0001"]


def test_optional_metadata_conflicts_and_configured_missing_are_explicit(tmp_path):
    cfg = config_for(tmp_path / "source")
    metadata = tmp_path / "metadata.tsv"
    cfg["cohorts"]["LS"]["metadata"] = str(metadata)
    with pytest.raises(FileNotFoundError):
        run_ingest(cfg, tmp_path / "missing")
    pd.DataFrame([{"participant_id": "0001", "age": "60.0", "sex": "F"}]).to_csv(metadata, sep="\t", index=False)
    run_ingest(cfg, tmp_path / "valid")
    pd.DataFrame([{"participant_id": "0001", "age": "61", "sex": "F"}]).to_csv(metadata, sep="\t", index=False)
    with pytest.raises(ValueError, match="冲突"):
        run_ingest(cfg, tmp_path / "conflict")


# ---------- 可移动包重导入、篡改检测和路径安全 ----------
def test_bundle_roundtrip_does_not_need_source_paths(tmp_path):
    cfg = config_for(tmp_path / "source")
    exported = export_inputs(cfg, "LS", tmp_path / "LS_compact")
    assert exported["included_count"] == 1
    original = copy.deepcopy(cfg)
    cfg["cohorts"]["LS"] = {"input_bundle": str(tmp_path / "LS_compact")}
    run_ingest(cfg, tmp_path / "run")
    assert read_tsv(tmp_path / "run/inputs/LS.tsv").subject_key.tolist() == ["LS:0001"]
    with gzip.open(tmp_path / "LS_compact/raw_records.jsonl.gz", "rt") as handle:
        raw = json.loads(handle.readline())
    assert len(raw["statuses"]) == 7
    assert "gm119_ml" in raw["features"]["t1"]
    with pytest.raises(FileExistsError):
        export_inputs(original, "LS", tmp_path / "LS_compact")


def test_bundle_tampering_fails_before_import(tmp_path):
    cfg = config_for(tmp_path / "source")
    export_inputs(cfg, "LS", tmp_path / "bundle")
    table = tmp_path / "bundle/cohort.tsv"
    table.write_text(table.read_text().replace("1500.0", "1000.0"))
    cfg["cohorts"]["LS"] = {"input_bundle": str(tmp_path / "bundle")}
    with pytest.raises(ValueError, match="哈希"):
        run_ingest(cfg, tmp_path / "run")
    manifest = tmp_path / "bundle/manifest.json"
    mutate_json(manifest, lambda value: value["files"].update({"cohort.tsv": sha256_file(table)}))
    with pytest.raises(ValueError, match="数值不一致"):
        run_ingest(cfg, tmp_path / "run2")


def test_identity_and_output_protection(tmp_path):
    cfg = config_for(tmp_path / "source")
    with pytest.raises(ValueError, match="重叠"):
        guard_output(cfg, Path(cfg["cohorts"]["LS"]["derivatives"]) / "new_run")
    run_ingest(cfg, tmp_path / "run")
    path = Path(cfg["cohorts"]["GB"]["derivatives"]) / "sub-0001/t1/t1_features.json"
    mutate_json(path, lambda value: value.update(participant_id="OTHER"))
    with pytest.raises(ValueError, match="不同输入"):
        run_ingest(cfg, tmp_path / "run")


def test_relative_feature_pointer_uses_explicit_source_root(tmp_path):
    cfg = config_for(tmp_path / "source")
    spec = cfg["cohorts"]["LS"]
    spec["source_root"] = str(tmp_path / "source/LS")
    path = Path(spec["derivatives"]) / "sub-0001/status/t1.json"
    mutate_json(path, lambda value: value["details"].update(feature_json="derivatives/sub-0001/t1/t1_features.json"))
    run_ingest(cfg, tmp_path / "run")
    assert len(read_tsv(tmp_path / "run/inputs/LS.tsv")) == 1


def test_relocated_pointers_use_new_subject_tree_and_preserve_original_json(tmp_path):
    cfg = config_for(tmp_path / "source")
    spec = cfg["cohorts"]["C3"]
    root = Path(spec["derivatives"])
    originals = {}
    for stage in ("wmh", "t1"):
        path = root / f"sub-0001/status/{stage}.json"
        mutate_json(path, lambda value, stage=stage: value["details"].update(
            feature_json=f"/old/workstation/derivatives/sub-0001/{stage}/{stage}_features.json"))
        originals[path] = path.read_bytes()
    before = run_ingest(cfg, tmp_path / "before")
    assert before["cohorts"]["C3"]["included"] == 0
    spec["relocate_feature_pointers"] = True
    after = run_ingest(cfg, tmp_path / "after")
    assert after["cohorts"]["C3"]["included"] == 1
    assert all(path.read_bytes() == content for path, content in originals.items())
    with gzip.open(tmp_path / "after/inputs/C3_raw_records.jsonl.gz", "rt") as stream:
        record = json.loads(stream.readline())
    resolution = record["feature_pointer_resolution"]["wmh"]
    assert resolution["recorded"].startswith("/old/workstation/")
    assert resolution["resolved"] == str(root / "sub-0001/wmh/wmh_features.json")
    assert resolution["relocated"] is True


@pytest.mark.parametrize("pointer", [
    "/old/sub-OTHER/wmh/wmh_features.json", "/old/sub-0001/../other.json",
    "/old/sub-0001/sub-0001/wmh.json", "/old/sub-0001/wmh.nii.gz",
])
def test_unsafe_relocation_is_excluded_not_guessed(tmp_path, pointer):
    cfg = config_for(tmp_path / "source")
    spec = cfg["cohorts"]["C3"]
    spec["relocate_feature_pointers"] = True
    mutate_json(Path(spec["derivatives"]) / "sub-0001/status/wmh.json",
                lambda value: value["details"].update(feature_json=pointer))
    run_ingest(cfg, tmp_path / "run")
    audit = read_tsv(tmp_path / "run/inputs/C3_audit.tsv")
    assert audit.reason.tolist() == ["unsafe_relocated_pointer"]


def test_relocated_json_subject_identity_is_still_checked(tmp_path):
    cfg = config_for(tmp_path / "source")
    spec = cfg["cohorts"]["C3"]
    spec["relocate_feature_pointers"] = True
    mutate_json(Path(spec["derivatives"]) / "sub-0001/wmh/wmh_features.json",
                lambda value: value.update(participant_id="OTHER"))
    run_ingest(cfg, tmp_path / "run")
    assert read_tsv(tmp_path / "run/inputs/C3_audit.tsv").reason.tolist() == ["feature_identity_mismatch"]


def test_relocated_source_never_falls_back_to_existing_old_copy(tmp_path):
    """两份副本都存在时，显式迁移模式只能使用新的配置根目录。"""
    import shutil
    cfg = config_for(tmp_path / "source")
    spec = cfg["cohorts"]["C3"]
    root = Path(spec["derivatives"])
    old = tmp_path / "old/sub-0001/wmh/wmh_features.json"
    old.parent.mkdir(parents=True)
    shutil.copyfile(root / "sub-0001/wmh/wmh_features.json", old)
    mutate_json(old, lambda value: value.update(participant_id="WRONG_OLD_COPY"))
    mutate_json(root / "sub-0001/status/wmh.json", lambda value: value["details"].update(feature_json=str(old)))
    spec["relocate_feature_pointers"] = True
    result = run_ingest(cfg, tmp_path / "run")
    assert result["cohorts"]["C3"]["included"] == 1


def test_relocated_pointer_rejects_symlink_escape(tmp_path):
    cfg = config_for(tmp_path / "source")
    spec = cfg["cohorts"]["C3"]
    root = Path(spec["derivatives"])
    external = tmp_path / "outside.json"
    external.write_text('{}')
    pointer = root / "sub-0001/escape.json"
    pointer.symlink_to(external)
    mutate_json(root / "sub-0001/status/wmh.json", lambda value: value["details"].update(feature_json=str(pointer)))
    spec["relocate_feature_pointers"] = True
    run_ingest(cfg, tmp_path / "run")
    assert read_tsv(tmp_path / "run/inputs/C3_audit.tsv").reason.tolist() == ["unsafe_relocated_pointer"]


def test_disabled_required_stage_is_rejected(tmp_path):
    cfg = config_for(tmp_path / "source")
    cfg["ingest"] = {"required_stages": ["wmh", "t1", "qc"]}
    with pytest.raises(ValueError, match="完整七阶段"):
        run_ingest(cfg, tmp_path / "run")
