"""只读导入分割结果，以及可独立重导入的轻量数据包。"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------- 稳定产物与既有分割特征契约 ----------
WMH_REGIONS = ("basal_ganglia", "frontal", "occipital", "temporal", "parietal")
WMH_FEATURES = tuple(f"wmh_{region}_layer{layer}_ml" for region in WMH_REGIONS for layer in range(1, 5))
REQUIRED_STAGES = ("skullstrip", "registration", "lesion", "wmh_seg", "wmh", "t1", "qc")
META_COLUMNS = ["participant_id", "subject_key", "cohort", "age", "sex", "icv_ml", "manual_qc_state"]
AUDIT_COLUMNS = ["participant_id", "subject_key", "cohort", "included", "reason", "detail", "manual_qc_state"]
SCHEMA_VERSION = "substain-sustain-inputs-v1"


class InputError(ValueError):
    """携带稳定审计原因的个体输入错误。"""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason


# ---------- 原子写入、哈希与输入目录保护 ----------
def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, canonical_json(value))


def atomic_tsv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, frame.to_csv(sep="\t", index=False, lineterminator="\n").encode("utf-8"))


def read_tsv(path: Path) -> pd.DataFrame:
    """始终把ID作为字符串读取，保留前导零。"""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8-sig")


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def guard_output(cfg: dict, output: Path) -> None:
    """输出不得覆盖源目录，也不得成为其祖先或写入其内部。"""
    output = Path(output).resolve()
    for cohort in cfg.get("cohorts", {}).values():
        for key in ("derivatives", "participants", "metadata", "manual_qc", "input_bundle"):
            if not cohort.get(key):
                continue
            source = Path(cohort[key]).resolve()
            is_directory = key in {"derivatives", "input_bundle"} or source.is_dir()
            if output == source or _within(source, output) or (is_directory and _within(output, source)):
                raise ValueError(f"输出与只读输入重叠: {output} / {source}")
    mapping = Path(cfg["paths"]["mapping"]).resolve()
    if output == mapping or _within(mapping, output):
        raise ValueError(f"输出目录包含输入mapping: {output}")


def _stamp(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


def _read_stable(path: Path, evidence: dict) -> bytes:
    path = path.resolve()
    before = _stamp(path)
    data = path.read_bytes()
    after = _stamp(path)
    if before != after:
        raise InputError("changed_during_read", str(path))
    evidence[str(path)] = {"stamp": list(after), "sha256": hashlib.sha256(data).hexdigest()}
    return data


def _read_json(path: Path, evidence: dict) -> dict:
    def invalid_constant(value: str) -> None:
        raise ValueError(f"JSON非有限值: {value}")
    value = json.loads(_read_stable(path, evidence).decode("utf-8-sig"), parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise InputError("invalid_json_schema", str(path))
    return value


def _verify_stable(evidence: dict) -> None:
    for path, record in evidence.items():
        if list(_stamp(Path(path))) != record["stamp"]:
            raise InputError("changed_during_export", path)


# ---------- 纯GM20映射与变量字典 ----------
def load_mapping(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = read_tsv(path)
    required = {"macro_index", "macro_id", "native_label"}
    if not required.issubset(mapping.columns):
        raise ValueError("mapping缺少macro_index/macro_id/native_label")
    if "in_official_gm119" in mapping:
        if not mapping["in_official_gm119"].str.lower().isin(["true", "1"]).all():
            raise ValueError("正式mapping必须只含GM119；请使用纯GM mapping而非旧macro20")
    for field in ("macro_index", "native_label"):
        numeric = pd.to_numeric(mapping[field], errors="raise")
        if not np.isfinite(numeric).all() or (numeric != np.floor(numeric)).any():
            raise ValueError(f"mapping的{field}必须为有限整数")
        mapping[field] = numeric.astype(int)
    if len(mapping) != 119 or mapping.native_label.nunique() != 119:
        raise ValueError("纯GM mapping必须含119个唯一原生标签")
    if set(mapping.native_label) & {35, 61, 62}:
        raise ValueError("纯GM mapping含旧macro20的非GM标签35/61/62")
    if set(mapping.macro_index) != set(range(1, 21)):
        raise ValueError("纯GM mapping必须覆盖全部1..20组")
    grouped = mapping.groupby("macro_index", sort=True)
    if any(group.macro_id.nunique() != 1 for _, group in grouped):
        raise ValueError("同一macro_index对应多个macro_id")
    if mapping.groupby("macro_id").macro_index.nunique().max() != 1:
        raise ValueError("同一macro_id对应多个macro_index")
    dictionary = []
    for index, feature in enumerate(WMH_FEATURES, 1):
        region = WMH_REGIONS[(index - 1) // 4].replace("_", " ").title()
        dictionary.append({"feature_id": feature, "label": f"WMH {region} layer {(index - 1) % 4 + 1}",
                           "modality": "WMH", "atlas_index": index})
    for index, group in grouped:
        row = group.iloc[0]
        label = row.get("macro_name_en", row.get("label", row["macro_id"].replace("_", " ").title()))
        dictionary.append({"feature_id": f"gm_{row['macro_id']}_ml", "label": str(label),
                           "modality": "GM", "atlas_index": int(index)})
    return mapping.sort_values(["macro_index", "native_label"]), pd.DataFrame(dictionary)


def validate_feature_dictionary(path: Path) -> pd.DataFrame:
    dictionary = read_tsv(path)
    if not {"feature_id", "label", "modality", "atlas_index"}.issubset(dictionary.columns):
        raise ValueError("变量字典字段不完整")
    if len(dictionary) != 40 or dictionary.feature_id.nunique() != 40:
        raise ValueError("变量字典必须有40个唯一特征")
    for modality in ("WMH", "GM"):
        block = dictionary.loc[dictionary.modality == modality]
        if len(block) != 20 or set(pd.to_numeric(block.atlas_index)) != set(range(1, 21)):
            raise ValueError(f"变量字典{modality}必须有1..20 atlas标签")
    return dictionary


def _number(value: Any, name: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise InputError("invalid_numeric_value", name)
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise InputError("invalid_numeric_value", name) from exc
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        raise InputError("invalid_numeric_value", name)
    return number


def _subject_key(cohort: str, participant_id: str) -> str:
    if not participant_id or any(part in participant_id for part in ("/", "\\", "\0", "\t", "\n", ":")) or participant_id in {".", ".."}:
        raise InputError("invalid_participant_id", repr(participant_id))
    return f"{cohort}:{participant_id}"


def _sex(value: Any, cfg: dict) -> str:
    default = {"male": "male", "female": "female", "m": "male", "f": "female", "男": "male", "女": "female"}
    labels = {str(key).strip().lower(): str(mapped) for key, mapped in cfg.get("ingest", {}).get("sex_mapping", default).items()}
    result = labels.get(str(value).strip().lower())
    if result not in {"male", "female"}:
        raise InputError("sex_missing_or_invalid", str(value))
    return result


def _features(wmh: dict, t1: dict, mapping: pd.DataFrame, cfg: dict) -> tuple[dict, float]:
    values = wmh.get("raw_ml")
    if not isinstance(values, dict) or set(values) != set(WMH_FEATURES):
        raise InputError("invalid_wmh20_columns", "raw_ml必须完整包含固定WMH20")
    features = {feature: _number(values[feature], feature) for feature in WMH_FEATURES}
    native = t1.get("gm119_ml")
    if not isinstance(native, dict):
        native = t1.get("native145_ml")
    if not isinstance(native, dict):
        raise InputError("missing_gm119", "缺少gm119_ml及native145_ml，不能使用旧macro20替代")
    missing = set(mapping.native_label.astype(str)) - set(native)
    if missing:
        raise InputError("missing_gm_labels", ",".join(sorted(missing, key=int)))
    gm = {str(label): _number(native[str(label)], f"GM{label}") for label in mapping.native_label}
    for _, group in mapping.groupby("macro_index", sort=True):
        name = f"gm_{group.iloc[0].macro_id}_ml"
        features[name] = _number(sum(gm[str(label)] for label in group.native_label), name, positive=True)
    icv = t1.get("dlicv_icv", {})
    if not isinstance(icv, dict) or icv.get("available") is not True:
        raise InputError("missing_true_icv", "需要DLICV ICV702，不能用非脑室组织体积代替")
    denominator = _number(icv.get("icv_label702_ml"), "DLICV ICV702 mL", positive=True)
    if "icv_label702_mm3" in icv:
        mm3 = _number(icv["icv_label702_mm3"], "DLICV ICV702 mm3", positive=True)
        tolerance = float(cfg.get("ingest", {}).get("icv_consistency_rtol", 1e-6))
        if not np.isclose(mm3 / 1000.0, denominator, rtol=tolerance, atol=0):
            raise InputError("icv_units_inconsistent", "ICV mL与mm3不一致")
    return features, denominator


# ---------- 源名册、状态指针与人工QC旁注 ----------
def _table_stable(path: Path, evidence: dict) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_read_stable(path, evidence)), sep="\t", dtype=str, keep_default_na=False, encoding="utf-8-sig")


def _unique_roster(frame: pd.DataFrame, name: str) -> None:
    if "participant_id" not in frame or frame.participant_id.duplicated().any() or frame.participant_id.eq("").any():
        raise ValueError(f"{name}必须含非空且唯一participant_id")


def _load_roster(spec: dict, evidence: dict, cfg: dict) -> pd.DataFrame:
    roster = _table_stable(Path(spec["participants"]), evidence)
    _unique_roster(roster, "participants")
    if spec.get("metadata"):
        metadata = _table_stable(Path(spec["metadata"]), evidence)
        _unique_roster(metadata, "metadata")
        metadata = metadata.set_index("participant_id")
        for field in ("age", "sex"):
            if field not in roster:
                roster[field] = ""
            if field in metadata:
                for index, row in roster.iterrows():
                    if row.participant_id not in metadata.index:
                        continue
                    existing, incoming = str(row[field]).strip(), str(metadata.at[row.participant_id, field]).strip()
                    if existing and incoming:
                        first = _number(existing, "age", positive=True) if field == "age" else _sex(existing, cfg)
                        second = _number(incoming, "age", positive=True) if field == "age" else _sex(incoming, cfg)
                        if first != second:
                            raise ValueError(f"metadata与participants冲突: {row.participant_id}/{field}")
                    if not existing:
                        roster.at[index, field] = incoming
    return roster


def _manual_states(spec: dict, cohort: str, evidence: dict) -> dict:
    if spec.get("manual_qc"):
        candidates = [Path(spec["manual_qc"])]
    else:
        tables = Path(spec["derivatives"]) / "tables"
        suffix = "_LS" if cohort == "LS" else ""
        candidates = [tables / f"qc_reviews{suffix}.tsv", tables / "qc_reviews.tsv", tables / f"subject_qc{suffix}.tsv"]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return {}
    frame = _table_stable(path, evidence)
    _unique_roster(frame, "manual_qc")
    field = next((name for name in ("review_state", "manual_qc_state") if name in frame), None)
    # 原始人工决定的理由、审核者和旧hash一并保留；这里不把它当入模门槛。
    return {row["participant_id"]: row for row in frame.to_dict("records")} if field else {}


def _pointer(raw: Any, spec: dict) -> Path:
    if not isinstance(raw, str) or not raw:
        raise InputError("missing_feature_pointer", "status.details.feature_json")
    for old, new in spec.get("path_prefix_map", {}).items():
        prefix = str(old).rstrip("/")
        if raw == prefix or raw.startswith(prefix + "/"):
            raw = str(new).rstrip("/") + raw[len(prefix):]
            break
    path = Path(raw)
    if path.is_absolute():
        return path
    project = Path(spec.get("source_root", Path(spec["participants"]).parent.parent))
    return project / path


def _collect_source(cfg: dict, cohort: str, mapping: pd.DataFrame, dictionary: pd.DataFrame) -> dict:
    spec = cfg["cohorts"][cohort]
    common_evidence: dict = {}
    roster = _load_roster(spec, common_evidence, cfg)
    manual = _manual_states(spec, cohort, common_evidence)
    rows, audits, snapshots, all_evidence = [], [], [], dict(common_evidence)
    accepted_evidence = dict(common_evidence)
    required = tuple(cfg.get("ingest", {}).get("required_stages", REQUIRED_STAGES))
    if not set(REQUIRED_STAGES).issubset(required):
        raise ValueError("自动入模必须检查完整七阶段，不允许关闭上游处理状态检查")
    for _, row in roster.iterrows():
        person = str(row.participant_id)
        subject_key = f"{cohort}:{person}"
        manual_record = manual.get(person, {})
        review = manual_record.get("review_state", manual_record.get("manual_qc_state", "unreviewed"))
        audit = dict(participant_id=person, subject_key=subject_key, cohort=cohort, included=False,
                     reason="", detail="", manual_qc_state=review)
        snapshot: dict = dict(participant_id=person, subject_key=subject_key, cohort=cohort, statuses={}, features={},
                              participant_metadata=row.to_dict(), manual_qc=manual_record)
        evidence: dict = {}
        try:
            subject_key = _subject_key(cohort, person)
            age = _number(row.get("age"), "age", positive=True)
            minimum = float(cfg.get("ingest", {}).get("age_valid_min", 0))
            maximum = float(cfg.get("ingest", {}).get("age_valid_max", 120))
            if not minimum < age <= maximum:
                raise InputError("age_missing_or_invalid", str(age))
            sex = _sex(row.get("sex"), cfg)
            directory = Path(spec["derivatives"]) / f"sub-{person}"
            for stage in required:
                state = _read_json(directory / "status" / f"{stage}.json", evidence)
                snapshot["statuses"][stage] = state
                if state.get("participant_id") != person or state.get("stage") != stage:
                    raise InputError("status_identity_mismatch", f"{person}/{stage}")
                if state.get("status") != "pass":
                    raise InputError("stage_not_pass", f"{stage}: {state.get('status')}")
                if not isinstance(state.get("details"), dict):
                    raise InputError("invalid_status_details", stage)
            for stage in ("wmh", "t1"):
                feature = _read_json(_pointer(snapshot["statuses"][stage]["details"].get("feature_json"), spec), evidence)
                snapshot["features"][stage] = feature
                if feature.get("participant_id") != person:
                    raise InputError("feature_identity_mismatch", f"{person}/{stage}")
            features, icv = _features(snapshot["features"]["wmh"], snapshot["features"]["t1"], mapping, cfg)
            _verify_stable(evidence)
            rows.append(dict(participant_id=person, subject_key=subject_key, cohort=cohort, age=age,
                             sex=sex, icv_ml=icv, manual_qc_state=review, **features))
            audit.update(included=True, reason="included_auto_qc_pass")
            accepted_evidence.update(evidence)
        except InputError as exc:
            audit.update(reason=exc.reason, detail=str(exc))
        except PermissionError as exc:
            audit.update(reason="permission_denied", detail=str(exc))
        except FileNotFoundError as exc:
            audit.update(reason="file_missing", detail=str(exc))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            audit.update(reason="read_or_format_error", detail=str(exc))
        snapshot["included"] = audit["included"]
        snapshot["source_evidence"] = evidence
        snapshots.append(snapshot)
        audits.append(audit)
        all_evidence.update(evidence)
    _verify_stable(accepted_evidence)
    frame = pd.DataFrame(rows, columns=META_COLUMNS + dictionary.feature_id.tolist())
    return {"frame": frame, "audit": pd.DataFrame(audits, columns=AUDIT_COLUMNS), "snapshots": snapshots,
            "source_evidence": all_evidence, "roster": roster}


# ---------- 压缩数据包：导出与完整重导入 ----------
def _gzip_jsonl(records: list[dict]) -> bytes:
    memory = io.BytesIO()
    with gzip.GzipFile(fileobj=memory, mode="wb", filename="", mtime=0) as stream:
        for record in records:
            stream.write((json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
    return memory.getvalue()


def _safe_bundle_member(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if Path(name).is_absolute() or not _within(path, root.resolve()):
        raise ValueError(f"数据包路径越界: {name}")
    return path


def _collect_bundle(cfg: dict, cohort: str, mapping: pd.DataFrame, dictionary: pd.DataFrame) -> dict:
    root = Path(cfg["cohorts"][cohort]["input_bundle"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("cohort") != cohort:
        raise ValueError("数据包版本或cohort身份不匹配")
    if manifest.get("mapping_sha256") != sha256_file(Path(cfg["paths"]["mapping"])):
        raise ValueError("数据包mapping与当前纯GM mapping不一致")
    required = {"cohort.tsv", "audit.tsv", "roster.tsv", "raw_records.jsonl.gz", "feature_dictionary.tsv"}
    if not required.issubset(manifest.get("files", {})):
        raise ValueError("轻量包缺少必要数据或原始JSON证据")
    for name, expected in manifest["files"].items():
        if sha256_file(_safe_bundle_member(root, name)) != expected:
            raise ValueError(f"数据包哈希不匹配: {name}")
    supplied = validate_feature_dictionary(root / "feature_dictionary.tsv")
    if supplied.feature_id.tolist() != dictionary.feature_id.tolist():
        raise ValueError("数据包特征顺序不一致")
    frame = read_tsv(root / "cohort.tsv")
    _unique_roster(frame, "bundle")
    expected_columns = META_COLUMNS + dictionary.feature_id.tolist()
    if frame.columns.tolist() != expected_columns or not frame.cohort.eq(cohort).all():
        raise ValueError("数据包表头或cohort不一致")
    if frame.subject_key.tolist() != [_subject_key(cohort, person) for person in frame.participant_id]:
        raise ValueError("数据包subject_key不一致")
    with gzip.open(root / "raw_records.jsonl.gz", "rt", encoding="utf-8") as stream:
        snapshots = [json.loads(line) for line in stream if line.strip()]
    included_records = [record for record in snapshots if record.get("included") is True]
    originals = {record["participant_id"]: record for record in included_records}
    if len(included_records) != len(frame) or len(originals) != len(frame) or set(originals) != set(frame.participant_id):
        raise ValueError("数据包表格与原始JSON身份不一致")
    for index, row in frame.iterrows():
        original = originals[row.participant_id]
        if original.get("cohort") != cohort or original.get("subject_key") != row.subject_key:
            raise ValueError("数据包JSON cohort身份不一致")
        required_stages = cfg.get("ingest", {}).get("required_stages", REQUIRED_STAGES)
        if not set(REQUIRED_STAGES).issubset(required_stages):
            raise ValueError("自动入模必须检查完整七阶段")
        for stage in required_stages:
            state = original["statuses"].get(stage, {})
            if state.get("participant_id") != row.participant_id or state.get("stage") != stage or state.get("status") != "pass":
                raise ValueError("数据包原始自动QC状态无效")
        features, icv = _features(original["features"]["wmh"], original["features"]["t1"], mapping, cfg)
        for feature, expected in dict(features, icv_ml=icv).items():
            actual = _number(row[feature], feature, positive=feature.startswith("gm_") or feature == "icv_ml")
            if not np.isclose(actual, expected, rtol=1e-12, atol=1e-12):
                raise ValueError(f"数据包表格与JSON数值不一致: {row.participant_id}/{feature}")
            frame.at[index, feature] = actual
        age, sex = _number(row.age, "age", positive=True), _sex(row.sex, cfg)
        original_metadata = original.get("participant_metadata", {})
        if age != _number(original_metadata.get("age"), "source age", positive=True) or sex != _sex(original_metadata.get("sex"), cfg):
            raise ValueError("数据包年龄/性别与原始名册不一致")
        if not float(cfg.get("ingest", {}).get("age_valid_min", 0)) < age <= float(cfg.get("ingest", {}).get("age_valid_max", 120)):
            raise ValueError("数据包年龄超出配置的有效范围")
        frame.at[index, "age"] = age
        frame.at[index, "sex"] = sex
    return {"frame": frame, "audit": read_tsv(root / "audit.tsv"), "snapshots": snapshots,
            "source_evidence": manifest.get("source_evidence", {}), "roster": read_tsv(root / "roster.tsv")}


def _collect(cfg: dict, cohort: str, mapping: pd.DataFrame, dictionary: pd.DataFrame) -> dict:
    if cohort not in cfg.get("cohorts", {}):
        raise ValueError(f"未配置队列: {cohort}")
    return (_collect_bundle if cfg["cohorts"][cohort].get("input_bundle") else _collect_source)(cfg, cohort, mapping, dictionary)


def _manifest_files(directory: Path) -> dict:
    return {path.name: sha256_file(path) for path in sorted(directory.iterdir()) if path.is_file() and path.name != "manifest.json"}


def export_inputs(cfg: dict, cohort: str, destination: Path) -> dict:
    """只读扫描源分割结果，原子创建一个自包含的轻量包。"""
    destination = Path(destination)
    guard_output(cfg, destination)
    if destination.exists():
        raise FileExistsError(f"导出目录已存在，避免覆盖请使用新目录: {destination}")
    mapping, dictionary = load_mapping(Path(cfg["paths"]["mapping"]))
    data = _collect(cfg, cohort, mapping, dictionary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + destination.name + ".", dir=destination.parent))
    try:
        atomic_tsv(temporary / "cohort.tsv", data["frame"])
        atomic_tsv(temporary / "audit.tsv", data["audit"])
        atomic_tsv(temporary / "roster.tsv", data["roster"])
        atomic_tsv(temporary / "feature_dictionary.tsv", dictionary)
        atomic_bytes(temporary / "raw_records.jsonl.gz", _gzip_jsonl(data["snapshots"]))
        manifest = {"schema_version": SCHEMA_VERSION, "cohort": cohort, "included_count": len(data["frame"]),
                    "roster_count": len(data["roster"]), "mapping_sha256": sha256_file(Path(cfg["paths"]["mapping"])),
                    "files": _manifest_files(temporary), "source_evidence": data["source_evidence"],
                    "manual_qc_gates_inclusion": False, "old_normative_z_used": False}
        manifest["input_hash"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
        atomic_json(temporary / "manifest.json", manifest)
        os.rename(temporary, destination)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


# ---------- 三队列统一导入 ----------
def run_ingest(cfg: dict, output: Path) -> dict:
    """建立不可静默换输入的本次分析矩阵，并保存每个排除原因。"""
    output = Path(output)
    guard_output(cfg, output)
    mapping_path = Path(cfg["paths"]["mapping"])
    mapping, dictionary = load_mapping(mapping_path)
    cohorts = list(cfg.get("cohorts", {}))
    if set(cohorts) != {"LS", "C3", "GB"}:
        raise ValueError("正式项目需恰好配置LS、C3、GB三个队列")
    collected = {cohort: _collect(cfg, cohort, mapping, dictionary) for cohort in cohorts}
    output.mkdir(parents=True, exist_ok=True)
    target = output / "inputs"
    temporary = Path(tempfile.mkdtemp(prefix=".inputs.", dir=output))
    try:
        atomic_tsv(temporary / "feature_dictionary.tsv", dictionary)
        counts = {}
        for cohort, data in collected.items():
            atomic_tsv(temporary / f"{cohort}.tsv", data["frame"])
            atomic_tsv(temporary / f"{cohort}_audit.tsv", data["audit"])
            atomic_tsv(temporary / f"{cohort}_roster.tsv", data["roster"])
            atomic_bytes(temporary / f"{cohort}_raw_records.jsonl.gz", _gzip_jsonl(data["snapshots"]))
            counts[cohort] = {"included": len(data["frame"]), "roster": len(data["roster"]),
                              "excluded": len(data["roster"]) - len(data["frame"])}
        manifest = {"schema_version": SCHEMA_VERSION, "mapping_sha256": sha256_file(mapping_path),
                    "feature_ids": dictionary.feature_id.tolist(), "cohorts": counts, "files": _manifest_files(temporary),
                    "source_specs": cfg["cohorts"], "source_evidence": {name: data["source_evidence"] for name, data in collected.items()},
                    "manual_qc_gates_inclusion": False, "old_normative_z_used": False}
        manifest["input_hash"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
        atomic_json(temporary / "manifest.json", manifest)
        if target.exists():
            old_path = target / "manifest.json"
            if not old_path.is_file() or json.loads(old_path.read_text(encoding="utf-8")).get("input_hash") != manifest["input_hash"]:
                raise ValueError("本输出目录已绑定不同输入，请为新输入选择新的输出目录")
            for name, expected in manifest["files"].items():
                if sha256_file(target / name) != expected:
                    raise ValueError(f"既有输入产物发生变化: {name}")
            return manifest
        os.rename(temporary, target)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
