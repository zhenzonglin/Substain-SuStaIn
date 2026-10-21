#!/usr/bin/env python3
"""只读诊断LS残差尺度失败：使用已导入表和原始JSON快照，不重跑分析。"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold
import yaml

from substain_sustain.config import load_config
from substain_sustain.ingest import read_tsv, sha256_file, validate_feature_dictionary
from substain_sustain.reference import _fit_design, _numeric_frame, _settings, _transformed


# ===== 导入审计：只读已保存的排除记录，示例替换原始ID =====
def input_audit(cfg: dict) -> dict:
    source = Path(cfg["output_dir"]) / "inputs"
    result = {}
    for cohort in ("LS", "C3", "GB"):
        path = source / f"{cohort}_audit.tsv"
        if not path.is_file():
            result[cohort] = {"error": "缺少导入审计文件"}
            continue
        frame = read_tsv(path)
        included = frame.included.str.lower().isin(["true", "1"])
        excluded = frame.loc[~included]
        examples = {}
        for reason, group in excluded.groupby("reason", sort=True):
            row = group.iloc[0]
            text = str(row.detail)
            person = str(row.participant_id)
            if person:
                text = text.replace(person, "<ID>")
            examples[str(reason)] = text
        spec = cfg["cohorts"][cohort]
        result[cohort] = {"roster_n": len(frame), "included_n": int(included.sum()),
                          "excluded_n": int((~included).sum()),
                          "reasons": excluded.reason.value_counts().to_dict(),
                          "one_redacted_example_per_reason": examples,
                          "configured_paths": {name: spec.get(name) for name in
                                               ("derivatives", "participants", "metadata", "input_bundle")}}
    return result


# ===== 一、完全沿用当前LS设计和OLS残差口径；只返回诊断数值 =====
def residual_summary(frame, dictionary, settings: dict, split: str) -> list[dict]:
    design, _ = _fit_design(frame, settings)
    ids = dictionary.feature_id.tolist()
    transformed = _transformed(frame, dictionary.modality.tolist(), ids)
    rows = []
    for index, feature in enumerate(ids):
        raw = frame[feature].to_numpy(dtype=float)
        fit = sm.OLS(transformed[:, index], design, hasconst=True, missing="raise").fit(method="pinv")
        sd = float(np.sqrt(np.sum(np.asarray(fit.resid) ** 2) / (len(frame) - design.shape[1])))
        valid = bool(np.isfinite(sd) and sd >= settings["min_residual_sd"])
        reason = ("pass" if valid else "all_zero" if np.all(raw == 0) else
                  "constant_nonzero" if len(np.unique(raw)) == 1 else "low_or_nonfinite_residual_sd")
        rows.append({"split": split, "feature_id": feature, "n": len(raw),
                     "n_zero": int(np.sum(raw == 0)), "n_positive": int(np.sum(raw > 0)),
                     "n_unique": int(len(np.unique(raw))), "min_ml": float(raw.min()),
                     "max_ml": float(raw.max()), "residual_sd": sd if np.isfinite(sd) else None,
                     "design_rank": int(np.linalg.matrix_rank(design)),
                     "residual_df": float(fit.df_resid), "status": reason})
    return rows


# ===== 二、对照保存的LS原始WMH快照；不访问或显示患者MRI和逐人信息 =====
def compare_snapshots(path: Path, frame, features: list[str], options: dict) -> dict:
    table = frame.set_index("subject_key")
    seen, duplicates, extra = set(), 0, 0
    counts = {name: {"compared": 0, "mismatch": 0, "missing_or_invalid": 0} for name in features}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("included") is not True:
                continue
            key = record.get("subject_key")
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            if key not in table.index:
                extra += 1
                continue
            raw = record.get("features", {}).get("wmh", {}).get("raw_ml", {})
            for name in features:
                try:
                    value = float(raw[name])
                    if not np.isfinite(value) or isinstance(raw[name], bool):
                        raise ValueError("invalid value")
                except (KeyError, TypeError, ValueError):
                    counts[name]["missing_or_invalid"] += 1
                    continue
                counts[name]["compared"] += 1
                counts[name]["mismatch"] += int(not np.isclose(
                    value, float(table.at[key, name]), rtol=options["comparison_rtol"],
                    atol=options["comparison_atol"]))
    return {"missing_subjects": len(set(table.index) - seen), "extra_subjects": extra,
            "duplicate_subjects": duplicates, "features": counts}


# ===== 三、全体LS与各训练折均检查；C3/GB仅显示分布，不参与估计 =====
def diagnose(cfg: dict, options: dict) -> dict:
    source = Path(cfg["output_dir"]) / "inputs"
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    names = ["feature_dictionary.tsv", "LS.tsv", "C3.tsv", "GB.tsv"]
    if options["verify_ls_raw_snapshots"]:
        names.append("LS_raw_records.jsonl.gz")
    for name in names:
        if name not in manifest["files"] or sha256_file(source / name) != manifest["files"][name]:
            raise ValueError("输入快照缺失或哈希改变：" + name)
    dictionary = validate_feature_dictionary(source / "feature_dictionary.tsv")
    ids = dictionary.feature_id.tolist()
    settings = _settings(cfg)
    frames = {name: _numeric_frame(read_tsv(source / f"{name}.tsv"), ids) for name in ("LS", "C3", "GB")}
    for name, frame in frames.items():
        if not frame.cohort.eq(name).all():
            raise ValueError("队列身份错误：" + name)
    ls = frames["LS"]
    ls = ls.loc[ls.age.between(settings["age_min"], settings["age_max"])].sort_values("subject_key").reset_index(drop=True)
    result = {"input_hashes": "pass", "age_range": [settings["age_min"], settings["age_max"]],
              "min_residual_sd": settings["min_residual_sd"], "ls_primary_n": len(ls),
              "ls_sex_counts": ls.sex.value_counts().to_dict(), "fit_errors": {}, "residuals": []}
    splits = [("LS_full", ls)] if len(ls) else []
    if len(ls) < settings["min_reference_n"]:
        result["fit_errors"]["sample_size"] = "LS人数低于min_reference_n"
    if len(ls) == 0 or ls.sex.value_counts().min() < settings["folds"]:
        result["fit_errors"]["OOF"] = "性别分层样本数不足"
    else:
        splitter = StratifiedKFold(n_splits=settings["folds"], shuffle=True, random_state=settings["seed"])
        splits.extend((f"LS_train_fold_{i}", ls.iloc[train]) for i, (train, _) in enumerate(splitter.split(ls, ls.sex)))
    all_rows = []
    for label, training in splits:
        try:
            all_rows.extend(residual_summary(training, dictionary, settings, label))
        except (ValueError, np.linalg.LinAlgError) as exc:
            result["fit_errors"][label] = str(exc)
    flagged = {row["feature_id"] for row in all_rows if row["status"] != "pass"}
    focus = set(options["focus_features"]) | flagged
    if focus - set(ids):
        raise ValueError("诊断配置中含不存在的focus_features")
    result["flagged_features"] = sorted(flagged)
    result["residuals"] = [row for row in all_rows if row["feature_id"] in focus]
    distributions = []
    for cohort, frame in frames.items():
        for scope, subset in (("all_ages", frame), ("primary", frame.loc[
                frame.age.between(settings["age_min"], settings["age_max"]) ])):
            for name in sorted(focus):
                values = subset[name].to_numpy(dtype=float)
                distributions.append({"cohort": cohort, "scope": scope, "feature_id": name, "n": len(values),
                                      "n_zero": int(np.sum(values == 0)), "n_positive": int(np.sum(values > 0)),
                                      "n_unique": len(np.unique(values)),
                                      "max_ml": float(values.max()) if len(values) else None})
    result["distributions"] = distributions
    wmh = dictionary.loc[(dictionary.modality == "WMH") & dictionary.feature_id.isin(focus)]
    if options["verify_ls_raw_snapshots"]:
        result["ls_raw_snapshot_check"] = compare_snapshots(
            source / "LS_raw_records.jsonl.gz", frames["LS"], wmh.feature_id.tolist(), options)
    if cfg.get("assets", {}).get("wmh_atlas"):
        try:
            image = nib.load(cfg["assets"]["wmh_atlas"])
            labels, counts = np.unique(np.asanyarray(image.dataobj), return_counts=True)
            lookup = dict(zip(labels.tolist(), counts.tolist()))
            result["display_atlas"] = [{"feature_id": row.feature_id, "label": int(row.atlas_index),
                                        "voxels": lookup.get(int(row.atlas_index), 0)} for row in wmh.itertuples()]
        except (OSError, ValueError) as exc:
            result["display_atlas_error"] = str(exc)
    return result


def print_summary(result: dict) -> None:
    print(f"输入哈希：{result['input_hashes']} | LS主年龄范围：{result['age_range']} | "
          f"LS人数：{result['ls_primary_n']} | 性别：{result['ls_sex_counts']}")
    print(f"残差SD门槛：{result['min_residual_sd']:.3g} | 全40维检查发现问题：{result['flagged_features']}")
    for label, problem in result["fit_errors"].items():
        print(f"设计/人数错误 {label}: {problem}")
    for feature in sorted({row["feature_id"] for row in result["distributions"]}):
        print(f"\n特征：{feature}")
        for row in result["residuals"]:
            if row["feature_id"] == feature:
                sd = f"{row['residual_sd']:.12g}" if row["residual_sd"] is not None else "非有限"
                print(f"  {row['split']}: n={row['n']}, 非零={row['n_positive']}, "
                      f"不同值={row['n_unique']}, 残差SD={sd}, 结果={row['status']}")
        for row in result["distributions"]:
            if row["feature_id"] == feature:
                print(f"  {row['cohort']}/{row['scope']}: n={row['n']}, 零={row['n_zero']}, "
                      f"非零={row['n_positive']}, 最大mL={row['max_ml']}")
    for key in ("ls_raw_snapshot_check", "display_atlas", "display_atlas_error"):
        if key in result:
            print(f"\n{key}: " + json.dumps(result[key], ensure_ascii=False))
    print("\n只读诊断完成。未更改数据、特征、阈值或模型；未启动SuStaIn。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--diagnostic-config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "config/reference_diagnostics.yaml")
    parser.add_argument("--json", action="store_true", help="输出不含逐人信息的完整汇总JSON")
    parser.add_argument("--inputs-only", action="store_true", help="仅显示三队列导入人数、排除原因和已隐藏ID的示例")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.inputs_only:
        print(json.dumps(input_audit(cfg), ensure_ascii=False, indent=2))
        return 0
    options = yaml.safe_load(args.diagnostic_config.read_text(encoding="utf-8"))
    if not isinstance(options["verify_ls_raw_snapshots"], bool):
        parser.error("verify_ls_raw_snapshots必须是布尔值")
    for name in ("comparison_atol", "comparison_rtol"):
        if not np.isfinite(options[name]) or options[name] < 0:
            parser.error(name + "必须是非负有限数")
    result = diagnose(cfg, options)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
