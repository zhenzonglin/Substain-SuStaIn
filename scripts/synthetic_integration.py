"""真实 LS 参考→官方 SuStaIn 的小型合成集成验证，不读取患者目录。

使用已验证的生产函数；只有数据规模和运行配置不同。官方每模型三轮
10000 次 MCMC 提议尺度调优保持不变。默认继续生成报告、展示包并独立重建验收。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from substain_sustain.config import load_config
from substain_sustain.ingest import atomic_json, atomic_tsv, canonical_json, sha256_file
from substain_sustain.model import run_crossvalidate, run_fit, run_validate
from substain_sustain.reference import (
    _algorithm_identity, _diagnostics, _oof_reference, _settings,
    fit_reference, transform_frame,
)


# ---------- 所有合成数据参数均从独立配置读取 ----------
def generate_cohorts(cfg: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    simulation = cfg["synthetic"]
    rng = np.random.default_rng(int(simulation["seed"]))
    feature_specs = simulation["features"]
    dictionary = pd.DataFrame(feature_specs)[["feature_id", "label", "modality", "atlas_index"]]
    loadings = np.asarray(simulation["subtype_loadings"], dtype=float)
    if loadings.ndim != 2 or loadings.shape[1] != len(dictionary):
        raise ValueError("合成亚型负荷必须与特征字典一致。")
    age_min, age_max = cfg["reference"]["age_min"], cfg["reference"]["age_max"]
    margin = simulation["age_interior_margin"]
    frames = {}
    for cohort, primary_n in simulation["cohort_primary_n"].items():
        primary_n = int(primary_n)
        primary_ages = rng.uniform(age_min + margin, age_max - margin, primary_n)
        if simulation["include_age_endpoints"]:
            primary_ages[:2] = [age_min, age_max]
        ages = np.r_[primary_ages, simulation["outside_reference_ages"]]
        n = len(ages)
        sexes = np.where(np.arange(n) % 2, "male", "female")
        icv = np.maximum(rng.normal(simulation["icv_mean_ml"], simulation["icv_sd_ml"], n), simulation["icv_min_ml"])
        frame = pd.DataFrame({"subject_key": [f"SYNTHETIC_{cohort}:{i:04d}" for i in range(n)],
                              "participant_id": [f"SYNTHETIC_{i:04d}" for i in range(n)],
                              "cohort": cohort, "age": ages, "sex": sexes, "icv_ml": icv})
        subtype = rng.integers(0, len(loadings), n)
        stages = np.zeros(n) if cohort == "LS" else rng.uniform(simulation["disease_stage_min"], simulation["disease_stage_max"], n)
        for j, spec in enumerate(feature_specs):
            baseline = (spec["transformed_mean"] + spec["age_slope"] * (ages - simulation["age_center"])
                        + spec["sex_effect"] * (sexes == "male")
                        + spec["log_icv_effect"] * np.log(icv / simulation["icv_mean_ml"]))
            direction = 1 if spec["modality"] == "WMH" else -1
            residual = rng.normal(0, spec["residual_sd"], n)
            transformed = baseline + residual + direction * stages * loadings[subtype, j] * spec["residual_sd"]
            if spec["modality"] == "WMH":
                volume = np.expm1(np.maximum(transformed, simulation["wmh_log_floor"]))
            else:
                volume = np.exp(transformed)
            frame[spec["feature_id"]] = volume
        frames[cohort] = frame
    if set(frames) != {"LS", "C3", "GB"}:
        raise ValueError("合成配置必须恰好包含 LS、C3、GB。")
    return dictionary, frames


# ---------- 调用真实参考模块；三维例外只存在于该明确合成入口 ----------
def prepare_reference(cfg: dict, output: Path) -> None:
    dictionary, frames = generate_cohorts(cfg)
    inputs = output / "inputs"
    reference = output / "reference"
    inputs.mkdir(parents=True)
    reference.mkdir()
    atomic_tsv(inputs / "feature_dictionary.tsv", dictionary)
    for cohort, frame in frames.items():
        atomic_tsv(inputs / f"{cohort}.tsv", frame)
        audit = frame[["subject_key", "participant_id", "cohort", "age", "sex"]].copy()
        audit["eligible"] = True
        audit["reason"] = "synthetic_generated_volume_not_patient_data"
        atomic_tsv(inputs / f"{cohort}_audit.tsv", audit)
    input_files = {p.name: sha256_file(p) for p in inputs.iterdir() if p.is_file()}
    input_hash = hashlib.sha256(canonical_json(input_files)).hexdigest()
    atomic_json(inputs / "manifest.json", {"input_hash": input_hash, "files": input_files,
                                          "profile": "synthetic_integration", "patient_data": False})
    settings = _settings(cfg)
    ls = frames["LS"].loc[frames["LS"].age.between(settings["age_min"], settings["age_max"], inclusive="both")].reset_index(drop=True)
    if len(ls) < settings["min_reference_n"]:
        raise ValueError("合成 LS 样本仍需满足设定的参考最小人数。")
    training_bytes = ls.to_csv(sep="\t", index=False, lineterminator="\n", float_format="%.17g").encode()
    identity = {"training_data_sha256": hashlib.sha256(training_bytes).hexdigest(),
                "feature_dictionary_sha256": sha256_file(inputs / "feature_dictionary.tsv"),
                "settings": settings, "algorithm_identity": _algorithm_identity()}
    reference_hash = hashlib.sha256(canonical_json(identity)).hexdigest()
    model = fit_reference(ls, dictionary, settings)
    model.update(reference_hash=reference_hash, training_data_sha256=identity["training_data_sha256"],
                 algorithm_identity=identity["algorithm_identity"], synthetic_integration=True)
    atomic_json(reference / "model.json", model)
    # 真正保存后重新读 JSON；不伪造健康参考系数、尺度或校准诊断。
    frozen = json.loads((reference / "model.json").read_text())
    oof, assignments, diagnostics, fold_models = _oof_reference(ls, dictionary, settings)
    fitted = transform_frame(ls, frozen)
    errors = fitted[dictionary.feature_id].to_numpy() * np.asarray(frozen["residual_sd"])
    diagnostics.extend(_diagnostics(ls, fitted, errors, dictionary, settings, "LS_fit"))
    counts = {}
    for cohort, frame in frames.items():
        z = transform_frame(frame, frozen)
        primary = z.loc[z.age_in_support].copy()
        atomic_tsv(reference / f"{cohort}_z.tsv", z)
        atomic_tsv(reference / f"{cohort}_primary_z.tsv", primary)
        counts[cohort] = {"all_ages": len(z), "primary": len(primary), "age_extrapolated": int((~z.age_in_support).sum())}
    atomic_tsv(reference / "LS_oof_z.tsv", oof)
    atomic_tsv(reference / "fold_assignments.tsv", assignments)
    atomic_tsv(reference / "diagnostics.tsv", pd.DataFrame(diagnostics))
    atomic_json(reference / "fold_models.json", {"models": fold_models})
    transform_hash = hashlib.sha256(canonical_json({"reference_hash": reference_hash, "input_snapshot_hash": input_hash})).hexdigest()
    atomic_json(reference / "manifest.json", {"reference_hash": reference_hash, **identity,
                                             "input_snapshot_hash": input_hash, "transform_hash": transform_hash,
                                             "training_n": len(ls), "cohorts": counts, "synthetic_integration": True,
                                             "files": {p.name: sha256_file(p) for p in reference.iterdir() if p.is_file()}})


# ---------- 模型执行与可恢复状态；不覆盖已有配置身份 ----------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "synthetic.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.get("runtime", {}).get("profile") != "synthetic_integration" or "synthetic" not in cfg:
        raise ValueError("此脚本只接受明确 synthetic_integration 配置。")
    output = Path(cfg["output_dir"])
    config_hash = hashlib.sha256(canonical_json(cfg)).hexdigest()
    provenance_path = output / "PROVENANCE.json"
    if output.exists():
        if not provenance_path.is_file() or json.loads(provenance_path.read_text()).get("config_hash") != config_hash:
            raise ValueError("已有合成输出不属于同一配置，拒绝覆盖。")
    else:
        output.mkdir(parents=True)
        atomic_json(provenance_path, {"artifact_type": "synthetic_reference_model_report_integration",
                                     "config_hash": config_hash, "formal_research_result": False, "patient_data": False,
                                     "feature_count": len(cfg["synthetic"]["features"]),
                                     "reference_fitting": "actual statsmodels OLS and held-out LS five-fold diagnostics",
                                     "sustain_core": "actual frozen upstream EM/MCMC; three times 10000 tuning unchanged",
                                     "scope": "software integration validation, not formal patient findings",
                                     "formal_contract_not_relaxed": "production retains raw 40 features, explicit LS-only exclusions and prespecified raw-volume GM merges"})
        atomic_json(output / "synthetic_config_resolved.json", cfg)
        prepare_reference(cfg, output)
    if not (output / "reference" / "manifest.json").is_file():
        raise ValueError("合成参考准备未完成；请保留证据并使用新的测试输出目录。")
    summary = {"crossvalidate": run_crossvalidate(cfg, output), "fit": run_fit(cfg, output),
               "validate": run_validate(cfg, output)}
    if cfg["synthetic"]["report_and_bundle"]:
        from substain_sustain.bundle import create_bundle, verify_bundle
        from substain_sustain.report import run_report

        summary["report"] = run_report(cfg, output)
        receipt_path = output / "synthetic_bundle_receipt.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
        else:
            receipt = create_bundle(cfg, output, Path(cfg["report"]["bundle_dir"]))
            atomic_json(receipt_path, receipt)
        summary["bundle"] = receipt
        summary["bundle_verification"] = verify_bundle(Path(receipt["path"]))
        if receipt["sha256"] != summary["bundle_verification"]["bundle_sha256"]:
            raise ValueError("展示包已改变，旧receipt不能冒充当前归档身份。")
    atomic_json(output / "synthetic_integration_summary.json", summary)
    print(json.dumps({"status": "integration_complete" if cfg["synthetic"]["report_and_bundle"] else "model_pipeline_complete",
                      "output": str(output), "patient_data": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
