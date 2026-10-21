"""LS独立常模：自然样条、交叉验证诊断与可JSON重载的冻结变换。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import cr
from sklearn.model_selection import StratifiedKFold

from .feature_selection import select_for_reference, validate_selection, model_frame, model_gm_mapping

from .ingest import (
    atomic_json, atomic_tsv, canonical_json, guard_output, read_tsv,
    sha256_file, validate_feature_dictionary,
)

MODEL_SCHEMA = "ls-natural-spline-reference-v1"


def _algorithm_identity() -> dict:
    """训练算法身份与三队列数据快照身份独立保存。"""
    return {
        "name": "statsmodels_ols_patsy_centered_natural_spline_v1",
        "estimator": "statsmodels.api.OLS",
        "solver": "pinv",
        "implementation_sha256": sha256_file(Path(__file__)),
        "feature_plan_sha256": sha256_file(Path(__file__).with_name("feature_selection.py")),
        "package_versions": {name: version(name) for name in ("statsmodels", "patsy", "numpy", "scikit-learn")},
    }


# ---------- 配置与输入校验 ----------
def _settings(cfg: dict) -> dict:
    defaults = {
        "age_min": 50, "age_max": 75, "folds": 5, "seed": 20261018,
        "spline_df": 4, "min_reference_n": 40, "min_residual_sd": 1e-8,
        "sex_encoding": {"female": 0, "male": 1}, "wmh_transform": "log1p", "gm_transform": "log",
        "covariates": ["age_spline", "sex", "log_icv"], "diagnostic_z_thresholds": [1, 2, 3],
        "diagnostic_coverage_z": 1.96,
    }
    settings = dict(defaults, **cfg.get("reference", {}))
    for name in ("folds", "seed", "spline_df", "min_reference_n"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"reference.{name}必须是整数")
    for name in ("age_min", "age_max", "min_residual_sd", "diagnostic_coverage_z"):
        if not np.isfinite(float(settings[name])):
            raise ValueError(f"reference.{name}必须为有限值")
    if not 0 <= float(settings["age_min"]) < float(settings["age_max"]):
        raise ValueError("参考年龄范围无效")
    if int(settings["spline_df"]) < 2 or int(settings["folds"]) < 2:
        raise ValueError("自然样条自由度与folds必须至少为2")
    if float(settings["min_residual_sd"]) <= 0 or int(settings["min_reference_n"]) <= 0:
        raise ValueError("参考人数与残差SD阈值必须为正")
    if settings["wmh_transform"] != "log1p" or settings["gm_transform"] != "log":
        raise ValueError("本版验证的变换为WMH log1p(mL)、GM log(mL)，不接受静默替代")
    covariates = settings["covariates"]
    if len(covariates) != len(set(covariates)) or set(covariates) - {"age_spline", "sex", "log_icv"}:
        raise ValueError("未知或重复常模协变量；本项目不建站点效应")
    encoding = settings["sex_encoding"]
    if set(encoding) != {"female", "male"} or not np.isfinite(list(encoding.values())).all() or encoding["female"] == encoding["male"]:
        raise ValueError("sex_encoding必须为不同的有限female/male数值")
    thresholds = settings["diagnostic_z_thresholds"]
    if len(set(thresholds)) != len(thresholds) or not all(np.isfinite(float(value)) and float(value) > 0 for value in thresholds):
        raise ValueError("诊断z阈值必须为正")
    if float(settings["diagnostic_coverage_z"]) <= 0:
        raise ValueError("诊断覆盖范围阈值必须为正")
    return settings


def _numeric_frame(frame: pd.DataFrame, feature_ids: list[str]) -> pd.DataFrame:
    required = {"participant_id", "subject_key", "cohort", "age", "sex", "icv_ml", *feature_ids}
    if not required.issubset(frame.columns):
        raise ValueError("常模输入缺少元数据或特征列")
    if frame.subject_key.duplicated().any() or frame.subject_key.eq("").any():
        raise ValueError("常模输入subject_key必须唯一且非空")
    result = frame.copy()
    for column in ["age", "icv_ml", *feature_ids]:
        result[column] = pd.to_numeric(result[column], errors="raise")
        if not np.isfinite(result[column].to_numpy(dtype=float)).all():
            raise ValueError(f"常模输入非有限值: {column}")
    if (result.icv_ml <= 0).any() or (result.age <= 0).any():
        raise ValueError("年龄与真实ICV必须为正")
    if not result.sex.isin(["female", "male"]).all():
        raise ValueError("常模sex只允许female/male")
    for column in feature_ids:
        if (result[column] < 0).any() or (column.startswith("gm_") and (result[column] <= 0).any()):
            raise ValueError(f"特征体积不合法: {column}")
    return result


def _transformed(frame: pd.DataFrame, modalities: list[str], feature_ids: list[str]) -> np.ndarray:
    values = frame[feature_ids].to_numpy(dtype=float)
    result = np.empty_like(values)
    for index, modality in enumerate(modalities):
        result[:, index] = np.log1p(values[:, index]) if modality == "WMH" else np.log(values[:, index])
    return result


# ---------- 自然样条设计；冻结节点及中心约束而非pickle DesignInfo ----------
def _fit_design(frame: pd.DataFrame, settings: dict) -> tuple[np.ndarray, dict]:
    age = frame.age.to_numpy(dtype=float)
    spec = {"covariates": settings["covariates"], "sex_encoding": settings["sex_encoding"],
            "age_min": float(settings["age_min"]), "age_max": float(settings["age_max"]),
            "spline_df": int(settings["spline_df"])}
    if "age_spline" in settings["covariates"]:
        # 中心约束消去一个自由度：df=4需要5个自然样条基函数、3个内部节点。
        probabilities = np.arange(1, spec["spline_df"]) / spec["spline_df"]
        knots = np.quantile(age, probabilities)
        if len(np.unique(knots)) != len(knots) or np.any(knots <= spec["age_min"]) or np.any(knots >= spec["age_max"]):
            raise ValueError("LS年龄分布不足以建立指定自然样条节点，禁止自动降阶")
        basis = np.asarray(cr(age, knots=knots, lower_bound=spec["age_min"], upper_bound=spec["age_max"]), dtype=float)
        spec["interior_knots"] = knots.tolist()
        spec["all_knots"] = [spec["age_min"], *knots.tolist(), spec["age_max"]]
        spec["constraint_matrix"] = basis.mean(axis=0, keepdims=True).tolist()
    design = _design(frame, spec)
    columns = ["intercept"]
    for covariate in settings["covariates"]:
        if covariate == "age_spline":
            columns.extend([f"age_spline_{index + 1}" for index in range(spec["spline_df"])])
        else:
            columns.append(covariate)
    spec["columns"] = columns
    if design.shape[1] != len(columns) or np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("LS常模设计矩阵不满秩；检查年龄、性别和ICV，不自动删协变量")
    if len(frame) <= design.shape[1]:
        raise ValueError("LS训练人数不足，无法估计残差自由度")
    return design, spec


def _design(frame: pd.DataFrame, spec: dict) -> np.ndarray:
    if len(frame) == 0:
        width = 1 + sum(spec["spline_df"] if name == "age_spline" else 1 for name in spec["covariates"])
        return np.empty((0, width))
    blocks = [np.ones((len(frame), 1))]
    for name in spec["covariates"]:
        if name == "age_spline":
            blocks.append(np.asarray(cr(frame.age.to_numpy(dtype=float), knots=np.asarray(spec["interior_knots"]),
                                        lower_bound=spec["age_min"], upper_bound=spec["age_max"],
                                        constraints=np.asarray(spec["constraint_matrix"], dtype=float))))
        elif name == "sex":
            blocks.append(frame.sex.map(spec["sex_encoding"]).to_numpy(dtype=float)[:, None])
        elif name == "log_icv":
            blocks.append(np.log(frame.icv_ml.to_numpy(dtype=float))[:, None])
        else:
            raise ValueError(f"未知协变量: {name}")
    design = np.column_stack(blocks)
    if not np.isfinite(design).all():
        raise ValueError("常模设计矩阵含非有限值")
    return design


# ---------- LS拟合与冻结转换 ----------
def fit_reference(frame: pd.DataFrame, dictionary: pd.DataFrame, settings: dict, selection: dict | None = None) -> dict:
    """仅用传入的LS训练子集估计参数；不接受C3或GB学习常模。"""
    feature_ids = dictionary.feature_id.tolist()
    data = _numeric_frame(model_frame(frame, selection) if selection is not None else frame, feature_ids)
    if not data.cohort.eq("LS").all():
        raise ValueError("常模只能在LS训练")
    if not data.age.between(settings["age_min"], settings["age_max"], inclusive="both").all():
        raise ValueError("LS常模训练不得包含支持年龄区间外样本")
    design, spec = _fit_design(data, settings)
    modalities = dictionary.modality.tolist()
    values = _transformed(data, modalities, feature_ids)
    # 每个特征调用成熟statsmodels OLS；残差SD仍使用SSE/(n-rank)。
    coefficients = np.empty((design.shape[1], len(feature_ids)))
    scale = np.empty(len(feature_ids))
    rank = design.shape[1]
    for index, feature_id in enumerate(feature_ids):
        fitted = sm.OLS(values[:, index], design, hasconst=True, missing="raise").fit(method="pinv")
        if int(fitted.model.rank) != rank or int(fitted.df_resid) != len(data) - rank:
            raise ValueError(f"statsmodels OLS自由度与已验证设计矩阵不一致: {feature_id}")
        coefficients[:, index] = np.asarray(fitted.params)
        scale[index] = np.sqrt(np.sum(np.asarray(fitted.resid) ** 2) / (len(data) - rank))
    invalid = np.flatnonzero(~np.isfinite(scale) | (scale < float(settings["min_residual_sd"])))
    if len(invalid):
        raise ValueError("LS残差SD低于设定阈值: " + ",".join(feature_ids[index] for index in invalid))
    result = {"schema_version": MODEL_SCHEMA, "estimator": "statsmodels.api.OLS", "design": spec, "settings": settings,
            "feature_ids": feature_ids, "modalities": modalities, "coefficients": coefficients.tolist(),
            "residual_sd": scale.tolist(), "disease_direction": [1 if item == "WMH" else -1 for item in modalities],
            "training_n": len(data), "training_rank": int(rank), "residual_degrees_of_freedom": int(len(data) - rank),
            "training_age_observed": [float(data.age.min()), float(data.age.max())],
            "training_subject_keys": data.subject_key.tolist(), "site_effects": False,
            "age_extrapolation": "natural_cubic_spline_linear_outside_boundary_knots"}
    if selection is not None:
        result["feature_selection"] = selection
    return result


def load_reference(path: Path) -> dict:
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    if model.get("schema_version") != MODEL_SCHEMA:
        raise ValueError("不支持的LS常模版本")
    features = model.get("feature_ids", [])
    coefficients = np.asarray(model.get("coefficients"), dtype=float)
    scale = np.asarray(model.get("residual_sd"), dtype=float)
    selection = model.get("feature_selection")
    if selection is not None:
        validate_selection(selection)
        if features != selection["model_feature_ids"]:
            raise ValueError("LS模型特征与冻结排除清单不一致")
    expected = len(selection["model_feature_ids"]) if selection is not None else 40
    if len(features) != expected or len(set(features)) != expected or coefficients.ndim != 2 or coefficients.shape[1] != expected:
        raise ValueError("LS常模特征或系数维数无效")
    if scale.shape != (expected,) or not np.isfinite(coefficients).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("LS常模包含无效系数或残差SD")
    return model


def transform_frame(frame: pd.DataFrame, model: dict) -> pd.DataFrame:
    """使用冻结LS模型；新队列不得重新估计节点、中心约束、系数或SD。"""
    selection = model.get("feature_selection")
    data = _numeric_frame(model_frame(frame, selection) if selection is not None else frame, model["feature_ids"])
    values = _transformed(data, model["modalities"], model["feature_ids"])
    prediction = _design(data, model["design"]) @ np.asarray(model["coefficients"])
    z = (values - prediction) / np.asarray(model["residual_sd"]) * np.asarray(model["disease_direction"])
    if not np.isfinite(z).all():
        raise ValueError("冻结常模产生非有限z值")
    # z表仅含实际建模特征；被排除的原始体积仍在inputs，不混入z列。
    excluded = [item["feature_id"] for item in model.get("feature_selection", {}).get("excluded", [])]
    result = data.drop(columns=excluded, errors="ignore").copy()
    result.loc[:, model["feature_ids"]] = z
    result["age_in_support"] = result.age.between(model["settings"]["age_min"], model["settings"]["age_max"], inclusive="both")
    result["reference_role"] = result.cohort
    return result


# ---------- 五折独立诊断；不以训练内校准替代OOF校准 ----------
def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _diagnostics(raw: pd.DataFrame, zframe: pd.DataFrame, errors: np.ndarray,
                 dictionary: pd.DataFrame, settings: dict, split: str) -> list[dict]:
    records = []
    age = raw.age.to_numpy(dtype=float)
    sex = raw.sex.map(settings["sex_encoding"]).to_numpy(dtype=float)
    log_icv = np.log(raw.icv_ml.to_numpy(dtype=float))
    for index, row in dictionary.reset_index(drop=True).iterrows():
        z = zframe[row.feature_id].to_numpy(dtype=float)
        record = {"feature_id": row.feature_id, "modality": row.modality, "split": split, "n": len(z),
                  "mean_z": float(np.mean(z)), "sd_z": float(np.std(z, ddof=1)) if len(z) > 1 else None,
                  "rmse_transformed": float(np.sqrt(np.mean(errors[:, index] ** 2))),
                  "coverage_95": float(np.mean(np.abs(z) <= settings["diagnostic_coverage_z"])),
                  "coverage_z_threshold": float(settings["diagnostic_coverage_z"]),
                  "correlation_age": _correlation(z, age), "correlation_sex": _correlation(z, sex),
                  "correlation_log_icv": _correlation(z, log_icv)}
        for threshold in settings["diagnostic_z_thresholds"]:
            record[f"proportion_z_gt_{float(threshold):g}"] = float(np.mean(z > float(threshold)))
        records.append(record)
    return records


def _oof_reference(frame: pd.DataFrame, dictionary: pd.DataFrame, settings: dict,
                   selection: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], list[dict]]:
    folds = int(settings["folds"])
    if frame.sex.value_counts().min() < folds:
        raise ValueError("LS每个性别至少需要folds名，无法保持五折性别分层")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(settings["seed"]))
    assignments = frame[["participant_id", "subject_key", "cohort", "age", "sex"]].copy()
    assignments["fold"] = -1
    oof = model_frame(frame, selection) if selection is not None else frame.copy()
    oof["fold"] = -1
    errors = np.empty((len(frame), len(dictionary)))
    diagnostics, models = [], []
    for fold, (train_index, test_index) in enumerate(splitter.split(frame, frame.sex)):
        training, heldout = frame.iloc[train_index], frame.iloc[test_index]
        model = fit_reference(training, dictionary, settings, selection)
        if selection is not None:
            model["feature_selection"] = selection
        prediction = transform_frame(heldout, model)
        oof.loc[test_index, dictionary.feature_id.tolist()] = prediction[dictionary.feature_id.tolist()].to_numpy()
        oof.loc[test_index, "fold"] = fold
        assignments.loc[test_index, "fold"] = fold
        errors[test_index] = prediction[dictionary.feature_id.tolist()].to_numpy(dtype=float) * np.asarray(model["residual_sd"])
        diagnostics.extend(_diagnostics(heldout, prediction, errors[test_index], dictionary, settings, f"fold_{fold}"))
        models.append(model)
    oof["age_in_support"] = True
    oof["reference_role"] = "LS_out_of_fold"
    diagnostics.extend(_diagnostics(frame, oof, errors, dictionary, settings, "LS_oof"))
    return oof, assignments, diagnostics, models


# ---------- 常模阶段产物与输入身份绑定 ----------
def run_reference(cfg: dict, output: Path) -> dict:
    output = Path(output)
    guard_output(cfg, output)
    settings = _settings(cfg)
    source = output / "inputs"
    input_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in input_manifest["files"].items():
        if sha256_file(source / name) != expected:
            raise ValueError(f"常模输入文件哈希改变: {name}")
    dictionary = validate_feature_dictionary(source / "feature_dictionary.tsv")
    frames = {cohort: _numeric_frame(read_tsv(source / f"{cohort}.tsv"), dictionary.feature_id.tolist()) for cohort in ("LS", "C3", "GB")}
    for cohort, frame in frames.items():
        if not frame.cohort.eq(cohort).all():
            raise ValueError(f"常模输入队列身份不一致: {cohort}")
        if frame.empty or not frame.age.between(settings["age_min"], settings["age_max"]).any():
            raise ValueError(f"{cohort}主年龄范围内导入人数为0；先检查inputs/{cohort}_audit.tsv，不能继续正式分析")
    eligible = frames["LS"].age.between(settings["age_min"], settings["age_max"], inclusive="both")
    ls = frames["LS"].loc[eligible].sort_values("subject_key").reset_index(drop=True)
    if len(ls) < int(settings["min_reference_n"]):
        raise ValueError(f"LS支持年龄内合格人数{len(ls)}低于min_reference_n={settings['min_reference_n']}")
    dictionary, selection = select_for_reference(dictionary, ls, cfg)
    gm_mapping = None
    if selection["merged"]:
        gm_mapping = model_gm_mapping(pd.read_csv(cfg["paths"]["mapping"], sep="\t"), dictionary, selection)
        configured = cfg["paths"].get("model_mapping")
        if configured:
            expected_mapping = pd.read_csv(configured, sep="\t").astype(str)
            if not expected_mapping.equals(gm_mapping.astype(str)):
                raise ValueError("配置的GM19映射与冻结合并方案不一致；不能使用旧映射")
    training_columns = ["participant_id", "subject_key", "cohort", "age", "sex", "icv_ml", *dictionary.feature_id.tolist()]
    training_bytes = model_frame(ls, selection)[training_columns].to_csv(sep="\t", index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
    # 常模身份只取LS支持范围内真正参与训练的数据；GB变化不影响冻结参数或训练身份。
    identity = {"training_data_sha256": hashlib.sha256(training_bytes).hexdigest(), "settings": settings,
                "feature_dictionary_sha256": sha256_file(source / "feature_dictionary.tsv"),
                "feature_selection": selection,
                "algorithm_identity": _algorithm_identity()}
    reference_hash = hashlib.sha256(canonical_json(identity)).hexdigest()
    snapshot_identity = {"input_manifest_hash": input_manifest["input_hash"],
                         "cohort_table_sha256": {cohort: sha256_file(source / f"{cohort}.tsv") for cohort in frames}}
    input_snapshot_hash = hashlib.sha256(canonical_json(snapshot_identity)).hexdigest()
    transform_hash = hashlib.sha256(canonical_json({"reference_hash": reference_hash,
                                                   "input_snapshot_hash": input_snapshot_hash})).hexdigest()
    target = output / "reference"
    if target.exists():
        path = target / "manifest.json"
        if not path.is_file():
            raise ValueError("已有reference目录缺少完成manifest，请使用新输出目录")
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("reference_hash") != reference_hash:
            raise ValueError("已有常模绑定不同输入或参数，请使用新的输出目录")
        if old.get("transform_hash") != transform_hash:
            raise ValueError("常模训练身份未变，但三队列转换输入已改变，请使用新的输出目录")
        for name, expected in old["files"].items():
            if sha256_file(target / name) != expected:
                raise ValueError(f"已有常模产物哈希改变: {name}")
        return old
    model = fit_reference(ls, dictionary, settings, selection)
    model.update(reference_hash=reference_hash, training_data_sha256=identity["training_data_sha256"],
                 algorithm_identity=identity["algorithm_identity"], feature_selection=selection)
    oof, assignments, diagnostics, fold_models = _oof_reference(ls, dictionary, settings, selection)
    fitted = transform_frame(ls, model)
    errors = fitted[dictionary.feature_id.tolist()].to_numpy() * np.asarray(model["residual_sd"])
    diagnostics.extend(_diagnostics(ls, fitted, errors, dictionary, settings, "LS_fit"))
    temporary = Path(tempfile.mkdtemp(prefix=".reference.", dir=output))
    try:
        atomic_json(temporary / "model.json", model)
        atomic_json(temporary / "feature_selection.json", selection)
        atomic_tsv(temporary / "feature_dictionary.tsv", dictionary)
        if gm_mapping is not None:
            atomic_tsv(temporary / "gm_model_mapping.tsv", gm_mapping)
        atomic_tsv(temporary / "feature_exclusions.tsv", pd.DataFrame(selection["excluded"], columns=[
            "feature_id", "reason", "criterion", "evidence_cohort", "ls_primary_n",
            "ls_primary_nonzero_n", "ls_primary_unique_values"]))
        # 以真正保存再加载的JSON模型完成三队列转换，验证可移植的节点和中心约束。
        frozen = load_reference(temporary / "model.json")
        counts = {}
        for cohort, frame in frames.items():
            atomic_tsv(temporary / f"{cohort}_model_raw.tsv", model_frame(frame, selection))
            transformed = transform_frame(frame, frozen)
            primary = transformed.loc[transformed.age_in_support].copy()
            atomic_tsv(temporary / f"{cohort}_z.tsv", transformed)
            atomic_tsv(temporary / f"{cohort}_primary_z.tsv", primary)
            counts[cohort] = {"all_ages": len(transformed), "primary": len(primary), "age_extrapolated": len(transformed) - len(primary)}
        atomic_tsv(temporary / "LS_oof_z.tsv", oof)
        atomic_tsv(temporary / "fold_assignments.tsv", assignments)
        atomic_tsv(temporary / "diagnostics.tsv", pd.DataFrame(diagnostics))
        atomic_json(temporary / "fold_models.json", {"models": fold_models})
        support = frames["LS"][["participant_id", "subject_key", "age", "sex"]].copy()
        support["reference_training_eligible"] = eligible
        support["reason"] = np.where(eligible, "LS_in_reference_age_support", "LS_outside_reference_age_support")
        atomic_tsv(temporary / "reference_training_audit.tsv", support)
        manifest = {"schema_version": MODEL_SCHEMA, "reference_hash": reference_hash, **identity,
                    "input_snapshot_hash": input_snapshot_hash, "transform_hash": transform_hash,
                    "input_snapshot_identity": snapshot_identity,
                    "model_sha256": sha256_file(temporary / "model.json"), "feature_ids": dictionary.feature_id.tolist(),
                    "training_n": len(ls), "cohorts": counts, "manual_qc_gates_inclusion": False,
                    "site_effects": False, "clinical_outcomes_used": False,
                    "files": {path.name: sha256_file(path) for path in sorted(temporary.iterdir()) if path.is_file()}}
        atomic_json(temporary / "manifest.json", manifest)
        os.rename(temporary, target)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
