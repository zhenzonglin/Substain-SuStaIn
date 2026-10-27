"""冻结模型的稳定数值评估；不改变训练用的官方EM/MCMC似然。"""
from __future__ import annotations

import math
from pathlib import Path
from types import MethodType


# ===== 一、仅控制评估方式、分块、诊断及一致性容差 =====
def numerical_settings(cfg):
    defaults = {"backend": "logsumexp_v1", "block_rows": 32, "relative_tolerance": 1e-10,
                "absolute_log_tolerance": 1e-8, "z_warning_threshold": 5,
                "top_features": 5, "heldout_dir": "heldout_evaluation"}
    supplied = cfg.get("numerical_evaluation", {})
    if set(supplied) - set(defaults):
        raise ValueError("numerical_evaluation包含未知参数")
    result = {**defaults, **supplied}
    if result["backend"] != "logsumexp_v1":
        raise ValueError("只支持已验证的logsumexp_v1评估")
    for key in ("block_rows", "top_features"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"numerical_evaluation.{key}必须为正整数")
    for key in ("relative_tolerance", "absolute_log_tolerance", "z_warning_threshold"):
        if type(result[key]) not in (int, float) or not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"numerical_evaluation.{key}必须为有限正数")
    name = result["heldout_dir"]
    if not isinstance(name, str) or Path(name).name != name or not name.startswith("heldout") or "\\" in name:
        raise ValueError("heldout_dir必须是以heldout开头的单一目录名")
    return result


# ===== 二、复用已核验官方轨迹及逐特征算式，只推迟exp运算 =====
def build_log_kernel(engine, block_rows):
    import numpy as np
    from pySuStaIn.ZscoreSustain import ZscoreSustain
    from .acceleration import _verified_code
    from .model import _upstream_provenance
    _upstream_provenance()
    if type(block_rows) is not int or block_rows < 1:
        raise ValueError("评估分块必须为正整数")

    def evaluate_blocks(data, stage_value, sigmat, factor, coeff):
        result = np.empty((len(data), stage_value.shape[1]), dtype=float)
        for first in range(0, len(data), block_rows):
            x = (data[first:first+block_rows, :, None] - stage_value) / sigmat[None, :, None]
            result[first:first+block_rows] = coeff + np.sum(factor[None, :, None] - 0.5*np.square(x), 1)
        if not np.isfinite(result).all():
            raise FloatingPointError("对数似然仍非有限；检查极端z/输入，禁止裁剪或填小数")
        return result

    namespace = {"np": np, "ZscoreSustain": ZscoreSustain, "evaluate_blocks": evaluate_blocks}
    exec(_verified_code(), namespace)
    return MethodType(namespace["blocked_likelihood"], engine)


def log_joint(kernel, data, sequence, fractions):
    import numpy as np
    fractions = np.asarray(fractions, dtype=float).reshape(-1)
    if (fractions.shape != (len(sequence),) or not np.isfinite(fractions).all()
            or np.any(fractions < 0) or not np.isclose(fractions.sum(), 1)):
        raise ValueError("冻结亚型比例无效")
    stages = np.stack([kernel(data, row) for row in sequence], axis=2)
    with np.errstate(divide="ignore"):
        weighted = stages + np.log(fractions)[None, None, :]
    return stages, weighted


def normalize_log_joint(weighted, fractions=None):
    import numpy as np
    from scipy.special import logsumexp
    offset = np.max(weighted, axis=(1, 2), keepdims=True)
    if not np.isfinite(offset).all():
        raise FloatingPointError("对数域联合后验无有限支持")
    # 先移去共同的大负数，再加入log(f)；极端z也不能让小亚型权重被舍入掉。
    shifted = weighted-offset
    if fractions is not None:
        fractions = np.asarray(fractions, dtype=float)
        if (fractions.shape != (weighted.shape[2],) or not np.isfinite(fractions).all()
                or np.any(fractions < 0) or not np.isclose(fractions.sum(), 1)):
            raise ValueError("归一化亚型比例无效")
        with np.errstate(divide="ignore"):
            shifted = shifted+np.log(fractions)[None, None, :]
    normalizer = logsumexp(shifted, axis=(1, 2), keepdims=True)
    if not np.isfinite(normalizer).all():
        raise FloatingPointError("对数域联合后验无有限支持")
    return np.exp(shifted-normalizer)


# ===== 三、严格沿用固定官方版本的CV链抽样规则，不使用展示burn-in =====
def official_cv_indices(n_iterations):
    import numpy as np
    if n_iterations < 1:
        raise ValueError("MCMC链为空")
    count = (int(np.round(n_iterations/1000)) if n_iterations >= 1e6 else
             int(np.round(n_iterations/100)) if n_iterations >= 1e5 else n_iterations)
    return np.round(np.linspace(0, n_iterations-1, count)).astype(int)


def evaluate_samples(engine, data, sequence, fractions, settings, progress=None):
    import numpy as np
    from scipy.special import logsumexp
    indices = official_cv_indices(sequence.shape[2])
    kernel = build_log_kernel(engine, settings["block_rows"])
    raw = np.empty((data.getNumSamples(), len(indices)))
    logs = np.empty_like(raw)
    for j, t in enumerate(indices):
        stages, weighted = log_joint(kernel, data, sequence[:, :, t], fractions[:, t])
        # 与官方运算顺序相同，单独保留原线性结果，不能把exp(log均值)冒充原缓存。
        raw[:, j] = np.sum(np.sum(np.exp(stages)*fractions[:, t][None, None, :], axis=2), axis=1)
        logs[:, j] = logsumexp(weighted, axis=(1, 2))
        if progress is not None:
            progress.update(phase="heldout_log_evaluation", evaluation_draw=j+1, evaluation_draws=len(indices))
    if not np.isfinite(raw).all() or np.any(raw < 0) or not np.isfinite(logs).all():
        raise FloatingPointError("冻结链评估产生非有限值或负似然")
    return {"indices": indices, "raw": raw, "logs": logs, "raw_mean": raw.mean(axis=1),
            "log_mean": logsumexp(logs, axis=1)-np.log(len(indices)),
            "raw_zero_draws": (raw == 0).sum(axis=1)}


def check_raw_mean(cached, evaluated, n_subjects, settings):
    import numpy as np
    value = np.asarray(cached, dtype=float)
    if value.shape not in ((n_subjects,), (n_subjects, 1)):
        raise ValueError(f"缓存留出似然形状{value.shape}与{n_subjects}名受试者不符")
    value = value.reshape(-1)
    if not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError(f"缓存留出似然含负值={int((value<0).sum())}、非有限={int((~np.isfinite(value)).sum())}")
    if not np.allclose(value, evaluated["raw_mean"], atol=0, rtol=settings["relative_tolerance"]):
        raise ValueError("冻结链重新计算的线性似然与原缓存不一致；不能将任意零值当作下溢放行")
    if not np.isfinite(evaluated["log_mean"]).all():
        raise FloatingPointError("稳定留出似然非有限")
    supported = value >= np.finfo(float).tiny
    if not np.allclose(np.log(value[supported]), evaluated["log_mean"][supported],
                       rtol=0, atol=settings["absolute_log_tolerance"]):
        raise ValueError("可表示范围内的原始与对数域留出似然不一致")
    return value


def z_diagnostics(data, feature_ids, settings):
    """描述所有大正/负z；只标记，不删除受试者或裁剪数据。"""
    import numpy as np
    absolute = np.abs(data)
    maxima = absolute.max(axis=0)
    order = np.argsort(maxima)[::-1][:settings["top_features"]]
    return {"subjects": len(data), "max_abs_z": float(absolute.max()),
            "subjects_beyond_warning": int(np.any(absolute > settings["z_warning_threshold"], axis=1).sum()),
            "warning_threshold": settings["z_warning_threshold"],
            "top_features": [{"feature_id": str(feature_ids[i]), "max_abs_z": float(maxima[i]),
                              "min_z": float(data[:, i].min()), "max_z": float(data[:, i].max())} for i in order]}


def training_floor_audit(engine, data, sequence, fractions, settings):
    """只评估保存的ML-EM序列一次；不是重跑训练，也不推断所有历史迭代的情况。"""
    import numpy as np
    result = evaluate_samples(engine, data, np.asarray(sequence)[:, :, None],
                              np.asarray(fractions)[:, None], settings)
    logs, raw = result["log_mean"], result["raw_mean"]
    floored = np.log(raw+1e-250)  # 固定官方训练实现的常数，不新增模型超参数。
    return {"scope": "saved_ML_EM_sequence_only", "n_train": len(raw),
            "linear_zero_subjects": int((raw == 0).sum()),
            "below_official_floor_subjects": int((logs < np.log(1e-250)).sum()),
            "official_floored_loglikelihood": float(floored.sum()),
            "stable_loglikelihood": float(logs.sum()),
            "training_not_refitted": True}


# ===== 四、仅官方分型/分期评估期间使用稳定比例；训练函数和随机流始终恢复 =====
def stable_subtyping(engine, data, sequence, fractions, n_samples):
    import numpy as np
    from scipy.special import logsumexp
    settings = getattr(engine, "_numerical_settings", numerical_settings({}))
    kernel = build_log_kernel(engine, settings["block_rows"])
    original_likelihood = engine._calculate_likelihood
    original_subtyping = getattr(engine, "_raw_subtyping", engine.subtype_and_stage_individuals)

    def ratios(sustain_data, this_sequence, this_f):
        stages, weighted = log_joint(kernel, sustain_data, this_sequence, this_f)
        log_total = logsumexp(weighted, axis=(1, 2))
        joint = normalize_log_joint(stages, this_f)
        # 原官方联合输出未乘f；单独保留其定义，不能用加权联合后验替换它。
        unweighted = normalize_log_joint(stages)
        return (float(log_total.sum()), np.ones(len(joint)), joint.sum(axis=2),
                np.squeeze(joint.sum(axis=1)), unweighted)

    engine._calculate_likelihood = ratios
    try:
        return original_subtyping(data, sequence, fractions, n_samples)
    finally:
        engine._calculate_likelihood = original_likelihood


def install_stable_subtyping(engine, settings):
    engine._numerical_settings = settings
    engine._raw_subtyping = engine.subtype_and_stage_individuals
    def evaluate(data, sequence, fractions, n_samples):
        return stable_subtyping(engine, data, sequence, fractions, n_samples)
    engine.subtype_and_stage_individuals = evaluate
