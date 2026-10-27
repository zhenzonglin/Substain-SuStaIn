"""官方 SuStaIn 输出留存与亚型比例加权的联合后验。

不修改官方 EM/MCMC。逐个后验样本先归一化，再平均；避免把平均似然
归一化误当成对模型不确定性的积分。
"""
from __future__ import annotations

from typing import Any

import numpy as np

OFFICIAL_NAMES = (
    "ml_subtype", "prob_ml_subtype", "ml_stage", "prob_ml_stage",
    "prob_subtype", "prob_stage", "prob_subtype_stage",
)


# ---------- 样本选择：烧入比例与展示抽样均明确记录 ----------
def sample_indices(n_iterations: int, n_samples: int, burn_in_fraction: float = 0.1) -> np.ndarray:
    """选择不重复、均匀分布的保留链样本。"""
    if n_iterations < 1 or n_samples < 1 or not 0 <= burn_in_fraction < 1:
        raise ValueError("MCMC 样本数必须为正，burn_in_fraction 必须属于 [0,1)。")
    first = int(n_iterations * burn_in_fraction)
    count = min(n_samples, n_iterations - first)
    return np.unique(np.rint(np.linspace(first, n_iterations - 1, count)).astype(int))


def normalize_weighted_likelihood(likelihood: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """L[N,stage,K] 乘 f[K] 后按每例归一化，拒绝无数值支持。"""
    likelihood = np.asarray(likelihood, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    if likelihood.ndim != 3 or fractions.shape != (likelihood.shape[2],):
        raise ValueError("似然与亚型比例维度不一致。")
    if not np.all(np.isfinite(likelihood)) or np.any(likelihood < 0):
        raise ValueError("似然必须是有限的非负数。")
    if not np.all(np.isfinite(fractions)) or np.any(fractions < 0) or fractions.sum() <= 0:
        raise ValueError("亚型比例无效。")
    # 先按最大值缩放，避免求和溢出；不伪造全零似然的支持。
    weighted = likelihood * fractions[None, None, :]
    scale = weighted.max(axis=(1, 2), keepdims=True)
    if np.any(scale <= 0):
        raise FloatingPointError("个体的所有阶段/亚型似然均为零；请检查极端 z 或参考尺度。")
    weighted = weighted / scale
    return weighted / weighted.sum(axis=(1, 2), keepdims=True)


# ---------- 一致的派生后验与报告标签 ----------
def summarize_joint(joint: np.ndarray) -> dict[str, np.ndarray]:
    """所有汇总都从同一个加权联合后验导出。"""
    joint = np.asarray(joint, dtype=float)
    if joint.ndim != 3 or not np.all(np.isfinite(joint)) or np.any(joint < 0):
        raise ValueError("联合后验必须是有限非负的 N×stage×K 数组。")
    if not np.allclose(joint.sum(axis=(1, 2)), 1, rtol=1e-10, atol=1e-12):
        raise ValueError("联合后验的每个受试者总概率必须为 1。")
    subtype = joint.sum(axis=1)
    stage = joint.sum(axis=2)
    rows = np.arange(len(joint))
    subtype_raw = subtype.argmax(axis=1)
    conditional = joint[rows, :, subtype_raw] / subtype[rows, subtype_raw, None]
    stage_map = conditional.argmax(axis=1)
    entropy = -np.sum(np.where(subtype > 0, subtype * np.log(np.maximum(subtype, np.finfo(float).tiny)), 0), axis=1)
    return {
        "posterior_joint": joint,
        "posterior_subtype": subtype,
        "posterior_stage": stage,
        "subtype_raw_zero_based": subtype_raw,
        "subtype": np.where(stage_map == 0, 0, subtype_raw + 1),
        "stage": stage_map,
        "stage_zero": stage_map == 0,
        "subtype_probability": subtype[rows, subtype_raw],
        "stage_conditional_probability": conditional[rows, stage_map],
        "stage_zero_probability": stage[:, 0],
        "subtype_entropy": entropy,
        "stage_expected": stage @ np.arange(joint.shape[1]),
    }


def infer_posterior(
    engine: Any, data: np.ndarray, samples_sequence: np.ndarray,
    samples_f: np.ndarray, *, n_samples: int = 1000,
    burn_in_fraction: float = 0.1, batch_size: int = 128,
) -> dict[str, np.ndarray]:
    """分批调用官方 API，同时保留原始输出与修正后的联合后验。

    official_* 使用与派生结果相同的抽样链及亚型排序，便于核查。
    官方原始全链运行结果仍由 pickle 单独保存。
    """
    from pySuStaIn.ZscoreSustain import ZScoreSustainData

    data = np.asarray(data, dtype=float)
    sequence = np.asarray(samples_sequence)
    fractions = np.asarray(samples_f, dtype=float)
    if data.ndim != 2 or not np.all(np.isfinite(data)) or len(data) == 0:
        raise ValueError("推断数据必须是非空、有限的二维矩阵。")
    if sequence.ndim != 3 or fractions.shape != (sequence.shape[0], sequence.shape[2]):
        raise ValueError("MCMC 序列与亚型比例维度不一致。")
    if batch_size < 1:
        raise ValueError("posterior_batch_size 必须为正整数。")
    indices = sample_indices(sequence.shape[2], n_samples, burn_in_fraction)
    seq = sequence[:, :, indices]
    frac = fractions[:, indices]
    # 官方分型按所用链的平均亚型比例排序，不逐样本重新换标签。
    order = np.argsort(frac.mean(axis=1))[::-1]
    n_events = seq.shape[1]
    official_batches: dict[str, list[np.ndarray]] = {k: [] for k in OFFICIAL_NAMES}
    stable_batches: dict[str, list[np.ndarray]] = {k: [] for k in OFFICIAL_NAMES}
    raw_valid_batches = []
    raw_available_batches = []
    joint_batches = []
    from .log_evaluation import (build_log_kernel, log_joint, normalize_log_joint,
                                 numerical_settings, stable_subtyping)
    evaluation = getattr(engine, "_numerical_settings", numerical_settings({}))
    log_kernel = build_log_kernel(engine, evaluation["block_rows"])
    for first in range(0, len(data), batch_size):
        batch = data[first:first + batch_size]
        sustain_data = ZScoreSustainData(batch, n_events)
        joint_mean = np.zeros((len(batch), n_events + 1, sequence.shape[0]))
        raw_valid = np.ones(len(batch), dtype=bool)
        raw_available = np.ones(len(batch), dtype=bool)
        for t in range(len(indices)):
            stages, weighted = log_joint(log_kernel, sustain_data, seq[order, :, t], frac[order, t])
            # 分母进入subnormal区即标记原线性输出缺乏数值支持；不填补它。
            raw_likelihood = np.exp(stages)
            raw_total = (raw_likelihood*frac[order, t][None, None, :]).sum(axis=(1, 2))
            raw_valid &= raw_total >= np.finfo(float).tiny
            raw_available &= raw_total > 0
            joint_t = normalize_log_joint(stages, frac[order, t])
            joint_mean += (joint_t - joint_mean) / (t + 1)
        shapes = {"ml_subtype": (1,), "prob_ml_subtype": (1,), "ml_stage": (1,), "prob_ml_stage": (1,),
                  "prob_subtype": (len(order),), "prob_stage": (n_events+1,), "prob_subtype_stage": (n_events+1, len(order))}
        raw_arrays = {key: np.full((len(batch), *shapes[key]), np.nan) for key in OFFICIAL_NAMES}
        stable_arrays = {key: np.empty((len(batch), *shapes[key])) for key in OFFICIAL_NAMES}
        # 所有可计算的官方原始输出均保留；低精度subnormal另标记，真正全零者原始字段为NaN。
        for original in (True, False):
            mask = raw_available if original else ~raw_valid
            if not mask.any():
                continue
            subset = batch[mask]
            # 固定官方版本的N=1 squeeze缺陷：临时重复后截回，不改变受试者数。
            official_input = np.repeat(subset, 2, axis=0) if len(subset) == 1 else subset
            target_data = ZScoreSustainData(official_input, n_events)
            if original:
                method = getattr(engine, "_raw_subtyping", engine.subtype_and_stage_individuals)
                result = method(target_data, seq, frac, len(indices))
            else:
                result = stable_subtyping(engine, target_data, seq, frac, len(indices))
            for key, value in zip(OFFICIAL_NAMES, result):
                value = np.asarray(value)[:len(subset)]
                if not np.isfinite(value).all():
                    raise FloatingPointError(f"{'原始' if original else '稳定'}官方{key}仍非有限")
                if original:
                    raw_arrays[key][mask] = value
                    stable_arrays[key][raw_valid] = raw_arrays[key][raw_valid]
                else:
                    stable_arrays[key][mask] = value
        for key in OFFICIAL_NAMES:
            official_batches[key].append(raw_arrays[key])
            stable_batches[key].append(stable_arrays[key])
        raw_valid_batches.append(raw_valid)
        raw_available_batches.append(raw_available)
        joint_batches.append(joint_mean)
    result = summarize_joint(np.concatenate(joint_batches, axis=0))
    result.update({f"official_{key}": np.concatenate(value, axis=0)
                   for key, value in official_batches.items()})
    result.update({f"official_stable_{key}": np.concatenate(value, axis=0)
                   for key, value in stable_batches.items()})
    result["official_raw_valid"] = np.concatenate(raw_valid_batches)
    result["official_raw_available"] = np.concatenate(raw_available_batches)
    result["subtype_order"] = order
    result["selected_indices"] = indices
    # 官方 marginal 已乘 f；这两个核对定位联合概率修正的范围。
    for key in ("subtype", "stage"):
        if not np.allclose(result[f"posterior_{key}"], result[f"official_stable_prob_{key}"], rtol=1e-8, atol=1e-10):
            raise AssertionError(f"派生后验与稳定官方 {key} 边际不一致。")
    return result
