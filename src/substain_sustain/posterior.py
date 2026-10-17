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
    joint_batches = []
    for first in range(0, len(data), batch_size):
        batch = data[first:first + batch_size]
        # 固定官方版本对 N=1 的 squeeze/reshape 有缺陷。仅在推断调用内部
        # 临时重复一行并截回原行数；概率按受试者独立，不参与训练或计数。
        # 测试要求单例、整批和尾批得到同一结果；输出始终只包含原始患者。
        official_input = np.repeat(batch, 2, axis=0) if len(batch) == 1 else batch
        raw = engine.subtype_and_stage_individuals_newData(official_input, seq, frac, len(indices))
        for key, array in zip(OFFICIAL_NAMES, raw):
            array = np.asarray(array)[:len(batch)]
            if not np.all(np.isfinite(array)):
                raise FloatingPointError(f"官方 {key} 返回非有限值。")
            official_batches[key].append(array)
        sustain_data = ZScoreSustainData(batch, n_events)
        joint_mean = np.zeros((len(batch), n_events + 1, sequence.shape[0]))
        for t in range(len(indices)):
            likelihood = np.stack([
                engine._calculate_likelihood_stage(sustain_data, seq[s, :, t])
                for s in order
            ], axis=2)
            joint_t = normalize_weighted_likelihood(likelihood, frac[order, t])
            joint_mean += (joint_t - joint_mean) / (t + 1)
        joint_batches.append(joint_mean)
    result = summarize_joint(np.concatenate(joint_batches, axis=0))
    result.update({f"official_{key}": np.concatenate(value, axis=0)
                   for key, value in official_batches.items()})
    result["subtype_order"] = order
    result["selected_indices"] = indices
    # 官方 marginal 已乘 f；这两个核对定位联合概率修正的范围。
    for key in ("subtype", "stage"):
        if not np.allclose(result[f"posterior_{key}"], result[f"official_prob_{key}"], rtol=1e-8, atol=1e-10):
            raise AssertionError(f"派生后验与官方 {key} 边际不一致。")
    return result
