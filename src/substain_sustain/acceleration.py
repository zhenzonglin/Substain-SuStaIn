"""沿受试者分块计算固定官方似然，保持每例的算式和求和顺序。"""
from __future__ import annotations

import ast
import functools
import inspect
import math
import textwrap
from types import MethodType

# ===== 一、冻结源码核验；复用官方轨迹生成代码，不重写EM/MCMC =====
@functools.lru_cache(maxsize=1)
def _verified_code():
    from pySuStaIn.ZscoreSustain import ZscoreSustain
    from substain_sustain.model import _upstream_provenance
    _upstream_provenance()
    source = textwrap.dedent(inspect.getsource(ZscoreSustain._calculate_likelihood_stage))
    tree = ast.parse(source)
    function = tree.body[0]
    # 在已校验哈希的官方函数中保留轨迹、标准差和常数计算，替换最后的批量表达式。
    starts = [i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
              and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "x"]
    if len(starts) != 1:
        raise ValueError("官方似然结构与已验证版本不符")
    function.body = function.body[:starts[0]] + ast.parse(
        "return evaluate_blocks(sustainData.data, stage_value, sigmat, factor, coeff)").body
    function.name = "blocked_likelihood"
    ast.fix_missing_locations(tree)
    return compile(tree, "<verified-official-row-blocks>", "exec")


def build_blocked_kernel(engine, block_rows: int):
    import numpy as np  # 配置解析阶段不加载BLAS，先由CLI设置线程上限。
    from pySuStaIn.ZscoreSustain import ZscoreSustain
    from .model import _upstream_provenance
    if type(block_rows) is not int or block_rows < 1:
        raise ValueError("block_rows必须是正整数")
    _upstream_provenance()
    code = _verified_code()

    # ===== 二、仅沿受试者维度分块；保留每个受试者的运算及求和顺序 =====
    def evaluate_blocks(data, stage_value, sigmat, factor, coeff):
        result = np.empty((len(data), stage_value.shape[1]), dtype=np.float64)
        for start in range(0, len(data), block_rows):
            x = (data[start:start + block_rows, :, None] - stage_value) / sigmat[None, :, None]
            result[start:start + block_rows] = np.exp(
                coeff + np.sum(factor[None, :, None] - 0.5 * np.square(x), 1))
        return result

    namespace = {"np": np, "ZscoreSustain": ZscoreSustain, "evaluate_blocks": evaluate_blocks}
    exec(code, namespace)
    return MethodType(namespace["blocked_likelihood"], engine)


# ===== 三、执行配置不改科学参数和随机流 =====
def execution_settings(cfg: dict) -> dict:
    from pathlib import Path
    settings = {"backend": "official", "block_rows": 32, "progress_enabled": False,
                "heartbeat_seconds": 5.0, "progress_filename": "progress.json", "events_filename": "progress_events.jsonl"}
    supplied = cfg.get("acceleration", {})
    if set(supplied) - set(settings):
        raise ValueError("acceleration包含未知参数")
    settings.update(supplied)
    if settings["backend"] not in ("official", "row_blocks"):
        raise ValueError("acceleration.backend必须为official或row_blocks")
    if type(settings["block_rows"]) is not int or settings["block_rows"] <= 0:
        raise ValueError("acceleration.block_rows必须为正整数")
    if type(settings["progress_enabled"]) is not bool:
        raise ValueError("acceleration.progress_enabled必须为布尔值")
    if (type(settings["heartbeat_seconds"]) not in (int, float)
            or not math.isfinite(settings["heartbeat_seconds"]) or settings["heartbeat_seconds"] <= 0):
        raise ValueError("acceleration.heartbeat_seconds必须为正数")
    for key in ("progress_filename", "events_filename"):
        value = settings[key]
        if not isinstance(value, str) or Path(value).name != value or not value.startswith("progress") or "\\" in value:
            raise ValueError(f"acceleration.{key}必须为以progress开头的单一文件名")
    if settings["progress_filename"] == settings["events_filename"]:
        raise ValueError("进度快照和事件日志不能使用同一文件")
    return settings
