"""配置加载：集中解析路径、覆盖关系和正式分析参数。"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

# ===== 配置合并与路径解析 =====
PATH_KEYS = {
    "mapping", "model_mapping", "derivatives", "participants", "metadata", "input_bundle", "manual_qc", "source_root",
    "wmh_atlas", "gm_atlas", "background", "manifest", "source_wmh_atlas", "source_background",
    "source_gm_segmentation", "template_python", "template_tool", "template_model_dir",
    "template_work_dir", "template_entry", "bundle_dir", "bundled_dir", "bundled_manifest",
}


def deep_merge(base: dict, updates: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read(path: Path, stack: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in stack:
        raise ValueError("配置extends存在循环引用")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("配置文件必须为YAML映射")
    parent = data.pop("extends", None)
    if parent:
        data = deep_merge(_read(path.parent / parent, (*stack, path)), data)
    return data


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    cfg = _read(path)
    project = Path(os.path.expandvars(str(cfg.get("project_dir", "..")))).expanduser()
    if not project.is_absolute():
        project = path.parent / project
    project = project.resolve()

    def resolve(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {k: resolve(v, k) for k, v in value.items()}
        if value is not None and key in PATH_KEYS:
            item = Path(os.path.expandvars(str(value))).expanduser()
            return str((project / item).resolve() if not item.is_absolute() else item.resolve())
        return value

    cfg = resolve(cfg)
    cfg["project_dir"] = str(project)
    out = Path(str(cfg.get("output_dir", "runs/formal_v1"))).expanduser()
    cfg["output_dir"] = str((project / out).resolve() if not out.is_absolute() else out.resolve())
    cfg["_config_file"] = str(path)
    validate_config(cfg)
    return cfg


# ===== 参数校验：不默默纠正或降级研究规格 =====
def validate_config(cfg: dict) -> None:
    from .feature_selection import selection_policy
    selection_policy(cfg)
    if set(cfg.get("cohorts", {})) != {"LS", "C3", "GB"}:
        raise ValueError("cohorts必须恰含LS、C3、GB")
    ref = cfg["reference"]
    if not 0 <= ref["age_min"] < ref["age_max"] <= 120:
        raise ValueError("参考年龄范围无效")
    mod = cfg["model"]
    for key in ("k_max", "folds", "startpoints", "cv_iterations", "fit_iterations",
                "stability_iterations", "posterior_samples", "display_samples",
                "posterior_batch_size", "workers"):
        if isinstance(mod[key], bool) or not isinstance(mod[key], int) or mod[key] <= 0:
            raise ValueError(f"model.{key}必须是正整数")
    if not 0 <= mod["burn_in_fraction"] < 1:
        raise ValueError("burn_in_fraction必须在[0,1)")
    thresholds = mod["z_vals"]
    if not thresholds or thresholds != sorted(set(thresholds)) or min(thresholds) <= 0:
        raise ValueError("z_vals必须为严格递增的正值")
    if mod["z_max"] <= max(thresholds):
        raise ValueError("z_max必须大于最高事件阈值")
    if not 0 < cfg["runtime"]["max_memory_fraction"] <= 1:
        raise ValueError("max_memory_fraction必须在(0,1]")
    if cfg["ingest"].get("manual_qc_required"):
        raise ValueError("当前已批准规格为自动QC入模，人工QC仅记录")
