"""保留原始40维；显式排除WMH并在原始mL层合并GM，冻结为模型输入。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


# ===== 配置合同：不自动按C3/GB分布或拟合结果选择特征 =====
def selection_policy(cfg: dict) -> dict:
    policy = cfg.get("feature_selection", {"expected_model_features": 40, "exclude": []})
    count, exclusions = policy.get("expected_model_features"), policy.get("exclude")
    merges = policy.get("merge", [])
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 40:
        raise ValueError("feature_selection.expected_model_features必须为1..40的整数")
    if not isinstance(exclusions, list):
        raise ValueError("feature_selection.exclude必须是列表")
    names = []
    for item in exclusions:
        if not isinstance(item, dict) or set(item) != {"feature_id", "reason", "criterion"}:
            raise ValueError("每项排除必须明确feature_id、reason、criterion")
        if not isinstance(item["feature_id"], str) or not item["feature_id"]:
            raise ValueError("排除特征名称无效")
        if not isinstance(item["reason"], str) or not item["reason"].strip():
            raise ValueError("排除特征必须有书面理由")
        if item["criterion"] != "LS_primary_all_zero":
            raise ValueError("当前仅支持事先明确指定的LS_primary_all_zero排除依据")
        names.append(item["feature_id"])
    if not isinstance(merges, list):
        raise ValueError("feature_selection.merge必须是列表")
    sources, targets = [], []
    for item in merges:
        required = {"feature_id", "source_features", "label", "label_cn", "operation", "mapping_version", "reason"}
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError("合并项必须完整指定来源、目标、名称、版本、理由与operation")
        if item["operation"] != "sum_raw_ml":
            raise ValueError("GM合并只支持sum_raw_ml：先加原始体积，再log和LS标准化")
        src = item["source_features"]
        if (not isinstance(src, list) or len(src) < 2 or any(not isinstance(s, str) or not s.startswith("gm_") for s in src)
                or any(not isinstance(item[k], str) or not item[k].strip() for k in required - {"source_features"})):
            raise ValueError("合并必须指定至少两个不同GM原始特征及非空说明")
        if not item["feature_id"].startswith("gm_"):
            raise ValueError("合并目标必须是GM特征")
        sources.extend(src)
        targets.append(item["feature_id"])
    if (len(names) != len(set(names)) or len(sources) != len(set(sources)) or len(targets) != len(set(targets))
            or set(names) & set(sources) or set(targets) & (set(names) | set(sources))):
        raise ValueError("排除或合并的来源/目标不能重复、交叉或递归")
    if count != 40 - len(names) - len(sources) + len(targets):
        raise ValueError("模型特征数必须等于原始40维减去明确排除项及合并减少的维数")
    return {"expected_model_features": count, "exclude": exclusions, "merge": merges}


# ===== 确定性构造模型字典；GM19新编号和原GM20图谱编号同时保存 =====
def model_dictionary(raw: pd.DataFrame, policy: dict) -> pd.DataFrame:
    import pandas as pd
    ids = raw.feature_id.tolist()
    excluded = {item["feature_id"] for item in policy["exclude"]}
    by_source = {source: item for item in policy["merge"] for source in item["source_features"]}
    if len(ids) != 40 or len(set(ids)) != 40 or (excluded | set(by_source)) - set(ids):
        raise ValueError("排除/合并来源必须来自完整原始40维字典")
    if {item["feature_id"] for item in policy["merge"]} & set(ids):
        raise ValueError("合并目标不得覆盖原始特征")
    records, emitted = [], set()
    for row in raw.to_dict("records"):
        source = row["feature_id"]
        if source in excluded:
            continue
        item = by_source.get(source)
        if item and item["feature_id"] in emitted:
            continue
        source_ids = item["source_features"] if item else [source]
        source_rows = raw.set_index("feature_id").loc[source_ids]
        if item and not source_rows.modality.eq("GM").all():
            raise ValueError("合并来源必须全为GM")
        if policy["merge"]:
            row["source_feature_ids"] = json.dumps(source_ids)
            row["source_atlas_indices"] = json.dumps([int(v) for v in source_rows.atlas_index])
        if item:
            row.update(feature_id=item["feature_id"], label=item["label"])
            emitted.add(item["feature_id"])
        records.append(row)
    result = pd.DataFrame(records)
    if policy["merge"]:
        gm = result.modality.eq("GM")
        result.loc[gm, "atlas_index"] = list(range(1, int(gm.sum()) + 1))
    if len(result) != policy["expected_model_features"]:
        raise ValueError("模型字典维数与声明不符")
    return result.reset_index(drop=True)


def model_frame(raw: pd.DataFrame, selection: dict) -> pd.DataFrame:
    """严格从完整原始体积求和；不接受只提供旧z值或缺少任一来源。"""
    import numpy as np
    import pandas as pd
    result = raw.copy()
    for item in selection["policy"].get("merge", []):
        sources = item["source_features"]
        if not set(sources).issubset(result.columns):
            raise ValueError("GM合并缺少原始体积列：" + ",".join(sources))
        values = result[sources].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("GM合并来源必须是有限的正体积，缺失不能当零")
        if item["feature_id"] in result:
            raise ValueError("输入已含合并目标，禁止重复加工或混用旧模型表")
        result[item["feature_id"]] = values.sum(axis=1)
        if not np.isfinite(result[item["feature_id"]]).all():
            raise ValueError("GM合并体积溢出")
        result = result.drop(columns=sources)
    excluded = [item["feature_id"] for item in selection["excluded"]]
    result = result.drop(columns=excluded, errors="ignore")
    features = selection["model_feature_ids"]
    metadata = [c for c in result.columns if c not in features]
    return result[metadata + features]


def source_atlas_labels(row) -> list[int]:
    """新模型编号与原图谱编号分开；脑图按原始标签的并集赋同一个模型值。"""
    value = getattr(row, "source_atlas_indices", None)
    return [int(v) for v in json.loads(value)] if value is not None else [int(row.atlas_index)]


def model_gm_mapping(raw_mapping: pd.DataFrame, dictionary: pd.DataFrame, selection: dict) -> pd.DataFrame:
    """由原GM119映射构造GM19；保留每个原生标签的原GM20归属。"""
    import pandas as pd
    blocks = []
    merges = {item["feature_id"]: item for item in selection["merged"]}
    for row in dictionary[dictionary.modality == "GM"].itertuples(index=False):
        block = raw_mapping.loc[raw_mapping.macro_index.isin(source_atlas_labels(row))].copy()
        block["source_macro_index"] = block.macro_index
        block["source_macro_id"] = block.macro_id
        block["macro_index"] = int(row.atlas_index)
        block["macro_id"] = row.feature_id.removeprefix("gm_").removesuffix("_ml")
        block["macro_name_en"] = row.label
        if row.feature_id in merges:
            block["macro_name_cn"] = merges[row.feature_id]["label_cn"]
        if selection["merged"]:
            block["mapping_version"] = selection["merged"][0]["mapping_version"]
        blocks.append(block)
    result = pd.concat(blocks, ignore_index=True)
    if len(result) != 119 or result.native_label.nunique() != 119 or set(result.native_label) != set(raw_mapping.native_label):
        raise ValueError("模型GM映射必须完整且互斥地保留119个原生灰质标签")
    return result


def select_for_reference(raw: pd.DataFrame, ls, cfg: dict) -> tuple[pd.DataFrame, dict]:
    import numpy as np
    policy = selection_policy(cfg)
    ids = raw.feature_id.tolist()
    exclusions = [item["feature_id"] for item in policy["exclude"]]
    if len(ids) != 40 or set(exclusions) - set(ids):
        raise ValueError("排除项必须来自完整原始40维字典")
    records = []
    for item in policy["exclude"]:
        values = ls[item["feature_id"]].to_numpy(dtype=float)
        if not len(values) or not np.isfinite(values).all() or np.any(values != 0):
            raise ValueError(f"排除依据不成立：{item['feature_id']}在当前LS主年龄范围内并非全部为零")
        records.append({**item, "evidence_cohort": "LS", "ls_primary_n": len(values),
                        "ls_primary_nonzero_n": 0, "ls_primary_unique_values": 1})
    selected = model_dictionary(raw, policy)
    audit = {"schema_version": "ls-explicit-feature-selection-v2", "policy": policy,
             "raw_feature_ids": ids, "model_feature_ids": selected.feature_id.tolist(),
             "raw_dictionary": raw.to_dict("records"), "model_dictionary": selected.to_dict("records"),
             "merged": policy["merge"], "merge_before_transform": True,
             "excluded": records, "selection_uses_C3": False, "selection_uses_GB": False,
             "age_support": [cfg.get("reference", {}).get("age_min", 50),
                             cfg.get("reference", {}).get("age_max", 75)]}
    return selected, audit


# ===== 冻结字典的读取与一致性校验；本模块也随展示包离线分发 =====
def validate_selection(audit: dict) -> None:
    import pandas as pd
    if audit.get("schema_version") != "ls-explicit-feature-selection-v2":
        raise ValueError("未知特征排除记录版本")
    policy = selection_policy({"feature_selection": audit["policy"]})
    raw, active = audit["raw_feature_ids"], audit["model_feature_ids"]
    excluded = [item["feature_id"] for item in policy["exclude"]]
    if (len(raw) != 40 or len(set(raw)) != 40 or set(excluded) - set(raw)
            or len(active) != policy["expected_model_features"]):
        raise ValueError("冻结原始/模型特征顺序或排除清单不一致")
    original = pd.DataFrame(audit["raw_dictionary"])
    expected = model_dictionary(original, policy)
    if (original.feature_id.tolist() != raw or expected.feature_id.tolist() != active
            or expected.to_dict("records") != audit["model_dictionary"]
            or audit.get("merged") != policy["merge"] or audit.get("merge_before_transform") is not True):
        raise ValueError("冻结GM合并来源、操作、字典或新旧编号不一致")
    if audit.get("selection_uses_C3") is not False or audit.get("selection_uses_GB") is not False:
        raise ValueError("特征排除不得使用C3/GB作决策")
    records = audit["excluded"]
    if [record["feature_id"] for record in records] != excluded:
        raise ValueError("冻结特征排除记录缺失或顺序错误")
    for item, record in zip(policy["exclude"], records):
        if any(record.get(key) != value for key, value in item.items()):
            raise ValueError("冻结排除理由与配置不一致")
        if (record.get("evidence_cohort") != "LS" or record.get("ls_primary_n", 0) <= 0
                or record.get("ls_primary_nonzero_n") != 0 or record.get("ls_primary_unique_values") != 1):
            raise ValueError("冻结排除记录缺少LS全零证据")


def frozen_dictionary(output: Path, cfg: dict | None = None) -> pd.DataFrame:
    import pandas as pd
    output = Path(output)
    raw = pd.read_csv(output / "inputs/feature_dictionary.tsv", sep="\t")
    path = output / "reference/feature_selection.json"
    model = json.loads((output / "reference/model.json").read_text(encoding="utf-8"))
    if not path.is_file():
        if model.get("feature_selection") is not None or (cfg and (selection_policy(cfg)["exclude"] or selection_policy(cfg)["merge"])):
            raise ValueError("缺少冻结特征排除记录，不能复用旧40维结果")
        return raw  # 旧40维结果和独立合成夹具保留兼容路径。
    audit = json.loads(path.read_text(encoding="utf-8"))
    validate_selection(audit)
    if cfg is not None and selection_policy(cfg) != audit["policy"]:
        raise ValueError("当前特征排除配置与冻结LS参考不一致；请使用新输出目录")
    if model.get("feature_selection") != audit or model.get("feature_ids") != audit["model_feature_ids"]:
        raise ValueError("LS模型与冻结特征排除记录不一致")
    if raw.feature_id.tolist() != audit["raw_feature_ids"]:
        raise ValueError("原始特征字典已改变")
    selected_path = output / "reference/feature_dictionary.tsv"
    selected = pd.read_csv(selected_path, sep="\t")
    expected = model_dictionary(raw, audit["policy"])
    if not selected.equals(expected):
        raise ValueError("模型字典必须符合冻结的原始/合并特征顺序和atlas标签")
    manifest = json.loads((output / "reference/manifest.json").read_text(encoding="utf-8"))
    for name in ("feature_selection.json", "feature_dictionary.tsv"):
        actual = hashlib.sha256((output / "reference" / name).read_bytes()).hexdigest()
        if manifest.get("files", {}).get(name) != actual:
            raise ValueError("冻结特征文件哈希改变：" + name)
    return selected
