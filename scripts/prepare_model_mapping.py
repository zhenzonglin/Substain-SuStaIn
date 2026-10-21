#!/usr/bin/env python3
"""从配置的GM合并方案生成可追溯的GM119→GM19映射，不读取患者数据。"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from substain_sustain.config import load_config
from substain_sustain.feature_selection import model_dictionary, model_gm_mapping, selection_policy
from substain_sustain.ingest import atomic_tsv, load_mapping


# ===== 原始119标签仅改变宏区归属，原始GM20映射保持只读 =====
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    raw_mapping, raw = load_mapping(Path(cfg["paths"]["mapping"]))
    policy = selection_policy(cfg)
    dictionary = model_dictionary(raw, policy)
    mapping = model_gm_mapping(raw_mapping, dictionary, {"merged": policy["merge"]})
    target = Path(cfg["paths"]["model_mapping"])
    # 不能覆盖原始映射，也不能替换已有不同版本。
    if target.resolve() == Path(cfg["paths"]["mapping"]).resolve():
        raise ValueError("模型映射输出不能覆盖原始映射")
    if target.exists():
        pd.testing.assert_frame_equal(pd.read_csv(target, sep="\t").astype(str), mapping.astype(str), check_dtype=False)
    else:
        atomic_tsv(target, mapping)
    print(f"模型GM区域：{mapping.macro_index.nunique()}；互斥原生标签：{mapping.native_label.nunique()}；{target}")


if __name__ == "__main__":
    main()
