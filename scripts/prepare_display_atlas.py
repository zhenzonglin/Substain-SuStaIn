#!/usr/bin/env python3
"""仅将已分割的公开模板生成宏区展示图谱；输入与输出路径均来自配置。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from substain_sustain.assets import prepare_gm_display_atlas
from substain_sustain.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(json.dumps(prepare_gm_display_atlas(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
