#!/usr/bin/env python3
"""立即保留已核验K=1及更高K缓存，切换到30物理核/60逻辑CPU的起点并行。"""
from pathlib import Path
from switch_acceleration import main

if __name__ == "__main__":
    main(default_settings=Path(__file__).resolve().parents[1]/"config/startpoints30.yaml",
         entry_script=Path(__file__).resolve())
