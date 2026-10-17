"""允许 python -m substain_sustain 调用统一命令行。"""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
