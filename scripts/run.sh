#!/usr/bin/env bash
# ===== 统一入口：工作站不依赖当前shell的激活环境 =====
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${project_dir}/.venv/bin/python" -m substain_sustain "$@"
