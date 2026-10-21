#!/usr/bin/env python3
"""备份工作站配置并安全快进更新；保留本机设置，启用最终GM17+19方案。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import uuid

import yaml


# ===== Git操作仅针对当前仓库及analysis.yaml，不清理其他修改 =====
def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def upgrade(root: Path, source: Path, target: Path) -> dict:
    from substain_sustain.config import load_config

    root = root.resolve()
    if Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("请在SuStaIn仓库根目录执行")
    source = (root / source).resolve() if not source.is_absolute() else source.resolve()
    target = (root / target).resolve() if not target.is_absolute() else target.resolve()
    for path in (source, target):
        if path.parent != root / "config" or not path.name.startswith("local") or path.suffix != ".yaml":
            raise ValueError("源/目标必须是本仓库config/local*.yaml本机配置")
    analysis = root / "config/analysis.yaml"
    staged = git(root, "diff", "--cached", "--name-only", "--", "config/analysis.yaml")
    if staged:
        raise ValueError("analysis.yaml存在已暂存修改；为保留暂存决定，本脚本不处理此情况")
    if not source.is_file():
        raise FileNotFoundError("缺少当前本机配置：" + str(source))
    head = git(root, "rev-parse", "HEAD")
    before = analysis.read_bytes()
    effective = load_config(source)
    effective.pop("_config_file", None)
    if Path(effective["project_dir"]).resolve() != root:
        raise ValueError("当前配置project_dir不指向本仓库，不能自动迁移")

    # ===== 保存生效配置与原文件；时间加随机后缀，重复运行不会覆盖备份 =====
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    backup = root / ".cache/config_upgrades" / stamp
    backup.mkdir(parents=True)
    preserved = root / "config" / f"local.preserved_{stamp}.yaml"
    originals = []
    for path in dict.fromkeys((analysis, source, target)):
        if path.is_file():
            shutil.copy2(path, backup / path.name)
            originals.append({"path": str(path.relative_to(root)),
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (backup / "analysis.patch").write_text(git(root, "diff", "--binary", "--", "config/analysis.yaml") + "\n", encoding="utf-8")
    preserved.write_text("# 更新前实际生效的本机配置；原始文件与注释另有完整备份。\n" +
                         yaml.safe_dump(effective, allow_unicode=True, sort_keys=False), encoding="utf-8")
    receipt = {"status": "backed_up", "head_before": head, "backup_dir": str(backup),
               "preserved_config": str(preserved), "target_config": str(target), "original_files": originals}
    receipt_path = backup / "receipt.json"

    def save_receipt() -> None:
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")

    save_receipt()
    restored = False
    try:
        if analysis.read_bytes() != before:
            raise ValueError("analysis.yaml在备份期间发生变化，停止更新")
        git(root, "restore", "--source=HEAD", "--worktree", "--", "config/analysis.yaml")
        restored = True
        subprocess.run(["git", "-C", str(root), "pull", "--ff-only"], check=True)
        template = yaml.safe_load((root / "config/formal38_migrated.example.yaml").read_text(encoding="utf-8"))
        plan = template["feature_selection"]
        if (plan["expected_model_features"] != 38 or len(plan["merge"]) != 1 or
                plan["merge"][0]["source_features"] != ["gm_striatopallidal_ml", "gm_basal_forebrain_ml"]):
            raise ValueError("更新后的模板不符合最终GM17+GM19合并方案")
        # 最终模板覆盖分区/C3/结果目录，其他字段继承更新前的实际生效值。
        template["extends"] = preserved.name
        temporary = backup / "local38.generated.yaml"
        temporary.write_text("# 保留本机路径与参数；最终38维/C3迁移配置覆盖保存快照。\n" +
                             yaml.safe_dump(template, allow_unicode=True, sort_keys=False), encoding="utf-8")
        temporary.replace(target)
        receipt.update(status="complete", head_after=git(root, "rev-parse", "HEAD"))
        save_receipt()
        return receipt
    except BaseException as exc:
        # 快进未发生时恢复原工作区配置，避免更新失败后留下默认路径。
        if restored and git(root, "rev-parse", "HEAD") == head:
            analysis.write_bytes(before)
            receipt["analysis_restored_after_failure"] = True
        receipt.update(status="failed", error=str(exc))
        save_receipt()
        raise


# ===== 所有输入配置位置可从命令行指定；不启动任何分析或分割 =====
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("config/local.yaml"))
    parser.add_argument("--target-config", type=Path, default=Path("config/local.38.yaml"))
    args = parser.parse_args()
    result = upgrade(args.project_dir, args.config, args.target_config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
