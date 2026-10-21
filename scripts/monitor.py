#!/usr/bin/env python3
"""只读监控：阶段记录、真实进程树、文件锁与资源采样；不加载大型后验。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import psutil
import yaml

from substain_sustain.config import load_config

# ===== 一、阶段合同；不修改分析代码、状态文件或研究配置 =====
STAGES = {
    "ingest": "数据导入", "reference": "LS参考", "crossvalidate": "C3交叉验证",
    "fit": "C3主模型/稳定性", "validate": "GB冻结验证", "report": "图表与报告",
    "bundle": "展示包生成", "verify-bundle": "展示包离线验收",
}


def read_json(path: Path) -> tuple[dict, str]:
    """状态写入瞬间可能不完整；读取失败显示未知，不能当作失败或完成。"""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return {}, "JSON不是对象"
        return value, ""
    except FileNotFoundError:
        return {}, ""
    except (OSError, ValueError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def timestamp(value: str | None) -> float:
    try:
        return datetime.fromisoformat(value or "").timestamp()
    except ValueError:
        return 0.0


def elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"


def argument(tokens: list[str], name: str, default: str = "") -> str:
    for i, token in enumerate(tokens):
        if token == name and i + 1 < len(tokens):
            return tokens[i + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return default


def resolved(value: str, cwd: str) -> Path:
    path = Path(value).expanduser()
    return (Path(cwd) / path).resolve() if not path.is_absolute() else path.resolve()


def process_command(tokens: list[str], cwd: str, cfg: dict, bundle: Path) -> str | None:
    """精确匹配Python模块/控制台入口及配置输出目录，排除监控自身和其他项目。"""
    start = -1
    if len(tokens) >= 3 and tokens[1:3] == ["-m", "substain_sustain"]:
        start = 3
    elif tokens and Path(tokens[0]).name == "substain-sustain":
        start = 1
    elif len(tokens) > 1 and Path(tokens[1]).name == "substain-sustain":
        start = 2
    if start < 0 or len(tokens) <= start:
        return None
    command = tokens[start]
    if command == "verify-bundle":
        candidate = argument(tokens, "--bundle")
        return command if candidate and resolved(candidate, cwd) == bundle else None
    if command not in STAGES and command != "all":
        return None
    path = resolved(argument(tokens, "--config", "config/analysis.yaml"), cwd)
    if path == Path(cfg["_config_file"]):
        return command
    try:
        other = load_config(path)
        if other["output_dir"] == cfg["output_dir"] and other["project_dir"] == cfg["project_dir"]:
            return command
    except (OSError, ValueError, KeyError, yaml.YAMLError):
        pass
    return None


def collect_processes(cfg: dict, bundle: Path) -> tuple[list[dict], list[dict], int]:
    """先定位实际分析入口，再按PPID收集子进程；PID重用由创建时间区分。"""
    table, roots, inaccessible = {}, [], 0
    for process in psutil.process_iter():
        try:
            with process.oneshot():
                if process.status() == psutil.STATUS_ZOMBIE:
                    continue
                row = {"pid": process.pid, "ppid": process.ppid(), "created": process.create_time(),
                       "name": process.name(), "state": process.status()}
                table[process.pid] = row
                tokens = process.cmdline()
                # 只对可能的入口查询cwd，避免全机扫描时大量权限错误。
                if "substain_sustain" not in tokens and not any(
                        Path(t).name == "substain-sustain" for t in tokens[:2]):
                    continue
                command = process_command(tokens, process.cwd(), cfg, bundle)
                if command:
                    roots.append({**row, "command": command})
        except psutil.AccessDenied:
            inaccessible += 1
        except (psutil.NoSuchProcess, OSError):
            continue
    selected = {row["pid"] for row in roots}
    while True:
        added = {pid for pid, row in table.items() if row["ppid"] in selected} - selected
        if not added:
            break
        selected.update(added)
    rows = []
    for pid in sorted(selected):
        try:
            process = psutil.Process(pid)
            with process.oneshot():
                if process.create_time() != table[pid]["created"]:
                    continue
                times = process.cpu_times()
                rows.append({**table[pid], "cpu_time": times.user + times.system,
                             "rss": process.memory_info().rss, "threads": process.num_threads()})
        except psutil.AccessDenied:
            inaccessible += 1
        except psutil.NoSuchProcess:
            continue
    live = {row["pid"] for row in rows}
    return rows, [row for row in roots if row["pid"] in live], inaccessible


def cpu_rates(previous: list[dict], current: list[dict], interval: float) -> None:
    old = {(row["pid"], row["created"]): row for row in previous}
    for row in current:
        before = old.get((row["pid"], row["created"]))
        row["cpu"] = (max(0.0, row["cpu_time"] - before["cpu_time"]) / interval * 100
                      if before and interval > 0 else None)


def lock_owners() -> tuple[dict, str]:
    """只读/proc/locks，不打开或申请项目锁，避免干扰正在运行的任务。"""
    owners = {}
    try:
        lines = Path("/proc/locks").read_text().splitlines()
        for line in lines:
            fields = line.split()
            if len(fields) < 6 or fields[1] != "FLOCK":
                continue  # 等待锁的记录有 -> 前缀，不代表持锁。
            major, minor, inode = fields[5].split(":")
            key = (int(major, 16), int(minor, 16), int(inode))
            owners.setdefault(key, set()).add(int(fields[4]))
        return owners, ""
    except (OSError, ValueError) as exc:
        return {}, str(exc)


def locked_by(path: Path, owners: dict, pids: set[int]) -> bool:
    try:
        stat = path.stat()
        key = (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino)
        return bool(owners.get(key, set()) & pids)
    except OSError:
        return False


# ===== 二、阶段和折状态：优先保留明确失败，中断不能显示成运行 =====
def stage_state(command: str, record: dict, roots: list[dict], identity: str,
                tolerance: float, read_error: str = "") -> tuple[str, str]:
    if read_error:
        return "未知", "状态文件暂不可读：" + read_error
    if record.get("config_sha256") and record["config_sha256"] != identity:
        return "配置不一致", "保存记录属于其他配置，请核对--config"
    started = timestamp(record.get("started_utc"))
    direct = any(root["command"] == command for root in roots)
    active = any((root["command"] == command or
                  (root["command"] == "all" and command not in {"bundle", "verify-bundle"}))
                 and started >= root["created"] - tolerance for root in roots)
    status = record.get("status")
    if direct and (not record or (started and not active)):
        return "启动中", "已找到本阶段进程，等待新状态记录"
    if status == "failed":
        return "失败", f"{record.get('error_type', '')}: {record.get('error', '')}"
    if status == "complete":
        duration = timestamp(record.get("finished_utc")) - started
        return "完成", "耗时 " + elapsed(duration)
    if status == "running":
        return ("运行中", "已运行 " + elapsed(time.time() - started)) if active else (
            "中断/进程退出", "残留running记录；未发现匹配的活跃分析入口")
    return "待运行", ""


def verify_state(bundle: Path, roots: list[dict]) -> tuple[str, str]:
    if any(root["command"] == "verify-bundle" for root in roots):
        return "运行中", "正在从展示包离线重画并验收"
    receipt, error = read_json(bundle.with_name(bundle.name + ".verification.json"))
    created, create_error = read_json(bundle.with_name(bundle.name + ".json"))
    if error or create_error:
        return "未知", error or create_error
    # 不每5秒重新哈希大包；对比两份正式回执和文件时间，监控不替代完整验收。
    try:
        stat = bundle.stat()
        verified = bundle.with_name(bundle.name + ".verification.json").stat()
        if (receipt.get("status") == "pass" and created.get("status") == "created"
                and created.get("sha256") == receipt.get("bundle_sha256")
                and created.get("bytes") == stat.st_size and verified.st_mtime >= stat.st_mtime):
            return "完成", "离线验收pass回执与生成回执一致"
    except FileNotFoundError:
        pass
    except OSError as exc:
        return "未知", str(exc)
    return "未验收", "旧版验收失败不写阶段状态；无进程/无有效回执时不能判定成功"


def model_progress(cfg: dict, rows: list[dict]) -> list[str]:
    output, model = Path(cfg["output_dir"]), cfg["model"]
    pids = {row["pid"] for row in rows}
    owners, error = lock_owners()
    lines = ["", "交叉验证（折号0起；缓存文件数不等于通过验收的模型数）"]
    complete = 0
    for fold in range(model["folds"]):
        folder = output / "model/cv" / f"fold_{fold}"
        marker, problem = read_json(folder / "complete.json")
        held = locked_by(folder / ".lock", owners, pids)
        state = "运行中" if held else "完成" if marker.get("status") == "complete" else (
            "未知" if problem else "已开始/待续" if folder.exists() else "待运行")
        complete += state == "完成"
        caches = sum((folder / "pickle_files" / f"C3_fold{fold}_subtype{k}.pickle").is_file()
                     for k in range(model["k_max"]))
        lines.append(f"  fold {fold:02d}  {state}  | 缓存 {caches}/{model['k_max']}")
    lines.append(f"  已完成折：{complete}/{model['folds']}")
    selection, _ = read_json(output / "model/cv/selection.json")
    chosen = selection.get("n_subtypes")
    if isinstance(chosen, int) and 1 <= chosen <= model["k_max"]:
        lines.append(f"C3拟合（选定K={chosen}；全部种子结束后仍需计算后验和稳定性）")
        locations = [("primary", "主种子", model["fit_iterations"])] + [
            (f"seed_{seed}", f"稳定性种子{seed}", model["stability_iterations"])
            for seed in model.get("stability_seeds", [])]
        for name, label, iterations in locations:
            folder = output / "model/fit" / name
            caches = sum((folder / "pickle_files" / f"C3_subtype{k}.pickle").is_file() for k in range(chosen))
            state = "运行中" if locked_by(folder / ".lock", owners, pids) else (
                "链缓存已写出" if caches == chosen else "已开始/待续" if folder.exists() else "待运行")
            lines.append(f"  {label}: {state} | 缓存 {caches}/{chosen} | 每模型MCMC {iterations:,}")
    if error:
        lines.append("文件锁不可读，折/种子的运行状态无法确认：" + error)
    return lines


# ===== 三、终端仪表盘：CPU按采样区间计算，100%代表一个逻辑核心 =====
def render(cfg: dict, settings: dict, rows: list[dict], roots: list[dict], inaccessible: int) -> str:
    bundle = Path(cfg["report"]["bundle_dir"]) / settings["bundle_filename"]
    snapshot = {k: v for k, v in cfg.items() if not k.startswith("_")}
    identity = hashlib.sha256(json.dumps(snapshot, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    memory = psutil.virtual_memory()
    known = [row["cpu"] for row in rows if row.get("cpu") is not None]
    cpu = f"{sum(known):.1f}%" + (" + 新进程采样中" if len(known) != len(rows) else "")
    rss = sum(row["rss"] for row in rows)
    lines = [f"SuStaIn只读监控 | {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}",
             f"配置：{cfg['_config_file']}", f"输出：{cfg['output_dir']}",
             f"分析进程：{len(rows)} | CPU合计 {cpu}（100%=1逻辑核） | RSS合计 {rss / 2**30:.2f} GiB",
             f"整机：{psutil.cpu_count()}逻辑核 | CPU {psutil.cpu_percent():.1f}% | "
             f"内存 {memory.percent:.1f}% | 可用/总计 {memory.available / 2**30:.2f}/{memory.total / 2**30:.2f} GiB",
             "RSS含共享页重复计数；Ctrl+C仅关闭本监控。", "", "阶段状态"]
    resources, resource_error = read_json(Path(cfg["output_dir"]) / "model/cv/resources.json")
    allocation = str(resources.get("workers", "尚无记录")) if not resource_error else "记录不可读"
    lines.insert(3, f"CV并行：配置上限 {cfg['model']['workers']} | 最近启动实际分配 {allocation} | "
                 "待处理折少于上限时活跃worker会减少")
    if not roots:
        lines.append("  当前未检测到该分析的活跃入口；完成/失败状态以已保存记录为准。")
    for command, label in STAGES.items():
        if command == "verify-bundle":
            state, detail = verify_state(bundle, roots)
        else:
            record, error = read_json(Path(cfg["output_dir"]) / "execution" / f"{command}.json")
            state, detail = stage_state(command, record, roots, identity,
                                        settings["process_start_tolerance_seconds"], error)
        # 错误只显示类型和简短原因，不输出原始受试者清单或整份日志。
        detail = " ".join(detail.split())
        lines.append(f"  [{state}] {command} / {label}" + (f" | {detail}" if detail else ""))
    lines.extend(model_progress(cfg, rows))
    lines.extend(["", "进程树明细（CPU降序；CPU为区间值，RSS为驻留内存）",
                  "  PID     PPID    CPU%       RSS GiB  线程  存活时间       状态 / 程序"])
    for row in sorted(rows, key=lambda r: r.get("cpu") or 0, reverse=True)[:settings["max_process_rows"]]:
        value = f"{row['cpu']:.1f}" if row.get("cpu") is not None else "采样中"
        lines.append(f"  {row['pid']:<7} {row['ppid']:<7} {value:<10} {row['rss'] / 2**30:<8.2f} "
                     f"{row['threads']:<5} {elapsed(time.time() - row['created'])}  {row['state']} / {row['name']}")
    if len(rows) > settings["max_process_rows"]:
        lines.append(f"  仅展示前{settings['max_process_rows']}个进程；资源合计包含全部{len(rows)}个。")
    if inaccessible:
        lines.append(f"全机扫描有{inaccessible}次权限不足；如任务属于其他用户，进程数据可能不完整。")
    lines.append("未读取MCMC链，不推算链内百分比；CPU为0也不能单独判定任务失败。")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="与正式启动命令相同的分析配置")
    parser.add_argument("--monitor-config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "config/monitor.yaml")
    parser.add_argument("--once", action="store_true", help="采样一次后退出，便于保存状态或排障")
    args = parser.parse_args()
    cfg = load_config(args.config)
    settings = yaml.safe_load(args.monitor_config.read_text(encoding="utf-8"))
    for name in ("refresh_seconds", "cpu_sample_seconds", "max_process_rows", "process_start_tolerance_seconds"):
        if isinstance(settings[name], bool) or not isinstance(settings[name], (int, float)) or settings[name] <= 0:
            parser.error(f"监控配置{name}必须为正数")
    if not isinstance(settings["max_process_rows"], int):
        parser.error("max_process_rows必须为整数")
    bundle = Path(cfg["report"]["bundle_dir"]) / settings["bundle_filename"]
    previous, _, _ = collect_processes(cfg, bundle)
    sampled = time.monotonic()
    psutil.cpu_percent()
    try:
        time.sleep(settings["cpu_sample_seconds"])
        while True:
            current, roots, inaccessible = collect_processes(cfg, bundle)
            now = time.monotonic()
            cpu_rates(previous, current, now - sampled)
            if not args.once and settings["clear_screen"] and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(render(cfg, settings, current, roots, inaccessible), flush=True)
            if args.once:
                break
            previous, sampled = current, now
            time.sleep(settings["refresh_seconds"])
    except KeyboardInterrupt:
        print("\n监控已退出，分析进程继续运行。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
