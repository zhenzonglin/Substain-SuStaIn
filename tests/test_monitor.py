"""只读监控验收：真实进程匹配、残留状态、文件锁和资源口径。"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("monitor", ROOT / "scripts/monitor.py")
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_stale_running_is_interrupted():
    record = {"status": "running", "started_utc": "2026-10-21T00:00:00+00:00"}
    assert monitor.stage_state("fit", record, [], "hash", 3)[0] == "中断/进程退出"


def test_matching_live_all_and_completed_stages():
    record = {"status": "running", "started_utc": "2026-10-21T00:00:00+00:00"}
    roots = [{"command": "all", "created": monitor.timestamp(record["started_utc"]) - 20}]
    assert monitor.stage_state("fit", record, roots, "hash", 3)[0] == "运行中"
    assert monitor.stage_state("bundle", record, roots, "hash", 3)[0] == "中断/进程退出"
    record.update(status="complete", finished_utc="2026-10-21T00:01:00+00:00")
    assert monitor.stage_state("fit", record, roots, "hash", 3) == ("完成", "耗时 0d 00:01:00")


def test_restart_does_not_validate_old_running():
    record = {"status": "running", "started_utc": "2026-10-21T00:00:00+00:00"}
    roots = [{"command": "all", "created": monitor.timestamp(record["started_utc"]) + 3600}]
    assert monitor.stage_state("fit", record, roots, "hash", 3)[0] == "中断/进程退出"
    roots[0]["command"] = "fit"
    assert monitor.stage_state("fit", record, roots, "hash", 3)[0] == "启动中"


def test_failure_read_error_and_wrong_config():
    record = {"status": "failed", "error_type": "ValueError", "error": "missing feature"}
    assert monitor.stage_state("fit", record, [], "hash", 3) == ("失败", "ValueError: missing feature")
    assert monitor.stage_state("fit", {}, [], "hash", 3, "partial JSON")[0] == "未知"
    record["config_sha256"] = "old"
    assert monitor.stage_state("fit", record, [], "hash", 3)[0] == "配置不一致"


def test_process_identity_requires_entry_and_config(tmp_path):
    cfg = {"_config_file": str(tmp_path / "formal.yaml")}
    bundle = tmp_path / "bundle.tar.gz"
    command = ["python", "-m", "substain_sustain", "all", "--config", "formal.yaml"]
    assert monitor.process_command(command, str(tmp_path), cfg, bundle) == "all"
    assert monitor.process_command(command[:4] + ["--config", "other.yaml"], str(tmp_path), cfg, bundle) is None
    assert monitor.process_command(["python", "scripts/monitor.py", "--config", "formal.yaml"],
                                   str(tmp_path), cfg, bundle) is None
    verify = ["python", "-m", "substain_sustain", "verify-bundle", "--bundle", str(bundle)]
    assert monitor.process_command(verify, str(tmp_path), cfg, bundle) == "verify-bundle"
    assert monitor.process_command(verify, str(tmp_path), cfg, tmp_path / "another.tar.gz") is None


def test_cpu_interval_and_pid_reuse():
    old = [{"pid": 10, "created": 1, "cpu_time": 2}]
    current = [{"pid": 10, "created": 1, "cpu_time": 6},
               {"pid": 10, "created": 2, "cpu_time": 1}]
    monitor.cpu_rates(old, current, 2)
    assert current[0]["cpu"] == 200
    assert current[1]["cpu"] is None


def test_partial_json_is_unknown(tmp_path):
    path = tmp_path / "fit.json"
    assert monitor.read_json(path) == ({}, "")
    path.write_text('{"status":')
    assert monitor.read_json(path)[1].startswith("JSONDecodeError")


def test_lock_owner_requires_live_matching_pid(tmp_path):
    fcntl = pytest.importorskip("fcntl")
    path = tmp_path / ".lock"
    with path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        owners, error = monitor.lock_owners()
        assert not error
        assert monitor.locked_by(path, owners, {os.getpid()})
        assert not monitor.locked_by(path, owners, {99999999})
    owners, _ = monitor.lock_owners()
    assert not monitor.locked_by(path, owners, {os.getpid()})


def test_verification_requires_matching_fresh_receipts(tmp_path):
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"test")
    created = bundle.with_name(bundle.name + ".json")
    checked = bundle.with_name(bundle.name + ".verification.json")
    created.write_text(json.dumps({"status": "created", "sha256": "new", "bytes": 4}))
    checked.write_text(json.dumps({"status": "pass", "bundle_sha256": "old"}))
    assert monitor.verify_state(bundle, [])[0] == "未验收"
    checked.write_text(json.dumps({"status": "pass", "bundle_sha256": "new"}))
    assert monitor.verify_state(bundle, [])[0] == "完成"
    assert monitor.verify_state(bundle, [{"command": "verify-bundle"}])[0] == "运行中"
    os.utime(bundle, (time.time() + 10, time.time() + 10))
    assert monitor.verify_state(bundle, [])[0] == "未验收"


def test_pickle_presence_is_not_fold_completion(tmp_path):
    folder = tmp_path / "model/cv/fold_0/pickle_files"
    folder.mkdir(parents=True)
    # 监控不能反序列化缓存；即使不完整文件也只统计为“缓存”，不标记完成。
    (folder / "C3_fold0_subtype0.pickle").write_bytes(b"incomplete")
    cfg = {"output_dir": str(tmp_path), "model": {"folds": 2, "k_max": 1}}
    lines = "\n".join(monitor.model_progress(cfg, []))
    assert "缓存 1/1" in lines and "已完成折：0/2" in lines
    (folder.parent / "complete.json").write_text(json.dumps({"status": "complete"}))
    assert "已完成折：1/2" in "\n".join(monitor.model_progress(cfg, []))


def test_live_process_and_child_are_counted_without_mutation(tmp_path):
    # 构造独立的无分析代码小进程，验证OS进程树匹配；不启动真实SuStaIn。
    package = tmp_path / "substain_sustain"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text(
        "import subprocess, sys\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2)'])\n"
        "p.wait()\n")
    cfg = {"_config_file": str(tmp_path / "formal.yaml")}
    with subprocess.Popen([sys.executable, "-m", "substain_sustain", "all", "--config", "formal.yaml"],
                          cwd=tmp_path) as process:
        try:
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                rows, roots, _ = monitor.collect_processes(cfg, tmp_path / "bundle.tar.gz")
                if len(rows) >= 2:
                    break
                time.sleep(0.02)
            assert any(row["pid"] == process.pid for row in roots)
            assert any(row["ppid"] == process.pid for row in rows)
            assert sum(row["rss"] for row in rows) > 0
            assert process.poll() is None
        finally:
            process.wait(timeout=5)
    assert monitor.collect_processes(cfg, tmp_path / "bundle.tar.gz")[0] == []
