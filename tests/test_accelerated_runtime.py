"""正式分块及心跳：原官方数值不变，内部进度可读，旧缓存安全迁移。"""
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import psutil
import pytest
import yaml

from substain_sustain.acceleration import execution_settings
from substain_sustain.model import _engine, _settings, _identity, _fold_worker, run_crossvalidate, run_fit, run_validate
from pySuStaIn.ZscoreSustain import ZScoreSustainData
from test_model import _synthetic_project
from test_cv_probe import station, snapshot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import switch_acceleration as switch  # noqa: E402
import monitor  # noqa: E402

ACCEL = {"backend": "row_blocks", "block_rows": 32, "progress_enabled": True,
         "heartbeat_seconds": 0.05, "progress_filename": "progress.json", "events_filename": "progress_events.jsonl"}
pytestmark = pytest.mark.filterwarnings(
    "ignore:Conversion of an array with ndim > 0 to a scalar:DeprecationWarning:pySuStaIn")


def test_progress_records_real_em_startpoints_and_mcmc_without_changing_values(tmp_path):
    root = tmp_path/"formal"
    root.mkdir()
    cfg = _synthetic_project(root)
    cfg["model"]["startpoints"] = 2
    rng = np.random.default_rng(12)
    values = rng.normal(0.8, 0.9, (12, 3))
    import pandas as pd
    features = pd.read_csv(root/"inputs/feature_dictionary.tsv", sep="\t")
    plain = _engine(values, features, _settings(cfg, 3), tmp_path/"plain", 42, 30, 1)
    fast = _engine(values, features, _settings({**cfg, "acceleration": ACCEL}, 3), tmp_path/"fast", 42, 30, 1)
    data = ZScoreSustainData(values, 3)
    expected = plain._find_ml(data)
    actual = fast._find_ml(data)
    for a, b in zip(expected, actual):
        np.testing.assert_array_equal(a, b)
    expected_mcmc = plain._perform_mcmc(data, expected[0], expected[1], 30, 1, .01)
    actual_mcmc = fast._perform_mcmc(data, actual[0], actual[1], 30, 1, .01)
    for a, b in zip(expected_mcmc, actual_mcmc):
        np.testing.assert_array_equal(a, b)
    events = [json.loads(x) for x in (tmp_path/"fast/progress_events.jsonl").read_text().splitlines()]
    starts = [x for x in events if x["event"] == "startpoint_started"]
    assert [x["startpoint"] for x in starts] == [1, 2]
    sweeps = [x for x in events if x["event"] == "em_sweep_finished"]
    assert sweeps and all(x["em_iteration"] > 0 and np.isfinite(x["candidate_log_likelihood"]) for x in sweeps)
    assert all(x["event_rank"] == 3 and x["candidate"] == x["candidate_total"] for x in sweeps)
    final = json.loads((tmp_path/"fast/progress.json").read_text())
    assert final["mcmc_iteration_current"] == final["mcmc_iterations"] == 30
    assert final["process_created"] == psutil.Process().create_time()
    assert final["status"] == "operation_complete"


def test_monitor_marks_stale_or_wrong_pid_as_history(tmp_path):
    progress = {"pid": 123, "process_created": 10, "phase": "EM", "target_k": 2,
                "updated_utc": "2000-01-01T00:00:00+00:00", "startpoint": 4,
                "startpoints_total": 25, "em_iteration": 3, "event_rank": 50, "event_total": 114,
                "candidate": 2, "candidate_total": 20, "likelihood_calls_completed": 600}
    (tmp_path/"progress.json").write_text(json.dumps(progress))
    cfg = {"acceleration": ACCEL}
    assert "历史进度" in monitor.internal_progress(tmp_path, cfg, [{"pid": 123, "created": 11}])
    shown = monitor.internal_progress(tmp_path, cfg, [{"pid": 123, "created": 10}])
    assert "心跳延迟" in shown and "起点4/25" in shown and "事件50/114" in shown


@pytest.mark.parametrize("update", [{"progress_filename": "manifest.json"}, {"backend": "unknown"},
                                    {"heartbeat_seconds": float("nan")}, {"block_rows": 0}])
def test_execution_config_rejects_unsafe_values(update):
    with pytest.raises(ValueError):
        execution_settings({"acceleration": {**ACCEL, **update}})


# ===== 旧版来源夹具：只在合成数据中模拟旧身份；正式迁移须命中固定源码哈希 =====
def legacy_station(tmp_path):
    cfg, _ = station(tmp_path)
    cfg.pop("acceleration", None)
    (tmp_path/"config").mkdir()
    path = tmp_path/"config/local.38.yaml"
    cfg["_config_file"] = str(path)
    path.write_text(yaml.safe_dump(switch.parallel.snapshot(cfg)))
    settings = yaml.safe_load((ROOT/"config/acceleration_switch.yaml").read_text())
    settings["workers"] = cfg["model"]["workers"]
    settings["acceleration"] = ACCEL
    output = Path(cfg["output_dir"])
    manifest = output/"model/cv/manifest.json"
    identity = _identity(cfg, output, [output/"reference/C3_primary_z.tsv"], "C3_crossvalidation")
    identity.pop("acceleration_config")
    identity["wrapper_source_sha256"] = settings["legacy_wrapper_source_sha256"]
    manifest.write_text(json.dumps(identity))
    execution = output/"execution"
    execution.mkdir()
    digest = switch.parallel.config_hash(cfg)
    (execution/f"config_{digest}.json").write_bytes(switch.parallel.encoded(switch.parallel.snapshot(cfg)))
    for name in ("ingest", "reference"):
        (execution/f"{name}.json").write_text(json.dumps({"status": "complete", "config_sha256": digest}))
    return cfg, settings, identity


def test_legacy_cached_fold_survives_switch_and_real_resume(tmp_path):
    cfg, settings, identity = legacy_station(tmp_path)
    output = Path(cfg["output_dir"])
    folds = json.loads((output/"model/cv/folds.json").read_text())
    _fold_worker({"cfg": cfg, "output": str(output), "fold": 0, "splits": [f["indices"] for f in folds], "identity": identity})
    chain = output/"model/cv/fold_0/pickle_files/C3_fold0_subtype0.pickle"
    cached = chain.read_bytes(), chain.stat().st_mtime_ns
    immutable = snapshot(output/"reference")
    target = switch.target_config(cfg, settings)
    backup = tmp_path/"backup"
    backup.mkdir()
    check = switch.equivalence(target, settings, backup/"check")
    assert check["bitwise_equal"]
    transaction = switch.prepare_backup(target, settings, backup)
    switch.migrate(target, transaction, backup)
    assert cfg["model"] == target["model"] and cfg["runtime"] == target["runtime"]
    assert run_crossvalidate(target, output)["n_subtypes"] == 1
    assert (chain.read_bytes(), chain.stat().st_mtime_ns) == cached
    assert snapshot(output/"reference") == immutable
    run_fit(target, output)
    assert run_validate(target, output)["status"] == "complete"
    for path in (output/"model/cv/fold_1/progress_events.jsonl", output/"model/fit/primary/progress_events.jsonl"):
        events = [json.loads(line) for line in path.read_text().splitlines()]
        assert {x.get("tuning_pass") for x in events if x.get("phase") == "mcmc_tuning"} >= {1, 2, 3}
    assert json.loads((backup/"receipt.json").read_text())["caches"][0]["complete_shape"]


@pytest.mark.parametrize("change", ["seed", "workers", "source", "data", "fit"])
def test_migration_rejects_unapproved_changes_before_stop(tmp_path, change):
    cfg, settings, _ = legacy_station(tmp_path)
    target = switch.target_config(cfg, settings)
    output = Path(cfg["output_dir"])
    if change in ("seed", "workers"):
        target["model"][change] += 1
    elif change == "source":
        path = output/"model/cv/manifest.json"
        value = json.loads(path.read_text())
        value["wrapper_source_sha256"]["model.py"] = "unapproved"
        path.write_text(json.dumps(value))
    elif change == "data":
        path = output/"reference/C3_primary_z.tsv"
        path.write_text(path.read_text()+"\n")
    else:
        (output/"model/fit").mkdir()
    before = snapshot(output)
    with pytest.raises(ValueError):
        switch.plan(target, settings)
    assert snapshot(output) == before


def test_transaction_failure_restores_manifest_and_local_config(tmp_path, monkeypatch):
    cfg, settings, _ = legacy_station(tmp_path)
    target = switch.target_config(cfg, settings)
    backup = tmp_path/"backup"
    backup.mkdir()
    transaction = switch.prepare_backup(target, settings, backup)
    real = switch.parallel.atomic_bytes
    failed = False
    def fail_once(path, value):
        nonlocal failed
        if path == Path(cfg["_config_file"]) and not failed:
            failed = True
            raise OSError("synthetic write failure")
        real(path, value)
    monkeypatch.setattr(switch.parallel, "atomic_bytes", fail_once)
    with pytest.raises(OSError):
        switch.migrate(target, transaction, backup)
    for path, value in transaction[1].items():
        assert (path.read_bytes() if path.exists() else None) == value


def test_real_old_resume_coordinator_and_child_are_both_stopped(tmp_path):
    cfg, settings, _ = legacy_station(tmp_path)
    scripts = tmp_path/"scripts"
    scripts.mkdir()
    # 真正的独立进程树；只模拟等待，不执行任何真实队列任务。
    (tmp_path/"substain_sustain.py").write_text("from pathlib import Path\nimport time\nPath('ready').touch()\ntime.sleep(60)\n")
    launcher = scripts/"switch_parallel.py"
    lock_dir = Path(cfg["output_dir"])/"execution/parallel_resume"
    lock_dir.mkdir()
    launcher.write_text("import subprocess,sys,time,fcntl\n"
                        f"lock=open({str(lock_dir/'.lock')!r},'a+');fcntl.flock(lock,fcntl.LOCK_EX)\n"
                        "p=subprocess.Popen([sys.executable,'-m','substain_sustain',"
                        "'crossvalidate','--config',sys.argv[-1]])\ntime.sleep(60)\n")
    parent = subprocess.Popen([sys.executable, str(launcher), "--resume", "--config", cfg["_config_file"]], cwd=tmp_path)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
    try:
        deadline = time.monotonic()+5
        while not (tmp_path/"ready").exists() and time.monotonic() < deadline:
            time.sleep(.02)
        selected = switch.matching_processes(cfg)
        assert parent.pid in {p.pid for p in selected} and len(selected) == 2
        switch.parallel.stop_processes(selected, {**settings, "terminate_timeout_seconds": 1, "kill_timeout_seconds": 1})
        assert all(not p.is_running() or p.status() == psutil.STATUS_ZOMBIE for p in selected)
        from substain_sustain.model import _lock
        with _lock(lock_dir):
            pass
        assert unrelated.poll() is None
    finally:
        for process in (parent, unrelated):
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)


def test_pending_stop_in_io_is_cancelled_and_never_terminated():
    class BusyIO:
        pid = 999999999
        resumed = False
        def ppid(self):
            return 1
        def is_running(self):
            return True
        def status(self):
            return psutil.STATUS_DISK_SLEEP
        def suspend(self):
            pass
        def resume(self):
            self.resumed = True
        def children(self, recursive=False):
            return []
        def terminate(self):
            pytest.fail("尚未确认暂停时不得终止")
    process = BusyIO()
    with pytest.raises(RuntimeError, match="未确认"):
        switch.parallel.stop_processes([process], {"stop_ack_timeout_seconds": .01, "stop_poll_seconds": .001})
    assert process.resumed
