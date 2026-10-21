"""并行切换：科学身份隔离、原子回退、精确停止及真实官方缓存恢复。"""
import importlib.util
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest
import yaml
from sklearn.model_selection import KFold

from substain_sustain.config import load_config
from substain_sustain.model import _identity, _model_seed, _guard, _fold_worker, run_crossvalidate
from test_model import _synthetic_project

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("switch_parallel", ROOT / "scripts/switch_parallel.py")
switch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(switch)

pytestmark = pytest.mark.filterwarnings(
    "ignore:Conversion of an array with ndim > 0 to a scalar:DeprecationWarning:pySuStaIn")


# ===== 一、合成工作站夹具：保留完整的前置阶段和真实身份 =====
@pytest.fixture
def station(tmp_path):
    output = tmp_path / "runs/formal_test"
    output.mkdir(parents=True)
    synthetic = _synthetic_project(output)
    cfg = yaml.safe_load((ROOT / "config/analysis.yaml").read_text())
    cfg.update(synthetic)
    cfg.update(project_dir=str(tmp_path), output_dir=str(output))
    cfg["feature_selection"] = {"expected_model_features": 40, "exclude": [], "merge": []}
    cfg["model"]["folds"] = 10
    cfg["model"]["workers"] = 4
    (tmp_path / "config").mkdir()
    config = tmp_path / "config/local.38.yaml"
    config.write_text(yaml.safe_dump(cfg))
    cfg = load_config(config)
    (output / "execution").mkdir()
    snapshot_hash = switch.config_hash(cfg)
    (output / "execution" / f"config_{snapshot_hash}.json").write_bytes(switch.encoded(switch.snapshot(cfg)))
    for stage in ("ingest", "reference"):
        (output / "execution" / f"{stage}.json").write_bytes(switch.encoded(
            {"command": stage, "status": "complete", "config_sha256": snapshot_hash}))
    cv = output / "model/cv"
    cv.mkdir(parents=True)
    identity = _identity(cfg, output, [output / "reference/C3_primary_z.tsv"], "C3_crossvalidation")
    _guard(cv, identity)
    splits = [t.tolist() for _, t in KFold(n_splits=10, shuffle=True, random_state=2026).split(range(20))]
    folds = [{"fold": i, "indices": t, "subject_keys": [f"C3:{j:03d}" for j in t],
              "model_seeds": {"1": _model_seed(2026, i, 1)}} for i, t in enumerate(splits)]
    (cv / "folds.json").write_bytes(switch.encoded(folds))
    settings = switch.read_settings(ROOT / "config/parallel10.yaml")
    return cfg, settings, splits, identity


def file_state(output):
    return {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.rglob("*") if p.is_file()}


# ===== 二、仅资源参数可迁移，任何输入/算法/种子变化都拒绝 =====
def test_migration_keeps_raw_reference_and_pickle_bytes(station):
    cfg, settings, splits, identity = station
    output = Path(cfg["output_dir"])
    folder = output / "model/cv/fold_0"
    folder.mkdir()
    _guard(folder, {"parent": identity, "fold": 0, "indices": splits[0]})
    sentinel = folder / "synthetic_cached_model.pickle"
    sentinel.write_bytes(b"preserved synthetic bytes, never interpreted as a complete chain")
    protected = [sentinel, output / "reference/model.json", output / "reference/C3_primary_z.tsv"]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in protected}
    target = switch.target_config(cfg, settings)
    backup = Path(cfg["project_dir"]) / ".cache/switch"
    assert switch.migrate(target, settings, backup)["status"] == "complete"
    assert load_config(cfg["_config_file"])["model"]["workers"] == 10
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in protected} == before
    new = _identity(target, output, [output / "reference/C3_primary_z.tsv"], "C3_crossvalidation")
    _guard(output / "model/cv", new)
    _guard(folder, {"parent": new, "fold": 0, "indices": splits[0]})
    record = switch.read_json(output / "execution/ingest.json")
    assert record["status"] == "complete" and record["config_sha256"] == switch.config_hash(target)
    assert record["reused_from_config_sha256"] == switch.config_hash(cfg)
    assert len(list(backup.glob("*"))) >= 5


@pytest.mark.parametrize("change", ["seed", "input", "source", "split", "stage"])
def test_mismatch_rejected_before_mutation(station, change):
    cfg, settings, _, _ = station
    output = Path(cfg["output_dir"])
    target = switch.target_config(cfg, settings)
    if change == "seed":
        target["model"]["seed"] += 1
    elif change == "input":
        with (output / "reference/C3_primary_z.tsv").open("a") as stream:
            stream.write("\n")
    elif change == "source":
        path = output / "model/cv/manifest.json"
        value = switch.read_json(path)
        value["wrapper_source_sha256"]["model.py"] = "different algorithm"
        path.write_bytes(switch.encoded(value))
    elif change == "split":
        (output / "model/cv/folds.json").write_text("[]")
    else:
        (output / "model/fit").mkdir()
    before = file_state(output)
    with pytest.raises(ValueError):
        switch.migration_plan(target)
    assert file_state(output) == before


def test_insufficient_resources_refuses_ten(station, monkeypatch):
    from substain_sustain import model
    cfg, settings, _, _ = station
    target = switch.target_config(cfg, settings)
    monkeypatch.setattr(model, "_resources", lambda *a, **k: {"workers": 8})
    with pytest.raises(MemoryError, match="只允许8"):
        switch.resource_check(target, settings)


def test_write_failure_restores_original_files(station, monkeypatch):
    cfg, settings, _, _ = station
    target = switch.target_config(cfg, settings)
    output = Path(cfg["output_dir"])
    before = {p: p.read_bytes() for p in switch.migration_plan(target)}
    config = Path(cfg["_config_file"])
    before[config] = config.read_bytes()
    real = switch.atomic_bytes
    failed = False

    def fail_once(path, value):
        nonlocal failed
        if path == config and not failed:
            failed = True
            raise OSError("synthetic interrupted config write")
        real(path, value)

    monkeypatch.setattr(switch, "atomic_bytes", fail_once)
    with pytest.raises(OSError, match="synthetic"):
        switch.migrate(target, settings, Path(cfg["project_dir"]) / ".cache/rollback")
    assert {p: p.read_bytes() for p in before} == before
    assert not (output / "execution" / f"config_{switch.config_hash(target)}.json").exists()


# ===== 三、真实进程控制仅作用于指定进程树，预检失败会恢复暂停任务 =====
def test_real_process_tree_stop_and_unrelated_process_survives(tmp_path):
    child_file = tmp_path / "child.pid"
    code = ("import subprocess,sys,time,pathlib; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); "
            f"pathlib.Path({str(child_file)!r}).write_text(str(p.pid)); time.sleep(120)")
    parent = subprocess.Popen([sys.executable, "-c", code])
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        deadline = time.monotonic() + 5
        while not child_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        child = psutil.Process(int(child_file.read_text()))
        proc = psutil.Process(parent.pid)
        with pytest.raises(ValueError):
            switch.stop_processes([proc], {"terminate_timeout_seconds": 1, "kill_timeout_seconds": 1},
                                  verify=lambda: (_ for _ in ()).throw(ValueError("stage advanced")))
        assert proc.is_running() and proc.status() != psutil.STATUS_STOPPED
        assert child.is_running() and child.status() != psutil.STATUS_STOPPED
        switch.stop_processes([proc], {"terminate_timeout_seconds": 1, "kill_timeout_seconds": 1})
        assert not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
        assert unrelated.poll() is None
    finally:
        for process in (parent, unrelated):
            if process.poll() is None:
                process.kill()
            process.wait()


def test_matching_entry_uses_current_project_config_and_process_identity(station, monkeypatch):
    cfg, _, _, _ = station
    root = Path(cfg["project_dir"])
    # 临时假入口只等待，不运行分析；命令行形式与正式入口一致。
    (root / "substain_sustain.py").write_text("from pathlib import Path\nimport time\n"
                                              "Path('ready').touch()\ntime.sleep(120)\n")
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    process = subprocess.Popen([sys.executable, "-m", "substain_sustain", "all", "--config", cfg["_config_file"]],
                               cwd=root)
    try:
        deadline = time.monotonic() + 5
        while not (root / "ready").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        found = switch.matching_processes(cfg)
        assert [p.pid for p in found] == [process.pid]
        other = dict(cfg, _config_file=str(root / "config/other.yaml"), output_dir=str(root / "other-output"))
        assert switch.matching_processes(other) == []
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def test_resume_order_stops_on_failed_stage(station, monkeypatch):
    cfg, settings, _, _ = station
    calls = []
    monkeypatch.setattr(switch, "resource_check", lambda *a: {})

    def run(command, **kwargs):
        stage = command[3]
        calls.append(stage)
        if stage == "validate":
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(switch.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        switch.resume(cfg, settings)
    assert calls == ["crossvalidate", "fit", "validate"]


def test_main_releases_resume_lock_before_background_launch(station, monkeypatch):
    from substain_sustain.model import _lock
    cfg, _, _, _ = station
    calls = []
    monkeypatch.chdir(cfg["project_dir"])
    monkeypatch.setattr(sys, "argv", ["switch_parallel.py", "--config", cfg["_config_file"], "--apply"])
    monkeypatch.setattr(switch, "matching_processes", lambda cfg: [])

    def launch(command, **kwargs):
        # 用另一文件句柄申请同一锁；若父进程未释放，会和真实子进程一样失败。
        with _lock(Path(cfg["output_dir"]) / "execution/parallel_resume"):
            calls.append(command)
        return type("SyntheticProcess", (), {"pid": 99999999})()

    monkeypatch.setattr(switch.subprocess, "Popen", launch)
    switch.main()
    assert len(calls) == 1 and "--resume" in calls[0]
    assert load_config(cfg["_config_file"])["model"]["workers"] == 10
    assert len(list((Path(cfg["project_dir"]) / "logs").glob("*.latest_log"))) == 1


# ===== 四、固定上游真实计算：已有一折保持原文件，剩余九折改10并行完成 =====
@pytest.mark.integration
def test_real_official_partial_cv_resumes_with_ten_workers(station):
    cfg, settings, splits, identity = station
    output = Path(cfg["output_dir"])
    _fold_worker({"cfg": cfg, "output": str(output), "fold": 0, "splits": splits, "identity": identity})
    chain = output / "model/cv/fold_0/pickle_files/C3_fold0_subtype0.pickle"
    before = (chain.read_bytes(), chain.stat().st_mtime_ns)
    target = switch.target_config(cfg, settings)
    switch.migrate(target, settings, Path(cfg["project_dir"]) / ".cache/official_migration")
    assert run_crossvalidate(target, output)["n_subtypes"] == 1
    assert switch.read_json(output / "model/cv/resources.json")["workers"] == 10
    assert len(list((output / "model/cv").glob("fold_*/complete.json"))) == 10
    assert (chain.read_bytes(), chain.stat().st_mtime_ns) == before
