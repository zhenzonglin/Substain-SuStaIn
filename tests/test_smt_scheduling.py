"""SMT调度、内存拒绝和完整起点跨版本复用；全部使用合成数据。"""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from substain_sustain import model, startpoints
from test_startpoints import pair, assert_equal
from test_startpoint_migration import migration_station, create_caches
from test_cv_probe import snapshot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import startpoint_cache_migration as checkpoints  # noqa: E402
import switch_acceleration as switch  # noqa: E402
import check_smt_capacity as capacity_tool  # noqa: E402

pytestmark = pytest.mark.filterwarnings(
    "ignore:Conversion of an array with ndim > 0 to a scalar:DeprecationWarning:pySuStaIn")


# ===== 一、60槽位共享30个物理核；不能虚构60个物理核或重复绑定CPU =====
def topology():
    return {"physical_cores": 30, "logical_cpus": 60,
            "core_cpu_groups": [[i, i+32] for i in range(30)],
            "cpu_ids": list(range(30)) + list(range(32, 62)), "quota_cores": None}


def test_logical_slots_cover_60_cpus_and_spread_first_layer():
    physical = startpoints.worker_cpu_groups(topology(), {"worker_unit": "physical_core"})
    logical = startpoints.worker_cpu_groups(topology(), {"worker_unit": "logical_cpu"})
    assert len(physical) == 30 and len(logical) == 60
    assert logical[:30] == [[i] for i in range(30)]
    assert [cpu for group in logical for cpu in group] == topology()["cpu_ids"]
    folds = [logical[i*6:(i+1)*6] for i in range(10)]
    assert all(len({cpu for group in fold for cpu in group}) == 6 for fold in folds)


@pytest.mark.parametrize("change", [{"worker_unit": "bad"}, {"fit_workers": 61}])
def test_invalid_smt_configuration(change):
    with pytest.raises(ValueError):
        startpoints.parallel_settings({"startpoint_parallel": {"worker_unit": "logical_cpu", **change}})


@pytest.mark.parametrize("available_gib, expected_processes", [(256, 60), (116.21, 24)])
def test_smt_uses_60_only_when_memory_allows(tmp_path, monkeypatch, available_gib, expected_processes):
    cfg = yaml.safe_load((ROOT / "config/analysis.yaml").read_text())
    settings = yaml.safe_load((ROOT / "config/startpoints60.yaml").read_text())
    cfg = switch.target_config(cfg, settings)
    monkeypatch.setattr(startpoints, "cpu_topology", lambda _: topology())
    monkeypatch.setattr(model, "_available_memory", lambda: int(available_gib * 1024**3))
    resource = model._resources(cfg, tmp_path, 6792, 38, 100000, 6, 10, 10)
    assert resource["startpoint_workers_per_model"] == 6
    assert resource["max_compute_processes"] == expected_processes
    assert resource["worker_unit"] == "logical_cpu" and resource["compute_slots"] == 60
    assert resource["workers"] * resource["estimated_worker_bytes"] <= resource["memory_budget_bytes"]
    (tmp_path / "resources.json").write_text(json.dumps(resource, indent=2))


def test_declining_60_does_not_touch_inputs_or_caches(tmp_path, monkeypatch):
    cfg, _, _, _ = migration_station(tmp_path)
    saved = snapshot(Path(cfg["output_dir"]))
    monkeypatch.setattr(model, "_resources", lambda *args: {"workers": 4, "startpoint_workers_per_model": 6,
                       "estimated_worker_bytes": 20 * 1024**3, "memory_budget_bytes": 93 * 1024**3})
    with pytest.raises(MemoryError, match="尚未停止旧进程"):
        switch.parallel.resource_check(cfg, {"workers": 10, "require_requested_workers": True})
    assert snapshot(Path(cfg["output_dir"])) == saved


def test_capacity_tool_is_read_only_and_supports_old_settings(tmp_path, monkeypatch):
    cfg, _, _, _ = migration_station(tmp_path)
    cfg["model"].update(workers=10, folds=10, startpoints=25)
    settings = yaml.safe_load((ROOT / "config/startpoints60.yaml").read_text())
    saved_cfg, saved_files = copy.deepcopy(cfg), snapshot(Path(cfg["output_dir"]))
    def budget(target, *args):
        assert "worker_unit" not in target["startpoint_parallel"]
        assert target["startpoint_parallel"]["workers_per_fold"] == 6
        return {"cpu_topology": topology(), "startpoint_workers_per_model": 6,
                "estimated_worker_bytes": 20 * 1024**3, "memory_budget_bytes": 93 * 1024**3,
                "estimated_parent_bytes": 2 * 1024**3, "estimated_em_worker_bytes": 3 * 1024**3}
    monkeypatch.setattr(model, "_resources", budget)
    result = capacity_tool.capacity(cfg, settings)
    assert result["requested_compute_processes"] == 60 and result["allowed_compute_processes"] == 24
    assert result["full_request_allowed"] is False
    assert cfg == saved_cfg and snapshot(Path(cfg["output_dir"])) == saved_files


def test_unbuffered_resume_coordinator_is_recognized(tmp_path, monkeypatch):
    cfg, _, _, _ = migration_station(tmp_path)
    process = SimpleNamespace(pid=987654,
        cmdline=lambda: [sys.executable, "-u", "scripts/switch_startpoints.py", "--config", cfg["_config_file"], "--resume"],
        cwd=lambda: cfg["project_dir"], exe=lambda: sys.executable, uids=lambda: SimpleNamespace(real=os.getuid()))
    monkeypatch.setattr(switch.parallel, "matching_processes", lambda _: [])
    monkeypatch.setattr(switch.psutil, "process_iter", lambda: [process])
    assert switch.matching_processes(cfg) == [process]


# ===== 二、旧起点的科学代码必须与本次修改保持相同 =====
@pytest.mark.parametrize("filename,names", [
    ("model.py", ["_engine", "_model_seed"]),
    ("startpoints.py", ["_run_startpoint", "OrderedStartpointPool", "_validate_result", "load_checkpoint"]),
])
def test_training_functions_unchanged_since_7e38ac1(filename, names):
    previous = subprocess.check_output(["git", "show", f"7e38ac117ada349eae11cb0818f11ede4ed61270:src/substain_sustain/{filename}"], cwd=ROOT).decode()
    current = (ROOT / "src/substain_sustain" / filename).read_text()
    def nodes(source):
        return {node.name: ast.dump(node) for node in ast.parse(source).body if hasattr(node, "name")}
    old, new = nodes(previous), nodes(current)
    assert all(old[name] == new[name] for name in names)


def legacy_point_fixture(tmp_path, monkeypatch, incomplete=False):
    """先用未改的官方起点生成数据，再编码成已发布旧版内容寻址格式。"""
    _, engine, data = pair(tmp_path)
    expected = engine._find_ml(data)
    fold = tmp_path / "parallel"
    original = next((fold / "startpoints/k_2").glob("*/manifest.json")).parent
    identity = json.loads((original / "manifest.json").read_text())
    new_source = copy.deepcopy(identity["source"])
    legacy = yaml.safe_load((ROOT / "config/startpoints60.yaml").read_text())["legacy_wrapper_source_sha256"]
    identity["source"] = {key: legacy[key] for key in new_source}
    digest = checkpoints.batch_digest(identity)
    legacy_folder = original.with_name(digest)
    original.rename(legacy_folder)
    (legacy_folder / "manifest.json").write_text(json.dumps(identity))
    for receipt in legacy_folder.glob("point_*/complete.json"):
        record = json.loads(receipt.read_text())
        record["identity"]["batch"] = digest
        receipt.write_text(json.dumps(record))
    if incomplete:
        (legacy_folder / "point_001/complete.json").unlink()
    # 将独立测试批次放到正式扫描布局；恢复时相同引擎仍通过新批次哈希读取。
    cv = tmp_path / "output/model/cv"
    cv.mkdir(parents=True)
    fold.rename(cv / "fold_0")
    folder = cv / "fold_0/startpoints/k_2" / digest
    monkeypatch.setattr(model, "_features", lambda *args: range(3))
    target = {"output_dir": str(tmp_path / "output"), "model": {"z_vals": [1, 2]}}
    old = {"wrapper_source_sha256": legacy}
    new = {"wrapper_source_sha256": new_source, "upstream_source_sha256": identity["upstream"],
           "dependency_versions": identity["dependencies"]}
    return target, old, new, folder, expected, data


@pytest.mark.parametrize("incomplete", [False, True])
def test_migrated_complete_points_reused_in_new_pool(tmp_path, monkeypatch, incomplete):
    target, old, new, folder, expected, data = legacy_point_fixture(tmp_path, monkeypatch, incomplete)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.rglob("*") if p.is_file()}
    planned = checkpoints.checkpoint_plan(target, old, new)
    assert len(planned) == 1 and len(planned[0]["points"]) == (2 if incomplete else 3)
    assert checkpoints.install_checkpoints(planned) == (2 if incomplete else 3)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    _, engine, _ = pair(tmp_path / "restart")
    engine.pool.directory = folder.parents[2]
    engine.pool.settings["worker_unit"] = "logical_cpu"
    cpus = sorted(os.sched_getaffinity(0))[:3]
    engine.pool.allocation = {"workers": len(cpus), "core_cpu_groups": [[cpu] for cpu in cpus]}
    if not incomplete:
        monkeypatch.setattr(startpoints, "ProcessPoolExecutor", lambda **kwargs: pytest.fail("已完成的旧起点不应重新启动进程"))
    assert_equal(expected, engine._find_ml(data))
    state = json.loads((engine.pool.directory / "progress_pool.json").read_text())
    assert state["reused"] == ([0, 2] if incomplete else [0, 1, 2])
    assert checkpoints.checkpoint_plan(target, old, new) == []
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


def test_damaged_complete_point_rejected_before_copy(tmp_path, monkeypatch):
    target, old, new, folder, _, _ = legacy_point_fixture(tmp_path, monkeypatch)
    (folder / "point_000/result.npz").write_bytes(b"test corruption")
    with pytest.raises(ValueError, match="哈希"):
        checkpoints.checkpoint_plan(target, old, new)
    assert len(list(folder.parent.glob("*/manifest.json"))) == 1


def test_smt_cv_migration_preserves_model_caches_and_reference(tmp_path):
    cfg, settings, old, folds = migration_station(tmp_path, k_max=2)
    create_caches(cfg, old, folds, through_k=1)
    settings["startpoint_parallel"]["worker_unit"] = "logical_cpu"
    settings["startpoint_parallel"]["workers_per_fold"] = 2
    output = Path(cfg["output_dir"])
    caches = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (output / "model/cv").glob("fold_*/pickle_files/*.pickle")}
    reference, inputs = snapshot(output / "reference"), snapshot(output / "inputs")
    target = switch.target_config(cfg, settings)
    backup = tmp_path / "backup"
    backup.mkdir()
    transaction = switch.prepare_backup(target, settings, backup)
    switch.migrate(target, transaction, backup)
    model.run_crossvalidate(target, output)
    assert caches == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in caches}
    assert reference == snapshot(output / "reference") and inputs == snapshot(output / "inputs")
    resources = json.loads((output / "model/cv/resources.json").read_text())
    assert resources["worker_unit"] == "logical_cpu" and resources["max_compute_processes"] == 4
    model.run_fit(target, output)
    assert model.run_validate(target, output)["status"] == "complete"
