"""官方起点的串并行等价、随机流、检查点与30物理核资源边界。"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pySuStaIn.ZscoreSustain import ZScoreSustainData

from substain_sustain.acceleration import execution_settings
from substain_sustain.model import _engine
from substain_sustain.startpoints import parallel_settings, cpu_topology

pytestmark = pytest.mark.filterwarnings(
    "ignore:Conversion of an array with ndim > 0 to a scalar:DeprecationWarning:pySuStaIn")


def small_settings():
    return {**parallel_settings({}), "enabled": True, "workers_per_fold": 2, "fit_workers": 2,
            "physical_core_limit": 2, "logical_cpu_limit": 2}


def pair(tmp_path, target_k=2, count=3):
    settings = {"z_vals": [1, 2], "z_max": 5, "startpoints": count,
                "_acceleration": execution_settings({"acceleration": {"backend": "row_blocks", "progress_enabled": True}})}
    cpu_ids = sorted(os.sched_getaffinity(0))[:2]
    parallel = {**settings, "_startpoint_parallel": small_settings(),
                "_startpoint_runtime": {"workers": len(cpu_ids), "core_cpu_groups": [[cpu] for cpu in cpu_ids]}}
    values = np.random.default_rng(1).normal(1, 1.3, (14, 3))
    features = pd.DataFrame({"label": ["A", "B", "C"]})
    a = _engine(values, features, settings, tmp_path/"serial", 20261027, 30, target_k)
    b = _engine(values, features, parallel, tmp_path/"parallel", 20261027, 30, target_k)
    return a, b, ZScoreSustainData(values, 6)


def assert_equal(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert np.array_equal(x, y, equal_nan=True)


@pytest.mark.parametrize("method", ["_find_ml", "_find_ml_split", "_find_ml_mixture"])
def test_official_startpoints_and_following_mcmc_bitwise_equal(tmp_path, method):
    serial, parallel, data = pair(tmp_path)
    args = [data]
    if method == "_find_ml_mixture":
        rng = np.random.default_rng(19)
        sequence = np.concatenate([serial._initialise_sequence(data, rng) for _ in range(2)])
        args.extend([sequence, np.array([0.6, 0.4])])
    expected = getattr(serial, method)(*args)
    actual = getattr(parallel, method)(*args)
    assert_equal(expected, actual)
    assert serial.global_rng.bit_generator.state == parallel.global_rng.bit_generator.state
    # 并列最优的官方mixture可能返回三维数组；链测试使用其中第一个最优点。
    sequence = expected[0] if expected[0].ndim == 2 else expected[0][:, :, 0]
    fractions = expected[1] if expected[1].ndim == 1 else expected[1][:, 0]
    expected_chain = serial._perform_mcmc(data, sequence, fractions, 30, 1, 0.01)
    actual_chain = parallel._perform_mcmc(data, sequence, fractions, 30, 1, 0.01)
    assert_equal(expected_chain, actual_chain)
    records = list((tmp_path/"parallel/startpoints").rglob("complete.json"))
    assert len(records) == 3
    for record in records:
        assert json.loads(record.read_text())["pid"] != os.getpid()
        progress = json.loads((record.parent/"progress.json").read_text())
        assert progress["em_sweeps_completed"] > 0


def test_completed_startpoints_reused_without_new_processes(tmp_path, monkeypatch):
    _, engine, data = pair(tmp_path)
    expected = engine._find_ml(data)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (tmp_path/"parallel/startpoints").rglob("result.npz")}
    _, restarted, _ = pair(tmp_path)
    import substain_sustain.startpoints as module
    monkeypatch.setattr(module, "ProcessPoolExecutor", lambda **kwargs: pytest.fail("完整起点不得重算"))
    assert_equal(expected, restarted._find_ml(data))
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    state = json.loads((tmp_path/"parallel/progress_pool.json").read_text())
    assert state["reused"] == [0, 1, 2] and state["running"] == [] and state["status"] == "complete"


def test_partial_batch_reuses_only_completed_points(tmp_path):
    _, engine, data = pair(tmp_path)
    expected = engine._find_ml(data)
    receipt = next((tmp_path/"parallel/startpoints").rglob("point_001/complete.json"))
    receipt.unlink()  # 合成中断：只有原子回执落盘后才算完成。
    _, restarted, _ = pair(tmp_path)
    assert_equal(expected, restarted._find_ml(data))
    state = json.loads((tmp_path/"parallel/progress_pool.json").read_text())
    assert state["reused"] == [0, 2] and sorted(state["completed"]) == [0, 1, 2]


def test_corrupt_checkpoint_fails_without_overwriting(tmp_path):
    _, engine, data = pair(tmp_path)
    engine._find_ml(data)
    checkpoint = next((tmp_path/"parallel/startpoints").rglob("result.npz"))
    checkpoint.write_bytes(b"synthetic corruption")
    _, restarted, _ = pair(tmp_path)
    with pytest.raises(ValueError, match="哈希"):
        restarted._find_ml(data)
    assert checkpoint.read_bytes() == b"synthetic corruption"


def test_changed_input_has_a_different_checkpoint_identity(tmp_path):
    _, engine, data = pair(tmp_path)
    engine._find_ml(data)
    old = set((tmp_path/"parallel/startpoints/k_2").iterdir())
    _, restarted, _ = pair(tmp_path)
    values = data.data.copy()
    values[0, 0] += 0.1
    restarted._find_ml(ZScoreSustainData(values, 6))
    assert len(set((tmp_path/"parallel/startpoints/k_2").iterdir()) - old) == 1


def test_topology_selects_30_real_cores_and_60_siblings(tmp_path):
    for cpu in range(64):
        folder = tmp_path/f"cpu{cpu}/topology"
        folder.mkdir(parents=True)
        (folder/"physical_package_id").write_text(str((cpu % 32)//16))
        (folder/"core_id").write_text(str(cpu % 16))
    topology = cpu_topology(parallel_settings({}), allowed=list(range(64)), sys_root=tmp_path, quota=False)
    assert topology["physical_cores"] == 30 and len(topology["cpu_ids"]) == 60
    assert len(topology["core_cpu_groups"]) == 30
    assert all(len(group) == 2 and group[1]-group[0] == 32 for group in topology["core_cpu_groups"])
    with pytest.raises(RuntimeError, match="不足"):
        cpu_topology(parallel_settings({}), allowed=list(range(32)), sys_root=tmp_path, quota=False)


@pytest.mark.parametrize("update", [{"workers_per_fold": 0}, {"physical_core_limit": 61},
                                   {"checkpoint_dir": "../startpoints"}, {"enabled": "yes"}])
def test_invalid_parallel_config_is_rejected(update):
    with pytest.raises(ValueError):
        parallel_settings({"startpoint_parallel": update})


def test_monitor_identifies_live_children_and_stale_parent(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
    import monitor
    _, engine, data = pair(tmp_path)
    engine._find_ml(data)
    text = monitor.startpoint_progress(tmp_path/"parallel", {}, [])
    assert "历史批次" in text[0] and "完成3/3" in text[0]
    state = json.loads((tmp_path/"parallel/progress_pool.json").read_text())
    shown = monitor.startpoint_progress(tmp_path/"parallel", {}, [{"pid": state["pid"], "created": state["process_created"]}])
    assert "历史批次" not in shown[0]


def test_38_features_114_events_official_startpoints_are_identical(tmp_path):
    values = np.random.default_rng(39).normal(1, 0.8, (8, 38))
    features = pd.DataFrame({"label": [f"F{i}" for i in range(38)]})
    execution = execution_settings({"acceleration": {"backend": "row_blocks", "progress_enabled": True}})
    settings = {"z_vals": [1, 2, 3], "z_max": 5, "startpoints": 2, "_acceleration": execution}
    cpus = sorted(os.sched_getaffinity(0))[:2]
    concurrent = {**settings, "_startpoint_parallel": small_settings(),
                  "_startpoint_runtime": {"workers": len(cpus), "core_cpu_groups": [[cpu] for cpu in cpus]}}
    serial = _engine(values, features, settings, tmp_path/"serial", 456, 10, 1)
    parallel = _engine(values, features, concurrent, tmp_path/"parallel", 456, 10, 1)
    data = ZScoreSustainData(values, 114)
    expected, actual = serial._find_ml(data), parallel._find_ml(data)
    assert_equal(expected, actual)
    assert expected[0].shape == (1, 114)
    assert serial.global_rng.bit_generator.state == parallel.global_rng.bit_generator.state


def test_child_failure_marks_batch_failed_and_stops_owned_workers(tmp_path):
    import psutil
    _, engine, data = pair(tmp_path)
    engine.pool.allocation["core_cpu_groups"] = [[99999999], [99999998]]
    with pytest.raises(OSError):
        engine._find_ml(data)
    state = json.loads((tmp_path/"parallel/progress_pool.json").read_text())
    assert state["status"] == "failed" and state["running"] == []
    children = psutil.Process().children(recursive=True)
    assert not any("spawn_main" in " ".join(p.cmdline()) for p in children if p.is_running())
