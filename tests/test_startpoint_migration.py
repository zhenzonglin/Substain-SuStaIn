"""旧版K=1缓存的真实官方续跑、严格迁移与资源预算测试。"""
import copy
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest
import yaml

from substain_sustain.model import (_engine, _settings, _features, _read_z, _identity, _guard,
                                  _model_seed, run_crossvalidate, run_fit, run_validate)
from test_cv_probe import station, snapshot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import switch_acceleration as switch  # noqa: E402

pytestmark = pytest.mark.filterwarnings(
    "ignore:Conversion of an array with ndim > 0 to a scalar:DeprecationWarning:pySuStaIn")


def migration_station(tmp_path, k_max=3):
    cfg, _ = station(tmp_path)
    cfg["model"].update(k_max=k_max, startpoints=2, workers=2)
    cfg["acceleration"].update(progress_enabled=True, heartbeat_seconds=0.1)
    cfg.pop("startpoint_parallel", None)
    (tmp_path/"config").mkdir()
    config = tmp_path/"config/local.38.yaml"
    cfg["_config_file"] = str(config)
    config.write_text(yaml.safe_dump(switch.parallel.snapshot(cfg)))
    settings = yaml.safe_load((ROOT/"config/startpoints30.yaml").read_text())
    settings["acceleration"] = copy.deepcopy(cfg["acceleration"])
    settings["startpoint_parallel"].update(workers_per_fold=2, fit_workers=2, physical_core_limit=4, logical_cpu_limit=4)
    settings["workers"] = cfg["model"]["workers"]
    output = Path(cfg["output_dir"])
    old = _identity(cfg, output, [output/"reference/C3_primary_z.tsv"], "C3_crossvalidation")
    old.pop("startpoint_parallel_config")
    old["wrapper_source_sha256"] = settings["legacy_wrapper_source_sha256"]
    cv = output/"model/cv"
    (cv/"manifest.json").write_text(json.dumps(old))
    folds = json.loads((cv/"folds.json").read_text())
    for fold in folds:
        fold["model_seeds"] = {str(k): _model_seed(cfg["model"]["seed"], fold["fold"], k) for k in range(1, k_max+1)}
    (cv/"folds.json").write_text(json.dumps(folds))
    execution = output/"execution"
    execution.mkdir()
    digest = switch.parallel.config_hash(cfg)
    (execution/f"config_{digest}.json").write_bytes(switch.parallel.encoded(switch.parallel.snapshot(cfg)))
    for name in ("ingest", "reference"):
        (execution/f"{name}.json").write_text(json.dumps({"status": "complete", "config_sha256": digest}))
    return cfg, settings, old, folds


def create_caches(cfg, old, folds, through_k=2):
    output = Path(cfg["output_dir"])
    features = _features(output, cfg)
    _, values = _read_z(output/"reference/C3_primary_z.tsv", features)
    all_folds = json.loads((output/"model/cv/folds.json").read_text())
    indices = [np.asarray(f["indices"]) for f in all_folds]
    for fold in folds:
        index = fold["fold"]
        directory = output/f"model/cv/fold_{index}"
        directory.mkdir()
        _guard(directory, {"parent": old, "fold": index, "indices": fold["indices"]})
        for k in range(1, through_k+1):
            engine = _engine(values, features, _settings(cfg, len(features)), directory,
                             _model_seed(cfg["model"]["seed"], index, k), cfg["model"]["cv_iterations"], k)
            engine.cross_validate_sustain_model(indices, select_fold=[index], plot=False)


def test_k1_k2_preserved_then_real_nested_cv_fit_and_frozen_gb(tmp_path):
    cfg, settings, old, folds = migration_station(tmp_path)
    create_caches(cfg, old, folds)
    output = Path(cfg["output_dir"])
    saved = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (output/"model/cv").glob("fold_*/pickle_files/*.pickle")}
    reference, inputs = snapshot(output/"reference"), snapshot(output/"inputs")
    target = switch.target_config(cfg, settings)
    assert target["model"] == cfg["model"] and target["runtime"] == cfg["runtime"]
    assert len(switch.audit_cv_caches(target, settings)) == 4
    backup = tmp_path/"backup"
    backup.mkdir()
    assert switch.parallel_equivalence(target, settings, backup/"probe")["bitwise_equal"]
    transaction = switch.prepare_backup(target, settings, backup)
    switch.migrate(target, transaction, backup)
    run_crossvalidate(target, output)
    assert saved == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in saved}
    assert reference == snapshot(output/"reference") and inputs == snapshot(output/"inputs")
    assert len(switch.audit_cv_caches(target, settings)) == 6
    resources = json.loads((output/"model/cv/resources.json").read_text())
    assert resources["max_compute_processes"] == 4
    for fold in folds:
        checkpoint_root = output/f"model/cv/fold_{fold['fold']}/startpoints/k_3"
        assert list(checkpoint_root.rglob("complete.json"))
    run_fit(target, output)
    assert run_validate(target, output)["status"] == "complete"
    assert saved == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in saved}


@pytest.mark.parametrize("changed", ["seed", "workers", "reference", "acceleration", "source", "input", "fit"])
def test_preflight_rejects_changes_outside_execution_scope(tmp_path, changed):
    cfg, settings, _, _ = migration_station(tmp_path)
    target = switch.target_config(cfg, settings)
    output = Path(cfg["output_dir"])
    if changed in ("seed", "workers"):
        target["model"][changed] += 1
    elif changed == "reference":
        target["reference"]["age_min"] += 1
    elif changed == "acceleration":
        target["acceleration"]["block_rows"] += 1
    elif changed == "source":
        path = output/"model/cv/manifest.json"
        value = json.loads(path.read_text())
        value["wrapper_source_sha256"]["model.py"] = "unapproved"
        path.write_text(json.dumps(value))
    elif changed == "input":
        path = output/"reference/C3_primary_z.tsv"
        path.write_text(path.read_text()+"\n")
    else:
        (output/"model/fit").mkdir()
    before = snapshot(output)
    with pytest.raises(ValueError):
        switch.plan(target, settings)
    assert before == snapshot(output)


def test_missing_or_bad_completed_cache_never_passes(tmp_path):
    cfg, settings, old, folds = migration_station(tmp_path, k_max=2)
    with pytest.raises(ValueError, match="缺少"):
        switch.audit_cv_caches(cfg, settings)
    create_caches(cfg, old, folds, through_k=1)
    path = Path(cfg["output_dir"])/"model/cv/fold_0/pickle_files/C3_fold0_subtype0.pickle"
    with path.open("rb") as stream:
        model = pickle.load(stream)
    model["samples_sequence"][0, 1, 0] = model["samples_sequence"][0, 0, 0]
    path.write_bytes(pickle.dumps(model))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="完整排列"):
        switch.audit_cv_caches(cfg, settings)
    assert path.read_bytes() == before


def test_30core_memory_budget_never_oversubscribes(tmp_path, monkeypatch):
    from substain_sustain import model, startpoints
    cfg = yaml.safe_load((ROOT/"config/analysis.yaml").read_text())
    settings = yaml.safe_load((ROOT/"config/startpoints30.yaml").read_text())
    cfg = switch.target_config(cfg, settings)
    topology = {"physical_cores": 30, "logical_cpus": 60, "core_cpu_groups": [[i, i+32] for i in range(30)],
                "cpu_ids": list(range(30))+list(range(32, 62)), "quota_cores": None}
    monkeypatch.setattr(startpoints, "cpu_topology", lambda _: topology)
    monkeypatch.setattr(model, "_available_memory", lambda: int(116.21 * 1024**3))
    resource = model._resources(cfg, tmp_path, 6792, 38, 100000, 6, 10, 10)
    assert resource["startpoint_workers_per_model"] == 3
    assert 1 <= resource["workers"] <= 10
    assert resource["max_compute_processes"] <= 30
    assert resource["estimated_worker_bytes"] * resource["workers"] <= resource["memory_budget_bytes"]
    (tmp_path/"resource_example.json").write_text(json.dumps(resource, indent=2))
