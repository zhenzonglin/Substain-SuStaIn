"""仅在已核验源码迁移时复制完整起点；旧批次、结果字节和修改时间保持不变。"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys


# ===== 一、只接受内容寻址的完整旧批次，不从进度文件推断完成 =====
def batch_digest(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def checkpoint_plan(target: dict, old: dict, new: dict) -> list[dict]:
    from substain_sustain.model import _hash, _features
    from substain_sustain.startpoints import load_checkpoint, parallel_settings

    cv = Path(target["output_dir"]) / "model/cv"
    name = parallel_settings(target)["checkpoint_dir"]
    events = len(_features(Path(target["output_dir"]), target)) * len(target["model"]["z_vals"])
    result = []
    for manifest in sorted(cv.glob(f"fold_*/{name}/k_*/*/manifest.json")):
        folder = manifest.parent
        identity = json.loads(manifest.read_text())
        if folder.name != batch_digest(identity):
            raise ValueError(f"起点批次身份哈希不符：{folder}")
        source = identity["source"]
        expected_old = {key: old["wrapper_source_sha256"].get(key) for key in source}
        expected_new = {key: new["wrapper_source_sha256"].get(key) for key in source}
        if set(source) != {"model.py", "acceleration.py", "progress.py", "startpoints.py"} or source not in (expected_old, expected_new):
            raise ValueError(f"起点源码不属于已核验的旧版/新版：{folder}")
        if identity["upstream"] != new["upstream_source_sha256"]:
            raise ValueError(f"起点官方源码不符：{folder}")
        if any(value != new["dependency_versions"].get(key) for key, value in identity["dependencies"].items()):
            raise ValueError(f"起点依赖版本不符：{folder}")
        if identity["python"] != sys.version:
            raise ValueError(f"起点Python构建版本不符：{folder}")
        upgraded = copy.deepcopy(identity)
        upgraded["source"] = expected_new
        new_digest = batch_digest(upgraded)
        destination = folder.with_name(new_digest)
        if destination.exists() and json.loads((destination / "manifest.json").read_text()) != upgraded:
            raise ValueError(f"目标起点批次冲突：{destination}")
        points = []
        for point in sorted(folder.glob("point_*")):
            receipt = point / "complete.json"
            if not receipt.exists():
                continue  # 没有原子完成回执的点留在原位，下次由官方入口重算。
            index = int(point.name.removeprefix("point_"))
            if not 0 <= index < len(identity["seeds"]):
                raise ValueError(f"起点编号超出子种子范围：{point}")
            before = _hash(receipt)
            load_checkpoint(point, {"batch": folder.name, "index": index}, events)
            record = json.loads(receipt.read_text())
            if _hash(receipt) != before:
                raise ValueError(f"检查时起点回执发生变化：{point}")
            dest_point = destination / point.name
            if (dest_point / "complete.json").exists():
                load_checkpoint(dest_point, {"batch": new_digest, "index": index}, events)
                continue
            if dest_point.exists():
                raise ValueError(f"目标起点含未完成文件，保留两份证据并停止：{dest_point}")
            record.update(identity={"batch": new_digest, "index": index},
                          migrated_from={"batch": folder.name, "receipt_sha256": before})
            points.append({"source": str(point), "destination": str(dest_point),
                           "sha256": _hash(point / "result.npz"), "receipt_sha256": before,
                           "mtime_ns": (point / "result.npz").stat().st_mtime_ns, "receipt": record})
        if destination != folder and points:
            result.append({"source": str(folder), "destination": str(destination),
                           "source_manifest_sha256": _hash(manifest), "identity": upgraded, "points": points})
    return result


# ===== 二、暂停确认后复制轻量起点文件；不移动或改写原始批次 =====
def install_checkpoints(items: list[dict]) -> int:
    from substain_sustain.model import _hash, _json, _guard

    copied = 0
    for item in items:
        original, destination = Path(item["source"]), Path(item["destination"])
        if _hash(original / "manifest.json") != item["source_manifest_sha256"]:
            raise ValueError("暂停后起点批次身份发生变化，拒绝迁移")
        destination.mkdir(parents=True, exist_ok=True)
        _guard(destination, item["identity"])
        for point in item["points"]:
            source, target = Path(point["source"]), Path(point["destination"])
            if (_hash(source / "result.npz") != point["sha256"]
                    or _hash(source / "complete.json") != point["receipt_sha256"]
                    or (source / "result.npz").stat().st_mtime_ns != point["mtime_ns"]):
                raise ValueError("暂停后完整起点被修改，拒绝迁移")
            if target.exists():
                raise ValueError(f"检查后目标起点出现并发写入：{target}")
            target.mkdir()
            try:
                # 单起点NPZ是轻量结果；不复制MCMC链，不创建可连带修改原件的硬链接。
                shutil.copy2(source / "result.npz", target / "result.npz")
                if _hash(target / "result.npz") != point["sha256"]:
                    raise ValueError("起点副本哈希不一致")
                _json(target / "complete.json", point["receipt"])
            except BaseException:
                # 留证据到独立名称，避免未完成的迁移副本挡住下一次重试。
                import uuid
                target.rename(target.with_name(f".migration_failed_{target.name}_{uuid.uuid4().hex}"))
                raise
            copied += 1
    return copied
