"""公开模板图谱的来源核验与空间显示适配；不读取任何患者影像。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to

ASSET_KEYS = ("wmh_atlas", "gm_atlas", "background")


def file_sha256(path: Path) -> str:
    """流式哈希避免将模板或概率矩阵整体读入内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_asset_manifest(assets: dict, destination: Path, source: str = "public templates only") -> dict:
    """生成可移植来源清单，配置决定路径，清单仅提供内容校验值。"""
    result = {"schema_version": 1, "source": source, "assets": {}}
    for key in ASSET_KEYS:
        path = Path(assets[key])
        image = nib.load(str(path))
        result["assets"][key] = {"filename": path.name, "sha256": file_sha256(path),
                                  "shape": list(image.shape), "public_template": True}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def validate_assets(cfg: dict, feature_dictionary: pd.DataFrame | None = None) -> dict:
    """正式报告需要三个完整、可追溯的公开资源，缺失时明确停止。"""
    assets = cfg.get("assets", {})
    missing = [key for key in (*ASSET_KEYS, "manifest") if not assets.get(key) or not Path(assets[key]).is_file()]
    if missing:
        raise FileNotFoundError("Display assets are incomplete: " + ", ".join(missing))
    manifest = json.loads(Path(assets["manifest"]).read_text(encoding="utf-8"))
    if cfg.get("runtime", {}).get("profile", "formal") == "formal" and "synthetic" in str(manifest.get("source", "")).lower():
        raise ValueError("Synthetic display fixtures cannot be used for a formal report")
    records = manifest.get("assets", {})
    for key in ASSET_KEYS:
        expected = records.get(key, {}).get("sha256")
        if not expected or expected != file_sha256(Path(assets[key])):
            raise ValueError("Display asset SHA256 mismatch: " + key)
        image = nib.load(str(assets[key]))
        if len(image.shape) != 3 or not np.isfinite(image.affine).all():
            raise ValueError("Display asset must be a finite-affine 3D NIfTI: " + key)
        data = image.get_fdata(dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError("Display asset contains non-finite voxels: " + key)
        if key != "background":
            if not np.allclose(data, np.rint(data)) or data.min() < 0:
                raise ValueError("Atlas must contain nonnegative integer labels: " + key)
            if feature_dictionary is not None:
                modality = "WMH" if key == "wmh_atlas" else "GM"
                rows = feature_dictionary[feature_dictionary.modality.str.upper() == modality]
                required = set(rows.atlas_index.astype(int))
                absent = required - set(np.unique(data).astype(int))
                if absent:
                    raise ValueError(f"{key} lacks feature atlas labels: {sorted(absent)}")
    return manifest


def display_arrays(cfg: dict, modality: str) -> tuple[np.ndarray, np.ndarray]:
    """统一至公开背景网格，标签使用最近邻，按RAS方向展示。"""
    assets = cfg["assets"]
    background = nib.as_closest_canonical(nib.load(str(assets["background"])))
    atlas_key = "wmh_atlas" if modality.upper() == "WMH" else "gm_atlas"
    atlas = nib.load(str(assets[atlas_key]))
    if atlas.shape != background.shape or not np.allclose(atlas.affine, background.affine):
        atlas = resample_from_to(atlas, background, order=0)
    return background.get_fdata(dtype=np.float32), np.rint(atlas.get_fdata()).astype(np.int16)


def prepare_gm_display_atlas(cfg: dict) -> dict:
    """一次性把公开模板的DLMUSE原生标签映射为GM宏20区；不运行分割。"""
    assets = cfg["assets"]
    if assets.get("public_template_kind") != "public_template":
        raise ValueError("Set assets.public_template_kind=public_template; patient images are not accepted")
    template = Path(assets.get("source_background", assets.get("public_template", "")))
    segmentation = Path(assets.get("source_gm_segmentation", assets.get("native_segmentation", "")))
    mapping_path = Path(assets.get("macro_mapping", cfg.get("paths", {}).get("mapping", "")))
    target = Path(assets["gm_atlas"])
    for path in (template, segmentation, mapping_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_image = nib.load(str(segmentation))
    template_image = nib.load(str(template))
    if source_image.shape != template_image.shape or not np.allclose(source_image.affine, template_image.affine):
        raise ValueError("DLMUSE segmentation must be in the declared public template's native grid")
    mapping = pd.read_csv(mapping_path, sep="\t")
    if not {"native_label", "macro_index"}.issubset(mapping.columns) or mapping.native_label.duplicated().any():
        raise ValueError("Invalid or duplicate native-label mapping")
    native = np.rint(source_image.get_fdata()).astype(np.int16)
    macro = np.zeros(native.shape, dtype=np.uint8)
    for row in mapping.itertuples(index=False):
        macro[native == int(row.native_label)] = int(row.macro_index)
    present = sorted(int(value) for value in np.unique(macro) if value)
    if present != list(range(1, 21)):
        raise ValueError("Public-template GM atlas must contain labels 1..20: " + str(present))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = nib.load(str(target))
        if existing.shape != macro.shape or not np.allclose(existing.affine, source_image.affine) or not np.array_equal(existing.get_fdata(), macro):
            raise FileExistsError("Existing display atlas differs; preserve it: " + str(target))
    else:
        header = source_image.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(macro, source_image.affine, header), str(target))
    record = {"public_template": True, "source_template_sha256": file_sha256(template),
              "native_segmentation_sha256": file_sha256(segmentation),
              "macro_mapping_sha256": file_sha256(mapping_path), "output_sha256": file_sha256(target),
              "labels": present, "mapping_scope": "15 official cortical + 5 project-defined noncortical regions"}
    target.with_suffix(target.suffix + ".provenance.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # 仅复制两个公开资源；不同内容的已存在目标不得静默覆盖。
    for key, source_key in (("wmh_atlas", "source_wmh_atlas"), ("background", "source_background")):
        source, destination = Path(assets[source_key]), Path(assets[key])
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            if file_sha256(source) != file_sha256(destination):
                raise FileExistsError("Existing public resource differs: " + str(destination))
        else:
            shutil.copyfile(source, destination)
    manifest = write_asset_manifest(assets, Path(assets["manifest"]),
                                    assets.get("source_citation", "Public templates from the existing feature-extraction project"))
    manifest["assets"]["gm_atlas"]["provenance"] = record
    Path(assets["manifest"]).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    validate_assets(cfg)
    return {"status": "complete", "gm_atlas": record, "manifest": manifest}
