"""由保存的数值源重建SuStaIn结果：静态HTML、出版图和逐图数据表。"""
from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .assets import display_arrays, file_sha256, validate_assets
from .feature_selection import frozen_dictionary, source_atlas_labels

DEFAULTS = {
    "formats": ["png", "svg", "tiff"], "raster_dpi": 600, "preview_dpi": 160,
    "figure_width": 10.0, "figure_height": 6.0, "font_size": 10,
    "max_posterior_draws": 1000, "stage_points": [0.25, 0.5, 0.75],
    "slice_fractions": [0.25, 0.4, 0.55, 0.7], "wmh_cmap": "YlOrRd",
    "gm_cmap": "magma", "pvd_cmap": "viridis", "distribution_cmap": "Blues",
    "cohort_colors": {"LS": "#5A7C65", "C3": "#32699B", "GB": "#B85F45"},
    "strict": True, "credible_interval": 0.95,
}

# -------- 文件合同与透明完成状态 --------
REQUIRED_FILES = (
    "inputs/feature_dictionary.tsv", "inputs/LS.tsv", "inputs/C3.tsv", "inputs/GB.tsv",
    "reference/model.json", "reference/diagnostics.tsv", "reference/LS_z.tsv", "reference/LS_oof_z.tsv",
    "reference/C3_z.tsv", "reference/GB_z.tsv", "reference/C3_primary_z.tsv", "reference/GB_primary_z.tsv",
    "model/cv/cvic.tsv", "model/cv/fold_loglikelihood.tsv", "model/cv/selection.json",
    "model/fit/summary.json", "model/fit/selected_model.npz", "model/fit/stability.tsv",
    "model/fit/assignments_C3_primary.tsv", "model/fit/assignments_C3_all.tsv",
    "model/fit/posterior_C3_primary.npz", "validation/posterior_GB_primary.npz",
    "validation/assignments_GB_primary.tsv", "validation/assignments_GB_all.tsv", "validation/summary.json",
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype={"participant_id": str, "subject_key": str})


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _reference_settings(cfg: dict, output: Path) -> dict:
    """报告使用实际冻结参考参数；样式可变，科学配置不可改写历史结果。"""
    model = _json(output / "reference/model.json")
    saved = model.get("settings")
    if not isinstance(saved, dict) or not {"age_min", "age_max"}.issubset(saved):
        raise ValueError("Saved reference model lacks its frozen scientific settings")
    current = cfg.get("reference", {})
    mismatched = sorted(key for key in set(saved) | set(current) if saved.get(key) != current.get(key))
    if mismatched:
        raise ValueError("Reference scientific configuration differs from the fitted model: " + ", ".join(mismatched))
    return saved


def _synthetic(cfg: dict) -> bool:
    return str(cfg.get("runtime", {}).get("profile", "formal")).startswith("synthetic")


def _status_page(folder: Path, missing: list[str], synthetic: bool) -> None:
    title = "TECHNICAL DEMONSTRATION — SYNTHETIC DATA" if synthetic else "FORMAL REPORT INCOMPLETE"
    content = "<h1>" + title + "</h1><p>No complete scientific result is claimed.</p><ul>"
    content += "".join("<li>" + html.escape(item) + "</li>" for item in missing) + "</ul>"
    (folder / "index.html").write_text("<!doctype html><meta charset='utf-8'>" + content, encoding="utf-8")


class ReportWriter:
    """统一导出每幅图的数值来源与三种格式，不依赖网络字体或外部脚本。"""
    def __init__(self, folder: Path, settings: dict):
        self.folder, self.settings, self.items = folder, settings, []
        (folder / "figures").mkdir(parents=True, exist_ok=True)
        (folder / "tables").mkdir(parents=True, exist_ok=True)

    def figure(self, name: str, title: str, figure, table: pd.DataFrame, note: str = "") -> None:
        if table.empty:
            raise ValueError("Cannot present an empty result as completed: " + name)
        table_path = self.folder / "tables" / (name + ".tsv")
        table.to_csv(table_path, sep="\t", index=False, float_format="%.10g")
        displayed_title = ("SYNTHETIC DATA — " if self.settings.get("_synthetic", False) else "") + title
        figure.suptitle(displayed_title, fontsize=self.settings["font_size"] + 2)
        if not getattr(figure, "_sustain_layout_ready", False):
            figure.tight_layout(rect=(0, 0, 1, 0.95))
        formats = list(dict.fromkeys(["png", *self.settings["formats"]]))
        files = []
        for extension in formats:
            if extension not in {"png", "svg", "tiff"}:
                raise ValueError("Report formats must be png, svg, or tiff")
            path = self.folder / "figures" / (name + "." + extension)
            kwargs = {"dpi": self.settings["preview_dpi"] if extension == "png" else self.settings["raster_dpi"]}
            if extension == "tiff":
                kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
            elif extension == "svg":
                kwargs["metadata"] = {"Date": None}
            figure.savefig(path, **kwargs)
            if extension == "tiff":
                from PIL import Image
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                    rgb.save(path, dpi=(self.settings["raster_dpi"], self.settings["raster_dpi"]), compression="tiff_lzw")
                    rgb.close()
            files.append(path.relative_to(self.folder).as_posix())
        plt.close(figure)
        self.items.append({"name": name, "title": title, "note": note, "files": files,
                           "table": table_path.relative_to(self.folder).as_posix(), "rows": len(table)})

    def axes(self, nrows: int = 1, ncols: int = 1, height: float = 1.0):
        return plt.subplots(nrows, ncols, squeeze=False,
                            figsize=(self.settings["figure_width"], self.settings["figure_height"] * height))


# -------- 纳入流程、健康参考与模型数选择 --------
def _plot_flow(writer: ReportWriter, output: Path) -> None:
    rows = []
    for cohort in ("LS", "C3", "GB"):
        audit = output / "inputs" / (cohort + "_audit.tsv")
        if audit.is_file():
            rows.append({"cohort": cohort, "step": "Source roster", "n": len(_read(audit))})
        stages = [("Imported", output / "inputs" / (cohort + ".tsv")),
                  ("Reference transformed", output / "reference" / (cohort + "_z.tsv")),
                  ("Primary age support", output / "reference" / (cohort + "_primary_z.tsv"))]
        for label, path in stages:
            if path.is_file():
                rows.append({"cohort": cohort, "step": label, "n": len(_read(path))})
        assignment = output / ("validation/assignments_GB_primary.tsv" if cohort == "GB" else "model/fit/assignments_C3_primary.tsv")
        if cohort != "LS" and assignment.is_file():
            rows.append({"cohort": cohort, "step": "Assigned primary", "n": len(_read(assignment))})
    table = pd.DataFrame(rows)
    fig, axes = writer.axes(1, 3)
    for ax, cohort in zip(axes.flat, ("LS", "C3", "GB")):
        frame = table[table.cohort == cohort]
        ax.barh(frame.step, frame.n, color=writer.settings["cohort_colors"][cohort])
        ax.invert_yaxis()
        for i, n in enumerate(frame.n):
            ax.text(n, i, f" {n:,}", va="center")
        ax.set(title=cohort, xlabel="Participants")
        ax.margins(x=.2)
    writer.figure("01_cohort_flow", "Cohort flow", fig, table,
                  "Source roster counts are shown when the cohort audit is available. Detailed reasons remain in inputs/{cohort}_audit.tsv; technical completeness does not imply manual QC approval.")


def _plot_reference(writer: ReportWriter, output: Path, dictionary: pd.DataFrame) -> None:
    data = _read(output / "reference/LS_oof_z.tsv")
    tail = (1 - float(writer.settings["credible_interval"])) / 2
    rows = []
    for item in dictionary.itertuples(index=False):
        values = pd.to_numeric(data[item.feature_id], errors="raise").dropna().to_numpy()
        if not len(values):
            raise ValueError("LS calibration has no finite observations for " + item.feature_id)
        rows.append({"feature_id": item.feature_id, "label": item.label, "modality": item.modality,
                     "n": len(values), "mean": np.mean(values), "sd": np.std(values, ddof=1),
                     "interval_lower": np.quantile(values, tail), "q25": np.quantile(values, .25),
                     "median": np.median(values), "q75": np.quantile(values, .75), "interval_upper": np.quantile(values, 1-tail)})
    table = pd.DataFrame(rows)
    fig, axes = writer.axes(2, 1, height=1.5)
    for ax, modality in zip(axes.flat, ("WMH", "GM")):
        subset = table[table.modality.str.upper() == modality]
        x = np.arange(len(subset))
        ax.vlines(x, subset.interval_lower, subset.interval_upper, color="#b8c6ca", linewidth=2)
        ax.vlines(x, subset.q25, subset.q75, color=writer.settings["cohort_colors"]["LS"], linewidth=5)
        ax.scatter(x, subset["median"], color="black", s=8, zorder=3)
        ax.axhline(0, color="gray", ls="--", lw=.8)
        ax.set_xticks(x, subset.label, rotation=65, ha="right", fontsize=7)
        ax.set(title=modality, ylabel="LS out-of-fold z")
    writer.figure("02_ls_calibration", "LS out-of-fold reference distributions", fig, table,
                  f"Lines show central {100 * writer.settings['credible_interval']:g}% and 50% intervals; dots are medians. These use held-out LS predictions; diagnostics.tsv retains calibration and covariate-dependence checks.")
    diagnostics = _read(output / "reference/diagnostics.tsv")
    columns = ["mean_z", "sd_z", "coverage_95", "correlation_age", "correlation_sex", "correlation_log_icv"]
    if not {"split", *columns}.issubset(diagnostics.columns):
        raise ValueError("LS calibration diagnostics require OOF means, SD, coverage and covariate correlations")
    subset = diagnostics[diagnostics.split == "LS_oof"].set_index("feature_id").reindex(dictionary.feature_id)
    if subset[columns].isna().all(axis=1).any():
        raise ValueError("LS OOF diagnostics lack required feature rows")
    fig, axes = writer.axes(1, 3, height=1.3)
    # 未定义的相关（例如某折某变量常量）以缺失展示，绝不填成零相关。
    groups = [("Calibration", ["mean_z", "sd_z"], -1, 2),
              ("Central 95% coverage", ["coverage_95"], 0, 1),
              ("Residual covariate correlation", ["correlation_age", "correlation_sex", "correlation_log_icv"], -1, 1)]
    for ax, (title, cols, low, high) in zip(axes.flat, groups):
        cmap = plt.get_cmap("RdBu_r" if "correlation" in title.lower() else writer.settings["distribution_cmap"]).copy()
        cmap.set_bad("#e4e4e4")
        matrix = subset[cols].to_numpy(float)
        im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, vmin=low, vmax=high)
        ax.set_xticks(np.arange(len(cols)), [name.replace("correlation_", "").replace("_", " ") for name in cols], rotation=35, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(dictionary)), dictionary.label, fontsize=6)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=.7)
    writer.figure("02b_ls_diagnostics", "LS out-of-fold calibration diagnostics", fig, subset.reset_index(),
                  "Gray cells are undefined diagnostics. Mean, SD, coverage and residual covariate dependence are descriptive checks; model calibration must be judged before interpreting cohort results.")


def _plot_cv(writer: ReportWriter, output: Path) -> None:
    cvic = _read(output / "model/cv/cvic.tsv")
    likelihood = _read(output / "model/cv/fold_loglikelihood.tsv")
    fig, axes = writer.axes(1, 2)
    axes[0, 0].plot(cvic.n_subtypes, cvic.cvic, "o-", color="#32699B")
    axes[0, 0].set(xlabel="Number of subtypes", ylabel="CVIC (lower is better)")
    for fold, group in likelihood.groupby("fold"):
        axes[0, 1].plot(group.n_subtypes, group.log_likelihood, "o-", alpha=.65, label=f"Fold {fold}")
    axes[0, 1].set(xlabel="Number of subtypes", ylabel="Held-out log likelihood")
    axes[0, 1].legend(fontsize=7)
    table = likelihood.merge(cvic, on="n_subtypes", how="left")
    writer.figure("03_model_selection", "C3 cross-validation and model selection", fig, table,
                  "Model selection uses C3 only. GB is not used to choose the subtype count. "
                  "When evaluation_backend is logsumexp_v1, CVIC uses stable log mean likelihoods from the same frozen draws; "
                  "raw zero counts and original floored log likelihoods remain in the source table.")


def _plot_posterior_trace(writer: ReportWriter, model: dict) -> None:
    """展示保存的后验抽样轨迹；抽稀轨迹不被称为充分混合证明。"""
    fractions = np.asarray(model["samples_f"], dtype=float)
    likelihood = np.asarray(model["samples_likelihood"], dtype=float).reshape(-1)
    if fractions.ndim != 2 or not fractions.shape[1] or np.any(fractions < 0) or not np.allclose(fractions.sum(axis=0), 1):
        raise ValueError("Posterior fractions must have normalized subtype-by-display-draw shape")
    indices = np.asarray(model.get("sample_indices", np.arange(fractions.shape[1]))).reshape(-1)
    if len(likelihood) != fractions.shape[1] or len(indices) != len(likelihood):
        raise ValueError("Posterior display likelihood, fraction, and draw indices differ")
    if not np.isfinite(likelihood).all() or not np.isfinite(fractions).all():
        raise ValueError("Posterior display arrays contain non-finite samples")
    if not np.isfinite(indices).all() or np.any(np.diff(indices) <= 0):
        raise ValueError("Posterior display draw indices must be finite and strictly increasing")
    fig, axes = writer.axes(2, 1)
    axes[0, 0].plot(indices, likelihood, color="#586e7b", lw=.8)
    axes[0, 0].set(ylabel="Log likelihood", xlabel="Saved posterior draw index")
    rows = []
    for subtype, values in enumerate(fractions):
        axes[1, 0].plot(indices, values, lw=.8, label=f"Subtype {subtype + 1}")
        rows.extend({"draw_index": int(index), "subtype": subtype + 1, "fraction": float(value),
                     "log_likelihood": float(ll)} for index, value, ll in zip(indices, values, likelihood))
    axes[1, 0].set(ylabel="Subtype fraction", xlabel="Saved posterior draw index", ylim=(0, 1))
    axes[1, 0].legend(fontsize=7)
    writer.figure("03b_posterior_trace", "Saved posterior likelihood and subtype-fraction traces", fig, pd.DataFrame(rows),
                  "These saved, thinned samples support inspection only. Full chains remain in the workstation archive; this panel does not certify convergence or mixing.")


# -------- 后验事件顺序、不确定性和空间轨迹 --------
def _posterior_events(model: dict, maximum: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sequences = np.asarray(model["samples_sequence"], dtype=int)
    if sequences.ndim != 3:
        raise ValueError("samples_sequence must have shape (subtypes, events, draws)")
    take = np.unique(np.linspace(0, sequences.shape[2] - 1, min(maximum, sequences.shape[2])).astype(int))
    sequences = sequences[:, :, take]
    for values in sequences.transpose(0, 2, 1).reshape(-1, sequences.shape[1]):
        if not np.array_equal(np.sort(values), np.arange(sequences.shape[1])):
            raise ValueError("Posterior sequence is not an event permutation")
    zvalues = np.asarray(model["Z_vals"], dtype=float)
    event_z = zvalues.T.ravel()
    event_biomarker = np.tile(np.arange(len(zvalues)), zvalues.shape[1])
    active = event_z > 0
    if active.sum() != sequences.shape[1]:
        raise ValueError("Z_vals event count does not match posterior sequences")
    return sequences, event_biomarker[active], event_z[active], take


def _plot_pvd(writer: ReportWriter, model: dict, dictionary: pd.DataFrame, prefix: str = "04_pvd", context: str = "") -> None:
    sequences, biomarkers, thresholds, _ = _posterior_events(model, writer.settings["max_posterior_draws"])
    names = dictionary.set_index("feature_id").label.to_dict()
    ids = np.asarray(model["feature_ids"]).astype(str)
    positions = np.argsort(sequences, axis=1)
    tail = (1 - float(writer.settings["credible_interval"])) / 2
    for subtype in range(len(sequences)):
        order = np.argsort(positions[subtype].mean(axis=1), kind="stable")
        probabilities = np.stack([(positions[subtype] == index).mean(axis=1) for index in range(sequences.shape[1])], axis=1)
        rows = []
        for event in range(sequences.shape[1]):
            for position in range(sequences.shape[1]):
                rows.append({"subtype": subtype + 1, "event_id": event, "feature_id": ids[biomarkers[event]],
                             "threshold_z": thresholds[event], "position": position + 1,
                             "probability": probabilities[event, position],
                             "mean_position": positions[subtype, event].mean() + 1,
                             "interval_lower_position": np.quantile(positions[subtype, event] + 1, tail),
                             "interval_upper_position": np.quantile(positions[subtype, event] + 1, 1-tail)})
        fig, axes = writer.axes(height=max(1.2, sequences.shape[1] / 45))
        ax = axes[0, 0]
        plot = ax.imshow(probabilities[order], aspect="auto", interpolation="nearest",
                         cmap=writer.settings["pvd_cmap"], vmin=0, vmax=1)
        ax.set_yticks(np.arange(len(order)), [f"{names[ids[biomarkers[i]]]}  z={thresholds[i]:g}" for i in order], fontsize=5)
        ax.set(xlabel="Event position (1-based)", ylabel="Biomarker threshold event")
        ticks = np.unique(np.linspace(0, sequences.shape[1] - 1, 9).astype(int))
        ax.set_xticks(ticks, ticks + 1)
        fig.colorbar(plot, ax=ax, label="Posterior position probability", shrink=.6)
        writer.figure(f"{prefix}_subtype_{subtype + 1:02d}", f"{context}Subtype {subtype + 1}: event order uncertainty", fig, pd.DataFrame(rows),
                      "Displayed posterior draws are a deterministic subset of the saved chain. Event position is not elapsed disease time.")


def _trajectory_table(model: dict, settings: dict) -> pd.DataFrame:
    sequences, biomarkers, thresholds, _ = _posterior_events(model, settings["max_posterior_draws"])
    ids = np.asarray(model["feature_ids"]).astype(str)
    stages = np.unique(np.rint(np.asarray(settings["stage_points"]) * sequences.shape[1]).astype(int))
    if np.any(stages < 0) or np.any(stages > sequences.shape[1]):
        raise ValueError("report.stage_points must contain fractions between zero and one")
    positions = np.argsort(sequences, axis=1) + 1
    tail = (1 - float(settings.get("credible_interval", .95))) / 2
    rows = []
    for subtype in range(len(sequences)):
        for biomarker, feature in enumerate(ids):
            events = np.where(biomarkers == biomarker)[0]
            event_order = events[np.argsort(thresholds[events])]
            values = []
            for draw in range(sequences.shape[2]):
                x = np.r_[0, positions[subtype, event_order, draw], sequences.shape[1] + 1]
                y = np.r_[0, thresholds[event_order], model["Z_max"][biomarker]]
                if np.any(np.diff(x) <= 0):
                    raise ValueError("Posterior violates within-biomarker threshold ordering")
                # 官方ZscoreSustain用相邻event-boundary值的均值定义离散stage。
                values.append(.5 * np.interp(stages, x, y) + .5 * np.interp(stages + 1, x, y))
            values = np.asarray(values)
            for index, stage in enumerate(stages):
                rows.append({"subtype": subtype + 1, "stage": stage, "feature_id": feature,
                             "mean_z": np.mean(values[:, index]), "interval_lower_z": np.quantile(values[:, index], tail),
                             "interval_upper_z": np.quantile(values[:, index], 1-tail)})
    return pd.DataFrame(rows)


def _plot_spatial(writer: ReportWriter, cfg: dict, model: dict, dictionary: pd.DataFrame) -> None:
    trajectory = _trajectory_table(model, writer.settings)
    maximum = float(np.max(model["Z_max"]))
    for modality in ("WMH", "GM"):
        background, atlas = display_arrays(cfg, modality)
        regions = dictionary[dictionary.modality.str.upper() == modality]
        if regions.empty:
            raise ValueError("Feature dictionary lacks modality " + modality)
        bounds = np.where(np.any(atlas > 0, axis=(0, 1)))[0]
        if not len(bounds):
            raise ValueError("Empty spatial display atlas")
        slices = np.rint(bounds[0] + np.asarray(writer.settings["slice_fractions"]) * (bounds[-1] - bounds[0])).astype(int)
        if np.any(slices < 0) or np.any(slices >= atlas.shape[2]):
            raise ValueError("report.slice_fractions must lie within the atlas")
        for subtype, table in trajectory[trajectory.feature_id.isin(regions.feature_id)].groupby("subtype"):
            stages = sorted(table.stage.unique())
            fig, axes = writer.axes(len(stages), len(slices), height=max(1, len(stages) * .42))
            for row, stage in enumerate(stages):
                values = table[table.stage == stage].set_index("feature_id").mean_z
                volume = np.full(atlas.shape, np.nan, dtype=np.float32)
                for region in regions.itertuples(index=False):
                    volume[np.isin(atlas, source_atlas_labels(region))] = values[region.feature_id]
                for col, plane in enumerate(slices):
                    ax = axes[row, col]
                    ax.imshow(background[:, :, plane].T, origin="lower", cmap="gray")
                    im = ax.imshow(np.ma.masked_invalid(volume[:, :, plane].T), origin="lower", cmap=writer.settings[modality.lower() + "_cmap"], vmin=0, vmax=maximum, alpha=.85)
                    ax.axis("off")
                    if col == 0:
                        ax.set_title(f"Stage {stage}", fontsize=8)
                    if row == len(stages) - 1:
                        ax.text(.02, .02, "L", transform=ax.transAxes, color="white", fontsize=7)
                        ax.text(.94, .02, "R", transform=ax.transAxes, color="white", fontsize=7)
            fig.subplots_adjust(left=.02, right=.88, bottom=.02, top=.90, hspace=.12, wspace=.03)
            color_axis = fig.add_axes([.91, .24, .015, .5])
            fig.colorbar(im, cax=color_axis, label="Model-implied z")
            fig._sustain_layout_ready = True
            writer.figure(f"05_spatial_{modality.lower()}_subtype_{int(subtype):02d}",
                          f"Subtype {int(subtype)}: {modality} model-implied regional trajectory", fig, table,
                          "Public-template regional display of official linear-z model stage values averaged over posterior draws. Only included feature ROIs are colored; other regions are blank. This is not patient anatomy or measured longitudinal progression. GM positive z denotes greater atrophy.")


# -------- 稳定性、冻模验证与年龄支持范围 --------
def _plot_stability(writer: ReportWriter, output: Path) -> None:
    rows = []
    candidates = [("seed", output / "model/fit/stability.tsv"),
                  ("fold", output / "model/cv/cv_stability.tsv"),
                  ("fold", output / "model/fit/cv_stability.tsv")]
    for kind, path in candidates:
        if path.is_file():
            frame = _read(path)
            frame["comparison"] = kind
            rows.append(frame)
    if not rows or not any((row.comparison == "fold").any() for row in rows):
        raise FileNotFoundError("Cross-fold stability artifact is required: model/cv/cv_stability.tsv")
    table = pd.concat(rows, ignore_index=True)
    if "kendall_tau" not in table:
        raise ValueError("Stability table requires kendall_tau")
    fig, axes = writer.axes(1, 2)
    for ax, kind in zip(axes.flat, ("seed", "fold")):
        frame = table[table.comparison == kind]
        for subtype, group in frame.groupby("subtype"):
            ax.scatter(np.arange(len(group)) + .06 * int(subtype), group.kendall_tau, label=f"Subtype {subtype}")
        ax.axhline(1, color="gray", ls="--", lw=.8)
        ax.set(title=f"Across {kind}s", xlabel="Comparison", ylabel="Matched sequence Kendall tau", ylim=(-1.05, 1.05))
        ax.legend(fontsize=7)
    writer.figure("06_stability", "Matched subtype sequence stability", fig, table,
                  "Subtypes are matched before comparison. This describes fitted sequence stability, not causal or longitudinal validation.")


def _assignment_tables(output: Path, scope: str) -> pd.DataFrame:
    c3 = _read(output / f"model/fit/assignments_C3_{scope}.tsv")
    gb = _read(output / f"validation/assignments_GB_{scope}.tsv")
    c3["cohort"], gb["cohort"] = "C3", "GB"
    return pd.concat([c3, gb], ignore_index=True)


def _plot_assignments(writer: ReportWriter, output: Path) -> None:
    data = _assignment_tables(output, "primary")
    tail = (1 - float(writer.settings["credible_interval"])) / 2
    intervals = []
    for cohort, relative in (("C3", "model/fit"), ("GB", "validation")):
        with np.load(output / relative / f"posterior_{cohort}_primary.npz", allow_pickle=False) as source:
            stage = np.asarray(source["posterior_stage"], dtype=float)
            keys = np.asarray(source["subject_keys"]).astype(str)
        if stage.ndim != 2 or len(stage) != len(keys) or not np.isfinite(stage).all() or np.any(stage < 0) or not np.allclose(stage.sum(axis=1), 1):
            raise ValueError("Invalid full stage posterior: " + cohort)
        cumulative = np.cumsum(stage, axis=1)
        lower, upper = (cumulative >= tail).argmax(axis=1), (cumulative >= 1-tail).argmax(axis=1)
        intervals.extend({"subject_key": key, "stage_posterior_lower": int(lo), "stage_posterior_upper": int(hi),
                          "stage_interval_width": int(hi - lo)} for key, lo, hi in zip(keys, lower, upper))
    data = data.merge(pd.DataFrame(intervals), on="subject_key", validate="one_to_one", how="left")
    if data.stage_interval_width.isna().any():
        raise ValueError("Stage posterior keys do not match assignment keys")
    table = data.groupby(["cohort", "subtype", "stage"], dropna=False).size().rename("n").reset_index()
    table["cohort_fraction"] = table.n / table.groupby("cohort").n.transform("sum")
    fig, axes = writer.axes(1, 2)
    subtypes = sorted(table.subtype.unique())
    for offset, cohort in enumerate(("C3", "GB")):
        group = table[table.cohort == cohort].groupby("subtype").cohort_fraction.sum().reindex(subtypes, fill_value=0)
        axes[0, 0].bar(np.arange(len(subtypes)) + (offset - .5) * .35, group, width=.35,
                       label=cohort, color=writer.settings["cohort_colors"][cohort])
        stage = table[table.cohort == cohort].groupby("stage").cohort_fraction.sum()
        axes[0, 1].plot(stage.index, stage.values, label=cohort, color=writer.settings["cohort_colors"][cohort])
    axes[0, 0].set_xticks(np.arange(len(subtypes)), ["Undifferentiated" if int(value) == 0 else f"Subtype {int(value)}" for value in subtypes], rotation=20)
    axes[0, 0].set(ylabel="Cohort proportion")
    axes[0, 1].set(xlabel="SuStaIn stage", ylabel="Cohort proportion")
    for ax in axes.flat:
        ax.legend()
    writer.figure("07_frozen_validation", "C3 discovery and frozen-model GB validation", fig, table,
                  "Stage zero is displayed as undifferentiated. GB uses the frozen LS reference and C3 model; subtype proportions may differ.")
    fig, axes = writer.axes(1, 3)
    uncertainty = data[[key for key in ("subject_key", "participant_id", "cohort", "subtype", "stage", "subtype_probability", "subtype_entropy", "stage_expected", "stage_zero_probability", "stage_posterior_lower", "stage_posterior_upper", "stage_interval_width") if key in data]].copy()
    for ax, column in zip(axes.flat, ("subtype_probability", "subtype_entropy", "stage_interval_width")):
        values = [pd.to_numeric(data.loc[data.cohort == cohort, column]).dropna().to_numpy() for cohort in ("C3", "GB")]
        ax.boxplot(values, tick_labels=["C3", "GB"], showfliers=False)
        ax.set(ylabel=column.replace("_", " "))
    writer.figure("08_assignment_uncertainty", "Individual assignment uncertainty", fig, uncertainty,
                  f"Stage interval width uses the central {100 * writer.settings['credible_interval']:g}% marginal stage posterior. Full joint subtype-stage probabilities are retained in posterior NPZ files. Maximum-probability labels do not replace posterior uncertainty.")


def _plot_age(writer: ReportWriter, cfg: dict, output: Path) -> None:
    primary = _assignment_tables(output, "primary")
    all_data = _assignment_tables(output, "all")
    # 支持范围优先来自实际LS模型，避免把旧GenMIND年龄范围沿用到新常模。
    settings = _reference_settings(cfg, output)
    support = [settings["age_min"], settings["age_max"]]
    all_data["age_group"] = np.where(all_data.age < float(support[0]), "Below LS support",
                                     np.where(all_data.age > float(support[1]), "Above LS support", "Within LS support"))
    table = all_data.groupby(["cohort", "age_group", "subtype"], dropna=False).agg(
        n=("stage", "size"), median_stage=("stage", "median"), mean_stage=("stage", "mean"),
        mean_subtype_probability=("subtype_probability", "mean")).reset_index()
    table["age_support_min"], table["age_support_max"] = support
    table["n_primary"] = table.cohort.map(primary.groupby("cohort").size())
    fig, axes = writer.axes(1, 2)
    groups = ["Below LS support", "Within LS support", "Above LS support"]
    for ax, cohort in zip(axes.flat, ("C3", "GB")):
        counts = all_data[all_data.cohort == cohort].groupby("age_group").size().reindex(groups, fill_value=0)
        ax.bar(groups, counts, color=writer.settings["cohort_colors"][cohort])
        ax.set_xticks(np.arange(3), groups, rotation=25, ha="right")
        ax.set(title=cohort, ylabel="Participants")
    writer.figure("09_age_support", "Primary age-supported analysis and extrapolation sensitivity", fig, table,
                  f"LS reference support is {support[0]}–{support[1]} years. Out-of-support scores are separately labelled sensitivity/extrapolation; this figure is a descriptive comparison, not evidence of all-age calibration.")


# -------- 静态报告发布入口 --------
def _excluded_raw_table(output: Path, selection: dict) -> pd.DataFrame:
    """排除区域仅描述原始mL，绝不伪造z或分期；GB分布不参与排除决策。"""
    rows = []
    low, high = selection["age_support"]
    for cohort in ("LS", "C3", "GB"):
        frame = _read(output / "inputs" / f"{cohort}.tsv")
        for scope, group in (("all_ages", frame), ("primary", frame.loc[frame.age.between(low, high)])):
            for item in selection["excluded"]:
                values = group[item["feature_id"]].to_numpy(dtype=float)
                rows.append({"cohort": cohort, "scope": scope, "feature_id": item["feature_id"],
                             "n": len(values), "n_zero": int(np.sum(values == 0)),
                             "n_nonzero": int(np.sum(values != 0)),
                             "max_raw_ml": float(values.max()) if len(values) else None,
                             "reason": item["reason"], "used_for_model": False})
    return pd.DataFrame(rows)


def run_report(cfg: dict, output: Path, bundle_only: bool = False) -> dict:
    """读取冻结结果生成报告；正式缺失文件时写明未完成并失败。"""
    output = Path(output)
    folder = output / "report"
    folder.mkdir(parents=True, exist_ok=True)
    # 立即撤销上次complete标记，防止本轮预检失败后仍展示旧成功状态。
    _write_json(folder / "summary.json", {"status": "incomplete", "reason": "Report regeneration has not completed"})
    _status_page(folder, ["Report regeneration has not completed"], _synthetic(cfg))
    settings = dict(DEFAULTS)
    settings.update(cfg.get("report", {}))
    if not 0 < settings["credible_interval"] < 1:
        raise ValueError("report.credible_interval must lie strictly between zero and one")
    for key in ("raster_dpi", "preview_dpi", "figure_width", "figure_height", "font_size", "max_posterior_draws"):
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError("report." + key + " must be finite and positive")
    for key in ("stage_points", "slice_fractions"):
        values = np.asarray(settings[key], dtype=float)
        if not len(values) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError("report." + key + " must be a nonempty list of fractions in [0,1]")
    missing = [name for name in REQUIRED_FILES if not (output / name).is_file()]
    cv_display_path = next((output / name for name in ("model/fit/cv_display.npz", "model/cv/cv_display.npz")
                            if (output / name).is_file()), None)
    if cv_display_path is None:
        missing.append("model/fit/cv_display.npz")
    synthetic = _synthetic(cfg)
    settings["_synthetic"] = synthetic
    if missing:
        summary = {"status": "incomplete", "synthetic": synthetic, "missing": missing}
        _write_json(folder / "summary.json", summary)
        _status_page(folder, missing, synthetic)
        if settings["strict"] or not synthetic:
            raise FileNotFoundError("Report inputs are incomplete: " + ", ".join(missing))
        return summary
    dictionary = frozen_dictionary(output, cfg)
    required_columns = {"feature_id", "label", "modality", "atlas_index"}
    if not required_columns.issubset(dictionary.columns) or dictionary.feature_id.duplicated().any():
        raise ValueError("Invalid feature dictionary for reporting")
    try:
        _reference_settings(cfg, output)
        validate_assets(cfg, dictionary)
    except (FileNotFoundError, ValueError) as exc:
        _write_json(folder / "summary.json", {"status": "incomplete", "synthetic": synthetic, "missing": [str(exc)]})
        _status_page(folder, [str(exc)], synthetic)
        raise
    with np.load(output / "model/fit/selected_model.npz", allow_pickle=False) as source:
        model = {key: source[key] for key in source.files}
    # 官方保存原始亚型顺序；展示编号必须匹配最终assignment的subtype_order。
    order = np.asarray(model.get("subtype_order", np.arange(len(model["samples_sequence"]))), dtype=int)
    if not np.array_equal(np.sort(order), np.arange(len(model["samples_sequence"]))):
        raise ValueError("Invalid posterior subtype_order")
    model["samples_sequence"] = model["samples_sequence"][order]
    model["samples_f"] = model["samples_f"][order]
    if model["samples_f"].shape != (model["samples_sequence"].shape[0], model["samples_sequence"].shape[2]):
        raise ValueError("Posterior sequence and fraction sample axes differ")
    ids = model["feature_ids"].astype(str).tolist()
    if set(ids) != set(dictionary.feature_id):
        raise ValueError("Saved posterior and feature dictionary differ")
    dictionary = dictionary.set_index("feature_id").loc[ids].reset_index()
    plt.rcParams.update({"font.family": settings.get("font_family", "DejaVu Sans"), "font.size": settings["font_size"],
                         "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "substain-sustain-v1"})
    writer = ReportWriter(folder, settings)
    try:
        _plot_flow(writer, output)
        _plot_reference(writer, output, dictionary)
        _plot_cv(writer, output)
        _plot_posterior_trace(writer, model)
        _plot_pvd(writer, model, dictionary)
        with np.load(cv_display_path, allow_pickle=False) as source:
            crossfold_model = {key: source[key] for key in source.files}
        for key in ("feature_ids", "labels", "Z_max"):
            crossfold_model.setdefault(key, model[key])
        _plot_pvd(writer, crossfold_model, dictionary, "04b_crossfold_pvd", "Cross-fold ")
        _plot_spatial(writer, cfg, model, dictionary)
        _plot_stability(writer, output)
        _plot_assignments(writer, output)
        _plot_age(writer, cfg, output)
    except Exception as exc:
        plt.close("all")
        _write_json(folder / "summary.json", {"status": "incomplete", "synthetic": synthetic,
                                              "missing": [str(exc)], "completed_figures": len(writer.items)})
        _status_page(folder, [str(exc)], synthetic)
        raise
    title = "TECHNICAL DEMONSTRATION — SYNTHETIC DATA" if synthetic else settings.get("title", "LS–C3–GB SuStaIn analysis")
    sections = []
    for item in writer.items:
        links = " · ".join(f'<a href="{html.escape(path)}">{html.escape(Path(path).suffix[1:].upper())}</a>' for path in item["files"])
        sections.append(f'<section><h2>{html.escape(item["title"])}</h2><p>{html.escape(item["note"])}</p>'
                        f'<img loading="lazy" src="figures/{item["name"]}.png" alt="{html.escape(item["title"])}">'
                        f'<p>{links} · <a href="{item["table"]}">Source TSV</a></p></section>')
    warning = "Synthetic demonstration validates software only; no cohort findings are claimed." if synthetic else "Stage is a relative disease-pattern index, not elapsed time or a progression rate. No site-effect adjustment was applied."
    fit_summary = _json(output / "model/fit/summary.json")
    evidence_rows = [("Subtype count", fit_summary.get("n_subtypes", "unavailable")),
                     ("Candidate subtype counts", fit_summary.get("candidate_k", "see selection.json")),
                     ("Selection status", fit_summary.get("selection_status", "see selection.json")),
                     ("Features", len(dictionary)), ("Formal cohort findings", "No — synthetic data" if synthetic else "Interpret with QC and calibration diagnostics")]
    overview = '<table><tbody>' + ''.join('<tr><th style="text-align:left;padding-right:24px">' + html.escape(str(key)) + '</th><td>' + html.escape(str(value)) + '</td></tr>' for key, value in evidence_rows) + '</tbody></table>'
    numerical_path = output/"model/cv/numerical_evaluation.json"
    if numerical_path.is_file():
        numerical = _json(numerical_path)
        counts = numerical.get("raw_zero_subjects_per_k", {})
        overview += '<p class="notice">Frozen-model likelihoods and posterior probabilities use log-sum-exp evaluation. '
        overview += 'Original zero likelihoods remain in model caches; no arbitrary epsilon replaces them in CVIC. '
        overview += 'Official EM/MCMC training, including its numerical floor, is unchanged. '
        overview += 'This numerical repair does not establish adequate reference calibration or model fit.</p>'
        overview += pd.DataFrame([{"K": key, "Held-out subjects with raw zero mean likelihood": value}
                                  for key, value in counts.items()]).to_html(index=False, escape=True)
        overview += '<p>Raw official posterior fields are unavailable (NaN with official_raw_available=false) when all linear '
        overview += 'support is lost. Finite raw outputs with subnormal denominators are retained but marked official_raw_valid=false. '
        overview += 'Separately named stable official fields and weighted joint probabilities are retained.</p>'
        training = numerical.get("training_floor_counts_per_fold_k", [])
        if training:
            maximum = max(row["training_below_floor_subjects"] for row in training)
            overview += f'<p>Largest count below the official training floor at a saved ML-EM solution: {maximum} '
            overview += 'subjects within one fold/model. Counts overlap across folds; these are not unique cohort totals or all-iteration diagnostics.</p>'
    selection_path = output / "reference/feature_selection.json"
    exclusion_table_path = None
    if selection_path.is_file():
        selection = _json(selection_path)
        if selection.get("merged"):
            overview += '<p class="notice">GM regions are merged before normalization: sum source volumes (mL), '
            overview += 'then log-transform and fit the LS reference. No z scores are averaged or summed.</p><ul>'
            for item in selection["merged"]:
                overview += '<li>' + html.escape(item["feature_id"] + ' = ' + ' + '.join(item["source_features"])) + '</li>'
            overview += '</ul><p>Source atlas labels and model ROI numbers are frozen in the model dictionary.</p>'
        if selection["excluded"]:
            overview += '<p class="notice">Raw inputs retain all 40 features. The model uses ' + str(len(dictionary))
            overview += ' features. Explicit exclusions based only on the LS reference:</p><ul>'
            for item in selection["excluded"]:
                overview += '<li>' + html.escape(item["feature_id"] + ': ' + item["reason"])
                overview += f' (LS primary n={item["ls_primary_n"]}, nonzero=0)</li>'
            overview += '</ul><p>Excluded ROIs remain uncolored; they do not represent normal z=0. '
            overview += 'Raw values are retained for descriptive reporting and are not staged.</p>'
            table = _excluded_raw_table(output, selection)
            exclusion_table_path = folder / "tables/00_excluded_raw_features.tsv"
            table.to_csv(exclusion_table_path, sep="\t", index=False, float_format="%.10g")
            overview += table.drop(columns=["reason", "used_for_model"]).to_html(index=False, escape=True)
            overview += '<p><a href="tables/00_excluded_raw_features.tsv">Excluded feature raw-volume summary TSV</a></p>'
    if fit_summary.get("selection_status") == "boundary_optimum":
        overview += '<p class="notice">The selected subtype count is at the upper candidate boundary. This boundary result must be considered before scientific interpretation.</p>'
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    document += '<title>' + html.escape(title) + '</title><style>body{max-width:1200px;margin:36px auto;padding:0 24px;font:16px/1.6 system-ui;color:#202c35;background:#fbfbfa}h1{font-size:32px}h2{font-size:23px}section{margin:36px 0;padding:20px;background:white;border:1px solid #dce1e4}img{display:block;max-width:100%;height:auto}a{color:#235e8c}.notice{padding:18px;background:#edf1e9;border-left:5px solid #6a8067}footer{font-size:13px}</style>'
    document += f'<h1>{html.escape(title)}</h1><p class="notice">{html.escape(warning)}</p>' + overview + ''.join(sections)
    document += '<footer>All figures are regenerated from saved numerical sources and public display atlases. This report loads no remote scripts, fonts, or assets. Full-chain posterior files remain in the analysis archive.</footer></html>'
    (folder / "index.html").write_text(document, encoding="utf-8")
    artifacts = []
    if exclusion_table_path is not None:
        artifacts.append({"path": exclusion_table_path.relative_to(folder).as_posix(),
                          "sha256": file_sha256(exclusion_table_path), "bytes": exclusion_table_path.stat().st_size})
    for item in writer.items:
        for relative in [*item["files"], item["table"]]:
            path = folder / relative
            artifacts.append({"path": relative, "sha256": file_sha256(path), "bytes": path.stat().st_size})
    summary = {"status": "technical_demo_complete" if synthetic else "complete", "synthetic": synthetic,
               "figures": len(writer.items), "items": writer.items, "artifacts": artifacts,
               "posterior_draw_limit": settings["max_posterior_draws"], "selection_status": fit_summary.get("selection_status"),
               "bundle_reproduction": bool(bundle_only)}
    _write_json(folder / "summary.json", summary)
    return summary
