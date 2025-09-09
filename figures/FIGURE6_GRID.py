#!/usr/bin/env python3

import os
import sys
import pickle
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator, MultipleLocator, FuncFormatter, FixedLocator, FormatStrFormatter


def try_register_nimbus_fonts() -> bool:
    """Attempt to register Nimbus Sans fonts from common Linux paths."""
    candidate_paths = [
        "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
        "/usr/share/fonts/truetype/urw-base35/NimbusSans-Regular.ttf",
        "/usr/share/fonts/truetype/nimbus/NimbusSans-Regular.ttf",
        "/usr/local/share/fonts/NimbusSans-Regular.ttf",
    ]
    registered = False
    for path in candidate_paths:
        if os.path.isfile(path):
            try:
                font_manager.fontManager.addfont(path)
                registered = True
            except Exception:
                pass
    return registered


def ensure_nimbus_sans() -> str:
    available_names = {f.name for f in font_manager.fontManager.ttflist}
    for fam in ["Nimbus Sans", "Nimbus Sans L"]:
        if fam in available_names:
            return fam
    # Try to register, then check again
    if try_register_nimbus_fonts():
        available_names = {f.name for f in font_manager.fontManager.ttflist}
        for fam in ["Nimbus Sans", "Nimbus Sans L"]:
            if fam in available_names:
                return fam
    return "DejaVu Sans"


def set_global_font() -> str:
    family = ensure_nimbus_sans()
    # Prefer Nimbus; fall back gracefully. Force Nimbus family when available.
    preferred_list = [family, 'Nimbus Sans L', 'Nimbus Sans', 'DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['font.family'] = preferred_list[0]
    plt.rcParams['font.sans-serif'] = preferred_list
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = family
    plt.rcParams['mathtext.it'] = family
    plt.rcParams['mathtext.bf'] = family
    plt.rcParams['mathtext.sf'] = family
    plt.rcParams['mathtext.default'] = 'regular'
    # Increase global font sizes
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 18
    # Informative log if Nimbus not available
    if family not in ("Nimbus Sans", "Nimbus Sans L"):
        print(f"[WARN] Nimbus Sans not found. Using fallback family: {family}")
    return family


def derive_composite_name_from_path(pkl_path: str) -> str:
    basename = os.path.basename(pkl_path)
    if not basename.endswith("_all_results.pkl"):
        raise ValueError(f"Expected filename to end with '_all_results.pkl', got: {basename}")
    composite_name = basename.replace("_all_results.pkl", "")
    if len(composite_name) == 0:
        raise ValueError(f"Could not derive composite name from filename: {basename}")
    return composite_name


def load_reranking_pickle(pkl_path: str) -> List[Tuple[str, str, pd.DataFrame]]:
    print(f"[INFO] Loading reranking pickle: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, (list, tuple)):
        raise TypeError(f"Pickle did not contain a list/tuple. Type: {type(data)}")
    print(f"[INFO] Loaded {len(data)} reranked annotation entries")
    return data


def load_experimental_dict(experimental_pkl_path: str) -> Dict[str, tuple]:
    print(f"[INFO] Loading experimental lookup dict: {experimental_pkl_path}")
    with open(experimental_pkl_path, "rb") as f:
        experimental_dict = pickle.load(f)
    if not isinstance(experimental_dict, dict):
        raise TypeError(f"Experimental pickle did not contain a dict. Type: {type(experimental_dict)}")
    print(f"[INFO] Experimental entries: {len(experimental_dict)}")
    return experimental_dict


def filter_reranked_by_npeaks(
    reranked_annotations: List[Tuple[str, str, pd.DataFrame]],
    experimental_dict: Dict[str, tuple],
    min_peaks: int,
) -> List[Tuple[str, str, pd.DataFrame]]:
    print(f"[INFO] Filtering annotations by min peaks > {min_peaks}")
    filtered: List[Tuple[str, str, pd.DataFrame]] = []
    missing = 0
    for file_name, exp_smiles, reranked_df in reranked_annotations:
        if file_name not in experimental_dict:
            missing += 1
            continue
        peaks = experimental_dict[file_name][3]
        n_peaks = peaks.shape[0] if hasattr(peaks, "shape") else len(peaks)
        if n_peaks > min_peaks:
            filtered.append((file_name, exp_smiles, reranked_df))
    print(f"[INFO] Kept {len(filtered)} / {len(reranked_annotations)} (missing exp: {missing})")
    return filtered


def compute_topk_averages(
    reranked_annotations: List[Tuple[str, str, pd.DataFrame]],
    metric_score_column: str,
    topk_values: List[int],
    require_min_rows: int,
):
    print(f"[INFO] Computing averages for k in {topk_values} using metric column: '{metric_score_column}'")
    before_avgs_by_k = {k: [] for k in topk_values}
    after_avgs_by_k = {k: [] for k in topk_values}

    for idx, (file_name, exp_smiles, reranked_df) in enumerate(reranked_annotations):
        if not isinstance(reranked_df, pd.DataFrame):
            raise TypeError(f"Entry {idx} for file '{file_name}' does not contain a pandas DataFrame")

        # Validate required columns and minimum rows
        for col in ["hybrid_score", metric_score_column]:
            if col not in reranked_df.columns:
                raise KeyError(
                    f"Missing required column '{col}' in reranked DataFrame for file '{file_name}'.\n"
                    f"Available columns: {list(reranked_df.columns)}"
                )
        if len(reranked_df) < require_min_rows:
            raise ValueError(
                f"Reranked DataFrame for file '{file_name}' has only {len(reranked_df)} rows; "
                f"need at least {require_min_rows} to compute top-{max(topk_values)} averages"
            )

        # Before: sort by 'hungarian_distance' (ascending)
        df_before = reranked_df.sort_values('hungarian_distance', ascending=True)

        # After: sort by metric score column (descending)
        df_after = reranked_df.sort_values(metric_score_column, ascending=False)

        # Compute averages per k
        for k in topk_values:
            before_mean = float(df_before.head(k)["hybrid_score"].mean())
            after_mean = float(df_after.head(k)["hybrid_score"].mean())
            before_avgs_by_k[k].append(before_mean)
            after_avgs_by_k[k].append(after_mean)

    # Aggregate across all annotations
    agg_before = {k: float(np.mean(vals)) for k, vals in before_avgs_by_k.items()}
    agg_after = {k: float(np.mean(vals)) for k, vals in after_avgs_by_k.items()}
    return agg_before, agg_after


def compute_topk_max_hybrids(
    reranked_annotations: List[Tuple[str, str, pd.DataFrame]],
    metric_score_column: str,
    topk_values: List[int],
    require_min_rows: int,
):
    print(f"[INFO] Computing MAX hybrid scores for k in {topk_values} using metric column: '{metric_score_column}'")
    before_max_by_k = {k: [] for k in topk_values}
    after_max_by_k = {k: [] for k in topk_values}

    for idx, (file_name, exp_smiles, reranked_df) in enumerate(reranked_annotations):
        if not isinstance(reranked_df, pd.DataFrame):
            raise TypeError(f"Entry {idx} for file '{file_name}' does not contain a pandas DataFrame")
        for col in ["hybrid_score", metric_score_column, "hungarian_distance"]:
            if col not in reranked_df.columns:
                raise KeyError(
                    f"Missing required column '{col}' in reranked DataFrame for file '{file_name}'.\n"
                    f"Available columns: {list(reranked_df.columns)}"
                )
        if len(reranked_df) < require_min_rows:
            raise ValueError(
                f"Reranked DataFrame for file '{file_name}' has only {len(reranked_df)} rows; "
                f"need at least {require_min_rows} to compute top-{max(topk_values)} max values"
            )

        df_before = reranked_df.sort_values('hungarian_distance', ascending=True)
        df_after = reranked_df.sort_values(metric_score_column, ascending=False)

        for k in topk_values:
            before_max = float(df_before.head(k)["hybrid_score"].max())
            after_max = float(df_after.head(k)["hybrid_score"].max())
            before_max_by_k[k].append(before_max)
            after_max_by_k[k].append(after_max)

    agg_before_max = {k: float(np.mean(vals)) for k, vals in before_max_by_k.items()}
    agg_after_max = {k: float(np.mean(vals)) for k, vals in after_max_by_k.items()}
    return agg_before_max, agg_after_max


def compute_structural_efficiency(
    reranked_annotations: List[Tuple[str, str, pd.DataFrame]],
    metric_score_column: str,
    k_values: List[int],
    top100: int = 100,
) -> Dict[str, List[float]]:
    print(f"[INFO] Computing structural efficiency for top-k {k_values} vs top-{top100}")
    efficiencies: Dict[str, List[float]] = {f"before_top{k}": [] for k in k_values}
    efficiencies.update({f"after_top{k}": [] for k in k_values})

    for idx, (file_name, exp_smiles, reranked_df) in enumerate(reranked_annotations):
        if "hungarian_distance" not in reranked_df.columns:
            raise KeyError(
                f"Missing required column 'hungarian_distance' for file '{file_name}'."
            )
        df_before = reranked_df.sort_values('hungarian_distance', ascending=True)
        df_after = reranked_df.sort_values(metric_score_column, ascending=False)

        denom_before = float(df_before.head(top100)['hybrid_score'].max())
        denom_after = float(df_after.head(top100)['hybrid_score'].max())
        if denom_before <= 0 or denom_after <= 0:
            raise ValueError(f"Non-positive denominator(s) for file '{file_name}': before={denom_before}, after={denom_after}")

        for k in k_values:
            num_before = float(df_before.head(k)['hybrid_score'].max())
            num_after = float(df_after.head(k)['hybrid_score'].max())
            eff_before = num_before / denom_before
            eff_after = num_after / denom_after
            efficiencies[f"before_top{k}"].append(eff_before)
            efficiencies[f"after_top{k}"].append(eff_after)

    return efficiencies


def draw_delta_bracket(
    ax: plt.Axes,
    x_left: float,
    x_right: float,
    y_left: float,
    y_right: float,
    color: str = "#000000",
    y_offset_fraction: float = 0.02,
    font_size: int = 16,
    bold: bool = True,
) -> None:
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min
    y_bracket = max(y_left, y_right) + y_offset_fraction * y_span
    ax.plot([x_left, x_left, x_right, x_right], [y_left, y_bracket, y_bracket, y_right], color=color, linewidth=2)
    if y_left > 0:
        delta_pct = (y_right - y_left) / y_left * 100.0
        label = f"{delta_pct:+.1f}%"
    else:
        delta = y_right - y_left
        label = f"Δ{delta:+.3f}"
    ax.text((x_left + x_right) / 2.0, y_bracket + y_offset_fraction * y_span,
            label, ha='center', va='bottom', fontsize=font_size, color=color,
            fontweight='bold' if bold else None)


def draw_grouped_bars_on_axis(
    ax: plt.Axes,
    agg_before: Dict[int, float],
    agg_after: Dict[int, float],
    y_label: str,
    y_lim: Tuple[float, float],
    show_legend: bool = False,
):
    ks = sorted(agg_before.keys())
    before_vals = [agg_before[k] for k in ks]
    after_vals = [agg_after[k] for k in ks]

    x = np.arange(len(ks))
    width = 0.35

    ax.bar(x - width/2, before_vals, width, label="Top-K Lookup", color="#ecae00", alpha=0.8)
    ax.bar(x + width/2, after_vals, width, label="Algo. Mol. Net.", color="#0086c0", alpha=0.8)

    ax.set_ylabel(y_label, fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{k}" for k in ks], fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(True, axis="y", alpha=0.3, color="#a0a0a0")
    ax.set_ylim(*y_lim)
    # Exactly 4 y-ticks, one decimal, hide zero label
    ax.yaxis.set_major_locator(MultipleLocator(0.4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: ("" if abs(y) < 1e-9 else f"{y:.1f}")))

    for i, k in enumerate(ks):
        b = before_vals[i]
        a = after_vals[i]
        x0 = x[i] - width/2
        x1 = x[i] + width/2
        draw_delta_bracket(ax, x0, x1, b, a, color="#000000", y_offset_fraction=0.03, font_size=16, bold=True)

    if show_legend:
        ax.legend(fontsize=18, loc='upper left')


def main():
    # CLI: [low_results_pkl] [high_results_pkl] [experimental_lookup_pkl]
    if len(sys.argv) < 3:
        experimental_pkl_path = \
            "/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/ExperimentalHSQC_Lookup.pkl"
        low_pkl_path = \
            "/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/annotation_reranking_results_final_low_quality/ra_weighted_product_hungarian_nn_composite_all_results.pkl"
        high_pkl_path = \
            "/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/annotation_reranking_results_final_high_quality/ra_weighted_product_hungarian_nn_composite_all_results.pkl"
        print("[WARN] No CLI args provided. Using default low/high paths and experimental lookup.")
    else:
        low_pkl_path = sys.argv[1]
        high_pkl_path = sys.argv[2]
        experimental_pkl_path = sys.argv[3] if len(sys.argv) > 3 else \
            "/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/ExperimentalHSQC_Lookup.pkl"

    # Validate files
    for path in [low_pkl_path, high_pkl_path, experimental_pkl_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

    # Derive composite names and ensure they match
    low_composite = derive_composite_name_from_path(low_pkl_path)
    high_composite = derive_composite_name_from_path(high_pkl_path)
    if low_composite != high_composite:
        raise ValueError(f"Composite names differ: low='{low_composite}', high='{high_composite}'")
    metric_score_column = f"{low_composite}_score"
    print(f"[INFO] Composite name: {low_composite}")
    print(f"[INFO] Metric score column: '{metric_score_column}'")

    set_global_font()

    # Load data
    low_annotations = load_reranking_pickle(low_pkl_path)
    high_annotations = load_reranking_pickle(high_pkl_path)
    experimental_dict = load_experimental_dict(experimental_pkl_path)

    # Filter by min peaks
    min_peaks = 15
    low_annotations = filter_reranked_by_npeaks(low_annotations, experimental_dict, min_peaks=min_peaks)
    high_annotations = filter_reranked_by_npeaks(high_annotations, experimental_dict, min_peaks=min_peaks)

    # Compute metrics
    topk_values = [3, 10]
    low_avg_before, low_avg_after = compute_topk_averages(low_annotations, metric_score_column, topk_values, require_min_rows=max(topk_values))
    high_avg_before, high_avg_after = compute_topk_averages(high_annotations, metric_score_column, topk_values, require_min_rows=max(topk_values))

    low_max_before, low_max_after = compute_topk_max_hybrids(low_annotations, metric_score_column, topk_values, require_min_rows=max(topk_values))
    high_max_before, high_max_after = compute_topk_max_hybrids(high_annotations, metric_score_column, topk_values, require_min_rows=max(topk_values))

    low_eff = compute_structural_efficiency(low_annotations, metric_score_column, k_values=topk_values, top100=100)
    high_eff = compute_structural_efficiency(high_annotations, metric_score_column, k_values=topk_values, top100=100)

    # Aggregate efficiencies
    low_eff_before = {k: float(np.mean(low_eff[f"before_top{k}"])) for k in topk_values}
    low_eff_after = {k: float(np.mean(low_eff[f"after_top{k}"])) for k in topk_values}
    high_eff_before = {k: float(np.mean(high_eff[f"before_top{k}"])) for k in topk_values}
    high_eff_after = {k: float(np.mean(high_eff[f"after_top{k}"])) for k in topk_values}

    # Build 3x2 grid
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12), sharex=True, sharey='row')
    # Remove spacing between columns and rows
    fig.subplots_adjust(wspace=0.0, hspace=0.0)

    # Row 1: Max Hybrid (y range 0..1.05 like original)
    draw_grouped_bars_on_axis(
        axes[0, 0], low_max_before, low_max_after, y_label="Max. Hybrid", y_lim=(0, 1.05), show_legend=False
    )
    axes[0, 0].set_title("Low-efficiency", fontsize=20)
    draw_grouped_bars_on_axis(
        axes[0, 1], high_max_before, high_max_after, y_label="", y_lim=(0, 1.05), show_legend=False
    )
    axes[0, 1].set_title("High-efficiency", fontsize=20)

    # Row 2: Average Hybrid (y range 0..0.75 like original)
    draw_grouped_bars_on_axis(
        axes[1, 0], low_avg_before, low_avg_after, y_label="Avg. Hybrid", y_lim=(0, 0.95), show_legend=True
    )
    draw_grouped_bars_on_axis(
        axes[1, 1], high_avg_before, high_avg_after, y_label="", y_lim=(0, 0.95), show_legend=False
    )
    # Override middle row y-ticks to 0.3, 0.6, 0.9 with one decimal
    for ax_mid in (axes[1, 0], axes[1, 1]):
        ax_mid.yaxis.set_major_locator(FixedLocator([0.3, 0.6, 0.9]))
        ax_mid.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Row 3: Structural Efficiency (eta), y range 0..1.3 like original
    draw_grouped_bars_on_axis(
        axes[2, 0], low_eff_before, low_eff_after, y_label=r"$\eta$ (Struc. Eff.)", y_lim=(0, 1.15), show_legend=False
    )
    draw_grouped_bars_on_axis(
        axes[2, 1], high_eff_before, high_eff_after, y_label="", y_lim=(0, 1.15), show_legend=False
    )

    # Make bottom row show x tick labels prominently
    for col in range(2):
        for row in [0, 1]:
            for label in axes[row, col].get_xticklabels():
                label.set_visible(False)

    # Shared suptitle
    # fig.suptitle(f"Low vs High efficiency — {low_composite}", fontsize=16)

    # Save figure
    out_dir = Path("/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/figures/paper_figures/comparison_low_vs_high/")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{low_composite}_3_10_V2_AUG20.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved 3x2 grid figure: {out_path}")
    plt.show()

if __name__ == "__main__":
    main()


