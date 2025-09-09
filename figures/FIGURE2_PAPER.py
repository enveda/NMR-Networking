#!/usr/bin/env python3
"""
Standalone 2x3 KDE Plotter for Annotation Comparison

Creates a 2x3 grid of overlaid KDEs comparing Modified-Hungarian vs Hungarian-NN
annotations for top-1 and top-5 values across exact, close, and poor match regimes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
"""
Compatibility shim for Matplotlib >= 3.9 where cm.register_cmap was moved to
matplotlib.colormaps.register. This ensures older third-party code (e.g.,
Seaborn versions expecting cm.register_cmap) can still import safely.
"""
try:
    import matplotlib.cm as _mpl_cm
    try:
        import matplotlib.colormaps as _mpl_colormaps
    except Exception:  # pragma: no cover
        _mpl_colormaps = None
    if not hasattr(_mpl_cm, 'register_cmap') and _mpl_colormaps is not None and hasattr(_mpl_colormaps, 'register'):
        def _register_cmap(name=None, cmap=None, data=None, lut=None, override_builtin=False):
            if cmap is None:
                return
            try:
                _mpl_colormaps.register(cmap, name=name, override_builtin=override_builtin)
            except Exception:
                pass
        _mpl_cm.register_cmap = _register_cmap  # type: ignore[attr-defined]
except Exception:
    pass

import seaborn as sns
from scipy.stats import gaussian_kde
import pickle
import warnings
warnings.filterwarnings('ignore')

# New imports for font handling
import os
from matplotlib import font_manager
from urllib.request import urlopen
from matplotlib.font_manager import FontProperties

# =====================================================
# CONFIGURABLE PARAMETERS
# =====================================================

# High-contrast, consistent color palette for all figures
COLOR_MOD_HUNG = '#f3d39c'
COLOR_HUNG_NN  = '#8199a5'
ALPHA_FILL     = 0.28           # Transparency for filled KDE areas
LINEWIDTH_KDE  = 2.0

# Consistent grid styling
GRID_ALPHA = 0.3
GRID_LINESTYLE = '--'
GRID_LINEWIDTH = 0.8

# Consistent background and spine styling
BACKGROUND_COLOR = '#fafafa'
SPINE_COLOR = '#cccccc'
SPINE_LINEWIDTH = 1

# Consistent text styling
TEXTBOX_STYLE = dict(boxstyle="round,pad=0.8", facecolor='white', alpha=0.95, edgecolor='gray', linewidth=1)

# Figure size and layout
DEFAULT_FIGSIZE = (12, 8)
DEFAULT_HSPACE = 0.15
DEFAULT_WSPACE = 0.0
# Legend styling
LEGEND_FONTSIZE = 16

# KDE controls
KDE_BW_ADJUST = 0.7      # seaborn bw_adjust
KDE_CLIP = (0.0, 1.0)    # clip to valid hybrid_fraction range
KDE_GRID = np.linspace(0, 1, 512)  # grid to estimate global y-lims

# =====================================================
# FONT MANAGEMENT
# =====================================================

def _register_downloaded_font(font_path: str) -> None:
    try:
        font_manager.fontManager.addfont(font_path)
        # Rebuild the font cache so Matplotlib sees the new font
        font_manager._rebuild()
    except Exception:
        pass


def ensure_nimbus_sans() -> str:
    """Ensure Nimbus Sans (or Nimbus Sans L) is available; download if missing.

    Returns the best available family name to use in Matplotlib rcParams.
    """
    preferred_families = ["Nimbus Sans", "Nimbus Sans L"]

    available_names = {f.name for f in font_manager.fontManager.ttflist}
    for fam in preferred_families:
        if fam in available_names:
            return fam

    # Attempt to download URW's Nimbus Sans (OTF) locally and register
    fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts', 'nimbus_sans')
    os.makedirs(fonts_dir, exist_ok=True)

    font_files = {
        'NimbusSans-Regular.otf': 'https://github.com/ArtifexSoftware/urw-base35-fonts/raw/master/fonts/NimbusSans-Regular.otf',
        'NimbusSans-Bold.otf': 'https://github.com/ArtifexSoftware/urw-base35-fonts/raw/master/fonts/NimbusSans-Bold.otf',
        'NimbusSans-Italic.otf': 'https://github.com/ArtifexSoftware/urw-base35-fonts/raw/master/fonts/NimbusSans-Italic.otf',
        'NimbusSans-BoldItalic.otf': 'https://github.com/ArtifexSoftware/urw-base35-fonts/raw/master/fonts/NimbusSans-BoldItalic.otf',
    }

    for fname, url in font_files.items():
        local_path = os.path.join(fonts_dir, fname)
        if not os.path.exists(local_path):
            try:
                with urlopen(url, timeout=15) as resp:
                    data = resp.read()
                with open(local_path, 'wb') as f:
                    f.write(data)
            except Exception:
                # If download fails, continue; we'll fallback later
                continue
        _register_downloaded_font(local_path)

    # Recheck availability after attempted download
    available_names = {f.name for f in font_manager.fontManager.ttflist}
    for fam in preferred_families:
        if fam in available_names:
            return fam

    # Fallback to common default
    return 'DejaVu Sans'


def set_global_font():
    family = ensure_nimbus_sans()
    # Set rcParams to prefer Nimbus Sans (or fallback) across the figure
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [family, 'Nimbus Sans L', 'Nimbus Sans', 'DejaVu Sans', 'Arial', 'Liberation Sans']
    # Configure math text to use the same family (when possible)
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = family
    plt.rcParams['mathtext.it'] = family
    plt.rcParams['mathtext.bf'] = family
    plt.rcParams['mathtext.sf'] = family
    plt.rcParams['mathtext.default'] = 'regular'
    return family

# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def bin_by_max_hybrid(df):
    """Group annotations into bins based on max_hybrid score."""
    bins = {
        '>0.8': df[df['max_hybrid'] > 0.8],
        '0.7-0.8': df[(df['max_hybrid'] <= 0.8) & (df['max_hybrid'] > 0.6)],
        '0.6-0.7': df[(df['max_hybrid'] <= 0.8) & (df['max_hybrid'] > 0.6)],
        '<0.6': df[df['max_hybrid'] <= 0.6]
    }
    return bins

def add_hybrid_fraction_columns(df):
    """Add hybrid fraction columns to DataFrame."""
    # guard against divide-by-zero if needed
    eps = 1e-12
    denom = df['max_hybrid'].clip(lower=eps)
    df['hybrid_fraction_1']  = df['top_1_hybrid']       / denom
    df['hybrid_fraction_3']  = df['top_3_max_hybrid']   / denom
    df['hybrid_fraction_5']  = df['top_5_max_hybrid']   / denom
    df['hybrid_fraction_10'] = df['top_10_max_hybrid']  / denom
    return df

def _kde_max_density(values_list, grid=KDE_GRID):
    """
    Compute the maximum KDE density across a list of 1D arrays on a common grid.
    Uses scipy Gaussian KDE directly to pre-compute consistent y-lims.
    """
    ymax = 0.0
    for arr in values_list:
        arr = np.asarray(arr, float)
        arr = arr[(arr >= 0.0) & (arr <= 1.0)]
        if arr.size >= 2 and np.std(arr) > 0:
            kde = gaussian_kde(arr)
            y = kde(grid)
            ymax = max(ymax, float(np.max(y)))
    return ymax

# Helper to apply font properties explicitly to axes elements

def apply_font_to_axes(axes, family: str) -> None:
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    for ax in axes.flat:
        # Titles and axis labels: set family without touching size
        if ax.get_title():
            try:
                ax.title.set_fontfamily(family)
            except Exception:
                pass
        try:
            ax.xaxis.label.set_fontfamily(family)
            ax.yaxis.label.set_fontfamily(family)
        except Exception:
            pass
        # Tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            try:
                label.set_fontfamily(family)
            except Exception:
                pass
        # Legend text
        leg = ax.get_legend()
        if leg is not None:
            for text in leg.get_texts():
                try:
                    text.set_fontfamily(family)
                except Exception:
                    pass
        # Any text artists already on the axes
        for txt in ax.texts:
            try:
                txt.set_fontfamily(family)
            except Exception:
                pass

# =====================================================
# MAIN PLOTTING FUNCTION (KDE VERSION)
# =====================================================

def plot_comparison_kdes_2x3(bins_bestfinal, bins_nn, 
                             figsize=DEFAULT_FIGSIZE,
                             save_path=None,
                             show_plot=True):
    """
    2x3 grid of normalized KDEs comparing Mod-Hung vs Hung-NN for top-1 and top-5
    across Exact, Very Close, and Poor match bins.
    """
    # seaborn aesthetics (keeps your palette but improves defaults)
    sns.set_theme(context="talk", style="white")

    k_values   = [1, 5]  # Only top-1 and top-5
    bin_names  = ['>0.8', '0.7-0.8', '<0.6']  # Exact, Close, Poor matches
    bin_labels = ['Exact Match', 'Close Match', 'Poor Match']
    
    fig, axes = plt.subplots(2, 3, figsize=figsize, sharex='col')
    # fig.patch.set_edgecolor('black')
    # fig.patch.set_linewidth(4)

    # Determine font family (ensures Nimbus is available)
    family = ensure_nimbus_sans()

    # -------- First pass: compute global y-axis for consistent scaling --------
    all_row_col_ymax = {}
    for row, k in enumerate(k_values):
        for col, bin_name in enumerate(bin_names):
            col_name = f'hybrid_fraction_{k}'
            arrs = []
            if bin_name in bins_bestfinal and len(bins_bestfinal[bin_name]) > 0:
                arrs.append(bins_bestfinal[bin_name][col_name].dropna().values)
            if bin_name in bins_nn and len(bins_nn[bin_name]) > 0:
                arrs.append(bins_nn[bin_name][col_name].dropna().values)
            y_max = _kde_max_density(arrs, KDE_GRID)
            # add 10% headroom; if empty, fall back to 1.0
            all_row_col_ymax[(row, col)] = (y_max * 1.1) if y_max > 0 else 1.0

    # Use a uniform y-limit across all subplots
    uniform_ymax = max(all_row_col_ymax.values()) if len(all_row_col_ymax) > 0 else 1.0

    # -------- Second pass: draw KDEs --------
    for row, k in enumerate(k_values):
        for col, (bin_name, bin_label) in enumerate(zip(bin_names, bin_labels)):
            ax = axes[row, col]
            ax.set_facecolor(BACKGROUND_COLOR)

            # Data
            col_name = f'hybrid_fraction_{k}'
            data_best = bins_bestfinal.get(bin_name, pd.DataFrame())
            data_nn   = bins_nn.get(bin_name, pd.DataFrame())
            series_best = data_best[col_name] if col_name in data_best else pd.Series([], dtype=float)
            series_nn   = data_nn[col_name]   if col_name in data_nn   else pd.Series([], dtype=float)

            # Plot NN first so Mod-Hung overlays
            if series_nn.size > 1 and series_nn.std(ddof=0) > 0:
                sns.kdeplot(
                    x=series_nn.clip(0, 1),
                    ax=ax, bw_adjust=KDE_BW_ADJUST, clip=KDE_CLIP,
                    fill=True, alpha=ALPHA_FILL, linewidth=LINEWIDTH_KDE,
                    color=COLOR_HUNG_NN, label=f'Hung-NN (n={series_nn.size})'
                )
            if series_best.size > 1 and series_best.std(ddof=0) > 0:
                sns.kdeplot(
                    x=series_best.clip(0, 1),
                    ax=ax, bw_adjust=KDE_BW_ADJUST, clip=KDE_CLIP,
                    fill=True, alpha=ALPHA_FILL, linewidth=LINEWIDTH_KDE,
                    color=COLOR_MOD_HUNG, label=f'Mod-Hung (n={series_best.size})'
                )

            # Means + delta box (only if both present)
            if series_best.size > 0 and series_nn.size > 0:
                mean_best = float(series_best.mean())
                mean_nn   = float(series_nn.mean())
                improvement = mean_best - mean_nn
                # ax.text(
                #     0.25, 0.95,
                #     f'Mod-Hung μ={mean_best:.2f}\nHung-NN μ={mean_nn:.2f}\nΔ=+{improvement:.2f}',
                #     transform=ax.transAxes, va='top', ha='center', fontsize=8, =TEXTBOX_STYLE
                # )

            # Subplot labels (A–F)
            subplot_labels = ['A', 'B', 'C', 'D', 'E', 'F']
            label_idx = row * 3 + col
            label_text = ax.text(0.05, 0.91, subplot_labels[label_idx], transform=ax.transAxes,
                                  fontsize=24, ha='center', va='center')
            # Ensure family without overriding size
            try:
                label_text.set_fontfamily(family)
            except Exception:
                pass

            # Column headers (top row only)
            if row == 0:
                t = ax.set_title(f'{bin_label}', pad=5, fontsize=20)
                ax.set_xlabel('')
                try:
                    t.set_fontfamily(family)
                except Exception:
                    pass

            # Row headers (left column only)
            if col == 0:
                ax.set_ylabel(f'Top-{k} \n \n Density', fontsize=20)
                try:
                    ax.yaxis.label.set_fontfamily(family)
                except Exception:
                    pass
            else:
                # Remove y-axis label for non-leftmost plots
                ax.set_ylabel("")

            # X label for bottom row (all columns keep x-labels)
            if row == 1:
                ax.set_xlabel('${\\eta}$ (Struct. Eff.)', fontsize=20)
                try:
                    ax.xaxis.label.set_fontfamily(family)
                except Exception:
                    pass

            # Hide ONLY y-axis labels and ticks for non-leftmost columns
            # Turn off y-axis ticks/labels for all subplots
            ax.set_yticklabels([])
            # Ensure no axis offset text (e.g., stray '0') appears on the figure
            try:
                ax.ticklabel_format(axis='both', style='plain', useOffset=False)
            except Exception:
                pass
            try:
                ax.yaxis.get_major_formatter().set_useOffset(False)
                ax.xaxis.get_major_formatter().set_useOffset(False)
            except Exception:
                pass
            try:
                ax.yaxis.offsetText.set_visible(False)
                ax.xaxis.offsetText.set_visible(False)
            except Exception:
                pass
            # Ensure y-axis tick marks are visible
            ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=False,
                           direction='out', length=4, width=0.8)
            try:
                ax.yaxis.set_ticks_position('left')
            except Exception:
                pass
            # Ensure the left spine is visible so ticks anchor correctly
            try:
                ax.spines['left'].set_visible(True)
            except Exception:
                pass

            # Keep x-axis ticks for all; hide labels on the top row
            if row == 0:
                ax.tick_params(axis='x', which='both', bottom=True, labelbottom=False)
            else:
                ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)

            # Styling
            ax.set_xlim(0.01, 1.02)
            ax.set_ylim(0, uniform_ymax+2.7)
            ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_xticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=18)
            ax.set_xticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=18)
            ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE,
                    linewidth=GRID_LINEWIDTH, which='major', axis='x')
            for spine in ax.spines.values():
                spine.set_edgecolor(SPINE_COLOR)
                spine.set_linewidth(SPINE_LINEWIDTH)

            # Legends (one per column, top row — to keep space below clear)
            if row == 0:
                ax.legend(
                    fontsize=LEGEND_FONTSIZE,
                    loc='upper right',
                    framealpha=0.9,
                    ncol=1,)
                #     labelspacing=0.05,     # vertical spacing between entries
                #     borderpad=0.2,         # padding inside the legend box
                #     handletextpad=0.2,     # spacing between handles and text
                #     borderaxespad=0.1      # padding between axes and legend box
                # )
                # Ensure legend text uses the desired font
                leg = ax.get_legend()
                if leg is not None:
                    for text in leg.get_texts():
                        try:
                            text.set_fontfamily(family)
                        except Exception:
                            pass

    # Apply font to any remaining elements
    apply_font_to_axes(axes, family)

    # Remove vertical spacing between rows and set zero horizontal space
    plt.subplots_adjust(hspace=0.0, wspace=DEFAULT_WSPACE, bottom=0.2)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show_plot:
        plt.show()

    return fig

# =====================================================
# DATA LOADING AND PROCESSING FUNCTIONS
# =====================================================

def load_annotation_data(annotations_path, experimental_path):
    """
    Load and process annotation data from pickle files.
    Returns (df_bestfinal, bins_bestfinal)
    """
    with open(annotations_path, 'rb') as f:
        annotations = pickle.load(f)
    with open(experimental_path, 'rb') as f:
        experimental_dict = pickle.load(f)
    
    query_files = []
    experimental_hsqc = []
    top_k = []
    for annotation in annotations:
        query = annotation[0]
        hsqc = experimental_dict[query][3]
        if len(hsqc) > 5:  # Only keep HSQCs with more than 5 peaks
            query_files.append(query)
            experimental_hsqc.append(hsqc)
            top_k.append(annotation[2])
    input_data = list(zip(query_files, experimental_hsqc, top_k))
    df_bestfinal = summarize_annotations(input_data)
    df_bestfinal = add_hybrid_fraction_columns(df_bestfinal)
    bins_bestfinal = bin_by_max_hybrid(df_bestfinal)
    return df_bestfinal, bins_bestfinal

def summarize_annotations(annotations):
    """
    Generate a summary DataFrame of annotation statistics.
    annotations: list of (query_file, query_hsqc, top_k_matches_list)
    """
    records = []
    for query_file, query_hsqc, top_k_matches in annotations:
        num_peaks = query_hsqc.shape[0]
        file_id = (
            query_file.split('/')[-1]
                      .split('.')[0]
                      .split(' ')[0]
                      .split('_')[0]
        )
        dfm = pd.DataFrame(top_k_matches)
        dfm['hybrid_score'] = (dfm['mcs_score_d'] + dfm['tanimoto_similarity']) / 2
        rec = {
            'query_file':            file_id,
            'num_peaks':             num_peaks,
            'top_1_tanimoto':        dfm['tanimoto_similarity'].iloc[0],
            'top_1_hybrid':          dfm['hybrid_score'].iloc[0],
            'top_3_max_tanimoto':    dfm['tanimoto_similarity'].iloc[:3].max(),
            'top_3_max_hybrid':      dfm['hybrid_score'].iloc[:3].max(),
            'top_5_max_tanimoto':    dfm['tanimoto_similarity'].iloc[:5].max(),
            'top_5_max_hybrid':      dfm['hybrid_score'].iloc[:5].max(),
            'top_10_max_tanimoto':   dfm['tanimoto_similarity'].iloc[:10].max(),
            'top_10_max_hybrid':     dfm['hybrid_score'].iloc[:10].max(),
            'max_tanimoto':          dfm['tanimoto_similarity'].max(),
            'max_hybrid':            dfm['hybrid_score'].max(),
            'k_max_tani':            int(dfm['tanimoto_similarity'].idxmax()) + 1,
            'k_max_hybrid':          int(dfm['hybrid_score'].idxmax()) + 1,
        }
        records.append(rec)
    return pd.DataFrame(records)

# =====================================================
# MAIN EXECUTION FUNCTION
# =====================================================

def main():
    print("2x3 KDE Plotter for Annotation Comparison")
    print("=" * 50)

    # Ensure Nimbus Sans is available and set globally
    family = set_global_font()

    annotations_path = '/home/cailum.stienstra/HSQC_Models/Networking_HSQC/ablation_results/expt_annotations_1.0_hybrid_modified_hungarian_100k_BEST.pkl'
    experimental_path = '/home/cailum.stienstra/HSQC_Models/Networking_HSQC/expt_filtered_lookup.pkl'
    hung_nn_path      = '/home/cailum.stienstra/HSQC_Models/Networking_HSQC/ablation_results/expt_annotations_1.0_hybrid_hungarian_nn_mean_100k_BEST.pkl'

    print("Loading annotation data...")
    df_bestfinal, bins_bestfinal = load_annotation_data(annotations_path, experimental_path)
    df_hung_nn,   bins_nn        = load_annotation_data(hung_nn_path, experimental_path)

    print("Creating 2x3 KDE plot...")
    fig = plot_comparison_kdes_2x3(
        bins_bestfinal, bins_nn,
        figsize=(14, 8),
        save_path='/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/figures/paper_figures/Figure2_August20_KDE.png',
        show_plot=True
    )
    print("Plot created successfully!")

if __name__ == "__main__":
    main()
