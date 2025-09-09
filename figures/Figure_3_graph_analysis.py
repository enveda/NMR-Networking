#!/usr/bin/env python3
"""
Figure 3: Graph Analysis with Mass Differences
==============================================

This script loads a graph with mass differences and creates comprehensive histograms
showing the distributions of all edge features and node degree statistics.

Edge features analyzed:
- Hungarian distance (weight)
- Tanimoto similarity
- MCS similarity  
- Hybrid similarity
- Mass difference

Node features analyzed:
- Node degree distribution
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx
from collections import Counter
from matplotlib.ticker import MaxNLocator, LogLocator
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

# Add the graph module to path
sys.path.append(os.path.dirname(__file__))

from graph.graph_utils import GraphBuilder

# ===== Consistent style with Figure 2 =====
# Colors and styling borrowed from figures/FIGURE2_PAPER.py
COLOR_MOD_HUNG = '#f3d39c'      # Deep orange for Mod-Hung
COLOR_HUNG_NN  = '#8199a5'      # Muted blue-gray for Hung-NN
ALPHA_FILL     = 0.28           # Transparency for filled KDE areas
LINEWIDTH_KDE  = 2.0

BACKGROUND_COLOR = '#ffffff'
SPINE_COLOR      = '#cccccc'
SPINE_LINEWIDTH  = 1

GRID_ALPHA     = 0.3
GRID_LINESTYLE = '--'
GRID_LINEWIDTH = 0.8

# KDE parameters aligned with Figure 2
KDE_BW_ADJUST = 1.0
KDE_CLIP = (0.0, 1.0)

# Build a compatible colormap for histograms blending the two key colors
HIST_CMAP = LinearSegmentedColormap.from_list('figure2_hist', [COLOR_HUNG_NN, COLOR_MOD_HUNG])

# Distinct qualitative palette for KDE curves (reds, greens, purples, etc.)
KDE_PALETTE = [
    '#0086c0',  # red'
    '#ecae00',  # green
    '#9467bd',  # purple
    '#ff7f0e',  # orange
    '#00ab86',  # cyan
    '#8c564b',  # brown
]

    
# Use a consistent, slightly-rectangular box aspect (height/width)
SUBPLOT_BOX_ASPECT = 0.85


# ---------- Font utilities (mirroring Figure 2) ----------
from matplotlib import font_manager
from urllib.request import urlopen

def _register_downloaded_font(font_path: str) -> None:
    try:
        font_manager.fontManager.addfont(font_path)
        font_manager._rebuild()
    except Exception:
        pass


def ensure_nimbus_sans() -> str:
    preferred_families = ["Nimbus Sans", "Nimbus Sans L"]
    available_names = {f.name for f in font_manager.fontManager.ttflist}
    for fam in preferred_families:
        if fam in available_names:
            return fam

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
                continue
        _register_downloaded_font(local_path)

    available_names = {f.name for f in font_manager.fontManager.ttflist}
    for fam in preferred_families:
        if fam in available_names:
            return fam
    return 'DejaVu Sans'


def set_global_font():
    family = ensure_nimbus_sans()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [family, 'Nimbus Sans L', 'Nimbus Sans', 'DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = family
    plt.rcParams['mathtext.it'] = family
    plt.rcParams['mathtext.bf'] = family
    plt.rcParams['mathtext.sf'] = family
    plt.rcParams['mathtext.default'] = 'regular'
    return family


def apply_font_to_axes(axes, family: str) -> None:
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    for ax in axes.flat:
        # Skip axes that have been removed from the figure
        if getattr(ax, 'figure', None) is None:
            continue
        # Titles/labels: set family only
        try:
            ax.title.set_fontfamily(family)
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
        # Any text artists
        for txt in ax.texts:
            try:
                txt.set_fontfamily(family)
            except Exception:
                pass


def set_figure2_style() -> None:
    """Apply styling consistent with Figure 2 using matplotlib rcParams only."""
    plt.rcParams['axes.facecolor']   = BACKGROUND_COLOR
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.edgecolor'] = 'white'
    plt.rcParams['axes.edgecolor']   = SPINE_COLOR
    plt.rcParams['axes.linewidth']   = SPINE_LINEWIDTH


def apply_axes_style(ax: plt.Axes) -> None:
    """Style a single axes like Figure 2 (background, grid, spines)."""
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE,
            linewidth=GRID_LINEWIDTH, which='major', axis='x')
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)
        spine.set_linewidth(SPINE_LINEWIDTH)


def get_kde_palette(num_colors: int):
    """Return a list of distinct qualitative colors from KDE_PALETTE."""
    if num_colors <= 0:
        return []
    repeats = (num_colors + len(KDE_PALETTE) - 1) // len(KDE_PALETTE)
    return (KDE_PALETTE * repeats)[:num_colors]


def plot_kde_only(ax: plt.Axes,
                  data: np.ndarray,
                  is_similarity: bool,
                  color: str) -> None:
    """Plot a KDE only (no histogram) using scipy gaussian_kde (Figure 2-like style)."""
    arr = np.asarray(data, dtype=float)
    if is_similarity:
        arr = np.clip(arr, 0.0, 1.0)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1 or np.std(arr) == 0:
        return
    kde = gaussian_kde(arr)
    # Apply bandwidth adjustment similar to seaborn's bw_adjust
    try:
        kde.set_bandwidth(kde.scotts_factor() * KDE_BW_ADJUST)
    except Exception:
        factor = kde.scotts_factor() * KDE_BW_ADJUST
        kde.covariance_factor = lambda: factor
        kde._compute_covariance()
    # Restrict evaluation strictly to observed data bounds to avoid smoothing
    # beyond hard cutoffs/thresholds (akin to seaborn's cut=0 behavior).
    x_min, x_max = float(np.min(arr)), float(np.max(arr))
    if np.isclose(x_min, x_max):
        x_max = x_min + 1e-6
    x_grid = np.linspace(x_min, x_max, 512)
    y = kde.evaluate(x_grid)
    ax.fill_between(x_grid, y, color=color, alpha=ALPHA_FILL)
    ax.plot(x_grid, y, color=color, linewidth=LINEWIDTH_KDE)


# Removed seaborn-dependent log-kde helper (not used)

# Figure 2 sets fonts via seaborn theme; no custom font plumbing needed here


def load_graph_with_mass_differences(graph_path: str) -> GraphBuilder:
    """
    Load a graph that contains mass differences.
    
    Parameters:
    -----------
    graph_path : str
        Path to the graph file with mass differences
        
    Returns:
    --------
    GraphBuilder : Loaded graph builder object
    """
    print(f"📁 Loading graph from: {graph_path}")
    
    builder = GraphBuilder()
    builder.load_graph(graph_path)
    
    print(f"📊 Graph loaded successfully!")
    print(f"   Nodes: {builder.graph.number_of_nodes():,}")
    print(f"   Edges: {builder.graph.number_of_edges():,}")
    
    return builder


def extract_edge_features(graph: nx.Graph) -> dict:
    """
    Extract all edge features from the graph.
    
    Parameters:
    -----------
    graph : nx.Graph
        NetworkX graph with edge attributes
        
    Returns:
    --------
    dict : Dictionary containing arrays of each edge feature
    """
    print("🔍 Extracting edge features...")
    
    # Initialize feature arrays
    features = {
        'hungarian_distance': [],
        'tanimoto_similarity': [],
        'mcs_similarity': [],
        'hybrid_similarity': [],
        'mass_difference': []
    }
    
    # Extract features from each edge
    for node1, node2, data in graph.edges(data=True):
        features['hungarian_distance'].append(data.get('weight', 0.0))
        features['tanimoto_similarity'].append(data.get('tanimoto_similarity', 0.0))
        features['mcs_similarity'].append(data.get('mcs_similarity', 0.0))
        features['hybrid_similarity'].append(data.get('hybrid_similarity', 0.0))
        features['mass_difference'].append(data.get('mass_difference', 0.0))
    
    # Convert to numpy arrays
    for key in features:
        features[key] = np.array(features[key])
    
    print(f"✅ Extracted {len(features['hungarian_distance']):,} edge features")
    
    return features


def extract_node_features(graph: nx.Graph) -> dict:
    """
    Extract node degree information from the graph.
    
    Parameters:
    -----------
    graph : nx.Graph
        NetworkX graph
        
    Returns:
    --------
    dict : Dictionary containing node degree statistics
    """
    print("🔍 Extracting node features...")
    
    # Calculate node degrees
    degrees = [graph.degree(node) for node in graph.nodes()]
    degrees = np.array(degrees)
    
    # Calculate degree statistics
    degree_stats = {
        'degrees': degrees,
        'mean_degree': np.mean(degrees),
        'median_degree': np.median(degrees),
        'std_degree': np.std(degrees),
        'min_degree': np.min(degrees),
        'max_degree': np.max(degrees),
        'degree_distribution': Counter(degrees)
    }
    
    print(f"✅ Extracted degree information for {len(degrees):,} nodes")
    print(f"   Mean degree: {degree_stats['mean_degree']:.2f}")
    print(f"   Median degree: {degree_stats['median_degree']:.2f}")
    print(f"   Degree range: {degree_stats['min_degree']} - {degree_stats['max_degree']}")
    
    return degree_stats


def create_edge_feature_histograms(edge_features: dict, output_dir: str = "figures"):
    """
    Create histograms for all edge features.
    
    Parameters:
    -----------
    edge_features : dict
        Dictionary containing arrays of edge features
    output_dir : str
        Output directory for saving figures
    """
    print("📊 Creating edge feature histograms...")
    
    # Apply Figure 2 styling and fonts
    set_figure2_style()
    family = set_global_font()
    
    # Set up the figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot each feature using KDE-only with distinct colors
    features_to_plot = [
        ('hungarian_distance', 'Mod. Hung.', axes[0, 0]),
        ('tanimoto_similarity', 'Tanimoto Similarity', axes[0, 1]),
        ('mcs_similarity', 'MCS Similarity', axes[0, 2]),
        ('hybrid_similarity', 'Hybrid Similarity', axes[1, 0]),
        ('mass_difference', 'Mass Difference (Da)', axes[1, 1])
    ]

    # Use distinct qualitative colors instead of sampling between two
    kde_colors = KDE_PALETTE[:len(features_to_plot)]

    for (feature_key, title, ax), color in zip(features_to_plot, kde_colors):
        if feature_key in edge_features:
            data = edge_features[feature_key]
            
            # Filter out zeros for similarity metrics
            is_similarity = feature_key in {'tanimoto_similarity', 'mcs_similarity', 'hybrid_similarity', 'modified_hungarian_distance'}
            if is_similarity:
                non_zero_data = data[data > 0]
                data_to_plot = non_zero_data if len(non_zero_data) > 0 else data
            else:
                data_to_plot = data
            
            apply_axes_style(ax)
            ax.set_box_aspect(SUBPLOT_BOX_ASPECT)
            plot_kde_only(ax, data_to_plot, is_similarity=is_similarity, color=color)
            # Labels and ticks
            ax.set_xlabel(title, fontsize=32)
            ax.set_ylabel('Density', fontsize=32)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.tick_params(axis='both', which='major', labelsize=24, width=1.0, length=5)

            # Hide y-tick labels but keep tick marks (to match Figure 2 behavior)
            ax.set_yticklabels([])
            ax.tick_params(axis='y', which='both', left=True, labelleft=False)

            # Statistics box
            mean_val = float(np.mean(data_to_plot)) if np.size(data_to_plot) else 0.0
            std_val = float(np.std(data_to_plot)) if np.size(data_to_plot) else 0.0
            median_val = float(np.median(data_to_plot)) if np.size(data_to_plot) else 0.0
            if title == 'Hybrid Similarity':
            
                ax.set_xlim(0.6, 1.005)
                ax.text(
                    0.65,
                    0.95,
                    f'μ: {mean_val:.2f}\nσ: {std_val:.2f}\nMedian: {median_val:.2f}',
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                    ha='center', fontsize=30
                )
            else:
                # Place stats box; top-right for Modified Hungarian, default elsewhere
                if title == 'Mod. Hung.':
                    pos_x = 0.3
                    ha_align = 'center'
                    ax.set_xlim(0, 25)
                else:
                    pos_x = (0.3 if is_similarity else 0.65)
                    ha_align = 'center'
                ax.text(
                    pos_x,
                    0.95,
                    f'μ: {mean_val:.2f}\nσ: {std_val:.2f}\nMedian: {median_val:.2f}',
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                    ha=ha_align, fontsize=30   
                )
            # Increase x-axis tick label size
            ax.tick_params(axis='x', labelsize=30)
                
    
    # Remove the last subplot (bottom right)
    axes[1, 2].remove()
    
    # Apply font family to active axes only (skip removed)
    apply_font_to_axes(np.array(fig.axes), family)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir) / "edge_feature_histograms.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Edge feature histograms saved to: {output_path}")
    
    plt.show()


def create_node_degree_histogram(degree_stats: dict, output_dir: str = "figures", graph: nx.Graph = None):
    """
    Create histogram for node degree distribution.
    
    Parameters:
    -----------
    degree_stats : dict
        Dictionary containing node degree statistics
    output_dir : str
        Output directory for saving figures
    graph : nx.Graph, optional
        NetworkX graph object for additional statistics
    """
    print("📊 Creating node degree histogram...")
    
    # Apply Figure 2 styling and fonts
    set_figure2_style()
    family = set_global_font()
    
    # Set up the figure (square subplots)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    degrees = degree_stats['degrees']
    
    # Main degree KDE-only
    apply_axes_style(ax1)
    plot_kde_only(ax1, degrees, is_similarity=False, color=COLOR_MOD_HUNG)
    ax1.set_xlabel('Node Degree', fontsize=28, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=28, fontweight='bold')
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax1.tick_params(axis='both', which='major', labelsize=24, width=1.0, length=5)
    # Make subplot slightly rectangular and consistent with edge plots
    ax1.set_box_aspect(SUBPLOT_BOX_ASPECT)
    
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    # Add statistics text with Greek symbols
    stats_text = f"μ: {degree_stats['mean_degree']:.2f}\n"
    stats_text += f"Median: {degree_stats['median_degree']:.2f}\n"
    stats_text += f"σ: {degree_stats['std_degree']:.2f}"
    stats_text += f"\nNodes: {n_nodes:,}\n"
    stats_text += f"Edges: {n_edges:,}"
    ax1.text(0.25, 0.95, stats_text, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), 
             fontsize=28, ha='center')
    
    # Secondary subplot: histogram with log-scaled y-axis (style similar to KDE)
    apply_axes_style(ax2)
    n, bins, patches = ax2.hist(
        degrees,
        bins=50,
        density=False,
        color='#00ab86',
        alpha=0.5,
        edgecolor='none',
        linewidth=0.0,
    )
    ax2.set_yscale('log')
    # Reduce number of y-axis ticks on the log scale and turn off minor ticks
    ax2.yaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
    ax2.minorticks_off()
    ax2.set_xlabel('Node Degree', fontsize=32)
    ax2.set_ylabel('Log10(Frequency)', fontsize=32)
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax2.tick_params(axis='both', which='major', labelsize=30, width=1.0, length=5)
    ax2.set_box_aspect(SUBPLOT_BOX_ASPECT)
    ax2.text(0.7, 0.95, stats_text, transform=ax2.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), 
            fontsize=28, ha='center')
    
    # Apply font family
    apply_font_to_axes(np.array([[ax1, ax2]]), family)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = Path(output_dir) / "node_degree_histogram.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Node degree histogram saved to: {output_path}")
    
    plt.show()


def print_feature_statistics(edge_features: dict, degree_stats: dict):
    """
    Print comprehensive statistics for all features.
    
    Parameters:
    -----------
    edge_features : dict
        Dictionary containing arrays of edge features
    degree_stats : dict
        Dictionary containing node degree statistics
    """
    print("\n" + "="*60)
    print("📈 COMPREHENSIVE FEATURE STATISTICS")
    print("="*60)
    
    # Edge feature statistics
    print("\n🔗 EDGE FEATURES:")
    print("-" * 40)
    
    for feature_name, data in edge_features.items():
        print(f"\n{feature_name.replace('_', ' ').title()}:")
        print(f"  Count: {len(data):,}")
        print(f"  Mean: {np.mean(data):.3f}")
        print(f"  Median: {np.median(data):.3f}")
        print(f"  Std: {np.std(data):.3f}")
        print(f"  Min: {np.min(data):.3f}")
        print(f"  Max: {np.max(data):.3f}")
        
        # For similarity metrics, show non-zero statistics
        if 'similarity' in feature_name:
            non_zero = data[data > 0]
            if len(non_zero) > 0:
                print(f"  Non-zero count: {len(non_zero):,}")
                print(f"  Non-zero mean: {np.mean(non_zero):.3f}")
                print(f"  Non-zero median: {np.median(non_zero):.3f}")
    
    # Node degree statistics
    print("\n🔄 NODE DEGREE STATISTICS:")
    print("-" * 40)
    degrees = degree_stats['degrees']
    print(f"  Total nodes: {len(degrees):,}")
    print(f"  Mean degree: {degree_stats['mean_degree']:.2f}")
    print(f"  Median degree: {degree_stats['median_degree']:.2f}")
    print(f"  Std degree: {degree_stats['std_degree']:.2f}")
    print(f"  Min degree: {degree_stats['min_degree']}")
    print(f"  Max degree: {degree_stats['max_degree']}")
    
    # Degree distribution summary
    degree_counts = degree_stats['degree_distribution']
    print(f"  Unique degrees: {len(degree_counts)}")
    print(f"  Most common degree: {degree_counts.most_common(1)[0] if degree_counts else 'N/A'}")
    
    print("\n" + "="*60)


def main():
    """
    Main function to run the complete graph analysis.
    """
    print("🧪 Figure 3: Graph Analysis with Mass Differences")
    print("=" * 60)
    
    # Use same global style and fonts as Figure 2
    set_figure2_style()
    set_global_font()
    
    # Define paths
    graph_path = '/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/storage/GRAPH_30_0.6_LEIDEN_FINAL_100k_with_mass_diff.pkl'
    output_dir = "/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/figures/paper_figures"
    
    # Check if graph file exists
    if not Path(graph_path).exists():
        print(f"❌ Graph file not found: {graph_path}")
        print("Please ensure the graph file with mass differences exists.")
        return
    
    try:
        # Load the graph
        builder = load_graph_with_mass_differences(graph_path)
        
        # Extract features
        edge_features = extract_edge_features(builder.graph)
        degree_stats = extract_node_features(builder.graph)
        
        # Print comprehensive statistics
        print_feature_statistics(edge_features, degree_stats)
        
        # Create histograms
        create_edge_feature_histograms(edge_features, output_dir)
        create_node_degree_histogram(degree_stats, output_dir, graph=builder.graph)
        
        print("\n✅ Analysis completed successfully!")
        print(f"📁 Figures saved to: {output_dir}/")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 