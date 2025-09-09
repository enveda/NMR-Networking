import pandas as pd
import numpy as np
import sys
import random
import pickle
from tabulate import tabulate
from io import BytesIO

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

import matplotlib.pyplot as plt
from IPython.display import display, SVG
import cairosvg

def raise_exception(msg):
    """
    Helper function to raise an exception with a custom message.
    
    Args:
        msg (str): Message to display in the exception
    
    Raises:
        Exception: Always raises an exception with the provided message
    """
    raise Exception(msg)


def summarize_annotations(annotations):
    """
    Generate a summary DataFrame of annotation statistics.

    Parameters:
        annotations (list of tuples): each tuple is 
            (query_file_path, query_hsqc, top_k_matches_list)
            - query_file_path : str
            - query_hsqc       : array-like or DataFrame with .shape[0] = # peaks
            - top_k_matches_list: list of dicts with keys 'mcs_score_d' and 'tanimoto_similarity'

    Returns:
        pandas.DataFrame with columns:
            query_file,
            num_peaks,
            top_1_tanimoto,
            top_1_hybrid,
            top_3_max_tanimoto,
            top_3_max_hybrid,
            top_5_max_tanimoto,
            top_5_max_hybrid,
            top_10_max_tanimoto,
            top_10_max_hybrid,
            max_tanimoto,
            max_hybrid,
            k_max_tani,
            k_max_hybrid
    """
    records = []
    for query_file, query_hsqc, top_k_matches in annotations:
        # number of peaks directly from the HSQC object
        num_peaks = query_hsqc.shape[0]
        # simplify file name to an ID
        file_id = (
            query_file.split('/')[-1]
                      .split('.')[0]
                      .split(' ')[0]
                      .split('_')[0]
        )
        
        # build DataFrame of matches and compute hybrid score
        dfm = pd.DataFrame(top_k_matches)
        dfm['hybrid_score'] = (dfm['mcs_score_d'] + dfm['tanimoto_similarity']) / 2
        
        # gather all metrics
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


def plot_hybrid_histograms(bins, column, xlabel):
    """
    Plot a 2×2 grid of histograms for each bin in `bins`, drawing the specified `column`.
    
    Parameters
    ----------
    bins : dict
        Mapping from bin name → DataFrame.
    column : str
        Which DataFrame column to histogram (e.g. 'top_1_hybrid', 'top_3_max_hybrid', …).
    xlabel : str
        Label for the x-axis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Add thick black border to entire figure
    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(4)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, ((bin_name, df), ax) in enumerate(zip(bins.items(), axes.flat)):
        # Histogram
        ax.hist(df[column], bins=20, edgecolor='black',
                alpha=0.7, color=colors[i])

        # Titles & labels
        ax.set_title(f'Max Hybrid Score: {bin_name}   (n={len(df)})')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_facecolor('#f0f0f0')

        # White border
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(2)

        # Mean line (always using top_1_hybrid mean, per original code)
        μ = df[column].mean()
        ax.axvline(μ, linestyle='--', alpha=0.8, color='red')
        ax.text(0.18, 0.95, f'μ = {μ:.2f}',
                transform=ax.transAxes,
                horizontalalignment='right',
                verticalalignment='top')

    plt.show()

def plot_hybrid_fraction_histograms(bins, k):
    """
    Plot a 2×2 grid of histograms showing the fraction of max hybrid score
    realized by the top-k hit for each bin.

    Parameters
    ----------
    bins : dict
        Mapping from bin name → DataFrame (must already contain 'hybrid_fraction_{k}' columns).
    k : int
        Which top-k fraction to plot (1, 3, 5, or 10).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Add thick black border to entire figure
    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(4)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, ((bin_name, df), ax) in enumerate(zip(bins.items(), axes.flat)):
        col = f'hybrid_fraction_{k}'
        ax.hist(df[col], bins=20, edgecolor='black',
                alpha=0.7, color=colors[i])

        ax.set_title(f'Max Hybrid {bin_name}   (n={len(df)})')
        ax.set_xlabel(f'Fraction of Max Hybrid Score\nRealized by Top-{k} Hit')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_facecolor('#f0f0f0')

        # white border
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(2)

        μ = df[col].mean()
        n_below = len(df[df[col] < 0.80])
        ax.axvline(μ, color='red', linestyle='--', alpha=0.8)
        ax.text(0.14, 0.95, f'μ = {μ:.2f}\nn < 0.80: {n_below}',
                transform=ax.transAxes,
                horizontalalignment='center',
                verticalalignment='top')

    plt.show()


def display_two_column_annotations(query_file, query_smiles, annotations,
                                   top_k=5, hybrid_k=5, num_peaks=None,
                                   sub_img_size=(400, 400),
                                   legend_font_size=8, save_path=None):
    """
    Three-column layout with separate text boxes under each molecule:
      • Column 0: query
      • Column 1: top_k in original order
      • Column 2: top hybrid_k by hybrid score
    """
    # Prepare DataFram
    df = annotations

    # Ensure hybrid column
    if 'hybrid' not in df.columns:
        if 'hybrid_score' in df.columns:
            df = df.rename(columns={'hybrid_score': 'hybrid'})
        else:
            df['hybrid'] = (df['tanimoto_similarity'] + df['mcs_score_d']) / 2.0

    # Split
    mid_df = df.head(top_k).reset_index(drop=True)
    right_df = df.nlargest(hybrid_k, 'hybrid').reset_index(drop=True)

    # RDKit options (no internal legends)
    opts = rdMolDraw2D.MolDrawOptions()
    opts.drawLegend = False

    def _image_and_legends(sub_df, include_index=False):
        mols = [Chem.MolFromSmiles(s) for s in sub_df.lookup_smiles]
        legends = [
            f"{i+1}. k={int(row.k)}, Dist={row.hungarian_distance:.1f}\n"
            f"Tan={row.tanimoto_similarity:.2f}, MCS={row.mcs_score_d:.2f}\n"
            f"Hybrid={row.hybrid:.2f}"
            for i, row in enumerate(sub_df.itertuples())
        ]
        grid = Draw.MolsToGridImage(
            mols,
            molsPerRow=1,
            subImgSize=sub_img_size,
            drawOptions=opts,
            useSVG=True
        )
        png = cairosvg.svg2png(bytestring=grid.data.encode('utf-8'))
        img = plt.imread(BytesIO(png))
        return img, legends

    # Prepare images & legends
    # Query image (no legend boxes)
    q_mol = Chem.MolFromSmiles(query_smiles)
    q_grid = Draw.MolsToGridImage([q_mol], molsPerRow=1, subImgSize=sub_img_size,
                                  drawOptions=opts, useSVG=True)
    q_png = cairosvg.svg2png(bytestring=q_grid.data.encode('utf-8'))
    q_img = plt.imread(BytesIO(q_png))

    mid_img, mid_legends = _image_and_legends(mid_df)
    right_img, right_legends = _image_and_legends(right_df)

    # Plot
    height = max(1, top_k, hybrid_k) * (sub_img_size[1] / 100)
    fig, axes = plt.subplots(1, 3, figsize=(6, height), constrained_layout=True)
    
    # Add thick black border to entire figure
    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(4)
    
    for ax in axes:
        ax.axis('off')
    axQ, axM, axH = axes

    # Show query
    axQ.imshow(q_img)
    query_file = query_file.split('/')[-1].split('.')[0].split(' ')[0].split('_')[0]
    axQ.set_title(f"Query: {query_file}\n Num Peaks: {num_peaks}", fontsize=legend_font_size + 2)

    # Show mid and annotate
    axM.imshow(mid_img)
    axM.set_title(f"Top {top_k}", fontsize=legend_font_size + 4)
    cell_height = mid_img.shape[0] / len(mid_legends)
    for i, text in enumerate(mid_legends):
        y = (i + 1) * cell_height - legend_font_size * 1.2 - 15
        x = mid_img.shape[1] / 2
        axM.text(x, y, text, ha='center', va='top',
                 fontsize=legend_font_size,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # Show hybrid and annotate
    axH.imshow(right_img)
    axH.set_title(f"Hybrid Top {hybrid_k}", fontsize=legend_font_size + 2)
    cell_height = right_img.shape[0] / len(right_legends)
    for i, text in enumerate(right_legends):
        y = (i + 1) * cell_height - legend_font_size * 1.2 - 15
        x = right_img.shape[1] / 2
        axH.text(x, y, text, ha='center', va='top',
                 fontsize=legend_font_size,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

# Shuffle annotations randomly
