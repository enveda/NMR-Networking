"""
Efficiency dataset generation utilities.

This module ports the functionality of demo/annotate/efficiency_lookup_ablation.py
into the installable package, providing:

- sample_experiments: sample N experimental queries from a pickle lookup
- build_lookup_subsets: create nested random subsets from a MNova lookup parquet or dict

Outputs are designed for ablation/efficiency testing.
"""

from __future__ import annotations

import ast
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def ensure_dir(path: Path | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def sample_experiments(
    experimental_pkl_path: Path | str,
    output_dir: Path | str,
    sample_size: int = 500,
    seed: int = 42,
) -> Path:
    """Sample experimental spectra and write to a static pickle file.

    Returns the output path of the written pickle.
    """
    experimental_pkl_path = Path(experimental_pkl_path)
    output_dir = Path(output_dir)
    if not experimental_pkl_path.exists():
        raise FileNotFoundError(f"Experimental pickle not found: {experimental_pkl_path}")

    with open(experimental_pkl_path, "rb") as f:
        experimental_dict: Dict[str, Any] = pickle.load(f)

    all_keys: List[str] = list(experimental_dict.keys())
    total = len(all_keys)
    if total == 0:
        raise ValueError("Experimental dictionary is empty.")

    rng = np.random.default_rng(seed)
    actual_n = min(sample_size, total)
    selected_indices = rng.choice(total, size=actual_n, replace=False)
    selected_keys = [all_keys[i] for i in selected_indices]

    sampled_dict: Dict[str, Any] = {k: experimental_dict[k] for k in selected_keys}

    ensure_dir(output_dir)
    out_path = output_dir / f"experiments_sample_{actual_n}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(sampled_dict, f)

    return out_path


def _iter_lookup_rows_from_parquet(parquet_path: Path | str) -> Iterable[Tuple[str, Dict[str, Any]]]:
    df = pd.read_parquet(parquet_path)
    # Expect columns id, smiles, hsqc
    for row in df.itertuples(index=False):
        key = str(row.id)
        try:
            hsqc = row.hsqc
            if isinstance(hsqc, str):
                hsqc_list = ast.literal_eval(hsqc)
            else:
                hsqc_list = hsqc
        except Exception:
            hsqc_list = None
        yield key, {
            'smiles': [getattr(row, 'smiles', None)],
            'gt': [np.array(hsqc_list, dtype=float) if hsqc_list is not None else np.empty((0, 2))],
        }


def build_lookup_subsets(
    mnova_lookup: Path | str | Dict[str, Dict[str, Any]],
    output_dir: Path | str,
    sizes: List[int],
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate nested random subsets from a lookup source (parquet or dict).

    Returns a manifest describing the permutation and subset files.
    """
    output_dir = Path(output_dir)
    subsets_dir = output_dir / 'lookup_subsets'
    ensure_dir(subsets_dir)

    # Load source into a list of (key, row_dict)
    if isinstance(mnova_lookup, (str, Path)):
        source_items = list(_iter_lookup_rows_from_parquet(mnova_lookup))
    else:
        source_items = list(mnova_lookup.items())

    total = len(source_items)
    if total == 0:
        raise ValueError("Lookup source is empty.")

    # Single permutation to ensure nested supersets
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total)
    keys_perm = [source_items[i][0] for i in perm]

    manifest = {
        'total': total,
        'seed': seed,
        'sizes': sizes,
        'permutation_first_20': keys_perm[:20],
        'subsets': [],
    }

    # Write subsets
    for n in sizes:
        n_eff = min(n, total)
        selected_keys = keys_perm[:n_eff]
        subset_dict = {k: dict(source_items[perm[i]][1]) for i, k in enumerate(selected_keys)}
        out_path = subsets_dir / f'lookup_subset_{n_eff}.pkl'
        with open(out_path, 'wb') as f:
            pickle.dump(subset_dict, f)
        manifest['subsets'].append({'size': n_eff, 'path': str(out_path)})

    # Save manifest
    with open(output_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main():
    import argparse
    p = argparse.ArgumentParser(description='Generate efficiency datasets (experiment samples and lookup subsets).')
    p.add_argument('--experimental-pkl', required=True, help='Path to experimental pickle dict')
    p.add_argument('--lookup-parquet', required=True, help='Path to MNova lookup parquet')
    p.add_argument('--output-dir', required=True, help='Base output directory')
    p.add_argument('--sample-size', type=int, default=500)
    p.add_argument('--sizes', type=str, default='5000,10000,25000,50000,100000')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    sizes = [int(x) for x in args.sizes.split(',') if x.strip()]
    out = Path(args.output_dir)
    ensure_dir(out)

    print('Sampling experiments...')
    exp_path = sample_experiments(args.experimental_pkl, out, sample_size=args.sample_size, seed=args.seed)
    print(' ->', exp_path)

    print('Building lookup subsets...')
    manifest = build_lookup_subsets(args.lookup_parquet, out, sizes=sizes, seed=args.seed)
    print(' -> manifest:', out / 'manifest.json')


if __name__ == '__main__':
    main()


