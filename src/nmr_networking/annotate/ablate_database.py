"""
Database ablation utilities and iterative annotation generator.

Ported for the minimal NMR-Networking package:
- Uses local similarity mapping from nmr_networking.similarity
- Uses RDKit Tanimoto as the default ranking signal (fallback if MCS packages are unavailable)
- Provides a simple CLI for running ablation for a single query
"""

from __future__ import annotations

import copy
import pickle
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS

from nmr_networking.similarity import (
    DISTANCE_FUNCTION_MAP as DISTANCE_FUNCTIONS,
    DISTANCE_FUNCTION_PARAMS,
)


def _mcs_sim_rdkit(smiles_a: str, smiles_b: str) -> float:
    """Compute a simple MCS similarity in [0,1] using RDKit's FMCS.

    Returns fraction of atoms in the smaller molecule covered by the MCS.
    """
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0
    try:
        res = rdFMCS.FindMCS([mol_a, mol_b], completeRingsOnly=False, timeout=10)
        smarts = res.smartsString
        if not smarts:
            return 0.0
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            return 0.0
        mcs_atoms = patt.GetNumAtoms()
        denom = min(mol_a.GetNumAtoms(), mol_b.GetNumAtoms()) or 1
        return float(mcs_atoms) / float(denom)
    except Exception:
        return 0.0


def _compute_full_metrics_tanimoto(args: Tuple[str, Dict[str, Any], str, Any, float]):
    """Compute Tanimoto similarity for a lookup candidate and package a record.

    Returns dict containing lookup key, smiles, hungarian distance, and tanimoto (MCS optional).
    """
    key, row, exp_smiles, exp_fp, dist = args
    lookup_smiles = row['smiles'][0]
    # Tanimoto
    tan = 0.0
    try:
        if exp_fp is not None:
            mol = Chem.MolFromSmiles(lookup_smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                tan = DataStructs.TanimotoSimilarity(exp_fp, fp)
    except Exception:
        tan = 0.0
    # Optional MCS
    sim_d = _mcs_sim_rdkit(exp_smiles, lookup_smiles)
    return {
        'lookup_key': key,
        'lookup_smiles': lookup_smiles,
        'hungarian_distance': dist,
        'mcs_score_d': sim_d,
        'tanimoto_similarity': tan,
    }


def iterative_mcs_ablation(
    exp_hsqc_shifts: np.ndarray,
    exp_smiles: str,
    exp_fp: Any,
    lookup_dict: Dict[str, Dict[str, Any]],
    threshold: float,
    rank_by: str = 'tanimoto',
    k: int = 5,
    distance_function: str = 'modified_hungarian',
    distance_params: Dict[str, Any] | None = None,
):
    """Iteratively compute distances, select top-k, compute MCS, prune by threshold, and repeat.

    Returns (surviving_lookup_dict, ablated_lookup_dict, final_annotation_df)
    """
    surviving = copy.deepcopy(lookup_dict)
    ablated: Dict[str, Dict[str, Any]] = {}

    if distance_function not in DISTANCE_FUNCTIONS:
        raise ValueError(f"Unknown distance function: {distance_function}. Available: {list(DISTANCE_FUNCTIONS.keys())}")

    dist_func = DISTANCE_FUNCTIONS[distance_function]
    if distance_params is None:
        distance_params = DISTANCE_FUNCTION_PARAMS.get(distance_function, {})

    # Precompute distances to all lookups
    if distance_params:
        dist_dict = {
            key: dist_func(exp_hsqc_shifts, row['gt'][0], **distance_params)
            for key, row in tqdm(surviving.items(), desc="Precompute Hungarian")
        }
    else:
        dist_dict = {
            key: dist_func(exp_hsqc_shifts, row['gt'][0])
            for key, row in tqdm(surviving.items(), desc="Precompute Hungarian")
        }

    iteration = 0
    final_annotation = None

    while True:
        iteration += 1

        # 1) pick top-k by Hungarian
        topk_keys = sorted(surviving.keys(), key=lambda x: dist_dict[x])[:k]

        # 2) compute similarities (tqdm over list comp is fine for small k)
        args = [(key, surviving[key], exp_smiles, exp_fp, dist_dict[key]) for key in topk_keys]
        records = [_compute_full_metrics_tanimoto(a) for a in tqdm(args, total=len(args), desc=f"Iter {iteration} sims")]

        # 3) assemble DataFrame
        final_annotation = pd.DataFrame(records)
        final_annotation['k'] = [topk_keys.index(r['lookup_key']) + 1 for r in records]

        # Hybrid score = avg(tanimoto, mcs); safe even if mcs is 0.0
        final_annotation['hybrid_score'] = 0.5 * (final_annotation['tanimoto_similarity'] + final_annotation['mcs_score_d'])

        # 4) choose column for pruning
        prune_col = 'hybrid_score'
        if rank_by == 'tanimoto':
            prune_col = 'tanimoto_similarity'
        elif rank_by == 'mcs':
            prune_col = 'mcs_score_d'

        # 5) remove entries above threshold in the chosen metric
        to_remove = final_annotation[final_annotation[prune_col] > threshold]['lookup_key'].tolist()
        if not to_remove:
            break
        for key in to_remove:
            ablated[key] = surviving.pop(key)

    # Tanimoto already computed above; ensure column exists
    if 'tanimoto_similarity' not in final_annotation.columns:
        final_annotation['tanimoto_similarity'] = 0.0
    final_annotation['hybrid_score'] = 0.5 * (final_annotation['tanimoto_similarity'] + final_annotation['mcs_score_d'])

    return surviving, ablated, final_annotation


def main():
    """CLI to run ablation for a single query.

    Example:
      python -m nmr_networking.annotate.ablate_database \
        --experimental-pkl ExperimentalHSQC_Lookup.pkl \
        --lookup-pkl lookup_dict_MNova.pkl \
        --file-key example_id \
        --threshold 35 --k 5 --distance-function modified_hungarian
    """
    import argparse

    p = argparse.ArgumentParser(description="Run iterative ablation for a single experimental query (default ranking by Tanimoto).")
    p.add_argument('--experimental-pkl', required=True, help='Pickle file with experimental lookup dict')
    p.add_argument('--lookup-pkl', required=True, help='Pickle file with candidate lookup dict')
    p.add_argument('--file-key', required=True, help='Key for experimental query to ablate')
    p.add_argument('--threshold', type=float, default=35.0)
    p.add_argument('-k', type=int, default=5)
    p.add_argument('--rank-by', choices=['hybrid', 'tanimoto', 'mcs'], default='tanimoto')
    p.add_argument('--distance-function', default='modified_hungarian')
    p.add_argument('--out-pkl', required=False, help='Path to write annotation results (pkl)')
    args = p.parse_args()

    with open(args.experimental_pkl, 'rb') as f:
        experimental_dict = pickle.load(f)
    with open(args.lookup_pkl, 'rb') as f:
        lookup_dict = pickle.load(f)

    if args.file_key not in experimental_dict:
        raise SystemExit(f"file_key {args.file_key} not found in experimental data")

    exp_vals = experimental_dict[args.file_key]
    exp_smiles = exp_vals[0]
    exp_fp = exp_vals[2]
    exp_hsqc = exp_vals[3][:, ::-1]

    survivors, removed, final_df = iterative_mcs_ablation(
        exp_hsqc_shifts=exp_hsqc,
        exp_smiles=exp_smiles,
        exp_fp=exp_fp,
        lookup_dict=lookup_dict,
        threshold=args.threshold,
        rank_by=args.rank_by,
        k=args.k,
        distance_function=args.distance_function,
        distance_params=DISTANCE_FUNCTION_PARAMS.get(args.distance_function, {}),
    )

    result = [(args.file_key, exp_smiles, final_df)]
    if args.out_pkl:
        with open(args.out_pkl, 'wb') as f:
            pickle.dump(result, f)
    else:
        print(final_df.head())


if __name__ == '__main__':
    main()


