"""
Top-k annotation generator (no iterative ablation).

Given an experimental HSQC query and a lookup dictionary, compute distances to all
lookup entries, select top-k, and compute structural similarity (Tanimoto, optional MCS)
for those top-k. Returns an annotation object compatible with AnnotationAnalyzer reranking.
"""

from __future__ import annotations

import argparse
import pickle
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS

from nmr_networking.similarity import (
    DISTANCE_FUNCTION_MAP as DISTANCE_FUNCTIONS,
    DISTANCE_FUNCTION_PARAMS,
)


def _mcs_sim_rdkit(smiles_a: str, smiles_b: str) -> float:
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0
    try:
        res = rdFMCS.FindMCS([mol_a, mol_b], completeRingsOnly=False, timeout=5)
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


def generate_topk_annotations(
    exp_hsqc_shifts: np.ndarray,
    exp_smiles: str,
    exp_fp: Any,
    lookup_dict: Dict[str, Dict[str, Any]],
    k: int = 100,
    distance_function: str = 'modified_hungarian',
    distance_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    if distance_function not in DISTANCE_FUNCTIONS:
        raise ValueError(f"Unknown distance function: {distance_function}")
    dist_func = DISTANCE_FUNCTIONS[distance_function]
    if distance_params is None:
        distance_params = DISTANCE_FUNCTION_PARAMS.get(distance_function, {})

    # Compute distances to all lookup candidates
    distances: List[Tuple[str, float]] = []
    for key, row in lookup_dict.items():
        try:
            gt = row['gt'][0]
            d = dist_func(exp_hsqc_shifts, gt, **distance_params) if distance_params else dist_func(exp_hsqc_shifts, gt)
            if isinstance(d, tuple):
                d = d[0]
            distances.append((key, float(d)))
        except Exception:
            continue

    # Select top-k (smallest distances)
    distances.sort(key=lambda x: x[1])
    top_keys = [k_ for k_, _ in distances[:k]]

    # Compute structural similarities for top-k
    records: List[Dict[str, Any]] = []
    for rank, key in enumerate(top_keys, start=1):
        row = lookup_dict[key]
        lookup_smiles = row['smiles'][0]
        tan = 0.0
        try:
            if exp_fp is not None:
                mol = Chem.MolFromSmiles(lookup_smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                    tan = DataStructs.TanimotoSimilarity(exp_fp, fp)
        except Exception:
            tan = 0.0
        mcs_d = _mcs_sim_rdkit(exp_smiles, lookup_smiles)
        records.append({
            'lookup_key': key,
            'lookup_smiles': lookup_smiles,
            'hungarian_distance': dict(distances)[key],
            'tanimoto_similarity': tan,
            'mcs_score_d': mcs_d,
            'k': rank,
            'hybrid_score': 0.5 * (tan + mcs_d),
        })

    return pd.DataFrame(records)


def main():
    p = argparse.ArgumentParser(description="Generate top-k annotations for a single query (no iteration).")
    p.add_argument('--experimental-pkl', required=True)
    p.add_argument('--lookup-pkl', required=True)
    p.add_argument('--file-key', required=True)
    p.add_argument('-k', type=int, default=100)
    p.add_argument('--distance-function', default='modified_hungarian')
    p.add_argument('--out-pkl', required=True)
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

    df = generate_topk_annotations(
        exp_hsqc_shifts=exp_hsqc,
        exp_smiles=exp_smiles,
        exp_fp=exp_fp,
        lookup_dict=lookup_dict,
        k=args.k,
        distance_function=args.distance_function,
    )

    with open(args.out_pkl, 'wb') as f:
        pickle.dump([(args.file_key, exp_smiles, df)], f)


if __name__ == '__main__':
    main()


