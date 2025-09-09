#!/usr/bin/env python3

"""
Prepare minimal sample data for NMR-Networking examples.

Creates:
- experimental.pkl: {'Q1': [smiles, None, exp_fp, exp_hsqc_np]}
- lookup.pkl: {'L1'|'L2'|'L3': {'smiles':[smi], 'gt':[hsqc_np]}}
- distances.csv: small edge list between lookup entries with Hungarian distances
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from nmr_networking.similarity import calculate_hungarian_distance


def make_hsqc(peaks: list[tuple[float, float]]):
    return np.array(peaks, dtype=float)


def main():
    out = Path(__file__).parent

    # Define simple molecules and HSQC peak sets (H, C)
    query_smiles = 'CCO'            # ethanol
    l1_smiles = 'CCO'               # ethanol (same)
    l2_smiles = 'CC(C)O'            # isopropanol
    l3_smiles = 'CC(=O)C'           # acetone

    query_hsqc = make_hsqc([(3.7, 58.0), (1.2, 25.0)])
    l1_hsqc = make_hsqc([(3.7, 58.0), (1.2, 25.0)])
    l2_hsqc = make_hsqc([(3.6, 67.0), (1.0, 23.0), (1.1, 24.0)])
    l3_hsqc = make_hsqc([(2.1, 30.0), (2.2, 31.0), (2.3, 20.0)])

    # Build experimental dict
    qmol = Chem.MolFromSmiles(query_smiles)
    exp_fp = AllChem.GetMorganFingerprintAsBitVect(qmol, 2, 2048)
    experimental = {
        'Q1': [query_smiles, None, exp_fp, query_hsqc[:, ::-1]],  # stored as (C,H) in some datasets
    }
    with open(out / 'experimental.pkl', 'wb') as f:
        pickle.dump(experimental, f)

    # Build lookup dict
    lookup = {
        'L1': {'smiles': [l1_smiles], 'gt': [l1_hsqc]},
        'L2': {'smiles': [l2_smiles], 'gt': [l2_hsqc]},
        'L3': {'smiles': [l3_smiles], 'gt': [l3_hsqc]},
    }
    with open(out / 'lookup.pkl', 'wb') as f:
        pickle.dump(lookup, f)

    # Create small edge list between lookups
    pairs = [('L1', 'L2'), ('L1', 'L3'), ('L2', 'L3')]
    rows = []
    for a, b in pairs:
        da = lookup[a]['gt'][0]
        db = lookup[b]['gt'][0]
        d = calculate_hungarian_distance(da, db, strategy='nn', reduction='sum')
        rows.append({'File1': a, 'File2': b, 'Hungarian_Distance': float(d)})
    pd.DataFrame(rows).to_csv(out / 'distances.csv', index=False)

    print('Wrote:')
    print(' -', out / 'experimental.pkl')
    print(' -', out / 'lookup.pkl')
    print(' -', out / 'distances.csv')


if __name__ == '__main__':
    main()


