import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import random
from scipy.spatial.distance import euclidean
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def calculate_hungarian_distance(pred_shifts, gt_shifts, strategy='nn', reduction='sum'):
    """
    Compute the assignment distances between two HSQC spectra,
    using one of three strategies to handle unequal peak counts:

    Strategies:
      - 'trunc':  only the optimal 1:1 matches for min(n1,n2) peaks (ignore extras)
      - 'zero':   pad the shorter list with (0,0) peaks before matching
      - 'nn':     after Hungarian matching, assign any unmatched peaks to their nearest neighbor

    Parameters
    ----------W
    pred_shifts : array-like of shape (n1,2)
        Predicted (¹H,¹³C) shift pairs.
    gt_shifts : array-like of shape (n2,2)
        Ground-truth (¹H,¹³C) shift pairs.
    strategy : {'trunc', 'zero', 'nn'}
        Which method to use for unmatched peaks.
    reduction : {'sum', 'mean'}
        Whether to return sum or mean of distances.

    Returns
    -------
    total_distance : float
        Sum or mean of absolute distances for all assigned pairs (including unmatched via padding or NN).
    """
    # Convert to numpy arrays
    pred = np.asarray(pred_shifts, dtype=float)
    gt   = np.asarray(gt_shifts, dtype=float)
    n1, n2 = pred.shape[0], gt.shape[0]

    # Quick return if both empty
    if n1 == 0 and n2 == 0:
        return 0.0

    # Compute full pairwise distance matrix
    C = cdist(pred, gt, metric='euclidean')

    if strategy == 'trunc':
        # Only take the optimal min(n1,n2) matches
        row, col = linear_sum_assignment(C)
        matched = np.abs(C[row, col])
        total = float(matched.sum())
        return total if reduction == 'sum' else total / len(matched)

    elif strategy == 'zero':
        # Pad the shorter list with (0,0) peaks
        if n1 < n2:
            pad = np.zeros((n2 - n1, 2), dtype=float)
            pred_z = np.vstack((pred, pad))
            gt_z   = gt
        elif n2 < n1:
            pad = np.zeros((n1 - n2, 2), dtype=float)
            gt_z   = np.vstack((gt, pad))
            pred_z = pred
        else:
            pred_z, gt_z = pred, gt

        C_z = cdist(pred_z, gt_z, metric='euclidean')
        row, col = linear_sum_assignment(C_z)
        total = float(np.abs(C_z[row, col]).sum())
        return total if reduction == 'sum' else total / max(n1, n2)

    elif strategy == 'nn':
        # Hungarian for core matches
        row, col = linear_sum_assignment(C)
        total = np.abs(C[row, col]).sum()

        # Identify unmatched indices
        matched_i = set(row)
        matched_j = set(col)
        unm_i = [i for i in range(n1) if i not in matched_i]
        unm_j = [j for j in range(n2) if j not in matched_j]

        # Optimized: Assign unmatched pred peaks to nearest gt
        if unm_i:
            # Only compute argmin for unmatched rows, not full matrix scan
            unm_i_array = np.array(unm_i)
            nnj = C[unm_i_array].argmin(axis=1)
            total += np.abs(C[unm_i_array, nnj]).sum()

        # Optimized: Assign unmatched gt peaks to nearest pred  
        if unm_j:
            # Only compute argmin for unmatched columns, not full matrix scan
            unm_j_array = np.array(unm_j)
            nni = C[:, unm_j_array].argmin(axis=0)
            total += np.abs(C[nni, unm_j_array]).sum()

        return float(total) if reduction == 'sum' else float(total) / max(n1, n2)

    else:
        raise ValueError(f"Unknown strategy '{strategy}' (choose 'trunc', 'zero', or 'nn')")


def modified_hungarian_distance(peaks1, peaks2,
                                sigma_H=0.01, sigma_C=0.2,
                                func_H=0.5, func_C=2.5,
                                penalty_factor=1.0,
                                matching='zero'):
    """
    Compute an HSQC-spectrum distance & matched fraction under
    three possible assignment schemes:
      - 'zero':   pad & Hungarian (original)
      - 'nn':      Hungarian + NN double‐assign unmatched
      - 'trunc':   rectangular Hungarian only (drop extras)

    matching : str, one of {'zero','nn','trunc'}
    """

    p1 = np.asarray(peaks1, float)
    p2 = np.asarray(peaks2, float)
    n1, n2 = len(p1), len(p2)

    # 1) compute normalized threshold T
    fHn = func_H / sigma_H
    fCn = func_C / sigma_C
    T   = np.sqrt(fHn*fHn + fCn*fCn)

    # empty‐spectrum corner
    if n1 == 0 or n2 == 0:
        # no real matches possible
        return T, 0.0

    # 2) normalize coords & pairwise distances
    coords1 = p1 / np.array([sigma_H, sigma_C])
    coords2 = p2 / np.array([sigma_H, sigma_C])
    D = cdist(coords1, coords2, metric="euclidean")

    # 3) base cost matrix
    C = np.where(D <= T, D, D + penalty_factor)

    # 4) get initial Hungarian assignment
    #    SciPy handles rectangular C directly
    row_idx, col_idx = linear_sum_assignment(C)

    # build up list of all (i,j) pairs we'll score:
    assignments = list(zip(row_idx, col_idx))

    if matching == 'zero':
        # --- original dummy‐slot padding approach ---
        # rebuild square matrix with dummy padding at cost T, then re‐solve:
        N = max(n1, n2)
        if n1 > n2:
            pad = np.full((n1, n1 - n2), T)
            C_pad = np.hstack((C, pad))
            # Pad distance matrix D as well
            D_pad = np.hstack((D, np.full((n1, n1 - n2), T)))
        elif n2 > n1:
            pad = np.full((n2 - n1, n2), T)
            C_pad = np.vstack((C, pad))
            # Pad distance matrix D as well
            D_pad = np.vstack((D, np.full((n2 - n1, n2), T)))
        else:
            C_pad = C
            D_pad = D
        row2, col2 = linear_sum_assignment(C_pad)
        assignments = list(zip(row2, col2))
        denom = N
        # Use the padded distance matrix for scoring
        D = D_pad

    elif matching == 'nn':
        # --- Hungarian + nearest‐neighbor for leftovers ---
        N = max(n1, n2)
        # find which peaks in the larger set were left out
        if n1 > n2:
            unmatched = set(range(n1)) - set(row_idx)
            for i in unmatched:
                j_nn = np.argmin(D[i, :])      # allow reuse
                assignments.append((i, j_nn))
        elif n2 > n1:
            unmatched = set(range(n2)) - set(col_idx)
            for j in unmatched:
                i_nn = np.argmin(D[:, j])
                assignments.append((i_nn, j))
        denom = N

    elif matching == 'trunc':
        # --- rectangular Hungarian only, drop extra peaks ---
        # assignments already = zip(row_idx, col_idx)
        denom = min(n1, n2)

    else:
        raise ValueError(f"Unknown matching strategy: {matching!r}")

    # 5) compute total penalty & matched count
    total_penalty = 0.0
    good_matches  = 0
    for i, j in assignments:
        dij = D[i, j]
        pen = dij if dij <= T else dij + penalty_factor
        total_penalty += pen
        if dij <= T:
            good_matches += 1

    distance = total_penalty / denom
    matched_fraction = good_matches / denom

    return distance, matched_fraction

def scaled_modified_hungarian_distance(peaks1, peaks2,
                                sigma_H=0.01, sigma_C=0.2,
                                func_H=0.5, func_C=2.5,
                                penalty_factor=1.0,
                                matching='nn',
                                H_range=(1.0, 10.0),
                                C_range=(1.0, 200.0),
                                penalty_scale=0.1):
    """
    Compute HSQC distance & matched fraction, scaling each assignment's cost
    by the average ppm position normalized to its 1H (1–10 ppm) and 13C
    (1–200 ppm) ranges, then reducing the overall weighted penalty by penalty_scale.
    """
    p1 = np.asarray(peaks1, float)
    p2 = np.asarray(peaks2, float)
    n1, n2 = len(p1), len(p2)

    # threshold in normalized units
    fHn = func_H / sigma_H
    fCn = func_C / sigma_C
    T = np.sqrt(fHn**2 + fCn**2)

    # empty-spectrum corner
    if n1 == 0 or n2 == 0:
        return T, 0.0

    # normalize & pairwise distances
    coords1 = p1 / np.array([sigma_H, sigma_C])
    coords2 = p2 / np.array([sigma_H, sigma_C])
    D = cdist(coords1, coords2, metric="euclidean")

    # cost matrix with penalty
    C = np.where(D <= T, D, D + penalty_factor)

    # Hungarian assignment (trunc)
    row_idx, col_idx = linear_sum_assignment(C)
    assignments = list(zip(row_idx, col_idx))
    denom = min(n1, n2)

    # compute scaled distance & matched fraction
    total_penalty = 0.0
    good_matches = 0
    H_min, H_max = H_range
    C_min, C_max = C_range

    for i, j in assignments:
        dij = D[i, j]
        base_pen = dij if dij <= T else dij + penalty_factor

        # average ppm
        H_avg = (p1[i, 0] + p2[j, 0]) / 2.0
        C_avg = (p1[i, 1] + p2[j, 1]) / 2.0

        # normalize within ranges
        H_norm = (H_avg - H_min) / (H_max - H_min)
        C_norm = (C_avg - C_min) / (C_max - C_min)
        weight = max(0.0, H_norm) + max(0.0, C_norm)

        # apply penalty scale
        total_penalty += base_pen * weight * penalty_scale
        if dij <= T:
            good_matches += 1

    distance = total_penalty / denom
    matched_fraction = good_matches / denom

    return distance, matched_fraction