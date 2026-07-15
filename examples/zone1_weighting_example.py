"""
Example: prioritizing the downfield "zone 1" region when matching HSQC spectra.
=================================================================================

Zone 1 = the left-lower quadrant of an HSQC plot: downfield of 3 ppm (1H) and
50 ppm (13C). It holds the aromatic / olefinic / heteroaromatic and oxygenated
(O-CH, anomeric) peaks, which are more diagnostic than the crowded aliphatic
region below those thresholds. The `modified_hungarian_zone1` distance makes
matches in this region dominate the score, so a query is ranked mainly by how
well its diagnostic downfield peaks line up.

How it works
------------
Every matched peak pair's penalty is multiplied by a position weight
    w = zone_floor + (1 - zone_floor) * z**zone_gamma
where z in [0, 1] is the peak's "zone-1-ness" (1 at the downfield corner, 0 at
the aliphatic corner). The total is a *weighted mean*, so the distance stays on
the same numeric scale regardless of the weighting.

    zone_floor   weight of a top-right/aliphatic peak relative to a zone-1 peak
                 1.0 = weighting OFF (identical to plain modified_hungarian)
                 0.0 = "true domination": aliphatic peaks are (near) ignored
    zone_gamma   >1 concentrates weight even more tightly on the corner
    zone_combine 'avg'     -> z = 0.5*(h + c)   (high in EITHER dim counts)
                 'product' -> z = h * c         (strict quadrant: BOTH high)

The registered `modified_hungarian_zone1` defaults to zone_floor=0.03,
zone_gamma=2.0, zone_combine='avg', with the ramp anchored to H_range=(3, 10)
and C_range=(50, 200) -- i.e. zone 1 is everything downfield of 3 ppm (1H) and
50 ppm (13C). On a 500k-compound reference library the ~42% of peaks in this
region carry ~80% of the total scoring influence; the sub-threshold aliphatic
peaks keep the remaining ~20% as a tie-breaker. Lower zone_floor toward 0.0 for
near-total domination (~99%), or raise it (~0.1 -> ~63%) to give the aliphatic
region more say.

Run:
    PYTHONPATH=src python examples/zone1_weighting_example.py
"""
import numpy as np

from nmr_networking.similarity import (
    modified_hungarian_distance,
    resolve_distance_params,
)

# Peaks are (1H_ppm, 13C_ppm) pairs -- the orientation used throughout the pipeline.
# QUERY: two diagnostic aromatic peaks (zone 1) + two aliphatic peaks.
query = np.array([
    [7.50, 130.0],   # aromatic  (zone 1)
    [7.20, 128.0],   # aromatic  (zone 1)
    [2.10, 30.0],    # aliphatic
    [1.30, 25.0],    # aliphatic
])

# Candidate A: matches the aromatic peaks well, aliphatic peaks off by a lot.
candidate_A = np.array([
    [7.52, 130.3],   # aromatic  -> good
    [7.18, 127.7],   # aromatic  -> good
    [2.60, 38.0],    # aliphatic -> poor
    [1.80, 33.0],    # aliphatic -> poor
])

# Candidate B: aromatic peaks off, aliphatic peaks match well (the opposite).
candidate_B = np.array([
    [7.90, 138.0],   # aromatic  -> poor
    [6.80, 121.0],   # aromatic  -> poor
    [2.11, 30.1],    # aliphatic -> good
    [1.31, 25.1],    # aliphatic -> good
])

candidates = {'A (aromatic match)': candidate_A, 'B (aliphatic match)': candidate_B}


def rank(distance_function, **overrides):
    params = resolve_distance_params(distance_function, **overrides)
    scored = []
    for name, cand in candidates.items():
        dist, matched = modified_hungarian_distance(query, cand, **params)
        scored.append((name, dist, matched))
    scored.sort(key=lambda r: r[1])  # smaller distance = better match
    return scored


def show(title, scored):
    print(f"\n{title}")
    for pos, (name, dist, matched) in enumerate(scored, 1):
        print(f"   {pos}. {name:22s}  distance={dist:8.3f}  matched_frac={matched:.2f}")
    print(f"   -> best match: {scored[0][0]}")


if __name__ == '__main__':
    # 1) Plain scorer: aliphatic and aromatic peaks count equally.
    show("Plain 'modified_hungarian' (no zone weighting):",
         rank('modified_hungarian'))

    # 2) True domination (registered defaults): downfield region controls ranking.
    show("'modified_hungarian_zone1' (true domination, floor=0.0, gamma=2.0):",
         rank('modified_hungarian_zone1'))

    # 3) Softer variant: keep the aliphatic region as a tie-breaker (floor=0.1).
    show("'modified_hungarian_zone1' with --zone-floor 0.1 override:",
         rank('modified_hungarian_zone1', zone_floor=0.1))

    print("\nNote how zone weighting flips the winner toward the candidate whose")
    print("aromatic (zone-1) peaks agree, even though its aliphatic peaks are worse.")
