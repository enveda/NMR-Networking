#!/usr/bin/env python3

"""
Rerank annotations by network and annotation metrics with optional composite scoring.

Streamlined for the minimal package:
- Relies on AnnotationAnalyzer for graph integration and scoring
- Provides CLI to process annotation result PKLs with experimental and lookup data
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from rich.console import Console

from nmr_networking.annotate.annotation_analyzer import AnnotationAnalyzer, load_annotation_results


def rerank_cli(
    annotation_path: str,
    experimental_path: str,
    lookup_hsqc_path: str,
    graph_path: str,
    metric: str = 'jaccard',
    secondary_metric: Optional[str] = None,
    primary_weight: float = 0.5,
    secondary_weight: float = 0.5,
    similarity_threshold: float = 50.0,
    max_neighbors: int = 20,
    min_connections: int = 3,
    distance_column: str = 'hung_modified_2',
    out_dir: Optional[str] = None,
):
    console = Console()

    annotations = load_annotation_results(annotation_path)
    with open(experimental_path, 'rb') as f:
        experimental_dict = pickle.load(f)
    with open(lookup_hsqc_path, 'rb') as f:
        lookup_dict = pickle.load(f)

    lookup_hsqc_data = {
        key: value['gt'][0] for key, value in lookup_dict.items() if 'gt' in value and len(value['gt']) > 0
    }

    analyzer = AnnotationAnalyzer(
        graph_path=graph_path,
        distance_column=distance_column,
        verbose=True,
    )

    reranked = analyzer.rerank_annotations_by_metric(
        annotations=annotations,
        experimental_data=experimental_dict,
        lookup_hsqc_data=lookup_hsqc_data,
        metric=metric,
        secondary_metric=secondary_metric,
        primary_weight=primary_weight,
        secondary_weight=secondary_weight,
        similarity_threshold=similarity_threshold,
        max_neighbors=max_neighbors,
        min_connections=min_connections,
    )

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / 'reranked_annotations.pkl', 'wb') as f:
            pickle.dump(reranked, f)

    return reranked


def main():
    import argparse

    p = argparse.ArgumentParser(description="Rerank annotations by network metrics and composite scores.")
    p.add_argument('--annotations', required=True)
    p.add_argument('--experimental', required=True)
    p.add_argument('--lookup', required=True)
    p.add_argument('--graph', required=True)
    p.add_argument('--metric', default='jaccard')
    p.add_argument('--secondary-metric', default=None)
    p.add_argument('--primary-weight', type=float, default=0.5, help='Weight for primary metric (0..1)')
    p.add_argument('--secondary-weight', type=float, default=0.5, help='Weight for secondary metric (0..1)')
    p.add_argument('--weights', type=str, default=None, help='Composite weights as "PRIMARY:SECONDARY" (e.g., 0.8:0.2). Overrides individual weights if provided.')
    p.add_argument('--similarity-threshold', type=float, default=50.0)
    p.add_argument('--max-neighbors', type=int, default=20)
    p.add_argument('--min-connections', type=int, default=3)
    p.add_argument('--distance-column', default='hung_modified_2')
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()

    # Allow --weights to override individual weights
    if args.weights:
        try:
            sep = ':' if ':' in args.weights else ','
            w1_str, w2_str = [x.strip() for x in args.weights.split(sep, 1)]
            w1 = float(w1_str)
            w2 = float(w2_str)
            total = (w1 + w2) or 1.0
            args.primary_weight = w1 / total
            args.secondary_weight = w2 / total
        except Exception:
            pass

    rerank_cli(
        annotation_path=args.annotations,
        experimental_path=args.experimental,
        lookup_hsqc_path=args.lookup,
        graph_path=args.graph,
        metric=args.metric,
        secondary_metric=args.secondary_metric,
        primary_weight=args.primary_weight,
        secondary_weight=args.secondary_weight,
        similarity_threshold=args.similarity_threshold,
        max_neighbors=args.max_neighbors,
        min_connections=args.min_connections,
        distance_column=args.distance_column,
        out_dir=args.out_dir,
    )


if __name__ == '__main__':
    main()


