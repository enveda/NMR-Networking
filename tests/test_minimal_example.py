import os
import pickle
from pathlib import Path

import pandas as pd

from nmr_networking import GraphBuilder
from nmr_networking.annotate.ablate_dataset import generate_topk_annotations
from nmr_networking.annotate.annotation_reranking import rerank_cli


def test_minimal_end_to_end(tmp_path: Path):
    # Prepare minimal data by invoking the example script
    project_root = Path(__file__).resolve().parents[1]
    examples = project_root / 'examples'

    # Ensure we run in project root so example writes files to expected paths
    cwd = os.getcwd()
    os.chdir(project_root)
    try:
        import runpy
        runpy.run_path(str(examples / 'prepare_data.py'))
    finally:
        os.chdir(cwd)

    experimental_pkl = examples / 'experimental.pkl'
    lookup_pkl = examples / 'lookup.pkl'
    distances_csv = examples / 'distances.csv'

    assert experimental_pkl.exists() and lookup_pkl.exists() and distances_csv.exists()

    # a) build graph
    df = pd.read_csv(distances_csv)
    builder = GraphBuilder()
    G = builder.create_graph(df, threshold=100.0, node_col1='File1', node_col2='File2', weight_col='Hungarian_Distance')
    assert G.number_of_nodes() >= 2

    # b) generate top-k annotations
    with open(experimental_pkl, 'rb') as f:
        experimental = pickle.load(f)
    with open(lookup_pkl, 'rb') as f:
        lookup = pickle.load(f)
    key = 'Q1'
    exp_vals = experimental[key]
    exp_smiles = exp_vals[0]
    exp_fp = exp_vals[2]
    exp_hsqc = exp_vals[3][:, ::-1]

    df_ann = generate_topk_annotations(
        exp_hsqc_shifts=exp_hsqc,
        exp_smiles=exp_smiles,
        exp_fp=exp_fp,
        lookup_dict=lookup,
        k=3,
    )
    assert {'lookup_key','lookup_smiles','hungarian_distance','tanimoto_similarity','mcs_score_d','k','hybrid_score'} <= set(df_ann.columns)
    assert len(df_ann) == 3

    # c) ensure graph export works
    builder.save_graph(tmp_path / 'graph.pkl')
    assert (tmp_path / 'graph.pkl').exists()

    # d) rerank with an existing annotations file if available
    ann_path = Path('/home/cailum.stienstra/HSQC_Models/Networking_HSQC/NMR-Networking/examples/low_quality_annotations_hybrid_score_le_0_8.pkl')
    if ann_path.exists():
        reranked = rerank_cli(
            annotation_path=str(ann_path),
            experimental_path=str(experimental_pkl),
            lookup_hsqc_path=str(lookup_pkl),
            graph_path=str(tmp_path / 'graph.pkl'),
            metric='jaccard',
            secondary_metric='hungarian_nn',
            primary_weight=0.8,
            secondary_weight=0.2,
            similarity_threshold=100.0,
            max_neighbors=10,
            min_connections=0,
            distance_column='hungarian_nn_sum',
            out_dir=None,
        )
        assert isinstance(reranked, list)
        assert len(reranked) >= 0


