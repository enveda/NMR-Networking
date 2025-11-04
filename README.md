![HSQC Networking Schematic](/figures/HSQC_Schematic.png)

# NMR-Networking

This is a coding package to support the implementation of HSQC Molecular Networking as described in the paper
'Structure Characterization with NMR Molecular Networking'. If you use any of the resources described in this work, please cite out paper which is currently available as a [preprint](https://chemrxiv.org/engage/chemrxiv/article-details/68c0a2679008f1a46773dc60)

## Install
### Prerequisites
- Python 3.9–3.12 
- OS: Linux/macOS recommended

### Poetry-based install:

1) Install Poetry:

   - Via official installer:
     ```bash
     curl -sSL https://install.python-poetry.org | python3 -
     ```
2) Create the virtualenv and install deps:
   ```bash
    poetry env use 3.12
    poetry install
   ```
3) Run tests to validate the environment:
   ```bash
   poetry run pytest -q
   ```


### What this toolkit does:
NMR-Networking provides an end-to-end pipeline for HSQC-based molecular networking:
- Build spectral similarity graphs (nodes = compounds, edges = HSQC similarity via Modified Hungarian distance).
- Embedding structural descriptors into edges (MCS, Tanimoto, Hybrid) 
- Add [Leiden community](https://www.nature.com/articles/s41598-019-41695-z) annotations as node features 
- Annotate compouds using top-k lookup and various distance metrics (Hungarian Distance and Modified Hungarian) 
- Query an arbitrary HSQC - get annotation candidates via top-k; rerank them with graph-aware metrics.

### Codebase Structure
- Graph construction and export: `nmr_networking.graph.builder.GraphBuilder`
  - Create graphs from DataFrame/dict/edge list distances.
  - Thresholding and custom edge filters.
  - Community detection (Leiden - node feature).
  - Structural annotations: Tanimoto, MCS, Hybrid, molecular formula, mass differences.

- HSQC similarity functions: `nmr_networking.similarity`
  - Hungarian family with multiple unmatched strategies (nn/trunc/zero) and reductions (sum/mean).
  - Modified/normalized Hungarian variants (e.g., `hung_nn`).

- Annotation reranking: `nmr_networking.annotate.annotation_analyzer.AnnotationAnalyzer`
  - Network metrics: Jaccard, Adamic-Adar, common neighbors, preferential attachment, RA, PWRA.
  - Composite scoring: 80% Hungarian-derived similarity + 20% normalized network metric by default.
  - Diagnostics: average and maximum improvements across Hybrid/Tanimoto/MCS for top-k.

## Data Availablility
Prebuilt networks, complete simulated library (MestreNova, n = 373k), sample annotations are available at https://zenodo.org/records/17081209

Experimental HSQC Spectra from the Human Metabolome Database (HMDB) cannot be shared with this study (with their associated annotation and reranking objects), but they can freely accessed at https://www.hmdb.ca/downloads  

## HSQC similarity functions - Hungarian Algorithms
```python
from nmr_networking.similarity import DISTANCE_FUNCTION_MAP, DISTANCE_FUNCTION_PARAMS
peaks_q = ...  # ndarray (N,2) (H,C) or (C,H)
peaks_r = ...
dist = DISTANCE_FUNCTION_MAP['hung_modified_2'](peaks_q, peaks_r, **DISTANCE_FUNCTION_PARAMS['hung_modified_2'])
sim  = 1.0 / (1.0 + dist)  # Hungarian-derived similarity used in composites
```

## Building NMR Molecular Networks
```python
import pandas as pd
from nmr_networking.graph.builder import GraphBuilder

df = pd.read_parquet('pairwise_hsqc_distances.parquet')  # expects File1, File2,(inchikeys) and Hungarian_Distance
b = GraphBuilder()
G = b.create_graph(
  data=df,
  threshold=35.0,
  node_col1='File1', node_col2='File2', weight_col='Hungarian_Distance',
  threshold_mode='less_than')

b.annotate_leiden_communities(resolution=0.1)  # optional
b.save_graph('network.pkl')
```

## Annotation (ablate) and Reranking examples
Two Poetry scripts for Reranking: `nmr-ablate` (annotation/top-k) and `nmr-rerank` (graph-aware reranking). These are the batch operations used to perform the studies that generate Figures 2, 3, and 6 in the main text. 

Prerequisites:
- An MNova lookup dict (pickle), e.g., `/path/to/lookup_dict_MNova.pkl` (maps `id -> payload` with `gt[0]` HSQC and optionally `smiles`). - see Zenodo
- An Experimental HSQC lookup dict (pickle), e.g., `/path/to/ExperimentalHSQC_Lookup.pkl` (maps query ids to metadata; expects HSQC at index 3, SMILES at index 0 by convention). This can be assembled using the HMDB spectra and simply contains HSQC spectra and associated SMILES. 
- A prebuilt graph (pickle) created with `GraphBuilder.save_graph` (see graph build section).

### Top-k Lookup
Top-k candidates per query based on HSQC distance (e.g., modified Hungarian):
```bash
poetry run nmr-ablate \
  --experimental-pkl /path/to/ExperimentalHSQC_Lookup.pkl \
  --lookup-pkl /path/to/lookup_dict_MNova.pkl \
  --file-key QUERY_001 \
  --k 100 \
  --distance-function hung_modified_2 \
  --out-pkl annotations_QUERY_001.pkl
```
Expected output:
- A pickle `annotations_QUERY_001.pkl` containing a list with one tuple: `(file_key, exp_smiles, df)`
- The DataFrame `df` includes columns like `lookup_key`, `hungarian_distance`, `tanimoto_similarity`, `mcs_score_d`, `hybrid_score`, and `k` (rank)
- 'k' describes how many candidates you want to calculate structural similarites for - MCS calculations can be computationally slow

### Rerank using the spectral graph (network metrics or composite)
Use a molecular network to perform algorithmic molecular networking and rerank the annotations based on a chosen metric (e.g., PWRA via `ra_weighted_product`) or composite with Hungarian similarity.
```bash
poetry run nmr-rerank \
  --annotations annotations_QUERY_001.pkl \
  --experimental /path/to/ExperimentalHSQC_Lookup.pkl \
  --lookup /path/to/lookup_dict_MNova.pkl \
  --graph /path/to/network.pkl \
  --metric ra_weighted_product \
  --secondary-metric hungarian_nn \
  --weights 0.2:0.8 \
  --out-dir rerank_results_QUERY_001
```
Notes:
- `--metric ra_weighted_product` computes PWRA coherence on the existing graph.
- `--secondary-metric hungarian_nn` uses inverse-distance similarity derived from the query–candidate edge.
- `--weights 0.2:0.8` sets composite weights as 20% network, 80% Hungarian-derived similarity (you can swap to 0.8:0.2 if desired).

Expected output in `rerank_results_QUERY_001/`:
- A reranked annotation pickle mirroring the input structure but with metric/composite columns added (e.g., `{metric}_score`, `{metric}_{secondary_metric}_composite_score`).

- If `query_connections` is low, consider relaxing thresholds or ensuring that the graph includes sufficient neighbors for the query region.

## Data Structures
- Creating Molecular Networks (NetworkX):
  - Nodes: IDs (InChIKeys). Attributes: `smiles`, community labels, structural formulas, etc.
  - Edges: HSQC distances in `weight` (alias `hungarian_distance`). Optional `tanimoto_similarity`, `mcs_similarity`, `hybrid_similarity`, `mass_difference`.

- MNova lookup dict (pickle):
  - Dict `id -> payload` where `payload` includes HSQC ground truth (`gt[0]`) and may include `smiles` or a DataFrame containing a `smiles` column.

- Annotation tuples (pickle):
  - List of `(file_key: str, exp_smiles: str, df: pandas.DataFrame)`; `df` columns include: `lookup_key`, `hungarian_distance`, `tanimoto_similarity`, `mcs_score_d`, `hybrid_score`, `k`. Optional reranked attributes


## Troubleshooting
- `ModuleNotFoundError: nmr_networking`: run from project root or add `src` to `PYTHONPATH`.
- Missing RDKit: install RDKit to enable Tanimoto/MCS/Hybrid and PNG rendering.
- PNG shows placeholders: SMILES could not be resolved from the MNova lookup entry.
- No rerank due to connectivity: relax `--min-connections` or increase `--threshold`/`--n-rerank`.

## Debugging Installation

Use any (or all) of the following smoke tests.

1) Import tests (Python):
```bash
poetry run python - <<'PY'
from nmr_networking.graph.builder import GraphBuilder
from nmr_networking.similarity import DISTANCE_FUNCTION_MAP, DISTANCE_FUNCTION_PARAMS
print("GraphBuilder OK")
print("Similarity functions:", sorted(list(DISTANCE_FUNCTION_MAP.keys()))[:5], "...")
PY
```
Expected output contains:
- `GraphBuilder OK`
- A non-empty list of similarity functions

2) Build a tiny graph from synthetic data:
```python
import pandas as pd
from nmr_networking.graph.builder import GraphBuilder

df = pd.DataFrame({
  'File1': ['A','A','B'],
  'File2': ['B','C','C'],
  'Hungarian_Distance': [10.0, 20.0, 12.0]
})

b = GraphBuilder()
G = b.create_graph(df, threshold=25.0, node_col1='File1', node_col2='File2', weight_col='Hungarian_Distance', threshold_mode='less_than')
print("Nodes, Edges:", G.number_of_nodes(), G.number_of_edges())
```
Expected output: `Nodes, Edges: 3 3`

3) Similarity function call:
```python
from nmr_networking.similarity import DISTANCE_FUNCTION_MAP, DISTANCE_FUNCTION_PARAMS

peaks_a = np.array([[1.0, 10.0],[2.0, 20.0],[3.0, 30.0]])  # (H,C)
peaks_b = np.array([[1.1, 10.2],[2.1, 20.1],[3.2, 29.9]])
fn = DISTANCE_FUNCTION_MAP['hung_modified_2']
params = DISTANCE_FUNCTION_PARAMS['hung_modified_2']
d = fn(peaks_a, peaks_b, **params)
print("hung_modified_2 distance:", float(d if not isinstance(d, tuple) else d[0]))
PY
```
Expected: a distance



