# Consolidated extras used by GraphBuilder: leiden community detection and MolNet conversion

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd

# Optional RDKit for mass/formula
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False


# ---- Minimal Leiden wrapper (fallback to networkx if leiden unavailable) ----
try:
    import igraph as ig
    import leidenalg
    _LEIDEN_AVAILABLE = True
except Exception:
    _LEIDEN_AVAILABLE = False


class NodeClustering:
    def __init__(self, communities: List[List[Any]]):
        self.communities = communities


def leiden(g_original: nx.Graph, resolution_parameter: float = 1.0) -> NodeClustering:
    """Return NodeClustering with communities using leiden if available; otherwise connected components.
    """
    if _LEIDEN_AVAILABLE:
        g = ig.Graph()
        g.add_vertices(list(map(str, g_original.nodes())))
        name_to_idx = {str(v['name']) if isinstance(v, dict) and 'name' in v else str(v): i for i, v in enumerate(g.vs)}
        edges = [(name_to_idx[str(u)], name_to_idx[str(v)]) for u, v in g_original.edges()]
        g.add_edges(edges)
        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution_parameter)
        comms: List[List[Any]] = []
        for comm in part:
            comms.append([g.vs[idx]['name'] for idx in comm])
        return NodeClustering(comms)
    # Fallback: connected components
    comms = [list(c) for c in nx.connected_components(g_original)]
    return NodeClustering(comms)


# ---- Minimal MolNet converter used by builder ----
class MolNetConverter:
    def __init__(self, mnova_lookup_path: Optional[Union[str, Path]] = None, experimental_lookup_path: Optional[Union[str, Path]] = None):
        self.mnova_data: Dict[str, Dict[str, Any]] = {}
        self.experimental_data: Dict[str, Dict[str, Any]] = {}
        if mnova_lookup_path:
            self.load_mnova_lookup(mnova_lookup_path)
        if experimental_lookup_path:
            self.load_experimental_lookup(experimental_lookup_path)

    def load_mnova_lookup(self, filepath: Union[str, Path]):
        fp = Path(filepath)
        if fp.suffix == '.parquet':
            df = pd.read_parquet(fp)
            for _, row in df.iterrows():
                self.mnova_data[str(row['id'])] = {'smiles': row.get('smiles'), 'hsqc': row.get('hsqc')}
        else:
            import pickle
            with open(fp, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, dict):
                        self.mnova_data[str(k)] = {'smiles': v.get('smiles'), 'hsqc': v.get('gt')}

    def load_experimental_lookup(self, filepath: Union[str, Path]):
        import pickle
        fp = Path(filepath)
        with open(fp, 'rb') as f:
            data = pickle.load(f)
        for k, v in data.items():
            if isinstance(v, (list, tuple)) and len(v) > 3:
                self.experimental_data[str(k)] = {'smiles': v[0], 'hsqc': v[3]}

    def _calc_mass(self, smiles: Optional[str]) -> Optional[float]:
        if not (RDKIT_AVAILABLE and smiles):
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Descriptors.MolWt(mol) if mol is not None else None
        except Exception:
            return None

    def _parse_hsqc(self, hsqc: Any) -> List[List[float]]:
        try:
            arr = np.asarray(hsqc, dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 2:
                return arr.tolist()
            return []
        except Exception:
            return []

    def convert_graph_to_molnet(self, graph: nx.Graph, output_dir: Union[str, Path] = 'molnet_output', edge_weight_column: str = 'weight', additional_edge_attributes: Optional[List[str]] = None, node_attributes: Optional[List[str]] = None, preview_count: int = 0, show_progress: bool = True) -> Dict[str, Any]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        nodes = []
        edges = []
        for nid, nattrs in graph.nodes(data=True):
            nid_s = str(nid)
            smiles = nattrs.get('smiles')
            if not smiles:
                smiles = (self.mnova_data.get(nid_s) or {}).get('smiles') or (self.experimental_data.get(nid_s) or {}).get('smiles')
            hsqc = (self.mnova_data.get(nid_s) or {}).get('hsqc') or (self.experimental_data.get(nid_s) or {}).get('hsqc')
            node_entry = {
                'id': nid_s,
                'hsqc_coordinates': self._parse_hsqc(hsqc) if hsqc is not None else [],
                'metadata': {
                    'smiles': smiles,
                    'molecular_mass': self._calc_mass(smiles),
                }
            }
            if node_attributes:
                for attr in node_attributes:
                    if attr in nattrs:
                        node_entry['metadata'][attr] = nattrs[attr]
            nodes.append(node_entry)
        for s, t, eattrs in graph.edges(data=True):
            entry = {'source_id': str(s), 'target_id': str(t), edge_weight_column: float(eattrs.get(edge_weight_column, 1.0))}
            if additional_edge_attributes is None:
                for k, v in eattrs.items():
                    if k != edge_weight_column:
                        try:
                            entry[k] = float(v) if isinstance(v, (int, float, np.floating)) else v
                        except Exception:
                            entry[k] = str(v)
            else:
                for attr in additional_edge_attributes:
                    if attr in eattrs:
                        entry[attr] = eattrs[attr]
            edges.append(entry)
        nodes_file = out / 'molnet_nodes.json'
        edges_file = out / 'molnet_edges.json'
        with open(nodes_file, 'w') as f:
            json.dump(nodes, f, indent=2)
        with open(edges_file, 'w') as f:
            json.dump(edges, f, indent=2)
        return {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'edge_weight_column': edge_weight_column,
            'files_created': [str(nodes_file), str(edges_file)],
            'output_directory': str(out),
        }
