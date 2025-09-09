import networkx as nx
import pandas as pd
import numpy as np
import pickle
import json
from typing import Union, Dict, List, Tuple, Optional, Any
from pathlib import Path
from tqdm import tqdm
import sys
import os
import multiprocessing as mp
from functools import partial

# RDKit imports for molecular similarity calculations
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors, rdFMCS
    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: Could not import RDKit. Structural similarity calculations will not be available.")
    RDKIT_AVAILABLE = False

# Molecular similarity imports
try:
    from molecular_graph_edit.molecular_similarity import mcs_sim_largest_connected as mcs_sim_d
    MCS_AVAILABLE = True
except ImportError:
    print("Warning: Could not import molecular_graph_edit. MCS similarity calculations will not be available.")
    MCS_AVAILABLE = False
    def mcs_sim_d(*args, **kwargs):
        return 0.0

try:
    from .extras import leiden
    LEIDEN_AVAILABLE = True
except Exception:
    LEIDEN_AVAILABLE = False
    def leiden(*args, **kwargs):
        # Quietly raise without printing extra guidance
        raise ImportError("Leiden algorithm not available.")

try:
    from .extras import MolNetConverter
    MOLNET_AVAILABLE = True
except Exception:
    MOLNET_AVAILABLE = False

# Import distance functions (project-local mapping)
try:
    from ..similarity import DISTANCE_FUNCTION_MAP, DISTANCE_FUNCTION_PARAMS
except Exception:
    print("Warning: Could not import distance functions. add_query_to_graph_and_export will not work.")
    DISTANCE_FUNCTION_MAP = {}
    DISTANCE_FUNCTION_PARAMS = {}


def smiles_mass_difference(smiles1, smiles2):
    """Return the monoisotopic mass difference between two SMILES."""
    if not RDKIT_AVAILABLE:
        return 0.0
        
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    mass1 = rdMolDescriptors.CalcExactMolWt(mol1)
    mass2 = rdMolDescriptors.CalcExactMolWt(mol2)
    return abs(mass1 - mass2)


def _compute_edge_similarities(edge, smiles_lookup):
    """
    Compute structural similarities for a single edge.
    
    Parameters:
    -----------
    edge : tuple
        Edge tuple (node1, node2, weight)
    smiles_lookup : dict
        Dictionary mapping node IDs to SMILES strings
        
    Returns:
    --------
    tuple : (node1, node2, weight, tanimoto_sim, mcs_sim, hybrid_sim, mass_diff)
    """
    node1, node2, weight = edge
    
    # Get SMILES strings for both nodes
    smiles1 = smiles_lookup.get(node1)
    smiles2 = smiles_lookup.get(node2)
    
    # Initialize similarity scores
    tanimoto_sim = 0.0
    mcs_sim = 0.0
    hybrid_sim = 0.0
    mass_diff = 0.0
    
    # Compute Tanimoto similarity if RDKit is available and SMILES are valid
    if RDKIT_AVAILABLE and smiles1 and smiles2:
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            
            if mol1 is not None and mol2 is not None:
                fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 2048)
                tanimoto_sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        except Exception as e:
            # Silently handle errors in similarity calculation
            pass
    
    # Compute MCS similarity if available and SMILES are valid
    if MCS_AVAILABLE and smiles1 and smiles2:
        try:
            mcs_sim = mcs_sim_d(
                smiles1, smiles2, 
                num_tries=10, 
                num_keep_m=10, 
                num_keep_max_m=2
            )
        except Exception as e:
            # Silently handle errors in similarity calculation
            pass
    
    # Compute mass difference if SMILES are valid
    if smiles1 and smiles2:
        try:
            mass_diff = smiles_mass_difference(smiles1, smiles2)
        except Exception as e:
            # Silently handle errors in mass difference calculation
            pass
    
    # Compute hybrid similarity as average of tanimoto and MCS
    hybrid_sim = 0.5 * (tanimoto_sim + mcs_sim)
    
    return (node1, node2, weight, tanimoto_sim, mcs_sim, hybrid_sim, mass_diff)


class GraphBuilder:
    """
    A comprehensive framework for creating and managing NetworkX graphs from various data sources.
    
    Supports different input formats including DataFrames, dictionaries, and edge lists.
    Provides thresholding, filtering, and graph persistence capabilities.
    """
    
    def __init__(self, directed: bool = False, graph_type: str = 'networkx'):
        """
        Initialize the GraphBuilder.
        
        Parameters:
        -----------
        directed : bool, default=False
            Whether to create directed graphs
        graph_type : str, default='networkx'
            Type of graph to create ('networkx', 'weighted', 'simple')
        """
        self.directed = directed
        self.graph_type = graph_type
        self.graph = None
        self.metadata = {}
        
    def create_graph(self, 
                    data: Union[pd.DataFrame, Dict, List[Tuple]], 
                    threshold: Optional[float] = None,
                    node_col1: str = 'File1',
                    node_col2: str = 'File2', 
                    weight_col: str = 'Hungarian_Distance',
                    threshold_mode: str = 'less_than',
                    show_progress: bool = True,
                    **kwargs) -> nx.Graph:
        """
        Create a NetworkX graph from input data with optional thresholding.
        
        Parameters:
        -----------
        data : pd.DataFrame, dict, or list of tuples
            Input data containing node pairs and edge weights
        threshold : float, optional
            Threshold value for filtering edges
        node_col1 : str, default='File1'
            Name of first node column (for DataFrame input)
        node_col2 : str, default='File2' 
            Name of second node column (for DataFrame input)
        weight_col : str, default='Hungarian_Distance'
            Name of weight column (for DataFrame input)
        threshold_mode : str, default='less_than'
            How to apply threshold: 'greater_than', 'less_than', 'equal', 'between'
        show_progress : bool, default=True
            Whether to show progress bars during graph creation
        **kwargs : dict
            Additional arguments for graph creation including:
            - include_structural_similarity: bool, whether to compute structural similarities
            - mnova_lookup_path: str, path to SMILES lookup data
            - num_processes: int, number of processes for parallel computation
            
        Returns:
        --------
        nx.Graph : The created NetworkX graph
        """
        
        if show_progress:
            print("Creating graph from input data...")
        
        # Initialize smiles_lookup if structural similarity is requested
        include_structural_similarity = kwargs.get('include_structural_similarity', False)
        if include_structural_similarity:
            mnova_lookup_path = kwargs.get('mnova_lookup_path')
            if mnova_lookup_path is None:
                raise ValueError("mnova_lookup_path must be provided when include_structural_similarity=True")
            self.smiles_lookup = self._load_smiles_lookup(mnova_lookup_path)
        else:
            self.smiles_lookup = {}
        
        # Convert input data to standardized format
        edges = self._parse_input_data(data, node_col1, node_col2, weight_col, show_progress)
        
        # Apply threshold filtering
        if threshold is not None:
            edges = self._apply_threshold(edges, threshold, threshold_mode, show_progress, **kwargs)
            
        # Create the graph
        self.graph = self._build_networkx_graph(edges, include_structural_similarity, show_progress=show_progress, **kwargs)
        
        # Store metadata
        self._update_metadata(data, threshold, threshold_mode, **kwargs)
        
        if show_progress:
            print(f"Graph creation complete! Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")
        
        return self.graph
    
    def _parse_input_data(self, data: Union[pd.DataFrame, Dict, List[Tuple]], 
                         node_col1: str, node_col2: str, weight_col: str, show_progress: bool = True) -> List[Tuple]:
        """Parse different input formats into standardized edge list."""
        
        if isinstance(data, pd.DataFrame):
            # Handle DataFrame input
            if node_col1 not in data.columns or node_col2 not in data.columns:
                raise ValueError(f"Columns {node_col1} and {node_col2} must exist in DataFrame")
                
            if show_progress:
                print(f"Parsing DataFrame with {len(data):,} rows...")
                
            if weight_col in data.columns:
                # Use vectorized operations for better performance
                edges = list(zip(data[node_col1].values, data[node_col2].values, data[weight_col].values))
            else:
                # No weights provided, use default weight of 1
                edges = list(zip(data[node_col1].values, data[node_col2].values, [1.0] * len(data)))
                        
        elif isinstance(data, dict):
            # Handle dictionary input: {(node1, node2): weight, ...}
            if show_progress:
                print(f"Parsing dictionary with {len(data):,} entries...")
            edges = [(k[0], k[1], v) for k, v in tqdm(data.items(), desc="Converting dict to edges", disable=not show_progress)]
            
        elif isinstance(data, list):
            # Handle list of tuples: [(node1, node2, weight), ...]
            if show_progress:
                print(f"Using edge list with {len(data):,} edges...")
            edges = data
            
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
            
        return edges
    

       
    def filter_edges_by_condition(self, condition_func=None, 
                                 inplace: bool = False,
                                 show_progress: bool = True) -> nx.Graph:
        """
        Filter edges based on a custom boolean condition function.
        
        Parameters:
        -----------
        condition_func : callable, optional
            Function that takes edge data and returns True to keep the edge, False to remove.
            If None, a placeholder function is used that keeps all edges.
            Function signature: condition_func(node1, node2, edge_data) -> bool
        inplace : bool, default=False
            Whether to modify the current graph in place or return a new graph
        show_progress : bool, default=True
            Whether to show progress bars during filtering
            
        Returns:
        --------
        nx.Graph : Filtered graph (new graph if inplace=False, current graph if inplace=True)
        
        Examples:
        ---------
        # Remove edges with low Tanimoto similarity
        def low_tanimoto_filter(node1, node2, edge_data):
            return edge_data.get('tanimoto_similarity', 0) > 0.5
            
        filtered_graph = builder.filter_edges_by_condition(low_tanimoto_filter)
        
        # Remove edges with high weight and low structural similarity
        def complex_filter(node1, node2, edge_data):
            weight = edge_data.get('weight', 0)
            hybrid_sim = edge_data.get('hybrid_similarity', 0)
            return weight < 20 and hybrid_sim > 0.3
        """
        
        if self.graph is None:
            raise ValueError("No graph has been loaded. Call create_graph() or load_graph() first.")
        
        if condition_func is None:
            # Placeholder condition - user should replace this
            def condition_func(node1, node2, edge_data):
                # TODO: Replace this with your custom condition
                # Examples:
                # return edge_data.get('tanimoto_similarity', 0) > 0.5
                return edge_data.get('weight', 0) < 25 and edge_data.get('hybrid_similarity', 0) > 0.5
                # return edge_data.get('mcs_similarity', 0) > 0.2
                return True  # Keep all edges by default
        
        if show_progress:
            print(f"Filtering edges based on custom condition...")
            print(f"Original graph: {self.graph.number_of_nodes():,} nodes, {self.graph.number_of_edges():,} edges")
        
        # Create new graph with same type and attributes
        if inplace:
            filtered_graph = self.graph
        else:
            if self.directed:
                filtered_graph = nx.DiGraph()
            else:
                filtered_graph = nx.Graph()
            
            # Copy graph attributes
            filtered_graph.graph.update(self.graph.graph)
            
            # Add all nodes with their attributes
            for node, data in self.graph.nodes(data=True):
                filtered_graph.add_node(node, **data)
        
        # Filter edges based on condition
        edges_to_keep = []
        edges_to_remove = []
        
        edge_iterator = tqdm(self.graph.edges(data=True), 
                           desc="Evaluating edge conditions", 
                           disable=not show_progress,
                           total=self.graph.number_of_edges())
        
        for node1, node2, edge_data in edge_iterator:
            try:
                if condition_func(node1, node2, edge_data):
                    edges_to_keep.append((node1, node2, edge_data))
                else:
                    edges_to_remove.append((node1, node2))
            except Exception as e:
                if show_progress:
                    print(f"Warning: Error evaluating condition for edge {node1}-{node2}: {e}")
                # Keep edge if condition evaluation fails
                edges_to_keep.append((node1, node2, edge_data))
        
        if inplace:
            # Remove edges that don't meet condition
            filtered_graph.remove_edges_from(edges_to_remove)
        else:
            # Add edges that meet condition
            for node1, node2, edge_data in edges_to_keep:
                filtered_graph.add_edge(node1, node2, **edge_data)
        
        if show_progress:
            print(f"Filtered graph: {filtered_graph.number_of_nodes():,} nodes, {filtered_graph.number_of_edges():,} edges")
            print(f"Removed {len(edges_to_remove):,} edges ({len(edges_to_remove)/self.graph.number_of_edges()*100:.1f}%)")
        
        if not inplace:
            return filtered_graph
        else:
            return self.graph

    def _apply_threshold(self, edges: List[Tuple], threshold: float, 
                        mode: str, show_progress: bool = True, **kwargs) -> List[Tuple]:
        """Apply threshold filtering to edges."""
        
        if show_progress:
            print(f"Applying threshold filtering ({mode} {threshold}) to {len(edges):,} edges...")
        
        filtered_edges = []
        
        # Create progress bar for threshold filtering
        edge_iterator = tqdm(edges, desc=f"Filtering edges ({mode} {threshold})", disable=not show_progress)
        
        for edge in edge_iterator:
            node1, node2, weight = edge
            
            if mode == 'greater_than':
                if weight > threshold:
                    filtered_edges.append(edge)
            elif mode == 'less_than':
                if weight < threshold:
                    filtered_edges.append(edge)
            elif mode == 'equal':
                if abs(weight - threshold) < kwargs.get('tolerance', 1e-6):
                    filtered_edges.append(edge)
            elif mode == 'between':
                min_val = kwargs.get('min_threshold', 0)
                max_val = kwargs.get('max_threshold', threshold)
                if min_val <= weight <= max_val:
                    filtered_edges.append(edge)
            else:
                raise ValueError(f"Unknown threshold mode: {mode}")
        
        if show_progress:
            print(f"Kept {len(filtered_edges):,} edges after filtering ({len(filtered_edges)/len(edges)*100:.1f}%)")
                
        return filtered_edges
    def _load_smiles_lookup(self, mnova_lookup_path: Union[str, Path]) -> Dict[str, str]:
        """Load SMILES lookup from parquet file."""
        print(f"Loading SMILES lookup from: {mnova_lookup_path}")
        lookup_df = pd.read_parquet(mnova_lookup_path)
        print(f"Loaded {len(lookup_df)} SMILES entries")
        
        # Create dictionary mapping id -> smiles
        smiles_dict = dict(zip(lookup_df['id'], lookup_df['smiles']))
        return smiles_dict
    
    def _compute_tanimoto_similarity(self, smiles1: str, smiles2: str) -> float:
        """Compute Tanimoto similarity between two SMILES strings."""
        if not RDKIT_AVAILABLE:
            return 0.0
            
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 2048)
        
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    
    def _compute_mcs_similarity(self, smiles1: str, smiles2: str) -> float:
        """Compute MCS similarity between two SMILES strings.

        Preference order:
        1) Library MCS (molecular_graph_edit) if available and valid
        2) RDKit MCS Tanimoto on maximum common substructure (normalized to [0,1])
        3) Fallback 0.0
        """
        # Try external MCS first
        if MCS_AVAILABLE:
            try:
                score = float(mcs_sim_d(
                    smiles1, smiles2,
                    num_tries=10,
                    num_keep_m=10,
                    num_keep_max_m=2
                ))
                if np.isfinite(score) and score >= 0.0:
                    # assume library returns a similarity in [0,1]
                    return max(0.0, min(1.0, score))
            except Exception:
                pass

        # RDKit-based MCS Tanimoto similarity (normalized)
        if RDKIT_AVAILABLE:
            try:
                mol1 = Chem.MolFromSmiles(smiles1)
                mol2 = Chem.MolFromSmiles(smiles2)
                if mol1 is None or mol2 is None:
                    return 0.0
                mcs = rdFMCS.FindMCS([mol1, mol2],
                                     atomCompare=rdFMCS.AtomCompare.CompareElements,
                                     bondCompare=rdFMCS.BondCompare.CompareOrder,
                                     completeRingsOnly=False,
                                     ringMatchesRingOnly=True,
                                     timeout=10)
                if not mcs.smartsString:
                    return 0.0
                mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
                if mcs_mol is None:
                    return 0.0
                mcs_atoms = mcs_mol.GetNumAtoms()
                n_atoms1 = mol1.GetNumAtoms()
                n_atoms2 = mol2.GetNumAtoms()
                denom = float(n_atoms1 + n_atoms2 - mcs_atoms)
                if denom <= 0:
                    return 0.0
                return max(0.0, min(1.0, mcs_atoms / denom))
            except Exception:
                pass

        return 0.0
    
    def _compute_hybrid_similarity(self, tanimoto_sim: float, mcs_sim: float) -> float:
        """Compute hybrid similarity as average of tanimoto and MCS."""
        return 0.5 * (tanimoto_sim + mcs_sim)
    
    def _build_networkx_graph(self, edges: List[Tuple], include_structural_similarity: bool, 
                              num_processes: Optional[int] = None, show_progress: bool = True, **kwargs) -> nx.Graph:
        """Build NetworkX graph from edge list."""
        
        if show_progress:
            print(f"Building NetworkX graph from {len(edges):,} edges...")
        
        # Choose graph type
        if self.directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
            
        # Handle structural similarity calculations
        if include_structural_similarity:
            if show_progress:
                print("Computing structural similarities...")
            
            # Use multiprocessing for structural similarity calculations if enabled
            if num_processes is not None and num_processes > 1:
                with mp.Pool(num_processes) as pool:
                    # Create a partial function that passes the smiles_lookup
                    worker_func = partial(_compute_edge_similarities, smiles_lookup=self.smiles_lookup)
                    
                    # Apply the partial function to each edge in parallel
                    results = list(tqdm(pool.imap(worker_func, edges), 
                                       desc="Computing structural similarities", 
                                       total=len(edges), 
                                       disable=not show_progress))
            else:
                # Single-process computation
                results = []
                edge_iterator = tqdm(edges, desc="Computing structural similarities", disable=not show_progress)
                for edge in edge_iterator:
                    results.append(_compute_edge_similarities(edge, self.smiles_lookup))
        else:
            # No structural similarity - just format edges
            results = []
            for edge in edges:
                if len(edge) == 3:
                    node1, node2, weight = edge
                else:
                    node1, node2 = edge
                    weight = 1.0
                results.append((node1, node2, weight, 0.0, 0.0, 0.0, 0.0))
                
        # Add edges to graph
        if show_progress:
            print("Adding edges to graph...")
        for node1, node2, weight, tanimoto_sim, mcs_sim, hybrid_sim, mass_diff in results:
            edge_attrs = {'weight': weight}
            
            if include_structural_similarity:
                edge_attrs.update({
                    'tanimoto_similarity': tanimoto_sim,
                    'mcs_similarity': mcs_sim,
                    'hybrid_similarity': hybrid_sim,
                    'mass_difference': mass_diff
                })
            
            G.add_edge(node1, node2, **edge_attrs)
                
        # Add node attributes if provided
        if 'node_attributes' in kwargs:
            if show_progress:
                print("Adding node attributes...")
            nx.set_node_attributes(G, kwargs['node_attributes'])
            
        # Add graph attributes
        if 'graph_attributes' in kwargs:
            G.graph.update(kwargs['graph_attributes'])
            
        return G



    def _update_metadata(self, data: Any, threshold: float, mode: str, **kwargs):
        """Update metadata about the graph creation process."""
        
        self.metadata = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'directed': self.directed,
            'threshold': threshold,
            'threshold_mode': mode,
            'input_type': type(data).__name__,
            'include_structural_similarity': kwargs.get('include_structural_similarity', False),
            'mnova_lookup_path': kwargs.get('mnova_lookup_path'),
            'num_processes': kwargs.get('num_processes'),
            'distance_function': kwargs.get('distance_function'),
            'weight_col': kwargs.get('weight_col'),
            'creation_params': kwargs
        }
    
    
    def save_graph(self, filepath: Union[str, Path], 
                  format: str = 'pickle', 
                  include_metadata: bool = True,
                  show_progress: bool = True):
        """
        Save the graph to disk.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to save the graph
        format : str, default='pickle'
            Format for saving: 'pickle', 'graphml', 'gexf', 'json'
        include_metadata : bool, default=True
            Whether to save metadata alongside the graph
        show_progress : bool, default=True
            Whether to show progress for large graphs
        """
        
        if self.graph is None:
            raise ValueError("No graph has been created yet. Call create_graph() first.")
            
        # Convert to Path and resolve to absolute path
        filepath = Path(filepath).resolve()
        
        # Create parent directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if show_progress:
            print(f"Saving graph with {self.graph.number_of_nodes():,} nodes and {self.graph.number_of_edges():,} edges...")
            print(f"Save location: {filepath}")
        
        try:
            if format == 'pickle':
                data_to_save = {
                    'graph': self.graph,
                    'metadata': self.metadata if include_metadata else None
                }
                with open(filepath, 'wb') as f:
                    pickle.dump(data_to_save, f)
                    
            elif format == 'graphml':
                nx.write_graphml(self.graph, filepath)
                
            elif format == 'gexf':
                nx.write_gexf(self.graph, filepath)
                
            elif format == 'json':
                # Convert to JSON-serializable format
                data = nx.node_link_data(self.graph)
                if include_metadata:
                    data['metadata'] = self.metadata
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                    
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            if show_progress:
                file_size = filepath.stat().st_size
                print(f"✅ Graph successfully saved to: {filepath}")
                print(f"📁 File size: {file_size / (1024**2):.2f} MB")
                
        except Exception as e:
            print(f"❌ Error saving graph: {e}")
            print(f"📁 Attempted save location: {filepath}")
            print(f"📁 Current working directory: {Path.cwd()}")
            raise
    
    def load_graph(self, filepath: Union[str, Path], format: str = 'pickle'):
        """
        Load a graph from disk.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the saved graph
        format : str, default='pickle'
            Format of the saved graph
        """
        
        filepath = Path(filepath).resolve()
        
        # Check if file exists
        if not filepath.exists():
            raise FileNotFoundError(f"Graph file not found: {filepath}")
        
        print(f"Loading graph from: {filepath}")
        
        try:
            if format == 'pickle':
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                self.graph = data['graph']
                self.metadata = data.get('metadata', {})
                
            elif format == 'graphml':
                self.graph = nx.read_graphml(filepath)
                
            elif format == 'gexf':
                self.graph = nx.read_gexf(filepath)
                
            elif format == 'json':
                with open(filepath, 'r') as f:
                    data = json.load(f)
                self.graph = nx.node_link_graph(data)
                self.metadata = data.get('metadata', {})
                
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            print(f"✅ Graph successfully loaded!")
            print(f"📊 Nodes: {self.graph.number_of_nodes():,}, Edges: {self.graph.number_of_edges():,}")
            
        except Exception as e:
            print(f"❌ Error loading graph: {e}")
            print(f"📁 Attempted load location: {filepath}")
            raise
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the graph."""
        
        if self.graph is None:
            return {}
            
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_connected(self.graph) if not self.directed else nx.is_strongly_connected(self.graph)
        }
        
        # Add weight statistics if graph is weighted
        if nx.is_weighted(self.graph):
            weights = [data['weight'] for _, _, data in self.graph.edges(data=True)]
            stats.update({
                'min_weight': min(weights),
                'max_weight': max(weights),
                'mean_weight': np.mean(weights),
                'median_weight': np.median(weights)
            })
            
        return stats
    
    def convert_to_molnet(self, 
                         output_dir: Union[str, Path] = "molnet_output",
                         mnova_lookup_path: Optional[Union[str, Path]] = None,
                         experimental_lookup_path: Optional[Union[str, Path]] = None,
                         edge_weight_column: str = "weight",
                         show_progress: bool = True) -> Dict[str, Any]:
        """
        Convert the current graph to molnet format with HSQC spectral data.
        Automatically includes similarity metrics and Leiden communities if they exist.
        
        Parameters:
        -----------
        output_dir : str or Path, default='molnet_output'
            Output directory for molnet files
        mnova_lookup_path : str or Path, optional
            Path to MNova lookup data (parquet or pickle format)
        experimental_lookup_path : str or Path, optional
            Path to experimental lookup data (pickle format)
        edge_weight_column : str, default='weight'
            Edge attribute to use as primary similarity score
        show_progress : bool, default=True
            Whether to show progress bars
            
        Returns:
        --------
        dict : Summary of conversion results
        """
        
        if self.graph is None:
            raise ValueError("No graph has been created yet. Call create_graph() first.")
        
        if not MOLNET_AVAILABLE:
            raise ImportError("MolNet converter not available. Please check molnet_converter.py file.")
        
        # Export all edge features by default; detection not required anymore
        available_similarity_metrics = None
        
        # Export all node attributes by default
        available_node_attributes = None
        
        # Initialize converter
        converter = MolNetConverter(
            mnova_lookup_path=mnova_lookup_path,
            experimental_lookup_path=experimental_lookup_path
        )
        
        # Convert graph with detected similarity metrics and node attributes
        summary = converter.convert_graph_to_molnet(
            graph=self.graph,
            output_dir=output_dir,
            edge_weight_column=edge_weight_column,
            additional_edge_attributes=available_similarity_metrics,
            node_attributes=available_node_attributes,
            preview_count=5,
            show_progress=show_progress
        )
        
        # Add information about included features to summary
        summary['included_similarity_metrics'] = available_similarity_metrics
        summary['included_node_attributes'] = available_node_attributes
        summary['primary_edge_weight'] = edge_weight_column
        
        return summary
    
    def convert_to_molnet_with_hybrid_weights(self, 
                                            output_dir: Union[str, Path] = "molnet_output",
                                            mnova_lookup_path: Optional[Union[str, Path]] = None,
                                            experimental_lookup_path: Optional[Union[str, Path]] = None,
                                            hybrid_weight_mode: str = "hybrid_similarity",
                                            fallback_weight: str = "weight",
                                            show_progress: bool = True) -> Dict[str, Any]:
        """
        Convert to molnet format using hybrid similarity as primary edge weight.
        
        This is a convenience method that prioritizes structural similarity metrics
        over Hungarian distance for the primary edge weights.
        
        Parameters:
        -----------
        output_dir : str or Path, default='molnet_output'
            Output directory for molnet files
        mnova_lookup_path : str or Path, optional
            Path to MNova lookup data
        experimental_lookup_path : str or Path, optional
            Path to experimental lookup data
        hybrid_weight_mode : str, default='hybrid_similarity'
            Primary similarity metric to use as edge weight:
            - 'hybrid_similarity': Combined tanimoto + MCS similarity
            - 'tanimoto_similarity': Tanimoto fingerprint similarity
            - 'mcs_similarity': Maximum common substructure similarity
        fallback_weight : str, default='weight'
            Fallback edge attribute if hybrid_weight_mode is not available
        show_progress : bool, default=True
            Whether to show progress bars
            
        Returns:
        --------
        dict : Summary of conversion results
        """
        
        if self.graph is None:
            raise ValueError("No graph has been created yet. Call create_graph() first.")
        
        # Check if the requested hybrid weight mode is available
        sample_edge = next(iter(self.graph.edges(data=True)), None)
        if sample_edge and hybrid_weight_mode in sample_edge[2]:
            primary_weight = hybrid_weight_mode
            if show_progress:
                print(f"Using {hybrid_weight_mode} as primary edge weight")
        else:
            primary_weight = fallback_weight
            if show_progress:
                print(f"⚠️  {hybrid_weight_mode} not found, falling back to {fallback_weight}")
        
        # Convert with all similarity metrics included
        return self.convert_to_molnet(
            output_dir=output_dir,
            mnova_lookup_path=mnova_lookup_path,
            experimental_lookup_path=experimental_lookup_path,
            edge_weight_column=primary_weight,
            show_progress=show_progress
        )

    def add_query_to_graph_and_export(self,
                                     query_hsqc_data: np.ndarray,
                                     lookup_hsqc_data: Dict[str, np.ndarray],
                                     query_id: str,
                                     similarity_threshold: float = 50.0,
                                     distance_function: Optional[str] = None,
                                     query_smiles: Optional[str] = None,
                                     output_dir: Optional[Union[str, Path]] = None,
                                     export_format: str = 'json',
                                     show_progress: bool = True) -> Dict[str, Any]:
        """
        Add a query compound to the graph and export ONLY the new nodes and edges.
        
        This function:
        1. Adds the query compound as a new node to the graph
        2. Calculates distances to all nodes using the specified distance function
        3. Creates edges to nodes within the similarity threshold
        4. Extracts ONLY the query subgraph (query node + connected neighbors)
        5. Exports the subgraph to the specified format
        
        Parameters:
        -----------
        query_hsqc_data : np.ndarray
            HSQC peak data for the query compound, shape (n_peaks, 2)
        lookup_hsqc_data : Dict[str, np.ndarray]
            Mapping of {node_id: hsqc_array} for calculating distances
        query_id : str
            Unique identifier for the query compound
        similarity_threshold : float, default=50.0
            Maximum distance to create an edge (lower = more similar)
        distance_function : str, optional
            Distance function to use. If None, will try to detect from graph metadata.
            Options include: 'modified_hungarian', 'hungarian_nn_sum', 'hungarian_nn_mean', etc.
        query_smiles : str, optional
            SMILES string for the query compound
        output_dir : str or Path, optional
            Output directory. If None, uses f"query_subgraph_{query_id}"
        export_format : str, default='json'
            Export format: 'json', 'molnet', 'graphml'
        show_progress : bool, default=True
            Whether to show progress bars
            
        Returns:
        --------
        dict : Summary of results including file paths and statistics
        """
        
        if self.graph is None:
            raise ValueError("No graph loaded. Call load_graph() or create_graph() first.")
        
        # Auto-detect distance function from graph metadata if not provided
        if distance_function is None:
            # Try to get from graph metadata
            graph_distance_func = self.metadata.get('distance_function')
            if graph_distance_func:
                distance_function = graph_distance_func
                if show_progress:
                    print(f"🔍 Auto-detected distance function from graph metadata: {distance_function}")
            else:
                # Try to get from graph attributes
                graph_distance_func = self.graph.graph.get('distance_function')
                if graph_distance_func:
                    distance_function = graph_distance_func
                    if show_progress:
                        print(f"🔍 Auto-detected distance function from graph attributes: {distance_function}")
                else:
                    # Default fallback
                    distance_function = 'modified_hungarian'
                    if show_progress:
                        print(f"⚠️  Could not detect distance function, using default: {distance_function}")
        
        if distance_function not in DISTANCE_FUNCTION_MAP:
            raise ValueError(f"Unknown distance function: {distance_function}. "
                           f"Available options: {list(DISTANCE_FUNCTION_MAP.keys())}")
        
        if show_progress:
            print(f"🔗 Adding query '{query_id}' to graph...")
            print(f"   📊 Original graph: {self.graph.number_of_nodes():,} nodes, {self.graph.number_of_edges():,} edges")
            print(f"   📏 Distance function: {distance_function}")
            print(f"   🎯 Similarity threshold: {similarity_threshold}")
            print(f"   🧪 HSQC peaks: {query_hsqc_data.shape[0]}")
        
        # Get distance function and parameters
        dist_func = DISTANCE_FUNCTION_MAP[distance_function]
        dist_params = DISTANCE_FUNCTION_PARAMS[distance_function].copy()
        
        # Create a copy of the graph
        updated_graph = self.graph.copy()
        
        # Check what attributes exist on existing nodes/edges for consistency
        sample_node = next(iter(updated_graph.nodes(data=True)), (None, {}))
        sample_edge = next(iter(updated_graph.edges(data=True)), (None, None, {}))
        
        node_attrs = sample_node[1] if sample_node[0] else {}
        edge_attrs = sample_edge[2] if sample_edge[0] else {}
        
        # Prepare query node attributes
        query_node_attrs = {
            'node_type': 'query',
            'smiles': query_smiles or 'Unknown',
        }
        
        # Add any similarity attributes that exist on other nodes
        similarity_attrs = ['hybrid_score', 'mcs_score', 'tanimoto_similarity']
        for attr in similarity_attrs:
            if attr in node_attrs:
                query_node_attrs[attr] = 1.0  # Perfect similarity with query
        
        # Add query node
        updated_graph.add_node(query_id, **query_node_attrs)
        
        # Calculate distances and add edges
        edges_added = 0
        calculated_distances = {}
        
        if show_progress:
            print(f"   🔄 Calculating distances to {len(lookup_hsqc_data):,} compounds...")
        
        for node_id in tqdm(updated_graph.nodes(), desc="Computing distances", disable=not show_progress):
            # Skip the query node itself
            if node_id == query_id:
                continue
                
            if node_id not in lookup_hsqc_data:
                continue
            
            hsqc_array = lookup_hsqc_data[node_id]
                
            try:
                # Calculate distance using the specified function
                if len(dist_params) == 0:
                    distance = dist_func(query_hsqc_data, hsqc_array)
                else:
                    distance = dist_func(query_hsqc_data, hsqc_array, **dist_params)
                
                # Extract distance value if it's a tuple
                if isinstance(distance, tuple):
                    distance = distance[0]
                
                calculated_distances[node_id] = distance
                
                # Add edge if within threshold
                if distance <= similarity_threshold:
                    # Prepare edge attributes
                    new_edge_attrs = {
                        'weight': distance,
                        'hungarian_distance': distance,
                        'edge_type': 'query_connection',
                        'distance_function': distance_function
                    }
                    
                    # Set similarity scores to 1.0 for query edges
                    new_edge_attrs['hybrid_similarity'] = 1.0
                    new_edge_attrs['tanimoto_similarity'] = 1.0
                    new_edge_attrs['mcs_similarity'] = 1.0
                    
                    # Add any other edge attributes that exist in the graph
                    for attr in ['hybrid_score', 'mcs_score']:
                        if attr in edge_attrs:
                            new_edge_attrs[attr] = 1.0
                    
                    updated_graph.add_edge(query_id, node_id, **new_edge_attrs)
                    edges_added += 1
                    
            except Exception as e:
                if show_progress:
                    print(f"   ⚠️  Error calculating distance for {node_id}: {e}")
                continue
        
        if show_progress:
            print(f"   ✅ Added {edges_added} edges to query node")
        
        # Extract query subgraph (ONLY new nodes and edges)
        query_neighbors = list(updated_graph.neighbors(query_id))
        subgraph_nodes = [query_id] + query_neighbors
        query_subgraph = updated_graph.subgraph(subgraph_nodes).copy()
        
        if show_progress:
            print(f"   ✂️  Extracted subgraph: {query_subgraph.number_of_nodes()} nodes, {query_subgraph.number_of_edges()} edges")
            reduction_pct = (1 - query_subgraph.number_of_nodes() / updated_graph.number_of_nodes()) * 100
            print(f"   🎯 Size reduction: {reduction_pct:.1f}% fewer nodes")
        
        # Set up output directory
        if output_dir is None:
            output_dir = f"query_subgraph_{query_id.replace('/', '_').replace('.', '_')}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export subgraph
        files_created = []
        
        if export_format == 'json':
            # Export as JSON
            nodes_file = output_path / 'nodes.json'
            edges_file = output_path / 'edges.json'
            
            # Prepare nodes data - ONLY include the query node
            nodes_data = []
            query_node_attrs = query_subgraph.nodes[query_id]
            query_node_data = {'id': query_id, **query_node_attrs}
            nodes_data.append(query_node_data)
            
            # Prepare edges data - include all edges in the subgraph
            edges_data = []
            for source, target, attrs in query_subgraph.edges(data=True):
                edge_data = {'source': source, 'target': target, **attrs}
                edges_data.append(edge_data)
            
            # Save JSON files
            with open(nodes_file, 'w') as f:
                json.dump(nodes_data, f, indent=2, default=str)
            with open(edges_file, 'w') as f:
                json.dump(edges_data, f, indent=2, default=str)
            
            files_created = [str(nodes_file), str(edges_file)]
            
        elif export_format == 'molnet':
            # Export as MolNet format
            temp_builder = GraphBuilder()
            temp_builder.graph = query_subgraph
            
            molnet_summary = temp_builder.convert_to_molnet(
                output_dir=output_dir,
                show_progress=show_progress
            )
            files_created = molnet_summary.get('files_created', [])
            
        elif export_format == 'graphml':
            # Export as GraphML
            graphml_file = output_path / 'subgraph.graphml'
            nx.write_graphml(query_subgraph, graphml_file)
            files_created = [str(graphml_file)]
            
        else:
            raise ValueError(f"Unknown export format: {export_format}")
        
        # Create summary
        summary = {
            'query_id': query_id,
            'query_smiles': query_smiles,
            'query_hsqc_peaks': query_hsqc_data.shape[0],
            'distance_function': distance_function,
            'similarity_threshold': similarity_threshold,
            'original_graph_nodes': self.graph.number_of_nodes(),
            'original_graph_edges': self.graph.number_of_edges(),
            'query_connections': edges_added,
            'subgraph_nodes': query_subgraph.number_of_nodes(),
            'subgraph_edges': query_subgraph.number_of_edges(),
            'exported_nodes': 1,  # Only query node is exported to nodes.json
            'exported_edges': query_subgraph.number_of_edges(),  # All edges are exported
            'export_format': export_format,
            'output_directory': str(output_path),
            'files_created': files_created,
            'query_neighbors': query_neighbors,
            'size_reduction_percent': (1 - query_subgraph.number_of_nodes() / updated_graph.number_of_nodes()) * 100
        }
        
        if show_progress:
            print(f"   💾 Exported {export_format} files to: {output_path}")
            print(f"   📁 Files created: {len(files_created)}")
            for file_path in files_created:
                print(f"      • {file_path}")
            if export_format == 'json':
                print(f"   📊 Nodes file contains: 1 node (query only)")
                print(f"   📈 Edges file contains: {query_subgraph.number_of_edges()} edges (all connections)")
        
        return summary

    def add_molecular_formulas_to_nodes(self,
                                        mnova_lookup_path: Optional[Union[str, Path]] = None,
                                        smiles_attr: str = 'smiles',
                                        formula_attr: str = 'molecular_formula',
                                        show_progress: bool = True) -> Dict[str, Any]:
        """
        Compute molecular formulas (e.g., CxHyOz...) for nodes and store them as node attributes.

        This function uses RDKit to calculate the molecular formula for each node using a SMILES
        string. SMILES can be sourced from an existing node attribute or from a lookup file.

        Parameters:
        -----------
        mnova_lookup_path : str or Path, optional
            Path to a parquet file containing `id` and `smiles` columns to lookup SMILES by node id.
        smiles_attr : str, default='smiles'
            Node attribute name that may contain SMILES strings.
        formula_attr : str, default='molecular_formula'
            Node attribute name to store the computed molecular formula string.
        show_progress : bool, default=True
            Whether to show progress output.

        Returns:
        --------
        dict : Summary statistics of the operation.
        """
        if self.graph is None:
            raise ValueError("No graph has been loaded. Call load_graph() or create_graph() first.")

        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for molecular formula calculations. Please install RDKit.")

        # Optional SMILES lookup
        smiles_lookup: Dict[str, str] = {}
        if mnova_lookup_path is not None:
            smiles_lookup = self._load_smiles_lookup(mnova_lookup_path)

        if show_progress:
            print(f"🧪 Adding molecular formulas to {self.graph.number_of_nodes():,} nodes...")
            if mnova_lookup_path is not None:
                print(f"📖 Using SMILES lookup: {Path(mnova_lookup_path).resolve()}")

        total_nodes = self.graph.number_of_nodes()
        nodes_with_smiles = 0
        nodes_with_formula = 0
        nodes_without_smiles = 0
        nodes_failed = 0

        node_iterator = tqdm(self.graph.nodes(data=True),
                             desc="Computing molecular formulas",
                             disable=not show_progress,
                             total=total_nodes)

        for node_id, node_attrs in node_iterator:
            smiles: Optional[str] = None

            # Priority 1: existing node attribute
            if isinstance(node_attrs, dict):
                smiles = node_attrs.get(smiles_attr)

            # Priority 2: lookup table
            if not smiles and node_id in smiles_lookup:
                smiles = smiles_lookup[node_id]

            if not smiles or not isinstance(smiles, str):
                nodes_without_smiles += 1
                continue

            nodes_with_smiles += 1

            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    nodes_failed += 1
                    continue

                formula = rdMolDescriptors.CalcMolFormula(mol)
                self.graph.nodes[node_id][formula_attr] = formula
                nodes_with_formula += 1

            except Exception:
                nodes_failed += 1
                continue

        summary = {
            'total_nodes': total_nodes,
            'nodes_with_smiles': nodes_with_smiles,
            'nodes_with_formula': nodes_with_formula,
            'nodes_without_smiles': nodes_without_smiles,
            'nodes_failed': nodes_failed,
            'formula_attribute': formula_attr
        }

        if show_progress:
            print("✅ Molecular formula annotation complete!")
            print(f"   📊 Total nodes: {total_nodes:,}")
            print(f"   🧬 Nodes with SMILES: {nodes_with_smiles:,}")
            print(f"   🧾 Nodes annotated with formula: {nodes_with_formula:,}")
            print(f"   ⚠️  Nodes without SMILES: {nodes_without_smiles:,}")
            print(f"   ❌ Nodes failed to annotate: {nodes_failed:,}")

        return summary

    def annotate_leiden_communities(self, resolution: float = 0.1, show_progress: bool = True) -> Dict[str, Any]:
        """
        Annotate the graph with Leiden communities.
        Adds a 'leiden_community' attribute to each node in the graph.
        Returns a summary dictionary with community assignments.
        """
        if self.graph is None:
            raise ValueError("No graph has been created yet. Call create_graph() first.")

        leiden_communities = leiden(self.graph, resolution_parameter=resolution)
        # Assign community index to each node as an attribute
        for idx, community in enumerate(leiden_communities.communities):
            for node in community:
                self.graph.nodes[node]['leiden_community'] = idx

        # Optionally show progress or print summary
        if show_progress:
            print(f"Annotated {self.graph.number_of_nodes()} nodes with Leiden community indices.")
            print(f"Number of communities found: {len(leiden_communities.communities)}")

        # Return a summary dictionary
        node_to_community = {node: self.graph.nodes[node]['leiden_community'] for node in self.graph.nodes}
        return {
            "n_communities": len(leiden_communities.communities),
            "node_to_community": node_to_community,
            "communities": [list(community) for community in leiden_communities.communities]
        }

    def add_mass_differences_to_graph(self, 
                                    mnova_lookup_path: Union[str, Path],
                                    show_progress: bool = True) -> Dict[str, Any]:
        """
        Add mass differences to all edges in an already loaded graph.
        
        This function:
        1. Loads SMILES lookup data from the provided path
        2. Computes mass differences for all edges in the graph
        3. Adds 'mass_difference' attribute to each edge
        
        Parameters:
        -----------
        mnova_lookup_path : str or Path
            Path to MNova lookup data (parquet or pickle format)
        show_progress : bool, default=True
            Whether to show progress bars during computation
            
        Returns:
        --------
        dict : Summary of mass difference addition including statistics
        """
        if self.graph is None:
            raise ValueError("No graph has been loaded. Call load_graph() or create_graph() first.")
        
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for mass difference calculations. Please install RDKit.")
        
        if show_progress:
            print(f"🔬 Adding mass differences to graph with {self.graph.number_of_nodes():,} nodes and {self.graph.number_of_edges():,} edges...")
        
        # Load SMILES lookup data
        smiles_lookup = self._load_smiles_lookup(mnova_lookup_path)
        
        if show_progress:
            print(f"📊 Loaded {len(smiles_lookup):,} SMILES entries")
        
        # Compute mass differences for all edges
        edges_with_mass_diff = 0
        edges_without_smiles = 0
        mass_differences = []
        
        edge_iterator = tqdm(self.graph.edges(data=True), 
                           desc="Computing mass differences", 
                           disable=not show_progress,
                           total=self.graph.number_of_edges())
        
        for node1, node2, edge_data in edge_iterator:
            # Get SMILES strings for both nodes
            smiles1 = smiles_lookup.get(node1)
            smiles2 = smiles_lookup.get(node2)
            
            if smiles1 and smiles2:
                mass_diff = smiles_mass_difference(smiles1, smiles2)
                edge_data['mass_difference'] = mass_diff
                mass_differences.append(mass_diff)
                edges_with_mass_diff += 1

        
        # Calculate statistics
        non_zero_mass_diffs = [md for md in mass_differences if md > 0]
        
        summary = {
            'total_edges': self.graph.number_of_edges(),
            'edges_with_mass_diff': edges_with_mass_diff,
            'edges_without_smiles': edges_without_smiles,
            'mean_mass_difference': sum(mass_differences) / len(mass_differences) if mass_differences else 0.0,
            'max_mass_difference': max(mass_differences) if mass_differences else 0.0,
            'min_mass_difference': min(mass_differences) if mass_differences else 0.0,
            'non_zero_mass_diffs': len(non_zero_mass_diffs),
            'mean_non_zero_mass_diff': sum(non_zero_mass_diffs) / len(non_zero_mass_diffs) if non_zero_mass_diffs else 0.0
        }
        
        if show_progress:
            print(f"✅ Mass differences added successfully!")
            print(f"   📊 Total edges processed: {summary['total_edges']:,}")
            print(f"   🔬 Edges with mass differences: {summary['edges_with_mass_diff']:,}")
            print(f"   ⚠️  Edges without SMILES: {summary['edges_without_smiles']:,}")
            print(f"   📈 Mean mass difference: {summary['mean_mass_difference']:.3f}")
            print(f"   📈 Max mass difference: {summary['max_mass_difference']:.3f}")
            print(f"   📈 Min mass difference: {summary['min_mass_difference']:.3f}")
            if non_zero_mass_diffs:
                print(f"   📈 Mean non-zero mass difference: {summary['mean_non_zero_mass_diff']:.3f}")
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Example with sample data
    # data = pd.read_parquet('/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/consolidated_hsqc_edges_filtered_low_75_15.pq')
    
    # # Create graph builder
    builder = GraphBuilder()
    
    # # Create graph with structural similarity using parallel processing
    # threshold = 25
    # weight_col = 'func_low_dist'
    # save_path = f'storage/Graph_{weight_col}_{threshold}_structure_similarity.pkl'
    
    # print("=== Creating Graph with Structural Similarity (Parallel Processing) ===")
    # builder.create_graph(
    #     data=data, 
    #     threshold=threshold,
    #     node_col1='File1',
    #     node_col2='File2', 
    #     weight_col=weight_col,
    #     threshold_mode='less_than',
    #     include_structural_similarity=True,
    #     mnova_lookup_path='/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/annotate/MNova_lookup_subset.parquet',
    #     num_processes=None,  # Use all available cores
    #     show_progress=True
    # )
    
    # # Show sample edges with structural similarity
    # print("\n=== Sample Edges with Structural Similarity ===")
    # edge_count = 0
    # for node1, node2, data in builder.graph.edges(data=True):
    #     if edge_count < 5:  # Show first 5 edges
    #         print(f"Edge {node1} -> {node2}:")
    #         print(f"  Weight: {data.get('weight', 'N/A')}")
    #         print(f"  Tanimoto: {data.get('tanimoto_similarity', 'N/A'):.3f}")
    #         print(f"  MCS: {data.get('mcs_similarity', 'N/A'):.3f}")
    #         print(f"  Hybrid: {data.get('hybrid_similarity', 'N/A'):.3f}")
    #         edge_count += 1
    
    # # Save the graph
    # builder.save_graph(save_path)
    # print(f"\nGraph saved to: {save_path}")
    mnova_lookup_path = '/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/MNova_lookup_subset_100k.parquet'
    builder = GraphBuilder()
    builder.load_graph('/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/storage/GRAPH_30_0.6_LEIDEN_FINAL_100k_with_mass_diff.pkl')

    # Add molecular formulas to nodes
    builder.add_molecular_formulas_to_nodes(
        mnova_lookup_path=mnova_lookup_path,
        smiles_attr='smiles',
        formula_attr='molecular_formula',
        show_progress=True
    )
    c = 0
    builder.add_mass_differences_to_graph(
        mnova_lookup_path=mnova_lookup_path,
        show_progress=True
    )
    builder.annotate_leiden_communities(resolution=0.1, show_progress=True)
    # Export to MolNet format including all node and edge attributes with preview
    molnet_summary = builder.convert_to_molnet(
        output_dir="/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/molnet_output_full_export",
        mnova_lookup_path=mnova_lookup_path,
        experimental_lookup_path='/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/ExperimentalHSQC_Lookup.pkl',
        edge_weight_column="weight",
        show_progress=True
    )

    print("\nMolNet export summary (full export):")
    for key, value in molnet_summary.items():
        print(f"  {key}: {value}")


    # builder.filter_edges_by_condition(condition_func=lambda node1, node2, edge_data: edge_data.get('weight', 0) < 25 and edge_data.get('hybrid_similarity', 0) > 0.6, inplace=True)
    # builder.annotate_leiden_communities(resolution=0.1, show_progress=True)
    
    # Print sample edges and nodes
    print("\n=== Sample Nodes ===")
    print(f"Total nodes: {len(builder.graph.nodes())}")
    for i, node in enumerate(list(builder.graph.nodes())[:5]):
        print(f"\nNode {i+1}: {node}")
        node_data = builder.graph.nodes[node]
        for key, value in node_data.items():
            print(f"  {key}: {value}")
            
    print("\n=== Sample Edges ===") 
    print(f"Total edges: {len(builder.graph.edges())}")
    for i, (node1, node2, data) in enumerate(list(builder.graph.edges(data=True))[:5]):
        print(f"\nEdge {i+1}: {node1} -> {node2}")
        for key, value in data.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    
    builder.save_graph('/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/storage/Graph_func_low_dist_25_structure_similarity_filtered_06hybrid_leiden.pkl')
    
    # Example: Convert to molnet format with automatic similarity metric detection
    print("\n=== Converting to MolNet Format with Automatic Similarity Detection ===")
    
    # Method 1: Use Hungarian distance as primary weight (similarity metrics auto-detected)
    molnet_summary = builder.convert_to_molnet(
        output_dir="molnet_output_hungarian_primary_leiden",
        mnova_lookup_path=mnova_lookup_path,
        experimental_lookup_path='/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/ExperimentalHSQC_Lookup.pkl',
        edge_weight_column="weight",  # Hungarian distance as primary
        show_progress=True
    )
    
    # # Method 2: Use hybrid similarity as primary weight (convenience method)
    # molnet_summary_hybrid = builder.convert_to_molnet_with_hybrid_weights(
    #     output_dir="molnet_output_hybrid_primary",
    #     mnova_lookup_path='/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/annotate/MNova_lookup_subset.pkl',
    #     experimental_lookup_path='/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/expt_filtered_lookup.pkl',
    #     hybrid_weight_mode="hybrid_similarity",  # Use hybrid similarity as primary weight
    #     show_progress=True
    # )
    
    # print(f"MolNet conversion summaries:")
    # print(f"  Hungarian primary: {molnet_summary}")
    # print(f"  Hybrid primary: {molnet_summary_hybrid}")
    # print(f"  Auto-detected metrics: {molnet_summary.get('included_similarity_metrics', [])}")
    # Show metadata
    print("\n=== Graph Metadata ===")
    for key, value in builder.metadata.items():
        print(f"{key}: {value}")
    
    # Show statistics for structural similarity
    # Show a sample edge with all its data
    print("\n=== Sample Edge Data ===")
    edge = next(iter(builder.graph.edges(data=True)))
    print(f"Edge {edge[0]} -> {edge[1]}:")
    for key, value in edge[2].items():
        print(f"  {key}: {value}")
    
    if builder.metadata.get('include_structural_similarity'):
        print("\n=== Structural Similarity Statistics ===")
        
        # Collect all similarity values
        tanimoto_values = []
        mcs_values = []
        hybrid_values = []
        hungarian_values = []
        
        for _, _, data in builder.graph.edges(data=True):
            tanimoto_values.append(data.get('tanimoto_similarity', 0.0))
            mcs_values.append(data.get('mcs_similarity', 0.0))
            hybrid_values.append(data.get('hybrid_similarity', 0.0))
            hungarian_values.append(data.get('weight', 0.0))
        
        print(f"Tanimoto - Mean: {np.mean(tanimoto_values):.3f}, Std: {np.std(tanimoto_values):.3f}")
        print(f"MCS - Mean: {np.mean(mcs_values):.3f}, Std: {np.std(mcs_values):.3f}")
        print(f"Hybrid - Mean: {np.mean(hybrid_values):.3f}, Std: {np.std(hybrid_values):.3f}")
        
        # Count non-zero values
        non_zero_tanimoto = np.sum(np.array(tanimoto_values) > 0)
        non_zero_mcs = np.sum(np.array(mcs_values) > 0)
        non_zero_hybrid = np.sum(np.array(hybrid_values) > 0)
        # non_zero_hungarian = np.sum(np.array(hungarian_values) > 0)
        
        # print(f"Non-zero similarities: Tanimoto={non_zero_tanimoto}, MCS={non_zero_mcs}, Hybrid={non_zero_hybrid}, Hungarian={non_zero_hungarian}")
        # print(f"Total edges: {len(tanimoto_values)}")
        
        # Create histograms
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(21, 5))
        plt.subplots_adjust(wspace=0.3)  # Increase horizontal spacing between subplots

        print(len(tanimoto_values), len(mcs_values), len(hybrid_values))

        ax1.hist(tanimoto_values, bins=50, alpha=0.75)
        ax1.set_title(f'n_edges = {len(tanimoto_values)}')
        ax1.set_xlabel('Tanimoto Similarity')
        ax1.set_ylabel('Count')
        
        ax2.hist(mcs_values, bins=50, alpha=0.75)
        ax2.set_title(f'n_edges = {len(mcs_values)}')
        ax2.set_xlabel('MCS Similarity')
        ax2.set_ylabel('Count')
        
        ax3.hist(hybrid_values, bins=50, alpha=0.75)
        ax3.set_title(f'n_edges = {len(hybrid_values)}')
        ax3.set_xlabel('Hybrid Similarity')
        ax3.set_ylabel('Count')

        ax4.hist(hungarian_values, bins=50, alpha=0.75)
        ax4.set_title(f'n_edges = {len(hungarian_values)}')
        ax4.set_xlabel('Modified Hungarian Distance')
        ax4.set_ylabel('Count')
        
        plt.savefig(f'/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/figures/histogram_structural_filtered.png')