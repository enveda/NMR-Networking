import json
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import math
from typing import List, Dict, Tuple, Optional, Any
import warnings
from scipy.sparse import csr_matrix
import bisect
import random

# Legacy numba/njit removed – no acceleration layer required

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from nmr_networking.graph.builder import GraphBuilder
from nmr_networking.similarity import DISTANCE_FUNCTION_MAP, DISTANCE_FUNCTION_PARAMS

"""
Distance Function Architecture:

This module now exclusively uses the distance functions from cost_functions.py, providing
a unified interface for Hungarian distance calculations with different strategies:

1. Standard Hungarian distances with different unmatched peak handling strategies:
   - 'nn' (nearest neighbor): unmatched peaks assigned to nearest neighbor
   - 'trunc' (truncation): only optimal 1:1 matches for min(n1,n2) peaks
   - 'zero' (zero-padding): pad shorter list with (0,0) peaks
   
2. Distance reduction methods:
   - 'sum': sum of all distances
   - 'mean': mean of all distances
   
3. Modified Hungarian distance with uncertainty normalization and functional group matching

Available distance functions:
- hungarian_nn_sum/mean: Standard Hungarian with nearest neighbor strategy
- hungarian_trunc_sum/mean: Standard Hungarian with truncation strategy  
- hungarian_zero_sum/mean: Standard Hungarian with zero-padding strategy
- modified_hungarian: Uncertainty-normalized Hungarian with functional group matching (default strategy)
- modified_hungarian_zero: Uncertainty-normalized Hungarian with zero-padding strategy
- modified_hungarian_nn: Uncertainty-normalized Hungarian with nearest neighbor strategy
- modified_hungarian_trunc: Uncertainty-normalized Hungarian with truncation strategy

Legacy function names are maintained for backward compatibility.
"""

class AnnotationAnalyzer:
    """
    A class for analyzing annotation objects from ablate_database.py within the context 
    of a molecular similarity graph.
    
    Handles parsing of annotation tuples, integration with NetworkX graphs,
    and structural similarity analysis of query compounds and their neighbors.
    """
    
    def __init__(self, graph: Optional[nx.Graph] = None, graph_path: Optional[str] = None,
                 distance_column: str = 'hungarian_nn_sum', verbose: bool = True):
        """
        Initialize the AnnotationAnalyzer.
        
        Parameters:
        -----------
        graph : nx.Graph, optional
            Pre-loaded NetworkX graph
        graph_path : str, optional
            Path to saved graph file to load
        distance_column : str, default='hungarian_nn_sum'
            Which distance column/function was used to create the graph.
            Standard options: 'hungarian_nn_sum', 'hungarian_nn_mean', 'hungarian_trunc_sum', 
                             'hungarian_trunc_mean', 'hungarian_zero_sum', 'hungarian_zero_mean'
            Modified options: 'modified_hungarian', 'modified_hungarian_zero', 'modified_hungarian_nn',
                             'modified_hungarian_trunc'
            Legacy options (for backward compatibility): 'Hungarian_Distance', 'hung_norm', 
                                                        'hung_modified', 'hung_modified_2', 'hung_sum'
        verbose : bool, default=True
            Whether to print verbose output during analysis.
        """
        self.graph = None
        self.annotation_data = None
        self.query_info = None
        self.lookup_results = None
        self.last_calculated_distances = {}  # Store distances from last add_query_to_graph call
        self.verbose = verbose
        
        
        # Distance function configuration
        self.distance_column = distance_column
        if distance_column not in DISTANCE_FUNCTION_MAP:
            raise ValueError(f"Unknown distance column: {distance_column}. "
                           f"Available options: {list(DISTANCE_FUNCTION_MAP.keys())}")
        
        self.distance_function = DISTANCE_FUNCTION_MAP[distance_column]
        self.distance_params = DISTANCE_FUNCTION_PARAMS[distance_column].copy()
        
        if self.verbose:
            print(f"Using distance function for '{distance_column}': {self.distance_function.__name__}")
            if self.distance_params:
                print(f"Distance function parameters: {self.distance_params}")
        
        # Load graph
        if graph is not None:
            self.graph = graph
        elif graph_path is not None:
            self.load_graph(graph_path)
        else:
            warnings.warn("No graph provided. Use load_graph() or set_graph() before analysis.")
    
    def load_graph(self, graph_path: str, format: str = 'pickle'):
        """
        Load a graph from file.
        
        Parameters:
        -----------
        graph_path : str
            Path to the graph file
        format : str, default='pickle'
            Format of the graph file
        """
        builder = GraphBuilder()
        builder.load_graph(graph_path, format=format)
        self.graph = builder.graph
        if self.verbose:
            print(f"Graph loaded: {self.graph.number_of_nodes():,} nodes, {self.graph.number_of_edges():,} edges")
    
    def parse_annotation(self, annotation_tuple: Tuple[str, str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Parse an annotation tuple from ablate_database.py output.
        
        Parameters:
        -----------
        annotation_tuple : tuple
            (file, exp_smiles, final_df) from ablate_database.py
            
        Returns:
        --------
        dict : Parsed annotation data
        """
        file, exp_smiles, final_df = annotation_tuple
        
        # Store annotation data
        self.annotation_data = annotation_tuple
        self.query_info = {
            'query_id': file,
            'query_smiles': exp_smiles,
            'n_results': len(final_df)
        }
        self.lookup_results = final_df.copy()
        
        # Sort by hybrid score (descending - higher is better) if it exists
        if len(self.lookup_results) > 0 and 'hybrid_score' in self.lookup_results.columns:
            self.lookup_results = self.lookup_results.sort_values('hybrid_score', ascending=False)
        elif len(self.lookup_results) > 0:
            # Try to create hybrid_score if we have the required columns
            if 'mcs_score_d' in self.lookup_results.columns and 'tanimoto_similarity' in self.lookup_results.columns:
                self.lookup_results['hybrid_score'] = (self.lookup_results['mcs_score_d'] + self.lookup_results['tanimoto_similarity']) / 2
                self.lookup_results = self.lookup_results.sort_values('hybrid_score', ascending=False)
            else:
                # Just keep original order if we can't create hybrid_score
                pass
        
        # Helper function to extract numeric values from potentially tuple entries
        def extract_numeric_value(value):
            """Extract numeric value from potential tuple (distance, matched_fraction)"""
            if isinstance(value, tuple):
                return value[0]  # Return the first element (distance)
            return value
        
        # Clean hungarian_distance column if it contains tuples
        if 'hungarian_distance' in final_df.columns:
            hungarian_distances = final_df['hungarian_distance'].apply(extract_numeric_value)
        else:
            hungarian_distances = pd.Series([0] * len(final_df))  # Default values if column missing
        
        # Prepare statistics with hybrid_score if available
        top_result = None
        mean_hybrid = 0
        max_hybrid = 0
        
        if len(final_df) > 0 and 'hybrid_score' in self.lookup_results.columns:
            top_result = self.lookup_results.loc[self.lookup_results['hybrid_score'].idxmax()]
            mean_hybrid = self.lookup_results['hybrid_score'].mean()
            max_hybrid = self.lookup_results['hybrid_score'].max()
        
        parsed_data = {
            'query_id': file,
            'query_smiles': exp_smiles,
            'lookup_results': final_df,
            'n_results': len(final_df),
            'top_result': top_result,
            'score_stats': {
                'mean_hybrid': mean_hybrid,
                'max_hybrid': max_hybrid,
                'mean_hungarian': hungarian_distances.mean() if len(final_df) > 0 else 0,
                'min_hungarian': hungarian_distances.min() if len(final_df) > 0 else 0
            }
        }
        
        return parsed_data
    
    def find_top_k_in_graph(self, k: int = 10, similarity_metric: str = 'hybrid_score') -> pd.DataFrame:
        """
        Find the top-k lookup results that exist as nodes in the graph.
        
        Parameters:
        -----------
        k : int, default=10
            Number of top results to return
        similarity_metric : str, default='hybrid_score'
            Column to rank by ('hybrid_score', 'mcs_score_d', 'tanimoto_similarity')
            
        Returns:
        --------
        pd.DataFrame : Top-k results that exist in the graph
        """
        if self.lookup_results is None:
            raise ValueError("No annotation data loaded. Call parse_annotation() first.")
        
        if self.graph is None:
            raise ValueError("No graph loaded. Use load_graph() or set_graph() first.")
        
        # Filter results to only include nodes that exist in the graph
        graph_nodes = set(self.graph.nodes())
        in_graph_mask = self.lookup_results['lookup_key'].isin(graph_nodes)
        in_graph_results = self.lookup_results[in_graph_mask].copy()
        
        # Sort by similarity metric (descending for similarity scores, ascending for distances)
        ascending = similarity_metric == 'hungarian_distance'
        top_k_results = in_graph_results.sort_values(similarity_metric, ascending=ascending).head(k)
        
        if self.verbose:
            print(f"Found {len(in_graph_results)} results in graph out of {len(self.lookup_results)} total results")
            print(f"Returning top-{k} by {similarity_metric}")
        
        return top_k_results
    
    def analyze_neighbors(self, node_ids: List[str], max_neighbors: int = 20) -> Dict[str, Any]:
        """
        Analyze the neighbors of given nodes in the graph.
        
        Parameters:
        -----------
        node_ids : list of str
            Node IDs to analyze neighbors for
        max_neighbors : int, default=20
            Maximum number of neighbors to return per node
            
        Returns:
        --------
        dict : Neighbor analysis results
        """
        if self.graph is None:
            raise ValueError("No graph loaded.")
        
        neighbor_analysis = {}
        all_neighbors = set()
        
        for node_id in node_ids:
            if node_id not in self.graph.nodes():
                print(f"Warning: Node {node_id} not found in graph")
                continue
                
            # Get neighbors and their edge weights
            neighbors = []
            for neighbor in self.graph.neighbors(node_id):
                edge_data = self.graph.get_edge_data(node_id, neighbor)
                weight = edge_data.get('weight', 1.0) if edge_data else 1.0
                neighbors.append((neighbor, weight))
            
            # Sort by edge weight (ascending - lower weights = higher similarity)
            neighbors.sort(key=lambda x: x[1])
            top_neighbors = neighbors[:max_neighbors]
            
            neighbor_analysis[node_id] = {
                'total_neighbors': len(neighbors),
                'top_neighbors': top_neighbors,
                'neighbor_weights': [w for _, w in top_neighbors],
                'mean_weight': np.mean([w for _, w in neighbors]) if neighbors else 0,
                'min_weight': min([w for _, w in neighbors]) if neighbors else 0
            }
            
            # Collect all unique neighbors
            all_neighbors.update([n for n, _ in top_neighbors])
        
        return {
            'node_analysis': neighbor_analysis,
            'unique_neighbors': list(all_neighbors),
            'total_unique_neighbors': len(all_neighbors)
        }
    
    def add_query_to_graph(self, query_hsqc_data: np.ndarray,
                          lookup_hsqc_data: Dict[str, np.ndarray] = None,
                          similarity_threshold: float = 50.0,
                          query_id: Optional[str] = None,
                          query_smiles: Optional[str] = None) -> nx.Graph:
        """
        Add the query compound as a new node to the graph with edges based on Hungarian distances.
        Calculates distances using the SAME distance function that was used to create the graph.
        
        Parameters:
        -----------
        query_hsqc_data : np.ndarray
            HSQC peak data for the query compound, shape (n_peaks, 2)
        lookup_hsqc_data : dict, optional
            Mapping of {node_id: hsqc_array} for calculating distances.
            If None, will try to extract from annotation results.
        similarity_threshold : float, default=50.0
            Maximum Hungarian distance to create an edge (lower = more similar)
        query_id : str, optional
            ID for the query node. If None, will use self.query_info['query_id']
        query_smiles : str, optional
            SMILES string for the query compound. If None, will use self.query_info['query_smiles']
            
        Returns:
        --------
        nx.Graph : Updated graph with query node added
        """
        if self.graph is None:
            raise ValueError("No graph loaded.")
        
        # Determine query_id and query_smiles
        if query_id is None:
            if self.query_info is None:
                raise ValueError("No query_id provided and no query info available. Either provide query_id parameter or call parse_annotation() first.")
            query_id = self.query_info['query_id']
            
        if query_smiles is None:
            if self.query_info is not None:
                query_smiles = self.query_info['query_smiles']
            else:
                query_smiles = "Unknown"  # Default fallback
        
        # Create a copy of the graph to avoid modifying the original
        updated_graph = self.graph.copy()
        
        # Check what similarity attributes exist on other nodes
        similarity_attributes = set()
        similarity_metrics = ['hybrid_score', 'mcs_score', 'tanimoto_similarity', 'mcs_score_d']
        
        # Sample a few nodes to see what attributes they have
        sample_nodes = list(updated_graph.nodes())[:10]  # Check first 10 nodes
        for node in sample_nodes:
            node_data = updated_graph.nodes[node]
            for metric in similarity_metrics:
                if metric in node_data:
                    similarity_attributes.add(metric)
        
        # Prepare node attributes for query
        query_node_attrs = {
            'smiles': query_smiles,
            'node_type': 'query'
        }
        
        # Add similarity metric attributes with value 1.0 if they exist on other nodes
        for attr in similarity_attributes:
            query_node_attrs[attr] = 1.0
            if self.verbose:
                print(f"Adding {attr}=1.0 to query node (attribute found on other nodes)")
        
        # Add query node with all attributes
        updated_graph.add_node(query_id, **query_node_attrs)
        
        # Calculate Hungarian distances using the same function as the graph
        if self.verbose:
            print(f"Calculating distances using {self.distance_function.__name__}...")
        
        edges_added = 0
        self.last_calculated_distances = {}  # Reset and store new distances
        
        # If lookup_hsqc_data is provided, use it; otherwise try to extract from annotation results
        if lookup_hsqc_data is None:
            if self.verbose:
                print("Warning: No HSQC lookup data provided. Cannot add query to graph.")
                print("Please provide lookup_hsqc_data parameter with {node_id: hsqc_array} mapping.")
            return updated_graph
        
        # Calculate distances for each node in the graph
        for node_id in updated_graph.nodes():
            # Skip the query node itself
            if node_id == query_id:
                continue
                
            if node_id not in lookup_hsqc_data:
                if self.verbose:
                    print(f"⚠️  Node {node_id} not in lookup data, skipping")
                continue
            
            hsqc_array = lookup_hsqc_data[node_id]
                
            try:
                # Calculate distance using the same function as the graph
                if len(self.distance_params) == 0:
                    # Simple functions with no extra parameters
                    distance = self.distance_function(query_hsqc_data, hsqc_array)
                else:
                    # Functions with parameters (uncertainty-based or modified Hungarian)
                    distance = self.distance_function(query_hsqc_data, hsqc_array, **self.distance_params)
                    # All wrapper functions now return only the distance value directly
                
                # Ensure distance is a single value (extract from tuple if needed)
                if isinstance(distance, tuple):
                    distance = distance[0]  # Extract distance value from (distance, matched_fraction) tuple
                
                # Store the calculated distance for debugging purposes
                self.last_calculated_distances[node_id] = distance
                
                # Add edge if within threshold
                if distance <= similarity_threshold:
                    # Prepare edge attributes with Hungarian distance
                    edge_attrs = {
                        'weight': distance,
                        'hungarian_distance': distance,  # Explicitly set Hungarian distance
                        'edge_type': 'hungarian',
                        'distance_function': self.distance_column
                    }
                    
                    # Set similarity attributes to 1.0 for query edges (perfect similarity with query)
                    edge_attrs['hybrid_similarity'] = 1.0
                    edge_attrs['tanimoto_similarity'] = 1.0
                    edge_attrs['mcs_similarity'] = 1.0
                    
                    # if self.verbose:
                    #     print(f"      Added edge {query_id}-{node_id}: hungarian_dist={distance:.4f}, "
                    #           f"hybrid=1.0000, tanimoto=1.0000, mcs=1.0000")
                    
                    # Add the edge with all attributes
                    updated_graph.add_edge(query_id, node_id, **edge_attrs)
                    edges_added += 1
                else:
                    continue
            except Exception as e:
                print(f"Error calculating distance for node {node_id}: {e}")
                continue
        
        if self.verbose:
            print(f"Added query node '{query_id}' with {edges_added} edges (threshold: {similarity_threshold})")
            print(f"Used distance function: {self.distance_function.__name__}")
        
        return updated_graph
    
    def summarize_analysis(self) -> Dict[str, Any]:
        """
        Provide a comprehensive summary of the annotation analysis.
        
        Returns:
        --------
        dict : Analysis summary
        """
        if self.query_info is None:
            return {"error": "No annotation data loaded"}
        
        summary = {
            'query_info': self.query_info,
            'graph_info': {
                'total_nodes': self.graph.number_of_nodes() if self.graph else 0,
                'total_edges': self.graph.number_of_edges() if self.graph else 0
            }
        }
        
        if self.lookup_results is not None:
            # Find how many results are in the graph
            if self.graph:
                graph_nodes = set(self.graph.nodes())
                in_graph_count = sum(1 for key in self.lookup_results['lookup_key'] if key in graph_nodes)
            else:
                in_graph_count = 0
            
            summary['results_info'] = {
                'total_results': len(self.lookup_results),
                'results_in_graph': in_graph_count,
                'coverage_percentage': (in_graph_count / len(self.lookup_results)) * 100 if len(self.lookup_results) > 0 else 0
            }
            
            # Top results summary
            top_5 = self.lookup_results.head(5)
            summary['top_5_results'] = [
                {
                    'lookup_key': row['lookup_key'],
                    'hybrid_score': row['hybrid_score'],
                    'hungarian_distance': row['hungarian_distance'],
                    'mcs_score': row['mcs_score_d'],
                    'tanimoto_score': row['tanimoto_similarity']
                }
                for _, row in top_5.iterrows()
            ]
        
        return summary
    
    def rerank_annotations(self, ranking_values: List[float], ranking_name: str = "custom", 
                          ascending: bool = False, compare_fraction_metrics: bool = True) -> Dict[str, Any]:
        """
        Rerank annotation results using custom ranking values and compare performance.
        
        Parameters:
        -----------
        ranking_values : List[float]
            List of ranking values (same length as annotation results)
            Higher values = better rank (unless ascending=True)
        ranking_name : str, default="custom"
            Name for the ranking method (for display purposes)
        ascending : bool, default=False
            If True, lower values get better ranks (like distances)
            If False, higher values get better ranks (like similarities)
        compare_fraction_metrics : bool, default=True
            Whether to calculate and compare fraction metrics before/after
            
        Returns:
        --------
        dict : Reranking results with before/after comparisons
        """
        if self.lookup_results is None:
            raise ValueError("No annotation data loaded. Call parse_annotation() first.")
        
        if len(ranking_values) != len(self.lookup_results):
            raise ValueError(f"Ranking values length ({len(ranking_values)}) must match "
                           f"annotation results length ({len(self.lookup_results)})")
        
        print(f"🔄 Reranking annotations using {ranking_name}...")
        
        # Store original results and k values
        original_results = self.lookup_results.copy()
        original_k_values = original_results['k'].tolist()
        
        # Calculate average metrics before reranking
        before_average_metrics = {}
        if compare_fraction_metrics:
            print(f"📊 Calculating average metrics BEFORE reranking...")
            for n in [1, 3, 5, 10]:
                # Average hybrid metrics
                avg_metrics = self.calculate_average_hybrid_top_n(top_n=n, rank_by='k')
                before_average_metrics[f'top_{n}'] = avg_metrics
                print(f"   - Top-{n} avg hybrid: {avg_metrics['average_hybrid_top_n']:.3f}")
        
        # Add ranking values to dataframe and rerank
        reranked_results = original_results.copy()
        reranked_results[ranking_name] = ranking_values
        
        # Sort by the new ranking values
        reranked_results = reranked_results.sort_values(ranking_name, ascending=ascending)
        
        # Update k values based on new ranking (1 = best rank)
        reranked_results['k_original'] = reranked_results['k']  # Store original k
        reranked_results['k'] = range(1, len(reranked_results) + 1)
        
        # Update the stored lookup results
        old_lookup_results = self.lookup_results
        self.lookup_results = reranked_results
        
        # Calculate average metrics after reranking
        after_average_metrics = {}
        if compare_fraction_metrics:
            print(f"📊 Calculating average metrics AFTER reranking...")
            for n in [1, 3, 5, 10]:
                # Average hybrid metrics
                avg_metrics = self.calculate_average_hybrid_top_n(top_n=n, rank_by='k')
                after_average_metrics[f'top_{n}'] = avg_metrics
                print(f"   - Top-{n} avg hybrid: {avg_metrics['average_hybrid_top_n']:.3f}")
        
        # Calculate average improvements
        average_improvements = {}
        if compare_fraction_metrics:
            print(f"\n📈 Reranking Performance Changes:")
            for n in [1, 3, 5, 10]:
                key = f'top_{n}'
                
                # Average hybrid improvements
                before_avg = before_average_metrics[key]['average_hybrid_top_n']
                after_avg = after_average_metrics[key]['average_hybrid_top_n']
                avg_improvement = after_avg - before_avg
                avg_improvement_pct = (avg_improvement / before_avg * 100) if before_avg > 0 else 0
                
                average_improvements[key] = {
                    'before': before_avg,
                    'after': after_avg,
                    'absolute_change': avg_improvement,
                    'percent_change': avg_improvement_pct
                }
                
                avg_status = "📈" if avg_improvement > 0 else "📉" if avg_improvement < 0 else "➡️"
                print(f"   {avg_status} Top-{n} avg hybrid: {before_avg:.3f} → {after_avg:.3f} "
                      f"(Δ: {avg_improvement:+.3f}, {avg_improvement_pct:+.1f}%)")
        
        # Show top-k changes
        print(f"\n🔄 Top-5 Ranking Changes:")
        print(f"{'Rank':<4} {'Lookup Key':<15} {'Hybrid':<8} {'Original k':<10} {ranking_name:<12}")
        print("-" * 65)
        
        for i, (_, row) in enumerate(reranked_results.head(5).iterrows(), 1):
            lookup_key = row['lookup_key'][:14]
            hybrid = f"{row['hybrid_score']:.4f}"
            orig_k = row['k_original']
            ranking_val = f"{row[ranking_name]:.4f}"
            print(f"{i:<4} {lookup_key:<15} {hybrid:<8} {orig_k:<10} {ranking_val:<12}")
        
        # Restore original lookup results
        self.lookup_results = old_lookup_results
        
        # Compile results
        rerank_results = {
            'ranking_method': ranking_name,
            'ranking_ascending': ascending,
            'original_results': original_results,
            'reranked_results': reranked_results,
            'before_average_metrics': before_average_metrics,
            'after_average_metrics': after_average_metrics,
            'average_improvements': average_improvements,
            'summary': {
                'total_results': len(reranked_results),
                'ranking_values_range': (min(ranking_values), max(ranking_values)),
                'best_avg_improvement': max([imp['absolute_change'] for imp in average_improvements.values()]) if average_improvements else 0,
                'overall_avg_improved': any([imp['absolute_change'] > 0 for imp in average_improvements.values()]) if average_improvements else False
            }
        }
        
        print(f"\n✅ Reranking complete!")
        if average_improvements:
            best_avg_improvement = max([imp['absolute_change'] for imp in average_improvements.values()])
            print(f"   Best average improvement: {best_avg_improvement:+.3f} hybrid score points")
        
        return rerank_results

    def rerank_by_network_metric(self, updated_graph: nx.Graph, metric: str = 'jaccard',
                                query_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Convenience function to rerank annotations using network topology metrics.
        
        Parameters:
        -----------
        updated_graph : nx.Graph
            Graph with query node added (from analyze_annotation_with_graph_integration)
        metric : str, default='jaccard'
            Network metric to use. Available options:
            'jaccard', 'adamic_adar', 'weighted_jaccard', 'cn', 'cos', 'si', 'hpi', 'hdi', 'lhn', 'pa', 'ra', 'sp',
            'ra_weighted_sum', 'ra_weighted_avg', 'ra_weighted_min', 'ra_weighted_max', 'ra_weighted_product'
        query_id : str, optional
            Query node ID (will use self.query_info if not provided)
            
        Returns:
        --------
        dict : Reranking results
        """
        if query_id is None:
            if self.query_info is None:
                raise ValueError("No query info available. Provide query_id or call parse_annotation() first.")
            query_id = self.query_info['query_id']
        
        if query_id not in updated_graph.nodes():
            raise ValueError(f"Query node '{query_id}' not found in graph")
        
        print(f"🌐 Calculating {metric} values for reranking...")
        
        # Calculate network metric values for each result
        ranking_values = []
        for _, row in self.lookup_results.iterrows():
            lookup_key = row['lookup_key']
            
            if lookup_key in updated_graph.nodes():
                value = self._calculate_network_metric(updated_graph, query_id, lookup_key, metric)
            else:
                # Node not in graph - give lowest possible value
                # For shortest path (distance metric), use inf; for others use 0
                if metric == 'sp':
                    value = float('inf')
                else:
                    value = 0.0
            
            ranking_values.append(value)
        
        print(f"   ✅ Calculated {metric} values (range: {min(ranking_values):.4f} - {max(ranking_values):.4f})")
        
        # Determine ranking order based on metric type
        # Shortest path is a distance metric (lower is better), others are similarity metrics (higher is better)
        ascending = (metric == 'sp')
        
        # Rerank using the calculated values
        return self.rerank_annotations(
            ranking_values=ranking_values,
            ranking_name=metric,
            ascending=ascending
        )

    def rerank_by_combined_metric(self, updated_graph: nx.Graph, metric: str = 'jaccard',
                                 query_id: Optional[str] = None, 
                                 network_weight: float = 0.5,
                                 distance_weight: float = 0.5,
                                 hungarian_similarity_method: str = 'inverse') -> Dict[str, Any]:
        """
        Rerank annotations using a combination of network topology metrics and Hungarian distance.
        
        This method combines network similarity (e.g., Jaccard) with Hungarian distance similarity
        to create a more comprehensive ranking that considers both topological and spectral similarity.
        
        Parameters:
        -----------
        updated_graph : nx.Graph
            Graph with query node added (from analyze_annotation_with_graph_integration)
        metric : str, default='jaccard'
            Network metric to use. Available options:
            'jaccard', 'adamic_adar', 'weighted_jaccard', 'cn', 'cos', 'si', 'hpi', 'hdi', 'lhn', 'pa', 'ra', 'sp',
            'ra_weighted_sum', 'ra_weighted_avg', 'ra_weighted_min', 'ra_weighted_max', 'ra_weighted_product'
        query_id : str, optional
            Query node ID (will use self.query_info if not provided)
        network_weight : float, default=0.5
            Weight for the network metric (0.0 to 1.0)
        distance_weight : float, default=0.5
            Weight for the Hungarian distance similarity (0.0 to 1.0)
        hungarian_similarity_method : str, default='inverse'
            Method to convert Hungarian distance to similarity:
            - 'inverse': 1/(1+distance)
            - 'exp': exp(-distance)
            - 'normalized_inverse': 1 - (distance / max_distance)
            
        Returns:
        --------
        dict : Reranking results with combined metric
        """
        if query_id is None:
            if self.query_info is None:
                raise ValueError("No query info available. Provide query_id or call parse_annotation() first.")
            query_id = self.query_info['query_id']
        
        if query_id not in updated_graph.nodes():
            raise ValueError(f"Query node '{query_id}' not found in graph")
        
        # Validate weights
        if not (0.0 <= network_weight <= 1.0) or not (0.0 <= distance_weight <= 1.0):
            raise ValueError("Weights must be between 0.0 and 1.0")
        
        # Normalize weights to sum to 1
        total_weight = network_weight + distance_weight
        if total_weight == 0:
            raise ValueError("At least one weight must be non-zero")
        network_weight = network_weight / total_weight
        distance_weight = distance_weight / total_weight
        
        print(f"🌐 Calculating combined metric: {metric} + Hungarian distance")
        print(f"   Network weight: {network_weight:.2f}, Distance weight: {distance_weight:.2f}")
        
        # Step 1: Calculate network metric values
        network_values = []
        for _, row in self.lookup_results.iterrows():
            lookup_key = row['lookup_key']
            
            if lookup_key in updated_graph.nodes():
                value = self._calculate_network_metric(updated_graph, query_id, lookup_key, metric)
                # For shortest path (distance metric), convert to similarity
                if metric == 'sp':
                    value = 1.0 / (1.0 + value) if value != float('inf') else 0.0
            else:
                # Node not in graph - give lowest possible value
                value = 0.0
            
            network_values.append(value)
        
        # Step 2: Get Hungarian distances and convert to similarities
        hungarian_distances = self.lookup_results['hungarian_distance'].values
        hungarian_similarities = []
        
        if hungarian_similarity_method == 'inverse':
            # Convert distance to similarity: 1/(1+distance)
            hungarian_similarities = [1.0 / (1.0 + dist) for dist in hungarian_distances]
        elif hungarian_similarity_method == 'exp':
            # Convert distance to similarity: exp(-distance)
            hungarian_similarities = [np.exp(-dist) for dist in hungarian_distances]
        elif hungarian_similarity_method == 'normalized_inverse':
            # Convert distance to similarity: 1 - (distance / max_distance)
            max_distance = max(hungarian_distances) if len(hungarian_distances) > 0 else 1.0
            hungarian_similarities = [1.0 - (dist / max_distance) for dist in hungarian_distances]
        else:
            raise ValueError(f"Unknown Hungarian similarity method: {hungarian_similarity_method}")
        
        # Step 3: Normalize network values to [0,1] if needed
        if len(network_values) > 0:
            if metric == 'adamic_adar':
                # Adamic-Adar can have values > 1, so normalize
                max_network = max(network_values) if max(network_values) > 0 else 1.0
                network_values = [v / max_network for v in network_values]
            # Jaccard and weighted_jaccard are already in [0,1] range
        
        # Step 4: Combine the metrics
        combined_values = []
        for i in range(len(network_values)):
            combined_score = (network_weight * network_values[i] + 
                            distance_weight * hungarian_similarities[i])
            combined_values.append(combined_score)
        
        # Step 5: Print diagnostics
        print(f"   ✅ Network metric ({metric}) range: {min(network_values):.4f} - {max(network_values):.4f}")
        print(f"   ✅ Hungarian similarities range: {min(hungarian_similarities):.4f} - {max(hungarian_similarities):.4f}")
        print(f"   ✅ Combined metric range: {min(combined_values):.4f} - {max(combined_values):.4f}")
        
        # Step 6: Rerank using the combined values
        combined_metric_name = f"{metric}_hungarian_combined"
        return self.rerank_annotations(
            ranking_values=combined_values,
            ranking_name=combined_metric_name,
            ascending=False  # Higher combined values = better ranks
        )

    def calculate_fraction_best_hybrid_realized(self, top_n: int = 3, rank_by: str = 'k') -> Dict[str, float]:
        """
        Calculate the fraction of best hybrid realized metric.
        
        This metric computes: (max hybrid value in top-N candidates) / (max hybrid value in entire set)
        
        Parameters:
        -----------
        top_n : int, default=3
            Number of top candidates to consider
        rank_by : str, default='k'
            How to rank candidates ('k' for annotation ranking, 'hybrid_score' for hybrid similarity)
            
        Returns:
        --------
        dict : Dictionary containing the fraction metrics and supporting data
        """
        if self.lookup_results is None:
            raise ValueError("No annotation data loaded. Call parse_annotation() first.")
        
        # Get the maximum hybrid score in the entire dataset
        max_hybrid_all = self.lookup_results['hybrid_score'].max()
        
        # Get top-N candidates based on ranking method
        if rank_by == 'k':
            # Sort by k (annotation ranking) - lower k values are better
            top_candidates = self.lookup_results.sort_values('k').head(top_n)
        elif rank_by == 'hybrid_score':
            # Sort by hybrid score - higher values are better
            top_candidates = self.lookup_results.sort_values('hybrid_score', ascending=False).head(top_n)
        else:
            raise ValueError(f"Unknown ranking method: {rank_by}. Use 'k' or 'hybrid_score'")
        
        # Get the maximum hybrid score in the top-N candidates
        max_hybrid_top_n = top_candidates['hybrid_score'].max()
        
        # Calculate the fraction
        fraction_realized = max_hybrid_top_n / max_hybrid_all if max_hybrid_all > 0 else 0.0
        
        return {
            'fraction_best_hybrid_realized': fraction_realized,
            'max_hybrid_all': max_hybrid_all,
            'max_hybrid_top_n': max_hybrid_top_n,
            'top_n_used': top_n,
            'ranking_method': rank_by,
            'top_n_candidates': top_candidates[['lookup_key', 'hybrid_score', 'k']].to_dict('records')
        }

    def calculate_average_hybrid_top_n(self, top_n: int = 3, rank_by: str = 'k') -> Dict[str, float]:
        """
        Calculate the average hybrid score of top-N candidates.
        
        This metric computes the mean hybrid score of the top-N candidates and compares it
        to the overall mean hybrid score to understand ranking quality.
        
        Parameters:
        -----------
        top_n : int, default=3
            Number of top candidates to consider
        rank_by : str, default='k'
            How to rank candidates ('k' for annotation ranking, 'hybrid_score' for hybrid similarity)
            
        Returns:
        --------
        dict : Dictionary containing the average hybrid metrics and supporting data
        """
        if self.lookup_results is None:
            raise ValueError("No annotation data loaded. Call parse_annotation() first.")
        
        # Get the average hybrid score in the entire dataset
        mean_hybrid_all = self.lookup_results['hybrid_score'].mean()
        max_hybrid_all = self.lookup_results['hybrid_score'].max()
        
        # Get top-N candidates based on ranking method
        if rank_by == 'k':
            # Sort by k (annotation ranking) - lower k values are better
            top_candidates = self.lookup_results.sort_values('k').head(top_n)
        elif rank_by == 'hybrid_score':
            # Sort by hybrid score - higher values are better
            top_candidates = self.lookup_results.sort_values('hybrid_score', ascending=False).head(top_n)
        else:
            raise ValueError(f"Unknown ranking method: {rank_by}. Use 'k' or 'hybrid_score'")
        
        # Get the average hybrid score in the top-N candidates
        mean_hybrid_top_n = top_candidates['hybrid_score'].mean()
        
        # Calculate metrics
        average_improvement = mean_hybrid_top_n - mean_hybrid_all
        relative_to_max = mean_hybrid_top_n / max_hybrid_all if max_hybrid_all > 0 else 0.0
        
        return {
            'average_hybrid_top_n': mean_hybrid_top_n,
            'average_hybrid_all': mean_hybrid_all,
            'max_hybrid_all': max_hybrid_all,
            'average_improvement': average_improvement,
            'relative_to_max': relative_to_max,
            'top_n_used': top_n,
            'ranking_method': rank_by,
            'top_n_candidates': top_candidates[['lookup_key', 'hybrid_score', 'k']].to_dict('records')
        }

    def rerank_annotations_by_metric(self, 
                                   annotations: List[Tuple[str, str, pd.DataFrame]], 
                                   experimental_data: Dict[str, Any],
                                   lookup_hsqc_data: Dict[str, np.ndarray],
                                   metric: str = 'jaccard',
                                   secondary_metric: Optional[str] = None,
                                   primary_weight: float = 0.5,
                                   secondary_weight: float = 0.5,
                                   similarity_threshold: float = 50.0,
                                   max_neighbors: int = 20,
                                   min_connections: int = 3,
                                   console: Console = Console(),
                                   use_leiden: bool = False) -> List[Tuple[str, str, pd.DataFrame]]:
        """
        Rerank a list of annotations by a single metric or composite of two metrics.
        
        Parameters:
        -----------
        annotations : List[Tuple[str, str, pd.DataFrame]]
            List of annotation tuples in format: (file_name, exp_smiles, final_df)
        experimental_data : Dict[str, Any]
            Dictionary mapping file names to experimental data (query HSQC data)
        lookup_hsqc_data : Dict[str, np.ndarray]
            Dictionary mapping SMILES to HSQC data
        metric : str, default='jaccard'
            Primary metric to use. Available options:
            Network: 'jaccard', 'adamic_adar', 'weighted_jaccard', 'cn', 'cos', 'si', 'hpi', 'hdi', 'lhn', 'pa', 'ra', 'sp',
                     'ra_weighted_sum', 'ra_weighted_avg', 'ra_weighted_min', 'ra_weighted_max', 'ra_weighted_product'
            Annotation: 'hungarian_nn', 'hybrid_score', 'tanimoto_similarity', 'mcs_score'
        secondary_metric : str, optional
            Secondary metric for composite ranking. If provided, creates composite ranking.
        primary_weight : float, default=0.5
            Weight for primary metric in composite ranking (0.0 to 1.0)
        secondary_weight : float, default=0.5
            Weight for secondary metric in composite ranking (0.0 to 1.0)
        similarity_threshold : float, default=50.0
            Similarity threshold for adding query to graph
        max_neighbors : int, default=20
            Maximum number of neighbors to consider for query
        min_connections : int, default=3
            Minimum number of graph connections required for query (annotations with fewer connections are skipped)
        use_leiden : bool, default=False
            Whether to account for Leiden clustering in the graph
            
        Returns:
        --------
        List[Tuple[str, str, pd.DataFrame]] : List of modified annotation tuples with new ranking columns
        """
        is_composite = secondary_metric is not None
        composite_name = f"{metric}_{secondary_metric}_composite" if is_composite else metric
        

        header_text = Text()
        if is_composite:
            header_text.append("🔄 Reranking ", style="bold blue")
            header_text.append(f"{len(annotations)}", style="bold yellow")
            header_text.append(" annotations using COMPOSITE metric: ", style="bold blue")
            header_text.append(f"{metric} + {secondary_metric}", style="bold green")
        else:
            header_text.append("🔄 Reranking ", style="bold blue")
            header_text.append(f"{len(annotations)}", style="bold yellow")
            header_text.append(" annotations using ", style="bold blue")
            header_text.append(f"{metric}", style="bold green")
            header_text.append(" metric", style="bold blue")
        
        header_panel = Panel(header_text, title="[bold cyan]Annotation Reranking[/bold cyan]", border_style="blue")
        console.print(header_panel)
        
        # Create configuration table
        config_table = Table(title="[bold cyan]Reranking Configuration[/bold cyan]", show_header=True, header_style="bold magenta")
        config_table.add_column("Parameter", style="cyan", no_wrap=True)
        config_table.add_column("Value", style="green")
        
        if is_composite:
            config_table.add_row("Primary Metric", metric)
            config_table.add_row("Secondary Metric", secondary_metric)
            config_table.add_row("Primary Weight", f"{primary_weight:.2f}")
            config_table.add_row("Secondary Weight", f"{secondary_weight:.2f}")
        else:
            config_table.add_row("Metric", metric)
        
        config_table.add_row("Similarity Threshold", str(similarity_threshold))
        config_table.add_row("Max Neighbors", str(max_neighbors))
        config_table.add_row("Min Connections", str(min_connections))
        config_table.add_row("Total Annotations", str(len(annotations)))
        
        console.print(config_table)
        console.print()
        
        reranked_annotations = []
        connection_filtered_count = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green", finished_style="bold green"),
            TaskProgressColumn(),
            console=console,
            transient=False
        ) as progress:
            task = progress.add_task("[cyan]Reranking annotations[/cyan]", total=len(annotations))
            
            for i, (file_name, exp_smiles, final_df) in enumerate(annotations):
                progress.update(task, description=f"[yellow]Processing[/yellow] {file_name} ({i+1}/{len(annotations)})")
                
                # Get query HSQC data
                if file_name not in experimental_data:
                    if self.verbose:
                        console.print(f"[red]⚠️  No experimental data for {file_name}, skipping[/red]")
                    # Return original annotation unchanged
                    reranked_annotations.append((file_name, exp_smiles, final_df))
                    progress.advance(task)
                    continue
                
                exp_vals = experimental_data[file_name]
                query_hsqc_data = exp_vals[3][:, ::-1]  # Swap columns to match expected format
                query_smiles = exp_vals[0]
                
                if self.verbose:
                    # Create query info panel
                    query_info = Table(title=f"[bold cyan]Query Info: {file_name}[/bold cyan]", show_header=False, box=None)
                    query_info.add_column("Property", style="cyan", no_wrap=True)
                    query_info.add_column("Value", style="green")
                    query_info.add_row("🧪 SMILES", query_smiles[:50] + "..." if len(query_smiles) > 50 else query_smiles)
                    query_info.add_row("🎯 HSQC Peaks", str(query_hsqc_data.shape[0]))
                    query_info.add_row("📊 Annotation Results", str(len(final_df)))
                    console.print(query_info)
                
                # Parse annotation (temporarily set internal state)
                self.parse_annotation((file_name, exp_smiles, final_df))
                
                # Add query to graph
                updated_graph = self.add_query_to_graph(
                    query_hsqc_data=query_hsqc_data,
                    lookup_hsqc_data=lookup_hsqc_data,
                    similarity_threshold=similarity_threshold
                )
                
                # Get query connections count
                query_connections = len(list(updated_graph.neighbors(file_name)))
                if self.verbose:
                    connection_status = "[green]✅ Sufficient[/green]" if query_connections >= min_connections else "[red]❌ Insufficient[/red]"
                    console.print(f"🌐 Query connections: [bold]{query_connections}[/bold] {connection_status}")
                
                # Filter out annotations with insufficient connections
                if query_connections < min_connections:
                    if self.verbose:
                        console.print(f"[red]❌ Insufficient connections ({query_connections} < {min_connections}), skipping annotation[/red]")
                    connection_filtered_count += 1
                    progress.advance(task)
                    continue
                
                # Calculate metric values for each result
                if is_composite:
                    # Composite ranking
                    if self.verbose:
                        console.print(f"[blue]🔍 Calculating composite metric: {metric} + {secondary_metric}[/blue]")
                    
                    # Normalize weights
                    total_weight = primary_weight + secondary_weight
                    if total_weight == 0:
                        total_weight = 1.0
                        primary_weight = secondary_weight = 0.5
                    else:
                        primary_weight = primary_weight / total_weight
                        secondary_weight = secondary_weight / total_weight
                    
                    primary_values = []
                    secondary_values = []
                    
                    for _, row in final_df.iterrows():
                        lookup_key = row['lookup_key']
                        
                        # Calculate primary metric
                        if lookup_key in updated_graph.nodes():
                            primary_val = self._calculate_metric(
                                updated_graph, file_name, lookup_key, metric, final_df
                            )
                            # Handle distance metrics
                            if metric == 'sp':
                                primary_val = 1.0 / (1.0 + primary_val) if primary_val != float('inf') else 0.0
                        else:
                            primary_val = 0.0
                        
                        # Calculate secondary metric
                        if secondary_metric in ['hungarian_nn', 'hybrid_score', 'tanimoto_similarity', 'mcs_score']:
                            # Annotation-based metric
                            secondary_val = self._calculate_metric(
                                updated_graph, file_name, lookup_key, secondary_metric, final_df
                            )
                        else:
                            # Network-based metric
                            if lookup_key in updated_graph.nodes():
                                secondary_val = self._calculate_metric(
                                    updated_graph, file_name, lookup_key, secondary_metric, final_df
                                )
                                # Handle distance metrics
                                if secondary_metric == 'sp':
                                    secondary_val = 1.0 / (1.0 + secondary_val) if secondary_val != float('inf') else 0.0
                            else:
                                secondary_val = 0.0
                        
                        primary_values.append(primary_val)
                        secondary_values.append(secondary_val)
                    
                    # Normalize and combine values
                    def normalize_values(values):
                        finite_values = [v for v in values if not (math.isinf(v) or math.isnan(v))]
                        if len(finite_values) == 0:
                            return [0.0] * len(values)
                        
                        min_val = min(finite_values)
                        max_val = max(finite_values)
                        
                        if max_val == min_val:
                            return [1.0 if not (math.isinf(v) or math.isnan(v)) else 0.0 for v in values]
                        
                        normalized = []
                        for v in values:
                            if math.isinf(v) or math.isnan(v):
                                normalized.append(0.0)
                            else:
                                normalized.append((v - min_val) / (max_val - min_val))
                        return normalized
                    
                    normalized_primary = normalize_values(primary_values)
                    normalized_secondary = normalize_values(secondary_values)
                    
                    metric_values = []
                    for j in range(len(normalized_primary)):
                        composite_score = (primary_weight * normalized_primary[j] + 
                                            secondary_weight * normalized_secondary[j])
                        metric_values.append(composite_score)
                    
                    if self.verbose:
                        # Create metric ranges table
                        ranges_table = Table(title="[bold cyan]Metric Ranges[/bold cyan]", show_header=True, header_style="bold magenta")
                        ranges_table.add_column("Metric", style="cyan", no_wrap=True)
                        ranges_table.add_column("Range", style="green")
                        ranges_table.add_row(f"Primary ({metric})", f"{min(primary_values):.4f} - {max(primary_values):.4f}")
                        ranges_table.add_row(f"Secondary ({secondary_metric})", f"{min(secondary_values):.4f} - {max(secondary_values):.4f}")
                        ranges_table.add_row("Composite", f"{min(metric_values):.4f} - {max(metric_values):.4f}")
                        ranges_table.add_row("Hybrid Score", f"{min(final_df['hybrid_score']):.4f} - {max(final_df['hybrid_score']):.4f}")
                        console.print(ranges_table)
                    
                else:
                    # Single metric ranking
                    metric_values = []
                    for _, row in final_df.iterrows():
                        lookup_key = row['lookup_key']
                        
                        if lookup_key in updated_graph.nodes():
                            metric_value = self._calculate_metric(
                                updated_graph, file_name, lookup_key, metric, final_df
                            )
                        else:
                            # Node not in graph - assign lowest value
                            # For shortest path (distance metric), use inf; for others use 0
                            if metric == 'sp':
                                metric_value = float('inf')
                            else:
                                metric_value = 0.0
                        
                        metric_values.append(metric_value)
                    
                    if self.verbose:
                        # Create metric ranges table for single metric
                        ranges_table = Table(title="[bold cyan]Metric Ranges[/bold cyan]", show_header=True, header_style="bold magenta")
                        ranges_table.add_column("Metric", style="cyan", no_wrap=True)
                        ranges_table.add_column("Range", style="green")
                        ranges_table.add_row(f"{metric.capitalize()}", f"{min(metric_values):.4f} - {max(metric_values):.4f}")
                        ranges_table.add_row("Hybrid Score", f"{min(final_df['hybrid_score']):.4f} - {max(final_df['hybrid_score']):.4f}")
                        console.print(ranges_table)
                
                # Create a copy of the DataFrame to add new columns
                reranked_df = final_df.copy()
                # Add metric values to DataFrame
                reranked_df[f'{composite_name}_score'] = metric_values
                
                # Add calculated distances from add_query_to_graph for debugging
                reranked_df['hung_distance_2'] = reranked_df['lookup_key'].map(
                    lambda x: self.last_calculated_distances.get(x, None)
                )

                ascending = (metric == 'sp' and not is_composite)  # Only sp as primary single metric is ascending
                reranked_df[f'{composite_name}_rank'] = reranked_df[f'{composite_name}_score'].rank(ascending=ascending, method='first')
                
                # Add original k-indices for comparison
                reranked_df['original_k_index'] = range(len(reranked_df))

                # Sort by the metric to show reranked order
                reranked_df_sorted = reranked_df.sort_values(f'{composite_name}_score', ascending=ascending)
                reranked_df_sorted[f'{composite_name}_k_index'] = range(len(reranked_df_sorted))
                reranked_df_sorted.to_csv('reranked_df_sorted.csv')

                # Add ranking change information
                reranked_df_sorted['rank_change'] = reranked_df_sorted['original_k_index'] - reranked_df_sorted[f'{composite_name}_k_index']
                
                # Add metadata about the reranking
                reranked_df_sorted['rerank_metric'] = composite_name
                reranked_df_sorted['rerank_timestamp'] = pd.Timestamp.now()
                reranked_df_sorted['query_connections'] = query_connections
                if is_composite:
                    reranked_df_sorted['primary_metric'] = metric
                    reranked_df_sorted['secondary_metric'] = secondary_metric
                    reranked_df_sorted['primary_weight'] = primary_weight
                    reranked_df_sorted['secondary_weight'] = secondary_weight
                
                # Calculate comprehensive improvements for all metrics
                if self.verbose:
                    console.print(f"[blue]📊 Calculating comprehensive improvements (Hybrid, Tanimoto, MCS)...[/blue]")
                    console.print(f"[yellow]ℹ️  Methodology: Using AVERAGE scores in top-K results (not maximum values)[/yellow]")
                
                # Before reranking (original k-index order)
                original_order = final_df.copy()  # Already sorted by hybrid score (best first)
                
                # Show Top 5 Ranked Results (original rankings)
                if self.verbose and len(original_order) > 0:
                    top_original = original_order.head(5)
                    if len(top_original) > 0:
                        original_table = Table(title="[bold cyan]Top 5 Ranked Results (Original)[/bold cyan]", show_header=True, header_style="bold magenta")
                        original_table.add_column("Rank", style="cyan", justify="right")
                        original_table.add_column("Lookup Key", style="yellow", no_wrap=True)
                        original_table.add_column("Hybrid Score", style="green", justify="right")
                        original_table.add_column("Tanimoto", style="blue", justify="right")
                        original_table.add_column("MCS Score", style="purple", justify="right")
                        
                        for i, (_, row) in enumerate(top_original.iterrows(), 1):
                            lookup_key = row['lookup_key'][:20] + "..." if len(row['lookup_key']) > 20 else row['lookup_key']
                            hybrid_score = f"{row['hybrid_score']:.4f}"
                            tanimoto_score = f"{row.get('tanimoto_similarity', 0.0):.4f}"
                            mcs_score = f"{row.get('mcs_score_d', 0.0):.4f}"
                            
                            original_table.add_row(str(i), lookup_key, hybrid_score, tanimoto_score, mcs_score)
                        
                        console.print(original_table)
                
                # Show Top 5 Reranked Results immediately after original results
                top_changes = reranked_df_sorted.head(5)
                if self.verbose and len(top_changes) > 0:
                    ranking_table = Table(title="[bold cyan]Top 5 Reranked Results[/bold cyan]", show_header=True, header_style="bold magenta")
                    ranking_table.add_column("New Rank", style="cyan", justify="right")
                    ranking_table.add_column("Lookup Key", style="yellow", no_wrap=True)
                    ranking_table.add_column("Hybrid Score", style="green", justify="right")
                    ranking_table.add_column("Original Rank", style="blue", justify="right")
                    ranking_table.add_column("Rank Change", style="magenta", justify="right")
                    
                    for i, (_, row) in enumerate(top_changes.iterrows(), 1):
                        lookup_key = row['lookup_key'][:20] + "..." if len(row['lookup_key']) > 20 else row['lookup_key']
                        hybrid_score = f"{row['hybrid_score']:.4f}"
                        original_rank = int(row['original_k_index']) + 1
                        rank_change = int(row['rank_change'])
                        
                        rank_change_str = f"[green]+{rank_change}[/green]" if rank_change > 0 else f"[red]{rank_change}[/red]" if rank_change < 0 else "[yellow]0[/yellow]"
                        
                        ranking_table.add_row(str(i), lookup_key, hybrid_score, str(original_rank), rank_change_str)
                    
                    console.print(ranking_table)
                
                # After reranking (new metric-based order)
                reranked_order = reranked_df_sorted.copy()
                
                # Calculate improvements for all metrics and top-k values
                all_improvements = {}
                metrics_to_track = {
                    'hybrid': 'hybrid_score',
                    'tanimoto': 'tanimoto_similarity', 
                    'mcs': 'mcs_score_d'
                }
                
                for metric_name, column_name in metrics_to_track.items():
                    all_improvements[metric_name] = {}
                    
                    for k in [1, 3, 5, 10]:
                        # Before reranking - scores in top-k by original ranking
                        top_k_original = original_order.head(k)
                        avg_score_before = top_k_original[column_name].mean()
                        max_score_before = top_k_original[column_name].max()
                        
                        # After reranking - scores in top-k by new metric ranking
                        top_k_reranked = reranked_order.head(k)
                        avg_score_after = top_k_reranked[column_name].mean()
                        max_score_after = top_k_reranked[column_name].max()
                        
                        # Calculate improvements for both average and maximum
                        avg_improvement = avg_score_after - avg_score_before
                        avg_percent_improvement = (avg_improvement / avg_score_before * 100) if avg_score_before > 0 else 0
                        
                        max_improvement = max_score_after - max_score_before
                        max_percent_improvement = (max_improvement / max_score_before * 100) if max_score_before > 0 else 0
                        
                        all_improvements[metric_name][f'top_{k}'] = {
                            f'avg_{metric_name}_before': float(avg_score_before),
                            f'avg_{metric_name}_after': float(avg_score_after),
                            f'max_{metric_name}_before': float(max_score_before),
                            f'max_{metric_name}_after': float(max_score_after),
                            'avg_improvement': float(avg_improvement),
                            'avg_percent_improvement': float(avg_percent_improvement),
                            'max_improvement': float(max_improvement),
                            'max_percent_improvement': float(max_percent_improvement),
                            'avg_improved': bool(avg_improvement > 0),
                            'max_improved': bool(max_improvement > 0)
                        }
                
                # Create improvements summary table
                if self.verbose:
                    improvements_table = Table(title="[bold cyan]Average & Maximum Score Improvements Summary[/bold cyan]", show_header=True, header_style="bold magenta")
                    improvements_table.add_column("Metric", style="cyan", no_wrap=True)
                    improvements_table.add_column("Top-1 Avg Change", style="green", justify="right")
                    improvements_table.add_column("Top-1 Max Change", style="blue", justify="right")
                    improvements_table.add_column("Top-3 Avg Change", style="green", justify="right")
                    improvements_table.add_column("Top-3 Max Change", style="blue", justify="right")
                    improvements_table.add_column("Top-5 Avg Change", style="green", justify="right")
                    improvements_table.add_column("Top-5 Max Change", style="blue", justify="right")
                    improvements_table.add_column("Top-10 Avg Change", style="green", justify="right")
                    improvements_table.add_column("Top-10 Max Change", style="blue", justify="right")
                    
                    for metric_name in metrics_to_track:
                        row_data = [metric_name.title()]
                        for k in [1, 3, 5, 10]:
                            key = f'top_{k}'
                            
                            # Average improvements
                            avg_improvement = all_improvements[metric_name][key]['avg_improvement']
                            avg_percent = all_improvements[metric_name][key]['avg_percent_improvement']
                            avg_improved = all_improvements[metric_name][key]['avg_improved']
                            
                            if avg_improved:
                                row_data.append(f"[green]+{avg_improvement:.4f}[/green] ({avg_percent:+.1f}%)")
                            else:
                                row_data.append(f"[red]{avg_improvement:.4f}[/red] ({avg_percent:+.1f}%)")
                            
                            # Maximum improvements
                            max_improvement = all_improvements[metric_name][key]['max_improvement']
                            max_percent = all_improvements[metric_name][key]['max_percent_improvement']
                            max_improved = all_improvements[metric_name][key]['max_improved']
                            
                            if max_improved:
                                row_data.append(f"[green]+{max_improvement:.4f}[/green] ({max_percent:+.1f}%)")
                            else:
                                row_data.append(f"[red]{max_improvement:.4f}[/red] ({max_percent:+.1f}%)")
                        
                        improvements_table.add_row(*row_data)
                    
                    console.print(improvements_table)
                    
                    # Create detailed before/after scores table
                    detailed_table = Table(title="[bold cyan]Average & Maximum Scores Before vs After Reranking[/bold cyan]", show_header=True, header_style="bold magenta")
                    detailed_table.add_column("Metric", style="cyan", no_wrap=True)
                    detailed_table.add_column("Top-K", style="yellow", justify="center")
                    detailed_table.add_column("Avg Before", style="blue", justify="right")
                    detailed_table.add_column("Avg After", style="green", justify="right")
                    detailed_table.add_column("Avg Change", style="magenta", justify="right")
                    detailed_table.add_column("Max Before", style="blue", justify="right")
                    detailed_table.add_column("Max After", style="green", justify="right")
                    detailed_table.add_column("Max Change", style="magenta", justify="right")
                    
                    for metric_name in metrics_to_track:
                        for k in [1, 3, 5, 10]:
                            key = f'top_{k}'
                            
                            # Average values
                            avg_before = all_improvements[metric_name][key][f'avg_{metric_name}_before']
                            avg_after = all_improvements[metric_name][key][f'avg_{metric_name}_after']
                            avg_improvement = all_improvements[metric_name][key]['avg_improvement']
                            avg_percent = all_improvements[metric_name][key]['avg_percent_improvement']
                            avg_improved = all_improvements[metric_name][key]['avg_improved']
                            
                            # Maximum values
                            max_before = all_improvements[metric_name][key][f'max_{metric_name}_before']
                            max_after = all_improvements[metric_name][key][f'max_{metric_name}_after']
                            max_improvement = all_improvements[metric_name][key]['max_improvement']
                            max_percent = all_improvements[metric_name][key]['max_percent_improvement']
                            max_improved = all_improvements[metric_name][key]['max_improved']
                            
                            avg_change_str = f"[green]+{avg_improvement:.4f}[/green] ({avg_percent:+.1f}%)" if avg_improved else f"[red]{avg_improvement:.4f}[/red] ({avg_percent:+.1f}%)"
                            max_change_str = f"[green]+{max_improvement:.4f}[/green] ({max_percent:+.1f}%)" if max_improved else f"[red]{max_improvement:.4f}[/red] ({max_percent:+.1f}%)"
                            
                            detailed_table.add_row(
                                metric_name.title() if k == 1 else "",  # Only show metric name on first row
                                f"Top-{k}",
                                f"{avg_before:.4f}",
                                f"{avg_after:.4f}",
                                avg_change_str,
                                f"{max_before:.4f}",
                                f"{max_after:.4f}",
                                max_change_str
                            )
                        
                        # Add separator between metrics
                        if metric_name != list(metrics_to_track.keys())[-1]:
                            detailed_table.add_row("", "", "", "", "", "", "", "")
                    
                    console.print(detailed_table)
                
                # Add all improvement data to the DataFrame (both average and maximum)
                for metric_name in metrics_to_track:
                    for k in [1, 3, 5, 10]:
                        key = f'top_{k}'
                        # Average improvements
                        reranked_df_sorted[f'avg_{metric_name}_before_{k}'] = all_improvements[metric_name][key][f'avg_{metric_name}_before']
                        reranked_df_sorted[f'avg_{metric_name}_after_{k}'] = all_improvements[metric_name][key][f'avg_{metric_name}_after']
                        reranked_df_sorted[f'{metric_name}_avg_improvement_{k}'] = all_improvements[metric_name][key]['avg_improvement']
                        reranked_df_sorted[f'{metric_name}_avg_percent_improvement_{k}'] = all_improvements[metric_name][key]['avg_percent_improvement']
                        reranked_df_sorted[f'{metric_name}_avg_improved_{k}'] = all_improvements[metric_name][key]['avg_improved']
                        
                        # Maximum improvements
                        reranked_df_sorted[f'max_{metric_name}_before_{k}'] = all_improvements[metric_name][key][f'max_{metric_name}_before']
                        reranked_df_sorted[f'max_{metric_name}_after_{k}'] = all_improvements[metric_name][key][f'max_{metric_name}_after']
                        reranked_df_sorted[f'{metric_name}_max_improvement_{k}'] = all_improvements[metric_name][key]['max_improvement']
                        reranked_df_sorted[f'{metric_name}_max_percent_improvement_{k}'] = all_improvements[metric_name][key]['max_percent_improvement']
                        reranked_df_sorted[f'{metric_name}_max_improved_{k}'] = all_improvements[metric_name][key]['max_improved']
                
                # Overall improvement summary (keep hybrid-focused for backwards compatibility, using average values)
                hybrid_improvements = all_improvements['hybrid']
                any_improvement = any(hybrid_improvements[f'top_{k}']['avg_improved'] for k in [1, 3, 5])
                best_improvement = max(hybrid_improvements[f'top_{k}']['avg_improvement'] for k in [1, 3, 5])
                best_percent_improvement = max(hybrid_improvements[f'top_{k}']['avg_percent_improvement'] for k in [1, 3, 5])
                
                reranked_df_sorted['any_hybrid_improvement'] = any_improvement
                reranked_df_sorted['best_hybrid_improvement'] = best_improvement
                reranked_df_sorted['best_hybrid_percent_improvement'] = best_percent_improvement
                
                # Identify specific compounds that contributed to improvements
                improved_compounds = []
                for k in [1, 3, 5]:
                    if hybrid_improvements[f'top_{k}']['avg_improved']:
                        # Find compounds in top-k after reranking that weren't in top-k before
                        top_k_after_smiles = set(reranked_order.head(k)['lookup_key'])
                        top_k_before_smiles = set(original_order.head(k)['lookup_key'])
                        
                        # New compounds that entered top-k
                        new_top_k = top_k_after_smiles - top_k_before_smiles
                        
                        # Check which of these new compounds have high hybrid scores
                        for smiles in new_top_k:
                            compound_data = reranked_order[reranked_order['lookup_key'] == smiles]
                            if len(compound_data) > 0:
                                hybrid_score = compound_data['hybrid_score'].iloc[0]
                                original_rank = compound_data['original_k_index'].iloc[0]
                                new_rank = compound_data[f'{composite_name}_k_index'].iloc[0]
                                
                                improved_compounds.append({
                                    'compound': smiles,
                                    'hybrid_score': float(hybrid_score),  # Convert to Python float
                                    'original_rank': int(original_rank),  # Convert to Python int
                                    'new_rank': int(new_rank),  # Convert to Python int
                                    'rank_improvement': int(original_rank - new_rank),  # Convert to Python int
                                    'top_k_level': int(k)  # Convert to Python int
                                })
                
                # Sort improved compounds by hybrid score (best first)
                improved_compounds.sort(key=lambda x: x['hybrid_score'], reverse=True)
                
                # Add comprehensive summary of improvements
                improvement_summary = {
                    'total_improvements': int(len([k for k in [1, 3, 5] if hybrid_improvements[f'top_{k}']['avg_improved']])),
                    'any_improvement': bool(any_improvement),
                    'best_improvement': float(best_improvement),
                    'best_percent_improvement': float(best_percent_improvement),
                    'improved_compounds': improved_compounds,
                    'hybrid_improvements': hybrid_improvements,
                    'all_improvements': all_improvements  # Include all metrics
                }
                
                if self.verbose:
                    summary_text = Text()
                    summary_text.append("📈 Average Score Improvement Summary\n", style="bold blue")
                    summary_text.append("(Based on average scores in top-K results, not maximum values)\n\n", style="italic yellow")
                    
                    summary_text.append(f"Any hybrid avg improvement: ", style="cyan")
                    summary_text.append(f"{any_improvement}", style="green" if any_improvement else "red")
                    summary_text.append("\n")
                    
                    summary_text.append(f"Best hybrid avg improvement: ", style="cyan")
                    summary_text.append(f"{best_improvement:+.4f} ({best_percent_improvement:+.2f}%)", 
                                      style="green" if best_improvement > 0 else "red")
                    summary_text.append("\n")
                    
                    summary_text.append(f"Improved compounds: ", style="cyan")
                    summary_text.append(f"{len(improved_compounds)}", style="yellow")
                    summary_text.append("\n\n")
                    
                    # Show summary for all metrics
                    for metric_name in metrics_to_track:
                        metric_improvements = all_improvements[metric_name]
                        any_metric_improvement = any(metric_improvements[f'top_{k}']['avg_improved'] for k in [1, 3, 5])
                        best_metric_improvement = max(metric_improvements[f'top_{k}']['avg_improvement'] for k in [1, 3, 5])
                        
                        summary_text.append(f"{metric_name.title()} any avg improvement: ", style="cyan")
                        summary_text.append(f"{any_metric_improvement}", style="green" if any_metric_improvement else "red")
                        summary_text.append(f", best avg: ", style="cyan")
                        summary_text.append(f"{best_metric_improvement:+.4f}", 
                                          style="green" if best_metric_improvement > 0 else "red")
                        summary_text.append("\n")
                    
                    if improved_compounds:
                        summary_text.append("\nTop improved compound: ", style="cyan")
                        summary_text.append(f"{improved_compounds[0]['compound'][:15]}...", style="yellow")
                        summary_text.append(f" (rank {improved_compounds[0]['original_rank']+1} → {improved_compounds[0]['new_rank']+1}, "
                                          f"hybrid: {improved_compounds[0]['hybrid_score']:.4f})", style="green")
                    
                    summary_panel = Panel(summary_text, title="[bold cyan]Average Score Improvement Summary[/bold cyan]", border_style="blue")
                    console.print(summary_panel)

                    # changes table

                    changes_table = Table(title="[bold cyan]Ranking Changes Summary[/bold cyan]", show_header=True, header_style="bold magenta")
                    changes_table.add_column("Metric", style="cyan", no_wrap=True)
                    changes_table.add_column("Value", style="green", justify="right")
                    
                    positive_changes = (reranked_df_sorted['rank_change'] > 0).sum()
                    negative_changes = (reranked_df_sorted['rank_change'] < 0).sum()
                    no_changes = (reranked_df_sorted['rank_change'] == 0).sum()
                    
                    changes_table.add_row("Positive rank changes", f"[green]{positive_changes}[/green]")
                    changes_table.add_row("Negative rank changes", f"[red]{negative_changes}[/red]")
                    changes_table.add_row("No rank changes", f"[yellow]{no_changes}[/yellow]")
                    changes_table.add_row("Total results", str(len(reranked_df_sorted)))
                    
                    console.print(changes_table)
                
                # Store improvement summary in the DataFrame as a JSON string for easy access
                reranked_df_sorted['improvement_summary'] = json.dumps(improvement_summary)
                
                # Create the reranked annotation tuple
                reranked_annotation = (file_name, exp_smiles, reranked_df_sorted)
                reranked_annotations.append(reranked_annotation)
                
                progress.advance(task)
    
        # Create final summary panel
        summary_text = Text()
        summary_text.append("✅ Reranking Complete!\n\n", style="bold green")
        summary_text.append(f"• Input annotations: ", style="cyan")
        summary_text.append(f"{len(annotations)}", style="yellow")
        summary_text.append("\n")
        summary_text.append(f"• Filtered out (< {min_connections} connections): ", style="cyan")
        summary_text.append(f"{connection_filtered_count}", style="red")
        summary_text.append("\n")
        summary_text.append(f"• Successfully processed: ", style="cyan")
        summary_text.append(f"{len(reranked_annotations)}", style="green")
        summary_text.append("\n")
        summary_text.append(f"• Success rate: ", style="cyan")
        success_rate = len(reranked_annotations)/len(annotations)*100
        success_color = "green" if success_rate >= 80 else "yellow" if success_rate >= 60 else "red"
        summary_text.append(f"{success_rate:.1f}%", style=success_color)
        
        summary_panel = Panel(summary_text, title="[bold cyan]Reranking Summary[/bold cyan]", border_style="green")
        console.print(summary_panel)
        
        return reranked_annotations

    def _calculate_metric(self, graph: nx.Graph, query_id: str, target_id: str, metric: str, 
                         annotation_data: pd.DataFrame = None) -> float:
        """
        Calculate a metric between query and target nodes (network-based or annotation-based).
        
        Parameters:
        -----------
        graph : nx.Graph
            Graph containing both nodes
        query_id : str
            Query node ID
        target_id : str
            Target node ID
        metric : str
            Metric to calculate
            Network options: 'jaccard', 'adamic_adar', 'weighted_jaccard', 'cn', 'cos', 'si', 'hpi', 'hdi', 'lhn', 'pa', 'ra', 'sp',
                             'ra_weighted_sum', 'ra_weighted_avg', 'ra_weighted_min', 'ra_weighted_max', 'ra_weighted_product'
            Annotation options: 'hungarian_nn', 'hybrid_score', 'tanimoto_similarity', 'mcs_score'
        annotation_data : pd.DataFrame, optional
            Annotation data for annotation-based metrics
            
        Returns:
        --------
        float : Metric value (normalized to 0-1 range where 1 is most similar)
        """
        # Handle annotation-based metrics
        if metric in ['hungarian_nn', 'hybrid_score', 'tanimoto_similarity', 'mcs_score']:
            return self._calculate_annotation_metric(target_id, metric, annotation_data, graph, query_id)
        
        # Handle network-based metrics
        return self._calculate_network_metric(graph, query_id, target_id, metric)
    
    def _calculate_annotation_metric(self, target_id: str, metric: str, annotation_data: pd.DataFrame, 
                                   graph: nx.Graph = None, query_id: str = None) -> float:
        """
        Calculate annotation-based metrics.
        
        Parameters:
        -----------
        target_id : str
            Target node ID (lookup_key)
        metric : str
            Annotation metric to calculate
        annotation_data : pd.DataFrame
            Annotation data containing the metrics
        graph : nx.Graph, optional
            Graph for extracting edge-based metrics (required for hungarian_nn)
        query_id : str, optional
            Query node ID (required for hungarian_nn)
            
        Returns:
        --------
        float : Metric value (normalized to 0-1 range where 1 is most similar)
        """
        if annotation_data is None:
            return 0.0
        
        if metric == 'hungarian_nn':
            # Hungarian distance - get from graph edges instead of annotation data
            if graph is None or query_id is None:
                return 0.0
            
            # Check if there's an edge between query and target in the graph
            if graph.has_edge(query_id, target_id):
                edge_data = graph.get_edge_data(query_id, target_id)
                hungarian_distance = edge_data.get('weight', float('inf'))
                # Use inverse normalization: 1/(1+distance)
                return 1.0 / (1.0 + hungarian_distance)
            else:
                # No edge exists - return 0 (no similarity)
                return 0.0
        
        # For other metrics, use annotation data
        # Find the row for this target_id
        target_row = annotation_data[annotation_data['lookup_key'] == target_id]
        if len(target_row) == 0:
            return 0.0
        
        target_row = target_row.iloc[0]
            
        if metric == 'hybrid_score':
            # Hybrid score is already in 0-1 range
            return float(target_row['hybrid_score'])
            
        elif metric == 'tanimoto_similarity':
            # Tanimoto similarity is already in 0-1 range
            return float(target_row['tanimoto_similarity'])
            
        elif metric == 'mcs_score':
            # MCS score (assuming mcs_score_d column)
            return float(target_row['mcs_score_d'])
            
        else:
            raise ValueError(f"Unknown annotation metric: {metric}")


    def _mc_loop(nbrs_idx, cdfs, C_pows, u_idx, v_idx, K, tol, num_samples):
        """
        Numba-accelerated sampling loop for CoSimRank.
        nbrs_idx: list of neighbor-index lists
        cdfs: list of numpy arrays with CDFs
        C_pows: 1D numpy array of C**k
        u_idx, v_idx: integer indices of start nodes
        K, tol, num_samples: as before
        """
        score_sum = 0.0
        for _ in range(num_samples):
            cu = u_idx
            cv = v_idx
            for k in range(K + 1):
                ck = C_pows[k]
                if ck < tol:
                    break
                if cu == cv:
                    score_sum += ck
                if k == K:
                    break
                # walk step for u
                nbrs_u = nbrs_idx[cu]
                if nbrs_u.size == 0:
                    break
                cdf_u = cdfs[cu]
                r = random.random()
                idx_u = np.searchsorted(cdf_u, r)
                cu = nbrs_u[idx_u]
                # walk step for v
                nbrs_v = nbrs_idx[cv]
                if nbrs_v.size == 0:
                    break
                cdf_v = cdfs[cv]
                r = random.random()
                idx_v = np.searchsorted(cdf_v, r)
                cv = nbrs_v[idx_v]
        return score_sum


    def weighted_cosimrank(self, G, u, v,
                        C=0.5, K=3, tol=1e-3,
                        weight_label='hybrid_score',
                        num_samples=1000):
        """
        Monte Carlo–based weighted CoSimRank similarity via random-walk sampling.

        Parameters
        ----------
        self : object
            Unused, for class method compatibility.
        G : networkx.Graph or networkx.DiGraph
            Input graph with edge-weights in G[x][y][weight_label].
        u, v : node
            The two nodes whose similarity is being computed.
        C : float, default=0.7
            Decay factor per hop (0 < C < 1).
        K : int, default=3
            Maximum hops to simulate.
        tol : float, default=1e-3
            Inner-loop early-stop threshold on decay (break if C**k < tol).
        weight_label : str, default='hybrid_score'
            Edge attribute key for weights; missing defaults to 1.0.
        num_samples : int, default=1000
            Number of random-walk pairs to sample.

        Returns
        -------
        score : float
            Approximate CoSimRank similarity s(u, v) in [0, 1].
        """
        # Precompute weighted neighbor distributions
        neighbors = {}
        for node in G.nodes():
            items = list(G[node].items())
            weights = [d.get(weight_label, 1.0) for _, d in items]
            total = sum(weights)
            if total > 0:
                nbrs = [nbr for nbr, _ in items]
                cdf = np.cumsum(weights) / total
                neighbors[node] = (nbrs, cdf.tolist())
            else:
                neighbors[node] = ([], [])

        score_sum = 0.0
        # Monte Carlo sampling
        for _ in range(num_samples):
            cu, cv = u, v
            for k in range(K + 1):
                ck = C**k
                if ck < tol:
                    break
                # accumulate if walkers meet
                if cu == cv:
                    score_sum += ck
                if k == K:
                    break
                # step walker from u
                nbrs_u, cdf_u = neighbors.get(cu, ([], []))
                if not nbrs_u:
                    break
                r = random.random()
                idx_u = bisect.bisect_left(cdf_u, r)
                cu = nbrs_u[idx_u]
                # step walker from v
                nbrs_v, cdf_v = neighbors.get(cv, ([], []))
                if not nbrs_v:
                    break
                r = random.random()
                idx_v = bisect.bisect_left(cdf_v, r)
                cv = nbrs_v[idx_v]

        return score_sum / float(num_samples)
        
    def _get_edge_weight(self, graph: nx.Graph, node1: str, node2: str, weight_attr: str, node1_is_query: bool) -> float:
        """
        Get edge weight between two nodes, with special handling for query nodes.
        
        Parameters
        ----------
        graph : networkx.Graph
            The graph
        node1 : str
            First node ID
        node2 : str
            Second node ID
        weight_attr : str
            Attribute name to look for
        node1_is_query : bool
            Whether node1 is a query node
            
        Returns
        -------
        float
            Edge weight value
        """
        if not graph.has_edge(node1, node2):
            return 1.0  # No edge exists
            
        edge_data = graph[node1][node2]
        
        # Step 1: If the requested attribute exists directly on the edge, use it
        if weight_attr in edge_data:
            return float(edge_data[weight_attr])
        
        # Step 2: Map weight_attr to corresponding edge attributes in your graph
        edge_attr_mapping = {
            'hybrid_score': 'hybrid_similarity',
            'tanimoto_similarity': 'tanimoto_similarity', 
            'mcs_score': 'mcs_similarity',
            'mcs_score_d': 'mcs_similarity'
        }
        
        # Check if we have the corresponding edge attribute
        if weight_attr in edge_attr_mapping:
            edge_attr_name = edge_attr_mapping[weight_attr]
            if edge_attr_name in edge_data:
                return float(edge_data[edge_attr_name])
        
        # Step 3: Special handling for query edges - get from annotation data
        if node1_is_query and weight_attr in ['hybrid_score', 'tanimoto_similarity', 'mcs_score', 'mcs_score_d']:
            if hasattr(self, 'lookup_results') and self.lookup_results is not None:
                target_row = self.lookup_results[self.lookup_results['lookup_key'] == node2]
                if len(target_row) > 0:
                    if weight_attr == 'hybrid_score':
                        return float(target_row['hybrid_score'].iloc[0])
                    elif weight_attr == 'tanimoto_similarity':
                        return float(target_row['tanimoto_similarity'].iloc[0])
                    elif weight_attr in ['mcs_score', 'mcs_score_d']:
                        return float(target_row['mcs_score_d'].iloc[0])
        
        # Step 4: Fallback - convert Hungarian distance to similarity
        hungarian_distance = None
        if 'hungarian_distance' in edge_data:
            hungarian_distance = float(edge_data['hungarian_distance'])
        elif 'weight' in edge_data:
            hungarian_distance = float(edge_data['weight'])
        
        if hungarian_distance is not None:
            return 1.0 / (1.0 + hungarian_distance)
        
        # Final fallback (should rarely be reached)
        return 1.0

    def _calculate_weighted_ra(self, graph: nx.Graph, u: str, v: str, mode: str = 'product', weight_attr: str = 'hybrid_score') -> float:
        """
        Compute a weighted Resource Allocation index between u and v.
        
        Parameters
        ----------
        graph : networkx.Graph
            Undirected graph.
        u, v : str
            Node IDs in graph
        mode : str
            How to combine the two edge‐attribute values for edges (u–w) and (v–w).
            Options: 'sum', 'avg', 'min', 'max', 'product'
        weight_attr : str
            Name of the edge attribute to use for weights.
        
        Returns
        -------
        float
            The weighted RA score.
        """
        # Check if both nodes exist in the graph
        if u not in graph.nodes() or v not in graph.nodes():
            return 0.0
        
        shared = set(graph[u]) & set(graph[v])
        score = 0.0
        
        # Check if one of the nodes is a query node (has node_type='query')
        u_is_query = graph.nodes[u].get('node_type') == 'query'
        v_is_query = graph.nodes[v].get('node_type') == 'query'
        
        for w in shared:
            deg_w = graph.degree(w)
            if deg_w == 0:
                continue

            # Get edge attributes with special handling for query nodes
            attr_uw = self._get_edge_weight(graph, u, w, weight_attr, u_is_query)
            attr_vw = self._get_edge_weight(graph, v, w, weight_attr, v_is_query)
            
            if mode == 'sum':
                ew = attr_uw + attr_vw
            elif mode == 'avg':
                ew = 0.5 * (attr_uw + attr_vw)
            elif mode == 'min':
                ew = min(attr_uw, attr_vw)
            elif mode == 'max':
                ew = max(attr_uw, attr_vw)
            elif mode == 'product':
                ew = attr_uw * attr_vw
            else: 
                raise ValueError(f"Unknown mode: {mode}. Available: 'sum', 'avg', 'min', 'max', 'product'")

            # Accumulate weighted resource allocation
            score += ew / deg_w

        return score

    def _calculate_ra_with_coherence(
        self,
        graph: nx.Graph,
        u: str,
        v: str,
        mode: str = 'product',                 # 'product' enforces coherence
        spec_attr: str = 'hungarian_distance', # spectral (distance → similarity)
        struct_attr: str = 'hybrid_similarity',# structural (already a similarity)
        alpha: float = 1.0,                    # hub penalty exponent (deg(w)^alpha)
        normalize: bool = True,
        dmax: float | None = 1.0,              # max dist for linear mapping; None -> 1/(1+d)
        eps: float = 1e-12,
        weight_attr: str | None = None
    ) -> float:
        """
        Resource Allocation (RA) with *coherence*: each shared neighbor contributes a product of
        spectral and structural similarities, discounted by a hub penalty (deg(w)^alpha).

        - For edges touching a query node, use spectral similarity (from spec_attr).
        - For edges touching a non-query node, use structural similarity (from struct_attr).
        - Combine the two legs (u–w and v–w) per `mode` ('product' recommended).
        """

        # ---------- guards ----------
        if u not in graph or v not in graph:
            return 0.0

        shared = set(graph[u]) & set(graph[v])
        if not shared:
            return 0.0

        u_is_query = graph.nodes[u].get('node_type') == 'query'
        v_is_query = graph.nodes[v].get('node_type') == 'query'

        # backward compatibility: allow overriding structural attribute via weight_attr
        if weight_attr is not None and isinstance(weight_attr, str) and len(weight_attr) > 0:
            struct_attr = weight_attr

        # ---------- helpers ----------
        def _edge_val(a, b, attr):
            return graph[a][b].get(attr, None)

        def _to_sim(x, attr_name):
            # Map distances -> similarity in [0,1]; pass similarities through and clip.
            if x is None:
                return 0.0
            x = float(x)
            name = (attr_name or '').lower()
            if 'dist' in name:  # treat as distance
                if dmax is None:
                    return 1.0 / (1.0 + max(0.0, x))
                if dmax <= 0:
                    return 0.0
                x = max(0.0, min(x, dmax))
                return 1.0 - (x / dmax)
            # assume already a similarity
            return max(0.0, min(1.0, x))

        def _combine(a, b, how):
            if how == 'sum':     return a + b
            if how == 'avg':     return 0.5 * (a + b)
            if how == 'min':     return min(a, b)
            if how == 'max':     return max(a, b)
            if how == 'product': return a * b
            raise ValueError(f"Unknown mode: {how}. Available: 'sum','avg','min','max','product'.")

        # ---------- accumulate ----------
        num, denom = 0.0, 0.0

        for w in shared:
            deg_w = graph.degree(w)
            if deg_w <= 0:
                continue
            inv_hub = (deg_w ** (-alpha)) if deg_w > 0 else 0.0

            # u–w leg: spectral if u is query, else structural
            uw_spec   = _to_sim(_edge_val(u, w, spec_attr),   spec_attr)
            uw_struct = _to_sim(_edge_val(u, w, struct_attr), struct_attr)
            leg_uw    = uw_spec if u_is_query else uw_struct

            # v–w leg: spectral if v is query, else structural
            vw_spec   = _to_sim(_edge_val(v, w, spec_attr),   spec_attr)
            vw_struct = _to_sim(_edge_val(v, w, struct_attr), struct_attr)
            leg_vw    = vw_spec if v_is_query else vw_struct

            # combine legs (product => PWRA coherence gate)
            ew = _combine(leg_uw, leg_vw, mode)

            num   += ew * inv_hub
            denom += inv_hub

        return num / (denom + eps) if normalize else num
    

    def _calculate_network_metric(self, graph: nx.Graph, query_id: str, target_id: str, metric: str) -> float:
        """
        Calculate a network metric between query and target nodes.
        
        Parameters:
        -----------
        graph : nx.Graph
            Graph containing both nodes
        query_id : str
            Query node ID
        target_id : str
            Target node ID
        metric : str
            Network metric to calculate
            Options: 'jaccard', 'adamic_adar', 'weighted_jaccard', 'cn', 'cos', 'si', 'hpi', 'hdi', 'lhn', 'pa', 'ra',
                     'ra_weighted_sum', 'ra_weighted_avg', 'ra_weighted_min', 'ra_weighted_max', 'ra_weighted_product', 'cosimrank'
            
        Returns:
        --------
        float : Network metric value
        """
        # Check if both nodes exist in the graph
        if query_id not in graph.nodes() or target_id not in graph.nodes():
            # Node not in graph - assign lowest value
            # For shortest path (distance metric), use inf; for others use 0
            if metric == 'sp':
                return float('inf')
            else:
                return 0.0
        
        # Get basic neighbor information for all metrics
        query_neighbors = set(graph.neighbors(query_id))
        target_neighbors = set(graph.neighbors(target_id))
        shared_nodes = query_neighbors.intersection(target_neighbors)
        total_nodes = query_neighbors.union(target_neighbors)
        
        if metric == 'jaccard':
            # Jaccard coefficient
            return len(shared_nodes) / len(total_nodes) if len(total_nodes) > 0 else 0.0
            
        elif metric == 'adamic_adar':
            # Adamic-Adar index
            value = 0.0
            for common_neighbor in shared_nodes:
                degree = graph.degree(common_neighbor)
                if degree > 1:
                    value += 1.0 / np.log(degree)
            return value
            
        elif metric == 'weighted_jaccard':
            # Weighted Jaccard coefficient
            query_neighbors_weighted = {}
            for neighbor in query_neighbors:
                weight = graph[query_id][neighbor].get('weight', 1.0)
                query_neighbors_weighted[neighbor] = weight
            
            target_neighbors_weighted = {}
            for neighbor in target_neighbors:
                weight = graph[target_id][neighbor].get('weight', 1.0)
                target_neighbors_weighted[neighbor] = weight
            
            # Calculate weighted Jaccard
            if len(shared_nodes) > 0 or len(query_neighbors) > 0 or len(target_neighbors) > 0:
                # For intersection: use minimum weight for each common neighbor
                intersection_weight = sum(min(query_neighbors_weighted.get(n, 0), target_neighbors_weighted.get(n, 0)) for n in shared_nodes)
                
                # For union: sum all weights, avoiding double counting
                union_weight = sum(max(query_neighbors_weighted.get(n, 0), target_neighbors_weighted.get(n, 0)) for n in total_nodes)
                
                return intersection_weight / union_weight if union_weight > 0 else 0.0
            else:
                return 0.0
        
        elif metric == 'cn':
            # Common Neighbors
            return float(len(shared_nodes))
            
        elif metric == 'cos':
            # Cosine Similarity
            denominator = math.sqrt(len(query_neighbors) * len(target_neighbors))
            return len(shared_nodes) / denominator if denominator > 0 else 0.0
            
        elif metric == 'si':
            # Sorensen index
            denominator = len(query_neighbors) + len(target_neighbors)
            return (2 * len(shared_nodes)) / denominator if denominator > 0 else 0.0
            
        elif metric == 'hpi':
            # Hub Promoted Index
            min_degree = min(len(query_neighbors), len(target_neighbors))
            return len(shared_nodes) / min_degree if min_degree > 0 else 0.0
            
        elif metric == 'hdi':
            # Hub Depressed Index
            max_degree = max(len(query_neighbors), len(target_neighbors))
            return len(shared_nodes) / max_degree if max_degree > 0 else 0.0
            
        elif metric == 'lhn':
            # Leicht–Holme–Newman Index
            denominator = len(query_neighbors) * len(target_neighbors)
            return len(shared_nodes) / denominator if denominator > 0 else 0.0
            
        elif metric == 'pa':
            # Preferential Attachment
            return float(len(query_neighbors) * len(target_neighbors))
            
        elif metric == 'ra':
            # Resource Allocation Index (unweighted)
            value = 0.0
            for common_neighbor in shared_nodes:
                neighbor_degree = graph.degree(common_neighbor)
                if neighbor_degree > 0:
                    value += 1.0 / neighbor_degree
            return value
            
        elif metric.startswith('ra_weighted_'):
            # Weighted Resource Allocation Index with different combination modes
            mode = metric.replace('ra_weighted_', '')

            return self._calculate_weighted_ra(graph, query_id, target_id, mode, weight_attr='hybrid_similarity')
            
        elif metric == 'cosimrank':
            # Weighted CoSimRank
            return self.weighted_cosimrank(graph, query_id, target_id)

        else:
            available_metrics = ['jaccard', 'adamic_adar', 'weighted_jaccard', 'cn', 'cos', 'si', 'hpi', 'hdi', 'lhn', 'pa', 'ra', 'ra_weighted_sum', 'ra_weighted_avg', 'ra_weighted_min', 'ra_weighted_max', 'ra_weighted_product', 'cosimrank']
            raise ValueError(f"Unknown metric: {metric}. Available options: {available_metrics}")

    def rerank_by_composite_metric(self, updated_graph: nx.Graph, 
                                  primary_metric: str = 'jaccard',
                                  secondary_metric: str = 'hungarian_nn',
                                  query_id: Optional[str] = None,
                                  primary_weight: float = 0.5,
                                  secondary_weight: float = 0.5) -> Dict[str, Any]:
        """
        Rerank annotations using a composite of two metrics with equal or custom weighting.
        
        Parameters:
        -----------
        updated_graph : nx.Graph
            Graph with query node added (from analyze_annotation_with_graph_integration)
        primary_metric : str, default='jaccard'
            First metric to use. Available options:
            Network: 'jaccard', 'adamic_adar', 'weighted_jaccard', 'cn', 'cos', 'si', 'hpi', 'hdi', 'lhn', 'pa', 'ra', 'sp',
                     'ra_weighted_sum', 'ra_weighted_avg', 'ra_weighted_min', 'ra_weighted_max', 'ra_weighted_product', 'cosimrank'
            Annotation: 'hungarian_nn', 'hybrid_score', 'tanimoto_similarity', 'mcs_score'
        secondary_metric : str, default='hungarian_nn'
            Second metric to use (same options as primary_metric)
        query_id : str, optional
            Query node ID (will use self.query_info if not provided)
        primary_weight : float, default=0.5
            Weight for primary metric (0.0 to 1.0)
        secondary_weight : float, default=0.5
            Weight for secondary metric (0.0 to 1.0)
            
        Returns:
        --------
        dict : Reranking results with composite metric
        """
        if query_id is None:
            if self.query_info is None:
                raise ValueError("No query info available. Provide query_id or call parse_annotation() first.")
            query_id = self.query_info['query_id']
        
        if query_id not in updated_graph.nodes():
            raise ValueError(f"Query node '{query_id}' not found in graph")
        
        # Normalize weights to sum to 1
        total_weight = primary_weight + secondary_weight
        if total_weight == 0:
            raise ValueError("At least one weight must be non-zero")
        primary_weight = primary_weight / total_weight
        secondary_weight = secondary_weight / total_weight
        
        print(f"🌐 Calculating composite metric: {primary_metric} + {secondary_metric}")
        print(f"   Primary weight: {primary_weight:.2f}, Secondary weight: {secondary_weight:.2f}")
        
        # Calculate both metric values for each result
        primary_values = []
        secondary_values = []
        
        for _, row in self.lookup_results.iterrows():
            lookup_key = row['lookup_key']
            
            # Calculate primary metric
            if lookup_key in updated_graph.nodes():
                primary_val = self._calculate_metric(
                    updated_graph, query_id, lookup_key, primary_metric, self.lookup_results
                )
                # Handle distance metrics (convert to similarity for combination)
                if primary_metric == 'sp':
                    primary_val = 1.0 / (1.0 + primary_val) if primary_val != float('inf') else 0.0
            else:
                primary_val = 0.0
            
            # Calculate secondary metric
            if secondary_metric in ['hungarian_nn', 'hybrid_score', 'tanimoto_similarity', 'mcs_score']:
                # Annotation-based metric
                secondary_val = self._calculate_metric(
                    updated_graph, query_id, lookup_key, secondary_metric, self.lookup_results
                )
            else:
                # Network-based metric
                if lookup_key in updated_graph.nodes():
                    secondary_val = self._calculate_metric(
                        updated_graph, query_id, lookup_key, secondary_metric, self.lookup_results
                    )
                    # Handle distance metrics (convert to similarity for combination)
                    if secondary_metric == 'sp':
                        secondary_val = 1.0 / (1.0 + secondary_val) if secondary_val != float('inf') else 0.0
                else:
                    secondary_val = 0.0
            
            primary_values.append(primary_val)
            secondary_values.append(secondary_val)
        
        # Normalize values to [0,1] range for fair combination
        def normalize_values(values):
            """Normalize values to [0,1] range using min-max normalization."""
            if len(values) == 0:
                return values
            
            # Handle special cases
            finite_values = [v for v in values if not (math.isinf(v) or math.isnan(v))]
            if len(finite_values) == 0:
                return [0.0] * len(values)
            
            min_val = min(finite_values)
            max_val = max(finite_values)
            
            if max_val == min_val:
                # All values are the same
                return [1.0 if not (math.isinf(v) or math.isnan(v)) else 0.0 for v in values]
            
            # Min-max normalization
            normalized = []
            for v in values:
                if math.isinf(v) or math.isnan(v):
                    normalized.append(0.0)
                else:
                    normalized.append((v - min_val) / (max_val - min_val))
            
            return normalized
        
        # Normalize both sets of values
        normalized_primary = normalize_values(primary_values)
        normalized_secondary = normalize_values(secondary_values)
        
        # Combine the normalized values
        composite_values = []
        for i in range(len(normalized_primary)):
            composite_score = (primary_weight * normalized_primary[i] + 
                             secondary_weight * normalized_secondary[i])
            composite_values.append(composite_score)
        
        print(f"   ✅ Primary metric ({primary_metric}) range: {min(primary_values):.4f} - {max(primary_values):.4f}")
        print(f"   ✅ Secondary metric ({secondary_metric}) range: {min(secondary_values):.4f} - {max(secondary_values):.4f}")
        print(f"   ✅ Composite metric range: {min(composite_values):.4f} - {max(composite_values):.4f}")
        
        # Rerank using the composite values (higher values = better)
        composite_name = f"{primary_metric}_{secondary_metric}_composite"
        return self.rerank_annotations(
            ranking_values=composite_values,
            ranking_name=composite_name,
            ascending=False  # Higher composite values = better ranks
        )


# Utility functions for batch processing
def load_annotation_results(results_path: str) -> List[Tuple[str, str, pd.DataFrame]]:
    """
    Load annotation results from pickle file.
    
    Parameters:
    -----------
    results_path : str
        Path to the pickle file containing annotation results
        
    Returns:
    --------
    list : List of annotation tuples
    """
    with open(results_path, 'rb') as f:
        annotations = pickle.load(f)
    
    print(f"Loaded {len(annotations)} annotation results from {results_path}")
    return annotations
