import pickle
import numpy as np
from graph.graph_utils import GraphBuilder
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem import DataStructs
import matplotlib.pyplot as plt
import networkx as nx
import sys
import os
import pygraphviz as pgv
import pandas as pd
from molecular_graph_edit.molecular_similarity import mcs_sim_largest_connected as mcs_sim_d

from Similarity import DISTANCE_FUNCTION_MAP, DISTANCE_FUNCTION_PARAMS

root = '/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/annotation_reranking_results'
experimental_dict = pickle.load(open('/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/ExperimentalHSQC_Lookup.pkl', 'rb'))
MNOVA_dict = '/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/MNova_lookup_subset.pkl'
MNOVA_dict = pickle.load(open(MNOVA_dict, 'rb'))
RERANKED_TOP5 = False

graph = '/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/storage/Graph_func_low_dist_25_structure_similarity_filtered_06hybrid.pkl'

# Load the graph
graph_builder = GraphBuilder()
graph_builder.load_graph(graph)

# Load query data from enveda file (singular HSQC spectrum input)
file  = '/home/cailum.stienstra/HSQC_Models/enveda_data/Processed/ISLT-100_Tube17007003750_ HSQC.pkl'
data = pickle.load(open(file, 'rb'))
query_HSQC       = data['spectrum'][0][:, :2][:, ::-1]
print(f"Query HSQC shape: {query_HSQC.shape}")
exp_mol    = data['mol'][0]
exp_smiles = Chem.MolToSmiles(exp_mol)
print(f"Query SMILES: {exp_smiles}")

# Convert query HSQC to numpy array format expected by distance functions
query_hsqc_array = np.array(query_HSQC)

# Prepare lookup HSQC data from MNOVA_dict
lookup_hsqc_data = {}
print("Processing MNOVA_dict for lookup data...")
for lookup_key, lookup_data in MNOVA_dict.items():
    try:
        hsqc_peaks = lookup_data['gt'][0]  # Get HSQC peaks
        lookup_hsqc_data[lookup_key] = np.array(hsqc_peaks)
    except Exception as e:
        continue

print(f"Loaded HSQC lookup data for {len(lookup_hsqc_data)} compounds")

# Add query to graph manually (using approach from case study)
query_id = "QUERY"

# Add query node to the main graph
graph_builder.graph.add_node(query_id, node_type='query', smiles=exp_smiles)

# Calculate distances and add edges manually
distance_function = 'modified_hungarian'
dist_func = DISTANCE_FUNCTION_MAP[distance_function]
dist_params = DISTANCE_FUNCTION_PARAMS[distance_function].copy()
edges_added = 0
similarity_threshold = 35.0

print(f"Calculating distances from query to compounds in graph...")

for compound_id in graph_builder.graph.nodes():
    # Skip the query node itself if it's already in the graph
    if compound_id == query_id:
        continue
        
    # Skip if no lookup HSQC data available for this compound
    if compound_id not in lookup_hsqc_data:
        continue
        
    hsqc_array = lookup_hsqc_data[compound_id]

    try:
        # Calculate distance
        if len(dist_params) == 0:
            distance = dist_func(query_hsqc_array, hsqc_array)
        else:
            distance = dist_func(query_hsqc_array, hsqc_array, **dist_params)
        
        # Extract distance value if it's a tuple
        if isinstance(distance, tuple):
            distance = distance[0]
        
        # Add edge if within threshold
        if distance <= similarity_threshold:
            graph_builder.graph.add_edge(query_id, compound_id, 
                                        weight=distance, 
                                        hungarian_distance=distance, 
                                        edge_type='query_connection')
            edges_added += 1
            
    except Exception as e:
        continue

print(f"Query added to graph with {edges_added} connections")

# Collect all distances for ranking (not just those within threshold)
print("Collecting all distances for ranking...")
all_distances = []
for compound_id in graph_builder.graph.nodes():
    # Skip the query node itself if it's already in the graph
    if compound_id == query_id:
        continue
        
    # Skip if no lookup HSQC data available for this compound
    if compound_id not in lookup_hsqc_data:
        continue
        
    hsqc_array = lookup_hsqc_data[compound_id]

    try:
        # Calculate distance
        if len(dist_params) == 0:
            distance = dist_func(query_hsqc_array, hsqc_array)
        else:
            distance = dist_func(query_hsqc_array, hsqc_array, **dist_params)
        
        # Extract distance value if it's a tuple
        if isinstance(distance, tuple):
            distance = distance[0]
        
        # Store all distances for ranking
        compound_smiles = graph_builder.graph.nodes[compound_id].get('smiles', 'Unknown')
        all_distances.append({
            'compound_id': compound_id,
            'smiles': compound_smiles,
            'distance': distance,
            'within_threshold': distance <= similarity_threshold
        })
            
    except Exception as e:
        continue

# Sort by distance and get top 100
all_distances.sort(key=lambda x: x['distance'])
top_100_ranked = all_distances[:2000]

# Get all one-hop and two-hop neighbors from the graph
print("Identifying one-hop and two-hop neighbors...")
one_hop_neighbors = set(graph_builder.graph.neighbors(query_id))
two_hop_neighbors = set()
for neighbor in one_hop_neighbors:
    two_hop_neighbors.update(graph_builder.graph.neighbors(neighbor))
two_hop_neighbors -= one_hop_neighbors  # Remove one-hop neighbors from two-hop set
two_hop_neighbors.discard(query_id)  # Remove query node

all_graph_neighbors = one_hop_neighbors.union(two_hop_neighbors)
print(f"Found {len(one_hop_neighbors)} one-hop neighbors and {len(two_hop_neighbors)} two-hop neighbors")

# Ensure all graph neighbors are in the top 100 list
existing_compound_ids = {item['compound_id'] for item in top_100_ranked}
missing_neighbors = all_graph_neighbors - existing_compound_ids

if missing_neighbors:
    print(f"Adding {len(missing_neighbors)} missing graph neighbors to the list...")
    print(f"Using distance function: {distance_function}")
    if dist_params:
        print(f"Distance parameters: {dist_params}")
    
    # Get distance data for missing neighbors
    missing_distances = []
    for compound_id in missing_neighbors:
        if compound_id in lookup_hsqc_data:
            try:
                hsqc_array = lookup_hsqc_data[compound_id]
                
                # Calculate distance using the same function and parameters as original ranking
                if len(dist_params) == 0:
                    distance = dist_func(query_hsqc_array, hsqc_array)
                else:
                    distance = dist_func(query_hsqc_array, hsqc_array, **dist_params)
                
                # Extract distance value if it's a tuple
                if isinstance(distance, tuple):
                    distance = distance[0]
                
                # Get SMILES from graph node
                compound_smiles = graph_builder.graph.nodes[compound_id].get('smiles', 'Unknown')
                
                missing_distances.append({
                    'compound_id': compound_id,
                    'smiles': compound_smiles,
                    'distance': distance,
                    'within_threshold': distance <= similarity_threshold,
                    'is_graph_neighbor': True
                })
                
                # Debug: Print first few calculations to verify consistency
                if len(missing_distances) <= 3:
                    print(f"  {compound_id}: distance = {distance:.3f}")
                    
            except Exception as e:
                print(f"Error calculating distance for missing neighbor {compound_id}: {e}")
                continue
    
    # Add missing neighbors to the list
top_100_ranked.extend(missing_distances)
print(f"Added {len(missing_distances)} missing neighbors with consistent distance metric")

# Verify distance metric consistency by re-calculating a few original compounds
if len(top_100_ranked) > 0:
    print("Verifying distance metric consistency...")
    verification_count = min(3, len(top_100_ranked))
    for i in range(verification_count):
        item = top_100_ranked[i]
        compound_id = item['compound_id']
        if compound_id in lookup_hsqc_data:
            hsqc_array = lookup_hsqc_data[compound_id]
            
            # Re-calculate distance using same metric
            if len(dist_params) == 0:
                recalculated_distance = dist_func(query_hsqc_array, hsqc_array)
            else:
                recalculated_distance = dist_func(query_hsqc_array, hsqc_array, **dist_params)
            
            if isinstance(recalculated_distance, tuple):
                recalculated_distance = recalculated_distance[0]
            
            # Compare with stored distance
            original_distance = item['distance']
            if abs(recalculated_distance - original_distance) < 1e-10:
                print(f"  ✓ {compound_id}: distance consistent ({original_distance:.3f})")
            else:
                print(f"  ✗ {compound_id}: distance mismatch! Original: {original_distance:.3f}, Recalculated: {recalculated_distance:.3f}")

# Calculate additional similarity scores for all compounds in the list
print("Calculating MCS, Tanimoto, and hybrid similarity scores...")

# Get query fingerprint for Tanimoto calculations
exp_fp = AllChem.GetMorganFingerprintAsBitVect(exp_mol, 2, 2048)
save_top_100_ranked = False
if save_top_100_ranked:
    for item in top_100_ranked:
        # Get compound SMILES from MNOVA dict using compound_id
        compound_id = item['compound_id']
        compound_smiles = MNOVA_dict[compound_id]['smiles'][0]
        compound_mol = Chem.MolFromSmiles(compound_smiles)
        
        # Calculate MCS similarity
        mcs_score = mcs_sim_d(
            exp_smiles,
            compound_smiles,
            num_tries=10,
            num_keep_m=10,
            num_keep_max_m=2,
        )
        
        # Calculate Tanimoto similarity
        compound_fp = AllChem.GetMorganFingerprintAsBitVect(compound_mol, 2, 2048)
        tanimoto_similarity = DataStructs.TanimotoSimilarity(exp_fp, compound_fp)
        
        # Calculate hybrid score (average of MCS and Tanimoto)
        hybrid_score = 0.5 * (mcs_score + tanimoto_similarity)
        
        # Add scores to the item
        item['mcs_score'] = mcs_score
        item['tanimoto_similarity'] = tanimoto_similarity
        item['hybrid_score'] = hybrid_score

# Create DataFrame and save to CSV
if save_top_100_ranked:
    df_top_100 = pd.DataFrame(top_100_ranked).sort_values(by='distance', ascending=True)
    csv_output_path = '/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/figures/top_100_ranked_compounds.csv'
    df_top_100.to_csv(csv_output_path, index=False)
    print(f"Top 100 ranked compounds with similarity scores saved to: {csv_output_path}")

# Print top 10 for verification
    print(f"\n=== Top 10 Ranked Compounds (with similarity scores) ===")
    for i, item in enumerate(top_100_ranked[:10]):
        print(f"{i+1:2d}. {item['compound_id']:15s} | Distance: {item['distance']:.3f} | MCS: {item['mcs_score']:.3f} | Tanimoto: {item['tanimoto_similarity']:.3f} | Hybrid: {item['hybrid_score']:.3f}")

# Get the updated graph
updated_graph = graph_builder.graph

# Extract 2-hop neighbors around the query
def get_n_hop_subgraph(graph, center_node, n_hops):
    """Extract n-hop neighborhood subgraph around a center node."""
    if center_node not in graph:
        raise ValueError(f"Node {center_node} not found in graph")
    
    # Use BFS to find all nodes within n hops
    visited = set()
    current_level = {center_node}
    visited.add(center_node)
    
    for hop in range(n_hops):
        next_level = set()
        for node in current_level:
            neighbors = set(graph.neighbors(node))
            next_level.update(neighbors - visited)
            visited.update(neighbors)
        current_level = next_level
        
        if not current_level:  # No more nodes to explore
            break
     
    return graph.subgraph(visited).copy()

# Extract 2-hop subgraph
print("Extracting 2-hop subgraph...")
subgraph = get_n_hop_subgraph(updated_graph, query_id, n_hops=2)

# Print detailed subgraph node information
print(f"\n=== DETAILED SUBGRAPH NODE INFORMATION ===")
print(f"Total nodes in subgraph: {subgraph.number_of_nodes()}")
print(f"Total edges in subgraph: {subgraph.number_of_edges()}")
print(f"Query node: {query_id}")
print(f"Query SMILES: {exp_smiles}")
print("=" * 80)

# Print detailed information for each node in the subgraph
for i, node in enumerate(subgraph.nodes(), 1):
    node_data = subgraph.nodes[node]
    print(f"\nNode {i:3d}: {node}")
    print(f"  Node attributes:")
    for attr, value in node_data.items():
        print(f"    {attr}: {value}")
    
    # Determine node type based on distance from query
    if node == query_id:
        node_category = "QUERY NODE"
    elif node in updated_graph.neighbors(query_id):
        node_category = "1-HOP NEIGHBOR"
    else:
        node_category = "2-HOP NEIGHBOR"
    
    print(f"  Node category: {node_category}")
    
    # Print edge information for this node within subgraph
    neighbors = list(subgraph.neighbors(node))
    print(f"  Number of neighbors in subgraph: {len(neighbors)}")
    if neighbors:
        print(f"  Neighbors in subgraph: {neighbors}")
        
        # Print all edge attributes for connections within subgraph
        print(f"  Edge connections within subgraph:")
        for neighbor in neighbors:
            edge_data = subgraph.get_edge_data(node, neighbor)
            if edge_data:
                print(f"    {node} -> {neighbor}:")
                for attr, value in edge_data.items():
                    print(f"      {attr}: {value}")
            else:
                print(f"    {node} -> {neighbor}: No edge data")
    
    # If this is not the query node, show distance to query
    if node != query_id:
        try:
            # Get distance to query from the updated graph
            if node in updated_graph.neighbors(query_id):
                distance_to_query = updated_graph[query_id][node]['weight']
                print(f"  Distance to query: {distance_to_query:.3f}")
            else:
                print(f"  Distance to query: Not directly connected")
        except:
            print(f"  Distance to query: Unknown")
    
    print("-" * 60)

print("=" * 80)
print("END OF SUBGRAPH NODE INFORMATION")
print("=" * 80)

# Print all edges in the subgraph with their complete labels
print(f"\n=== ALL EDGES IN SUBGRAPH WITH COMPLETE LABELS ===")
print(f"Total edges: {subgraph.number_of_edges()}")
print("=" * 80)

for i, (u, v, data) in enumerate(subgraph.edges(data=True), 1):
    print(f"\nEdge {i:3d}: {u} -> {v}")
    print(f"  Edge attributes:")
    for attr, value in data.items():
        print(f"    {attr}: {value}")
    print("-" * 40)

print("=" * 80)
print("END OF SUBGRAPH EDGE INFORMATION")
print("=" * 80)

print(f"\n=== Subgraph Statistics ===")
print(f"Query node: {query_id}")
print(f"Query SMILES: {exp_smiles}")
print(f"Total nodes in subgraph: {subgraph.number_of_nodes()}")
print(f"Total edges in subgraph: {subgraph.number_of_edges()}")
print(f"Direct neighbors of query: {len(list(updated_graph.neighbors(query_id)))}")

# Show edge weights for direct connections to query
print(f"\n=== Direct Connections to Query (Top 10) ===")
query_edges = [(n, d['weight']) for n, d in updated_graph[query_id].items()]
query_edges.sort(key=lambda x: x[1])  # Sort by weight (distance)
for neighbor, weight in query_edges[:10]:
    neighbor_smiles = updated_graph.nodes[neighbor].get('smiles', 'Unknown')
    print(f"  {neighbor}: distance = {weight:.3f}, SMILES = {neighbor_smiles}")

# Create pygraphviz graph
print("\nCreating pygraphviz visualization...")
G = pgv.AGraph(directed=False)

# Set graph attributes
G.graph_attr.update({
    'rankdir': 'TB',  # Top to bottom layout for hierarchical positioning
    'splines': 'polyline',  # Curved edges
    'overlap': 'false',
    'sep': '+120,120',  # Tighter node separation for closer neighbors
    'nodesep': '1.2',  # Much tighter node separation
    'ranksep': '1.7',  # Much tighter rank separation
    'fontsize': '14',
    'fontname': 'Arial',
    'concentrate': 'true',  # Concentrate edges for cleaner layout
    'compound': 'true',  # Enable compound nodes for better grouping
    'pack': 'true',  # Pack nodes closer together
    'packmode': 'clust',  # Use cluster packing mode
})


# Add nodes with styling

c = 1
label_dict = {}
for node in subgraph.nodes():
    node_data = subgraph.nodes[node]
    if node == query_id:
        # Query node styling
        G.add_node(node, 
                  shape='circle',
                  style='filled',
                  fillcolor='#E24A33',
                  color='#B91C1C',
                  penwidth='3',
                  fontsize='32',
                  fontweight='bold',
                  label=f'QUERY')
    elif node in updated_graph.neighbors(query_id):
        # 1-hop neighbors
        G.add_node(node,
                  shape='circle',
                  size='32',
                  style='filled',
                  fillcolor='#FFA500',
                  color='#FF8C00',
                  penwidth='2',
                  fontsize='32',
                  label=f'{c}, {node}')

    else:
        # 2-hop neighbors
        G.add_node(node,
                  shape='circle',
                  style='filled',
                  size='32',
                  fillcolor='#87CEEB',
                  color='#4682B4',
                  penwidth='1.5',
                  fontsize='32',
                  label=f'{c}, {node}')
    label_dict[str] = node
    c += 1

# Position one-hop neighbors close to query using edge weights and lengths
# (No subgraph/cluster to avoid boxes around nodes)

# Add edges with feature labels
for u, v, data in subgraph.edges(data=True):
    hungarian_dist = data.get('weight', 0)
    hybrid_sim = data.get('hybrid_similarity', 0)
    
    # Create edge label with better formatting
    edge_label = f"H:{hungarian_dist:.1f}\nS:{hybrid_sim:.2f}"
    
    if query_id in (u, v):
        # Query connections - position one-hop neighbors close to query
        G.add_edge(u, v,
                  color='#B91C1C',
                  penwidth='3',
                  fontsize='32',
                  fontcolor='#B91C1C',
                  edge_label=edge_label,
                  len='2.5',
                  weight='3.0')  # Higher weight to keep very close
    else:
        # Other connections - keep all neighbors close together
        G.add_edge(u, v,
                  color='#374151',
                  penwidth='2',
                  fontsize='32',
                  fontcolor='#374151',
                  edge_label=edge_label,
                  len='3.0',
                  weight='2.0')  # Higher weight to keep neighbors close

# Layout and render
print("Computing layout...")
# Use neato layout algorithm for compact, force-directed positioning
G.layout(prog='neato')

# Save the graph.
output_path = '/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/figures/query_2hop_subgraph_pygraphviz.png'
G.draw(output_path, format='png')
print(f"PyGraphviz visualization saved to: {output_path}")

# Also save as SVG for vector format
# svg_output_path = '/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/figures/query_2hop_subgraph_pygraphviz.svg'
# G.draw(svg_output_path, format='svg')
# print(f"PyGraphviz SVG saved to: {svg_output_path}")
# Print some statistics about the pygraphviz graph
print(f"\n=== PyGraphviz Graph Statistics ===")
print(f"Nodes: {len(G.nodes())}")
print(f"Edges: {len(G.edges())}")
print(f"Layout algorithm: neato")
print(f"Graph attributes: {G.graph_attr}")