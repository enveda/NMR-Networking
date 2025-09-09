import pickle
import numpy as np
from graph.graph_utils import GraphBuilder
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import networkx as nx

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
similarity_threshold = 35

print(f"Calculating distances from query to {len(lookup_hsqc_data)} compounds...")

for compound_id, hsqc_array in lookup_hsqc_data.items():
    # Skip if compound not in graph
    if compound_id not in graph_builder.graph:
        continue

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

# Print node information
print(f"\n=== Subgraph Node Information ===")
print(f"Total nodes in subgraph: {subgraph.number_of_nodes()}")
print(f"Query node: {query_id}")
print(f"Query SMILES: {exp_smiles}")
print("\nAll nodes in subgraph:")
for node in subgraph.nodes():
    node_data = subgraph.nodes[node]
    smiles = node_data.get('smiles', 'Unknown')
    node_type = node_data.get('node_type', 'Unknown')
    print(f"  {node}: {smiles} (Type: {node_type})")

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

CUSTOM_PALETTE = {
    'query':    '#E24A33',  # vibrant red
    'one_hop':  '#348ABD',  # deep blue
    'two_hop':  '#988ED5',  # muted purple
    'edge_q':   '#E24A33',
    'edge_oth': '#BBBBBB',
}

# assume query_id, updated_graph, subgraph are already in scope

# 1) Compute a more "spread‐out" layout with increased spacing
# pos = nx.spring_layout(
#     subgraph,
#     k=50.0,          # increased from 2.5 for more spacing
#     iterations=1000, # more iterations for stability
#     seed=103,
#     scale=100,
# )
pos = nx.kamada_kawai_layout(subgraph, scale=15)

# 2) Classify nodes & sizes
node_colors = []
node_sizes  = []
for n in subgraph.nodes():
    if n == query_id:
        node_colors.append(CUSTOM_PALETTE['query'])
        node_sizes.append(600)
    elif n in updated_graph.neighbors(query_id):
        node_colors.append(CUSTOM_PALETTE['one_hop'])
        node_sizes.append(350)
    else:
        node_colors.append(CUSTOM_PALETTE['two_hop'])
        node_sizes.append(200)

# 3) Classify edges with darker colors and varied widths
edge_colors = []
edge_widths = []
edge_alphas = []
for u, v, data in subgraph.edges(data=True):
    if query_id in (u, v):
        edge_colors.append('#B91C1C')  # darker red
        edge_widths.append(3.5)
        edge_alphas.append(0.8)
    else:
        edge_colors.append('#374151')  # dark gray
        edge_widths.append(2.0)
        edge_alphas.append(0.6)

# 4) Plot setup
plt.figure(figsize=(10, 10))  # larger figure for better spacing
plt.rcParams.update({
    'font.family':    'sans-serif',
    'font.size':      11,
    'axes.linewidth': 0.8,
})
ax = plt.gca()
ax.set_facecolor('white')

# 5) Draw edges with improved styling and feature labels
edge_labels = {}
for i, (u, v, data) in enumerate(subgraph.edges(data=True)):
    x_coords = [pos[u][0], pos[v][0]]
    y_coords = [pos[u][1], pos[v][1]]
    
    # Get edge features
    hungarian_dist = data.get('weight', 0)
    hybrid_sim = data.get('hybrid_similarity', 0)
    
    # Create edge label with features
    if hungarian_dist > 0 or hybrid_sim > 0:
        edge_labels[(u, v)] = f"H:{hungarian_dist:.1f}\n\nS:{hybrid_sim:.2f}"
    
    # Draw edge with shadow effect
    ax.plot(x_coords, y_coords, 
           color='black', 
           linewidth=edge_widths[i] + 1, 
           alpha=0.2, 
           zorder=1)  # shadow
    
    ax.plot(x_coords, y_coords, 
           color=edge_colors[i], 
           linewidth=edge_widths[i], 
           alpha=edge_alphas[i], 
           zorder=2)  # main edge

# Draw nodes using matplotlib scatter for proper zorder control
for i, node in enumerate(subgraph.nodes()):
    x, y = pos[node]
    color = node_colors[i]
    size = node_sizes[i]
    
    # Draw node shadow
    ax.scatter(x, y, s=size*1.8, c='black', alpha=0.3)
    # Draw main node
    ax.scatter(x, y, s=size, c=color, edgecolors='k', linewidth=1)
    # if node != 'QUERY':
    try:
        print(node, MNOVA_dict[node]['smiles'][0])
    except:
        print(node, 'No SMILES')

# 6) Draw edge labels with features
nx.draw_networkx_edge_labels(
    subgraph, pos,
    edge_labels=edge_labels,
    font_size=6,
    font_color='darkred',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray'),
)

# 7) Add legend for edge features
legend_elements = [
    Line2D([0], [0], color='#B91C1C', lw=3.5, label='Query Connections'),
    Line2D([0], [0], color='#374151', lw=2.0, label='Other Connections'),
    Line2D([0], [0], color='none', label='\n H : Hungarian Distance'),
    Line2D([0], [0], color='none', label='\n S: Hybrid Similarity')
]

# plt.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

# 8) Explicitly label **all** nodes by their identifier
labels = {n: str(n) for n in subgraph.nodes()}
nx.draw_networkx_labels(
    subgraph, pos,
    labels=labels,
    font_size=9,
    font_weight='bold',
)

# 7) No legend or title per prior request
plt.axis('off')
plt.tight_layout()

# 8) Save
output_path = '/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/figures/query_2hop_pub.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {output_path}")