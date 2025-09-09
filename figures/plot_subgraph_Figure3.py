import pickle
import numpy as np
from graph_utils import GraphBuilder
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdFMCS
from rdkit.Chem import DataStructs
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
import networkx as nx
import sys
import os
import pygraphviz as pgv
import random
import json
from rdkit.Chem import rdMolDescriptors

# New imports for font handling
from matplotlib import font_manager
from urllib.request import urlopen


from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# ---------------- Font utilities (mirroring Figure 2) ---------------- #

def _register_downloaded_font(font_path: str) -> None:
    try:
        font_manager.fontManager.addfont(font_path)
        font_manager._rebuild()
    except Exception:
        pass


def ensure_nimbus_sans() -> str:
    preferred_families = ["Nimbus Sans", "Nimbus Sans L"]
    available_names = {f.name for f in font_manager.fontManager.ttflist}
    for fam in preferred_families:
        if fam in available_names:
            return fam

    fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts', 'nimbus_sans')
    os.makedirs(fonts_dir, exist_ok=True)

    font_files = {
        'NimbusSans-Regular.otf': 'https://github.com/ArtifexSoftware/urw-base35-fonts/raw/master/fonts/NimbusSans-Regular.otf',
        'NimbusSans-Bold.otf': 'https://github.com/ArtifexSoftware/urw-base35-fonts/raw/master/fonts/NimbusSans-Bold.otf',
        'NimbusSans-Italic.otf': 'https://github.com/ArtifexSoftware/urw-base35-fonts/raw/master/fonts/NimbusSans-Italic.otf',
        'NimbusSans-BoldItalic.otf': 'https://github.com/ArtifexSoftware/urw-base35-fonts/raw/master/fonts/NimbusSans-BoldItalic.otf',
    }

    for fname, url in font_files.items():
        local_path = os.path.join(fonts_dir, fname)
        if not os.path.exists(local_path):
            try:
                with urlopen(url, timeout=15) as resp:
                    data = resp.read()
                with open(local_path, 'wb') as f:
                    f.write(data)
            except Exception:
                continue
        _register_downloaded_font(local_path)

    available_names = {f.name for f in font_manager.fontManager.ttflist}
    for fam in preferred_families:
        if fam in available_names:
            return fam
    return 'DejaVu Sans'


def set_global_font():
    family = ensure_nimbus_sans()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [family, 'Nimbus Sans L', 'Nimbus Sans', 'DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = family
    plt.rcParams['mathtext.it'] = family
    plt.rcParams['mathtext.bf'] = family
    plt.rcParams['mathtext.sf'] = family
    plt.rcParams['mathtext.default'] = 'regular'
    return family


def apply_graphviz_fonts(G: pgv.AGraph, family: str) -> None:
    try:
        G.graph_attr.update({'fontname': family})
        G.node_attr.update({'fontname': family})
        G.edge_attr.update({'fontname': family})
    except Exception:
        pass



def calculate_mass_difference(smiles1, smiles2):
    """Return the monoisotopic mass difference between two SMILES."""
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES string(s) provided.")
    mass1 = rdMolDescriptors.CalcExactMolWt(mol1)
    mass2 = rdMolDescriptors.CalcExactMolWt(mol2)
    return abs(mass1 - mass2)



# from Similarity import DISTANCE_FUNCTION_MAP, DISTANCE_FUNCTION_PARAMS

root = '/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/annotation_reranking_results'
experimental_dict = pickle.load(open('/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/ExperimentalHSQC_Lookup.pkl', 'rb'))
MNOVA_dict = '/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/MNova_lookup_subset.pkl'
MNOVA_dict = pickle.load(open(MNOVA_dict, 'rb'))

# Load the MNOVA lookup dictionary for SMILES
mnova_lookup = pickle.load(open('/home/cailum.stienstra/HSQC_Models/Networking_HSQC/lookup_dict_MNova.pkl', 'rb'))

RERANKED_TOP5 = False

graph = '/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/storage/Graph_func_low_dist_25_structure_similarity_filtered_06hybrid.pkl'

# Load the graph
graph_builder = GraphBuilder()
graph_builder.load_graph(graph)

# Get the main graph
main_graph = graph_builder.graph

# Ensure fonts are set (Matplotlib side)
family = set_global_font()

# Randomly select a node from the graph with retry logic
print("Randomly selecting a node from the graph...")
all_nodes = list(main_graph.nodes())
max_attempts = 50  # Maximum number of attempts to find a suitable node
attempt = 0

while attempt < max_attempts:
    selected_node = random.choice(all_nodes)
    print(f"Attempt {attempt + 1}: Selected node: {selected_node}")
    
    # Get node information using MNOVA lookup
    selected_smiles = "Unknown"
    if selected_node in mnova_lookup:
        selected_smiles = mnova_lookup[selected_node]['smiles'][0]
    selected_node_type = main_graph.nodes[selected_node].get('node_type', 'Unknown')
    print(f"Selected node SMILES: {selected_smiles}")
    print(f"Selected node type: {selected_node_type}")

    # Extract 2-hop neighbors around the selected node
    def get_n_hop_subgraph(graph, center_node, n_hops, max_nodes=15, max_edges=30, min_nodes=8):
        """Extract n-hop neighborhood subgraph around a center node with size constraints."""
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
        
        # Create subgraph
        subgraph = graph.subgraph(visited).copy()
        
        # If subgraph is too large or too small, adjust it
        if subgraph.number_of_nodes() > max_nodes or subgraph.number_of_edges() > max_edges or subgraph.number_of_nodes() < min_nodes:
            print(f"Subgraph size ({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges) needs adjustment...")
            
            # Get 1-hop neighbors first
            one_hop_neighbors = set(main_graph.neighbors(center_node))
            one_hop_neighbors.add(center_node)
            
            # Add some 2-hop neighbors if space allows
            two_hop_candidates = visited - one_hop_neighbors
            two_hop_candidates = list(two_hop_candidates)
            
            # Try to add 2-hop nodes one by one until we hit the limit
            final_nodes = one_hop_neighbors.copy()
            
            # Ensure we have at least min_nodes
            if len(final_nodes) < min_nodes:
                # Add more 2-hop nodes to reach minimum
                for candidate in two_hop_candidates:
                    if len(final_nodes) >= min_nodes:
                        break
                    final_nodes.add(candidate)
            
            # If still too small, try to add more nodes even if it exceeds max_nodes
            if len(final_nodes) < min_nodes:
                print(f"Warning: Could not reach minimum {min_nodes} nodes. Current: {len(final_nodes)}")
            else:
                # Now try to stay within max_nodes and max_edges
                for candidate in two_hop_candidates:
                    if candidate in final_nodes:
                        continue
                        
                    temp_nodes = final_nodes.copy()
                    temp_nodes.add(candidate)
                    temp_subgraph = main_graph.subgraph(temp_nodes).copy()
                    
                    if temp_subgraph.number_of_nodes() <= max_nodes and temp_subgraph.number_of_edges() <= max_edges:
                        final_nodes = temp_nodes
                    else:
                        break
            
            subgraph = main_graph.subgraph(final_nodes).copy()
        
        return subgraph

    # Extract 2-hop subgraph with constraints
    print("Extracting 2-hop subgraph with size constraints...")
    subgraph = get_n_hop_subgraph(main_graph, selected_node, n_hops=2, max_nodes=20, max_edges=40, min_nodes=10)
    
    # Check if subgraph size is acceptable (between 10 and 20 nodes)
    num_nodes = subgraph.number_of_nodes()
    if 8 <= num_nodes <= 20:
        print(f"Found suitable subgraph with {num_nodes} nodes!")
        break
    else:
        print(f"Subgraph has {num_nodes} nodes (not in range 10-20). Trying another node...")
        attempt += 1

if attempt >= max_attempts:
    print(f"Warning: Could not find a suitable node after {max_attempts} attempts.")
    print(f"Using the last selected node with {subgraph.number_of_nodes()} nodes.")
else:
    print(f"Successfully found suitable subgraph after {attempt + 1} attempts.")

# Create node index mapping
node_list = list(subgraph.nodes())
node_to_index = {node: idx for idx, node in enumerate(node_list)}
index_to_node = {idx: node for node, idx in node_to_index.items()}

# Print node information
print(f"\n=== Subgraph Node Information ===")
print(f"Total nodes in subgraph: {subgraph.number_of_nodes()}")
print(f"Selected node: {selected_node} (Index: {node_to_index[selected_node]})")
print(f"Selected SMILES: {selected_smiles}")
print("\nAll nodes in subgraph:")
for node in subgraph.nodes():
    node_idx = node_to_index[node]
    smiles = "Unknown"
    if node in mnova_lookup:
        smiles = mnova_lookup[node]['smiles'][0]
    node_type = subgraph.nodes[node].get('node_type', 'Unknown')
    print(f"  Index {node_idx}: {node} - {smiles} (Type: {node_type})")

print(f"\n=== Subgraph Statistics ===")
print(f"Selected node: {selected_node} (Index: {node_to_index[selected_node]})")
print(f"Selected SMILES: {selected_smiles}")
print(f"Total nodes in subgraph: {subgraph.number_of_nodes()}")
print(f"Total edges in subgraph: {subgraph.number_of_edges()}")
print(f"Direct neighbors of selected node: {len(list(main_graph.neighbors(selected_node)))}")

# Show edge weights for direct connections to selected node
print(f"\n=== Direct Connections to Selected Node (Top 10) ===")
selected_edges = [(n, d['weight']) for n, d in main_graph[selected_node].items() if n in subgraph.nodes()]
selected_edges.sort(key=lambda x: x[1])  # Sort by weight (distance)
for neighbor, weight in selected_edges[:10]:
    neighbor_smiles = "Unknown"
    if neighbor in mnova_lookup:
        neighbor_smiles = mnova_lookup[neighbor]['smiles'][0]
    print(f"  {neighbor} (Index {node_to_index[neighbor]}): distance = {weight:.3f}, SMILES = {neighbor_smiles}")

# Save node and edge metadata to file
print("\nSaving metadata to file...")
metadata = {
    'selected_node': selected_node,
    'selected_node_index': node_to_index[selected_node],
    'selected_smiles': selected_smiles,
    'selected_node_type': selected_node_type,
    'subgraph_stats': {
        'num_nodes': subgraph.number_of_nodes(),
        'num_edges': subgraph.number_of_edges(),
        'direct_neighbors': len(list(main_graph.neighbors(selected_node)))
    },
    'node_index_mapping': node_to_index,
    'nodes': {},
    'edges': []
}

# Save node metadata
for node in subgraph.nodes():
    node_idx = node_to_index[node]
    smiles = "Unknown"
    if node in mnova_lookup:
        smiles = mnova_lookup[node]['smiles'][0]
    node_data = subgraph.nodes[node]
    metadata['nodes'][node] = {
        'index': node_idx,
        'smiles': smiles,
        'node_type': node_data.get('node_type', 'Unknown'),
        'is_selected': node == selected_node,
        'is_direct_neighbor': node in main_graph.neighbors(selected_node)
    }

# Save edge metadata
for u, v, data in subgraph.edges(data=True):
    # Get SMILES for mass difference calculation
    smiles_u = "Unknown"
    smiles_v = "Unknown"
    if u in mnova_lookup:
        smiles_u = mnova_lookup[u]['smiles'][0]
    if v in mnova_lookup:
        smiles_v = mnova_lookup[v]['smiles'][0]
    
    # Calculate mass difference
    mass_diff = None
    if smiles_u != "Unknown" and smiles_v != "Unknown":
        mass_diff = calculate_mass_difference(smiles_u, smiles_v)
    
    edge_info = {
        'source': u,
        'source_index': node_to_index[u],
        'target': v,
        'target_index': node_to_index[v],
        'weight': data.get('weight', 0),
        'hungarian_distance': data.get('hungarian_distance', 0),
        'hybrid_similarity': data.get('hybrid_similarity', 0),
        'edge_type': data.get('edge_type', 'unknown'),
        'involves_selected': selected_node in (u, v),
        'mass_difference': mass_diff,
        'smiles_u': smiles_u,
        'smiles_v': smiles_v
    }
    metadata['edges'].append(edge_info)

# Save metadata to JSON file
metadata_file = '/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/figures/paper_figures/random_subgraph_metadata.json'
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved to: {metadata_file}")

# Create pygraphviz graph
print("\nCreating pygraphviz visualization...")
G = pgv.AGraph(directed=False)

# Apply Nimbus Sans to Graphviz (graph, node, edge)
apply_graphviz_fonts(G, family)

# Ensure Graphviz can locate the downloaded Nimbus Sans fonts
fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts', 'nimbus_sans')
try:
    os.environ['GDFONTPATH'] = fonts_dir
    G.graph_attr.update({'fontpath': fonts_dir})
except Exception:
    pass

# Set graph attributes for better neighborhood emphasis and shorter edges
G.graph_attr.update({
    'rankdir': 'LR',  # Left to right layout for better neighborhood emphasis
    'overlap': 'false',  # Prevent node overlap
    'sep': '+35,35',  # Slightly increased node separation
    'nodesep': '0.7',  # Increased node separation
    'ranksep': '0.8',  # Increased rank separation
    'fontsize': '10',
    'fontname': family,
    'K': '0.8',  # Neato parameter for moderate clustering
    'packmode': 'clust',  # Cluster packing mode
    'concentrate': 'true'  # Concentrate edges to reduce crossings
})

# Color scheme from other figure
COLOR_MOD_HUNG = '#F29E4C'      # Deep orange for Mod-Hung
COLOR_HUNG_NN =  '#9467bd'      # Purple for Hung-NN

# Ensure node and edge defaults also use Nimbus Sans
G.node_attr.update({'fontname': family})
G.edge_attr.update({'fontname': family})

# Add nodes with purple color scheme
for node in subgraph.nodes():
    node_idx = node_to_index[node]
    G.add_node(node, 
              shape='circle',
              style='filled',
              fillcolor=COLOR_HUNG_NN,  # Purple for nodes
              color='#6B46C1',  # Darker purple border
              penwidth='1.5',
              fontsize='10',
              fontcolor='white',  # White text for contrast
              fontname=family,
              label=str(node_idx))

# Add edges with orange color scheme
for u, v, data in subgraph.edges(data=True):
    G.add_edge(u, v,
              color=COLOR_MOD_HUNG,  # Orange for edges
              penwidth='2')

# Layout and render using dot for single-layer layout
print("Computing layout with dot algorithm...")
G.layout(prog='neato')  # Use dot layout algorithm for single-layer layout

# Save the graph
output_path = '/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/figures/paper_figures/random_2hop_subgraph_pygraphviz.png'
G.draw(output_path, format='png')
print(f"PyGraphviz visualization saved to: {output_path}")

# Also save as SVG for vector format
svg_output_path = '/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/figures/paper_figures/random_2hop_subgraph_pygraphviz.svg'
G.draw(svg_output_path, format='svg')
print(f"PyGraphviz SVG saved to: {svg_output_path}")

# Print some statistics about the pygraphviz graph
print(f"\n=== PyGraphviz Graph Statistics ===")
print(f"Nodes: {len(G.nodes())}")
print(f"Edges: {len(G.edges())}")
print(f"Layout algorithm: dot")
print(f"Graph attributes: {G.graph_attr}")

# Function to calculate mass differences between compounds
def calculate_mass_difference(smiles1, smiles2):
    """Calculate mass difference between two SMILES strings using RDKit's exact molecular weight."""
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return None
        
        # Calculate exact molecular weights using RDKit
        mass1 = rdMolDescriptors.CalcExactMolWt(mol1)
        mass2 = rdMolDescriptors.CalcExactMolWt(mol2)
        
        # Calculate mass difference (absolute value)
        mass_diff = abs(mass1 - mass2)
        
        return mass_diff
        
    except Exception as e:
        print(f"Error calculating mass difference for {smiles1} and {smiles2}: {e}")
        return None

# Function to calculate MCS and Tanimoto similarity
def calculate_mcs_tanimoto(smiles1, smiles2):
    """Calculate MCS size and Tanimoto similarity between two SMILES strings."""
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0, 0
        
        # Calculate MCS
        mcs_result = rdFMCS.FindMCS([mol1, mol2], atomCompare=rdFMCS.AtomCompare.CompareElements,
                                   bondCompare=rdFMCS.BondCompare.CompareOrder,
                                   ringMatchesRingOnly=True, completeRingsOnly=True)
        
        mcs_size = mcs_result.numAtoms if mcs_result.numAtoms > 0 else 0
        
        # Calculate Tanimoto similarity
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)
        tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)
        
        return mcs_size, tanimoto
        
    except Exception as e:
        print(f"Error calculating MCS/Tanimoto for {smiles1} and {smiles2}: {e}")
        return 0, 0 