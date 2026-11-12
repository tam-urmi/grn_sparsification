import os
import pandas as pd
import networkx as nx
import random
import pickle

#### Setup Paths ####
# Get the directory where the Python file is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define base directories relative to the script's location
base_dir = os.path.dirname(os.path.dirname(script_dir))
data_dir = os.path.join(base_dir, "data", "raw", "grn")

# Ensure the original graph directory exists
original_graph_dir = os.path.join(base_dir, "results", "graphs", "original")

noisy_graph_dir = os.path.join(base_dir, "results", "graphs", "noisy")

#### Functions ####

# Function to save graph in both edgelist and pickle format
def save_graph(G, filename, directory):
    """
    Saves the graph in both edgelist and pickle format.

    Parameters:
        G (nx.DiGraph): The directed graph.
        filename (str): The base filename (without extension).
    """
    edgelist_path = os.path.join(directory, f"{filename}.edgelist")
    pkl_path = os.path.join(directory, f"{filename}.pkl")

    os.makedirs(directory, exist_ok=True)

    # Save as edgelist
    nx.write_edgelist(G, edgelist_path, data=True)
    print(f"Graph saved as edgelist: {edgelist_path}")

    # Save as pickle
    with open(pkl_path, "wb") as f:
        pickle.dump(G, f)
    print(f"Graph saved as pickle: {pkl_path}")

# Load and create graph from `ms_numeric.csv`
def create_graph_ms_numeric():
    file_path = os.path.join(data_dir, "ms_numeric.csv")
    data = pd.read_csv(file_path)

    G = nx.DiGraph()
    for _, row in data.iterrows():
        G.add_edge(row["Source"], row["Target"], weight=row["Weight"], label="real")

    save_graph(G, "ms_numeric_graph", original_graph_dir)
    return G

# Load and create graph from `EColi_NetworkRegulatorGene.tsv`
def create_graph_ecoli():
    file_path = os.path.join(data_dir, "ecoli_hiTRN.csv")
    data = pd.read_csv(file_path)

    G = nx.DiGraph()
    for _, row in data.iterrows():
        G.add_edge(row["TF"], row["gene"], weight=1, label="real")

    save_graph(G, "ecoli_graph", original_graph_dir)
    return G

# Load and create graph from `C-elegans-frontal.txt`
def create_graph_c_elegans():
    file_path = os.path.join(data_dir, "C-elegans-frontal.txt")
    data = pd.read_csv(file_path, delim_whitespace=True, comment="#", names=["source", "sink"])

    G = nx.DiGraph()
    for _, row in data.iterrows():
        G.add_edge(row["source"], row["sink"], weight=1, label="real")

    save_graph(G, "c_elegans_graph", original_graph_dir)
    return G

## Generate synthetic graphs

# Preferential Attachment Model
def create_preferential_attachment_graph(n=500, m=5):
    G = nx.DiGraph()
    undirected_graph = nx.barabasi_albert_graph(n, m)
    G.add_edges_from((u, v, {"label": "real"}) if random.random() < 0.5 else (v, u, {"label": "real"}) for u, v in undirected_graph.edges())

    save_graph(G, "preferential_attachment_graph", original_graph_dir)
    return G

# Stochastic Block Model
def create_stochastic_block_model_graph(n=500, sizes=[100, 200, 200], p_in=0.1, p_out=0.01):
    probs = [[p_in if i == j else p_out for j in range(len(sizes))] for i in range(len(sizes))]
    undirected_graph = nx.stochastic_block_model(sizes, probs)

    G = nx.DiGraph()
    G.add_edges_from((u, v, {"label": "real"}) if random.random() < 0.5 else (v, u, {"label": "real"}) for u, v in undirected_graph.edges())

    save_graph(G, "stochastic_block_model_graph", original_graph_dir)
    return G

# Degree-Corrected Block Model
def create_degree_corrected_block_model_graph(n=500, sizes=[100, 200, 200], p_in=0.1, p_out=0.01):
    probs = [[p_in if i == j else p_out for j in range(len(sizes))] for i in range(len(sizes))]
    undirected_graph = nx.generators.community.random_partition_graph(sizes, probs[0][0], probs[0][1], directed=False, seed=None)

    G = nx.DiGraph()
    G.add_edges_from((u, v, {"label": "real"}) if random.random() < 0.5 else (v, u, {"label": "real"}) for u, v in undirected_graph.edges())

    save_graph(G, "degree_corrected_block_model_graph", original_graph_dir)
    return G


def add_random_noise_edges(graph, graph_name):
    """
    Counts the existing edges in a directed graph and then adds the same number
    of random 'noise' edges between pairs of nodes, labeled as 'noise'.
    The resulting graph is saved as an edge list.

    Parameters:
        graph (nx.DiGraph): The input directed graph.
        output_file (str): The file path to save the edge list.

    Returns:
        nx.DiGraph: The graph with added noise edges.
    """
    graph_copy = graph.copy()
    # Count the existing edges
    original_edge_count = graph_copy.number_of_edges()
    nodes = list(graph_copy.nodes())
    node_count = len(nodes)

    # Convert graph to an adjacency matrix
    adj_matrix = nx.to_numpy_array(graph_copy, nodelist=nodes)

    # Find node pairs with no edges (where adj_matrix value is 0)
    no_edge_pairs = [(i, j) for i in range(node_count) for j in range(node_count)
                     if i != j and adj_matrix[i, j] == 0]
    print("No-edge pairs: ", len(no_edge_pairs))

    # Randomly sample pairs to match the original edge count
    noise_edges = random.sample(no_edge_pairs, min(original_edge_count, len(no_edge_pairs)))

    # Add the noise edges to the graph
    for i, j in noise_edges:
        graph_copy.add_edge(nodes[i], nodes[j], label='noise')

    print("Number of noisy edges added:", len(noise_edges), "Real edge count:", original_edge_count)

    # Save the graph as an edge list with labels
    print(f"Graph with noise edges saved to {noisy_graph_dir}")
    save_graph(graph_copy, graph_name, noisy_graph_dir)
    return graph_copy

def add_preferential_noise_edges(graph, graph_name):
    """
    Adds noise edges to a directed graph using a preferential attachment strategy.

    Parameters:
        graph (nx.DiGraph): The input directed graph.
        output_file (str): The file path to save the edge list.

    Returns:
        nx.DiGraph: The graph with added noise edges.
    """
    graph_copy = graph.copy()
    nodes = list(graph_copy.nodes())

    # Step 1: Compute initial out-degree distribution and prepare the destination list
    initial_out_degrees = {node: graph_copy.out_degree(node) for node in nodes}
    destination_nodes = set(nodes)  # Nodes available as destinations for noise edges

    # Step 2: Track noise edges per node
    noise_edges_count = {node: 0 for node in nodes}

    # Step 3: Add noise edges based on preferential attachment
    for node in nodes:
        d = initial_out_degrees[node]  # Number of edges originally attached to the node
        current_noise_edges = noise_edges_count[node]

        # Calculate the number of noise edges to add
        noise_to_add = max(0, d - current_noise_edges)

        if noise_to_add > 0:
            valid_destinations = list(destination_nodes - {node})  # Remove self-loop possibility

            if len(valid_destinations) == 0:  # If no valid destination exists, skip
                continue

            for _ in range(noise_to_add):
                if len(valid_destinations) == 0:  # If we run out of valid destinations, stop early
                    break

                # Select a random destination node
                destination = random.choice(valid_destinations)

                # Add the noise edge
                graph_copy.add_edge(node, destination, label='noise')
                noise_edges_count[node] += 1  # Update noise edge count
                noise_edges_count[destination] += 1  # Track noise edges on the destination

                # If destination reaches its max noise edges, remove it from available choices
                if noise_edges_count[destination] >= initial_out_degrees[destination]:
                    destination_nodes.discard(destination)
                    valid_destinations.remove(destination)  # Keep list in sync

    print(f"Total noise edges added: {sum(noise_edges_count.values()) // 2}")

    # Step 4: Save the graph as an edge list with labels
    save_graph(graph_copy, graph_name, noisy_graph_dir)
    print(f"Graph with preferential noise edges saved as {graph_name}")
    return graph_copy


def main():
    os.makedirs(original_graph_dir, exist_ok=True)
    os.makedirs(noisy_graph_dir, exist_ok=True)

    graphs = {
        "ms_numeric": create_graph_ms_numeric(),
        "ecoli": create_graph_ecoli(),
        "c_elegans": create_graph_c_elegans(),
        "pref_attachment": create_preferential_attachment_graph(),
        "stochastic_block": create_stochastic_block_model_graph(),
        "deg_corrected_block": create_degree_corrected_block_model_graph(),
    }

    noisy_graphs = []
    summary_data = []
    for graph_name, graph in graphs.items():
        for noise_type, add_noise in (
            ("random", add_random_noise_edges),
            ("pref", add_preferential_noise_edges),
        ):
            name = f"{noise_type}_noisyG_{graph_name}"
            noisy_graph = add_noise(graph, name)
            noisy_graphs.append((name, noisy_graph))
            summary_data.append({
                "graph_name": name,
                "real_edges": sum(
                    1 for _, _, data in noisy_graph.edges(data=True)
                    if data.get("label") == "real"
                ),
                "noise_edges": sum(
                    1 for _, _, data in noisy_graph.edges(data=True)
                    if data.get("label") == "noise"
                ),
            })

    tables_dir = os.path.join(base_dir, "results", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    graph_setup_init_df = pd.DataFrame(summary_data)
    graph_setup_init_df.to_csv(
        os.path.join(tables_dir, "graph_setup_init_df.csv"), index=False
    )
    print(graph_setup_init_df)


if __name__ == "__main__":
    main()
