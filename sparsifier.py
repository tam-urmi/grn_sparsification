import os
import pickle
import networkx as nx
import random
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import cg

# Define the base directory dynamically
base_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory dynamically
noisy_graph_dir = os.path.join(base_dir, "noisy_graph")

def load_noisy_graphs():
    """
    Loads all directed graph pickle files from the noisy_graph directory
    and returns a dictionary of directed graphs.

    Returns:
        dict: A dictionary where keys are filenames (without .pkl)
              and values are nx.DiGraph objects.
    """
    graphs = {}

    # Check if the directory exists
    if not os.path.exists(noisy_graph_dir):
        print(f"Directory '{noisy_graph_dir}' not found.")
        return graphs

    # Loop through all pickle files in the directory
    for filename in os.listdir(noisy_graph_dir):
        if filename.endswith(".pkl"):  # Only process pickle files
            file_path = os.path.join(noisy_graph_dir, filename)

            # Load the directed graph
            with open(file_path, "rb") as f:
                directed_graph = pickle.load(f)

            # Ensure it's a directed graph
            if not isinstance(directed_graph, nx.DiGraph):
                print(f"Warning: {filename} is not a directed graph. Converting to nx.DiGraph.")
                directed_graph = nx.DiGraph(directed_graph)

            # Store the graph with filename (without extension) as the key
            graph_name = filename.replace(".pkl", "")
            graphs[graph_name] = directed_graph

            print(f"Loaded directed graph: {graph_name}")

    return graphs

# Example usage
noisy_graphs_dict = load_noisy_graphs()
print("Graph names: ", [i for i in noisy_graphs_dict.keys()])
print(f"Total noisy graphs loaded: {len(noisy_graphs_dict)}")

### Define sparsifiers

# === Helper Functions === #
def convert_graph(G, target_type):
    """Converts the graph to directed or undirected based on the target type."""
    if target_type == 'directed' and isinstance(G, nx.Graph):
        return nx.DiGraph(G), True
    elif target_type == 'undirected' and isinstance(G, nx.DiGraph):
        return nx.Graph(G), True
    return G, False  # No conversion needed

def extract_edge_labels(nx_graph):
    """Extracts edge labels from a NetworkX graph."""
    edge_labels = {}
    for u, v, data in nx_graph.edges(data=True):
        label = data.get('label', 'real')  # Default to 'real' if no label
        edge_labels[(u, v)] = label
    return edge_labels

def count_edges_by_label(G):
    """Counts the number of 'real' and 'noise' edges in the graph."""
    real_edges = sum(1 for _, _, data in G.edges(data=True) if data.get('label', 'real') == 'real')
    noise_edges = sum(1 for _, _, data in G.edges(data=True) if data.get('label', 'real') == 'noise')
    return real_edges, noise_edges

def save_graph(graph, folder, filename):
    """Saves the graph in both pickle and edgelist formats, preserving edge labels."""
    os.makedirs(folder, exist_ok=True)  # Ensure the folder exists

    # Save as pickle
    pkl_path = os.path.join(folder, f"{filename}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"Saved graph: {pkl_path}")

    # Save as edgelist with labels
    edgelist_path = os.path.join(folder, f"{filename}.edgelist")
    with open(edgelist_path, "w") as f:
        for u, v, data in graph.edges(data=True):
            label = data.get('label', 'real')
            f.write(f"{u} {v} {label}\n")
    print(f"Saved graph: {edgelist_path}")

# === Sparsifiers === #
def KNeighborSparsifier(G, edge_labels, k):
    """Retains up to 'k' neighbors per node (Undirected)."""
    G, _ = convert_graph(G, "undirected")
    new_G = nx.Graph()

    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) > k:
            neighbors = random.sample(neighbors, k)
        for neighbor in neighbors:
            label = edge_labels.get((node, neighbor), 'real')
            new_G.add_edge(node, neighbor, label=label)

    return new_G, extract_edge_labels(new_G)

def RankDegreeSparsifier(G, edge_labels, targetRatio):
    """Retains high-degree edges (Directed)."""
    G, _ = convert_graph(G, "directed")
    new_G = nx.DiGraph()

    targetNum = int(len(G.edges()) * targetRatio)
    sorted_edges = sorted(G.edges(data=True), key=lambda e: G.degree(e[1]), reverse=True)
    selected_edges = sorted_edges[:targetNum]

    for u, v, data in selected_edges:
        label = data.get('label', 'real')
        new_G.add_edge(u, v, label=label)

    return new_G, extract_edge_labels(new_G)

def CutSparsifier(G, edge_labels, targetRatio):
    """Selects edges probabilistically while retaining a higher fraction."""
    G, _ = convert_graph(G, "undirected")
    new_G = nx.Graph()

    k = {edge: 10 for edge in G.edges()}
    epsilon = 5  # Lower epsilon reduces aggressive pruning
    rho = 12 * np.log(G.number_of_nodes()) / epsilon ** 2  # Reduce aggressive edge removal

    for u, v in G.edges():
        p = min(1, rho / (k[(u, v)] + 1))  # Adjust threshold to retain more edges
        if random.random() < p:
            new_G.add_edge(u, v, label=edge_labels.get((u, v), 'real'))

    return new_G, extract_edge_labels(new_G)

def RandomEdgeSparsifier(G, edge_labels, targetRatio):
    """Randomly selects a fraction of edges, adjusted to retain more edges."""
    new_G = nx.Graph() if isinstance(G, nx.Graph) else nx.DiGraph()

    edges = list(G.edges())
    sampled_edges = random.sample(edges, int(len(edges) * targetRatio))

    for u, v in sampled_edges:
        new_G.add_edge(u, v, label=edge_labels.get((u, v), 'real'))

    return new_G, extract_edge_labels(new_G)

def ForestFireSparsifier(G, edge_labels, targetRatio, burn_prob):
    """Uses a probabilistic model where nodes 'burn' fewer edges."""
    G, _ = convert_graph(G, "directed")
    new_G = nx.DiGraph()

    selected_edges = random.sample(list(G.edges()), int(len(G.edges()) * targetRatio))

    for u, v in selected_edges:
        new_G.add_edge(u, v, label=edge_labels.get((u, v), 'real'))

    return new_G, extract_edge_labels(new_G)

def EffRSparsifier(G, edge_labels, epsilon, method, output_folder, file_name, tol=1e-10, precon=False):
    """
    Effective Resistance Sparsifier:
    Computes an effective resistance-based sparsified graph by approximating resistances using different methods.

    Parameters:
        G (nx.Graph): The input graph.
        edge_labels (dict): Dictionary containing edge labels.
        epsilon (float): Accuracy parameter for approximation.
        method (str): 'ext', 'spl', or 'kts' specifying resistance approximation type.
        output_folder (str): Folder to save pruned graphs.
        file_name (str): Base name for output files.
        tol (float): Tolerance for solving linear systems.
        precon (bool): Whether to use preconditioning for solving Laplacian systems.

    Returns:
        g_pruned (nx.Graph): The sparsified graph.
        new_edge_labels (dict): Updated edge labels for pruned edges.
    """

    def Mtrx_Elist(A):
        """Convert adjacency matrix to edge list."""
        j, i = np.nonzero(np.triu(A))
        elist = np.vstack((i, j)).transpose()
        weights = A[np.triu(A) != 0]
        return elist, weights

    def Elist_Mtrx_s(E_list, weights):
        """Convert edge list to sparse adjacency matrix."""
        n = np.max(E_list) + 1
        A = sparse.csr_matrix((weights, (E_list[:, 0], E_list[:, 1])), shape=(n, n))
        return A + A.transpose()

    def Lap_s(A):
        """Compute Laplacian for a sparse adjacency matrix."""
        return sparse.csgraph.laplacian(A)

    def sVIM(E_list):
        """Compute signed-edge vertex incidence matrix."""
        m = E_list.shape[0]
        data = [1] * m + [-1] * m
        i = list(range(m)) * 2
        j = np.hstack([E_list[:, 0], E_list[:, 1]]).tolist()
        return sparse.csr_matrix((data, (i, j)), shape=(m, np.max(E_list) + 1))

    def WDiag(weights):
        """Compute diagonal weight matrix."""
        weights_sqrt = np.sqrt(weights)
        return sparse.dia_matrix((weights_sqrt, [0]), shape=(len(weights), len(weights)))

    # Convert graph to adjacency matrix
    E_list, weights = Mtrx_Elist(nx.to_numpy_array(G))
    labels = {edge: edge_labels.get((edge[0], edge[1]), 'real') for edge in G.edges}

    m, n = E_list.shape[0], np.max(E_list) + 1
    A = Elist_Mtrx_s(E_list, weights)
    L = Lap_s(A)
    B = sVIM(E_list)
    W = WDiag(weights)
    scale = np.ceil(np.log2(n)) / epsilon

    if precon:
        M_inverse = sparse.linalg.spilu(L)
        M = sparse.linalg.LinearOperator((n, n), M_inverse.solve)
    else:
        M = None

    if method == 'ext':
        effR = np.zeros(m)
        for i in range(m):
            Br = B[i, :].toarray()
            Z = cg(L, Br.T, tol=tol, M=M)[0]
            effR[i] = Br @ Z

    elif method == 'spl':
        Q = sparse.random(int(scale), m, density=1.0, format='csr', random_state=42)
        Q.data = np.where(Q.data > 0.5, 1, -1) / np.sqrt(scale)
        SYS = Q @ W @ B
        Z = np.zeros((int(scale), n))

        for i in range(int(scale)):
            SYSr = SYS[i, :].toarray()
            Z[i, :] = cg(L, SYSr.T, tol=tol, M=M)[0]

        effR = np.sum((Z[:, E_list[:, 0]] - Z[:, E_list[:, 1]]) ** 2, axis=0)

    elif method == 'kts':
        effR_res = np.zeros(m)
        for i in range(int(scale)):
            ons = sparse.random(1, m, density=1.0, format='csr', random_state=42)
            ons.data = np.where(ons.data > 0.5, 1, -1) / np.sqrt(scale)
            b = (ons @ W @ B).toarray()
            Z = cg(L, b.T, tol=tol, M=M)[0]

            if Z.ndim == 1:
                Z = Z.reshape(-1, 1)

            effR_res += np.abs((Z[E_list[:, 0]] - Z[E_list[:, 1]]) ** 2).flatten()
        effR = effR_res / int(scale)

    # Threshold-based pruning
    thres_cutoff = 60  # Keep the top 60% edges
    thres = np.percentile(effR, thres_cutoff)
    indices = [i for i, value in enumerate(effR) if value > thres]

    edges_to_keep = [(E_list[i, 0], E_list[i, 1]) for i in indices]
    g_pruned = nx.Graph()
    for u, v in edges_to_keep:
        g_pruned.add_edge(u, v, label=labels.get((u, v), 'real'))

    # Save effective resistance values to a file
    os.makedirs(output_folder, exist_ok=True)
    effR_txt_path = os.path.join(output_folder, f"{file_name}_{method}_effR.txt")
    with open(effR_txt_path, "w") as f:
        for value in effR:
            f.write(f"{value}\n")

    new_edge_labels = extract_edge_labels(g_pruned)
    return g_pruned, new_edge_labels

# === Applying Sparsifiers to Noisy Graphs === #
def extract_edge_labels(G):
    """Extracts a dictionary of edge labels keyed by (u, v)."""
    return {(u, v): d.get("label", "real") for u, v, d in G.edges(data=True)}

def count_edges_by_label(G):
    """Counts number of real and noise edges in a graph."""
    real = sum(1 for _, _, d in G.edges(data=True) if d.get("label") == "real")
    noise = sum(1 for _, _, d in G.edges(data=True) if d.get("label") == "noise")
    return real, noise

def save_graph(G, directory, filename):
    """Saves the graph in both edgelist and pickle format."""
    import pickle
    edgelist_path = os.path.join(directory, f"{filename}.edgelist")
    pkl_path = os.path.join(directory, f"{filename}.pkl")

    nx.write_edgelist(G, edgelist_path, data=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(G, f)
    print(f"Saved: {filename} → .edgelist & .pkl")

def process_noisy_graphs(noisy_graphs_dict, sparsifiers):
    """Applies all sparsifiers to graphs in noisy_graphs_dict, saves them, and logs edge counts."""
    pruned_graphs_dir = "pruned_graphs"
    os.makedirs(pruned_graphs_dir, exist_ok=True)

    for graph_name, G in noisy_graphs_dict.items():
        print(f"\n🔍 Processing graph: {graph_name}")

        graph_folder = os.path.join(pruned_graphs_dir, graph_name)
        os.makedirs(graph_folder, exist_ok=True)

        edge_labels = extract_edge_labels(G)
        real_before, noise_before = count_edges_by_label(G)

        print(f"Label check — Initial: {real_before} real, {noise_before} noise edges")

        for sparsifier_name, sparsifier_func, params in sparsifiers:
            print(f"\n✨ Applying {sparsifier_name} to {graph_name}...")

            pruned_graph, new_edge_labels = sparsifier_func(G, edge_labels, **params)

            # Count after pruning
            real_after, noise_after = count_edges_by_label(pruned_graph)

            print(f"Before pruning: {real_before} real, {noise_before} noise")
            print(f"After {sparsifier_name}: {real_after} real, {noise_after} noise")

            # Optional: check if any edges are missing a label
            unlabeled = sum(1 for _, _, d in pruned_graph.edges(data=True) if 'label' not in d)
            if unlabeled > 0:
                print(f"⚠️ Warning: {unlabeled} edges have no 'label' after pruning!")

            # Double-check: all edges should have label
            assert all('label' in d for _, _, d in pruned_graph.edges(data=True)), \
                f"Some edges are missing the 'label' attribute in {sparsifier_name}!"

            # Save pruned graph
            save_graph(pruned_graph, graph_folder, f"{sparsifier_name}")

# === Updated Sparsifiers List (All Parameters Set Here) === #
sparsifiers = [
    ("KNeighbor", KNeighborSparsifier, {"k": 6}),
    ("RankDegree", RankDegreeSparsifier, {"targetRatio": 0.65}),
    ("Cut", CutSparsifier, {"targetRatio": 0.65}),
    ("RandomEdge", RandomEdgeSparsifier, {"targetRatio": 0.65}),
    ("ForestFire", ForestFireSparsifier, {"targetRatio": 0.65, "burn_prob": 0.1}),
    ("EffR", EffRSparsifier, {"epsilon": 0.1, "method": "spl", "output_folder": "pruned_graphs", "file_name": "effr_sparsified"})
]

# Run processing
process_noisy_graphs(noisy_graphs_dict, sparsifiers)
print("\nAll graphs processed and saved!")
