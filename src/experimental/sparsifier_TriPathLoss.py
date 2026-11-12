
import os
import pandas as pd
import networkx as nx
import csv
import matplotlib.pyplot as plt
import numpy as np
import random
import networkit as nk
import scipy.sparse as sparse
from scipy.sparse.linalg import cg

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = os.path.join(project_dir, "data", "raw", "grn")
noisy_graph_dir = os.path.join(project_dir, "results", "graphs", "noisy")


#### Functions ####

# Load and create graph from `ms_numeric.csv`
def create_graph_ms_numeric():
    file_path = os.path.join(data_dir, 'ms_numeric.csv')
    data = pd.read_csv(file_path)
    # display(data)
    G = nx.DiGraph()
    for _, row in data.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'], label='real')
    return G

# Load and create graph from `EColi_NetworkRegulatorGene.tsv`
def create_graph_ecoli():
    file_path = os.path.join(data_dir, 'ecoli_hiTRN.csv')
    data = pd.read_csv(file_path)

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges using 'regulatorId' as the source, 'regulatedId' as the sink, and 'function' as the weight
    for _, row in data.iterrows():
        G.add_edge(row['TF'], row['gene'], weight=1, label='real')

    return G

# Load and create graph from `C-elegans-frontal.txt`
def create_graph_c_elegans():
    file_path = os.path.join(data_dir, 'C-elegans-frontal.txt')
    data = pd.read_csv(file_path, delim_whitespace=True, comment='#', names=['source', 'sink'])
    G = nx.DiGraph()
    for _, row in data.iterrows():
        G.add_edge(row['source'], row['sink'], weight=1, label='real')  # Assuming default weight = 1
    return G

## Generate synthetic graphs

# Preferential Attachment Model
def preferential_attachment_graph(n=500, m=5):
    """
    Generates a directed graph using the Preferential Attachment model.

    Parameters:
        n (int): Number of nodes.
        m (int): Number of edges to attach from a new node to existing nodes.

    Returns:
        DiGraph: Directed graph generated with preferential attachment and labeled edges.
    """
    undirected_graph = nx.barabasi_albert_graph(n, m)
    directed_graph = nx.DiGraph()
    directed_graph.add_edges_from(
        ((u, v, {"label": "real"}) if random.random() < 0.5 else (v, u, {"label": "real"}))
        for u, v in undirected_graph.edges()
    )
    return directed_graph

# Stochastic Block Model
def stochastic_block_model_graph(n=500, sizes=[100, 200, 200], p_in=0.1, p_out=0.01):
    """
    Generates a directed graph using the Stochastic Block Model.

    Parameters:
        n (int): Number of nodes.
        sizes (list): List of integers representing the sizes of each block.
        p_in (float): Probability of edges within blocks.
        p_out (float): Probability of edges between blocks.

    Returns:
        DiGraph: Directed graph generated with stochastic block model and labeled edges.
    """
    probs = [[p_in if i == j else p_out for j in range(len(sizes))] for i in range(len(sizes))]
    undirected_graph = nx.stochastic_block_model(sizes, probs)
    directed_graph = nx.DiGraph()
    directed_graph.add_edges_from(
        ((u, v, {"label": "real"}) if random.random() < 0.5 else (v, u, {"label": "real"}))
        for u, v in undirected_graph.edges()
    )
    return directed_graph

# Degree-Corrected Block Model
def degree_corrected_block_model_graph(n=500, sizes=[100, 200, 200], p_in=0.1, p_out=0.01):
    """
    Generates a directed graph using the Degree-Corrected Block Model.

    Parameters:
        n (int): Number of nodes.
        sizes (list): List of integers representing the sizes of each block.
        p_in (float): Probability of edges within blocks.
        p_out (float): Probability of edges between blocks.

    Returns:
        DiGraph: Directed graph generated with degree-corrected block model and labeled edges.
    """
    probs = [[p_in if i == j else p_out for j in range(len(sizes))] for i in range(len(sizes))]
    undirected_graph = nx.generators.community.random_partition_graph(sizes, probs[0][0], probs[0][1], directed=False, seed=None)
    directed_graph = nx.DiGraph()
    directed_graph.add_edges_from(
        ((u, v, {"label": "real"}) if random.random() < 0.5 else (v, u, {"label": "real"}))
        for u, v in undirected_graph.edges()
    )
    return directed_graph

# Random Geometric Graph (adapted for directed edges)
def random_geometric_graph(n=500, radius=0.1):
    """
    Generates a directed random geometric graph.

    Parameters:
        n (int): Number of nodes.
        radius (float): Distance threshold for edge creation.

    Returns:
        DiGraph: Directed geometric graph with labeled edges.
    """
    undirected_graph = nx.random_geometric_graph(n, radius)
    directed_graph = nx.DiGraph()
    directed_graph.add_edges_from(
        ((u, v, {"label": "real"}) if random.random() < 0.5 else (v, u, {"label": "real"}))
        for u, v in undirected_graph.edges()
    )
    return directed_graph

### Operations for graph properties

def count_motifs(graph):
    """Count motifs: directed motifs for directed graphs, triangles, and wedges for undirected graphs."""
    # Check if the input graph is in Networkit format, convert if necessary
    if isinstance(graph, nk.Graph):
        # Convert Networkit graph to NetworkX graph (directionality is preserved automatically)
        graph = nk.nxadapter.nk2nx(graph)

    # Proceed with motif counting based on the directionality of the graph
    if graph.is_directed():
        triads = nx.algorithms.triads.triadic_census(graph)
        motifs = {
            key: value for key, value in triads.items() if key in ["021D", "021U", "021C", "111D", "111U", "030T",
                                                                   "030C", "201", "120D", "120U", "120C", "210", "300"]
        }
    else:
        triangles = sum(nx.triangles(graph).values()) // 3  # Each triangle counted three times
        wedges = sum(deg * (deg - 1) // 2 for node, deg in graph.degree()) - triangles
        motifs = {"Triangles": triangles, "Wedges": wedges}

    return motifs

### Plot network properties

def plot_network_properties(graph, title, save_path):
    fig, axes = plt.subplots(1, 6, figsize=(26, 5))
    fig.suptitle(title, fontsize=16)

    # 1. Log-Log Degree Distribution
    degrees = [graph.degree(n) for n in graph.nodes()]
    degree_counts = np.bincount(degrees)
    x = np.nonzero(degree_counts)[0]
    y = degree_counts[x]
    axes[0].loglog(x, y, marker="o", linestyle="none", alpha=0.6)
    axes[0].set_title("Log-Log Degree Distribution")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Count")

    # 2. Motif Count
    motif_counts = count_motifs(graph)
    axes[1].bar(motif_counts.keys(), motif_counts.values(), color="blue", alpha=0.7)
    axes[1].set_title("Motif Count")
    axes[1].set_xlabel("Motif Type")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis='x', rotation=30)

    # 3. Node-wise Clustering Coefficient
    clustering_coeffs = nx.clustering(graph.to_undirected())
    axes[2].plot(list(clustering_coeffs.keys()), list(clustering_coeffs.values()), color='b', marker=".", linestyle="none")
    axes[2].set_title("Node-wise Clustering")
    axes[2].set_xlabel("Node")
    axes[2].set_ylabel("Clustering Coefficient")

    # 4. Laplacian Spectrum (only on largest connected component to avoid NaNs)
    largest_cc = max(nx.connected_components(graph.to_undirected()), key=len)
    subgraph = graph.subgraph(largest_cc).to_undirected()
    laplacian = nx.laplacian_matrix(subgraph).todense()

    if not np.isfinite(laplacian).all():
        print("Warning: Laplacian matrix contains NaNs or infinities. Skipping eigenvalue calculation.")
        eigenvalues = []
    else:
        eigenvalues = np.linalg.eigvals(laplacian)

    if len(eigenvalues) > 0:
        axes[3].hist(eigenvalues, bins=30, color="green", alpha=0.6)
        axes[3].set_title("Laplacian Spectrum")
        axes[3].set_xlabel("Eigenvalue")
        axes[3].set_ylabel("Frequency")
    else:
        axes[3].text(0.5, 0.5, "Laplacian Spectrum\nNot Computed", ha="center", va="center")
        axes[3].set_title("Laplacian Spectrum")
        axes[3].set_xlabel("Eigenvalue")
        axes[3].set_ylabel("Frequency")

    # 5. Degree vs Clustering Coefficient
    degree_values = list(dict(graph.degree()).values())
    clustering_values = list(nx.clustering(graph.to_undirected()).values())
    axes[4].scatter(degree_values, clustering_values, alpha=0.6, edgecolors='k')
    axes[4].set_title("Degree vs Clustering Coefficient")
    axes[4].set_xlabel("Degree")
    axes[4].set_ylabel("Clustering Coefficient")

    # 6. Path Length Distribution (for largest connected component)
    if nx.is_directed(graph):
        lengths = dict(nx.shortest_path_length(graph))
    else:
        lengths = dict(nx.shortest_path_length(graph.to_undirected()))

    path_lengths = []
    for source in lengths:
        for target in lengths[source]:
            if source != target:
                path_lengths.append(lengths[source][target])

    if path_lengths:
        axes[5].hist(path_lengths, bins=30, color="purple", alpha=0.6)
        axes[5].set_title("Path Length Distribution")
        axes[5].set_xlabel("Path Length")
        axes[5].set_ylabel("Frequency")
    else:
        axes[5].text(0.5, 0.5, "Path Length Distribution\nNot Computed", ha="center", va="center")
        axes[5].set_title("Path Length Distribution")
        axes[5].set_xlabel("Path Length")
        axes[5].set_ylabel("Frequency")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save and close the figure
    fig.savefig(save_path)
    plt.close(fig)

def add_random_noise_edges(graph, output_file):
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
    os.makedirs(noisy_graph_dir, exist_ok=True)
    output_path = os.path.join(noisy_graph_dir, os.path.basename(output_file))
    with open(output_path, 'w') as f:
        for u, v, data in graph_copy.edges(data=True):
            label = data.get('label', 'real')  # Default label to 'real' if not specified
            f.write(f"{u} {v} {label}\n")

    print(f"Graph with noise edges saved to {output_path}")
    return graph_copy



def KNeighborSparsifier(G, edge_labels, k, output_folder, file_name, print_info=False):
    if isinstance(G, (nx.Graph, nx.DiGraph)):
        G, edge_labels = nx_to_nk(G)

    new_G = nk.graph.Graph(G.numberOfNodes(), weighted=G.isWeighted(), directed=G.isDirected())
    new_edge_labels = {}

    for node in range(0, G.numberOfNodes()):
        neighbors = list(G.iterNeighbors(node))
        if len(neighbors) > k:
            neighbors = random.sample(neighbors, k)
        for neighbor in neighbors:
            if not new_G.hasEdge(node, neighbor):
                weight = G.weight(node, neighbor) if G.isWeighted() else 1
                label = edge_labels.get((node, neighbor), 'real')
                new_G.addEdge(node, neighbor, weight)
                new_edge_labels[(node, neighbor)] = label

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{file_name}.edgelist")
    nk.writeGraph(new_G, output_path, nk.Format.EdgeListSpaceZero)
    return new_G, new_edge_labels

def RankDegreeSparsifier(G, edge_labels, targetRatio, output_folder, file_name, rho=0.1, print_info=False):
    if isinstance(G, (nx.Graph, nx.DiGraph)):
        G, edge_labels = nx_to_nk(G)

    targetNum = int(G.numberOfEdges() * targetRatio)
    seeds, seen = set(), set()
    new_G = nk.graph.Graph(G.numberOfNodes(), weighted=G.isWeighted(), directed=G.isDirected())
    new_edge_labels = {}
    iter_count, count = 0, 0

    while new_G.numberOfEdges() < targetNum:
        iter_count += 1
        if not seeds:
            seeds = set(random.sample(list(range(0, G.numberOfNodes())), 10))

        if count > int(0.01 * targetNum):
            iter_count, count = 0, 0
        elif iter_count > 1000 and count <= int(0.01 * targetNum) and rho < 0.99:
            iter_count, count = 0, 0
            rho += 0.1
            seen = set()

        new_seeds = set()
        for seed in seeds:
            if seed in seen:
                continue
            seen.add(seed)
            neighbors = sorted(list(G.iterNeighbors(seed)), key=lambda x: G.degree(x), reverse=True)
            neighbors = neighbors[:max(1, int(len(neighbors) * rho))]
            for neighbor in neighbors:
                if not new_G.hasEdge(seed, neighbor):
                    weight = G.weight(seed, neighbor) if G.isWeighted() else 1
                    label = edge_labels.get((seed, neighbor), 'real')
                    new_G.addEdge(seed, neighbor, weight)
                    new_edge_labels[(seed, neighbor)] = label
                    new_seeds.add(neighbor)
                    count += 1
        seeds = new_seeds

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{file_name}.edgelist")
    nk.writeGraph(new_G, output_path, nk.Format.EdgeListSpaceZero)
    return new_G, new_edge_labels

def CutSparsifier(G, edge_labels, targetRatio, output_folder, file_name):
    if isinstance(G, (nx.Graph, nx.DiGraph)):
        G, edge_labels = nx_to_nk(G)

    new_G = nk.graph.Graph(G.numberOfNodes(), weighted=G.isWeighted(), directed=G.isDirected())
    new_edge_labels = {}
    k = {edge: 10 for edge in G.iterEdges()}
    epsilon = 10
    d = 1
    rho = 16 * (d + 2) * np.log(G.numberOfNodes()) / epsilon ** 2

    for edge in G.iterEdges():
        p = min(1, rho / k[edge])
        if random.random() < p:
            u, v = edge
            weight = G.weight(u, v) if G.isWeighted() else 1
            label = edge_labels.get((u, v), 'real')
            new_G.addEdge(u, v, weight)
            new_edge_labels[(u, v)] = label

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{file_name}.edgelist")
    nk.writeGraph(new_G, output_path, nk.Format.EdgeListSpaceZero)
    return new_G, new_edge_labels

def RandomEdgeSparsifier(G, edge_labels, targetRatio, output_folder, file_name, print_info=False):
    if isinstance(G, (nx.Graph, nx.DiGraph)):
        G, edge_labels = nx_to_nk(G)

    if not G.hasEdgeIds():
        G.indexEdges()

    randomEdgeSparsifier = nk.sparsification.RandomEdgeSparsifier()
    randomGraph = randomEdgeSparsifier.getSparsifiedGraphOfSize(G, targetRatio)
    new_edge_labels = {}

    for u, v in randomGraph.edges():
        label = edge_labels.get((u, v), 'real')
        new_edge_labels[(u, v)] = label

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{file_name}.edgelist")
    nk.writeGraph(randomGraph, output_path, nk.Format.EdgeListSpaceZero)
    return randomGraph, new_edge_labels

def LocalDegreeSparsifier(G, edge_labels, targetRatio, output_folder, file_name, print_info=False):
    if isinstance(G, (nx.Graph, nx.DiGraph)):
        G, edge_labels = nx_to_nk(G)

    localDegSparsifier = nk.sparsification.LocalDegreeSparsifier()
    localDegGraph = localDegSparsifier.getSparsifiedGraphOfSize(G, targetRatio)
    new_edge_labels = {}

    for u, v in localDegGraph.edges():
        label = edge_labels.get((u, v), 'real')
        new_edge_labels[(u, v)] = label

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{file_name}.edgelist")
    nk.writeGraph(localDegGraph, output_path, nk.Format.EdgeListSpaceZero)
    return localDegGraph, new_edge_labels

def ForestFireSparsifier(G, edge_labels, targetRatio, output_folder, file_name, print_info=False):
    if isinstance(G, (nx.Graph, nx.DiGraph)):
        G, edge_labels = nx_to_nk(G)

    if targetRatio > 0.5:
        fireSparsifier = nk.sparsification.ForestFireSparsifier(0.2, 5.0)
    else:
        fireSparsifier = nk.sparsification.ForestFireSparsifier(0.6, 5.0)
    fireGraph = fireSparsifier.getSparsifiedGraphOfSize(G, targetRatio)
    new_edge_labels = {}

    for u, v in fireGraph.edges():
        label = edge_labels.get((u, v), 'real')
        new_edge_labels[(u, v)] = label

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{file_name}.edgelist")
    nk.writeGraph(fireGraph, output_path, nk.Format.EdgeListSpaceZero)
    return fireGraph, new_edge_labels

def LocalSimilaritySparsifier(G, edge_labels, targetRatio, output_folder, file_name, print_info=False):
    if isinstance(G, (nx.Graph, nx.DiGraph)):
        G, edge_labels = nx_to_nk(G)

    localSimilaritySparsifier = nk.sparsification.LocalSimilaritySparsifier()
    localSimilarityGraph = localSimilaritySparsifier.getSparsifiedGraphOfSize(G, targetRatio)
    new_edge_labels = {}

    for u, v in localSimilarityGraph.edges():
        label = edge_labels.get((u, v), 'real')
        new_edge_labels[(u, v)] = label

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder)
    nk.writeGraph(localSimilarityGraph, output_path, nk.Format.EdgeListSpaceZero)

    return localSimilarityGraph, new_edge_labels

def Mtrx_Elist(A):
    """Convert adjacency matrix to edge list."""
    j, i = np.nonzero(np.triu(A))  # Find edges in the upper triangle
    elist = np.vstack((i, j)).transpose()  # Edge list as pairs of nodes
    weights = A[np.triu(A) != 0]  # Extract weights

    return elist, weights

def Elist_Mtrx(E_list, weights):
    """Convert edge list to adjacency matrix."""
    n = np.max(E_list) + 1
    A = np.zeros((n, n))

    for i in range(E_list.shape[0]):
        n1, n2 = E_list[i]
        w = weights[i]
        A[n1, n2], A[n2, n1] = w, w

    return A

def Elist_Mtrx_s(E_list, weights):
    """Convert edge list to sparse adjacency matrix."""
    n = np.max(E_list) + 1
    A = sparse.csr_matrix((weights, (E_list[:, 0], E_list[:, 1])), shape=(n, n))
    return A + A.transpose()

def Lap(A):
    """Compute Laplacian from adjacency matrix."""
    return np.diag(np.sum(np.abs(A), axis=1)) - A

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

def weight_list(graph):
    weights_list = np.array([data['weight'] for u, v, data in graph.edges(data=True) if 'weight' in data])
    return weights_list

def EffR(graph, edge_labels, epsilon, type, output_folder, file_name, tol=1e-10, precon=False):
    """Compute effective resistance approximation and save pruned graph with edge labels."""
    E_list, weights = Mtrx_Elist(nx.to_numpy_array(graph))
    labels = [edge_labels.get((u, v), 'real') for u, v in graph.edges]  # Extract labels

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

    if type == 'ext':
        effR = np.zeros(m)
        for i in range(m):
            Br = B[i, :].toarray()
            Z = cg(L, Br.T, tol=tol, M=M)[0]
            effR[i] = Br @ Z

    elif type == 'spl':
        Q = sparse.random(int(scale), m, density=1.0, format='csr', random_state=42)
        Q.data = np.where(Q.data > 0.5, 1, -1) / np.sqrt(scale)
        SYS = Q @ W @ B
        Z = np.zeros((int(scale), n))

        for i in range(int(scale)):
            SYSr = SYS[i, :].toarray()
            Z[i, :] = cg(L, SYSr.T, tol=tol, M=M)[0]

        effR = np.sum((Z[:, E_list[:, 0]] - Z[:, E_list[:, 1]]) ** 2, axis=0)

    elif type == 'kts':
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

    # Threshold cutoff and creation of pruned graphs with label preservation
    thres_cutoff = [30, 60, 90]

    for t, cutoff in enumerate(thres_cutoff):
        thres = np.percentile(effR, cutoff)
        indices = [i for i, value in enumerate(effR) if value > thres]

        edges_to_keep = [(E_list[i, 0], E_list[i, 1], labels[i]) for i in indices]  # Retain labels
        print(edges_to_keep)

        # Create a new pruned graph with labels
        g_pruned = nx.Graph()
        for u, v, label in edges_to_keep:
            g_pruned.add_edge(u, v, label=label)

        # Save the pruned graph with edge labels in .edgelist format
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"{file_name}_{type}_{t}.edgelist")
        with open(output_path, "w") as f:
            for u, v, label in edges_to_keep:
                # Writing each edge with source, target, and label directly
                f.write(f"{u} {v} {label}\n")

        # Save the effective resistance values to a text file
        effR_txt_path = os.path.join(output_folder, f"{file_name}_{type}_{cutoff}_effR.txt")
        with open(effR_txt_path, "w") as f:
            for value in effR:
                f.write(f"{value}\n")

    new_edge_labels = [edge_labels.get((u, v), 'real') for u, v in g_pruned.edges]

    return g_pruned, new_edge_labels, effR

### Function to apply the sparsifiers

def extract_edge_labels(nx_graph):
    """Extract edge labels from a NetworkX graph."""
    edge_labels = {}
    for u, v, data in nx_graph.edges(data=True):
        label = data.get('label', 'real')  # Default to 'real' if no label
        edge_labels[(u, v)] = label
    return edge_labels

def save_edge_labels_txt(edge_labels, file_path):
    """Save edge labels to a text file."""
    with open(file_path, 'w') as f:
        if isinstance(edge_labels, dict):
            for edge, label in edge_labels.items():
                f.write(f"{edge[0]} {edge[1]} {label}\n")
        elif isinstance(edge_labels, list):
            for edge in edge_labels:
                f.write(f"{edge[0]} {edge[1]} {edge[2]}\n")
    print(f"Edge labels saved as: {file_path}")

def save_effR_values_txt(effR_values, file_path):
    """Save effective resistance values to a text file."""
    with open(file_path, 'w') as f:
        for value in effR_values:
            f.write(f"{value}\n")
    print(f"Effective resistance values saved as: {file_path}")

def apply_sparsifiers(graph_list, sparsifiers, output_folder):
    for graph_name, G in graph_list:
        # Extract edge labels from the original graph
        edge_labels = extract_edge_labels(G)

        for sparsifier_name, sparsifier_func, params in sparsifiers:
            # Customize file name based on graph and sparsifier
            file_name = f"pruned_{graph_name}_{sparsifier_name}"
            additional_outputs = {}

            # Apply the sparsifier and retrieve pruned graph and any additional outputs
            print(f"Applying {sparsifier_name} to {graph_name}...")
            if sparsifier_name == "EffR":
                pruned_graph, new_edge_labels, effR_values = sparsifier_func(
                    G, edge_labels=edge_labels, output_folder=output_folder, file_name=file_name, **params
                )
                additional_outputs["effR_values"] = effR_values
            else:
                pruned_graph, new_edge_labels = sparsifier_func(
                    G, edge_labels=edge_labels, output_folder=output_folder, file_name=file_name, **params
                )

            additional_outputs["edge_labels"] = new_edge_labels

            # Save the pruned graph in edgelist format
            print(f"{sparsifier_name} applied. Pruned graph saved as: {file_name}.edgelist")

            # Save additional outputs if present
            if "edge_labels" in additional_outputs:
                edge_label_path = os.path.join(output_folder, f"{file_name}_edge_labels.txt")
                save_edge_labels_txt(additional_outputs["edge_labels"], edge_label_path)

            if "effR_values" in additional_outputs:
                effR_values_path = os.path.join(output_folder, f"{file_name}_effR_values.txt")
                save_effR_values_txt(additional_outputs["effR_values"], effR_values_path)

## New sparsifier to apply on noisy graphs

def calculate_triangle_path_score_nk(G, u, v, min_path_length, triangle_weight=1.0, path_weight=1.0):
    """
    Calculate the importance score of an edge based on its contribution to motifs
    and paths of length greater than or equal to min_path_length.

    Parameters:
        G (nk.Graph): The original NetworKit graph.
        u (int): Starting node of the edge.
        v (int): Ending node of the edge.
        min_path_length (int): Minimum length of paths to consider for path preservation.
        triangle_weight (float): Weight for the motif contribution.
        path_weight (float): Weight for the path contribution.

    Returns:
        float: The score for the edge.
    """
    # Count motifs around the edge (u, v) using the count_motif function
    motif_dict = count_motifs(G)
    motif_count = sum(motif_dict.values())

    # Path length calculation (convert to undirected graph temporarily if directed)
    path_count = 0
    if G.isDirected():
        G_undirected = nk.graphtools.toUndirected(G)
    else:
        G_undirected = G

    # BFS from v (excluding u) to find paths of length >= min_path_length
    current_nodes = {v}
    for path_length in range(1, min_path_length + 1):
        next_nodes = set()
        for node in current_nodes:
            next_nodes.update(set(G_undirected.iterNeighbors(node)) - {u})
        path_count += len(next_nodes) if path_length >= min_path_length else 0
        current_nodes = next_nodes

    # Calculate edge importance score
    score = triangle_weight * motif_count + path_weight * path_count
    return score

def TrianglePathLossSparsifier(G, edge_labels, targetRatio, output_folder, file_name, min_path_length, rho=0.1, triangle_weight=1.0, path_weight=1.0, print_info=False):
    if isinstance(G, (nx.Graph, nx.DiGraph)):
        G, edge_labels = nx_to_nk(G)

    targetNum = int(G.numberOfEdges() * targetRatio)
    seeds, seen = set(), set()

    # Create a new NetworKit graph to store the pruned edges and labels
    new_G = nk.Graph(G.numberOfNodes(), weighted=G.isWeighted(), directed=G.isDirected())
    new_edge_labels = {}
    iter_count, count = 0, 0

    while new_G.numberOfEdges() < targetNum:
        iter_count += 1
        if not seeds:
            seeds = set(random.sample(range(G.numberOfNodes()), 10))

        if count > int(0.01 * targetNum):
            iter_count, count = 0, 0
        elif iter_count > 1000 and count <= int(0.01 * targetNum) and rho < 0.99:
            iter_count, count = 0, 0
            rho += 0.1
            seen = set()

        new_seeds = set()
        for seed in seeds:
            if seed in seen:
                continue
            seen.add(seed)

            # Calculate scores for each neighboring edge based on motif and path importance
            neighbor_scores = [
                (neighbor, calculate_triangle_path_score_nk(G, seed, neighbor, min_path_length, triangle_weight, path_weight))
                for neighbor in G.iterNeighbors(seed)
            ]

            # Sort neighbors by score and take the top fraction based on rho
            sorted_neighbors = sorted(neighbor_scores, key=lambda x: x[1], reverse=True)
            selected_neighbors = [n[0] for n in sorted_neighbors[:max(1, int(len(sorted_neighbors) * rho))]]

            for neighbor in selected_neighbors:
                if not new_G.hasEdge(seed, neighbor):
                    weight = G.weight(seed, neighbor) if G.isWeighted() else 1
                    label = edge_labels.get((seed, neighbor), 'real')
                    new_G.addEdge(seed, neighbor, weight)
                    new_edge_labels[(seed, neighbor)] = label
                    new_seeds.add(neighbor)
                    count += 1
        seeds = new_seeds

    # Save the sparsified graph in NetworKit-compatible format
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{file_name}.edgelist")
    nk.writeGraph(new_G, output_path, nk.Format.EdgeListSpaceZero)

    return new_G, new_edge_labels

def main():
    graph_factories = [
        ("MS", "noisyG_ms_numeric", create_graph_ms_numeric),
        ("EColi", "noisyG_ecoli", create_graph_ecoli),
        ("CElegans", "noisyG_c_elegans", create_graph_c_elegans),
        ("PA", "noisyG_pref_attachment", preferential_attachment_graph),
        ("Stoch_block", "noisyG_stochastic_block", stochastic_block_model_graph),
        ("DC_block", "noisyG_deg_corrected_block", degree_corrected_block_model_graph),
        ("RandomGeom", "noisyG_random_geometric", random_geometric_graph),
    ]
    noisy_graphs = [
        (
            name,
            add_random_noise_edges(graph_factory(), f"{output_name}.edgelist"),
        )
        for name, output_name, graph_factory in graph_factories
    ]

    output_folder = os.path.join(
        project_dir, "results", "graphs", "sparsified", "triangle_path"
    )
    os.makedirs(output_folder, exist_ok=True)

    triangle_path_sparsifiers = [
        (
            "TrianglePathLoss",
            TrianglePathLossSparsifier,
            {
                "targetRatio": 0.5,
                "rho": 0.1,
                "triangle_weight": 0.5,
                "path_weight": 0.5,
                "min_path_length": 7,
            },
        )
    ]
    apply_sparsifiers(noisy_graphs, triangle_path_sparsifiers, output_folder)

    baseline_sparsifiers = [
        ("KNeighbor", KNeighborSparsifier, {"k": 5}),
        ("RankDegree", RankDegreeSparsifier, {"targetRatio": 0.5, "rho": 0.1}),
        ("Cut", CutSparsifier, {"targetRatio": 0.5}),
        ("RandomEdge", RandomEdgeSparsifier, {"targetRatio": 0.5}),
        ("LocalDegree", LocalDegreeSparsifier, {"targetRatio": 0.5}),
        ("ForestFire", ForestFireSparsifier, {"targetRatio": 0.5}),
        ("LocalSimilarity", LocalSimilaritySparsifier, {"targetRatio": 0.5}),
        (
            "EffR",
            EffR,
            {"epsilon": 0.1, "type": "kts", "tol": 1e-10, "precon": False},
        ),
    ]
    apply_sparsifiers(noisy_graphs, baseline_sparsifiers, output_folder)


if __name__ == "__main__":
    main()
