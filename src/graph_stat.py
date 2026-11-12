import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import eigvalsh
from collections import defaultdict
import ast
from grakel import GraphKernel

# Define the root directories
PRUNED_GRAPHS_DIR = "pruned_graphs"
ORIGINAL_GRAPH_DIR = "original_graph"
NOISY_GRAPHS_DIR = "noisy_graph"

def load_graph_from_edgelist(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                node1, node2 = parts[:2]
                attr_str = ' '.join(parts[2:])
                try:
                    attributes = ast.literal_eval(attr_str) if attr_str else {}
                except (SyntaxError, ValueError):
                    attributes = {}
                G.add_edge(node1, node2, **attributes)
    return G

def compute_graph_statistics(G):
    stats = {}
    degrees = [d for _, d in G.degree()]
    bins = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    hist, _ = np.histogram(degrees, bins=bins)
    for i in range(len(bins) - 1):
        stats[f"degree_{bins[i]}_{bins[i+1]}"] = hist[i]
    stats["triangle_count"] = sum(nx.triangles(G).values()) // 3
    stats["wedge_count"] = sum(d * (d - 1) // 2 for d in degrees)
    clustering = list(nx.clustering(G).values())
    clustering_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(clustering, bins=clustering_bins)
    for i in range(len(clustering_bins) - 1):
        stats[f"clustering_{clustering_bins[i]}_{clustering_bins[i+1]}"] = hist[i]
    path_stats = {}
    FIXED_PATH_BINS = range(1, 21)
    if nx.is_connected(G):
        lengths = dict(nx.shortest_path_length(G))
        path_lengths = [l for node in lengths for l in lengths[node].values() if l > 0]
        hist, _ = np.histogram(path_lengths, bins=list(FIXED_PATH_BINS) + [FIXED_PATH_BINS[-1] + 1])
        for i in range(len(FIXED_PATH_BINS)):
            key = f"path_{FIXED_PATH_BINS[i]}_{FIXED_PATH_BINS[i]+1}"
            path_stats[key] = int(hist[i]) if i < len(hist) else 0
    else:
        for i in range(len(FIXED_PATH_BINS)):
            key = f"path_{FIXED_PATH_BINS[i]}_{FIXED_PATH_BINS[i]+1}"
            path_stats[key] = 0
    laplacian_stats = {}
    laplacian = nx.laplacian_matrix(G).toarray()
    eigenvalues = eigvalsh(laplacian)
    if len(eigenvalues) == 0:
        spectrum_bins = [0, 10]
        laplacian_stats["laplacian_0.0_10.0"] = 0
    else:
        min_eig, max_eig = min(eigenvalues), max(eigenvalues)
        spectrum_bins = np.arange(min_eig, max_eig + 10, 10)
        hist, _ = np.histogram(eigenvalues, bins=spectrum_bins)
        for i in range(len(spectrum_bins) - 1):
            key = f"laplacian_{spectrum_bins[i]:.1f}_{spectrum_bins[i+1]:.1f}"
            laplacian_stats[key] = hist[i]
        for i in range(len(spectrum_bins) - 1):
            key = f"laplacian_{spectrum_bins[i]:.1f}_{spectrum_bins[i+1]:.1f}"
            laplacian_stats.setdefault(key, 0)
    full_stats = {**stats, **path_stats, **laplacian_stats}
    for k, v in full_stats.items():
        if isinstance(v, float) and np.isnan(v):
            full_stats[k] = 0
        elif isinstance(v, np.float64) and np.isnan(v):
            full_stats[k] = 0
    return full_stats

def extract_adjacency_features(G):
    features = {}
    n = G.number_of_nodes()
    m = G.number_of_edges()
    max_edges = n * (n - 1) / 2 if not G.is_directed() else n * (n - 1)
    features["adj_density"] = m / max_edges if max_edges else 0.0
    isolated_nodes = [node for node, degree in dict(G.degree()).items() if degree == 0]
    features["adj_isolation_ratio"] = len(isolated_nodes) / n if n else 0.0
    return features

def noise_reduction_stats(G):
    edge_labels = nx.get_edge_attributes(G, 'label')
    real_edges = sum(1 for label in edge_labels.values() if label == 'real')
    noise_edges = sum(1 for label in edge_labels.values() if label == 'noise')
    return {
        "num_real_edges": real_edges,
        "num_noise_edges": noise_edges
    }

def generate_wl_embeddings(graph_list, wl_embedding_size=15):
    formatted_graphs = []
    for G in graph_list:
        G = G.to_undirected()
        labels = {v: str(G.degree(v)) for v in G.nodes()}
        nx.set_node_attributes(G, labels, "label")
        adj_dict = {v: [] for v in G.nodes()}
        for u, v in G.edges():
            adj_dict[u].append(v)
            adj_dict[v].append(u)
        node_labels = {v: G.nodes[v]['label'] for v in G.nodes()}
        formatted_graphs.append((adj_dict, node_labels))
    gk = GraphKernel(kernel=[{"name": "weisfeiler_lehman", "n_iter": 3}, {"name": "subtree_wl"}], normalize=True)
    embeddings = gk.fit_transform(formatted_graphs)
    padded_embeddings = []
    for emb in embeddings:
        emb = emb[:wl_embedding_size]
        emb = np.pad(emb, (0, wl_embedding_size - len(emb)), constant_values=0)
        padded_embeddings.append(emb)
    return np.array(padded_embeddings)

def process_graphs():
    data = []
    all_graphs = []
    meta_info = []
    def add_graph(graph, graph_type, noise_type, sparsifier_type, state):
        stats = compute_graph_statistics(graph)
        adj_feats = extract_adjacency_features(graph)
        noise_feats = noise_reduction_stats(graph)
        features = {**stats, **adj_feats, **noise_feats}
        all_graphs.append(graph)
        meta_info.append({
            "graph_type": graph_type,
            "noise_type": noise_type,
            "sparsifier_type": sparsifier_type,
            "state": state,
            "features": features
        })
    for file in os.listdir(ORIGINAL_GRAPH_DIR):
        if file.endswith(".edgelist"):
            graph_type = file.split(".")[0]
            G = load_graph_from_edgelist(os.path.join(ORIGINAL_GRAPH_DIR, file))
            add_graph(G, graph_type, "NA", "NA", "original")
    for graph_folder in os.listdir(PRUNED_GRAPHS_DIR):
        graph_path = os.path.join(PRUNED_GRAPHS_DIR, graph_folder)
        if os.path.isdir(graph_path):
            parts = graph_folder.split("_")
            if len(parts) > 2:
                noise_type = "_".join(parts[0:1])
                graph_type = "pruned_" + "_".join(parts[2:])
                for file in os.listdir(graph_path):
                    if file.endswith(".edgelist"):
                        sparsifier_type = file.split(".")[0]
                        G = load_graph_from_edgelist(os.path.join(graph_path, file))
                        add_graph(G, graph_type, noise_type, sparsifier_type, "pruned")
    for file in os.listdir(NOISY_GRAPHS_DIR):
        if file.endswith(".edgelist"):
            filename = file.split(".")[0]
            if "_" in filename:
                noise_type, graph_type = filename.split("_", 1)
            else:
                noise_type = "unknown"
                graph_type = filename
            G = load_graph_from_edgelist(os.path.join(NOISY_GRAPHS_DIR, file))
            add_graph(G, graph_type, noise_type, "NA", "noised")
    wl_embeddings = generate_wl_embeddings(all_graphs, wl_embedding_size=15)
    for i, meta in enumerate(meta_info):
        emb = wl_embeddings[i]
        for j, val in enumerate(emb):
            meta["features"][f"wl_embed_{j}"] = val
        data.append({**meta["features"],
                     "graph_type": meta["graph_type"],
                     "noise_type": meta["noise_type"],
                     "sparsifier_type": meta["sparsifier_type"],
                     "state": meta["state"]})
    df = pd.DataFrame(data)
    column_order = ["graph_type", "noise_type", "sparsifier_type", "state"] + \
                   sorted([col for col in df.columns if col not in ("graph_type", "noise_type", "sparsifier_type", "state")])
    df = df[column_order]
    return df

# Run processing and save
if __name__ == "__main__":
    df_graphs = process_graphs()
    df_graphs = df_graphs.fillna(0)
    df_graphs.to_csv("graph_statistics.csv", index=False)
    print("Saved graph statistics.")
