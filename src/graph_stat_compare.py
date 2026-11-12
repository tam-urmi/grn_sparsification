import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import eigvalsh
from collections import defaultdict
import ast

# Define the root directories
PRUNED_GRAPHS_DIR = "pruned_graphs"
ORIGINAL_GRAPH_DIR = "original_graph"

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

    # Degree distribution
    degrees = [d for _, d in G.degree()]
    bins = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    hist, _ = np.histogram(degrees, bins=bins)
    for i in range(len(bins) - 1):
        stats[f"degree_{bins[i]}_{bins[i+1]}"] = hist[i]

    # Triangle count
    stats["triangle_count"] = sum(nx.triangles(G).values()) // 3

    # Wedge count (paths of length 2)
    stats["wedge_count"] = sum(d * (d - 1) // 2 for d in degrees)

    # Clustering coefficient distribution
    clustering = list(nx.clustering(G).values())
    clustering_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(clustering, bins=clustering_bins)
    for i in range(len(clustering_bins) - 1):
        stats[f"clustering_{clustering_bins[i]}_{clustering_bins[i+1]}"] = hist[i]

    # Path length distribution
    path_stats = {}
    if nx.is_connected(G):
        lengths = dict(nx.shortest_path_length(G))
        path_lengths = [l for node in lengths for l in lengths[node].values() if l > 0]
        path_bins = range(1, max(path_lengths) + 2)
        hist, _ = np.histogram(path_lengths, bins=path_bins)
        for i in range(len(path_bins) - 1):
            path_stats[f"path_{path_bins[i]}_{path_bins[i+1]}"] = hist[i]

    # Laplacian spectrum
    laplacian_stats = {}
    laplacian = nx.laplacian_matrix(G).toarray()
    eigenvalues = eigvalsh(laplacian)
    min_eig, max_eig = min(eigenvalues), max(eigenvalues)
    spectrum_bins = np.arange(min_eig, max_eig + 10, 10)
    hist, _ = np.histogram(eigenvalues, bins=spectrum_bins)
    for i in range(len(spectrum_bins) - 1):
        laplacian_stats[f"laplacian_{spectrum_bins[i]:.1f}_{spectrum_bins[i+1]:.1f}"] = hist[i]

    return {**stats, **path_stats, **laplacian_stats}

def process_graphs():
    data = []

    # Process original graphs
    for file in os.listdir(ORIGINAL_GRAPH_DIR):
        if file.endswith(".edgelist"):
            graph_type = file.split(".")[0]
            G = load_graph_from_edgelist(os.path.join(ORIGINAL_GRAPH_DIR, file))
            stats = compute_graph_statistics(G)
            stats.update({"graph_type": graph_type, "noisy_type": "NA", "sparsifier_type": "NA"})
            data.append(stats)

    # Process pruned graphs
    for graph_folder in os.listdir(PRUNED_GRAPHS_DIR):
        graph_path = os.path.join(PRUNED_GRAPHS_DIR, graph_folder)
        if os.path.isdir(graph_path):
            parts = graph_folder.split("_")
            if len(parts) > 2:
                noisy_type = "_".join(parts[0:1])
                graph_type = "_".join(parts[2:])
                for file in os.listdir(graph_path):
                    if file.endswith(".edgelist"):
                        sparsifier_type = file.split(".")[0]
                        G = load_graph_from_edgelist(os.path.join(graph_path, file))
                        stats = compute_graph_statistics(G)
                        stats.update({
                            "graph_type": graph_type,
                            "noisy_type": noisy_type,
                            "sparsifier_type": sparsifier_type
                        })
                        data.append(stats)

    df = pd.DataFrame(data)
    column_order = ["graph_type", "noisy_type", "sparsifier_type"] + \
                   sorted([col for col in df.columns if col.startswith("path_")]) + \
                   sorted([col for col in df.columns if col.startswith("laplacian_")]) + \
                   [col for col in df.columns if col not in ("graph_type", "noisy_type", "sparsifier_type") and not col.startswith("path_") and not col.startswith("laplacian_")]

    df = df[column_order]
    return df

# Generate and display the DataFrame
df_graphs = process_graphs()
df_graphs.to_csv("graph_statistics.csv", index=False)

# Generate and display the DataFrame
print(df_graphs.columns[0:20])
print(df_graphs.columns[21:40])
print(df_graphs.columns[41:60])
print(df_graphs.columns[61:80])
print(df_graphs.columns[81:103])
print(df_graphs.head())
