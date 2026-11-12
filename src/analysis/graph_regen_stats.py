
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns
import scipy.stats as stats
import netrd as nd
from netrd.distance import JaccardDistance

# Reload the CSV file
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# df = pd.read_csv(file_path)

# Define file paths
original_graph_path = os.path.join(
    project_dir, "results", "graphs", "original", "ecoli_graph.edgelist"
)
pruned_graph_path = os.path.join(
    project_dir,
    "results",
    "graphs",
    "sparsified",
    "v3",
    "random_noisyG_ecoli",
    "GRN_Sparsifier.edgelist",
)

# Load the original graph normally
try:
    G_original = nx.read_edgelist(original_graph_path, nodetype=str, data=True)
except Exception as e:
    print(f"Error loading original graph: {e}")

# Manually parse the pruned graph
G_pruned = nx.Graph()
try:
    with open(pruned_graph_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue  # Skip invalid lines

            node1, node2 = parts[:2]  # First two are node names
            edge_attrs = {}

            # Try to parse additional attributes
            if len(parts) > 2:
                try:
                    edge_attrs = eval(" ".join(parts[2:]))  # Convert string dictionary to dict
                    if not isinstance(edge_attrs, dict):
                        edge_attrs = {}
                except:
                    edge_attrs = {}

            G_pruned.add_edge(node1, node2, **edge_attrs)

except Exception as e:
    print(f"Error manually parsing pruned graph: {e}")

# Print sample edges to verify proper loading
print("Original Graph Sample Edges:", list(G_original.edges(data=True))[:5])
print("Pruned Graph Sample Edges:", list(G_pruned.edges(data=True))[:5])


# Extract degree distribution from the original graph
degree_sequence = [d for n, d in G_original.degree()]

# Generate 1000 configuration model graphs
config_models = [nx.configuration_model(degree_sequence) for _ in range(1000)]

# Initialize distance measures
jaccard = JaccardDistance()
hamming = nd.distance.Hamming()
polynomial_dissimilarity = nd.distance.PolynomialDissimilarity()
frobenius = nd.distance.Frobenius()

# Store distances
jaccard_distances = []
hamming_distances = []
polynomial_dissimilarity_distances = []
frobenius_distances = []

# Compute network distances for each configuration model
for i, config_graph in enumerate(config_models):
    # Convert multigraph to simple graph
    config_graph = nx.Graph(config_graph)
    config_graph.remove_edges_from(nx.selfloop_edges(config_graph))

    try:
        jaccard_distances.append(jaccard.dist(G_original, config_graph))
    except Exception as e:
        jaccard_distances.append(None)
        print(f"Error computing Jaccard distance for graph {i}: {e}")

    try:
        hamming_distances.append(hamming.dist(G_original, config_graph))
    except Exception as e:
        hamming_distances.append(None)
        print(f"Error computing Hamming distance for graph {i}: {e}")

    try:
        polynomial_dissimilarity_distances.append(polynomial_dissimilarity.dist(G_original, config_graph))
    except Exception as e:
        polynomial_dissimilarity_distances.append(None)
        print(f"Error computing Polynomial Dissimilarity distance for graph {i}: {e}")

    try:
        frobenius_distances.append(frobenius.dist(G_original, config_graph))
    except Exception as e:
        frobenius_distances.append(None)
        print(f"Error computing Frobenius distance for graph {i}: {e}")

# Plot the distances
plt.figure(figsize=(10, 5))
plt.plot(range(1, 1001), jaccard_distances, marker='o', linestyle='-', color='blue', alpha=0.7, label="Jaccard Distance")
plt.plot(range(1, 1001), hamming_distances, marker='s', linestyle='--', color='red', alpha=0.7, label="Hamming Distance")
plt.plot(range(1, 1001), polynomial_dissimilarity_distances, marker='^', linestyle='-.', color='green', alpha=0.7, label="Polynomial Dissimilarity Distance")
plt.plot(range(1, 1001), frobenius_distances, marker='d', linestyle=':', color='purple', alpha=0.7, label="Frobenius Distance")
plt.xlabel("Configuration Model Graphs (1-1000)")
plt.ylabel("Distance")
plt.title("Network Distances between Configuration Model Graphs and Original Ecoli Graph")
plt.legend()
plt.grid()
plt.show()
