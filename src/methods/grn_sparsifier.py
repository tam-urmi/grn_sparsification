
import os
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.preprocessing import MinMaxScaler
from networkx.algorithms.community import greedy_modularity_communities

# ---- PATHS ----
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
noisy_graph_dir = os.path.join(base_dir, "results", "graphs", "noisy")
pruned_graphs_dir = os.path.join(
    base_dir, "results", "graphs", "sparsified", "neural_v3"
)

# ---- UTILITIES ----
def save_graph(graph, edge_labels, folder, filename):
    os.makedirs(folder, exist_ok=True)
    pkl_path = os.path.join(folder, f"{filename}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump((graph, edge_labels), f)

    edgelist_path = os.path.join(folder, f"{filename}.edgelist")
    with open(edgelist_path, "w") as f:
        for u, v in graph.edges():
            label = edge_labels.get((u, v), edge_labels.get((v, u), "real"))
            f.write(f"{u} {v} {label}\n")


def load_noisy_graphs():
    graphs = {}
    if not os.path.exists(noisy_graph_dir):
        return graphs
    for file in os.listdir(noisy_graph_dir):
        if file.endswith(".pkl"):
            with open(os.path.join(noisy_graph_dir, file), "rb") as f:
                G = pickle.load(f)
                if isinstance(G, tuple):
                    G, labels = G
                    for (u, v), label in labels.items():
                        if G.has_edge(u, v):
                            G[u][v]['label'] = label
                graphs[file.replace(".pkl", "")] = G
    return graphs


# ---- NEURAL NETWORK ----
class GRNEdgeSparsifierNN(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(16, 1)
        self.w1 = nn.Parameter(torch.randn(1))
        self.w2 = nn.Parameter(torch.randn(1))
        self.w3 = nn.Parameter(torch.randn(1))

    def forward(self, x):
        weighted_score = self.w1 * x[:, 0] + self.w2 * x[:, 1] + self.w3 * x[:, 2]
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x + weighted_score.unsqueeze(1)


# ---- FEATURES ----
def compute_edge_scores(G):
    G_undirected = G.to_undirected()
    triangles = nx.triangles(G_undirected)
    betweenness = nx.edge_betweenness_centrality(G_undirected)
    articulation_points = set(nx.articulation_points(G_undirected))

    edge_features = []
    for u, v in G.edges():
        tri_score = min(triangles.get(u, 0), triangles.get(v, 0))
        bet_score = betweenness.get((u, v), betweenness.get((v, u), 0.0))
        ap_score = int(u in articulation_points) + int(v in articulation_points)
        edge_features.append([tri_score, bet_score, ap_score])
    return np.array(edge_features)


# ---- PENALTIES ----
def compute_theoretical_penalties(G, lambda_deg=1.0, lambda_mod=1.0, lambda_density=1.0):
    degrees = np.array([d for _, d in G.degree()])
    if len(degrees) == 0:
        return 1.0, 1.0, 1.0

    hist, bins = np.histogram(degrees, bins='auto', density=True)
    bins_center = (bins[:-1] + bins[1:]) / 2
    log_bins = np.log(bins_center + 1e-10)
    log_hist = np.log(hist + 1e-10)
    slope, _ = np.polyfit(log_bins, log_hist, 1)
    estimated_exponent = -slope

    deg_penalty = lambda_deg * (max(0, 2.5 - estimated_exponent) + max(0, estimated_exponent - 3))
    communities = list(greedy_modularity_communities(G))
    modularity = nx.algorithms.community.modularity(G, communities)
    mod_penalty = lambda_mod * max(0, 0.3 - modularity)

    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = 2 * m / (n * (n - 1)) if n > 1 else 0
    density_penalty = lambda_density * max(0, density - 0.07)

    return deg_penalty, mod_penalty, density_penalty


# ---- SPARSIFIER ----
def grn_nn_sparsifier(
    G, sparsity_ratio=0.5, lr=0.01,
    lambda_deg=1.0, lambda_mod=1.0, lambda_density=1.0,
    tol=1e-4, patience=30, max_epochs=1000
):
    edge_features = compute_edge_scores(G)
    scaler = MinMaxScaler()
    X_np = scaler.fit_transform(edge_features)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(np.sum(edge_features, axis=1), dtype=torch.float32).unsqueeze(1)

    model = GRNEdgeSparsifierNN(input_dim=3)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    edge_list = list(G.edges())
    num_keep = int(sparsity_ratio * len(edge_list))

    loss_history = []

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X).squeeze()
        loss_main = loss_fn(preds, y.squeeze())

        with torch.no_grad():
            scores_np = preds.detach().cpu().numpy()
        top_idx = np.argsort(scores_np)[-num_keep:]
        kept_edges = [edge_list[i] for i in top_idx]

        G_sparse = nx.Graph()
        G_sparse.add_edges_from(kept_edges)
        deg_penalty, mod_penalty, density_penalty = compute_theoretical_penalties(G_sparse, lambda_deg, lambda_mod, lambda_density)

        total_loss = loss_main + deg_penalty + mod_penalty + density_penalty
        total_loss.backward()
        optimizer.step()

        loss_history.append(total_loss.item())
        if len(loss_history) > patience and np.std(loss_history[-patience:]) < tol:
            print(f"Early stopping at epoch {epoch}")
            break

    with torch.no_grad():
        final_scores = model(X).squeeze().numpy()
    top_idx = np.argsort(final_scores)[-num_keep:]
    pruned = nx.Graph()
    edge_labels = {}
    for i in top_idx:
        u, v = edge_list[i]
        label = G[u][v].get("label", "real")
        pruned.add_edge(u, v, label=label)
        edge_labels[(u, v)] = label

    return pruned, edge_labels


# ---- PROCESS ----
def process_noisy_graphs(graph_dict, sparsifiers):
    for graph_name, G in graph_dict.items():
        print(f"\nProcessing {graph_name}")
        graph_folder = os.path.join(pruned_graphs_dir, graph_name)
        os.makedirs(graph_folder, exist_ok=True)
        for sparsifier_name, sparsifier_func, kwargs in sparsifiers:
            print(f"  - Applying {sparsifier_name}")
            pruned_graph, edge_labels = sparsifier_func(G, **kwargs)
            save_graph(pruned_graph, edge_labels, graph_folder, sparsifier_name)


# ---- MAIN ----
def main():
    noisy_graphs = load_noisy_graphs()
    sparsifiers = [
        ("GRN_Sparsifier", grn_nn_sparsifier, {
            "sparsity_ratio": 0.5,
            "lr": 0.02,
            "lambda_deg": 0.5,
            "lambda_mod": 1.0,
            "lambda_density": 1.0,
            "tol": 1e-4,
            "patience": 30,
            "max_epochs": 1000
        })
    ]
    process_noisy_graphs(noisy_graphs, sparsifiers)

if __name__ == "__main__":
    main()
