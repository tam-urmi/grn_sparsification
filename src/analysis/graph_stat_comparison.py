import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import get_cmap
import numpy as np
import os

# ---- SETTINGS ----
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
input_file = os.path.join(project_dir, "results", "tables", "graph_statistics.csv")
output_dir = os.path.join(project_dir, "results", "figures", "feature_subplots")
os.makedirs(output_dir, exist_ok=True)

# ---- METADATA COLUMNS ----
metadata_cols = [
    'graph_type', 'noise_type', 'sparsifier_type', 'state',
    'num_noise_edges', 'num_real_edges'
]

# ---- LOAD AND PROCESS ----
df = pd.read_csv(input_file)

# Group feature columns by first underscore
feature_groups = {}
for col in df.columns:
    if col not in metadata_cols:
        prefix = col.split('_')[0]
        feature_groups.setdefault(prefix, []).append(col)

# Get all unique pruned graph types
pruned_graph_types = df[df['state'] != 'original']['graph_type'].dropna().unique()

# ---- PLOT LOOP ----
for pruned_type in pruned_graph_types:
    if not pruned_type.startswith('pruned_'):
        continue

    # Extract base graph name (e.g., karate from pruned_karate)
    base_name = pruned_type.replace('pruned_', '')
    original_graph_type = f"{base_name}_graph"

    # Get data
    original = df[(df['graph_type'] == original_graph_type) & (df['state'] == 'original')]
    sparsified = df[(df['graph_type'] == pruned_type) & (df['state'] != 'original')]

    if original.empty and sparsified.empty:
        continue

    n_plots = len(feature_groups)
    fig, axs = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
    if n_plots == 1:
        axs = [axs]

    plotted_any = False

    for ax, (param, cols) in zip(axs, feature_groups.items()):
        cmap = get_cmap("Blues")
        plotted = False

        original_vals = original[cols].values.flatten()
        sparsifiers = sorted(sparsified['sparsifier_type'].dropna().unique())

        # --- BAR PLOT for scalar features ---
        if len(cols) == 1:
            values = []
            labels = []
            colors = []

            if original_vals.size > 0:
                values.append(original_vals.mean())
                labels.append("original")
                colors.append("black")
                plotted = True
                plotted_any = True

            for i, s_type in enumerate(sparsifiers):
                s_vals = sparsified[sparsified['sparsifier_type'] == s_type][cols].values.flatten()
                if s_vals.size > 0:
                    values.append(s_vals.mean())
                    labels.append(s_type)
                    color = 'red' if s_type == 'GRN_Sparsifier' else cmap((i + 1) / (len(sparsifiers) + 1))
                    colors.append(color)
                    plotted = True
                    plotted_any = True

            if plotted:
                ax.bar(labels, values, color=colors)
                ax.set_title(f"[Bar] {param} – {base_name}")
                ax.set_ylabel(param)
                ax.grid(True)
            else:
                ax.set_visible(False)

        # --- LOG-LOG Histogram for degree / clustering ---
        elif param in ['degree', 'clustering']:
            bins = 50

            def plot_log_hist(data, label, color, linestyle='-'):
                hist_vals, bin_edges = np.histogram(data, bins=bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                nonzero = hist_vals > 0
                ax.plot(bin_centers[nonzero], hist_vals[nonzero], label=label,
                        color=color, linestyle=linestyle, linewidth=2)

            if original_vals.size > 0:
                plot_log_hist(original_vals, 'original', 'black', linestyle='--')
                plotted = True
                plotted_any = True

            for i, s_type in enumerate(sparsifiers):
                s_vals = sparsified[sparsified['sparsifier_type'] == s_type][cols].values.flatten()
                if s_vals.size > 0:
                    color = 'red' if s_type == 'GRN_Sparsifier' else cmap((i + 1) / (len(sparsifiers) + 1))
                    plot_log_hist(s_vals, s_type, color)
                    plotted = True
                    plotted_any = True

            if plotted:
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_title(f"[Log-Log] {param} – {base_name}")
                ax.set_xlabel(param)
                ax.set_ylabel("Frequency")
                ax.legend()
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            else:
                ax.set_visible(False)

        # --- KDE for multi-column continuous features ---
        else:
            if original_vals.size > 0:
                sns.kdeplot(original_vals, ax=ax, color='black', linestyle='--', linewidth=2.5, label='original')
                plotted = True
                plotted_any = True

            for i, s_type in enumerate(sparsifiers):
                s_vals = sparsified[sparsified['sparsifier_type'] == s_type][cols].values.flatten()
                if s_vals.size > 0:
                    color = 'red' if s_type == 'GRN_Sparsifier' else cmap((i + 1) / (len(sparsifiers) + 1))
                    sns.kdeplot(s_vals, ax=ax, color=color, linewidth=2, label=s_type)
                    plotted = True
                    plotted_any = True

            if plotted:
                ax.set_title(f"{param} – {base_name}")
                ax.set_ylabel(param)
                ax.legend()
                ax.grid(True)
            else:
                ax.set_visible(False)

    if plotted_any:
        plt.tight_layout()
        plt.show()

    # if plotted_any:
    #     plt.tight_layout()
    #     filename = f"{base_name}_features_subplot.png".replace(" ", "_").lower()
    #     plt.savefig(os.path.join(output_dir, filename))
    #     plt.close()

# print(f"✅ All matched and plotted graphs saved to: {output_dir}")
