import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns
import scipy.stats as stats
import textwrap
import matplotlib.colors as mcolors

# --- FILE PATHS ---
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(project_dir, "results", "tables", "graph_statistics.csv")
plot_folder = os.path.join(project_dir, "results", "figures", "PCA_trend_analysis")
loading_plot_dir = os.path.join(plot_folder, "top_features")

# Create necessary folders
os.makedirs(plot_folder, exist_ok=True)
os.makedirs(loading_plot_dir, exist_ok=True)
os.makedirs(os.path.join(project_dir, "results", "tables"), exist_ok=True)

# --- LOAD DATA ---
df = pd.read_csv(data_path)
print(df.columns)
feature_cols = df.columns.difference(["graph_type", "noise_type", "sparsifier_type", "state", "num_real_edges", "num_noise_edges"])
X = df[feature_cols]

from sklearn.preprocessing import StandardScaler  # Add this near other imports

# --- SCALE AND NORMALIZE DATA BEFORE PCA ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- RUN PCA ---
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Determine number of components to retain ≥95% variance (capped at 10)
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
num_components = min(np.argmax(explained_variance_ratio >= 0.95) + 2, 10)
print(f"Selected top {num_components} components (≥95% variance explained)")

# Extract top components
X_pca_top = X_pca[:, :num_components]


# --- CONSTRUCT PCA RESULT DATAFRAME ---
print("DF", df.tail(15))
top_pc_df = df[["graph_type", "noise_type", "sparsifier_type", "state"]].copy()
for i in range(num_components):
    top_pc_df[f"PC{i+1}"] = X_pca_top[:, i]

print("TOP PC DF:", top_pc_df.tail())
top_pc_df.to_csv(
    os.path.join(project_dir, "results", "tables", "top_pc_df.csv")
)

# --- PLOT TOP FEATURE LOADINGS PER PC ---
loadings = np.abs(pca.components_[:num_components])
loading_df = pd.DataFrame(loadings, columns=feature_cols, index=[f"PC{i+1}" for i in range(num_components)])

for pc_idx in loading_df.index:
    top_features = loading_df.loc[pc_idx].sort_values(ascending=False).head(num_components)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features.values, y=top_features.index, hue = top_features.index, palette="viridis")
    plt.title(f"Top {num_components} Feature Contributions to {pc_idx}")
    plt.xlabel("Absolute Loading Value")
    plt.ylabel("Feature")
    plt.tight_layout()

    plot_path = os.path.join(loading_plot_dir, f"{pc_idx}_top_features.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")

# --- PLOTTING SECTION ---


# Plot PCA components for each graph type**

# Define color scheme
pruned_colors = sns.color_palette("husl", n_colors=len(top_pc_df['sparsifier_type'].unique()))  # Unique colors for sparsifiers
noised_colors = sns.light_palette("blue", n_colors=len(top_pc_df['noise_type'].unique()))  # Different shades of blue

# Number of PCA components to compare (now set to 4)
num_pca_compare = min(4, num_components)

# Compute the **global maximum absolute PCA value** for normalization
max_pca_value = np.abs(top_pc_df.iloc[:, 4:4+num_pca_compare]).max().max()


# --- PCA DIFFERENCE PLOTS  ---

# Clean graph names to remove "_graph" for consistent matching
top_pc_df["graph_type_clean"] = top_pc_df["graph_type"].str.replace("_graph", "", regex=False)

pca_diff_data = []
original_graphs = top_pc_df[top_pc_df["state"] == "original"]

# --- ORIGINAL vs NOISED ---
for _, original_row in original_graphs.iterrows():
    original_graph_name = original_row["graph_type_clean"]
    original_pca_values = original_row[[f"PC{i+1}" for i in range(num_pca_compare)]].values

    # Match based on everything after first "_" in noised name
    matched_noised_graphs = top_pc_df[
        (top_pc_df["state"] == "noised") &
        (top_pc_df["graph_type"].apply(lambda x: "_".join(x.split("_")[1:])) == original_graph_name)
    ]

    for _, noised_row in matched_noised_graphs.iterrows():
        noised_pca_values = noised_row[[f"PC{i+1}" for i in range(num_pca_compare)]].values
        diff_original_noised = np.abs(original_pca_values - noised_pca_values)
        print("noised diff", diff_original_noised)

        pca_diff_data.append({
            "graph_type": noised_row["graph_type"],
            "original_graph": original_row["graph_type"],
            "noise_type": noised_row["noise_type"],
            "sparsifier_type": noised_row["sparsifier_type"],
            "state": "Original vs Noised",
            **{f"PCA_{i+1}_diff": diff_original_noised[i] for i in range(num_pca_compare)}
        })

# --- ORIGINAL vs PRUNED ---
for _, original_row in original_graphs.iterrows():
    original_graph_name = original_row["graph_type"]
    original_graph_name_clean = original_row["graph_type_clean"]
    original_pca_values = original_row[[f"PC{i+1}" for i in range(num_pca_compare)]].values

    matched_pruned_graphs = top_pc_df[
        (top_pc_df["state"] == "pruned") &
        (top_pc_df["graph_type"].str.startswith("pruned_")) &
        (top_pc_df["graph_type"].str.endswith(original_graph_name_clean))
    ]

    if matched_pruned_graphs.empty:
        continue

    # Normalize across all matched + original graphs
    all_pca = [original_pca_values]
    for _, row in matched_pruned_graphs.iterrows():
        all_pca.append(row[[f"PC{i+1}" for i in range(num_pca_compare)]].values)
    max_pca_value = np.max(np.abs(np.concatenate(all_pca)))

    original_pca_values_norm = original_pca_values / max_pca_value

    for _, pruned_row in matched_pruned_graphs.iterrows():
        pruned_pca_values = pruned_row[[f"PC{i+1}" for i in range(num_pca_compare)]].values / max_pca_value
        diff_original_pruned = np.abs(original_pca_values_norm - pruned_pca_values)
        print("pruned diff", diff_original_pruned)

        pca_diff_data.append({
            "graph_type": pruned_row["graph_type"],
            "original_graph": original_graph_name,
            "noise_type": pruned_row["noise_type"],
            "sparsifier_type": pruned_row["sparsifier_type"],
            "state": "Original vs Pruned",
            **{f"PCA_{i+1}_diff": diff_original_pruned[i] for i in range(num_pca_compare)}
        })

# Create final DataFrame
pca_diff_df = pd.DataFrame(pca_diff_data)
print("Final PCA Difference DataFrame:")
print(pca_diff_df.head())
print(pca_diff_df.shape)



# Save to CSV
diff_csv_path = os.path.join(
    project_dir, "results", "tables", "PCA_difference_analysis.csv"
)
pca_diff_df.to_csv(diff_csv_path, index=False)
print(f"Saved PCA difference data to: {diff_csv_path}")

# breakpoint()

# Define two colormaps: Green for Random Noise, Brown for Pref Noise
random_cmap = mcolors.LinearSegmentedColormap.from_list("random_green", ["lightgreen", "darkgreen"])
pref_cmap = mcolors.LinearSegmentedColormap.from_list("pref_brown", ["sandybrown", "saddlebrown"])

# Get unique pruned graph types
pruned_graph_types = top_pc_df[top_pc_df["graph_type"].str.startswith("pruned_")]["graph_type"].unique()
print(pruned_graph_types)

for graph_type in pruned_graph_types:
    print(graph_type)
    df_graph = pca_diff_df[(pca_diff_df["graph_type"] == graph_type) & (pca_diff_df["state"] == "Original vs Pruned")]
    print(df_graph.shape)
    print(df_graph)

    # Extract original graph name from the pruned graph type
    original_graph_name = graph_type.replace("pruned_", "")

    # Count number of pruned variations
    num_pruned_variations = df_graph.shape[0]

    plt.figure(figsize=(10, 8))

    # To store label positions and avoid overlaps, tracking both noise types together
    label_positions = []
    pca_columns = [f"PCA_{i+1}_diff" for i in range(num_pca_compare)]

    for idx, (_, row) in enumerate(df_graph.iterrows()):
        pca_diff_values = row[pca_columns].values.astype(float)

        noise_type = row["noise_type"].replace(" ", "_")  # Ensure filename compatibility
        sparsifier_type = row["sparsifier_type"]

        # Choose colormap based on noise_type
        if "random" in row["noise_type"].lower():
            color = random_cmap(idx / max(1, num_pruned_variations - 1))  # Gradient in green
        else:  # Assume pref noise if not random
            color = pref_cmap(idx / max(1, num_pruned_variations - 1))  # Gradient in brown

        # Special styling for GRN_Sparsifier
        if "GRN_Sparsifier" in sparsifier_type:
            linestyle = "--"  # Dashed line for visibility
            linewidth = 2  # Slightly thicker line
            marker = "s"  # Square marker
        else:
            linestyle = "-"
            linewidth = 1
            marker = "o"

        # Plot the PCA difference values
        plt.plot(pca_diff_values, linestyle=linestyle, marker=marker, color=color, alpha=0.8, linewidth=linewidth)

        # # Adjust text position to avoid overlap (considering both random and pref labels)
        # x_pos = len(pca_diff_values) - 2  # Move labels slightly to the left
        # y_pos = pca_diff_values[-1]

        # # Keep labels inside the plot area
        # y_pos = max(y_min + padding, min(y_pos, y_max - padding))

        # # Ensure spacing between labels (considering all noise types together)
        # offset = 0.03 * (y_max - y_min)  # More space for readability
        # while any(abs(y_pos - existing_y) < offset for existing_y in label_positions):
        #     y_pos += offset if idx % 2 == 0 else -offset  # Alternate movement up/down

        # label_positions.append(y_pos)  # Store adjusted position

        # Add text label for the sparsifier type with an arrow pointing to the graph
        plt.annotate(
            sparsifier_type,
            xy=(len(pca_diff_values) - 1, pca_diff_values[-1]),  # Arrow points to last data point
            # xytext=(x_pos + 0.5, y_pos),  # Label positioned slightly to the left
            fontsize=8, color=color, fontweight="bold",
            ha="right", va="center",
            arrowprops=dict(facecolor=color, edgecolor='gray', arrowstyle="->", lw=1)
        )

    # Formatting
    plt.xlabel("PCA Component Index")
    plt.ylabel("PCA Difference (Original - Pruned)")
    plt.title(f"PCA Difference: {original_graph_name} (Original vs Pruned)")

    plt.tight_layout()  # Ensure proper layout

    # Save the figure with noise in the filename
    safe_noise_type = df_graph["noise_type"].iloc[0].replace(" ", "_")
    diff_plot_filename = os.path.join(plot_folder, f"PCA_Difference_{original_graph_name}_{safe_noise_type}_Pruned.png")
    plt.savefig(diff_plot_filename, dpi=300, bbox_inches="tight")

    plt.show()
    print(f"Saved PCA Difference plot: {diff_plot_filename}")
