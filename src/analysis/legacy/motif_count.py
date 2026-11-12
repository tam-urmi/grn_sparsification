
import numpy as np
import networkx as nx
import pandas as pd
import os
import matplotlib.pyplot as plt
from itertools import combinations
import csv
import random
import itertools
from collections import Counter

path = os.getcwd()

file_path = path + '/unpruned/human_gene2.txt'
# Read the .txt file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Split each line by space and create a list of lists
data = [line.strip().split(" ") for line in lines]

# Convert the list of lists into a DataFrame
human_gene = pd.DataFrame(data)
human_gene

human_gene.columns = ['source', 'sink', 'weight']
# Create a directed graph (use nx.Graph() for an undirected graph)
G_human_gene = nx.DiGraph()

# Add edges to the graph with weights
for index, row in human_gene.iterrows():
    G_human_gene.add_edge(row['source'], row['sink'], weight=row['weight'])

### Human gene

## Forest Fire 0.5
# Open and read the .wel file
file_path = path + '/pruned/ForestFire/0.5/0/udw.wel'
human_gene_prune_0_5 = pd.read_csv(file_path, delim_whitespace=True)
human_gene_prune_0_5.columns = ['source', 'sink', 'weight']

# Create a directed graph (use nx.Graph() for an undirected graph)
G_human_gene_ff0_5 = nx.DiGraph()

# Add edges to the graph with weights
for index, row in human_gene_prune_0_5.iterrows():
    G_human_gene_ff0_5.add_edge(row['source'], row['sink'], weight=row['weight'])

## Forest Fire 0.9

# Open and read the .wel file
file_path = path + '/pruned/ForestFire/0.9/0/udw.wel'
human_gene_prune_0_9 = pd.read_csv(file_path, delim_whitespace=True)
human_gene_prune_0_9.columns = ['source', 'sink', 'weight']

# Create a directed graph (use nx.Graph() for an undirected graph)
G_human_gene_ff0_9 = nx.DiGraph()

# Add edges to the graph with weights
for index, row in human_gene_prune_0_9.iterrows():
    G_human_gene_ff0_9.add_edge(row['source'], row['sink'], weight=row['weight'])

## KNN 0.5
file_path = path + '/pruned/KNeighbor/0.5/0/udw.wel'
human_gene_prune_0_5 = pd.read_csv(file_path, delim_whitespace=True)
human_gene_prune_0_5.columns = ['source', 'sink', 'weight']

# Create a directed graph (use nx.Graph() for an undirected graph)
G_human_gene_knn0_5 = nx.DiGraph()

# Add edges to the graph with weights
for index, row in human_gene_prune_0_5.iterrows():
    G_human_gene_knn0_5.add_edge(row['source'], row['sink'], weight=row['weight'])

## KNN 0.95
file_path = path + '/pruned/KNeighbor/0.95/0/udw.wel'
human_gene_prune_0_95 = pd.read_csv(file_path, delim_whitespace=True)
human_gene_prune_0_95.columns = ['source', 'sink', 'weight']

# Create a directed graph (use nx.Graph() for an undirected graph)
G_human_gene_knn0_95 = nx.DiGraph()

# Add edges to the graph with weights
for index, row in human_gene_prune_0_95.iterrows():
    G_human_gene_knn0_95.add_edge(row['source'], row['sink'], weight=row['weight'])


## Random 0.5
file_path = path + '/pruned/Random/0.5/0/udw.wel'
human_gene_prune_0_5 = pd.read_csv(file_path, delim_whitespace=True)
human_gene_prune_0_5.columns = ['source', 'sink', 'weight']

# Create a directed graph (use nx.Graph() for an undirected graph)
G_human_gene_random0_5 = nx.DiGraph()

# Add edges to the graph with weights
for index, row in human_gene_prune_0_5.iterrows():
    G_human_gene_random0_5.add_edge(row['source'], row['sink'], weight=row['weight'])

## Random 0.9
file_path = path + '/pruned/Random/0.9/0/udw.wel'
human_gene_prune_0_9 = pd.read_csv(file_path, delim_whitespace=True)
human_gene_prune_0_9.columns = ['source', 'sink', 'weight']

# Create a directed graph (use nx.Graph() for an undirected graph)
G_human_gene_random0_9 = nx.DiGraph()

# Add edges to the graph with weights
for index, row in human_gene_prune_0_9.iterrows():
    G_human_gene_random0_9.add_edge(row['source'], row['sink'], weight=row['weight'])

## ER 0.5
file_path = path + '/pruned/ER/0.5/0/dw.wel'
human_gene_prune_0_5 = pd.read_csv(file_path, delim_whitespace=True)
human_gene_prune_0_5.columns = ['source', 'sink', 'weight']

# Create a directed graph (use nx.Graph() for an undirected graph)
G_human_gene_ER0_5 = nx.DiGraph()

# Add edges to the graph with weights
for index, row in human_gene_prune_0_5.iterrows():
    G_human_gene_ER0_5.add_edge(row['source'], row['sink'], weight=row['weight'])

## ER 0.95
file_path = path + '/pruned/ER/0.95/0/dw.wel'
human_gene_prune_0_95 = pd.read_csv(file_path, delim_whitespace=True)
human_gene_prune_0_95.columns = ['source', 'sink', 'weight']

# Create a directed graph (use nx.Graph() for an undirected graph)
G_human_gene_ER0_95 = nx.DiGraph()

# Add edges to the graph with weights
for index, row in human_gene_prune_0_95.iterrows():
    G_human_gene_ER0_95.add_edge(row['source'], row['sink'], weight=row['weight'])


def count_directed_motifs(graph):
    triads = nx.algorithms.triads.triadic_census(graph)
    triads_15 = {key: value for key, value in triads.items() if ((key != '003') & (key != '012'))}
    return triads_15

# Plot motif counts
def plot_motif_counts(motif_counts, title):
    plt.figure(figsize = (4, 3))
    plt.bar(motif_counts.keys(), motif_counts.values(), color = 'blue', alpha = 0.7)
    plt.xticks(rotation = 45)
    plt.xlabel('Motif type')
    plt.ylabel('Number of motifs')
    plt.title(title)
    plt.show()

# Count motifs in each graph
motif_counts_G1 = count_directed_motifs(G_human_gene)
motif_counts_G2 = count_directed_motifs(G_human_gene_ER0_5)
motif_counts_G3 = count_directed_motifs(G_human_gene_ff0_5)
motif_counts_G4 = count_directed_motifs(G_human_gene_knn0_5)
motif_counts_G5 = count_directed_motifs(G_human_gene_random0_5)


plot_motif_counts(motif_counts_G1, 'Human gene')
plot_motif_counts(motif_counts_G2, 'ER 0.5 prune')
plot_motif_counts(motif_counts_G3, 'FF 0.5 prune')
plot_motif_counts(motif_counts_G4, 'Knn 0.5 prune')
plot_motif_counts(motif_counts_G5, 'Random 0.5 prune')
