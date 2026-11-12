# GRN sparsification

Research on removing noise from synthetic and real gene-regulatory networks while
preserving their structural properties, especially triangle motifs, longer paths,
and connectedness.

## Current state

This repository is an experimental workspace, not yet a unified or validated
algorithm package. The current approaches and their limitations are summarized in
[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md). In particular, the
`TrianglePathLossSparsifier` prototype is a heuristic edge-selection routine,
not yet an optimized loss function, and it does not explicitly constrain
connectedness.

## Code map

- `sparsifier_TriPathLoss.py` — experimental graph generation, noise injection,
  baseline sparsifiers, and the triangle/path scoring prototype.
- `grn_sparsifier.py` — separate neural edge-ranking experiment with degree,
  modularity, and density penalties.
- `create_graphs.py` — synthetic and real graph preparation and noise injection.
- `sparsifier.py` — additional sparsifier prototypes.
- `motif_count.py` — motif-counting experiments.
- `src/` — graph-statistics and PCA analysis scripts.
- `data_sources/download_yeastract.py` — YEASTRACT data-download utility from
  the related `biogrn_sparse` work.
- `sparsification/` — a separate baseline framework; it is not the GRN
  sparsification implementation and is not included in this repository snapshot.

## Data and results

Raw networks, third-party datasets, notebooks with saved outputs, generated
graphs, figures, and experiment results are intentionally kept out of version
control. Obtain datasets from their original sources and follow their terms.
Local ignore rules are in `.gitignore`; review dataset provenance and licenses
before adding any data to this repository.

## Reproducibility

There is not yet a validated, shared environment or end-to-end test suite.
Dependencies and input file locations differ across the experimental scripts.
Treat the current code as research prototypes until the setup and evaluation
workflow are consolidated.
