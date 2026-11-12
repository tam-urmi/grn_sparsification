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

## Project layout

- `src/data/` — graph preparation and the YEASTRACT downloader.
- `src/methods/` — baseline and neural sparsifier experiments.
- `src/analysis/` — graph-statistics and PCA analysis.
- `src/experimental/` — unfinished triangle/path prototype.
- `src/analysis/legacy/` — older scripts requiring legacy input layouts.
- `data/raw/` and `data/processed/` — local inputs and derived data; excluded
  from version control.
- `results/` — local graph outputs, metric tables, plots, and archived runs;
  excluded from version control.
- `lit/` — local papers, citations, reviews, and conference materials.
- `archive/` — local notebooks, duplicate code, and unrelated or superseded
  project material.
- `docs/WORKSPACE_LAYOUT.md` — detailed local organization and archive guide.
- `archive/baselines/sparsification-clone/` — the separate general-purpose
  baseline framework, not the GRN implementation. See the separate
  [baseline project](https://github.com/tam-urmi/sparsification).

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
workflow are consolidated. Old notebooks and environment snapshots are kept in
the local `archive/`, not in the published source tree.
