# Project status

## Research objective

Develop a sparsification method that removes spurious edges from noisy synthetic
and biological networks while preserving useful structure. The intended
objective includes triangle motifs, longer paths, and connectedness.

## What exists

- **Synthetic and real-network preparation:** `create_graphs.py` builds several
  synthetic graph families, loads real GRN inputs, and adds randomly sampled
  noise edges.
- **Baseline comparisons:** `sparsifier_TriPathLoss.py` contains multiple
  baseline sparsifiers and an experimental triangle/path edge-scoring routine.
- **Neural ranking experiment:** `grn_sparsifier.py` ranks edges from triangle,
  betweenness, and articulation-point features, then applies degree,
  modularity, and density penalties.
- **Evaluation:** scripts in `src/` analyze graph statistics and PCA summaries;
  notebook experiments and result files remain local pending curation.
- **Biological-network acquisition:** `biogrn_sparse` currently contributes a
  YEASTRACT downloader. Its local TCGA network files are data, not source code,
  and are excluded from publication.
- **Baseline framework:** the cluster `sparsification` directory has the same
  general-purpose framework layout as the separate nested `sparsification/`
  checkout. Keep it as a referenced baseline rather than merging or copying its
  unrelated upstream code and history. Its public project is
  [tam-urmi/sparsification](https://github.com/tam-urmi/sparsification).

## Main algorithm is still a prototype

The current `TrianglePathLossSparsifier` name overstates what is implemented:

- It selects edges by iterative neighborhood expansion; it does not optimize a
  differentiable or otherwise explicit loss.
- Its triangle contribution is currently based on a whole-graph motif count,
  rather than the motif change attributable to each candidate edge.
- It has no explicit connectedness term or constraint.
- The code path has unresolved integration issues (including input-path and
  NetworkX/NetworKit conversion assumptions), so it should not be treated as a
  reproducible end-to-end method yet.

The neural experiment is a separate line of work. Its listed penalties cover
degree distribution, modularity, and density; it does not currently implement
the intended long-path objective or an explicit connectedness constraint.

## Recommended next research milestone

1. Define the graph model and corruption protocol separately for directed and
   undirected data.
2. Specify measurable terms for motif preservation, long-path preservation,
   connectedness, and sparsity, including normalization and weights.
3. Implement a small import-safe core with deterministic toy-graph tests for
   each objective term and edge cases such as disconnected inputs.
4. Compare the method with the existing baselines at matched edge budgets on
   synthetic graphs before running the biological datasets.
5. Record dataset provenance, dependency versions, random seeds, and evaluation
   outputs in a reproducible experiment manifest.

These are follow-up research tasks, not claims that the current prototypes
already satisfy them.
