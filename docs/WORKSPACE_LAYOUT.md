# Local workspace layout

The public repository contains source code and documentation. Raw/processed
data, generated results, literature, and archived material remain local and
are excluded by `.gitignore`.

```text
src/
  data/           graph preparation and data acquisition
  methods/        baseline and neural sparsifier prototypes
  analysis/       graph statistics and PCA
    legacy/       scripts tied to historical input layouts
  experimental/   unfinished triangle/path experiment
data/
  raw/
    grn/          E. coli, C. elegans, human, and TRRUST inputs
    synthetic/    small plain/noisy example networks
    tcga_cancer/  local TCGA/PANDA network files from biogrn_sparse
    external/     inputs whose provenance still needs classification
  processed/
results/
  graphs/         original, noisy, and sparsified graph snapshots
  tables/         metrics and experiment tables
  figures/        plots from analysis runs
  archived/       historical generated results
lit/
  papers/
  citations/
  reviews/
  conferences/
archive/
  notebooks/      experiment notebooks, including saved outputs
  baselines/      separate upstream sparsification checkout
  third-party/    Grenadine source/tutorial copy
  dependencies/   old environment snapshots
  legacy-code/
  presentations/
  personal/       unrelated personal documents; never publish
  other-research/
```

## Result handling

- The former `pruned_graphs*`, `noisy_graph`, and `original_graph` folders are
  preserved under `results/graphs/`.
- The prior `results30jan2026` tree is under
  `results/archived/2026-01-30_31/`. Its `FILE_INDEX.md` documents intended
  deliverables but does not match the files actually present; treat the result
  claims as unverified until rerun.
- The compressed C. elegans input was byte-identical to the plain-text input;
  the duplicate is retained under `archive/raw-duplicates/`.
- Top-level generated metric CSVs and figures were moved into
  `results/tables/` and `results/figures/`.
- Notebooks are retained in `archive/notebooks/` because they contain saved
  outputs and depend on older paths. Their contents have not been scrubbed or
  made public.

The existing `myenv/` remains at the workspace root: moving a virtual
environment can break its interpreter paths. Rebuild a fresh environment from
a reviewed dependency specification before relying on it.

The large TCGA/PANDA files report their original logical sizes but currently
have no local disk blocks allocated in iCloud Drive. They may need to download
before tools can read them.
