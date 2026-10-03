# FITI: cross-client stability for federated decision trees

Artifact version: **1.0.0**.

Author: **Zhigao Huang**. Code is licensed under MIT (`LICENSE`). Original result data, synthetic data, figures, and non-software documentation are licensed under CC BY 4.0 (`LICENSE-DATA`). Third-party inputs retain their upstream terms. Citation metadata is provided in `CITATION.cff`.

Supports the manuscript **Beyond predictive gain: cross-client stability for interpretable federated decision trees**. FITI scores candidate rules by impurity gain minus cross-client variance in conditional label probability shifts.

## Contents

- `src/`: benchmark implementation, feature diagnostics, statistical comparisons, and plots.
- `exp/`: archived primary, weighting/voting, and fixed-penalty diagnostic results.
- `results/`: accuracy summaries derived from the archived seed-level results.
- `figures/`: six images used by the manuscript's five figures.
- `data/sources.json`: source URLs and SHA-256 hashes of the public input files.
- `data/synthetic_seed0.npz`: reproducible synthetic example with train and OOD arrays; seeds 0–9 can be regenerated from the supplied loader.
- `release-manifest.json`: file sizes and SHA-256 hashes.

## Environment and quick checks

Use Python 3.11 or newer. Install the dependencies in an isolated environment:

```sh
python -m venv .venv
# Activate the environment using your platform's standard command.
python -m pip install -r requirements.txt
python scripts/verify_artifact.py
python scripts/export_results.py
python src/compute_statistical_table.py
python src/plot_revision_figures.py
python scripts/smoke_synthetic.py
```

Run commands from the repository root. `requirements-verified.txt` records the Python 3.13 packages used for artifact verification; it is distinct from the manuscript's reported historical environment. Archived results are preserved rather than replaced by a new benchmark run.

## Public datasets and experiment reruns

```sh
python scripts/download_data.py
# Data defaults to ./data. FITI_DATA_DIR can point to an alternative directory.
python src/run_revision_experiments.py
python src/run_priority2_experiments.py
python src/feature_selection_frequency.py
```

The benchmark command runs seven settings and 10 seeds, including tuned pooled baselines, and overwrites `exp/supplementary_results.json`; use a separate working copy to preserve archived results. The priority experiment writes its own weighting/voting result file. The feature diagnostic uses fixed lambda 5 for Synthetic and 0.5 for Adult Spurious, rather than the main benchmark's validation-selected penalties. The earlier `run_supplementary_experiments.py` is included as a loader dependency; its standalone experiment mode uses a different selection procedure and should not be used to reproduce the primary table.

The Adult and German Credit inputs are available from UCI; COMPAS is available from ProPublica. The archive includes source instructions and aggregate results, not third-party person-level records. Upstream terms apply to downloaded inputs. File hashes identify the inputs used here.

## Implementation details and evidence scope

The primary archived implementation uses three threshold percentiles (25, 50, 75). The fixed-penalty feature diagnostic uses nine percentiles. The primary routine assigns zero gain and shift to a client whose candidate split has an empty child, and uses Laplace-smoothed shifts when both children are populated. These are properties of the released implementation and are not silently changed during packaging.

The loaders use ordinal encoding for categorical values. Adult textual missing-value markers remain encoded categories; COMPAS loaders discard records missing required fields. Adult Spurious fits its encoder on the original Adult training split before repartitioning the concatenated official training and test files; this does not strictly isolate encoder fitting from the subsequently constructed OOD clients. Natural-partition loaders fit encoders on their training groups. The supplied implementation and archived outputs should be assessed with these preprocessing details in view.

`iid_accuracy` uses subsets of training records and is not an independent held-out test metric. Main claims concern OOD accuracy. With two natural-partition clients, penalty selection trains on one client and validates on the other, leaving zero cross-client variance at the validation-training stage. Repeated-seed intervals characterize benchmark variability, not population replication. Pooled references, forest extensions, and weighting experiments have separate conditions; their result files are not interchangeable.

## Changes for portability

Only data-directory resolution, documentation, dependency declarations, and artifact verification/export utilities were added during packaging. Tree-scoring logic and archived benchmark result files are unchanged. Paths default to the repository's data directory and can be overridden with FITI_DATA_DIR or config.yaml. Manuscript and editorial materials, credentials, caches, and raw third-party inputs are excluded.
