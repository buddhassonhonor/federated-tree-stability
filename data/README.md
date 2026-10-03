# Data sources

The release contains aggregate experiment results in `exp/` and `results/`, plus a synthetic seed-0 example in `synthetic_seed0.npz`. The generator in `src/run_supplementary_experiments.py` recreates the synthetic benchmark for each seed.

The synthetic file contains `train_X_0` through `train_X_7` and matching labels, and `ood_X`/`ood_y` for the two concatenated OOD environments. Arrays contain no participant records.

Public real-world inputs remain at their original sources:

- Adult: Barry Becker and Ronny Kohavi, UCI Machine Learning Repository, https://doi.org/10.24432/C5XW20 . Source page: https://archive.ics.uci.edu/dataset/2/adult .
- Statlog (German Credit Data): Hans Hofmann, UCI Machine Learning Repository, https://doi.org/10.24432/C5NC77 . Source page: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data .
- COMPAS: ProPublica, data for its Machine Bias analysis, https://github.com/propublica/compas-analysis . The input CSV is pinned to its upstream commit in `sources.json`.

`python scripts/download_data.py` retrieves the exact input files into `data/` and verifies their hashes. UCI lists Adult and German Credit under CC BY 4.0; consult the upstream COMPAS repository for its applicable terms. The project archive does not relicense upstream inputs.
