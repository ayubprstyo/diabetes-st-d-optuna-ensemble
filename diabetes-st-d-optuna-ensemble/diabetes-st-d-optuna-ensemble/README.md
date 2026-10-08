# Discrete-aware SMOTE-Tomek, Optuna, and ensemble learning for diabetes risk prediction

Code and results for the manuscript

> **Optimizing diabetes risk prediction using discrete-aware SMOTE-Tomek, Optuna, and ensemble learning on health survey data**
> A. Prasetyo and F. Amalia. Submitted to *Jurnal Ilmiah Teknologi dan Rekayasa* (JITR), Universitas Gunadarma, 2026.

## Summary

Standard SMOTE-Tomek interpolates binary and ordinal survey features into fractional values that never occur in real respondents (for example, `HighBP = 0.37`). Tree-based models can isolate these artificial values, so balancing the training data barely changes their sensitivity. The discrete-aware variant (**ST-D**) rounds every synthetic value to a feasible value on the original scale before Tomek-link cleaning.

The study compares seven configurations for each of four ensemble models (XGBoost, LightGBM, CatBoost, and Random Forest) on 229 474 deduplicated records of the 2015 Behavioral Risk Factor Surveillance System (BRFSS):

| Code | Configuration | Resampling | Hyperparameters | Class weight |
|---|---|---|---|---|
| A | Baseline | None | Default | No |
| B | ST | Standard SMOTE-Tomek | Default | No |
| C | Optuna | None | Optuna (TPE) | No |
| D | ST + Optuna | Standard SMOTE-Tomek | Optuna (TPE) | No |
| E | CW + Optuna | None | Optuna (TPE) | Yes |
| F | **ST-D + Optuna (proposed)** | Discrete-aware SMOTE-Tomek | Optuna (TPE) | No |
| G | ST-D (ablation) | Discrete-aware SMOTE-Tomek | Default | No |

Main results on the 20% hold-out test set (n = 45 895):

| | Mean recall (4 models) | Mean F1 (4 models) |
|---|---|---|
| A Baseline | 0.175 | 0.265 |
| B Standard SMOTE-Tomek | 0.230 | 0.315 |
| G Discrete-aware SMOTE-Tomek | 0.658 | 0.440 |

The best proposed model (Random Forest, configuration F) reached recall 0.684, F1-score 0.457, and ROC-AUC 0.807. Full results, including PR-AUC, Brier score, and 95% bootstrap confidence intervals, are in [`results/`](results/).

## Leakage-free protocol

1. Remove 24 206 exact duplicate rows **before** splitting.
2. Stratified 80:20 train–test split (seed 42); the test set is used once.
3. Fit the z-score scaler on the training set only.
4. Resample (ST or ST-D) **only** training data, including the training part of every Optuna cross-validation fold.
5. Optuna TPE (seed 42), 3-fold stratified CV on a 30 000-row training subsample, 25 trials, objective = F1 of the positive class.

## Repository structure

```
notebooks/diabetes_st-d_optuna_ensemble.ipynb   Google Colab notebook (end-to-end, with QUICK mode)
src/run_experiment.py                           Script that produced the reported results
src/make_figures.py                             Reproduces Figures 1–5 of the manuscript
results/results.csv                             Test metrics for all 28 model × configuration runs
results/bootstrap_ci.csv                        95% bootstrap CIs (1000 replicates) for F1 and ROC-AUC
results/best_params.json                        Hyperparameters selected by Optuna
results/importance.csv                          Built-in and permutation importance (best proposed model)
results/info.json                               Data counts and resampling statistics
results/probs.npz                               Test-set labels and predicted probabilities for every run
figures/                                        Manuscript figures
data/README.md                                  How to obtain the dataset
```

## How to reproduce

**Option 1 – Google Colab.** Open the notebook in [`notebooks/`](notebooks/), run all cells. The dataset is downloaded from Kaggle with `kagglehub`, or you can upload the CSV manually. Set `QUICK = True` for a fast smoke test; the full run takes about one hour on a Colab CPU.

**Option 2 – Local.**

```bash
pip install -r requirements.txt
# download the CSV as described in data/README.md
python src/run_experiment.py --data data/diabetes_binary_health_indicators_BRFSS2015.csv --out results
python src/make_figures.py results figures
```

`--quick` runs on a 20 000-row subset with 2 Optuna trials for a smoke test. The reported results were produced on Python 3.11 with two CPU threads (`--n_jobs 2`); multithreaded gradient boosting can produce very small numerical differences on other hardware.

## Data

The data are the *Diabetes Health Indicators Dataset* by A. Teboul on Kaggle, a cleaned version of the 2015 BRFSS published by the U.S. Centers for Disease Control and Prevention. The dataset is not redistributed in this repository; see [`data/README.md`](data/README.md).

## Use of AI

An AI-based assistant (Claude, Anthropic) was used for programming assistance in developing and running the analysis code, for literature searching and reference checking, and for drafting and language editing of the manuscript. AI tools were not used to generate or modify data. The authors reviewed the code and results and take full responsibility for them.

## Citation

If you use this code, please cite the manuscript (citation details will be added after publication) and this repository; see [`CITATION.cff`](CITATION.cff).

## License

Code: MIT License (see [`LICENSE`](LICENSE)). The dataset is subject to the terms of its original sources.
