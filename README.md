# Spatial Water Consumption Modelling & Multi-Method Anomaly Detection

A reproducible Python case study for **data-readiness assessment, leakage-aware spatial modelling, failure analysis, and unsupervised anomaly screening** on LSOA-level domestic water-consumption data.

The project deliberately separates what the data can support from what they cannot. The available source is cross-sectional, so this repository does **not** present the work as temporal water-demand forecasting.

## Why this project

The aim is to answer two practical questions:

1. Is the available spatial/contextual information sufficient to outperform simple baselines for LSOA-level consumption modelling?
2. Without labelled anomalies, can complementary unsupervised methods identify a defensible shortlist of unusual consumption records for investigation?

A central finding is that a more complex model does not automatically create useful predictive information. In the reference analysis, the leakage-controlled Gradient Boosting benchmark did not outperform the simple mean baseline, while multi-method anomaly screening produced a more actionable result. The negative predictive result is retained rather than tuned away.

## Methods

- data-quality audit and structurally empty-column detection;
- identifier exclusion and explicit target-leakage safeguards;
- mean and median dummy baselines;
- exogenous/context-only Gradient Boosting benchmark when usable features exist;
- leakage-controlled spatial-context Gradient Boosting benchmark;
- held-out MAE, RMSE, R-squared and prediction-dispersion diagnostics;
- residual and top-failure-case analysis;
- held-out permutation importance;
- IQR, Local Outlier Factor and Isolation Forest anomaly screening;
- pairwise Jaccard/overlap analysis and consensus anomaly voting;
- limited scale-aware robustness perturbation on genuine context features only.

## Reference results

The preceding leakage-controlled reference run used 454 records and found that three source fields (`DATA_SOURCE`, `Year`, `NUMBER_OF_METERS`) were structurally empty.

Predictive modelling showed limited data readiness:

- mean-baseline MAE: ~17,989;
- spatial-context GBM MAE: ~19,038;
- spatial-context GBM R-squared: ~-0.12;
- prediction-versus-actual slope: ~0.09.

The model therefore compressed predictions toward the mean and failed particularly on extreme-consumption cases.

For anomaly screening with a 5% LOF/Isolation Forest screening assumption:

- IQR: 13 flagged records;
- LOF: 23;
- Isolation Forest: 23;
- consensus (at least two methods): 20;
- all three methods: 4.

These are **screening candidates**, not confirmed errors. No anomaly accuracy/F1 is claimed because labelled ground truth is unavailable.

> **Interpretation:** the available cross-sectional data support anomaly screening and data-readiness diagnosis more convincingly than reliable fine-grained consumption prediction.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   └── water_consumption_analysis.py
├── data/
│   └── README.md
├── figures/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── docs/
│   └── methodology.md
└── tests/
    └── test_analysis.py
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For tests:

```bash
pip install pytest
```

## Data

The source CSV is intentionally not included. See [`data/README.md`](data/README.md) for the expected schema and licensing note.

Example local path:

```text
data/Portsmouth_Water_Domestic_Consumption_25-26.csv
```

## Run the analysis

```bash
python src/water_consumption_analysis.py --data data/Portsmouth_Water_Domestic_Consumption_25-26.csv
```

Optional parameters:

```bash
python src/analysis.py \
  --data data/Portsmouth_Water_Domestic_Consumption_25-26.csv \
  --test-size 0.20 \
  --contamination 0.05 \
  --random-state 42 \
  --output-dir outputs \
  --figure-dir figures
```

`--contamination` is a screening assumption for LOF and Isolation Forest. It must not be interpreted as an estimate of true anomaly prevalence.

## Generated outputs

A run creates, among others:

- `figures/water_consumption_dashboard.png`
- `outputs/model_benchmarks.csv`
- `outputs/anomaly_method_agreement.csv`
- `outputs/consensus_anomalies.csv`
- `outputs/permutation_importance.csv`
- `outputs/failure_cases_top5.csv`
- `outputs/test_predictions.csv`
- `outputs/run_config.json`

Generated outputs are ignored by default so a repository does not accidentally publish source-derived records. Curated, non-sensitive summary artefacts can be committed deliberately after checking the source-data licence.

## Tests

```bash
pytest -q
```

The tests use small synthetic fixtures strictly for unit-testing deterministic software behaviour. The analysis pipeline itself does not generate synthetic fallback data.

## Scientific safeguards

The project intentionally avoids several common failure modes:

- no target-derived synthetic meter proxy;
- no random month/season variables;
- no row identifier as a predictor;
- no test-target information in spatial encodings;
- no anomaly accuracy/F1 without labelled anomalies;
- no claim of temporal forecasting from cross-sectional data;
- no claim that a consensus anomaly is automatically an erroneous observation.

See [`docs/methodology.md`](docs/methodology.md) for details.

## Limitations and next step

The main limitation is sparse independent explanatory information. Genuine forecasting would require repeated timestamped consumption histories, chronological/backtesting validation, and richer context where available and appropriately governed.

This repository should therefore be read as an example of **honest model benchmarking and anomaly screening under data limitations**, rather than as a production water-demand forecasting system.

## Licence

Code is released under the MIT License. Third-party/source data are not covered by the software licence; verify the original dataset licence before redistribution.
