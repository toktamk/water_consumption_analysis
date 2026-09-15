# Methodology

## Scope

This project is a **spatial/cross-sectional water-consumption modelling and anomaly-screening case study**. It is deliberately not described as temporal water-demand forecasting because the supplied data do not contain usable repeated timestamps.

## 1. Data audit

The pipeline checks required columns, duplicates, missingness, target type, structurally empty columns, and basic target distribution. Row identifiers are excluded from predictive modelling.

In the reference run, the source contained 454 records. `DATA_SOURCE`, `Year`, and `NUMBER_OF_METERS` were entirely missing. After identifier removal, the independent explanatory information was therefore very limited.

## 2. Leakage control

No predictor is manufactured directly from `TOTAL_CONSUMPTION`. Random month/season variables and target-derived meter proxies are prohibited.

For the optional spatial-context benchmark, group-level target summaries are handled as follows:

- training rows use leave-one-out group summaries, so a row does not directly encode its own target;
- held-out rows receive group summaries fitted on the training partition only;
- unseen groups fall back to training-global statistics.

These summaries are still target-derived contextual features. Therefore the project reports a separate exogenous/context-only benchmark and does not present the spatial-context model as a purely exogenous forecasting model.

## 3. Baselines and predictive benchmark

The pipeline evaluates:

1. dummy mean baseline;
2. dummy median baseline;
3. Gradient Boosting using only genuine available context features, when such features exist;
4. Gradient Boosting using the leakage-controlled spatial-context summaries.

Metrics are MAE, RMSE, and R-squared. A prediction-versus-actual slope is used only as a continuous diagnostic of compression/dispersion; it is not described as probabilistic calibration.

### Reference finding

In the preceding leakage-controlled reference run, the spatial-context GBM did not outperform the simple mean baseline: MAE was approximately 19,038 versus 17,989 and R-squared was approximately -0.12. The prediction-versus-actual slope was approximately 0.09. These results indicate strong regression toward the mean and insufficient independent information for reliable LSOA-level prediction.

The repository intentionally preserves this negative result. Model complexity is not treated as evidence of model value.

## 4. Failure analysis

The held-out residuals and largest absolute errors are inspected explicitly. In the reference run, high-consumption observations were often substantially underpredicted, while some low-consumption observations were overpredicted. This is consistent with the narrow prediction range visible in the actual-versus-predicted diagnostic.

Permutation importance is computed on the held-out set using MAE degradation. It is interpreted cautiously because correlated or target-derived contextual features can share importance.

## 5. Anomaly detection

Three complementary definitions are compared:

- **IQR rule**: global univariate extremes in observed consumption;
- **Local Outlier Factor (LOF)**: local-density anomalies after robust scaling;
- **Isolation Forest**: partition-based multivariate anomaly screening.

The LOF/Isolation Forest contamination parameter is an explicit screening assumption, not an estimate of real-world anomaly prevalence.

Because no gold-standard anomaly labels are available, the project does **not** report anomaly accuracy, precision, recall, F1, or AUROC. Instead it reports method counts, pairwise Jaccard agreement, pairwise overlap, anomaly votes, and consensus cases.

### Reference finding

With a 5% screening assumption in the reference run:

- IQR flagged 13 records;
- LOF flagged 23;
- Isolation Forest flagged 23;
- 20 records were supported by at least two methods;
- 11 records were shared by LOF and Isolation Forest;
- 4 records were supported by all three methods.

A consensus flag is an investigation priority, not proof that a record is erroneous.

## 6. Robustness

Where genuine continuous context features are available, the pipeline perturbs those features with small scale-aware noise and re-evaluates MAE. Target-derived spatial summaries are not perturbed and used to make inflated robustness claims.

With the current sparse feature set, this check is intentionally limited and should not be interpreted as comprehensive robustness validation.

## 7. What the data support

The current data support:

- reproducible data-quality auditing;
- leakage-aware spatial benchmarking;
- residual and failure-mode analysis;
- multi-method anomaly screening;
- identification of data-readiness limitations.

They do not by themselves support:

- genuine temporal forecasting;
- causal claims;
- labelled anomaly-classification performance;
- production deployment claims;
- strong prediction of fine-grained consumption from rich exogenous drivers.

## 8. Next data needed for forecasting

A genuine operational forecasting extension would require repeated timestamped consumption histories and chronological/backtesting validation. Useful candidate context could include weather, exposure/meter counts where appropriate, calendar effects, property/household context, operational events, and consistent spatial-temporal provenance, subject to availability and governance.
