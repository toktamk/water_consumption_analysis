import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
TARGET = 'TOTAL_CONSUMPTION'
ID_COL = 'LSOA_CODE'
OBJECT_ID_CANDIDATES = {'ObjectId', 'OBJECTID', 'OBJECT_ID', 'FID', 'ID', 'Id', 'id'}


# ============================================================
# 1. DATA LOADING AND AUDIT
# ============================================================
def load_and_audit(file_path: str) -> pd.DataFrame:
    """Load the real Portsmouth dataset and print a compact data-quality audit."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found: {file_path}. "
            "Provide the real Portsmouth dataset; this script does not generate synthetic fallback data."
        )

    df = pd.read_csv(file_path)

    print('=' * 72)
    print('1. DATA AUDIT')
    print('=' * 72)
    print(f'Rows: {df.shape[0]} | Columns: {df.shape[1]}')
    print(f'Duplicate rows: {df.duplicated().sum()}')

    if TARGET not in df.columns:
        raise ValueError(f"Required target column '{TARGET}' not found.")
    if ID_COL not in df.columns:
        raise ValueError(f"Required identifier column '{ID_COL}' not found.")

    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    print('\nMissing values:')
    print(missing if len(missing) else 'No missing values')

    all_missing = [c for c in df.columns if df[c].isna().all()]
    if all_missing:
        print('\nStructurally empty columns excluded from modelling:')
        print(all_missing)

    print('\nTarget summary:')
    print(df[TARGET].describe())

    return df


# ============================================================
# 2. SAFE BASE FEATURES
# ============================================================
def prepare_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create only source-derived/non-target features.

    Key safeguards:
    - no estimated meter count derived from TOTAL_CONSUMPTION;
    - no random month/season variables;
    - all-missing columns are excluded from modelling;
    - row identifiers such as ObjectId are not treated as predictive variables.
    """
    out = df.copy()
    out[ID_COL] = out[ID_COL].astype(str)

    # Coarse spatial grouping based only on the source identifier.
    # With the current data this produces 18 groups and is useful as contextual structure.
    out['LSOA_District'] = out[ID_COL].str.slice(0, 7)
    out['district_record_count'] = out.groupby('LSOA_District')[ID_COL].transform('count').astype(float)

    excluded = {TARGET, ID_COL, 'LSOA_District'} | OBJECT_ID_CANDIDATES
    banned_tokens = ('consumption', 'target', 'actual', 'predicted', 'residual')

    exogenous_numeric = []
    for col in out.columns:
        if col in excluded:
            continue
        if any(tok in col.lower() for tok in banned_tokens):
            continue
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue
        if out[col].notna().sum() == 0:
            continue
        if out[col].nunique(dropna=True) <= 1:
            continue
        exogenous_numeric.append(col)

    # district_record_count is legitimate and may be constant only in unusual datasets;
    # keep it if it has information.
    if (
        'district_record_count' not in exogenous_numeric
        and out['district_record_count'].nunique(dropna=True) > 1
    ):
        exogenous_numeric.append('district_record_count')

    print('\nUsable exogenous numeric source/context features:')
    print(exogenous_numeric if exogenous_numeric else 'None beyond spatial group structure')

    out.attrs['exogenous_numeric'] = exogenous_numeric
    return out


# ============================================================
# 3. LEAKAGE-CONTROLLED SPATIAL TARGET ENCODING
# ============================================================
def fit_spatial_encoder(train_df: pd.DataFrame) -> dict:
    """Fit group-level target summaries on TRAIN ONLY for application to validation/test."""
    global_stats = {
        'global_mean': float(train_df[TARGET].mean()),
        'global_median': float(train_df[TARGET].median()),
        'global_std': float(train_df[TARGET].std(ddof=0)),
    }

    grp = train_df.groupby('LSOA_District')[TARGET].agg(['mean', 'median', 'std', 'count'])
    grp = grp.rename(columns={
        'mean': 'district_mean_consumption',
        'median': 'district_median_consumption',
        'std': 'district_std_consumption',
        'count': 'district_count_target_encoder',
    })

    return {'global': global_stats, 'group': grp}


def apply_spatial_encoder(df: pd.DataFrame, encoder: dict) -> pd.DataFrame:
    """Apply train-fitted spatial summaries to validation/test data."""
    out = df.copy()
    grp = encoder['group']
    glob = encoder['global']

    out = out.join(grp, on='LSOA_District')
    out['district_mean_consumption'] = out['district_mean_consumption'].fillna(glob['global_mean'])
    out['district_median_consumption'] = out['district_median_consumption'].fillna(glob['global_median'])
    out['district_std_consumption'] = out['district_std_consumption'].fillna(glob['global_std'])
    out['district_count_target_encoder'] = out['district_count_target_encoder'].fillna(0.0)
    return out


def make_leave_one_out_training_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build leakage-controlled training target encodings.

    A standard train-group mean still contains each training row's own target. For the training
    rows we therefore use leave-one-out (LOO) group means/variances so a row does not directly
    encode itself. Test rows still use summaries fitted from the training set only.

    For the median, exact leave-one-out computation is used because the dataset is small (454 rows).
    """
    out = train_df.copy()
    glob_mean = float(train_df[TARGET].mean())
    glob_median = float(train_df[TARGET].median())
    glob_std = float(train_df[TARGET].std(ddof=0))

    g = train_df.groupby('LSOA_District')[TARGET]
    count = g.transform('count').astype(float)
    total = g.transform('sum').astype(float)
    total_sq = train_df.groupby('LSOA_District')[TARGET].transform(lambda s: np.sum(np.square(s))).astype(float)

    denom = count - 1.0
    loo_mean = (total - train_df[TARGET]) / denom.replace(0, np.nan)

    # Population variance of the group after removing the current row:
    # E[X^2] - E[X]^2. It is stable and sufficient as a contextual spread feature.
    loo_second_moment = (total_sq - np.square(train_df[TARGET])) / denom.replace(0, np.nan)
    loo_var = loo_second_moment - np.square(loo_mean)
    loo_var = loo_var.clip(lower=0)
    loo_std = np.sqrt(loo_var)

    # Exact LOO median; dataset is small enough that clarity is preferable to micro-optimisation.
    loo_median = pd.Series(index=train_df.index, dtype=float)
    for _, idx in train_df.groupby('LSOA_District').groups.items():
        idx = list(idx)
        vals = train_df.loc[idx, TARGET]
        if len(idx) <= 1:
            loo_median.loc[idx] = glob_median
        else:
            for row_idx in idx:
                loo_median.loc[row_idx] = vals.drop(index=row_idx).median()

    out['district_mean_consumption'] = loo_mean.fillna(glob_mean)
    out['district_median_consumption'] = loo_median.fillna(glob_median)
    out['district_std_consumption'] = pd.Series(loo_std, index=train_df.index).fillna(glob_std)
    out['district_count_target_encoder'] = denom.clip(lower=0).fillna(0.0)
    return out


# ============================================================
# 4. ANOMALY DETECTION
# ============================================================
def detect_anomalies(df: pd.DataFrame, context_cols: list[str]) -> pd.DataFrame:
    """
    Compare complementary unsupervised anomaly definitions:
      1) IQR: global univariate extreme consumption;
      2) LOF: local-density anomaly;
      3) Isolation Forest: multivariate partition-based anomaly.

    No accuracy/F1 is claimed because there are no gold-standard anomaly labels.
    """
    out = df.copy()

    # 1) Global univariate IQR flag
    q1 = out[TARGET].quantile(0.25)
    q3 = out[TARGET].quantile(0.75)
    iqr = q3 - q1
    out['is_iqr_outlier'] = (
        (out[TARGET] < q1 - 1.5 * iqr) |
        (out[TARGET] > q3 + 1.5 * iqr)
    ).astype(int)

    # Context matrix. Target is permitted here because anomaly detection describes unusual
    # observed consumption, not prediction of the target.
    model_cols = [TARGET] + [c for c in context_cols if c in out.columns and c != TARGET]
    model_cols = list(dict.fromkeys(model_cols))

    X = out[model_cols].replace([np.inf, -np.inf], np.nan).copy()
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    # Drop unusable columns after imputation (all-NaN or constant).
    usable = [c for c in X.columns if X[c].notna().all() and X[c].nunique(dropna=True) > 1]
    X = X[usable]
    if len(X.columns) == 0:
        raise ValueError('No usable numeric features available for anomaly detection.')

    # LOF is distance-based, so use robust scaling.
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    contamination = 0.05  # fixed, interpretable comparison: approximately 5% candidate anomalies

    n_neighbors = min(20, max(5, len(X) // 20))
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
    )
    lof_pred = lof.fit_predict(X_scaled)
    out['is_lof_outlier'] = (lof_pred == -1).astype(int)
    out['lof_score'] = -lof.negative_outlier_factor_

    iforest = IsolationForest(
        n_estimators=500,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    if_pred = iforest.fit_predict(X_scaled)
    out['is_iforest_outlier'] = (if_pred == -1).astype(int)
    out['iforest_score'] = -iforest.score_samples(X_scaled)

    out['anomaly_votes'] = (
        out['is_iqr_outlier'] +
        out['is_lof_outlier'] +
        out['is_iforest_outlier']
    )
    out['is_consensus_anomaly'] = (out['anomaly_votes'] >= 2).astype(int)

    # Agreement diagnostics useful in an interview/report.
    pair_lof_if = ((out['is_lof_outlier'] == 1) & (out['is_iforest_outlier'] == 1)).sum()
    all_three = (out['anomaly_votes'] == 3).sum()

    print('\n' + '=' * 72)
    print('2. ANOMALY DETECTION SUMMARY')
    print('=' * 72)
    print(f'Features used for LOF/Isolation Forest: {usable}')
    print(f'IQR outliers: {out["is_iqr_outlier"].sum()}')
    print(f'LOF outliers (5% contamination): {out["is_lof_outlier"].sum()}')
    print(f'Isolation Forest outliers (5% contamination): {out["is_iforest_outlier"].sum()}')
    print(f'LOF + Isolation Forest agreement: {pair_lof_if}')
    print(f'All three methods agree: {all_three}')
    print(f'Consensus anomalies (>=2 methods): {out["is_consensus_anomaly"].sum()}')

    return out


# ============================================================
# 5. MODELLING PIPELINE
# ============================================================
def train_and_evaluate(df: pd.DataFrame):
    """
    Spatial cross-sectional consumption modelling, not temporal forecasting.

    The model uses:
    - genuine exogenous/context numeric columns only if non-empty and non-ID;
    - leakage-controlled group target summaries:
        * leave-one-out encodings for training rows;
        * train-only group summaries for test rows.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    # Leakage-controlled train/test encoding.
    encoder = fit_spatial_encoder(train_df)
    train_enc = make_leave_one_out_training_features(train_df)
    test_enc = apply_spatial_encoder(test_df, encoder)

    # Genuine numeric exogenous/context variables.
    exogenous_cols = []
    excluded_exact = {
        TARGET, ID_COL, 'LSOA_District',
        'is_iqr_outlier', 'is_lof_outlier', 'is_iforest_outlier',
        'is_consensus_anomaly', 'anomaly_votes', 'lof_score', 'iforest_score',
    } | OBJECT_ID_CANDIDATES
    banned_tokens = ('consumption', 'target', 'actual', 'predicted', 'residual')

    for c in train_enc.columns:
        if c in excluded_exact:
            continue
        if any(tok in c.lower() for tok in banned_tokens):
            continue
        if not pd.api.types.is_numeric_dtype(train_enc[c]):
            continue
        if train_enc[c].notna().sum() == 0:
            continue
        if train_enc[c].nunique(dropna=True) <= 1:
            continue
        exogenous_cols.append(c)

    # Explicitly include the legitimate non-target-derived group size if informative.
    if (
        'district_record_count' in train_enc.columns
        and train_enc['district_record_count'].nunique(dropna=True) > 1
        and 'district_record_count' not in exogenous_cols
    ):
        exogenous_cols.append('district_record_count')

    spatial_cols = [
        'district_mean_consumption',
        'district_median_consumption',
        'district_std_consumption',
        'district_count_target_encoder',
    ]

    feature_cols = []
    for c in exogenous_cols + spatial_cols:
        if c in train_enc.columns and c not in feature_cols:
            feature_cols.append(c)

    print('\nUnique spatial groups in training data:', train_enc['LSOA_District'].nunique())
    print('Model features:', feature_cols)

    X_train = train_enc[feature_cols].copy()
    y_train = train_enc[TARGET].astype(float).copy()
    X_test = test_enc[feature_cols].copy()
    y_test = test_enc[TARGET].astype(float).copy()

    # Strict train-only imputation. Drop columns for which the training median is NaN.
    train_medians = X_train.median(numeric_only=True)
    valid_features = [c for c in feature_cols if c in train_medians.index and pd.notna(train_medians[c])]
    dropped = [c for c in feature_cols if c not in valid_features]
    if dropped:
        print('Dropped unusable/all-missing model features:', dropped)

    X_train = X_train[valid_features].fillna(train_medians[valid_features])
    X_test = X_test[valid_features].fillna(train_medians[valid_features])
    feature_cols = valid_features

    if X_train.isna().any().any() or X_test.isna().any().any():
        bad_train = X_train.columns[X_train.isna().any()].tolist()
        bad_test = X_test.columns[X_test.isna().any()].tolist()
        raise ValueError(f'NaN remained after train-only imputation. Train: {bad_train}; Test: {bad_test}')

    if len(feature_cols) == 0:
        raise ValueError('No valid predictive features remain after leakage and missingness safeguards.')

    # Baseline: training mean.
    baseline_pred = np.repeat(y_train.mean(), len(y_test))
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=3,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Calibration-style slope for continuous predictions. Treat this as a diagnostic,
    # not as probabilistic calibration.
    if np.std(y_pred) > 0:
        slope, intercept = np.polyfit(y_pred, y_test, 1)
    else:
        slope, intercept = np.nan, np.nan

    results = test_enc[[ID_COL, 'LSOA_District', TARGET]].copy()
    results['predicted'] = y_pred
    results['residual'] = results[TARGET] - results['predicted']
    results['absolute_error'] = results['residual'].abs()

    print('\n' + '=' * 72)
    print('3. MODEL PERFORMANCE')
    print('=' * 72)
    print(f'Baseline MAE:   {baseline_mae:.2f}')
    print(f'Baseline RMSE:  {baseline_rmse:.2f}')
    print(f'GBM Test MAE:   {mae:.2f}')
    print(f'GBM Test RMSE:  {rmse:.2f}')
    print(f'GBM Test R^2:   {r2:.4f}')
    print(f'Prediction-vs-actual slope: {slope:.3f} | intercept: {intercept:.2f}')

    improvement = 100 * (baseline_mae - mae) / baseline_mae if baseline_mae else np.nan
    print(f'MAE improvement vs mean baseline: {improvement:.2f}%')

    # Held-out permutation importance.
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=30,
        random_state=RANDOM_STATE,
        scoring='neg_mean_absolute_error',
    )
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance_mean': perm.importances_mean,
        'importance_std': perm.importances_std,
    }).sort_values('importance_mean', ascending=False)

    # Robustness: perturb only exogenous/context features, not target-encoded spatial statistics.
    rng = np.random.default_rng(RANDOM_STATE)
    robust_cols = [c for c in exogenous_cols if c in X_test.columns]
    if robust_cols:
        X_test_noisy = X_test.copy()
        stds = X_train[robust_cols].std(ddof=0).replace(0, 1.0)
        noise = rng.normal(0, 0.05, size=(len(X_test_noisy), len(robust_cols))) * stds.to_numpy()
        X_test_noisy.loc[:, robust_cols] = X_test_noisy[robust_cols].to_numpy() + noise
        noisy_pred = model.predict(X_test_noisy)
        noisy_mae = mean_absolute_error(y_test, noisy_pred)
        print('\nRobustness under 5% scale-aware noise on exogenous/context features:')
        print(f'Noisy MAE: {noisy_mae:.2f} | Delta MAE: {noisy_mae - mae:.2f}')
    else:
        noisy_mae = np.nan
        print('\nRobustness noise test skipped: no genuine continuous exogenous feature is available.')
        print('This is preferable to perturbing target-derived spatial encodings and overstating robustness.')

    worst_cases = results.nlargest(5, 'absolute_error')
    print('\nTop 5 failure cases:')
    print(worst_cases[[ID_COL, TARGET, 'predicted', 'residual', 'absolute_error']])

    metrics = {
        'baseline_mae': baseline_mae,
        'baseline_rmse': baseline_rmse,
        'gbm_mae': mae,
        'gbm_rmse': rmse,
        'gbm_r2': r2,
        'mae_improvement_percent_vs_baseline': improvement,
        'prediction_actual_slope': slope,
        'prediction_actual_intercept': intercept,
        'noisy_mae': noisy_mae,
    }

    return {
        'model': model,
        'feature_cols': feature_cols,
        'exogenous_cols': exogenous_cols,
        'train_encoded': train_enc,
        'test_encoded': test_enc,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'results': results,
        'importance_df': importance_df,
        'worst_cases': worst_cases,
        'metrics': metrics,
    }


# ============================================================
# 6. DASHBOARD
# ============================================================
def create_dashboard(full_df: pd.DataFrame, artifacts: dict, output_dir='figures'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = artifacts['results']
    importance_df = artifacts['importance_df']
    worst_cases = artifacts['worst_cases']
    slope = artifacts['metrics']['prediction_actual_slope']

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # 1) Actual vs Predicted
    ax = axes[0, 0]
    ax.scatter(results['predicted'], results[TARGET], alpha=0.7)
    lo = min(results['predicted'].min(), results[TARGET].min())
    hi = max(results['predicted'].max(), results[TARGET].max())
    ax.plot([lo, hi], [lo, hi], '--', linewidth=1.5)
    slope_label = f'{slope:.2f}' if pd.notna(slope) else 'NA'
    ax.set_title(f'Actual vs Predicted (Slope={slope_label})')
    ax.set_xlabel('Predicted Consumption')
    ax.set_ylabel('Actual Consumption')

    # 2) Global IQR outliers
    ax = axes[0, 1]
    normal = full_df[full_df['is_iqr_outlier'] == 0]
    out = full_df[full_df['is_iqr_outlier'] == 1]
    ax.scatter(normal.index, normal[TARGET], alpha=0.6, label='Normal')
    ax.scatter(out.index, out[TARGET], alpha=0.9, marker='x', s=55, label='IQR outlier')
    ax.set_title('Global Consumption Outliers: IQR')
    ax.set_xlabel('Record Index')
    ax.set_ylabel('Total Consumption')
    ax.legend()

    # 3) Anomaly method agreement summary
    ax = axes[0, 2]
    agreement = pd.Series({
        'IQR': int(full_df['is_iqr_outlier'].sum()),
        'LOF': int(full_df['is_lof_outlier'].sum()),
        'Isolation\nForest': int(full_df['is_iforest_outlier'].sum()),
        'Consensus\n>=2': int(full_df['is_consensus_anomaly'].sum()),
    })
    agreement.plot(kind='bar', ax=ax)
    ax.set_title('Anomaly Detection Summary')
    ax.set_ylabel('Flagged Records')
    ax.tick_params(axis='x', rotation=0)

    # 4) Residual diagnostics
    ax = axes[1, 0]
    ax.scatter(results['predicted'], results['residual'], alpha=0.7)
    ax.axhline(0, linestyle='--', linewidth=1)
    ax.set_title('Residual Diagnostics')
    ax.set_xlabel('Predicted Consumption')
    ax.set_ylabel('Residual (Actual - Predicted)')

    # 5) Failure cases
    ax = axes[1, 1]
    ax.scatter(results[TARGET], results['predicted'], alpha=0.45, label='Test cases')
    ax.scatter(
        worst_cases[TARGET],
        worst_cases['predicted'],
        s=100,
        marker='X',
        label='Top 5 failure cases',
    )
    lo = min(results[TARGET].min(), results['predicted'].min())
    hi = max(results[TARGET].max(), results['predicted'].max())
    ax.plot([lo, hi], [lo, hi], '--', linewidth=1)
    ax.set_title('Failure Case Analysis')
    ax.set_xlabel('Actual Consumption')
    ax.set_ylabel('Predicted Consumption')
    ax.legend()

    # 6) Permutation importance
    ax = axes[1, 2]
    imp = importance_df.sort_values('importance_mean').tail(10)
    ax.barh(imp['feature'], imp['importance_mean'], xerr=imp['importance_std'])
    ax.axvline(0, linewidth=1)
    ax.set_title('Held-out Permutation Importance')
    ax.set_xlabel('Increase in MAE after Permutation')

    plt.tight_layout()
    path = output_dir / 'portsmouth_water_consumption_dashboard.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nDashboard saved to: {path.resolve()}')


# ============================================================
# 7. EXPORT TABLES
# ============================================================
def export_outputs(full_df: pd.DataFrame, artifacts: dict, output_dir='outputs'):
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    full_df.to_csv(outdir / 'water_with_anomaly_flags.csv', index=False)
    artifacts['results'].to_csv(outdir / 'test_predictions.csv', index=False)
    artifacts['importance_df'].to_csv(outdir / 'permutation_importance.csv', index=False)
    artifacts['worst_cases'].to_csv(outdir / 'failure_cases_top5.csv', index=False)
    pd.DataFrame([artifacts['metrics']]).to_csv(outdir / 'model_metrics.csv', index=False)

    # Export consensus anomalies separately for rapid domain review.
    consensus = full_df[full_df['is_consensus_anomaly'] == 1].copy()
    consensus.to_csv(outdir / 'consensus_anomalies.csv', index=False)

    print(f'Outputs exported to: {outdir.resolve()}')


# ============================================================
# MAIN
# ============================================================
def main():
    candidate_paths = [
        'data/Portsmouth_Water_Domestic_Consumption_25-26.csv',
        'Portsmouth_Water_Domestic_Consumption_25-26.csv',
    ]
    file_path = next((p for p in candidate_paths if os.path.exists(p)), candidate_paths[0])

    raw = load_and_audit(file_path)
    base = prepare_base_features(raw)

    context_cols = list(base.attrs.get('exogenous_numeric', []))
    if 'district_record_count' not in context_cols and base['district_record_count'].nunique() > 1:
        context_cols.append('district_record_count')

    anomaly_df = detect_anomalies(base, context_cols=context_cols)
    artifacts = train_and_evaluate(anomaly_df)
    create_dashboard(anomaly_df, artifacts)
    export_outputs(anomaly_df, artifacts)

    print('\n' + '=' * 72)
    print('ANALYSIS COMPLETE')
    print('=' * 72)
    print('This analysis is spatial/cross-sectional consumption modelling plus anomaly detection.')
    print('It is deliberately NOT described as temporal forecasting because the supplied source')
    print('does not contain usable repeated timestamps. Genuine forecasting requires timestamped')
    print('consumption histories and chronological/backtesting validation.')


if __name__ == '__main__':
    main()
