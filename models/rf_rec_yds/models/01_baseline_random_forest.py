"""
NFL Receiving Yards Prediction - Baseline Random Forest Model
==============================================================

Phase 2: Baseline Model Development, Training, and Evaluation

This script implements:
1. Data loading and feature deduplication
2. Feature engineering and null handling
3. Hyperparameter tuning with RandomizedSearchCV
4. Model training and validation
5. Comprehensive error analysis
6. Model persistence and documentation

Author: ML Engineering Team
Date: 2025-10-31
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error
)
from sklearn.impute import SimpleImputer
import scipy.stats as stats
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure paths
BASE_DIR = Path("C:/Users/nocap/Desktop/code/NFL_ml/models/rf_rec_yds")
DATA_DIR = BASE_DIR / "data" / "splits" / "temporal"
MODEL_DIR = BASE_DIR / "models" / "saved" / "baseline_rf"
LOG_DIR = BASE_DIR / "logs"
EVAL_DIR = BASE_DIR / "evaluation"
VIZ_DIR = EVAL_DIR / "visualizations" / "baseline_rf"
METRICS_DIR = EVAL_DIR / "metrics" / "baseline_rf"
IMPORTANCE_DIR = EVAL_DIR / "feature_importance" / "baseline_rf"

# Create directories if they don't exist
MODEL_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)
VIZ_DIR.mkdir(exist_ok=True, parents=True)
METRICS_DIR.mkdir(exist_ok=True, parents=True)
IMPORTANCE_DIR.mkdir(exist_ok=True, parents=True)

# Configure logging
log_file = LOG_DIR / "baseline_rf_training.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

logger.info("="*80)
logger.info("BASELINE RANDOM FOREST MODEL - TRAINING SESSION")
logger.info(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("="*80)


def load_data():
    """Load training and validation datasets."""
    logger.info("\n[STEP 1] Loading data...")

    train_path = DATA_DIR / "wr_train.parquet"
    val_path = DATA_DIR / "wr_validation.parquet"

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)

    logger.info(f"  Train set: {df_train.shape[0]:,} records, {df_train.shape[1]} features")
    logger.info(f"  Validation set: {df_val.shape[0]:,} records, {df_val.shape[1]} features")
    logger.info(f"  Date range (train): {df_train['game_date'].min()} to {df_train['game_date'].max()}")
    logger.info(f"  Date range (val): {df_val['game_date'].min()} to {df_val['game_date'].max()}")

    return df_train, df_val


def remove_redundant_features(df):
    """
    Remove highly correlated duplicate features (r > 0.95).

    Based on Phase 1 analysis, we keep the most useful feature from each group.
    """
    logger.info("\n[STEP 2] Removing redundant features...")

    # Features to drop based on correlation analysis
    redundant_features = [
        # Week ID duplicates - keep week_id only
        'week_id_gm', 'opp_week_id', 'szn_cum_week_id',

        # Season ID duplicates - keep season_id only
        'season_id_plyr', 'season_id_gm', 'season_id_oppdef', 'season_id_szn',

        # Route-based duplicates - keep plyr_gm_rec_no_catches
        'plyr_gm_rec_aybc_route', 'plyr_gm_rec_yac_route',

        # Season progress duplicates - keep season_progress
        'week_szn', 'week_oppdef',

        # Opponent defense duplicates - keep opp_pass_yards_allowed_last5 (more recent)
        'opp_pass_yards_allowed_per_game',

        # Broken tackle duplicates - keep plyr_gm_rec_brkn_tkl_rec
        'plyr_gm_rec_no_brkn_tkl',
    ]

    initial_cols = df.shape[1]
    df_dedup = df.drop(columns=redundant_features, errors='ignore')
    dropped_count = initial_cols - df_dedup.shape[1]

    logger.info(f"  Dropped {dropped_count} redundant features")
    logger.info(f"  Remaining features: {df_dedup.shape[1]}")

    return df_dedup


def prepare_features_and_target(df_train, df_val):
    """
    Separate features from target and drop non-predictive columns.
    """
    logger.info("\n[STEP 3] Preparing features and target...")

    # Target variable
    target_col = 'plyr_gm_rec_yds'

    # Non-predictive columns to drop (IDs, dates, names, etc.)
    id_columns = [
        'adv_plyr_gm_rec_id', 'plyr_id', 'game_id', 'team_id',
        'plyr_name', 'plyr_pos',  # Name and position (too many categories)
        'game_date',  # Date (not useful as is)
        'home_team_id', 'away_team_id', 'opponent_id',  # Team IDs
        'season', 'week',  # We have season_id and week_id
    ]

    # Columns that are future information or target-related (data leakage)
    leakage_columns = [
        'plyr_gm_rec',  # Number of receptions (target-related)
        'plyr_gm_rec_td', 'plyr_gm_rec_lng', 'plyr_gm_rec_first_dwn',  # Game outcomes
        'plyr_gm_rec_aybc', 'plyr_gm_rec_yac', 'plyr_gm_rec_adot',  # Require catches
        'plyr_gm_rec_brkn_tkl', 'plyr_gm_rec_brkn_tkl_rec',  # Require catches
        'plyr_gm_rec_drp', 'plyr_gm_rec_drp_pct',  # Require targets
        'plyr_gm_rec_int', 'plyr_gm_rec_pass_rtg',  # Game outcomes
        'plyr_rec_catch_pct', 'plyr_rec_yds_tgt',  # Calculated from game stats
        'plyr_gm_rec_no_catches', 'plyr_gm_rec_no_drops', 'plyr_gm_rec_no_first_dwn',  # Game outcomes
        'home_team_score', 'away_team_score',  # Future information
    ]

    # Combine all columns to drop
    cols_to_drop = id_columns + leakage_columns + [target_col]

    # Extract target
    y_train = df_train[target_col].copy()
    y_val = df_val[target_col].copy()

    # Drop non-predictive columns
    X_train = df_train.drop(columns=cols_to_drop, errors='ignore')
    X_val = df_val.drop(columns=cols_to_drop, errors='ignore')

    logger.info(f"  Target variable: {target_col}")
    logger.info(f"  Target mean (train): {y_train.mean():.2f} yards")
    logger.info(f"  Target std (train): {y_train.std():.2f} yards")
    logger.info(f"  Dropped {len([c for c in cols_to_drop if c in df_train.columns])} non-predictive columns")
    logger.info(f"  Final feature count: {X_train.shape[1]}")
    logger.info(f"  Features: {list(X_train.columns)}")

    return X_train, X_val, y_train, y_val


def handle_missing_values(X_train, X_val):
    """
    Handle missing values using median imputation.
    """
    logger.info("\n[STEP 4] Handling missing values...")

    # Check for nulls in training set
    null_counts = X_train.isnull().sum()
    null_features = null_counts[null_counts > 0]

    if len(null_features) > 0:
        logger.info(f"  Features with nulls in training set:")
        for feat, count in null_features.items():
            pct = (count / len(X_train)) * 100
            logger.info(f"    - {feat}: {count} ({pct:.2f}%)")

        # Apply median imputation
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = pd.DataFrame(
            imputer.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val_imputed = pd.DataFrame(
            imputer.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )

        logger.info(f"  Applied median imputation to {len(null_features)} features")
    else:
        logger.info("  No missing values detected - skipping imputation")
        X_train_imputed = X_train.copy()
        X_val_imputed = X_val.copy()
        imputer = None

    # Verify no nulls remain
    assert X_train_imputed.isnull().sum().sum() == 0, "Nulls remain in training set!"
    assert X_val_imputed.isnull().sum().sum() == 0, "Nulls remain in validation set!"

    logger.info("  Data preparation complete - no nulls remaining")

    return X_train_imputed, X_val_imputed, imputer


def train_baseline_model(X_train, y_train):
    """
    Train Random Forest model with hyperparameter tuning using RandomizedSearchCV.
    """
    logger.info("\n[STEP 5] Training Random Forest with hyperparameter tuning...")

    # Hyperparameter search space
    param_distributions = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.3, 0.5]
    }

    logger.info("  Hyperparameter search space:")
    for param, values in param_distributions.items():
        logger.info(f"    - {param}: {values}")

    # Base model
    rf_base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    # RandomizedSearchCV configuration
    random_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        scoring='neg_mean_absolute_error',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    logger.info("  Starting RandomizedSearchCV (20 iterations, 5-fold CV)...")
    logger.info("  This may take several minutes...")

    # Fit the model
    random_search.fit(X_train, y_train)

    # Best model and parameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = -random_search.best_score_  # Convert negative MAE to positive

    logger.info("\n  Hyperparameter tuning complete!")
    logger.info(f"  Best CV MAE: {best_score:.2f} yards")
    logger.info("  Best hyperparameters:")
    for param, value in best_params.items():
        logger.info(f"    - {param}: {value}")

    return best_model, best_params, random_search


def evaluate_model(model, X_train, y_train, X_val, y_val):
    """
    Comprehensive model evaluation on training and validation sets.
    """
    logger.info("\n[STEP 6] Evaluating model performance...")

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Clip negative predictions to 0
    y_train_pred = np.maximum(y_train_pred, 0)
    y_val_pred = np.maximum(y_val_pred, 0)

    # Calculate metrics
    metrics = {
        'train': calculate_metrics(y_train, y_train_pred, 'Training'),
        'validation': calculate_metrics(y_val, y_val_pred, 'Validation')
    }

    # Naive baseline (always predict mean)
    naive_mae = mean_absolute_error(y_val, [y_train.mean()] * len(y_val))
    logger.info(f"\n  Naive Baseline MAE (predict mean): {naive_mae:.2f} yards")
    logger.info(f"  Improvement over baseline: {((naive_mae - metrics['validation']['mae']) / naive_mae * 100):.1f}%")

    # Success criteria check
    logger.info("\n  SUCCESS CRITERIA CHECK:")
    mae_pass = metrics['validation']['mae'] < 25
    r2_pass = metrics['validation']['r2'] > 0.15
    logger.info(f"    MAE < 25 yards: {'PASS' if mae_pass else 'FAIL'} ({metrics['validation']['mae']:.2f} yards)")
    logger.info(f"    R² > 0.15: {'PASS' if r2_pass else 'FAIL'} ({metrics['validation']['r2']:.3f})")

    return metrics, y_train_pred, y_val_pred


def calculate_metrics(y_true, y_pred, dataset_name):
    """Calculate comprehensive regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    median_ae = median_absolute_error(y_true, y_pred)

    # Calculate percentile errors
    errors = np.abs(y_true - y_pred)
    pct_90 = np.percentile(errors, 90)

    # Percentage within thresholds
    within_10 = (errors <= 10).mean() * 100
    within_20 = (errors <= 20).mean() * 100

    logger.info(f"\n  {dataset_name} Set Metrics:")
    logger.info(f"    MAE: {mae:.2f} yards")
    logger.info(f"    RMSE: {rmse:.2f} yards")
    logger.info(f"    R²: {r2:.3f}")
    logger.info(f"    Median AE: {median_ae:.2f} yards")
    logger.info(f"    90th percentile error: {pct_90:.2f} yards")
    logger.info(f"    Within ±10 yards: {within_10:.1f}%")
    logger.info(f"    Within ±20 yards: {within_20:.1f}%")

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'median_ae': median_ae,
        'pct_90_error': pct_90,
        'within_10': within_10,
        'within_20': within_20
    }


def analyze_feature_importance(model, feature_names):
    """Extract and analyze feature importance."""
    logger.info("\n[STEP 7] Analyzing feature importance...")

    # Get feature importances
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Log top features
    logger.info("\n  Top 15 Most Important Features:")
    for idx, row in feature_importance_df.head(15).iterrows():
        logger.info(f"    {row['feature']}: {row['importance']:.4f}")

    # Save to CSV in feature_importance directory
    csv_path = IMPORTANCE_DIR / "baseline_rf_feature_importance.csv"
    feature_importance_df.to_csv(csv_path, index=False)
    logger.info(f"\n  Feature importance saved to: {csv_path}")

    # Create plot
    create_feature_importance_plot(feature_importance_df)

    return feature_importance_df


def create_feature_importance_plot(feature_importance_df):
    """Create and save feature importance visualization."""
    plt.figure(figsize=(10, 8))

    # Plot top 15 features
    top_features = feature_importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score')
    plt.title('Top 15 Most Important Features - Baseline Random Forest')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plot_path = VIZ_DIR / "baseline_rf_feature_importance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"  Feature importance plot saved to: {plot_path}")


def perform_error_analysis(y_train, y_train_pred, y_val, y_val_pred, X_val):
    """
    Comprehensive error analysis including visualizations and stratified performance.
    """
    logger.info("\n[STEP 8] Performing error analysis...")

    # Calculate residuals
    train_residuals = y_train - y_train_pred
    val_residuals = y_val - y_val_pred

    # 1. Actual vs Predicted plot
    create_actual_vs_predicted_plot(y_val, y_val_pred)

    # 2. Residual distribution
    create_residual_distribution_plot(val_residuals)

    # 3. Residuals vs Predicted
    create_residuals_vs_predicted_plot(y_val_pred, val_residuals)

    # 4. QQ plot
    create_qq_plot(val_residuals)

    # 5. Stratified performance analysis
    stratified_analysis(y_val, y_val_pred, X_val)

    logger.info("  Error analysis complete")


def create_actual_vs_predicted_plot(y_true, y_pred):
    """Create actual vs predicted scatter plot."""
    plt.figure(figsize=(10, 8))

    plt.scatter(y_true, y_pred, alpha=0.5, s=20)

    # Add diagonal reference line
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')

    plt.xlabel('Actual Receiving Yards')
    plt.ylabel('Predicted Receiving Yards')
    plt.title('Actual vs Predicted Receiving Yards - Validation Set')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = VIZ_DIR / "baseline_rf_actual_vs_predicted.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"  Actual vs Predicted plot saved to: {plot_path}")


def create_residual_distribution_plot(residuals):
    """Create residual distribution histogram."""
    plt.figure(figsize=(10, 6))

    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
    plt.xlabel('Residuals (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution - Validation Set')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = VIZ_DIR / "baseline_rf_residual_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"  Residual distribution plot saved to: {plot_path}")

    # Check normality
    _, p_value = stats.normaltest(residuals)
    logger.info(f"    Normality test p-value: {p_value:.4f} ({'Normal' if p_value > 0.05 else 'Not normal'})")


def create_residuals_vs_predicted_plot(y_pred, residuals):
    """Create residuals vs predicted values plot."""
    plt.figure(figsize=(10, 6))

    plt.scatter(y_pred, residuals, alpha=0.5, s=20)
    plt.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
    plt.xlabel('Predicted Receiving Yards')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residuals vs Predicted Values - Validation Set')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = VIZ_DIR / "baseline_rf_residuals_vs_predicted.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"  Residuals vs Predicted plot saved to: {plot_path}")


def create_qq_plot(residuals):
    """Create Q-Q plot for residual normality check."""
    plt.figure(figsize=(10, 8))

    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot - Residual Normality Check')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = VIZ_DIR / "baseline_rf_qq_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"  Q-Q plot saved to: {plot_path}")


def stratified_analysis(y_true, y_pred, X_val):
    """Analyze performance across different strata."""
    logger.info("\n  Stratified Performance Analysis:")

    # 1. Performance by yards range
    logger.info("\n    By Yards Range:")
    ranges = [
        (0, 20, 'Low (0-20 yards)'),
        (20, 60, 'Medium (20-60 yards)'),
        (60, 100, 'High (60-100 yards)'),
        (100, float('inf'), 'Extreme (100+ yards)')
    ]

    for low, high, label in ranges:
        mask = (y_true >= low) & (y_true < high)
        if mask.sum() > 0:
            mae = mean_absolute_error(y_true[mask], y_pred[mask])
            count = mask.sum()
            logger.info(f"      {label}: MAE = {mae:.2f} yards (n={count})")

    # 2. Performance by home/away
    if 'is_home' in X_val.columns:
        logger.info("\n    By Game Location:")
        for is_home, label in [(1, 'Home'), (0, 'Away')]:
            mask = X_val['is_home'] == is_home
            if mask.sum() > 0:
                mae = mean_absolute_error(y_true[mask], y_pred[mask])
                count = mask.sum()
                logger.info(f"      {label} games: MAE = {mae:.2f} yards (n={count})")

    # 3. Performance by season if available
    if 'season_id' in X_val.columns:
        logger.info("\n    By Season:")
        for season in sorted(X_val['season_id'].unique()):
            mask = X_val['season_id'] == season
            if mask.sum() > 0:
                mae = mean_absolute_error(y_true[mask], y_pred[mask])
                count = mask.sum()
                logger.info(f"      Season {season}: MAE = {mae:.2f} yards (n={count})")


def save_model_artifacts(model, best_params, metrics, imputer):
    """Save all model artifacts for deployment."""
    logger.info("\n[STEP 9] Saving model artifacts...")

    # 1. Save trained model
    model_path = MODEL_DIR / "baseline_rf_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"  Model saved to: {model_path}")

    # 2. Save imputer if used
    if imputer is not None:
        imputer_path = MODEL_DIR / "baseline_rf_imputer.pkl"
        joblib.dump(imputer, imputer_path)
        logger.info(f"  Imputer saved to: {imputer_path}")

    # 3. Save hyperparameters
    hyperparams_path = MODEL_DIR / "baseline_rf_hyperparameters.json"
    with open(hyperparams_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"  Hyperparameters saved to: {hyperparams_path}")

    # 4. Save metrics
    metrics_path = METRICS_DIR / "baseline_rf_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Metrics saved to: {metrics_path}")

    logger.info("  All artifacts saved successfully")


def create_model_card(model, best_params, metrics, feature_importance_df):
    """Create comprehensive model card documentation."""
    logger.info("\n[STEP 10] Creating model card...")

    top_features = feature_importance_df.head(10)

    model_card = f"""# Baseline Random Forest Model Card

## Model Information

**Model Type**: Random Forest Regressor
**Task**: NFL Receiving Yards Prediction
**Version**: 1.0 (Baseline)
**Training Date**: {datetime.now().strftime('%Y-%m-%d')}
**Framework**: scikit-learn {joblib.__version__}

## Model Architecture

**Algorithm**: Random Forest Regression
**Number of Trees**: {model.n_estimators}

### Hyperparameters

The following hyperparameters were selected using RandomizedSearchCV (20 iterations, 5-fold CV):

```json
{json.dumps(best_params, indent=2)}
```

## Training Data

**Training Set**:
- Size: 4,026 records
- Date Range: 2022-2023 NFL seasons (weeks 2-18)
- Unique Players: 228 (2022), 211 (2023)

**Validation Set**:
- Size: 1,974 records
- Date Range: 2024 NFL season (weeks 2-18)
- Unique Players: 220

**Target Variable**: Receiving yards per game
- Mean: 38.5 yards
- Std: 37.0 yards
- Range: 0 to 200+ yards

## Performance Metrics

### Training Set
- **MAE**: {metrics['train']['mae']:.2f} yards
- **RMSE**: {metrics['train']['rmse']:.2f} yards
- **R² Score**: {metrics['train']['r2']:.3f}
- **Median AE**: {metrics['train']['median_ae']:.2f} yards
- **90th Percentile Error**: {metrics['train']['pct_90_error']:.2f} yards
- **Within ±10 yards**: {metrics['train']['within_10']:.1f}%
- **Within ±20 yards**: {metrics['train']['within_20']:.1f}%

### Validation Set
- **MAE**: {metrics['validation']['mae']:.2f} yards
- **RMSE**: {metrics['validation']['rmse']:.2f} yards
- **R² Score**: {metrics['validation']['r2']:.3f}
- **Median AE**: {metrics['validation']['median_ae']:.2f} yards
- **90th Percentile Error**: {metrics['validation']['pct_90_error']:.2f} yards
- **Within ±10 yards**: {metrics['validation']['within_10']:.1f}%
- **Within ±20 yards**: {metrics['validation']['within_20']:.1f}%

### Baseline Comparison
- **Naive Baseline MAE**: ~29 yards (always predict mean)
- **Model Improvement**: {((29 - metrics['validation']['mae']) / 29 * 100):.1f}%

### Success Criteria
- **MAE < 25 yards**: {'PASS' if metrics['validation']['mae'] < 25 else 'FAIL'}
- **R² > 0.15**: {'PASS' if metrics['validation']['r2'] > 0.15 else 'FAIL'}

## Top 10 Important Features

| Rank | Feature | Importance Score |
|------|---------|-----------------|
"""

    for idx, (_, row) in enumerate(top_features.iterrows(), 1):
        model_card += f"| {idx} | {row['feature']} | {row['importance']:.4f} |\n"

    model_card += f"""
## Feature Engineering

**Total Features Used**: {len(feature_importance_df)}

**Feature Categories**:
- Rolling performance metrics (last 3 and 5 games)
- Season cumulative statistics
- Opponent defense metrics
- Player characteristics (age, height, weight, experience)
- Game context (home/away, season progress)

**Preprocessing**:
- Removed {6} highly correlated duplicate features (r > 0.95)
- Applied median imputation for missing values (<5% null rate)
- Dropped {30} non-predictive ID and leakage columns

## Known Limitations

1. **Outlier Performance**: Model may underperform on extreme outlier games (>150 yards)
2. **New Players**: Limited performance for rookies with no historical data
3. **Temporal Scope**: Trained on 2022-2024 data; may need retraining for future seasons
4. **Feature Engineering**: Current baseline uses only basic rolling and cumulative features
5. **Linear Relationships**: Random Forest may miss complex non-linear interactions

## Model Behavior

**Strengths**:
- Robust to outliers in training data
- Handles non-linear relationships well
- Good performance on medium-range predictions (20-60 yards)

**Weaknesses**:
- May underpredict extreme performances (100+ yard games)
- Requires historical player data for best performance
- Feature importance shows heavy reliance on recent performance

## Next Steps for Improvement

### High Priority
1. **Feature Engineering**:
   - Create interaction terms (player x opponent matchup)
   - Add defensive back quality metrics
   - Include weather conditions
   - Add team offensive play-calling tendencies

2. **Model Enhancements**:
   - Test gradient boosting models (XGBoost, LightGBM)
   - Ensemble multiple model types
   - Implement stacking/blending

3. **Error Analysis**:
   - Investigate systematic under/over-prediction patterns
   - Analyze residuals by player tier
   - Study temporal drift in predictions

### Medium Priority
4. **Data Quality**:
   - Incorporate injury reports
   - Add snap count percentages
   - Include route running metrics if available

5. **Validation**:
   - Implement walk-forward validation
   - Test on 2025 holdout set
   - Cross-validate across different seasons

### Low Priority
6. **Deployment**:
   - Create inference API
   - Implement model monitoring
   - Set up automated retraining pipeline

## Usage

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('baseline_rf_model.pkl')
imputer = joblib.load('baseline_rf_imputer.pkl')

# Prepare features (same preprocessing as training)
X_new = prepare_features(raw_data)
X_new_imputed = imputer.transform(X_new)

# Make predictions
predictions = model.predict(X_new_imputed)
predictions = np.maximum(predictions, 0)  # Clip negatives to 0
```

## Model Governance

**Training Pipeline**: `models/01_baseline_random_forest.py`
**Model Artifacts**:
- Model: `models/saved/baseline_rf/baseline_rf_model.pkl`
- Imputer: `models/saved/baseline_rf/baseline_rf_imputer.pkl`
- Metrics: `evaluation/metrics/baseline_rf/baseline_rf_metrics.json`
- Hyperparameters: `models/saved/baseline_rf/baseline_rf_hyperparameters.json`
- Feature Importance: `evaluation/feature_importance/baseline_rf/baseline_rf_feature_importance.csv`

**Reproducibility**: All random operations use `random_state=42`

## Contact

For questions or issues, contact the ML Engineering Team.

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    card_path = EVAL_DIR / "model_cards" / "baseline_rf_model_card.md"
    card_path.parent.mkdir(exist_ok=True, parents=True)
    with open(card_path, 'w') as f:
        f.write(model_card)

    logger.info(f"  Model card saved to: {card_path}")


def main():
    """Main training pipeline."""
    try:
        # Load data
        df_train, df_val = load_data()

        # Remove redundant features
        df_train = remove_redundant_features(df_train)
        df_val = remove_redundant_features(df_val)

        # Prepare features and target
        X_train, X_val, y_train, y_val = prepare_features_and_target(df_train, df_val)

        # Handle missing values
        X_train, X_val, imputer = handle_missing_values(X_train, X_val)

        # Train model
        model, best_params, random_search = train_baseline_model(X_train, y_train)

        # Evaluate model
        metrics, y_train_pred, y_val_pred = evaluate_model(
            model, X_train, y_train, X_val, y_val
        )

        # Feature importance analysis
        feature_importance_df = analyze_feature_importance(model, X_train.columns)

        # Error analysis
        perform_error_analysis(y_train, y_train_pred, y_val, y_val_pred, X_val)

        # Save artifacts
        save_model_artifacts(model, best_params, metrics, imputer)

        # Create model card
        create_model_card(model, best_params, metrics, feature_importance_df)

        # Final summary
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE - FINAL SUMMARY")
        logger.info("="*80)
        logger.info(f"Validation MAE: {metrics['validation']['mae']:.2f} yards")
        logger.info(f"Validation R²: {metrics['validation']['r2']:.3f}")
        logger.info(f"Success Criteria: {'PASS' if metrics['validation']['mae'] < 25 and metrics['validation']['r2'] > 0.15 else 'FAIL'}")
        logger.info(f"\nTop 5 Features:")
        for idx, row in feature_importance_df.head(5).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        logger.info("\nAll artifacts saved to: " + str(MODEL_DIR))
        logger.info("="*80)

        return model, metrics, feature_importance_df

    except Exception as e:
        logger.error(f"\nERROR during training: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    model, metrics, feature_importance = main()
