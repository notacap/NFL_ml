"""
Baseline Random Forest model for NFL WR receiving yards prediction.

This module implements a Random Forest Regressor to predict next-week receiving yards
using time-based train/test splitting to prevent data leakage.

Data Leakage Prevention:
- Time-based splitting: Train on 2022-2023 seasons, test on 2024 season
- Features are pre-shifted rolling averages (safe to use)
- No current-week receiving stats used as features

Author: Generated for NFL_ml project
Date: 2024-11-25
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Feature definitions
ROLLING_FEATURES = [
    'roll_3g_yds', 'roll_3g_tgt', 'roll_3g_rec', 'roll_3g_yac',
    'roll_3g_first_dwn', 'roll_3g_aybc', 'roll_3g_td',
    'roll_5g_yds', 'roll_5g_tgt', 'roll_5g_rec', 'roll_5g_yac',
    'roll_5g_first_dwn', 'roll_5g_aybc', 'roll_5g_td'
]

EFFICIENCY_FEATURES = [
    'season_targets_per_game', 'season_yards_per_game',
    'yards_per_reception', 'yards_per_target'
]

TARGET_COLUMN = 'next_week_rec_yds'
FILTER_COLUMN = 'has_min_games_3g'

# Model hyperparameters
MODEL_CONFIG = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}


def get_feature_columns() -> List[str]:
    """
    Return the list of feature columns to use for training.

    Returns:
        List of 18 feature column names (14 rolling + 4 efficiency)
    """
    return ROLLING_FEATURES + EFFICIENCY_FEATURES


def load_and_filter_data(path: str) -> pd.DataFrame:
    """
    Load parquet data and apply required filters.

    Filters:
    - Only rows where has_min_games_3g == True (removes cold-start rows)
    - Removes rows with NaN in target or features

    Args:
        path: Path to the parquet file

    Returns:
        Filtered DataFrame ready for training
    """
    logger.info(f"Loading data from: {path}")
    df = pd.read_parquet(path)
    logger.info(f"Raw data shape: {df.shape}")

    # Apply filter for minimum games
    initial_count = len(df)
    df = df[df[FILTER_COLUMN] == True].copy()
    logger.info(f"After filtering {FILTER_COLUMN}==True: {len(df)} rows (removed {initial_count - len(df)})")

    # Get feature columns
    feature_cols = get_feature_columns()

    # Remove rows with NaN in features or target
    cols_to_check = feature_cols + [TARGET_COLUMN]
    initial_count = len(df)
    df = df.dropna(subset=cols_to_check)
    logger.info(f"After removing NaN rows: {len(df)} rows (removed {initial_count - len(df)})")

    return df


def create_time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create time-based train/test split to prevent data leakage.

    Split strategy:
    - Train: 2022-2023 seasons (year column values)
    - Test: 2024 season
    - Data is sorted chronologically before splitting

    Args:
        df: Filtered DataFrame

    Returns:
        Tuple of (train_df, test_df)
    """
    # Sort by season and week for proper time ordering
    df = df.sort_values(['year', 'week_id']).reset_index(drop=True)
    logger.info("Data sorted by year and week_id (chronological order)")

    # Time-based split using year column
    # Train: 2022, 2023 | Test: 2024
    train_years = [2022, 2023]
    test_years = [2024]

    train_df = df[df['year'].isin(train_years)].copy()
    test_df = df[df['year'].isin(test_years)].copy()

    logger.info(f"Train set: {len(train_df)} rows (years: {train_years})")
    logger.info(f"Test set: {len(test_df)} rows (years: {test_years})")

    # Verify no overlap
    train_max_year = train_df['year'].max()
    test_min_year = test_df['year'].min()
    assert train_max_year < test_min_year, "Data leakage detected: train/test overlap!"
    logger.info(f"Verified: No temporal overlap (train max year: {train_max_year}, test min year: {test_min_year})")

    return train_df, test_df


def prepare_features_target(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature matrix and target vector from DataFrame.

    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name

    Returns:
        Tuple of (X, y) as numpy arrays
    """
    X = df[feature_cols].values
    y = df[target_col].values
    return X, y


def train_model(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor model.

    Args:
        X_train: Training feature matrix
        y_train: Training target vector
        **kwargs: Optional hyperparameter overrides

    Returns:
        Trained RandomForestRegressor model
    """
    # Merge default config with any overrides
    config = {**MODEL_CONFIG, **kwargs}

    logger.info(f"Training Random Forest with config: {config}")
    model = RandomForestRegressor(**config)
    model.fit(X_train, y_train)
    logger.info("Model training complete")

    return model


def evaluate_model(
    model: RandomForestRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate model on test set and calculate metrics.

    Metrics:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - R2: R-squared (coefficient of determination)
    - MAPE: Mean Absolute Percentage Error (with floor of 1 to handle zeros)

    Args:
        model: Trained model
        X_test: Test feature matrix
        y_test: Test target vector

    Returns:
        Dictionary of metric names to values
    """
    logger.info("Generating predictions on test set...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # MAPE with floor of 1 to handle zero-yard games
    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'n_test_samples': len(y_test),
        'y_test_mean': float(np.mean(y_test)),
        'y_test_std': float(np.std(y_test)),
        'y_pred_mean': float(np.mean(y_pred)),
        'y_pred_std': float(np.std(y_pred))
    }

    logger.info(f"Test Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}, MAPE: {mape:.2f}%")

    return metrics, y_pred


def create_feature_importance_plot(
    model: RandomForestRegressor,
    feature_names: List[str],
    output_path: str,
    top_n: int = 15
) -> pd.DataFrame:
    """
    Create and save feature importance bar chart.

    Args:
        model: Trained model
        feature_names: List of feature names
        output_path: Path to save the plot
        top_n: Number of top features to show

    Returns:
        DataFrame with feature importances
    """
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Plot top N features
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(top_n)
    plt.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Feature Importance (Gini Impurity)')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importances - RF Baseline Model')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Feature importance plot saved to: {output_path}")
    return importance_df


def create_pred_vs_actual_plot(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    output_path: str
) -> None:
    """
    Create and save predicted vs actual scatter plot.

    Args:
        y_test: Actual target values
        y_pred: Predicted values
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, s=20, color='steelblue')

    # Add diagonal reference line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    plt.xlabel('Actual Receiving Yards')
    plt.ylabel('Predicted Receiving Yards')
    plt.title('Predicted vs Actual Receiving Yards - RF Baseline')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Pred vs Actual plot saved to: {output_path}")


def create_residuals_plot(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    output_path: str
) -> None:
    """
    Create and save residual distribution histogram.

    Args:
        y_test: Actual target values
        y_pred: Predicted values
        output_path: Path to save the plot
    """
    residuals = y_test - y_pred

    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    plt.axvline(x=np.mean(residuals), color='orange', linestyle='-', linewidth=2,
                label=f'Mean: {np.mean(residuals):.2f}')

    plt.xlabel('Residual (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution - RF Baseline')
    plt.legend()

    # Add stats text
    stats_text = f'Mean: {np.mean(residuals):.2f}\nStd: {np.std(residuals):.2f}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Residuals plot saved to: {output_path}")


def save_artifacts(
    model: RandomForestRegressor,
    metrics: Dict[str, float],
    feature_importance: pd.DataFrame,
    output_dir: str,
    model_dir: str
) -> Dict[str, str]:
    """
    Save model and all artifacts.

    Args:
        model: Trained model
        metrics: Dictionary of evaluation metrics
        feature_importance: DataFrame with feature importances
        output_dir: Directory for evaluation outputs
        model_dir: Directory for model artifacts

    Returns:
        Dictionary of artifact paths
    """
    paths = {}

    # Save model
    model_path = os.path.join(model_dir, 'rf_baseline_v1.joblib')
    joblib.dump(model, model_path)
    paths['model'] = model_path
    logger.info(f"Model saved to: {model_path}")

    # Save model config
    config_path = os.path.join(model_dir, 'model_config.yaml')
    config_data = {
        'model_type': 'RandomForestRegressor',
        'version': 'v1_baseline',
        'created_at': datetime.now().isoformat(),
        'hyperparameters': MODEL_CONFIG,
        'features': {
            'rolling_features': ROLLING_FEATURES,
            'efficiency_features': EFFICIENCY_FEATURES,
            'total_features': len(get_feature_columns())
        },
        'target': TARGET_COLUMN,
        'filter': FILTER_COLUMN,
        'train_years': [2022, 2023],
        'test_years': [2024]
    }
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)
    paths['config'] = config_path
    logger.info(f"Config saved to: {config_path}")

    # Save metrics report
    metrics_path = os.path.join(output_dir, 'metrics_report.txt')
    with open(metrics_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("RF BASELINE MODEL - EVALUATION METRICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        for key, value in MODEL_CONFIG.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        f.write("TEST SET METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  MAE:  {metrics['MAE']:.4f} yards\n")
        f.write(f"  RMSE: {metrics['RMSE']:.4f} yards\n")
        f.write(f"  R2:   {metrics['R2']:.4f}\n")
        f.write(f"  MAPE: {metrics['MAPE']:.2f}%\n\n")

        f.write("TEST SET STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  N samples:    {metrics['n_test_samples']}\n")
        f.write(f"  Actual mean:  {metrics['y_test_mean']:.2f} yards\n")
        f.write(f"  Actual std:   {metrics['y_test_std']:.2f} yards\n")
        f.write(f"  Pred mean:    {metrics['y_pred_mean']:.2f} yards\n")
        f.write(f"  Pred std:     {metrics['y_pred_std']:.2f} yards\n\n")

        f.write("FEATURE IMPORTANCE (Top 10)\n")
        f.write("-" * 40 + "\n")
        for idx, row in feature_importance.head(10).iterrows():
            f.write(f"  {row['feature']:30s}: {row['importance']:.4f}\n")
        f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("DATA LEAKAGE PREVENTION CHECKLIST\n")
        f.write("=" * 60 + "\n")
        f.write("[X] Time-based split (Train: 2022-2023, Test: 2024)\n")
        f.write("[X] Data sorted chronologically before split\n")
        f.write("[X] Features are pre-shifted rolling averages\n")
        f.write("[X] Filter applied: has_min_games_3g == True\n")
        f.write("[X] No current-week receiving stats in features\n")

    paths['metrics'] = metrics_path
    logger.info(f"Metrics report saved to: {metrics_path}")

    return paths


def main(data_path: str = None, base_dir: str = None) -> Dict[str, Any]:
    """
    Main training pipeline.

    Args:
        data_path: Path to input parquet file (optional, uses default if not provided)
        base_dir: Base directory for outputs (optional, uses default if not provided)

    Returns:
        Dictionary with model, metrics, and artifact paths
    """
    # Set default paths
    if base_dir is None:
        base_dir = r"C:\Users\nocap\Desktop\code\NFL_ml\models\rf_wr_yds"

    if data_path is None:
        data_path = os.path.join(base_dir, "data", "processed",
                                  "nfl_wr_features_v1_20251125_113246.parquet")

    output_dir = os.path.join(base_dir, "outputs", "model_evaluation")
    model_dir = os.path.join(base_dir, "models")

    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("RF BASELINE MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    # Step 1: Load and filter data
    df = load_and_filter_data(data_path)

    # Step 2: Create time-based split
    train_df, test_df = create_time_split(df)

    # Step 3: Prepare features and target
    feature_cols = get_feature_columns()
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

    X_train, y_train = prepare_features_target(train_df, feature_cols, TARGET_COLUMN)
    X_test, y_test = prepare_features_target(test_df, feature_cols, TARGET_COLUMN)

    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Verify no NaN in feature matrices
    assert not np.isnan(X_train).any(), "NaN detected in X_train!"
    assert not np.isnan(X_test).any(), "NaN detected in X_test!"
    logger.info("Verified: No NaN values in feature matrices")

    # Step 4: Train model
    model = train_model(X_train, y_train)

    # Step 5: Evaluate model
    metrics, y_pred = evaluate_model(model, X_test, y_test)

    # Step 6: Create visualizations
    importance_df = create_feature_importance_plot(
        model, feature_cols,
        os.path.join(output_dir, 'feature_importance.png')
    )

    create_pred_vs_actual_plot(
        y_test, y_pred,
        os.path.join(output_dir, 'pred_vs_actual.png')
    )

    create_residuals_plot(
        y_test, y_pred,
        os.path.join(output_dir, 'residuals.png')
    )

    # Step 7: Save artifacts
    artifact_paths = save_artifacts(model, metrics, importance_df, output_dir, model_dir)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final Test Metrics:")
    logger.info(f"  MAE:  {metrics['MAE']:.4f} yards")
    logger.info(f"  RMSE: {metrics['RMSE']:.4f} yards")
    logger.info(f"  R2:   {metrics['R2']:.4f}")
    logger.info(f"  MAPE: {metrics['MAPE']:.2f}%")

    return {
        'model': model,
        'metrics': metrics,
        'feature_importance': importance_df,
        'artifact_paths': artifact_paths,
        'predictions': y_pred,
        'y_test': y_test
    }


if __name__ == "__main__":
    results = main()
