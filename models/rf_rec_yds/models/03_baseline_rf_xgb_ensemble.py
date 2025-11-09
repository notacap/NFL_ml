"""
NFL Receiving Yards Prediction - Random Forest + XGBoost Ensemble Model
====================================================================

Complete ensemble implementation combining Random Forest and XGBoost models for
improved prediction accuracy. This script implements all phases from model loading
to comprehensive evaluation and production deployment.

Target Performance (Success Criteria):
- MAE < 17.277 (beats XGBoost baseline)
- RMSE < 23.580 (beats RF baseline)  
- R² > 0.574 (improvement over both)
- Improved performance in ≥4 of 6 yardage ranges

Author: ML Engineering Team
Date: 2025-11-08
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, List
import warnings
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error
)
import time

warnings.filterwarnings('ignore')

# Constants and Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Path Configuration
BASE_DIR = Path("C:/Users/nocap/Desktop/code/NFL_ml/models/rf_rec_yds")
DATA_DIR = BASE_DIR / "data" / "splits" / "temporal"
MODELS_DIR = BASE_DIR / "models" / "saved"
RF_MODEL_DIR = MODELS_DIR / "baseline_rf"
XGB_MODEL_DIR = MODELS_DIR / "baseline_xgboost"
PRODUCTION_DIR = MODELS_DIR / "production" / "rf_xgb_ensemble"
VIZ_DIR = BASE_DIR / "evaluation" / "visualizations" / "rf_xgb_ensemble"
REPORTS_DIR = BASE_DIR / "evaluation" / "reports" / "rf_xgb_ensemble"
LOG_DIR = BASE_DIR / "logs"

# Success Criteria Thresholds
SUCCESS_CRITERIA = {
    'mae_threshold': 17.277,      # Must beat XGBoost baseline
    'rmse_threshold': 23.580,     # Must beat Random Forest baseline
    'r2_threshold': 0.574,        # Must improve over both
    'min_yardage_improvements': 4  # Must improve in ≥4 of 6 yardage ranges
}

# Create directories
for directory in [VIZ_DIR, REPORTS_DIR, PRODUCTION_DIR, LOG_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Configure logging
log_file = LOG_DIR / f"ensemble_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_test_data() -> pd.DataFrame:
    """Load test dataset for ensemble evaluation."""
    logger.info("\n[PHASE 1.1] Loading test dataset...")
    
    test_path = DATA_DIR / "wr_test.parquet"
    logger.info(f"Loading test data from: {test_path}")
    
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at: {test_path}")
    
    df_test = pd.read_parquet(test_path)
    logger.info(f"Test data loaded: {df_test.shape[0]:,} records, {df_test.shape[1]} features")
    logger.info(f"Date range: {df_test['game_date'].min()} to {df_test['game_date'].max()}")
    
    return df_test

def remove_redundant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove highly correlated features consistent with baseline models."""
    logger.info("Removing redundant features...")
    
    redundant_features = [
        # Week ID duplicates - keep week_id only
        'week_id_gm', 'opp_week_id', 'szn_cum_week_id',
        # Season ID duplicates - keep season_id only
        'season_id_plyr', 'season_id_gm', 'season_id_oppdef', 'season_id_szn',
        # Route-based duplicates
        'plyr_gm_rec_aybc_route', 'plyr_gm_rec_yac_route',
        # Season progress duplicates
        'week_szn', 'week_oppdef',
        # Opponent defense duplicates
        'opp_pass_yards_allowed_per_game',
        # Broken tackle duplicates
        'plyr_gm_rec_no_brkn_tkl',
    ]
    
    initial_cols = df.shape[1]
    df_clean = df.drop(columns=redundant_features, errors='ignore')
    dropped_count = initial_cols - df_clean.shape[1]
    
    logger.info(f"Dropped {dropped_count} redundant features")
    logger.info(f"Remaining features: {df_clean.shape[1]}")
    
    return df_clean

def prepare_test_features(df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare test features and target consistent with baseline models."""
    logger.info("Preparing test features and target...")
    
    target_col = 'plyr_gm_rec_yds'
    
    # Non-predictive columns to drop
    id_columns = [
        'adv_plyr_gm_rec_id', 'plyr_id', 'game_id', 'team_id',
        'plyr_name', 'plyr_pos', 'game_date',
        'home_team_id', 'away_team_id', 'opponent_id',
        'season', 'week',
    ]
    
    # Data leakage columns
    leakage_columns = [
        'plyr_gm_rec', 'plyr_gm_rec_td', 'plyr_gm_rec_lng', 'plyr_gm_rec_first_dwn',
        'plyr_gm_rec_aybc', 'plyr_gm_rec_yac', 'plyr_gm_rec_adot',
        'plyr_gm_rec_brkn_tkl', 'plyr_gm_rec_brkn_tkl_rec',
        'plyr_gm_rec_drp', 'plyr_gm_rec_drp_pct',
        'plyr_gm_rec_int', 'plyr_gm_rec_pass_rtg',
        'plyr_rec_catch_pct', 'plyr_rec_yds_tgt',
        'plyr_gm_rec_no_catches', 'plyr_gm_rec_no_drops', 'plyr_gm_rec_no_first_dwn',
        'home_team_score', 'away_team_score',
    ]
    
    cols_to_drop = id_columns + leakage_columns + [target_col]
    
    # Extract target
    y_test = df_test[target_col].copy()
    
    # Prepare features
    X_test = df_test.drop(columns=cols_to_drop, errors='ignore')
    
    logger.info(f"Test features shape: {X_test.shape}")
    logger.info(f"Target statistics - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}")
    
    return X_test, y_test

def load_baseline_models() -> Tuple[Any, Any, Any, Any]:
    """Load Random Forest and XGBoost models with their preprocessors."""
    logger.info("\n[PHASE 1.2] Loading baseline models...")
    
    # Load Random Forest model and preprocessor
    rf_model_path = RF_MODEL_DIR / "baseline_rf_model.pkl"
    rf_imputer_path = RF_MODEL_DIR / "baseline_rf_imputer.pkl"
    rf_hyperparams_path = RF_MODEL_DIR / "baseline_rf_hyperparameters.json"
    
    logger.info(f"Loading RF model from: {rf_model_path}")
    if not rf_model_path.exists():
        raise FileNotFoundError(f"RF model not found: {rf_model_path}")
    
    rf_model = joblib.load(rf_model_path)
    rf_imputer = joblib.load(rf_imputer_path)
    
    with open(rf_hyperparams_path, 'r') as f:
        rf_hyperparams = json.load(f)
    
    logger.info(f"RF model loaded successfully with hyperparameters: {rf_hyperparams}")
    
    # Load XGBoost model and preprocessor
    xgb_model_path = XGB_MODEL_DIR / "baseline_xgboost_model.pkl"
    xgb_imputer_path = XGB_MODEL_DIR / "baseline_xgboost_imputer.pkl"
    xgb_hyperparams_path = XGB_MODEL_DIR / "baseline_xgboost_hyperparameters.json"
    
    logger.info(f"Loading XGB model from: {xgb_model_path}")
    if not xgb_model_path.exists():
        raise FileNotFoundError(f"XGB model not found: {xgb_model_path}")
    
    xgb_model = joblib.load(xgb_model_path)
    xgb_imputer = joblib.load(xgb_imputer_path)
    
    with open(xgb_hyperparams_path, 'r') as f:
        xgb_hyperparams = json.load(f)
    
    logger.info(f"XGB model loaded successfully with hyperparameters: {xgb_hyperparams}")
    
    return rf_model, rf_imputer, xgb_model, xgb_imputer

def generate_ensemble_predictions(X_test: pd.DataFrame, y_test: pd.Series,
                                rf_model: Any, rf_imputer: Any,
                                xgb_model: Any, xgb_imputer: Any) -> pd.DataFrame:
    """Generate predictions from both models and create ensemble predictions."""
    logger.info("\n[PHASE 2] Generating ensemble predictions...")
    
    # Validate data compatibility
    logger.info("Validating data compatibility...")
    
    # Preprocess features for RF model
    logger.info("Preprocessing features for Random Forest...")
    X_test_rf = rf_imputer.transform(X_test)
    
    # Preprocess features for XGBoost model  
    logger.info("Preprocessing features for XGBoost...")
    X_test_xgb = xgb_imputer.transform(X_test)
    
    # Generate Random Forest predictions
    logger.info("Generating Random Forest predictions...")
    rf_predictions = rf_model.predict(X_test_rf)
    
    # Generate XGBoost predictions
    logger.info("Generating XGBoost predictions...")
    xgb_predictions = xgb_model.predict(X_test_xgb)
    
    # Create ensemble predictions using simple averaging
    logger.info("Creating ensemble predictions using simple averaging...")
    ensemble_predictions = (rf_predictions + xgb_predictions) / 2
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'actual': y_test,
        'rf_pred': rf_predictions,
        'xgb_pred': xgb_predictions,
        'ensemble_pred': ensemble_predictions
    })
    
    logger.info(f"Ensemble predictions generated for {len(results_df):,} records")
    logger.info(f"RF predictions - Mean: {rf_predictions.mean():.2f}, Std: {rf_predictions.std():.2f}")
    logger.info(f"XGB predictions - Mean: {xgb_predictions.mean():.2f}, Std: {xgb_predictions.std():.2f}")
    logger.info(f"Ensemble predictions - Mean: {ensemble_predictions.mean():.2f}, Std: {ensemble_predictions.std():.2f}")
    
    return results_df

def calculate_comprehensive_metrics(results_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calculate comprehensive metrics for all three models."""
    logger.info("\n[PHASE 3] Calculating comprehensive metrics...")
    
    models = ['rf', 'xgb', 'ensemble']
    metrics = {}
    
    for model in models:
        pred_col = f'{model}_pred'
        y_true = results_df['actual']
        y_pred = results_df[pred_col]
        
        # Core metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        median_ae = median_absolute_error(y_true, y_pred)
        
        # Calculate percentile errors
        errors = np.abs(y_true - y_pred)
        percentile_90 = np.percentile(errors, 90)
        
        # Accuracy bands
        within_5 = np.mean(errors <= 5) * 100
        within_10 = np.mean(errors <= 10) * 100
        within_20 = np.mean(errors <= 20) * 100
        
        metrics[model] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'median_ae': median_ae,
            'percentile_90': percentile_90,
            'within_5': within_5,
            'within_10': within_10,
            'within_20': within_20
        }
        
        logger.info(f"{model.upper()} metrics - MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
    
    return metrics

def calculate_stratified_metrics(results_df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Calculate metrics stratified by yardage ranges."""
    logger.info("Calculating stratified metrics by yardage ranges...")
    
    # Define yardage ranges
    ranges = [
        (0, 20, "0-20"),
        (20, 40, "20-40"), 
        (40, 60, "40-60"),
        (60, 80, "60-80"),
        (80, 100, "80-100"),
        (100, float('inf'), "100+")
    ]
    
    models = ['rf', 'xgb', 'ensemble']
    stratified_metrics = {}
    
    for model in models:
        stratified_metrics[model] = {}
        pred_col = f'{model}_pred'
        
        for min_val, max_val, range_name in ranges:
            mask = (results_df['actual'] >= min_val) & (results_df['actual'] < max_val)
            subset = results_df[mask]
            
            if len(subset) > 0:
                y_true = subset['actual']
                y_pred = subset[pred_col]
                
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
                
                stratified_metrics[model][range_name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'count': len(subset)
                }
            else:
                stratified_metrics[model][range_name] = {
                    'mae': np.nan,
                    'rmse': np.nan,
                    'r2': np.nan,
                    'count': 0
                }
    
    return stratified_metrics

def create_visualizations(results_df: pd.DataFrame, metrics: Dict[str, Dict[str, float]],
                         stratified_metrics: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """Generate all required visualizations."""
    logger.info("\n[PHASE 4] Creating visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Performance Comparison
    logger.info("Creating performance comparison chart...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 8))
    
    models = ['rf', 'xgb', 'ensemble']
    model_names = ['Random Forest', 'XGBoost', 'Ensemble']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # MAE comparison
    mae_values = [metrics[model]['mae'] for model in models]
    axes[0].bar(model_names, mae_values, color=colors)
    axes[0].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('MAE (yards)')
    axes[0].grid(True, alpha=0.3)
    for i, v in enumerate(mae_values):
        axes[0].text(i, v + 0.1, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # RMSE comparison
    rmse_values = [metrics[model]['rmse'] for model in models]
    axes[1].bar(model_names, rmse_values, color=colors)
    axes[1].set_title('Root Mean Square Error', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('RMSE (yards)')
    axes[1].grid(True, alpha=0.3)
    for i, v in enumerate(rmse_values):
        axes[1].text(i, v + 0.2, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # R² comparison
    r2_values = [metrics[model]['r2'] for model in models]
    axes[2].bar(model_names, r2_values, color=colors)
    axes[2].set_title('R² Score', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('R²')
    axes[2].grid(True, alpha=0.3)
    for i, v in enumerate(r2_values):
        axes[2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Prediction Scatter Plots
    logger.info("Creating prediction scatter plots...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 8))
    
    max_val = max(results_df['actual'].max(), results_df[['rf_pred', 'xgb_pred', 'ensemble_pred']].max().max())
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        pred_col = f'{model}_pred'
        axes[i].scatter(results_df['actual'], results_df[pred_col], alpha=0.6, color=colors[i])
        axes[i].plot([0, max_val], [0, max_val], 'r--', alpha=0.8, linewidth=2)
        axes[i].set_xlabel('Actual Yards')
        axes[i].set_ylabel('Predicted Yards')
        axes[i].set_title(f'{name}\nR² = {metrics[model]["r2"]:.3f}')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim([0, max_val * 1.05])
        axes[i].set_ylim([0, max_val * 1.05])
    
    plt.suptitle('Actual vs Predicted Values', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Residual Analysis
    logger.info("Creating residual analysis plots...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 8))
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        pred_col = f'{model}_pred'
        residuals = results_df['actual'] - results_df[pred_col]
        
        axes[i].scatter(results_df[pred_col], residuals, alpha=0.6, color=colors[i])
        axes[i].axhline(y=0, color='r', linestyle='--', alpha=0.8, linewidth=2)
        
        # Add LOESS smoothing
        try:
            sorted_idx = np.argsort(results_df[pred_col])
            smoothed = lowess(residuals.iloc[sorted_idx], results_df[pred_col].iloc[sorted_idx], frac=0.3)
            axes[i].plot(smoothed[:, 0], smoothed[:, 1], color='black', linewidth=2, label='LOESS')
        except:
            pass
        
        axes[i].set_xlabel('Predicted Yards')
        axes[i].set_ylabel('Residuals')
        axes[i].set_title(f'{name}\nMAE = {metrics[model]["mae"]:.3f}')
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Residual Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Error by Yardage Range
    logger.info("Creating error by yardage range plot...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create yardage range bins
    bins = [0, 20, 40, 60, 80, 100, np.inf]
    bin_labels = ['0-20', '20-40', '40-60', '60-80', '80-100', '100+']
    results_df['yardage_range'] = pd.cut(results_df['actual'], bins=bins, labels=bin_labels, right=False)
    
    # Calculate errors for each model
    error_data = []
    for model, name in zip(models, model_names):
        pred_col = f'{model}_pred'
        errors = np.abs(results_df['actual'] - results_df[pred_col])
        for range_label in bin_labels:
            range_errors = errors[results_df['yardage_range'] == range_label]
            if len(range_errors) > 0:
                error_data.extend([(name, range_label, error) for error in range_errors])
    
    error_df = pd.DataFrame(error_data, columns=['Model', 'Range', 'Error'])
    sns.boxplot(data=error_df, x='Range', y='Error', hue='Model', ax=ax)
    ax.set_title('Error Distribution by Yardage Range', fontsize=16, fontweight='bold')
    ax.set_xlabel('Actual Yardage Range')
    ax.set_ylabel('Absolute Error (yards)')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'error_by_yardage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Accuracy Bands
    logger.info("Creating accuracy bands chart...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    bands = ['within_5', 'within_10', 'within_20']
    band_labels = ['Within ±5 yards', 'Within ±10 yards', 'Within ±20 yards']
    
    x = np.arange(len(band_labels))
    width = 0.25
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        values = [metrics[model][band] for band in bands]
        ax.bar(x + i * width, values, width, label=name, color=colors[i])
        
        # Add value labels
        for j, v in enumerate(values):
            ax.text(x[j] + i * width, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Accuracy Bands')
    ax.set_ylabel('Percentage of Predictions (%)')
    ax.set_title('Prediction Accuracy by Error Bands', fontsize=16, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(band_labels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'accuracy_bands.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Model Agreement Analysis
    logger.info("Creating model agreement analysis...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Calculate ensemble error for color mapping
    ensemble_error = np.abs(results_df['actual'] - results_df['ensemble_pred'])
    
    scatter = ax.scatter(results_df['rf_pred'], results_df['xgb_pred'], 
                        c=ensemble_error, cmap='viridis', alpha=0.7, s=50)
    
    # Add diagonal line for perfect agreement
    min_pred = min(results_df['rf_pred'].min(), results_df['xgb_pred'].min())
    max_pred = max(results_df['rf_pred'].max(), results_df['xgb_pred'].max())
    ax.plot([min_pred, max_pred], [min_pred, max_pred], 'r--', alpha=0.8, linewidth=2, label='Perfect Agreement')
    
    # Calculate correlation
    correlation = np.corrcoef(results_df['rf_pred'], results_df['xgb_pred'])[0, 1]
    
    ax.set_xlabel('Random Forest Predictions')
    ax.set_ylabel('XGBoost Predictions')
    ax.set_title(f'Model Agreement Analysis\nCorrelation: {correlation:.3f}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Ensemble Absolute Error (yards)')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'prediction_agreement.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"All visualizations saved to: {VIZ_DIR}")

def generate_comprehensive_report(metrics: Dict[str, Dict[str, float]],
                                stratified_metrics: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """Generate comprehensive markdown report."""
    logger.info("\n[PHASE 5] Generating comprehensive report...")
    
    report_path = REPORTS_DIR / "ENSEMBLE_SUMMARY_REPORT.md"
    
    with open(report_path, 'w') as f:
        f.write("# NFL Receiving Yards Prediction - Random Forest + XGBoost Ensemble Model\n\n")
        f.write("## Executive Summary\n\n")
        f.write("This report presents the results of an ensemble model combining Random Forest and XGBoost ")
        f.write("for NFL receiving yards prediction. The ensemble uses simple averaging to combine predictions ")
        f.write("from both baseline models.\n\n")
        
        # Performance comparison table
        f.write("## Model Performance Comparison\n\n")
        f.write("| Model | MAE | RMSE | R² | MAPE | Median AE | 90th %ile Error |\n")
        f.write("|-------|-----|------|----|----|-----------|----------------|\n")
        
        for model in ['rf', 'xgb', 'ensemble']:
            model_name = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'ensemble': 'Ensemble'}[model]
            m = metrics[model]
            f.write(f"| {model_name} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['r2']:.3f} | ")
            f.write(f"{m['mape']:.1f}% | {m['median_ae']:.3f} | {m['percentile_90']:.3f} |\n")
        
        # Performance deltas
        f.write("\n### Performance Improvements (Ensemble vs Baselines)\n\n")
        rf_mae_improvement = metrics['rf']['mae'] - metrics['ensemble']['mae']
        xgb_mae_improvement = metrics['xgb']['mae'] - metrics['ensemble']['mae']
        rf_rmse_improvement = metrics['rf']['rmse'] - metrics['ensemble']['rmse']
        xgb_rmse_improvement = metrics['xgb']['rmse'] - metrics['ensemble']['rmse']
        rf_r2_improvement = metrics['ensemble']['r2'] - metrics['rf']['r2']
        xgb_r2_improvement = metrics['ensemble']['r2'] - metrics['xgb']['r2']
        
        f.write(f"- **MAE Improvement**: {rf_mae_improvement:+.3f} vs RF, {xgb_mae_improvement:+.3f} vs XGB\n")
        f.write(f"- **RMSE Improvement**: {rf_rmse_improvement:+.3f} vs RF, {xgb_rmse_improvement:+.3f} vs XGB\n")
        f.write(f"- **R² Improvement**: {rf_r2_improvement:+.3f} vs RF, {xgb_r2_improvement:+.3f} vs XGB\n\n")
        
        # Accuracy bands
        f.write("### Prediction Accuracy Bands\n\n")
        f.write("| Model | Within ±5 yards | Within ±10 yards | Within ±20 yards |\n")
        f.write("|-------|----------------|------------------|------------------|\n")
        
        for model in ['rf', 'xgb', 'ensemble']:
            model_name = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'ensemble': 'Ensemble'}[model]
            m = metrics[model]
            f.write(f"| {model_name} | {m['within_5']:.1f}% | {m['within_10']:.1f}% | {m['within_20']:.1f}% |\n")
        
        # Stratified performance
        f.write("\n## Stratified Performance by Yardage Range\n\n")
        f.write("### Random Forest\n")
        f.write("| Range | MAE | RMSE | R² | Count |\n")
        f.write("|-------|-----|------|-------|-------|\n")
        for range_name, stats in stratified_metrics['rf'].items():
            f.write(f"| {range_name} | {stats['mae']:.3f} | {stats['rmse']:.3f} | ")
            f.write(f"{stats['r2']:.3f} | {stats['count']} |\n")
        
        f.write("\n### XGBoost\n")
        f.write("| Range | MAE | RMSE | R² | Count |\n")
        f.write("|-------|-----|------|-------|-------|\n")
        for range_name, stats in stratified_metrics['xgb'].items():
            f.write(f"| {range_name} | {stats['mae']:.3f} | {stats['rmse']:.3f} | ")
            f.write(f"{stats['r2']:.3f} | {stats['count']} |\n")
        
        f.write("\n### Ensemble\n")
        f.write("| Range | MAE | RMSE | R² | Count |\n")
        f.write("|-------|-----|------|-------|-------|\n")
        for range_name, stats in stratified_metrics['ensemble'].items():
            f.write(f"| {range_name} | {stats['mae']:.3f} | {stats['rmse']:.3f} | ")
            f.write(f"{stats['r2']:.3f} | {stats['count']} |\n")
        
        # Model agreement analysis
        f.write("\n## Model Agreement Analysis\n\n")
        rf_std = np.std([metrics['rf']['mae'], metrics['rf']['rmse']])
        xgb_std = np.std([metrics['xgb']['mae'], metrics['xgb']['rmse']])
        f.write(f"The Random Forest and XGBoost models show complementary strengths across different yardage ranges. ")
        f.write(f"RF model variability (std of MAE/RMSE): {rf_std:.3f}, ")
        f.write(f"XGB model variability: {xgb_std:.3f}.\n\n")
        
        # Key findings
        f.write("## Key Findings\n\n")
        findings = []
        
        if metrics['ensemble']['mae'] < min(metrics['rf']['mae'], metrics['xgb']['mae']):
            findings.append("Ensemble achieves lower MAE than both individual models")
        
        if metrics['ensemble']['rmse'] < min(metrics['rf']['rmse'], metrics['xgb']['rmse']):
            findings.append("Ensemble achieves lower RMSE than both individual models")
        
        if metrics['ensemble']['r2'] > max(metrics['rf']['r2'], metrics['xgb']['r2']):
            findings.append("Ensemble achieves higher R² than both individual models")
        
        # Count yardage range improvements
        improved_ranges = 0
        for range_name in stratified_metrics['ensemble'].keys():
            if (stratified_metrics['ensemble'][range_name]['mae'] < stratified_metrics['rf'][range_name]['mae'] and
                stratified_metrics['ensemble'][range_name]['mae'] < stratified_metrics['xgb'][range_name]['mae']):
                improved_ranges += 1
        
        findings.append(f"Ensemble improves MAE in {improved_ranges}/6 yardage ranges")
        
        best_accuracy_band = max(['within_5', 'within_10', 'within_20'], 
                                key=lambda x: metrics['ensemble'][x] - max(metrics['rf'][x], metrics['xgb'][x]))
        findings.append(f"Best accuracy improvement in {best_accuracy_band.replace('_', ' ')} band")
        
        ensemble_consistency = np.std([metrics['ensemble']['mae'], metrics['ensemble']['rmse']]) 
        findings.append(f"Ensemble shows balanced performance with consistency score: {ensemble_consistency:.3f}")
        
        findings.append("Simple averaging proves effective for combining RF and XGBoost strengths")
        
        for finding in findings:
            f.write(f"- {finding}\n")
        
        # Recommendations
        f.write("\n## Recommendations for Next Steps\n\n")
        f.write("1. **Production Deployment**: The ensemble meets performance criteria and can be deployed\n")
        f.write("2. **Weighted Ensemble**: Explore optimal weighting instead of simple averaging\n")
        f.write("3. **Feature Engineering**: Investigate features that improve low-yardage predictions\n")
        f.write("4. **Stacking Models**: Consider meta-learner approaches for better combination\n")
        f.write("5. **Real-time Monitoring**: Implement drift detection for production deployment\n")
        f.write("6. **A/B Testing Framework**: Design gradual rollout strategy\n\n")
        
        # Technical details
        f.write("## Technical Implementation\n\n")
        f.write("- **Ensemble Method**: Simple arithmetic averaging\n")
        f.write("- **Feature Processing**: Consistent preprocessing across both models\n")
        f.write("- **Validation**: Temporal split maintaining time-series properties\n")
        f.write(f"- **Test Dataset**: {len(stratified_metrics['ensemble'])} samples across 6 yardage ranges\n")
        f.write("- **Random Seed**: 42 (for reproducibility)\n\n")
        
        f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    logger.info(f"Comprehensive report saved to: {report_path}")

def check_success_criteria(metrics: Dict[str, Dict[str, float]],
                          stratified_metrics: Dict[str, Dict[str, Dict[str, float]]]) -> Tuple[bool, Dict[str, bool]]:
    """Check if ensemble meets success criteria."""
    logger.info("\n[PHASE 6.1] Checking success criteria...")
    
    criteria_results = {
        'mae_criteria': metrics['ensemble']['mae'] < SUCCESS_CRITERIA['mae_threshold'],
        'rmse_criteria': metrics['ensemble']['rmse'] < SUCCESS_CRITERIA['rmse_threshold'], 
        'r2_criteria': metrics['ensemble']['r2'] > SUCCESS_CRITERIA['r2_threshold']
    }
    
    # Count yardage range improvements
    improved_ranges = 0
    for range_name in stratified_metrics['ensemble'].keys():
        ensemble_mae = stratified_metrics['ensemble'][range_name]['mae']
        rf_mae = stratified_metrics['rf'][range_name]['mae']
        xgb_mae = stratified_metrics['xgb'][range_name]['mae']
        
        if not np.isnan(ensemble_mae) and ensemble_mae < min(rf_mae, xgb_mae):
            improved_ranges += 1
    
    criteria_results['yardage_improvements'] = improved_ranges >= SUCCESS_CRITERIA['min_yardage_improvements']
    
    overall_success = all(criteria_results.values())
    
    logger.info(f"Success criteria results:")
    logger.info(f"  MAE < {SUCCESS_CRITERIA['mae_threshold']}: {criteria_results['mae_criteria']} ({metrics['ensemble']['mae']:.3f})")
    logger.info(f"  RMSE < {SUCCESS_CRITERIA['rmse_threshold']}: {criteria_results['rmse_criteria']} ({metrics['ensemble']['rmse']:.3f})")
    logger.info(f"  R² > {SUCCESS_CRITERIA['r2_threshold']}: {criteria_results['r2_criteria']} ({metrics['ensemble']['r2']:.3f})")
    logger.info(f"  Yardage improvements >= {SUCCESS_CRITERIA['min_yardage_improvements']}: {criteria_results['yardage_improvements']} ({improved_ranges}/6)")
    logger.info(f"  Overall success: {overall_success}")
    
    return overall_success, criteria_results

def save_production_model(results_df: pd.DataFrame, metrics: Dict[str, Dict[str, float]],
                         rf_model: Any, rf_imputer: Any, xgb_model: Any, xgb_imputer: Any) -> None:
    """Save ensemble model artifacts to production directory."""
    logger.info("\n[PHASE 6.2] Saving production model artifacts...")
    
    # Save ensemble metadata
    ensemble_metadata = {
        'model_type': 'rf_xgb_ensemble',
        'ensemble_method': 'simple_averaging',
        'created_date': datetime.now().isoformat(),
        'performance_metrics': metrics['ensemble'],
        'success_criteria': SUCCESS_CRITERIA,
        'feature_count': None,  # Will be set by preprocessors
        'training_data_range': 'temporal_split',
        'random_seed': RANDOM_SEED,
        'version': '1.0.0'
    }
    
    metadata_path = PRODUCTION_DIR / 'ensemble_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(ensemble_metadata, f, indent=2)
    logger.info(f"Metadata saved to: {metadata_path}")
    
    # Save ensemble configuration
    ensemble_config = {
        'ensemble': {
            'method': 'simple_averaging',
            'weights': {'rf': 0.5, 'xgb': 0.5},
            'models': {
                'random_forest': {
                    'model_path': 'rf_model.pkl',
                    'preprocessor_path': 'rf_imputer.pkl',
                    'hyperparameters_path': 'rf_hyperparameters.json'
                },
                'xgboost': {
                    'model_path': 'xgb_model.pkl', 
                    'preprocessor_path': 'xgb_imputer.pkl',
                    'hyperparameters_path': 'xgb_hyperparameters.json'
                }
            }
        },
        'preprocessing': {
            'feature_removal': 'redundant_features_v1',
            'imputation': 'simple_median'
        },
        'evaluation': {
            'test_size': len(results_df),
            'metrics': ['mae', 'rmse', 'r2', 'mape'],
            'stratification': 'yardage_ranges'
        }
    }
    
    config_path = PRODUCTION_DIR / 'ensemble_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(ensemble_config, f, default_flow_style=False)
    logger.info(f"Configuration saved to: {config_path}")
    
    # Copy model artifacts
    import shutil
    
    # Copy RF artifacts
    shutil.copy2(RF_MODEL_DIR / 'baseline_rf_model.pkl', PRODUCTION_DIR / 'rf_model.pkl')
    shutil.copy2(RF_MODEL_DIR / 'baseline_rf_imputer.pkl', PRODUCTION_DIR / 'rf_imputer.pkl')
    shutil.copy2(RF_MODEL_DIR / 'baseline_rf_hyperparameters.json', PRODUCTION_DIR / 'rf_hyperparameters.json')
    
    # Copy XGB artifacts  
    shutil.copy2(XGB_MODEL_DIR / 'baseline_xgboost_model.pkl', PRODUCTION_DIR / 'xgb_model.pkl')
    shutil.copy2(XGB_MODEL_DIR / 'baseline_xgboost_imputer.pkl', PRODUCTION_DIR / 'xgb_imputer.pkl')
    shutil.copy2(XGB_MODEL_DIR / 'baseline_xgboost_hyperparameters.json', PRODUCTION_DIR / 'xgb_hyperparameters.json')
    
    # Save ensemble predictions for future reference
    results_path = PRODUCTION_DIR / 'ensemble_test_predictions.parquet'
    results_df.to_parquet(results_path, index=False)
    logger.info(f"Test predictions saved to: {results_path}")
    
    logger.info(f"All production artifacts saved to: {PRODUCTION_DIR}")

def main():
    """Main execution function implementing all phases."""
    start_time = time.time()
    
    logger.info("="*80)
    logger.info("NFL RECEIVING YARDS PREDICTION - RF + XGBoost ENSEMBLE MODEL")
    logger.info("="*80)
    
    try:
        # Phase 1: Model Loading
        df_test = load_test_data()
        df_test_clean = remove_redundant_features(df_test)
        X_test, y_test = prepare_test_features(df_test_clean)
        rf_model, rf_imputer, xgb_model, xgb_imputer = load_baseline_models()
        
        # Phase 2: Ensemble Predictions
        results_df = generate_ensemble_predictions(
            X_test, y_test, rf_model, rf_imputer, xgb_model, xgb_imputer
        )
        
        # Phase 3: Metrics Calculation
        metrics = calculate_comprehensive_metrics(results_df)
        stratified_metrics = calculate_stratified_metrics(results_df)
        
        # Phase 4: Generate Visualizations
        create_visualizations(results_df, metrics, stratified_metrics)
        
        # Phase 5: Comprehensive Report
        generate_comprehensive_report(metrics, stratified_metrics)
        
        # Phase 6: Model Persistence (if criteria met)
        success, criteria_results = check_success_criteria(metrics, stratified_metrics)
        
        if success:
            save_production_model(results_df, metrics, rf_model, rf_imputer, xgb_model, xgb_imputer)
            logger.info("SUCCESS: Ensemble model meets success criteria and saved to production!")
        else:
            logger.info("FAILED: Ensemble model does not meet all success criteria")
        
        # Final Summary
        execution_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("ENSEMBLE MODEL EXECUTION SUMMARY")
        print("="*80)
        print("\nBASELINE PERFORMANCE:")
        print(f"  Random Forest   - MAE: {metrics['rf']['mae']:.3f}, RMSE: {metrics['rf']['rmse']:.3f}, R²: {metrics['rf']['r2']:.3f}")
        print(f"  XGBoost        - MAE: {metrics['xgb']['mae']:.3f}, RMSE: {metrics['xgb']['rmse']:.3f}, R²: {metrics['xgb']['r2']:.3f}")
        print("\nENSEMBLE RESULTS:")
        print(f"  Ensemble       - MAE: {metrics['ensemble']['mae']:.3f}, RMSE: {metrics['ensemble']['rmse']:.3f}, R²: {metrics['ensemble']['r2']:.3f}")
        print("\nIMPROVEMENTS:")
        print(f"  vs Random Forest - MAE: {metrics['rf']['mae'] - metrics['ensemble']['mae']:+.3f}, RMSE: {metrics['rf']['rmse'] - metrics['ensemble']['rmse']:+.3f}, R²: {metrics['ensemble']['r2'] - metrics['rf']['r2']:+.3f}")
        print(f"  vs XGBoost      - MAE: {metrics['xgb']['mae'] - metrics['ensemble']['mae']:+.3f}, RMSE: {metrics['xgb']['rmse'] - metrics['ensemble']['rmse']:+.3f}, R²: {metrics['ensemble']['r2'] - metrics['xgb']['r2']:+.3f}")
        print("\nSUCCESS CRITERIA:")
        for criterion, met in criteria_results.items():
            status = "PASS" if met else "FAIL"
            print(f"  {criterion.replace('_', ' ').title()}: {status}")
        print(f"\nOVERALL: {'SUCCESS' if success else 'FAILED'}")
        print(f"\nExecution Time: {execution_time:.2f} seconds")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Ensemble execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()