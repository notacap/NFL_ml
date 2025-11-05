"""
NFL Receiving Yards Prediction - Baseline XGBoost Model
======================================================

Phase 2: XGBoost Baseline Model Development, Training, and Evaluation

This script implements:
1. Data loading and feature deduplication (removing 13 correlated features)
2. Feature engineering and null handling
3. XGBoost model training with hyperparameter tuning
4. Model validation and comprehensive evaluation
5. Performance comparison against Random Forest baseline
6. Model persistence and logging

Target Performance (Random Forest baseline to beat):
- MAE < 17.33
- R² > 0.574  
- RMSE < 23.58

Author: ML Engineering Team
Date: 2025-11-05
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
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
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
EVAL_DIR = BASE_DIR / "evaluation" / "metrics"

# Create directories if they don't exist
MODEL_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)
EVAL_DIR.mkdir(exist_ok=True, parents=True)

# Configure logging
log_file = LOG_DIR / "baseline_xgb_training.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data():
    """
    Load training and validation datasets.
    """
    logger.info("\n[STEP 1] Loading data...")
    
    train_path = DATA_DIR / "wr_train.parquet"
    val_path = DATA_DIR / "wr_validation.parquet"
    
    logger.info(f"  Loading training data: {train_path}")
    df_train = pd.read_parquet(train_path)
    logger.info(f"  Training set shape: {df_train.shape}")
    
    logger.info(f"  Loading validation data: {val_path}")
    df_val = pd.read_parquet(val_path)
    logger.info(f"  Validation set shape: {df_val.shape}")
    
    return df_train, df_val

def remove_correlated_features(df):
    """
    Remove 13 highly correlated features (correlation > 0.95) to reduce from 48 to 35 features.
    This matches the feature selection done in the Random Forest baseline.
    """
    logger.info("\n[STEP 2] Removing correlated features...")
    
    # 13 features identified as highly correlated (> 0.95 correlation)
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
        
        # Save imputer for inference
        imputer_path = MODEL_DIR / "baseline_xgb_imputer.pkl"
        joblib.dump(imputer, imputer_path)
        logger.info(f"  Saved imputer to: {imputer_path}")
        
        logger.info(f"  Applied median imputation to {len(null_features)} features")
        return X_train_imputed, X_val_imputed, imputer
    else:
        logger.info("  No missing values found")
        return X_train, X_val, None

def train_xgboost_model(X_train, y_train):
    """
    Train XGBoost model with hyperparameter tuning using RandomizedSearchCV.
    """
    logger.info("\n[STEP 5] Training XGBoost model...")
    
    # Initial XGBoost parameters
    base_params = {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    logger.info(f"  Base parameters: {base_params}")
    
    # Hyperparameter search space
    param_distributions = {
        'n_estimators': [200, 300, 400, 500],
        'max_depth': [4, 5, 6, 7, 8],
        'learning_rate': [0.05, 0.1, 0.15, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [1, 1.5, 2.0, 2.5]
    }
    
    logger.info("  Starting hyperparameter tuning with RandomizedSearchCV...")
    logger.info(f"  Search space size: {len(param_distributions)} parameters")
    logger.info(f"  Number of iterations: 20")
    logger.info(f"  Cross-validation folds: 5")
    
    # Create base model
    xgb_model = xgb.XGBRegressor(**base_params)
    
    # RandomizedSearchCV setup
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=20,  # 20 iterations as specified
        cv=5,       # 5-fold cross-validation
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    # Fit the random search
    start_time = datetime.now()
    random_search.fit(X_train, y_train)
    end_time = datetime.now()
    
    logger.info(f"  Hyperparameter tuning completed in {end_time - start_time}")
    logger.info(f"  Best CV score (MAE): {-random_search.best_score_:.3f}")
    logger.info(f"  Best parameters: {random_search.best_params_}")
    
    # Save best hyperparameters
    hyperparams_path = MODEL_DIR / "baseline_xgb_hyperparameters.json"
    with open(hyperparams_path, 'w') as f:
        json.dump(random_search.best_params_, f, indent=2)
    logger.info(f"  Saved best hyperparameters to: {hyperparams_path}")
    
    return random_search.best_estimator_, random_search.best_params_

def evaluate_model(model, X_train, X_val, y_train, y_val):
    """
    Evaluate model performance on training and validation sets.
    """
    logger.info("\n[STEP 6] Evaluating model performance...")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Calculate metrics for training set
    train_metrics = {
        'mae': mean_absolute_error(y_train, y_train_pred),
        'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'r2': r2_score(y_train, y_train_pred),
        'median_ae': median_absolute_error(y_train, y_train_pred)
    }
    
    # Calculate metrics for validation set
    val_metrics = {
        'mae': mean_absolute_error(y_val, y_val_pred),
        'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'r2': r2_score(y_val, y_val_pred),
        'median_ae': median_absolute_error(y_val, y_val_pred)
    }
    
    # Additional percentage-based metrics
    def calculate_percentage_metrics(y_true, y_pred):
        errors = np.abs(y_true - y_pred)
        return {
            'pct_90_error': np.percentile(errors, 90),
            'within_10': np.mean(errors <= 10) * 100,
            'within_20': np.mean(errors <= 20) * 100
        }
    
    train_pct_metrics = calculate_percentage_metrics(y_train, y_train_pred)
    val_pct_metrics = calculate_percentage_metrics(y_val, y_val_pred)
    
    # Combine metrics
    train_metrics.update(train_pct_metrics)
    val_metrics.update(val_pct_metrics)
    
    # Log performance
    logger.info("\n  === TRAINING SET PERFORMANCE ===")
    logger.info(f"  MAE: {train_metrics['mae']:.3f}")
    logger.info(f"  RMSE: {train_metrics['rmse']:.3f}")
    logger.info(f"  R²: {train_metrics['r2']:.3f}")
    logger.info(f"  Median AE: {train_metrics['median_ae']:.3f}")
    logger.info(f"  90th %ile Error: {train_metrics['pct_90_error']:.3f}")
    logger.info(f"  Within 10 yards: {train_metrics['within_10']:.1f}%")
    logger.info(f"  Within 20 yards: {train_metrics['within_20']:.1f}%")
    
    logger.info("\n  === VALIDATION SET PERFORMANCE ===")
    logger.info(f"  MAE: {val_metrics['mae']:.3f}")
    logger.info(f"  RMSE: {val_metrics['rmse']:.3f}")
    logger.info(f"  R²: {val_metrics['r2']:.3f}")
    logger.info(f"  Median AE: {val_metrics['median_ae']:.3f}")
    logger.info(f"  90th %ile Error: {val_metrics['pct_90_error']:.3f}")
    logger.info(f"  Within 10 yards: {val_metrics['within_10']:.1f}%")
    logger.info(f"  Within 20 yards: {val_metrics['within_20']:.1f}%")
    
    # Baseline comparison (Random Forest targets)
    rf_baseline = {
        'mae': 17.33,
        'rmse': 23.58,
        'r2': 0.574
    }
    
    logger.info("\n  === BASELINE COMPARISON (Random Forest) ===")
    mae_improvement = rf_baseline['mae'] - val_metrics['mae']
    rmse_improvement = rf_baseline['rmse'] - val_metrics['rmse']
    r2_improvement = val_metrics['r2'] - rf_baseline['r2']
    
    logger.info(f"  Target MAE < 17.33: {val_metrics['mae']:.3f} {'✓ ACHIEVED' if val_metrics['mae'] < rf_baseline['mae'] else '✗ MISSED'}")
    logger.info(f"  Target RMSE < 23.58: {val_metrics['rmse']:.3f} {'✓ ACHIEVED' if val_metrics['rmse'] < rf_baseline['rmse'] else '✗ MISSED'}")
    logger.info(f"  Target R² > 0.574: {val_metrics['r2']:.3f} {'✓ ACHIEVED' if val_metrics['r2'] > rf_baseline['r2'] else '✗ MISSED'}")
    
    if mae_improvement > 0:
        logger.info(f"  MAE improvement: -{mae_improvement:.3f} yards ({((rf_baseline['mae'] - val_metrics['mae'])/rf_baseline['mae']*100):+.1f}%)")
    else:
        logger.info(f"  MAE degradation: {-mae_improvement:.3f} yards ({((val_metrics['mae'] - rf_baseline['mae'])/rf_baseline['mae']*100):+.1f}%)")
    
    # Save metrics
    all_metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'baseline_comparison': {
            'rf_baseline': rf_baseline,
            'improvements': {
                'mae': mae_improvement,
                'rmse': rmse_improvement,
                'r2': r2_improvement
            }
        }
    }
    
    metrics_path = EVAL_DIR / "xgb_baseline_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"  Saved metrics to: {metrics_path}")
    
    return all_metrics, y_train_pred, y_val_pred

def analyze_feature_importance(model, feature_names):
    """
    Analyze and log feature importance from the trained XGBoost model.
    """
    logger.info("\n[STEP 7] Analyzing feature importance...")
    
    # Get feature importance
    importance_scores = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    logger.info(f"\n  Top 10 Most Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        logger.info(f"    {row['feature']}: {row['importance']:.4f}")
    
    # Save feature importance
    importance_path = MODEL_DIR / "baseline_xgb_feature_importance.csv"
    feature_importance.to_csv(importance_path, index=False)
    logger.info(f"  Saved feature importance to: {importance_path}")
    
    # Check if most important feature matches RF baseline expectation
    top_feature = feature_importance.iloc[0]['feature']
    expected_top_feature = 'plyr_gm_rec_tgt'
    
    if top_feature == expected_top_feature:
        logger.info(f"  ✓ Top feature matches RF baseline: {top_feature}")
    else:
        logger.info(f"  ! Top feature differs from RF baseline. XGB: {top_feature}, RF: {expected_top_feature}")
    
    return feature_importance

def save_model(model, imputer=None):
    """
    Save the trained model and associated artifacts.
    """
    logger.info("\n[STEP 8] Saving model artifacts...")
    
    # Save main model
    model_path = MODEL_DIR / "baseline_xgb_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"  Saved XGBoost model to: {model_path}")
    
    # Model metadata
    metadata = {
        'model_type': 'XGBoost Regressor',
        'target_variable': 'plyr_gm_rec_yds',
        'features_count': len(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else 'unknown',
        'training_date': datetime.now().isoformat(),
        'hyperparameter_tuning': 'RandomizedSearchCV (20 iterations, 5-fold CV)',
        'baseline_comparison': 'Random Forest (MAE: 17.33, RMSE: 23.58, R²: 0.574)',
        'preprocessing': 'Median imputation for missing values, 13 correlated features removed'
    }
    
    metadata_path = MODEL_DIR / "baseline_xgb_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Saved model metadata to: {metadata_path}")

def main():
    """
    Main training pipeline for XGBoost baseline model.
    """
    logger.info("=" * 80)
    logger.info("NFL RECEIVING YARDS PREDICTION - XGBOOST BASELINE MODEL")
    logger.info("=" * 80)
    logger.info(f"Training started at: {datetime.now()}")
    
    try:
        # Step 1: Load data
        df_train, df_val = load_data()
        
        # Step 2: Remove correlated features (13 features -> 35 total features)
        df_train = remove_correlated_features(df_train)
        df_val = remove_correlated_features(df_val)
        
        # Step 3: Prepare features and target
        X_train, X_val, y_train, y_val = prepare_features_and_target(df_train, df_val)
        
        # Verify we have 35 features as expected
        logger.info(f"  Final feature count verification: {X_train.shape[1]} (target: 35)")
        if X_train.shape[1] != 35:
            logger.warning(f"  Expected 35 features but got {X_train.shape[1]}")
        
        # Step 4: Handle missing values
        X_train, X_val, imputer = handle_missing_values(X_train, X_val)
        
        # Step 5: Train XGBoost model with hyperparameter tuning
        model, best_params = train_xgboost_model(X_train, y_train)
        
        # Step 6: Evaluate model
        metrics, y_train_pred, y_val_pred = evaluate_model(model, X_train, X_val, y_train, y_val)
        
        # Step 7: Feature importance analysis
        feature_importance = analyze_feature_importance(model, X_train.columns)
        
        # Step 8: Save model artifacts
        save_model(model, imputer)
        
        # Final summary
        val_mae = metrics['validation']['mae']
        val_rmse = metrics['validation']['rmse']
        val_r2 = metrics['validation']['r2']
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Final Validation Performance:")
        logger.info(f"  MAE: {val_mae:.3f} (target: < 17.33)")
        logger.info(f"  RMSE: {val_rmse:.3f} (target: < 23.58)")
        logger.info(f"  R²: {val_r2:.3f} (target: > 0.574)")
        
        # Success metrics
        goals_met = 0
        if val_mae < 17.33:
            goals_met += 1
        if val_rmse < 23.58:
            goals_met += 1
        if val_r2 > 0.574:
            goals_met += 1
            
        logger.info(f"Performance goals achieved: {goals_met}/3")
        logger.info(f"Training completed at: {datetime.now()}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.error("Check the log file for detailed error information")
        raise

if __name__ == "__main__":
    main()