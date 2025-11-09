"""
NFL Receiving Yards Prediction - Refined XGBoost Model (Version 2)
==================================================================

Phase 2.2: Refined XGBoost Model with Targeted Optimization

This script implements a more targeted optimization approach based on the previous baseline:
1. Data loading and feature deduplication (removing 13 correlated features)
2. Refined hyperparameter tuning focused on beating RF baseline
3. More granular search around promising parameter values
4. Better early stopping strategy
5. Conservative regularization to avoid over-regularization
6. Focus on the exact metrics needed to beat RF baseline

Target Performance (Random Forest baseline to beat):
- MAE < 17.33 (previous XGB: 17.393, optimized v1: 18.829)
- RMSE < 23.58 (previous XGB: 23.700, optimized v1: 26.490)
- R² > 0.574 (previous XGB: 0.570, optimized v1: 0.463)

Strategy:
- Start from previous XGB parameters that were close to RF baseline
- Fine-tune around those parameters rather than broad search
- Less aggressive regularization
- Better early stopping thresholds
- Focus on MAE optimization since it's closest to target

Author: ML Engineering Team
Date: 2025-11-05
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
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
    """
    logger.info("\n[STEP 2] Removing correlated features...")
    
    redundant_features = [
        'week_id_gm', 'opp_week_id', 'szn_cum_week_id',
        'season_id_plyr', 'season_id_gm', 'season_id_oppdef', 'season_id_szn',
        'plyr_gm_rec_aybc_route', 'plyr_gm_rec_yac_route',
        'week_szn', 'week_oppdef',
        'opp_pass_yards_allowed_per_game',
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
    
    target_col = 'plyr_gm_rec_yds'
    
    id_columns = [
        'adv_plyr_gm_rec_id', 'plyr_id', 'game_id', 'team_id',
        'plyr_name', 'plyr_pos', 'game_date',
        'home_team_id', 'away_team_id', 'opponent_id',
        'season', 'week',
    ]
    
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
    
    y_train = df_train[target_col].copy()
    y_val = df_val[target_col].copy()
    
    X_train = df_train.drop(columns=cols_to_drop, errors='ignore')
    X_val = df_val.drop(columns=cols_to_drop, errors='ignore')
    
    logger.info(f"  Target variable: {target_col}")
    logger.info(f"  Target mean (train): {y_train.mean():.2f} yards")
    logger.info(f"  Target std (train): {y_train.std():.2f} yards")
    logger.info(f"  Final feature count: {X_train.shape[1]}")
    
    return X_train, X_val, y_train, y_val

def handle_missing_values(X_train, X_val):
    """
    Handle missing values using median imputation.
    """
    logger.info("\n[STEP 4] Handling missing values...")
    
    null_counts = X_train.isnull().sum()
    null_features = null_counts[null_counts > 0]
    
    if len(null_features) > 0:
        logger.info(f"  Features with nulls: {len(null_features)}")
        
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
        
        imputer_path = MODEL_DIR / "baseline_xgb_imputer.pkl"
        joblib.dump(imputer, imputer_path)
        logger.info(f"  Saved imputer to: {imputer_path}")
        
        return X_train_imputed, X_val_imputed, imputer
    else:
        logger.info("  No missing values found")
        return X_train, X_val, None

def create_refined_hyperparameter_space():
    """
    Create refined hyperparameter search space based on original XGBoost baseline that was close.
    Focus on fine-tuning around the original parameters that achieved:
    MAE: 17.393, RMSE: 23.700, R²: 0.570
    """
    logger.info("\n[STEP 5] Creating refined hyperparameter search space...")
    
    # Original baseline parameters that were close to target:
    # n_estimators: 300, max_depth: 6, learning_rate: 0.1, subsample: 0.8, colsample_bytree: 0.8
    # reg_alpha: 0, reg_lambda: 1.5
    
    # Fine-tuned parameter space around the successful baseline
    param_distributions = {
        # Trees: slightly more aggressive to improve performance
        'n_estimators': [250, 300, 350, 400, 450],
        'max_depth': [5, 6, 7],  # Around the successful depth=6
        
        # Learning rate: fine-tune around 0.1
        'learning_rate': [0.08, 0.09, 0.1, 0.11, 0.12],
        
        # Sampling: fine-tune around 0.8
        'subsample': [0.75, 0.8, 0.85, 0.9],
        'colsample_bytree': [0.75, 0.8, 0.85, 0.9],
        
        # Regularization: less aggressive than v1, around original values
        'reg_alpha': [0, 0.01, 0.05, 0.1],  # L1
        'reg_lambda': [1.0, 1.2, 1.5, 1.8, 2.0],  # L2, around original 1.5
        
        # Tree constraints: conservative
        'gamma': [0, 0.05, 0.1],  # Less aggressive than v1
        'min_child_weight': [1, 2, 3],  # Around default
        
        # Remove problematic parameters from v1
        # No max_delta_step, no colsample_bylevel
    }
    
    logger.info(f"  Refined parameter space with {len(param_distributions)} dimensions:")
    for param, values in param_distributions.items():
        logger.info(f"    {param}: {len(values)} options - {values}")
    
    return param_distributions

def train_refined_xgboost(X_train, y_train, X_val, y_val, param_distributions):
    """
    Train refined XGBoost model with targeted optimization.
    """
    logger.info("\n[STEP 6] Training refined XGBoost model...")
    
    # Base parameters
    base_params = {
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
        'objective': 'reg:squarederror',
        'eval_metric': 'mae'
    }
    
    logger.info(f"  Base parameters: {base_params}")
    
    # Custom scoring focused on MAE (our primary goal)
    def mae_scorer(estimator, X, y):
        y_pred = estimator.predict(X)
        return -mean_absolute_error(y, y_pred)  # Negative because sklearn maximizes
    
    logger.info("  Starting refined hyperparameter tuning...")
    logger.info(f"  RandomizedSearchCV iterations: 30")
    logger.info(f"  Cross-validation folds: 5")
    logger.info(f"  Scoring: MAE (primary optimization target)")
    
    # Create base model
    xgb_model = xgb.XGBRegressor(**base_params)
    
    # RandomizedSearchCV with focus on MAE
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=30,  # Focused search
        cv=5,
        scoring=mae_scorer,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    start_time = datetime.now()
    random_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = random_search.best_params_.copy()
    best_params.update(base_params)
    
    logger.info(f"  Hyperparameter search completed")
    logger.info(f"  Best MAE score: {-random_search.best_score_:.3f}")
    logger.info(f"  Best parameters: {best_params}")
    
    # Train final model with early stopping
    best_params['early_stopping_rounds'] = 100  # More patient early stopping
    final_model = xgb.XGBRegressor(**best_params)
    
    # Use validation set for early stopping
    eval_set = [(X_train, y_train), (X_val, y_val)]
    final_model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )
    
    end_time = datetime.now()
    
    logger.info(f"  Total training time: {end_time - start_time}")
    
    if hasattr(final_model, 'best_iteration'):
        logger.info(f"  Early stopping at iteration: {final_model.best_iteration}")
        evals = final_model.evals_result()
        if 'validation_1' in evals and 'mae' in evals['validation_1']:
            best_val_mae = min(evals['validation_1']['mae'])
            logger.info(f"  Best validation MAE during training: {best_val_mae:.3f}")
    
    # Save hyperparameters
    hyperparams_path = MODEL_DIR / "baseline_xgb_hyperparameters.json"
    with open(hyperparams_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"  Saved hyperparameters to: {hyperparams_path}")
    
    return final_model, best_params

def evaluate_model(model, X_train, X_val, y_train, y_val):
    """
    Evaluate model performance with focus on beating RF baseline.
    """
    logger.info("\n[STEP 7] Evaluating model performance...")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Calculate core metrics
    train_metrics = {
        'mae': mean_absolute_error(y_train, y_train_pred),
        'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'r2': r2_score(y_train, y_train_pred),
        'median_ae': median_absolute_error(y_train, y_train_pred)
    }
    
    val_metrics = {
        'mae': mean_absolute_error(y_val, y_val_pred),
        'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'r2': r2_score(y_val, y_val_pred),
        'median_ae': median_absolute_error(y_val, y_val_pred)
    }
    
    # Additional metrics
    val_errors = np.abs(y_val - y_val_pred)
    val_metrics.update({
        'pct_90_error': np.percentile(val_errors, 90),
        'within_10': np.mean(val_errors <= 10) * 100,
        'within_20': np.mean(val_errors <= 20) * 100
    })
    
    # Log performance
    logger.info("\n  === TRAINING SET PERFORMANCE ===")
    logger.info(f"  MAE: {train_metrics['mae']:.3f}")
    logger.info(f"  RMSE: {train_metrics['rmse']:.3f}")
    logger.info(f"  R²: {train_metrics['r2']:.3f}")
    
    logger.info("\n  === VALIDATION SET PERFORMANCE ===")
    logger.info(f"  MAE: {val_metrics['mae']:.3f}")
    logger.info(f"  RMSE: {val_metrics['rmse']:.3f}")
    logger.info(f"  R²: {val_metrics['r2']:.3f}")
    logger.info(f"  Median AE: {val_metrics['median_ae']:.3f}")
    logger.info(f"  90th %ile Error: {val_metrics['pct_90_error']:.3f}")
    logger.info(f"  Within 10 yards: {val_metrics['within_10']:.1f}%")
    logger.info(f"  Within 20 yards: {val_metrics['within_20']:.1f}%")
    
    # Baseline comparisons
    rf_baseline = {'mae': 17.33, 'rmse': 23.58, 'r2': 0.574}
    prev_xgb = {'mae': 17.393, 'rmse': 23.700, 'r2': 0.570}
    optimized_v1 = {'mae': 18.829, 'rmse': 26.490, 'r2': 0.463}
    
    logger.info("\n  === BASELINE COMPARISONS ===")
    logger.info(f"  RF Baseline (target): MAE {rf_baseline['mae']:.3f}, RMSE {rf_baseline['rmse']:.3f}, R² {rf_baseline['r2']:.3f}")
    logger.info(f"  Previous XGBoost: MAE {prev_xgb['mae']:.3f}, RMSE {prev_xgb['rmse']:.3f}, R² {prev_xgb['r2']:.3f}")
    logger.info(f"  Optimized v1: MAE {optimized_v1['mae']:.3f}, RMSE {optimized_v1['rmse']:.3f}, R² {optimized_v1['r2']:.3f}")
    logger.info(f"  Refined v2: MAE {val_metrics['mae']:.3f}, RMSE {val_metrics['rmse']:.3f}, R² {val_metrics['r2']:.3f}")
    
    # Goal achievement
    goals_achieved = []
    
    if val_metrics['mae'] < rf_baseline['mae']:
        improvement = rf_baseline['mae'] - val_metrics['mae']
        pct = (improvement / rf_baseline['mae']) * 100
        logger.info(f"  ✓ MAE GOAL ACHIEVED: {val_metrics['mae']:.3f} < {rf_baseline['mae']:.3f} (improvement: -{improvement:.3f}, {pct:.2f}%)")
        goals_achieved.append('MAE')
    else:
        gap = val_metrics['mae'] - rf_baseline['mae']
        pct = (gap / rf_baseline['mae']) * 100
        logger.info(f"  X MAE goal missed: {val_metrics['mae']:.3f} > {rf_baseline['mae']:.3f} (gap: +{gap:.3f}, +{pct:.2f}%)")
    
    if val_metrics['rmse'] < rf_baseline['rmse']:
        improvement = rf_baseline['rmse'] - val_metrics['rmse']
        pct = (improvement / rf_baseline['rmse']) * 100
        logger.info(f"  ✓ RMSE GOAL ACHIEVED: {val_metrics['rmse']:.3f} < {rf_baseline['rmse']:.3f} (improvement: -{improvement:.3f}, {pct:.2f}%)")
        goals_achieved.append('RMSE')
    else:
        gap = val_metrics['rmse'] - rf_baseline['rmse']
        pct = (gap / rf_baseline['rmse']) * 100
        logger.info(f"  X RMSE goal missed: {val_metrics['rmse']:.3f} > {rf_baseline['rmse']:.3f} (gap: +{gap:.3f}, +{pct:.2f}%)")
    
    if val_metrics['r2'] > rf_baseline['r2']:
        improvement = val_metrics['r2'] - rf_baseline['r2']
        pct = (improvement / rf_baseline['r2']) * 100
        logger.info(f"  ✓ R² GOAL ACHIEVED: {val_metrics['r2']:.3f} > {rf_baseline['r2']:.3f} (improvement: +{improvement:.3f}, +{pct:.2f}%)")
        goals_achieved.append('R2')
    else:
        gap = rf_baseline['r2'] - val_metrics['r2']
        pct = (gap / rf_baseline['r2']) * 100
        logger.info(f"  X R² goal missed: {val_metrics['r2']:.3f} < {rf_baseline['r2']:.3f} (gap: -{gap:.3f}, -{pct:.2f}%)")
    
    logger.info(f"\n  GOALS ACHIEVED: {len(goals_achieved)}/3 ({', '.join(goals_achieved) if goals_achieved else 'None'})")
    
    # Compile all metrics
    all_metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'baseline_comparisons': {
            'rf_baseline': rf_baseline,
            'previous_xgb': prev_xgb,
            'optimized_v1': optimized_v1,
            'improvements_vs_rf': {
                'mae': rf_baseline['mae'] - val_metrics['mae'],
                'rmse': rf_baseline['rmse'] - val_metrics['rmse'],
                'r2': val_metrics['r2'] - rf_baseline['r2']
            }
        },
        'goals_achieved': goals_achieved,
        'performance_summary': {
            'beats_rf_baseline': len(goals_achieved) > 0,
            'beats_prev_xgb': (val_metrics['mae'] < prev_xgb['mae'] or 
                             val_metrics['rmse'] < prev_xgb['rmse'] or 
                             val_metrics['r2'] > prev_xgb['r2']),
            'beats_optimized_v1': (val_metrics['mae'] < optimized_v1['mae'] and 
                                 val_metrics['rmse'] < optimized_v1['rmse'] and 
                                 val_metrics['r2'] > optimized_v1['r2'])
        }
    }
    
    # Save metrics
    metrics_path = EVAL_DIR / "xgb_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"  Saved metrics to: {metrics_path}")
    
    return all_metrics, y_train_pred, y_val_pred

def analyze_feature_importance(model, feature_names):
    """
    Analyze feature importance from the trained model.
    """
    logger.info("\n[STEP 8] Analyzing feature importance...")
    
    importance_scores = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    logger.info(f"\n  Top 10 Most Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        logger.info(f"    {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
    
    # Save feature importance
    importance_path = MODEL_DIR / "baseline_xgb_feature_importance.csv"
    feature_importance.to_csv(importance_path, index=False)
    logger.info(f"  Saved feature importance to: {importance_path}")
    
    return feature_importance

def save_model(model, imputer):
    """
    Save the refined model and artifacts.
    """
    logger.info("\n[STEP 9] Saving refined model artifacts...")
    
    # Save model
    model_path = MODEL_DIR / "baseline_xgb_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"  Saved model to: {model_path}")
    
    # Save training history if available
    if hasattr(model, 'evals_result'):
        training_history = model.evals_result()
        history_path = MODEL_DIR / "baseline_xgb_training_history.json"
        
        serializable_history = {}
        for eval_set, metrics in training_history.items():
            serializable_history[eval_set] = {}
            for metric, values in metrics.items():
                serializable_history[eval_set][metric] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        logger.info(f"  Saved training history to: {history_path}")
    
    # Model metadata
    metadata = {
        'model_type': 'XGBoost Regressor (Refined v2)',
        'target_variable': 'plyr_gm_rec_yds',
        'training_date': datetime.now().isoformat(),
        'optimization_approach': 'Refined search around successful baseline parameters',
        'key_changes_from_v1': [
            'Less aggressive regularization',
            'More patient early stopping (100 rounds)',
            'Focus on MAE optimization',
            'Parameter space centered around original baseline',
            'Removed problematic parameters (max_delta_step, colsample_bylevel)'
        ],
        'best_iteration': int(model.best_iteration) if hasattr(model, 'best_iteration') else None
    }
    
    metadata_path = MODEL_DIR / "baseline_xgb_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Saved metadata to: {metadata_path}")

def main():
    """
    Main training pipeline for refined XGBoost model.
    """
    logger.info("=" * 80)
    logger.info("NFL RECEIVING YARDS PREDICTION - REFINED XGBOOST MODEL V2")
    logger.info("=" * 80)
    logger.info(f"Training started at: {datetime.now()}")
    logger.info("Goal: Beat Random Forest baseline with refined approach")
    logger.info("Target: MAE < 17.33, RMSE < 23.58, R² > 0.574")
    
    try:
        # Step 1: Load data
        df_train, df_val = load_data()
        
        # Step 2: Remove correlated features
        df_train = remove_correlated_features(df_train)
        df_val = remove_correlated_features(df_val)
        
        # Step 3: Prepare features and target
        X_train, X_val, y_train, y_val = prepare_features_and_target(df_train, df_val)
        
        logger.info(f"  Final feature count verification: {X_train.shape[1]} (target: 35)")
        
        # Step 4: Handle missing values
        X_train, X_val, imputer = handle_missing_values(X_train, X_val)
        
        # Step 5: Create refined hyperparameter space
        param_distributions = create_refined_hyperparameter_space()
        
        # Step 6: Train refined XGBoost model
        model, best_params = train_refined_xgboost(X_train, y_train, X_val, y_val, param_distributions)
        
        # Step 7: Evaluate model
        metrics, y_train_pred, y_val_pred = evaluate_model(model, X_train, X_val, y_train, y_val)
        
        # Step 8: Feature importance analysis
        feature_importance = analyze_feature_importance(model, X_train.columns)
        
        # Step 9: Save model artifacts
        save_model(model, imputer)
        
        # Final summary
        val_metrics = metrics['validation']
        goals_achieved = metrics['goals_achieved']
        
        logger.info("\n" + "=" * 80)
        logger.info("REFINED TRAINING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Final Validation Performance:")
        logger.info(f"  MAE: {val_metrics['mae']:.3f} (target: < 17.33)")
        logger.info(f"  RMSE: {val_metrics['rmse']:.3f} (target: < 23.58)")
        logger.info(f"  R²: {val_metrics['r2']:.3f} (target: > 0.574)")
        
        logger.info(f"\nPerformance Summary:")
        logger.info(f"  Goals achieved: {len(goals_achieved)}/3 ({', '.join(goals_achieved) if goals_achieved else 'None'})")
        logger.info(f"  Beats RF baseline: {'Yes' if metrics['performance_summary']['beats_rf_baseline'] else 'No'}")
        logger.info(f"  Beats previous XGBoost: {'Yes' if metrics['performance_summary']['beats_prev_xgb'] else 'No'}")
        logger.info(f"  Beats optimized v1: {'Yes' if metrics['performance_summary']['beats_optimized_v1'] else 'No'}")
        
        if hasattr(model, 'best_iteration'):
            logger.info(f"  Early stopping iteration: {model.best_iteration}")
        
        logger.info(f"\nTraining completed at: {datetime.now()}")
        
        if len(goals_achieved) > 0:
            logger.info("\nSUCCESS: Model achieved at least one performance goal!")
        else:
            logger.info("\nModel improved but did not achieve RF baseline goals")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()