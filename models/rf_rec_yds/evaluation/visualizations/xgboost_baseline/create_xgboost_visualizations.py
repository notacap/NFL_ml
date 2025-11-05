"""
NFL Receiving Yards Prediction - XGBoost Baseline Model Visualizations
=====================================================================

This script creates comprehensive visualizations for the optimized XGBoost model:
1. Actual vs Predicted scatter plot with perfect prediction line
2. Residual distribution histogram with normality test
3. Residuals vs Predicted values (heteroscedasticity check)
4. Q-Q plot for residual normality assessment
5. Feature importance plot (top 20 features)

The script loads the trained optimized XGBoost model and applies the same
preprocessing pipeline used during training.

Author: ML Engineering Team
Date: 2025-11-05
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.impute import SimpleImputer
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for high-quality plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11

# Configure paths
BASE_DIR = Path("C:/Users/nocap/Desktop/code/NFL_ml/models/rf_rec_yds")
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "splits" / "temporal"
VIZ_DIR = BASE_DIR / "evaluation" / "visualizations" / "xgboost_baseline"

# Create visualization directory if it doesn't exist
VIZ_DIR.mkdir(exist_ok=True, parents=True)

def load_model_and_preprocessors():
    """
    Load the trained XGBoost model and preprocessing artifacts.
    """
    print("Loading trained XGBoost model and preprocessors...")
    
    # Load model
    model_path = MODEL_DIR / "baseline_xgb_optimized_v2_model.pkl"
    model = joblib.load(model_path)
    print(f"  + Loaded model from: {model_path}")
    
    # Load imputer
    imputer_path = MODEL_DIR / "baseline_xgb_optimized_v2_imputer.pkl"
    imputer = joblib.load(imputer_path)
    print(f"  + Loaded imputer from: {imputer_path}")
    
    # Load feature importance
    importance_path = MODEL_DIR / "baseline_xgb_optimized_v2_feature_importance.csv"
    feature_importance = pd.read_csv(importance_path)
    print(f"  + Loaded feature importance from: {importance_path}")
    
    # Load metadata
    metadata_path = MODEL_DIR / "baseline_xgb_optimized_v2_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"  + Loaded metadata from: {metadata_path}")
    
    return model, imputer, feature_importance, metadata

def load_and_preprocess_validation_data():
    """
    Load validation data and apply the same preprocessing as during training.
    """
    print("\nLoading and preprocessing validation data...")
    
    # Load validation data
    val_path = DATA_DIR / "wr_validation.parquet"
    df_val = pd.read_parquet(val_path)
    print(f"  + Loaded validation data: {df_val.shape}")
    
    # Remove correlated features (same 13 features as training)
    redundant_features = [
        'week_id_gm', 'opp_week_id', 'szn_cum_week_id',
        'season_id_plyr', 'season_id_gm', 'season_id_oppdef', 'season_id_szn',
        'plyr_gm_rec_aybc_route', 'plyr_gm_rec_yac_route',
        'week_szn', 'week_oppdef',
        'opp_pass_yards_allowed_per_game',
        'plyr_gm_rec_no_brkn_tkl',
    ]
    
    df_val_dedup = df_val.drop(columns=redundant_features, errors='ignore')
    print(f"  + Removed {len(redundant_features)} correlated features")
    
    # Separate features and target
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
    
    y_val = df_val_dedup[target_col].copy()
    X_val = df_val_dedup.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"  + Extracted features: {X_val.shape[1]} features")
    print(f"  + Target statistics: mean={y_val.mean():.2f}, std={y_val.std():.2f}")
    
    return X_val, y_val

def generate_predictions(model, imputer, X_val):
    """
    Generate predictions using the trained model and preprocessor.
    """
    print("\nGenerating predictions...")
    
    # Apply imputation (same as training)
    X_val_imputed = pd.DataFrame(
        imputer.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    print(f"  + Applied imputation to handle missing values")
    
    # Generate predictions
    y_pred = model.predict(X_val_imputed)
    print(f"  + Generated {len(y_pred)} predictions")
    
    return y_pred

def create_actual_vs_predicted_plot(y_val, y_pred, save_path):
    """
    Create actual vs predicted scatter plot with perfect prediction line.
    """
    print("Creating actual vs predicted plot...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate metrics for display
    mae = np.mean(np.abs(y_val - y_pred))
    rmse = np.sqrt(np.mean((y_val - y_pred)**2))
    r2 = stats.pearsonr(y_val, y_pred)[0]**2
    
    # Create scatter plot
    ax.scatter(y_val, y_pred, alpha=0.6, s=30, color='steelblue', edgecolors='white', linewidth=0.5)
    
    # Perfect prediction line (y = x)
    min_val = min(y_val.min(), y_pred.min())
    max_val = max(y_val.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
            label='Perfect Prediction', alpha=0.8)
    
    # Fit line
    z = np.polyfit(y_val, y_pred, 1)
    p = np.poly1d(z)
    ax.plot(y_val, p(y_val), 'orange', linewidth=2, alpha=0.8, label=f'Fitted Line (y = {z[0]:.2f}x + {z[1]:.2f})')
    
    # Formatting
    ax.set_xlabel('Actual Receiving Yards', fontsize=12)
    ax.set_ylabel('Predicted Receiving Yards', fontsize=12)
    ax.set_title('XGBoost Model: Actual vs Predicted Receiving Yards\nValidation Set Performance', fontsize=14, pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add metrics text box
    metrics_text = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.3f}\nn = {len(y_val):,}'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Equal aspect ratio and limits
    ax.set_aspect('equal', adjustable='box')
    margin = (max_val - min_val) * 0.05
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  + Saved to: {save_path}")

def create_residual_distribution_plot(y_val, y_pred, save_path):
    """
    Create residual distribution histogram with normality test.
    """
    print("Creating residual distribution plot...")
    
    residuals = y_val - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram with normal overlay
    ax1.hist(residuals, bins=50, density=True, alpha=0.7, color='lightblue', 
             edgecolor='black', linewidth=0.5, label='Residuals')
    
    # Overlay normal distribution
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    normal_curve = stats.norm.pdf(x, mu, sigma)
    ax1.plot(x, normal_curve, 'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
    
    ax1.set_xlabel('Residuals (Actual - Predicted)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Distribution of Residuals', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    box_plot = ax2.boxplot(residuals, vert=True, patch_artist=True, 
                          boxprops=dict(facecolor='lightblue', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    ax2.set_title('Residuals Box Plot', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    # Statistical tests
    shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
    jarque_bera_stat, jarque_bera_p = stats.jarque_bera(residuals)
    
    # Add statistics text
    stats_text = f'Normality Tests:\n'
    stats_text += f'Shapiro-Wilk: p = {shapiro_p:.4f}\n'
    stats_text += f'Jarque-Bera: p = {jarque_bera_p:.4f}\n\n'
    stats_text += f'Descriptive Stats:\n'
    stats_text += f'Mean: {mu:.3f}\n'
    stats_text += f'Std: {sigma:.3f}\n'
    stats_text += f'Skewness: {stats.skew(residuals):.3f}\n'
    stats_text += f'Kurtosis: {stats.kurtosis(residuals):.3f}'
    
    fig.text(0.02, 0.98, stats_text, transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  + Saved to: {save_path}")

def create_residuals_vs_predicted_plot(y_pred, y_val, save_path):
    """
    Create residuals vs predicted values plot to check heteroscedasticity.
    """
    print("Creating residuals vs predicted plot...")
    
    residuals = y_val - y_pred
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(y_pred, residuals, alpha=0.6, s=30, color='steelblue', 
               edgecolors='white', linewidth=0.5)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Zero Residual')
    
    # Add LOWESS smooth line to show trend
    from scipy.signal import savgol_filter
    if len(y_pred) > 100:
        # Sort for smooth line
        sorted_indices = np.argsort(y_pred)
        y_pred_sorted = y_pred[sorted_indices]
        residuals_sorted = residuals.iloc[sorted_indices]
        
        # Apply smoothing
        try:
            window_length = min(101, len(y_pred_sorted) // 10)
            if window_length % 2 == 0:
                window_length += 1
            if window_length >= 3:
                smooth_residuals = savgol_filter(residuals_sorted, window_length, 3)
                ax.plot(y_pred_sorted, smooth_residuals, 'orange', linewidth=2, 
                       alpha=0.8, label='Trend Line')
        except:
            pass
    
    # Calculate Breusch-Pagan test for heteroscedasticity
    # Simple version: correlation between |residuals| and predicted values
    abs_residuals = np.abs(residuals)
    het_corr, het_p = stats.pearsonr(y_pred, abs_residuals)
    
    ax.set_xlabel('Predicted Values', fontsize=12)
    ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    ax.set_title('Residuals vs Predicted Values\nCheck for Heteroscedasticity', fontsize=14, pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add heteroscedasticity test results
    het_text = f'Heteroscedasticity Test:\n'
    het_text += f'Correlation(|residuals|, predicted): {het_corr:.4f}\n'
    het_text += f'p-value: {het_p:.4f}\n'
    if het_p < 0.05:
        het_text += 'Evidence of heteroscedasticity'
    else:
        het_text += 'No strong evidence of heteroscedasticity'
    
    ax.text(0.05, 0.95, het_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  + Saved to: {save_path}")

def create_qq_plot(y_val, y_pred, save_path):
    """
    Create Q-Q plot for residual normality assessment.
    """
    print("Creating Q-Q plot...")
    
    residuals = y_val - y_pred
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Generate Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax)
    
    # Customize the plot
    ax.set_title('Q-Q Plot: Residuals vs Normal Distribution\nAssessing Normality Assumption', 
                fontsize=14, pad=20)
    ax.set_xlabel('Theoretical Quantiles (Normal Distribution)', fontsize=12)
    ax.set_ylabel('Sample Quantiles (Residuals)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Calculate R² of Q-Q plot
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
    sample_quantiles = np.sort(residuals)
    qq_r2 = stats.pearsonr(theoretical_quantiles, sample_quantiles)[0]**2
    
    # Add R² text
    qq_text = f'Q-Q Plot R²: {qq_r2:.4f}\n'
    if qq_r2 > 0.95:
        qq_text += 'Excellent normality'
    elif qq_r2 > 0.90:
        qq_text += 'Good normality'
    elif qq_r2 > 0.80:
        qq_text += 'Moderate normality'
    else:
        qq_text += 'Poor normality'
    
    ax.text(0.05, 0.95, qq_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  + Saved to: {save_path}")

def create_feature_importance_plot(feature_importance, save_path):
    """
    Create feature importance plot for top 20 features.
    """
    print("Creating feature importance plot...")
    
    # Get top 20 features
    top_features = feature_importance.head(20).copy()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create horizontal bar plot
    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, top_features['importance'], 
                   color='steelblue', alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Customize the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'], fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('XGBoost Model: Top 20 Feature Importance\nOptimized Model (v2)', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', ha='left', va='center', fontsize=9)
    
    # Invert y-axis to show highest importance at top
    ax.invert_yaxis()
    
    # Add total importance covered
    total_importance = feature_importance['importance'].sum()
    top20_importance = top_features['importance'].sum()
    coverage = (top20_importance / total_importance) * 100
    
    coverage_text = f'Top 20 features explain {coverage:.1f}% of total importance'
    ax.text(0.98, 0.02, coverage_text, transform=ax.transAxes, fontsize=10,
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  + Saved to: {save_path}")

def calculate_performance_metrics(y_val, y_pred):
    """
    Calculate comprehensive performance metrics.
    """
    mae = np.mean(np.abs(y_val - y_pred))
    rmse = np.sqrt(np.mean((y_val - y_pred)**2))
    r2 = stats.pearsonr(y_val, y_pred)[0]**2
    median_ae = np.median(np.abs(y_val - y_pred))
    
    # Additional metrics
    errors = np.abs(y_val - y_pred)
    within_10 = np.mean(errors <= 10) * 100
    within_20 = np.mean(errors <= 20) * 100
    pct_90_error = np.percentile(errors, 90)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'median_ae': median_ae,
        'within_10': within_10,
        'within_20': within_20,
        'pct_90_error': pct_90_error
    }

def main():
    """
    Main function to create all XGBoost model visualizations.
    """
    print("=" * 80)
    print("CREATING XGBOOST BASELINE MODEL VISUALIZATIONS")
    print("=" * 80)
    
    try:
        # Step 1: Load model and preprocessors
        model, imputer, feature_importance, metadata = load_model_and_preprocessors()
        
        # Step 2: Load and preprocess validation data
        X_val, y_val = load_and_preprocess_validation_data()
        
        # Step 3: Generate predictions
        y_pred = generate_predictions(model, imputer, X_val)
        
        # Step 4: Calculate performance metrics
        metrics = calculate_performance_metrics(y_val, y_pred)
        print(f"\nModel Performance on Validation Set:")
        print(f"  MAE: {metrics['mae']:.3f}")
        print(f"  RMSE: {metrics['rmse']:.3f}")
        print(f"  R²: {metrics['r2']:.3f}")
        print(f"  Median AE: {metrics['median_ae']:.3f}")
        print(f"  Within 10 yards: {metrics['within_10']:.1f}%")
        print(f"  Within 20 yards: {metrics['within_20']:.1f}%")
        
        # Step 5: Create visualizations
        print(f"\nCreating visualizations in: {VIZ_DIR}")
        
        # 1. Actual vs Predicted
        actual_vs_pred_path = VIZ_DIR / "actual_vs_predicted.png"
        create_actual_vs_predicted_plot(y_val, y_pred, actual_vs_pred_path)
        
        # 2. Residual Distribution
        residual_dist_path = VIZ_DIR / "residual_distribution.png"
        create_residual_distribution_plot(y_val, y_pred, residual_dist_path)
        
        # 3. Residuals vs Predicted
        residuals_vs_pred_path = VIZ_DIR / "residuals_vs_predicted.png"
        create_residuals_vs_predicted_plot(y_pred, y_val, residuals_vs_pred_path)
        
        # 4. Q-Q Plot
        qq_plot_path = VIZ_DIR / "qq_plot.png"
        create_qq_plot(y_val, y_pred, qq_plot_path)
        
        # 5. Feature Importance
        feature_importance_path = VIZ_DIR / "feature_importance.png"
        create_feature_importance_plot(feature_importance, feature_importance_path)
        
        print("\n" + "=" * 80)
        print("VISUALIZATION CREATION COMPLETED")
        print("=" * 80)
        print(f"All visualizations saved to: {VIZ_DIR}")
        print("\nFiles created:")
        print(f"  1. actual_vs_predicted.png - Scatter plot with perfect prediction line")
        print(f"  2. residual_distribution.png - Histogram with normality test")
        print(f"  3. residuals_vs_predicted.png - Heteroscedasticity check")
        print(f"  4. qq_plot.png - Q-Q plot for residual normality")
        print(f"  5. feature_importance.png - Top 20 features")
        
        print(f"\nModel Information:")
        print(f"  Model Type: {metadata['model_type']}")
        print(f"  Training Date: {metadata['training_date']}")
        print(f"  Best Iteration: {metadata.get('best_iteration', 'N/A')}")
        print(f"  Final Features: {len(feature_importance)} features")
        
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()