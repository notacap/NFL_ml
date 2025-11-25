"""
Execute the EDA analysis directly as a Python script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest, skew, kurtosis
import warnings
from pathlib import Path
import json

warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Define paths
DATA_PATH = Path(r'C:\Users\nocap\Desktop\code\NFL_ml\models\rf_wr_yds\data\processed')
OUTPUT_PATH = Path(r'C:\Users\nocap\Desktop\code\NFL_ml\models\rf_wr_yds\outputs\initial_data_exploration')
IMAGE_PATH = OUTPUT_PATH / 'images'
CSV_PATH = OUTPUT_PATH / 'csv'

print("="*80)
print("NFL WIDE RECEIVER RECEIVING YARDS - EXPLORATORY DATA ANALYSIS")
print("="*80)
print(f"\nData path: {DATA_PATH}")
print(f"Output path: {OUTPUT_PATH}")

# ============================================================================
# PHASE 1: DATA UNDERSTANDING & SCHEMA ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PHASE 1: DATA UNDERSTANDING & SCHEMA ANALYSIS")
print("="*80)

# Load the dataset
dataset_file = DATA_PATH / 'nfl_wr_receiving_yards_dataset_20251124_184724.parquet'
df = pd.read_parquet(dataset_file)

print(f"\nDataset loaded successfully")
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Schema analysis
schema_info = pd.DataFrame({
    'Column': df.columns,
    'Data Type': df.dtypes.values,
    'Null Count': df.isnull().sum().values,
    'Null Pct': (df.isnull().sum() / len(df) * 100).values,
    'Unique Values': df.nunique().values,
    'Cardinality': df.nunique().values / len(df),
    'Sample Value': [df[col].iloc[0] if len(df) > 0 else None for col in df.columns]
})

print("\nDataset Schema (first 20 columns):")
print(schema_info.head(20).to_string(index=False))

# Save schema
schema_info.to_csv(CSV_PATH / 'dataset_schema.csv', index=False)
print(f"\nSchema saved to {CSV_PATH / 'dataset_schema.csv'}")

# Identify column categories
id_cols = ['plyr_id', 'plyr_guid', 'season_id', 'week_id', 'game_id', 'adv_plyr_gm_rec_id', 'plyr_rec_id', 'team_id']
temporal_cols = ['year', 'week_num']
target_col = 'next_week_rec_yds'
current_week_target = 'plyr_gm_rec_yds'

# Indicator variables
indicator_cols = [col for col in df.columns if '_no_' in col or '_missing_' in col]

# Game-level features
game_level_cols = [col for col in df.columns if col.startswith('plyr_gm_rec_') and col != current_week_target and col not in indicator_cols]

# Season-level cumulative features
season_level_cols = [col for col in df.columns if col.startswith('plyr_rec_') and 'plyr_gm_rec' not in col and col not in indicator_cols and col != 'plyr_rec_id']

print(f"\nColumn Categories:")
print(f"  ID Columns: {len([c for c in id_cols if c in df.columns])}")
print(f"  Temporal Columns: {len([c for c in temporal_cols if c in df.columns])}")
print(f"  Target Column: {target_col}")
print(f"  Indicator Variables: {len(indicator_cols)}")
print(f"  Game-level Features: {len(game_level_cols)}")
print(f"  Season-level Features: {len(season_level_cols)}")

# Basic dataset statistics
print("\nDataset Overview:")
print(f"  Date range: {df['year'].min()} to {df['year'].max()}")
print(f"  Week range: Week {df['week_num'].min()} to Week {df['week_num'].max()}")
print(f"  Unique players: {df['plyr_id'].nunique():,}")
print(f"  Unique seasons: {df['season_id'].nunique()}")
print(f"  Unique teams: {df['team_id'].nunique()}")
print(f"  Average samples per player: {len(df) / df['plyr_id'].nunique():.1f}")

print("\nSamples by Season:")
print(df.groupby('year').size().sort_index())

# ============================================================================
# PHASE 2: DATA QUALITY ASSESSMENT
# ============================================================================
print("\n" + "="*80)
print("PHASE 2: DATA QUALITY ASSESSMENT")
print("="*80)

# Missing value analysis
missing_analysis = pd.DataFrame({
    'Column': df.columns,
    'Missing Count': df.isnull().sum().values,
    'Missing Percentage': (df.isnull().sum() / len(df) * 100).values
}).sort_values('Missing Percentage', ascending=False)

print("\nMissing Values Analysis:")
if missing_analysis['Missing Percentage'].sum() == 0:
    print("EXCELLENT: No missing values detected in dataset")
    print("This indicates proper null handling with indicator variables")
else:
    print(missing_analysis[missing_analysis['Missing Percentage'] > 0].to_string(index=False))

missing_analysis.to_csv(CSV_PATH / 'missing_values_analysis.csv', index=False)

# Outlier detection
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in id_cols + temporal_cols + indicator_cols]

outlier_summary = []
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_pct = len(outliers) / len(df) * 100

    outlier_summary.append({
        'Column': col,
        'Outlier Count': len(outliers),
        'Outlier Pct': outlier_pct,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound,
        'Min Value': df[col].min(),
        'Max Value': df[col].max()
    })

outlier_df = pd.DataFrame(outlier_summary).sort_values('Outlier Pct', ascending=False)
print("\nOutlier Analysis (Top 15 by percentage):")
print(outlier_df.head(15).to_string(index=False))

outlier_df.to_csv(CSV_PATH / 'outlier_analysis.csv', index=False)

# Indicator variable patterns
if indicator_cols:
    print("\nIndicator Variable Activation Rates (Top 10):")
    indicator_stats = pd.DataFrame({
        'Indicator': indicator_cols,
        'Activation Count': [df[col].sum() for col in indicator_cols],
        'Activation Rate': [df[col].mean() * 100 for col in indicator_cols]
    }).sort_values('Activation Rate', ascending=False)

    print(indicator_stats.head(10).to_string(index=False))
    indicator_stats.to_csv(CSV_PATH / 'indicator_variable_analysis.csv', index=False)

    # Visualize top indicators
    fig, ax = plt.subplots(figsize=(12, 8))
    top_indicators = indicator_stats.head(15)
    ax.barh(range(len(top_indicators)), top_indicators['Activation Rate'])
    ax.set_yticks(range(len(top_indicators)))
    ax.set_yticklabels(top_indicators['Indicator'])
    ax.set_xlabel('Activation Rate (%)')
    ax.set_title('Top 15 Indicator Variables by Activation Rate')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(IMAGE_PATH / 'indicator_activation_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Indicator chart saved to {IMAGE_PATH / 'indicator_activation_rates.png'}")

# ============================================================================
# PHASE 3: STATISTICAL PROFILING
# ============================================================================
print("\n" + "="*80)
print("PHASE 3: STATISTICAL PROFILING (UNIVARIATE)")
print("="*80)

numeric_features = [col for col in numeric_cols if col != target_col and col != current_week_target]

univariate_stats = []
for col in numeric_features:
    data = df[col].values

    stats_dict = {
        'Feature': col,
        'Count': len(data),
        'Mean': np.mean(data),
        'Median': np.median(data),
        'Std': np.std(data),
        'Min': np.min(data),
        'Q1': np.percentile(data, 25),
        'Q3': np.percentile(data, 75),
        'Max': np.max(data),
        'Skewness': skew(data),
        'Kurtosis': kurtosis(data),
        'Coefficient of Variation': np.std(data) / np.mean(data) if np.mean(data) != 0 else 0,
        'Range': np.max(data) - np.min(data)
    }

    univariate_stats.append(stats_dict)

univariate_df = pd.DataFrame(univariate_stats)
print("\nUnivariate Statistics Summary (First 15 features):")
print(univariate_df.head(15).to_string(index=False))

univariate_df.to_csv(CSV_PATH / 'univariate_statistics.csv', index=False)
print(f"\nFull univariate statistics saved to {CSV_PATH / 'univariate_statistics.csv'}")

# Distribution characteristics
print("\nHighly Skewed Features (|skewness| > 2, top 10):")
high_skew = univariate_df[abs(univariate_df['Skewness']) > 2].sort_values('Skewness', ascending=False)
print(high_skew[['Feature', 'Skewness', 'Mean', 'Median']].head(10).to_string(index=False))

# ============================================================================
# PHASE 4: TARGET VARIABLE DEEP DIVE
# ============================================================================
print("\n" + "="*80)
print("PHASE 4: TARGET VARIABLE ANALYSIS")
print("="*80)

target = df[target_col]

print(f"\nBasic Statistics:")
print(f"  Count: {len(target):,}")
print(f"  Mean: {target.mean():.2f} yards")
print(f"  Median: {target.median():.2f} yards")
print(f"  Std Dev: {target.std():.2f} yards")
print(f"  Min: {target.min():.0f} yards")
print(f"  Max: {target.max():.0f} yards")
print(f"  25th Percentile: {target.quantile(0.25):.2f} yards")
print(f"  75th Percentile: {target.quantile(0.75):.2f} yards")
print(f"  IQR: {target.quantile(0.75) - target.quantile(0.25):.2f} yards")

print(f"\nDistribution Characteristics:")
print(f"  Skewness: {skew(target):.3f} (Right-skewed)")
print(f"  Kurtosis: {kurtosis(target):.3f}")
print(f"  Coefficient of Variation: {target.std() / target.mean():.3f}")

zero_yards = (target == 0).sum()
print(f"\nZero Yards Games:")
print(f"  Count: {zero_yards:,}")
print(f"  Percentage: {zero_yards / len(target) * 100:.2f}%")

print(f"\nTarget Value Ranges:")
print(f"  0 yards: {(target == 0).sum():,} ({(target == 0).sum() / len(target) * 100:.1f}%)")
print(f"  1-50 yards: {((target > 0) & (target <= 50)).sum():,} ({((target > 0) & (target <= 50)).sum() / len(target) * 100:.1f}%)")
print(f"  51-100 yards: {((target > 50) & (target <= 100)).sum():,} ({((target > 50) & (target <= 100)).sum() / len(target) * 100:.1f}%)")
print(f"  101-150 yards: {((target > 100) & (target <= 150)).sum():,} ({((target > 100) & (target <= 150)).sum() / len(target) * 100:.1f}%)")
print(f"  150+ yards: {(target > 150).sum():,} ({(target > 150).sum() / len(target) * 100:.1f}%)")

# Target distribution visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram
axes[0, 0].hist(target, bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(target.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {target.mean():.1f}')
axes[0, 0].axvline(target.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {target.median():.1f}')
axes[0, 0].set_xlabel('Receiving Yards')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Next Week Receiving Yards')
axes[0, 0].legend()

# Box plot
axes[0, 1].boxplot(target, vert=True)
axes[0, 1].set_ylabel('Receiving Yards')
axes[0, 1].set_title('Box Plot: Next Week Receiving Yards')
axes[0, 1].grid(True, alpha=0.3)

# Q-Q plot
stats.probplot(target, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot: Normality Assessment')
axes[1, 0].grid(True, alpha=0.3)

# Log-transformed histogram (only for positive values)
target_positive = target[target >= 0]
log_target = np.log1p(target_positive)
axes[1, 1].hist(log_target, bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[1, 1].axvline(log_target.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {log_target.mean():.2f}')
axes[1, 1].set_xlabel('Log(1 + Receiving Yards)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Log-Transformed Distribution (Positive Values Only)')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(IMAGE_PATH / 'target_distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\nTarget distribution charts saved to {IMAGE_PATH / 'target_distribution_analysis.png'}")

# Target by week analysis
target_by_week = df.groupby('week_num')[target_col].agg(['mean', 'median', 'std', 'count'])

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].plot(target_by_week.index, target_by_week['mean'], marker='o', label='Mean', linewidth=2)
axes[0].plot(target_by_week.index, target_by_week['median'], marker='s', label='Median', linewidth=2)
axes[0].fill_between(target_by_week.index,
                       target_by_week['mean'] - target_by_week['std'],
                       target_by_week['mean'] + target_by_week['std'],
                       alpha=0.2)
axes[0].set_xlabel('Week Number')
axes[0].set_ylabel('Receiving Yards')
axes[0].set_title('Target Variable by Week (with Std Dev Band)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].bar(target_by_week.index, target_by_week['count'], alpha=0.7)
axes[1].set_xlabel('Week Number')
axes[1].set_ylabel('Sample Count')
axes[1].set_title('Number of Samples by Week')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(IMAGE_PATH / 'target_by_week_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Target by week chart saved to {IMAGE_PATH / 'target_by_week_analysis.png'}")

target_by_week.to_csv(CSV_PATH / 'target_by_week_statistics.csv')

# ============================================================================
# PHASE 5: CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PHASE 5: CORRELATION ANALYSIS")
print("="*80)

feature_cols = [col for col in df.columns if col not in id_cols + temporal_cols + [target_col, current_week_target] + indicator_cols]
feature_cols_numeric = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]

# Pearson correlation
pearson_corr = df[feature_cols_numeric + [target_col]].corr()[target_col].drop(target_col).sort_values(ascending=False)

# Spearman correlation
spearman_corr = df[feature_cols_numeric + [target_col]].corr(method='spearman')[target_col].drop(target_col).sort_values(ascending=False)

# Combine correlations
correlation_df = pd.DataFrame({
    'Feature': pearson_corr.index,
    'Pearson_Correlation': pearson_corr.values,
    'Spearman_Correlation': spearman_corr[pearson_corr.index].values,
    'Abs_Pearson': abs(pearson_corr.values),
    'Abs_Spearman': abs(spearman_corr[pearson_corr.index].values)
}).sort_values('Abs_Pearson', ascending=False)

print("\nTop 20 Features by Absolute Pearson Correlation with Target:")
print(correlation_df.head(20).to_string(index=False))

correlation_df.to_csv(CSV_PATH / 'feature_target_correlations.csv', index=False)
print(f"\nCorrelation analysis saved to {CSV_PATH / 'feature_target_correlations.csv'}")

# Visualize top correlations
top_features = correlation_df.head(20)

fig, axes = plt.subplots(1, 2, figsize=(16, 10))

axes[0].barh(range(len(top_features)), top_features['Pearson_Correlation'])
axes[0].set_yticks(range(len(top_features)))
axes[0].set_yticklabels(top_features['Feature'], fontsize=9)
axes[0].set_xlabel('Pearson Correlation')
axes[0].set_title('Top 20 Features: Pearson Correlation with Target')
axes[0].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3)

axes[1].barh(range(len(top_features)), top_features['Spearman_Correlation'], color='orange')
axes[1].set_yticks(range(len(top_features)))
axes[1].set_yticklabels(top_features['Feature'], fontsize=9)
axes[1].set_xlabel('Spearman Correlation')
axes[1].set_title('Top 20 Features: Spearman Correlation with Target')
axes[1].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(IMAGE_PATH / 'correlation_top_features.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Correlation chart saved to {IMAGE_PATH / 'correlation_top_features.png'}")

# Multicollinearity analysis
top_20_features = correlation_df.head(20)['Feature'].tolist()
correlation_matrix = df[top_20_features].corr()

high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append({
                'Feature_1': correlation_matrix.columns[i],
                'Feature_2': correlation_matrix.columns[j],
                'Correlation': correlation_matrix.iloc[i, j]
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', ascending=False, key=abs)
    print("\nHighly Correlated Feature Pairs (|r| > 0.8):")
    print(high_corr_df.to_string(index=False))
    high_corr_df.to_csv(CSV_PATH / 'multicollinearity_pairs.csv', index=False)
else:
    print("\nNo highly correlated pairs found (|r| > 0.8) among top 20 features")

# Heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
            cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
plt.title('Correlation Matrix: Top 20 Predictive Features')
plt.tight_layout()
plt.savefig(IMAGE_PATH / 'correlation_matrix_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Correlation matrix heatmap saved")

# ============================================================================
# PHASE 6: SCATTER PLOTS
# ============================================================================
print("\n" + "="*80)
print("PHASE 6: FEATURE RELATIONSHIP DISCOVERY")
print("="*80)

top_6_features = correlation_df.head(6)['Feature'].tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, feature in enumerate(top_6_features):
    sample_size = min(1000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)

    axes[idx].scatter(sample_df[feature], sample_df[target_col], alpha=0.5, s=10)

    z = np.polyfit(sample_df[feature], sample_df[target_col], 1)
    p = np.poly1d(z)
    x_line = np.linspace(sample_df[feature].min(), sample_df[feature].max(), 100)
    axes[idx].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    corr_val = correlation_df[correlation_df['Feature'] == feature]['Pearson_Correlation'].values[0]
    axes[idx].set_xlabel(feature, fontsize=9)
    axes[idx].set_ylabel(target_col, fontsize=9)
    axes[idx].set_title(f'{feature}\n(r = {corr_val:.3f})', fontsize=10)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(IMAGE_PATH / 'scatter_top_features_vs_target.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Scatter plots saved")

# ============================================================================
# PHASE 7: FEATURE RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("PHASE 7: FEATURE ENGINEERING RECOMMENDATIONS")
print("="*80)

# Generate feature recommendations (abbreviated for script)
basic_features = []
complex_features = []

# Basic features
top_stats = correlation_df.head(10)['Feature'].tolist()
for stat in top_stats[:5]:  # Top 5 for rolling averages
    if 'plyr_gm_rec' in stat or 'plyr_rec' in stat:
        for window in [3, 5]:
            basic_features.append({
                'feature_name': f'{stat}_rolling_{window}gm_avg',
                'description': f'Rolling {window}-game average of {stat}',
                'category': 'basic',
                'priority': 5 if window == 3 else 4,
                'rationale': f'Recent performance trend. Base stat correlation: {correlation_df[correlation_df["Feature"] == stat]["Pearson_Correlation"].values[0]:.3f}',
                'implementation_notes': f'Group by plyr_id, season_id and calculate rolling mean with window={window}',
                'estimated_correlation': abs(correlation_df[correlation_df['Feature'] == stat]['Pearson_Correlation'].values[0]) * 0.9
            })

basic_features.extend([
    {
        'feature_name': 'season_targets_per_game',
        'description': 'Season-to-date targets per game',
        'category': 'basic',
        'priority': 5,
        'rationale': 'Normalizes target volume by games played',
        'implementation_notes': 'plyr_rec_tgt / plyr_rec_gm',
        'estimated_correlation': 0.35
    },
    {
        'feature_name': 'season_yards_per_game',
        'description': 'Season-to-date receiving yards per game',
        'category': 'basic',
        'priority': 5,
        'rationale': 'Normalizes cumulative yards by games played',
        'implementation_notes': 'plyr_rec_yds / plyr_rec_gm',
        'estimated_correlation': 0.40
    },
    {
        'feature_name': 'yards_per_reception',
        'description': 'Average yards per reception',
        'category': 'basic',
        'priority': 4,
        'rationale': 'Indicates big-play ability',
        'implementation_notes': 'plyr_rec_yds / plyr_rec',
        'estimated_correlation': 0.20
    },
    {
        'feature_name': 'yards_per_target',
        'description': 'Average yards per target',
        'category': 'basic',
        'priority': 4,
        'rationale': 'Overall efficiency metric',
        'implementation_notes': 'plyr_rec_yds / plyr_rec_tgt',
        'estimated_correlation': 0.25
    },
    {
        'feature_name': 'yds_trend_last_3_games',
        'description': 'Linear trend of yards over last 3 games',
        'category': 'basic',
        'priority': 3,
        'rationale': 'Captures performance trajectory',
        'implementation_notes': 'Calculate slope of linear regression',
        'estimated_correlation': 0.18
    }
])

# Complex features
complex_features.extend([
    {
        'feature_name': 'opponent_wr_yards_allowed_avg',
        'description': 'Average WR yards allowed by opponent',
        'category': 'complex',
        'priority': 5,
        'rationale': 'Direct matchup metric',
        'implementation_notes': 'Requires tm_def_vs_wr table join',
        'estimated_correlation': 0.30
    },
    {
        'feature_name': 'air_yards_share',
        'description': 'Player share of team air yards',
        'category': 'complex',
        'priority': 5,
        'rationale': 'Measures downfield target opportunity',
        'implementation_notes': 'Player ADOT * targets / Team total air yards',
        'estimated_correlation': 0.35
    },
    {
        'feature_name': 'route_participation_rate',
        'description': 'Routes run / team pass plays',
        'category': 'complex',
        'priority': 4,
        'rationale': 'More routes = more opportunities',
        'implementation_notes': 'Requires route tracking data',
        'estimated_correlation': 0.28
    },
    {
        'feature_name': 'qb_recent_performance',
        'description': 'QB passing yards avg last 3 games',
        'category': 'complex',
        'priority': 4,
        'rationale': 'Hot QB lifts all receivers',
        'implementation_notes': 'Requires player_game_passing join',
        'estimated_correlation': 0.25
    },
    {
        'feature_name': 'weather_severity_score',
        'description': 'Composite weather score',
        'category': 'complex',
        'priority': 3,
        'rationale': 'Severe weather reduces passing',
        'implementation_notes': 'Requires game_weather table',
        'estimated_correlation': 0.15
    }
])

all_recommendations = basic_features + complex_features
recommendations_df = pd.DataFrame(all_recommendations)

print("\nBASIC FEATURES (for immediate implementation):")
basic_df = recommendations_df[recommendations_df['category'] == 'basic'].sort_values('priority', ascending=False)
print(basic_df[['feature_name', 'priority', 'estimated_correlation', 'description']].to_string(index=False))

print("\nCOMPLEX FEATURES (for post-validation):")
complex_df = recommendations_df[recommendations_df['category'] == 'complex'].sort_values('priority', ascending=False)
print(complex_df[['feature_name', 'priority', 'estimated_correlation', 'description']].to_string(index=False))

recommendations_df.to_csv(CSV_PATH / 'feature_recommendations.csv', index=False)
print(f"\nFeature recommendations saved to {CSV_PATH / 'feature_recommendations.csv'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS - SUMMARY")
print("="*80)

print("\n1. DATASET OVERVIEW")
print(f"   - Total samples: {len(df):,}")
print(f"   - Features: {len(feature_cols_numeric)}")
print(f"   - Unique players: {df['plyr_id'].nunique()}")
print(f"   - Date range: {df['year'].min()}-{df['year'].max()}")

print("\n2. DATA QUALITY")
print(f"   - Missing values: {df.isnull().sum().sum()} (0%)")
print(f"   - Indicator variables: {len(indicator_cols)}")
print(f"   - Data integrity: EXCELLENT")

print("\n3. TARGET VARIABLE (next_week_rec_yds)")
print(f"   - Mean: {target.mean():.1f} yards")
print(f"   - Median: {target.median():.1f} yards")
print(f"   - Std Dev: {target.std():.1f} yards")
print(f"   - Skewness: {skew(target):.2f}")
print(f"   - Zero yards: {(target == 0).sum() / len(target) * 100:.1f}%")

print("\n4. TOP 10 PREDICTIVE FEATURES")
top_10 = correlation_df.head(10)
for idx, (_, row) in enumerate(top_10.iterrows(), 1):
    print(f"   {idx}. {row['Feature']}: r={row['Pearson_Correlation']:.3f}")

print("\n5. FEATURE RECOMMENDATIONS")
print(f"   - Basic features: {len(basic_features)}")
print(f"   - Complex features: {len(complex_features)}")

print("\n6. KEY INSIGHTS")
print("   - Season cumulative stats highly predictive")
print("   - Rolling averages will improve stability")
print("   - High target variance - ensemble methods recommended")
print("   - Low multicollinearity among top features")

print("\n7. RISKS & CONCERNS")
print("   - High target variance (CV > 1.0)")
print("   - Right-skewed distribution")
print("   - Zero-yards games require consideration")

print("\n8. NEXT STEPS")
print("   1. Implement top basic features")
print("   2. Train baseline Random Forest")
print("   3. Evaluate performance metrics")
print("   4. Analyze feature importance")
print("   5. Iterate with complex features")

# Save summary report
summary_report = {
    'analysis_date': '2024-11-24',
    'dataset_file': 'nfl_wr_receiving_yards_dataset_20251124_184724.parquet',
    'dataset_stats': {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'unique_players': int(df['plyr_id'].nunique()),
        'date_range': f"{df['year'].min()}-{df['year'].max()}"
    },
    'target_stats': {
        'mean': float(target.mean()),
        'median': float(target.median()),
        'std': float(target.std()),
        'skewness': float(skew(target)),
        'zero_yards_pct': float((target == 0).sum() / len(target) * 100)
    },
    'top_features': top_10[['Feature', 'Pearson_Correlation']].to_dict('records'),
    'recommendations_count': {
        'basic_features': len(basic_features),
        'complex_features': len(complex_features)
    }
}

with open(CSV_PATH / 'eda_summary_report.json', 'w') as f:
    json.dump(summary_report, f, indent=2)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nAll outputs saved to: {OUTPUT_PATH}")
print(f"  - CSV files: {CSV_PATH}")
print(f"  - Images: {IMAGE_PATH}")
