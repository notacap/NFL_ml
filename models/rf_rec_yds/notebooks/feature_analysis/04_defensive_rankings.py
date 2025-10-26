"""
Defensive Rankings Analysis
============================
Analyzes overall defensive metrics to identify which statistics best predict
receiver suppression and develops a comprehensive defensive ranking system.

Objectives:
1. Correlation analysis between defensive metrics and yards allowed
2. Compare volume vs efficiency metrics
3. Test rolling vs season-long statistics
4. Identify most stable and predictive defensive metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
ROOT_DIR = Path(r"C:\Users\nocap\Desktop\code\NFL_ml")
DATA_DIR = ROOT_DIR / "parquet_files" / "clean"
OUTPUT_DIR = ROOT_DIR / "outputs"
VIZ_DIR = OUTPUT_DIR / "visualizations"
PRED_DIR = OUTPUT_DIR / "predictions"

# Create output directories
VIZ_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("DEFENSIVE RANKINGS ANALYSIS")
print("="*80)

# ============================================================================
# PART 1: Load and Prepare Data
# ============================================================================

print("\n1. Loading defensive data...")

# Load defensive datasets
tm_def = pd.read_parquet(DATA_DIR / "tm_szn" / "tm_def")
tm_def_pass = pd.read_parquet(DATA_DIR / "tm_szn" / "tm_def_pass")
tm_def_vs_wr = pd.read_parquet(DATA_DIR / "tm_szn" / "tm_def_vs_wr")
tm_def_vs_te = pd.read_parquet(DATA_DIR / "tm_szn" / "tm_def_vs_te")
tm_def_vs_rb = pd.read_parquet(DATA_DIR / "tm_szn" / "tm_def_vs_rb")

print(f"   - Overall defense: {tm_def.shape}")
print(f"   - Pass defense: {tm_def_pass.shape}")
print(f"   - WR defense: {tm_def_vs_wr.shape}")
print(f"   - TE defense: {tm_def_vs_te.shape}")
print(f"   - RB defense: {tm_def_vs_rb.shape}")

# Merge defensive data
print("\n2. Merging defensive datasets...")
df_def = tm_def.merge(
    tm_def_pass,
    on=['team_id', 'season_id', 'week_id', 'season', 'week'],
    suffixes=('', '_pass')
)

df_def = df_def.merge(
    tm_def_vs_wr,
    on=['team_id', 'season_id', 'week_id', 'season', 'week'],
    how='left'
)

df_def = df_def.merge(
    tm_def_vs_te,
    on=['team_id', 'season_id', 'week_id', 'season', 'week'],
    how='left'
)

df_def = df_def.merge(
    tm_def_vs_rb,
    on=['team_id', 'season_id', 'week_id', 'season', 'week'],
    how='left'
)

print(f"   - Merged defense data: {df_def.shape}")

# Calculate per-game averages and derive metrics
print("\n3. Calculating defensive metrics...")

# Games played (week_id represents cumulative through that week)
df_def['games_played'] = df_def['week_id']

# Per-game metrics
df_def['pts_per_game'] = df_def['tm_def_pts_allwd'] / df_def['games_played']
df_def['yds_per_game'] = df_def['tm_def_yds'] / df_def['games_played']
df_def['pass_yds_per_game'] = df_def['tm_def_pass_yds'] / df_def['games_played']
df_def['wr_yds_per_game'] = df_def['tm_def_wr_yds'] / df_def['games_played']
df_def['te_yds_per_game'] = df_def['tm_def_te_yds'] / df_def['games_played']
df_def['rb_rec_yds_per_game'] = df_def['tm_def_rb_rec_yds'] / df_def['games_played']

# Efficiency metrics
df_def['sacks_per_game'] = df_def['tm_def_sk'] / df_def['games_played']
df_def['int_per_game'] = df_def['tm_def_int'] / df_def['games_played']
df_def['takeaways_per_game'] = df_def['tm_def_tkawy'] / df_def['games_played']

# Total receiving yards allowed (WR + TE + RB)
df_def['total_rec_yds_per_game'] = (
    df_def['wr_yds_per_game'].fillna(0) +
    df_def['te_yds_per_game'].fillna(0) +
    df_def['rb_rec_yds_per_game'].fillna(0)
)

print(f"   - Added {len(['pts_per_game', 'yds_per_game', 'pass_yds_per_game', 'wr_yds_per_game', 'te_yds_per_game', 'rb_rec_yds_per_game', 'sacks_per_game', 'int_per_game', 'takeaways_per_game', 'total_rec_yds_per_game'])} calculated metrics")

# ============================================================================
# PART 2: Defensive Metrics Correlation Analysis
# ============================================================================

print("\n" + "="*80)
print("CORRELATION ANALYSIS: Defensive Metrics vs Yards Allowed")
print("="*80)

# Define target variables (what we want to predict/minimize)
targets = {
    'pass_yds_per_game': 'Pass Yards Allowed',
    'wr_yds_per_game': 'WR Yards Allowed',
    'te_yds_per_game': 'TE Yards Allowed',
    'rb_rec_yds_per_game': 'RB Rec Yards Allowed',
    'total_rec_yds_per_game': 'Total Rec Yards Allowed'
}

# Define predictor variables
predictors = {
    # Volume metrics
    'tm_def_pts_allwd': 'Total Points Allowed',
    'tm_def_yds': 'Total Yards Allowed',
    'tm_def_pass_cmp': 'Completions Allowed',
    'tm_def_pass_att': 'Pass Attempts Faced',

    # Efficiency metrics
    'tm_def_ypp': 'Yards per Play',
    'tm_def_pass_cmp_pct': 'Completion % Allowed',
    'tm_def_pass_yds_att': 'Yards per Attempt',
    'tm_def_pass_ypc': 'Yards per Completion',
    'tm_def_pass_rtg': 'Passer Rating Allowed',

    # Pressure metrics
    'tm_def_sk': 'Sacks',
    'tm_def_sk_pct': 'Sack %',
    'tm_def_bltz_pct': 'Blitz %',
    'tm_def_prss_pct': 'Pressure %',
    'tm_def_hrry_pct': 'Hurry %',
    'tm_def_qbkd_pct': 'QB Knockdown %',

    # Turnover metrics
    'tm_def_int': 'Interceptions',
    'tm_def_int_pct': 'INT %',
    'tm_def_tkawy': 'Takeaways',
    'tm_def_to_pct': 'Turnover %',

    # Per-game metrics
    'pts_per_game': 'Points per Game',
    'sacks_per_game': 'Sacks per Game',
    'int_per_game': 'INTs per Game',
    'takeaways_per_game': 'Takeaways per Game'
}

# Calculate correlations
print("\nCalculating correlations between predictors and targets...")
correlation_results = []

for target_col, target_name in targets.items():
    for pred_col, pred_name in predictors.items():
        # Filter to non-null values
        valid_data = df_def[[target_col, pred_col]].dropna()

        if len(valid_data) > 30:  # Require at least 30 observations
            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(valid_data[target_col], valid_data[pred_col])

            # Spearman correlation (rank-based, robust to outliers)
            spearman_r, spearman_p = stats.spearmanr(valid_data[target_col], valid_data[pred_col])

            correlation_results.append({
                'Target': target_name,
                'Predictor': pred_name,
                'Predictor_Col': pred_col,
                'Pearson_r': pearson_r,
                'Pearson_p': pearson_p,
                'Spearman_r': spearman_r,
                'Spearman_p': spearman_p,
                'N': len(valid_data),
                'Abs_Pearson': abs(pearson_r),
                'Abs_Spearman': abs(spearman_r)
            })

corr_df = pd.DataFrame(correlation_results)

# Display top correlations for each target
print("\nTop 5 Predictors for Each Target Variable:")
print("-" * 80)

for target_name in targets.values():
    target_corr = corr_df[corr_df['Target'] == target_name].sort_values('Abs_Pearson', ascending=False)
    print(f"\n{target_name}:")
    print(target_corr[['Predictor', 'Pearson_r', 'Pearson_p', 'Spearman_r']].head(5).to_string(index=False))

# Save correlation results
corr_df.to_csv(PRED_DIR / 'defensive_metric_correlations.csv', index=False)
print(f"\nSaved correlation results to: {PRED_DIR / 'defensive_metric_correlations.csv'}")

# ============================================================================
# PART 3: Visualization - Correlation Heatmap
# ============================================================================

print("\n" + "="*80)
print("CREATING CORRELATION HEATMAP")
print("="*80)

# Create correlation matrix for key metrics
key_predictors = [
    'tm_def_pass_yds_att', 'tm_def_pass_rtg', 'tm_def_pass_cmp_pct',
    'tm_def_sk_pct', 'tm_def_prss_pct', 'tm_def_int_pct',
    'pts_per_game', 'sacks_per_game', 'tm_def_ypp'
]

key_targets = ['pass_yds_per_game', 'wr_yds_per_game', 'te_yds_per_game']

# Build correlation matrix
corr_matrix = pd.DataFrame(index=key_predictors, columns=key_targets)

for target in key_targets:
    for predictor in key_predictors:
        valid_data = df_def[[target, predictor]].dropna()
        if len(valid_data) > 30:
            corr_matrix.loc[predictor, target] = stats.pearsonr(valid_data[target], valid_data[predictor])[0]

corr_matrix = corr_matrix.astype(float)

# Create heatmap
fig, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.3f',
    cmap='RdYlGn_r',
    center=0,
    vmin=-1,
    vmax=1,
    cbar_kws={'label': 'Correlation Coefficient'},
    ax=ax,
    linewidths=0.5
)

ax.set_title('Defensive Metrics Correlation with Receiving Yards Allowed\n(Higher = More Yards Allowed)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Target Variables', fontsize=12, fontweight='bold')
ax.set_ylabel('Defensive Metrics', fontsize=12, fontweight='bold')

# Rename labels for better readability
target_labels = {
    'pass_yds_per_game': 'Pass Yds/G',
    'wr_yds_per_game': 'WR Yds/G',
    'te_yds_per_game': 'TE Yds/G'
}
ax.set_xticklabels([target_labels.get(col, col) for col in key_targets], rotation=45, ha='right')

predictor_labels = {
    'tm_def_pass_yds_att': 'Yds per Attempt',
    'tm_def_pass_rtg': 'Passer Rating',
    'tm_def_pass_cmp_pct': 'Completion %',
    'tm_def_sk_pct': 'Sack %',
    'tm_def_prss_pct': 'Pressure %',
    'tm_def_int_pct': 'INT %',
    'pts_per_game': 'Points/Game',
    'sacks_per_game': 'Sacks/Game',
    'tm_def_ypp': 'Yards per Play'
}
ax.set_yticklabels([predictor_labels.get(idx, idx) for idx in key_predictors], rotation=0)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'defensive_metrics_correlation.png', dpi=300, bbox_inches='tight')
print(f"Saved correlation heatmap to: {VIZ_DIR / 'defensive_metrics_correlation.png'}")
plt.close()

# ============================================================================
# PART 4: Defensive Rankings Development
# ============================================================================

print("\n" + "="*80)
print("DEVELOPING DEFENSIVE RANKINGS")
print("="*80)

# Focus on season-end data (week 18) for final rankings
season_end = df_def[df_def['week_id'] == 18].copy()

print(f"\nSeason-end data: {season_end.shape[0]} team-seasons")

# Define ranking metrics (lower is better for defense)
ranking_metrics = {
    # Volume metrics
    'pts_per_game': 'Points Allowed per Game',
    'yds_per_game': 'Yards Allowed per Game',
    'pass_yds_per_game': 'Pass Yards per Game',
    'wr_yds_per_game': 'WR Yards per Game',
    'te_yds_per_game': 'TE Yards per Game',

    # Efficiency metrics
    'tm_def_ypp': 'Yards per Play',
    'tm_def_pass_yds_att': 'Pass Yds per Attempt',
    'tm_def_pass_cmp_pct': 'Completion %',
    'tm_def_pass_rtg': 'Passer Rating',

    # Pressure (higher is better)
    'tm_def_sk_pct': 'Sack %',
    'tm_def_prss_pct': 'Pressure %',

    # Turnovers (higher is better)
    'tm_def_int_pct': 'INT %',
    'tm_def_to_pct': 'Turnover %'
}

# Calculate rankings for each metric
rankings_data = []

for season in season_end['season'].unique():
    season_data = season_end[season_end['season'] == season].copy()

    for metric, metric_name in ranking_metrics.items():
        if metric in season_data.columns:
            # Determine ranking direction
            if metric in ['tm_def_sk_pct', 'tm_def_prss_pct', 'tm_def_int_pct', 'tm_def_to_pct']:
                # Higher is better (better defense)
                season_data[f'{metric}_rank'] = season_data[metric].rank(ascending=False, method='min')
            else:
                # Lower is better (fewer yards/points allowed)
                season_data[f'{metric}_rank'] = season_data[metric].rank(ascending=True, method='min')

            # Calculate percentile (0-100, where 100 is best)
            if metric in ['tm_def_sk_pct', 'tm_def_prss_pct', 'tm_def_int_pct', 'tm_def_to_pct']:
                season_data[f'{metric}_pctile'] = season_data[metric].rank(pct=True) * 100
            else:
                season_data[f'{metric}_pctile'] = (1 - season_data[metric].rank(pct=True)) * 100

    rankings_data.append(season_data)

rankings_df = pd.concat(rankings_data, ignore_index=True)

# Create composite defensive scores
print("\nCalculating composite defensive scores...")

# Weight different metric categories
rankings_df['pass_defense_score'] = (
    rankings_df['pass_yds_per_game_pctile'] * 0.4 +
    rankings_df['tm_def_pass_yds_att_pctile'] * 0.3 +
    rankings_df['tm_def_pass_rtg_pctile'] * 0.3
)

rankings_df['wr_defense_score'] = rankings_df['wr_yds_per_game_pctile']
rankings_df['te_defense_score'] = rankings_df['te_yds_per_game_pctile']

rankings_df['pressure_score'] = (
    rankings_df['tm_def_sk_pct_pctile'] * 0.4 +
    rankings_df['tm_def_prss_pct_pctile'] * 0.6
)

rankings_df['overall_defense_score'] = (
    rankings_df['pass_defense_score'] * 0.5 +
    rankings_df['pressure_score'] * 0.2 +
    rankings_df['tm_def_to_pct_pctile'] * 0.15 +
    rankings_df['pts_per_game_pctile'] * 0.15
)

# Display top defenses by season
print("\nTop 5 Pass Defenses by Season:")
print("-" * 80)

for season in sorted(rankings_df['season'].unique()):
    season_ranks = rankings_df[rankings_df['season'] == season].sort_values('pass_defense_score', ascending=False)
    print(f"\n{season} Season:")
    print(season_ranks[['team_id', 'pass_defense_score', 'pass_yds_per_game', 'tm_def_pass_yds_att', 'tm_def_pass_rtg']].head(5).to_string(index=False))

# Save rankings
rankings_output = rankings_df[[
    'team_id', 'season', 'week',
    'pts_per_game', 'pts_per_game_rank', 'pts_per_game_pctile',
    'pass_yds_per_game', 'pass_yds_per_game_rank', 'pass_yds_per_game_pctile',
    'wr_yds_per_game', 'wr_yds_per_game_rank', 'wr_yds_per_game_pctile',
    'te_yds_per_game', 'te_yds_per_game_rank', 'te_yds_per_game_pctile',
    'tm_def_pass_yds_att', 'tm_def_pass_yds_att_rank', 'tm_def_pass_yds_att_pctile',
    'tm_def_pass_rtg', 'tm_def_pass_rtg_rank', 'tm_def_pass_rtg_pctile',
    'tm_def_sk_pct', 'tm_def_sk_pct_rank', 'tm_def_sk_pct_pctile',
    'tm_def_prss_pct', 'tm_def_prss_pct_rank', 'tm_def_prss_pct_pctile',
    'pass_defense_score', 'pressure_score', 'overall_defense_score'
]].copy()

rankings_output.to_csv(PRED_DIR / 'defensive_rankings_all.csv', index=False)
print(f"\nSaved defensive rankings to: {PRED_DIR / 'defensive_rankings_all.csv'}")

# ============================================================================
# PART 5: Rolling Window Analysis
# ============================================================================

print("\n" + "="*80)
print("ROLLING WINDOW ANALYSIS")
print("="*80)

# Calculate rolling averages for key metrics
print("\nCalculating rolling 3-game and 5-game averages...")

rolling_data = []

for team in df_def['team_id'].unique():
    for season in df_def['season'].unique():
        team_season = df_def[(df_def['team_id'] == team) & (df_def['season'] == season)].sort_values('week_id')

        if len(team_season) > 0:
            for idx, row in team_season.iterrows():
                week = row['week_id']

                # Get data from previous weeks
                prev_weeks = team_season[team_season['week_id'] < week]

                if len(prev_weeks) >= 3:
                    last_3 = prev_weeks.tail(3)
                    last_5 = prev_weeks.tail(5)

                    rolling_data.append({
                        'team_id': team,
                        'season': season,
                        'week': week,
                        'pass_yds_season_avg': row['pass_yds_per_game'],
                        'pass_yds_L3_avg': last_3['pass_yds_per_game'].mean(),
                        'pass_yds_L5_avg': last_5['pass_yds_per_game'].mean() if len(last_5) >= 5 else np.nan,
                        'wr_yds_season_avg': row['wr_yds_per_game'],
                        'wr_yds_L3_avg': last_3['wr_yds_per_game'].mean(),
                        'wr_yds_L5_avg': last_5['wr_yds_per_game'].mean() if len(last_5) >= 5 else np.nan,
                        'pass_rtg_season_avg': row['tm_def_pass_rtg'],
                        'pass_rtg_L3_avg': last_3['tm_def_pass_rtg'].mean(),
                        'pass_rtg_L5_avg': last_5['tm_def_pass_rtg'].mean() if len(last_5) >= 5 else np.nan,
                    })

rolling_df = pd.DataFrame(rolling_data)
rolling_df.to_csv(PRED_DIR / 'defensive_rolling_averages.csv', index=False)
print(f"Saved rolling averages to: {PRED_DIR / 'defensive_rolling_averages.csv'}")

# Analyze correlation between rolling and season averages
print("\nCorrelation between rolling vs season averages:")
print("-" * 80)

for metric in ['pass_yds', 'wr_yds', 'pass_rtg']:
    valid_L3 = rolling_df[[f'{metric}_season_avg', f'{metric}_L3_avg']].dropna()
    valid_L5 = rolling_df[[f'{metric}_season_avg', f'{metric}_L5_avg']].dropna()

    if len(valid_L3) > 30:
        r_L3 = stats.pearsonr(valid_L3[f'{metric}_season_avg'], valid_L3[f'{metric}_L3_avg'])[0]
        print(f"{metric} - Last 3 games vs Season: r = {r_L3:.3f}")

    if len(valid_L5) > 30:
        r_L5 = stats.pearsonr(valid_L5[f'{metric}_season_avg'], valid_L5[f'{metric}_L5_avg'])[0]
        print(f"{metric} - Last 5 games vs Season: r = {r_L5:.3f}")

# ============================================================================
# PART 6: Stability Analysis
# ============================================================================

print("\n" + "="*80)
print("DEFENSIVE METRIC STABILITY ANALYSIS")
print("="*80)

# Calculate week-to-week variance for each team-season
stability_data = []

for team in df_def['team_id'].unique():
    for season in df_def['season'].unique():
        team_season = df_def[(df_def['team_id'] == team) & (df_def['season'] == season)].sort_values('week_id')

        if len(team_season) >= 10:  # Require at least 10 weeks
            stability_data.append({
                'team_id': team,
                'season': season,
                'pass_yds_std': team_season['pass_yds_per_game'].std(),
                'pass_yds_cv': team_season['pass_yds_per_game'].std() / team_season['pass_yds_per_game'].mean(),
                'wr_yds_std': team_season['wr_yds_per_game'].std(),
                'wr_yds_cv': team_season['wr_yds_per_game'].std() / team_season['wr_yds_per_game'].mean() if team_season['wr_yds_per_game'].mean() > 0 else np.nan,
                'pass_rtg_std': team_season['tm_def_pass_rtg'].std(),
                'pass_rtg_cv': team_season['tm_def_pass_rtg'].std() / team_season['tm_def_pass_rtg'].mean()
            })

stability_df = pd.DataFrame(stability_data)

print("\nAverage Coefficient of Variation (CV) by Metric:")
print("-" * 80)
if len(stability_df) > 0:
    print(f"Pass Yards per Game: {stability_df['pass_yds_cv'].mean():.3f}")
    print(f"WR Yards per Game: {stability_df['wr_yds_cv'].mean():.3f}")
    print(f"Passer Rating: {stability_df['pass_rtg_cv'].mean():.3f}")
else:
    print("Insufficient data for stability analysis (need at least 10 weeks per team-season)")

stability_df.to_csv(PRED_DIR / 'defensive_stability_metrics.csv', index=False)
print(f"\nSaved stability metrics to: {PRED_DIR / 'defensive_stability_metrics.csv'}")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80)

# Top 5 most predictive metrics overall
top_metrics = corr_df.groupby('Predictor')['Abs_Pearson'].mean().sort_values(ascending=False).head(10)

print("\nTop 10 Most Predictive Defensive Metrics (Average Correlation):")
print("-" * 80)
for i, (metric, corr) in enumerate(top_metrics.items(), 1):
    print(f"{i:2d}. {metric:40s} - Avg |r| = {corr:.3f}")

# Volume vs Efficiency comparison
volume_metrics = ['tm_def_pts_allwd', 'tm_def_yds', 'tm_def_pass_cmp', 'tm_def_pass_att']
efficiency_metrics = ['tm_def_ypp', 'tm_def_pass_cmp_pct', 'tm_def_pass_yds_att', 'tm_def_pass_rtg']

volume_corr = corr_df[corr_df['Predictor_Col'].isin(volume_metrics)]['Abs_Pearson'].mean()
efficiency_corr = corr_df[corr_df['Predictor_Col'].isin(efficiency_metrics)]['Abs_Pearson'].mean()

print(f"\nVolume Metrics - Average |r| = {volume_corr:.3f}")
print(f"Efficiency Metrics - Average |r| = {efficiency_corr:.3f}")

if efficiency_corr > volume_corr:
    print("\nConclusion: Efficiency metrics are MORE predictive than volume metrics")
else:
    print("\nConclusion: Volume metrics are MORE predictive than efficiency metrics")

print("\n" + "="*80)
print("DEFENSIVE RANKINGS ANALYSIS COMPLETE")
print("="*80)
print(f"\nOutputs saved to:")
print(f"  - {PRED_DIR / 'defensive_metric_correlations.csv'}")
print(f"  - {PRED_DIR / 'defensive_rankings_all.csv'}")
print(f"  - {PRED_DIR / 'defensive_rolling_averages.csv'}")
print(f"  - {PRED_DIR / 'defensive_stability_metrics.csv'}")
print(f"  - {VIZ_DIR / 'defensive_metrics_correlation.png'}")
print("="*80)
