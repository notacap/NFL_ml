"""
Team Offensive Strength Analysis for NFL Receiver Production
=============================================================

Objective: Quantify team offensive capabilities that boost receiver production
and identify which team metrics best predict individual receiving yards.

Analysis includes:
1. Team passing volume vs efficiency metrics
2. Rolling team performance windows
3. Pass-funnel team identification
4. Correlation with aggregate receiver yards
5. QB performance impact
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
BASE_PATH = Path(r'C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\clean')
OUTPUT_VIZ = Path(r'C:\Users\nocap\Desktop\code\NFL_ml\outputs\visualizations')
OUTPUT_DATA = Path(r'C:\Users\nocap\Desktop\code\NFL_ml\outputs\predictions')

print("=" * 80)
print("TEAM OFFENSIVE METRICS ANALYSIS")
print("=" * 80)

# ============================================================================
# SECTION 1: Load and Prepare Data
# ============================================================================
print("\n[1/8] Loading data...")

def load_parquet_dataset(path, seasons=[2022, 2023, 2024]):
    """Load partitioned parquet dataset"""
    dfs = []
    for season in seasons:
        season_path = path / f'season={season}'
        if season_path.exists():
            try:
                df = pd.read_parquet(season_path)
                dfs.append(df)
            except:
                pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# Load team game stats
tm_gm_stats = load_parquet_dataset(BASE_PATH / 'tm_gm' / 'tm_gm_stats')
print(f"  - Team game stats: {tm_gm_stats.shape}")

# Load player receiving game stats
plyr_gm_rec = pd.read_parquet(BASE_PATH / 'plyr_gm' / 'plyr_gm_rec')
print(f"  - Player game receiving: {plyr_gm_rec.shape}")

# Load game info (for home/away, scores)
game_info = load_parquet_dataset(BASE_PATH / 'gm_info' / 'nfl_game')
print(f"  - Game info: {game_info.shape}")

# Load team info
team_info = pd.read_parquet(BASE_PATH / 'nfl_team.parquet')
print(f"  - Team info: {team_info.shape}")

# ============================================================================
# SECTION 2: Calculate Team Offensive Metrics
# ============================================================================
print("\n[2/8] Calculating team offensive metrics...")

# Sort by team and game date
tm_gm_stats = tm_gm_stats.sort_values(['team_id', 'season_id', 'week_id'])

# Calculate key metrics
tm_gm_stats['pass_attempts_per_game'] = tm_gm_stats['tm_gm_pass_att']
tm_gm_stats['pass_yards_per_game'] = tm_gm_stats['tm_gm_pass_yds']
tm_gm_stats['completion_pct'] = (tm_gm_stats['tm_gm_pass_cmp'] /
                                   tm_gm_stats['tm_gm_pass_att'].replace(0, np.nan))
tm_gm_stats['yards_per_attempt'] = (tm_gm_stats['tm_gm_pass_yds'] /
                                     tm_gm_stats['tm_gm_pass_att'].replace(0, np.nan))
tm_gm_stats['td_rate'] = (tm_gm_stats['tm_gm_pass_td'] /
                           tm_gm_stats['tm_gm_pass_att'].replace(0, np.nan))
tm_gm_stats['int_rate'] = (tm_gm_stats['tm_gm_pass_int'] /
                            tm_gm_stats['tm_gm_pass_att'].replace(0, np.nan))
tm_gm_stats['pass_run_ratio'] = (tm_gm_stats['tm_gm_pass_att'] /
                                  tm_gm_stats['tm_gm_rush_ply'].replace(0, np.nan))
tm_gm_stats['third_down_pct'] = (tm_gm_stats['tm_gm_third_dwn_conv'] /
                                  tm_gm_stats['tm_gm_third_dwn_att'].replace(0, np.nan))
tm_gm_stats['total_plays'] = tm_gm_stats['tm_gm_pass_att'] + tm_gm_stats['tm_gm_rush_ply']
tm_gm_stats['pass_play_pct'] = tm_gm_stats['tm_gm_pass_att'] / tm_gm_stats['total_plays']

# Get points scored (from game_info)
if 'home_score' in game_info.columns and 'away_score' in game_info.columns:
    # Create two records per game: one for home team, one for away team
    home_scores = game_info[['game_id', 'home_tm_id', 'home_score']].rename(
        columns={'home_tm_id': 'team_id', 'home_score': 'points_scored'}
    )
    away_scores = game_info[['game_id', 'away_tm_id', 'away_score']].rename(
        columns={'away_tm_id': 'team_id', 'away_score': 'points_scored'}
    )
    scores = pd.concat([home_scores, away_scores])
    tm_gm_stats = tm_gm_stats.merge(scores, on=['game_id', 'team_id'], how='left')

# Convert time of possession to seconds
if 'tm_gm_top' in tm_gm_stats.columns:
    tm_gm_stats['top_seconds'] = tm_gm_stats['tm_gm_top'].dt.total_seconds()
    tm_gm_stats['top_percentage'] = tm_gm_stats['top_seconds'] / 3600  # % of 60 minutes

print(f"  - Calculated {len([c for c in tm_gm_stats.columns if c not in tm_gm_stats.select_dtypes(include='object').columns])} metrics")

# ============================================================================
# SECTION 3: Calculate Rolling Averages (3-game and 5-game windows)
# ============================================================================
print("\n[3/8] Calculating rolling averages...")

rolling_cols = [
    'pass_attempts_per_game', 'pass_yards_per_game', 'completion_pct',
    'yards_per_attempt', 'td_rate', 'pass_run_ratio', 'third_down_pct',
    'pass_play_pct', 'top_percentage'
]

if 'points_scored' in tm_gm_stats.columns:
    rolling_cols.append('points_scored')

# Calculate rolling averages for each team
for col in rolling_cols:
    # 3-game rolling average (excluding current game)
    tm_gm_stats[f'{col}_roll3'] = (
        tm_gm_stats.groupby('team_id')[col]
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
    )
    # 5-game rolling average (excluding current game)
    tm_gm_stats[f'{col}_roll5'] = (
        tm_gm_stats.groupby('team_id')[col]
        .transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
    )
    # Season-to-date average (excluding current game)
    tm_gm_stats[f'{col}_szn_avg'] = (
        tm_gm_stats.groupby(['team_id', 'season_id'])[col]
        .transform(lambda x: x.expanding().mean().shift(1))
    )

print(f"  - Created rolling metrics for {len(rolling_cols)} features")

# ============================================================================
# SECTION 4: Aggregate Receiver Production by Team-Game
# ============================================================================
print("\n[4/8] Aggregating receiver production...")

# Aggregate all receiver yards for each team-game
rec_production = plyr_gm_rec.groupby(['team_id', 'game_id', 'season_id', 'week_id']).agg({
    'plyr_gm_rec_yds': ['sum', 'mean', 'max', 'count'],
    'plyr_gm_rec_tgt': 'sum',
    'plyr_gm_rec': 'sum'
}).reset_index()

rec_production.columns = [
    'team_id', 'game_id', 'season_id', 'week_id',
    'total_rec_yds', 'avg_rec_yds_per_player', 'max_rec_yds', 'num_receivers',
    'total_targets', 'total_receptions'
]

print(f"  - Aggregated receiving yards for {rec_production.shape[0]} team-games")

# ============================================================================
# SECTION 5: Merge Team Metrics with Receiver Production
# ============================================================================
print("\n[5/8] Merging datasets...")

# Merge team stats with receiver production
analysis_df = tm_gm_stats.merge(
    rec_production,
    on=['team_id', 'game_id', 'season_id', 'week_id'],
    how='inner'
)

print(f"  - Final analysis dataset: {analysis_df.shape}")
print(f"  - Date range: {analysis_df['season_id'].min()}-{analysis_df['season_id'].max()}")

# ============================================================================
# SECTION 6: Correlation Analysis
# ============================================================================
print("\n[6/8] Computing correlations with receiver production...")

# Define metric categories
volume_metrics = [
    'pass_attempts_per_game', 'pass_yards_per_game', 'total_plays', 'pass_play_pct'
]
efficiency_metrics = [
    'completion_pct', 'yards_per_attempt', 'td_rate', 'int_rate', 'third_down_pct'
]
rolling_metrics_3game = [f'{col}_roll3' for col in rolling_cols]
rolling_metrics_5game = [f'{col}_roll5' for col in rolling_cols]
rolling_metrics_szn = [f'{col}_szn_avg' for col in rolling_cols]

# Calculate correlations
def calculate_correlations(df, metrics, target='total_rec_yds'):
    """Calculate Pearson and Spearman correlations"""
    results = []
    for metric in metrics:
        if metric in df.columns:
            # Remove NaN values
            valid_data = df[[metric, target]].dropna()
            if len(valid_data) > 30:  # Minimum sample size
                pearson_corr, pearson_p = pearsonr(valid_data[metric], valid_data[target])
                spearman_corr, spearman_p = spearmanr(valid_data[metric], valid_data[target])

                results.append({
                    'metric': metric,
                    'pearson_corr': pearson_corr,
                    'pearson_pval': pearson_p,
                    'spearman_corr': spearman_corr,
                    'spearman_pval': spearman_p,
                    'n_samples': len(valid_data),
                    'abs_pearson': abs(pearson_corr)
                })

    return pd.DataFrame(results).sort_values('abs_pearson', ascending=False)

# Calculate correlations for each category
print("\n  Volume Metrics Correlations:")
volume_corr = calculate_correlations(analysis_df, volume_metrics)
print(volume_corr.head(10).to_string(index=False))

print("\n  Efficiency Metrics Correlations:")
efficiency_corr = calculate_correlations(analysis_df, efficiency_metrics)
print(efficiency_corr.head(10).to_string(index=False))

print("\n  3-Game Rolling Averages Correlations:")
roll3_corr = calculate_correlations(analysis_df, rolling_metrics_3game)
print(roll3_corr.head(5).to_string(index=False))

print("\n  5-Game Rolling Averages Correlations:")
roll5_corr = calculate_correlations(analysis_df, rolling_metrics_5game)
print(roll5_corr.head(5).to_string(index=False))

# Combine all correlations
all_corr = pd.concat([volume_corr, efficiency_corr, roll3_corr, roll5_corr],
                     ignore_index=True).sort_values('abs_pearson', ascending=False)

# ============================================================================
# SECTION 7: Identify Pass-Funnel Teams
# ============================================================================
print("\n[7/8] Identifying pass-funnel teams...")

# Calculate season averages for each team
team_season_stats = analysis_df.groupby(['team_id', 'season_id']).agg({
    'pass_attempts_per_game': 'mean',
    'pass_play_pct': 'mean',
    'pass_run_ratio': 'mean',
    'total_rec_yds': 'mean',
    'yards_per_attempt': 'mean'
}).reset_index()

# Add team names
team_season_stats = team_season_stats.merge(
    team_info[['team_id', 'team_name', 'abrv']],
    on='team_id',
    how='left'
)

# Define pass-funnel criteria (top 25% in pass play percentage)
pass_funnel_threshold = team_season_stats['pass_play_pct'].quantile(0.75)
team_season_stats['is_pass_funnel'] = team_season_stats['pass_play_pct'] >= pass_funnel_threshold

pass_funnel_teams = team_season_stats[team_season_stats['is_pass_funnel']].sort_values(
    'pass_play_pct', ascending=False
)

print(f"\n  Pass-Funnel Teams (>= {pass_funnel_threshold:.1%} pass plays):")
print(pass_funnel_teams[['season_id', 'abrv', 'pass_play_pct', 'pass_attempts_per_game',
                          'total_rec_yds']].head(15).to_string(index=False))

# ============================================================================
# SECTION 8: Create Visualizations
# ============================================================================
print("\n[8/8] Creating visualizations...")

# Figure 1: Team Pass Volume vs Aggregate Receiver Yards
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Team Offensive Metrics vs Receiver Production', fontsize=16, fontweight='bold')

# Plot 1: Pass Attempts vs Total Receiver Yards
ax1 = axes[0, 0]
scatter1 = ax1.scatter(analysis_df['pass_attempts_per_game'],
                       analysis_df['total_rec_yds'],
                       alpha=0.5, s=50)
z1 = np.polyfit(analysis_df['pass_attempts_per_game'].dropna(),
                analysis_df.loc[analysis_df['pass_attempts_per_game'].notna(), 'total_rec_yds'], 1)
p1 = np.poly1d(z1)
ax1.plot(analysis_df['pass_attempts_per_game'].sort_values(),
         p1(analysis_df['pass_attempts_per_game'].sort_values()),
         "r--", linewidth=2, label=f'Trend Line')
corr1 = analysis_df[['pass_attempts_per_game', 'total_rec_yds']].corr().iloc[0, 1]
ax1.set_xlabel('Pass Attempts per Game', fontsize=11)
ax1.set_ylabel('Total Receiving Yards', fontsize=11)
ax1.set_title(f'Pass Attempts vs Receiver Production\n(r = {corr1:.3f})', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Yards per Attempt vs Total Receiver Yards
ax2 = axes[0, 1]
scatter2 = ax2.scatter(analysis_df['yards_per_attempt'],
                       analysis_df['total_rec_yds'],
                       alpha=0.5, s=50, color='orange')
valid_data2 = analysis_df[['yards_per_attempt', 'total_rec_yds']].dropna()
z2 = np.polyfit(valid_data2['yards_per_attempt'], valid_data2['total_rec_yds'], 1)
p2 = np.poly1d(z2)
ax2.plot(valid_data2['yards_per_attempt'].sort_values(),
         p2(valid_data2['yards_per_attempt'].sort_values()),
         "r--", linewidth=2, label='Trend Line')
corr2 = valid_data2.corr().iloc[0, 1]
ax2.set_xlabel('Yards per Attempt', fontsize=11)
ax2.set_ylabel('Total Receiving Yards', fontsize=11)
ax2.set_title(f'Efficiency (YPA) vs Receiver Production\n(r = {corr2:.3f})', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Pass Play Percentage vs Total Receiver Yards
ax3 = axes[1, 0]
scatter3 = ax3.scatter(analysis_df['pass_play_pct'] * 100,
                       analysis_df['total_rec_yds'],
                       alpha=0.5, s=50, color='green')
valid_data3 = analysis_df[['pass_play_pct', 'total_rec_yds']].dropna()
z3 = np.polyfit(valid_data3['pass_play_pct'] * 100, valid_data3['total_rec_yds'], 1)
p3 = np.poly1d(z3)
ax3.plot(valid_data3['pass_play_pct'].sort_values() * 100,
         p3(valid_data3['pass_play_pct'].sort_values() * 100),
         "r--", linewidth=2, label='Trend Line')
corr3 = valid_data3.corr().iloc[0, 1]
ax3.set_xlabel('Pass Play % of Total Plays', fontsize=11)
ax3.set_ylabel('Total Receiving Yards', fontsize=11)
ax3.set_title(f'Pass-Heavy Offense vs Receiver Production\n(r = {corr3:.3f})', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Completion % vs Total Receiver Yards
ax4 = axes[1, 1]
valid_data4 = analysis_df[['completion_pct', 'total_rec_yds']].dropna()
scatter4 = ax4.scatter(valid_data4['completion_pct'] * 100,
                       valid_data4['total_rec_yds'],
                       alpha=0.5, s=50, color='purple')
z4 = np.polyfit(valid_data4['completion_pct'] * 100, valid_data4['total_rec_yds'], 1)
p4 = np.poly1d(z4)
ax4.plot(valid_data4['completion_pct'].sort_values() * 100,
         p4(valid_data4['completion_pct'].sort_values() * 100),
         "r--", linewidth=2, label='Trend Line')
corr4 = valid_data4.corr().iloc[0, 1]
ax4.set_xlabel('Completion %', fontsize=11)
ax4.set_ylabel('Total Receiving Yards', fontsize=11)
ax4.set_title(f'Completion % vs Receiver Production\n(r = {corr4:.3f})', fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_VIZ / 'team_pass_volume_vs_rec_yds.png', dpi=300, bbox_inches='tight')
print(f"  - Saved: team_pass_volume_vs_rec_yds.png")
plt.close()

# Figure 2: Rolling Averages Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Rolling Average Metrics vs Receiver Production', fontsize=16, fontweight='bold')

# Compare current game vs 3-game rolling vs 5-game rolling vs season average
metrics_to_compare = ['pass_attempts_per_game', 'yards_per_attempt']

for idx, metric in enumerate(metrics_to_compare):
    row = idx

    # Current game value
    ax1 = axes[row, 0]
    valid_current = analysis_df[[metric, 'total_rec_yds']].dropna()
    ax1.scatter(valid_current[metric], valid_current['total_rec_yds'],
               alpha=0.4, s=30, label='Current Game')
    corr_current = valid_current.corr().iloc[0, 1]

    # 3-game rolling
    metric_roll3 = f'{metric}_roll3'
    if metric_roll3 in analysis_df.columns:
        valid_roll3 = analysis_df[[metric_roll3, 'total_rec_yds']].dropna()
        ax1.scatter(valid_roll3[metric_roll3], valid_roll3['total_rec_yds'],
                   alpha=0.4, s=30, label='3-Game Avg', marker='^')
        corr_roll3 = valid_roll3.corr().iloc[0, 1]

    ax1.set_xlabel(metric.replace('_', ' ').title(), fontsize=11)
    ax1.set_ylabel('Total Receiving Yards', fontsize=11)
    ax1.set_title(f'Current vs 3-Game Avg\n(r_current={corr_current:.3f}, r_3game={corr_roll3:.3f})',
                  fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 5-game rolling vs season average
    ax2 = axes[row, 1]
    metric_roll5 = f'{metric}_roll5'
    metric_szn = f'{metric}_szn_avg'

    if metric_roll5 in analysis_df.columns:
        valid_roll5 = analysis_df[[metric_roll5, 'total_rec_yds']].dropna()
        ax2.scatter(valid_roll5[metric_roll5], valid_roll5['total_rec_yds'],
                   alpha=0.4, s=30, label='5-Game Avg', marker='s')
        corr_roll5 = valid_roll5.corr().iloc[0, 1]

    if metric_szn in analysis_df.columns:
        valid_szn = analysis_df[[metric_szn, 'total_rec_yds']].dropna()
        ax2.scatter(valid_szn[metric_szn], valid_szn['total_rec_yds'],
                   alpha=0.4, s=30, label='Season Avg', marker='d')
        corr_szn = valid_szn.corr().iloc[0, 1]

    ax2.set_xlabel(metric.replace('_', ' ').title(), fontsize=11)
    ax2.set_ylabel('Total Receiving Yards', fontsize=11)
    ax2.set_title(f'5-Game vs Season Avg\n(r_5game={corr_roll5:.3f}, r_season={corr_szn:.3f})',
                  fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_VIZ / 'rolling_avg_comparison.png', dpi=300, bbox_inches='tight')
print(f"  - Saved: rolling_avg_comparison.png")
plt.close()

# ============================================================================
# SECTION 9: Save Outputs
# ============================================================================
print("\n[9/9] Saving analysis outputs...")

# Save correlation results
all_corr.to_csv(OUTPUT_DATA / 'team_offensive_correlations.csv', index=False)
print(f"  - Saved: team_offensive_correlations.csv")

# Save team rankings
team_rankings = team_season_stats.sort_values(
    ['season_id', 'total_rec_yds'], ascending=[True, False]
)
team_rankings.to_csv(OUTPUT_DATA / 'team_offensive_rankings.csv', index=False)
print(f"  - Saved: team_offensive_rankings.csv")

# Save pass-funnel teams
pass_funnel_teams.to_csv(OUTPUT_DATA / 'pass_funnel_teams.csv', index=False)
print(f"  - Saved: pass_funnel_teams.csv")

print("\n" + "=" * 80)
print("KEY FINDINGS:")
print("=" * 80)
print(f"\n1. STRONGEST PREDICTORS (Top 5):")
for idx, row in all_corr.head(5).iterrows():
    print(f"   {idx+1}. {row['metric']}: r = {row['pearson_corr']:.3f} (p < {row['pearson_pval']:.4f})")

print(f"\n2. VOLUME vs EFFICIENCY:")
volume_avg = volume_corr['abs_pearson'].mean()
efficiency_avg = efficiency_corr['abs_pearson'].mean()
print(f"   - Average |correlation| for volume metrics: {volume_avg:.3f}")
print(f"   - Average |correlation| for efficiency metrics: {efficiency_avg:.3f}")
if volume_avg > efficiency_avg:
    print(f"   => VOLUME metrics are stronger predictors (+{(volume_avg-efficiency_avg):.3f})")
else:
    print(f"   => EFFICIENCY metrics are stronger predictors (+{(efficiency_avg-volume_avg):.3f})")

print(f"\n3. ROLLING AVERAGES:")
roll3_avg = roll3_corr['abs_pearson'].mean()
roll5_avg = roll5_corr['abs_pearson'].mean()
print(f"   - 3-game rolling average correlation: {roll3_avg:.3f}")
print(f"   - 5-game rolling average correlation: {roll5_avg:.3f}")
if roll3_avg > roll5_avg:
    print(f"   => 3-game window captures recent form better")
else:
    print(f"   => 5-game window provides better stability")

print(f"\n4. PASS-FUNNEL TEAMS:")
print(f"   - {len(pass_funnel_teams)} team-seasons identified as pass-funnel")
print(f"   - Average receiver yards (pass-funnel): {pass_funnel_teams['total_rec_yds'].mean():.1f}")
print(f"   - Average receiver yards (all teams): {team_season_stats['total_rec_yds'].mean():.1f}")
diff_pct = ((pass_funnel_teams['total_rec_yds'].mean() / team_season_stats['total_rec_yds'].mean()) - 1) * 100
print(f"   => Pass-funnel teams boost receiver production by {diff_pct:+.1f}%")

print("\n" + "=" * 80)
print("Analysis complete! Check outputs folder for detailed results.")
print("=" * 80)
