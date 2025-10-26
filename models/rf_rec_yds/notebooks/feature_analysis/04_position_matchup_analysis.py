"""
Position-Specific Matchup Analysis
===================================
Analyzes whether position-specific defensive metrics (WR/TE/RB defense) provide
better predictions than overall pass defense metrics.

Objectives:
1. Compare position-specific vs overall defensive metrics
2. Identify defensive weaknesses by position
3. Quantify matchup advantages
4. Develop position-specific defensive rankings
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

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("POSITION-SPECIFIC MATCHUP ANALYSIS")
print("="*80)

# ============================================================================
# PART 1: Load and Merge Data
# ============================================================================

print("\n1. Loading data...")

# Load player game receiving data
plyr_rec = pd.read_parquet(DATA_DIR / "plyr_gm" / "plyr_gm_rec")

# Load game data for opponent matching
games = pd.read_parquet(DATA_DIR / "gm_info" / "nfl_game")

# Load defensive data
tm_def_pass = pd.read_parquet(DATA_DIR / "tm_szn" / "tm_def_pass")
tm_def_vs_wr = pd.read_parquet(DATA_DIR / "tm_szn" / "tm_def_vs_wr")
tm_def_vs_te = pd.read_parquet(DATA_DIR / "tm_szn" / "tm_def_vs_te")
tm_def_vs_rb = pd.read_parquet(DATA_DIR / "tm_szn" / "tm_def_vs_rb")

# Load player info to get positions
plyr_info = pd.read_parquet(DATA_DIR / "players" / "plyr")

print(f"   - Player receiving games: {plyr_rec.shape}")
print(f"   - Games: {games.shape}")
print(f"   - Pass defense: {tm_def_pass.shape}")
print(f"   - WR defense: {tm_def_vs_wr.shape}")
print(f"   - TE defense: {tm_def_vs_te.shape}")
print(f"   - RB defense: {tm_def_vs_rb.shape}")
print(f"   - Player info: {plyr_info.shape}")

# ============================================================================
# PART 2: Match Players to Opponent Defense
# ============================================================================

print("\n2. Matching players to opponent defenses...")

# Merge player games with game info to get opponent
plyr_rec_games = plyr_rec.merge(
    games[['game_id', 'home_team_id', 'away_team_id', 'season', 'week']],
    on='game_id',
    how='inner'
)

# Determine opponent team
plyr_rec_games['opponent_id'] = plyr_rec_games.apply(
    lambda row: row['away_team_id'] if row['team_id'] == row['home_team_id'] else row['home_team_id'],
    axis=1
)

print(f"   - Matched {plyr_rec_games.shape[0]} player-games to opponents")

# Add player position
plyr_rec_games = plyr_rec_games.merge(
    plyr_info[['plyr_id', 'plyr_pos']],
    on='plyr_id',
    how='left'
)

print(f"   - Added player positions")

# Filter to WR, TE, RB positions
position_filter = plyr_rec_games['plyr_pos'].isin(['WR', 'TE', 'RB'])
plyr_rec_games = plyr_rec_games[position_filter].copy()

print(f"   - Filtered to WR/TE/RB: {plyr_rec_games.shape[0]} player-games")
print(f"   - Position breakdown:")
print(plyr_rec_games['plyr_pos'].value_counts())

# ============================================================================
# PART 3: Merge Opponent Defensive Stats
# ============================================================================

print("\n3. Merging opponent defensive statistics...")

# For each player game, we want the opponent's defensive stats BEFORE the game
# Use week-1 data (cumulative stats through previous week)

# Adjust week for defensive stats lookup
plyr_rec_games['opponent_def_week'] = plyr_rec_games['week_id'] - 1

# Overall pass defense
plyr_rec_games = plyr_rec_games.merge(
    tm_def_pass,
    left_on=['opponent_id', 'season_id', 'opponent_def_week'],
    right_on=['team_id', 'season_id', 'week_id'],
    how='left',
    suffixes=('', '_def_pass')
)

# Position-specific defense
plyr_rec_games = plyr_rec_games.merge(
    tm_def_vs_wr,
    left_on=['opponent_id', 'season_id', 'opponent_def_week'],
    right_on=['team_id', 'season_id', 'week_id'],
    how='left',
    suffixes=('', '_def_wr')
)

plyr_rec_games = plyr_rec_games.merge(
    tm_def_vs_te,
    left_on=['opponent_id', 'season_id', 'opponent_def_week'],
    right_on=['team_id', 'season_id', 'week_id'],
    how='left',
    suffixes=('', '_def_te')
)

plyr_rec_games = plyr_rec_games.merge(
    tm_def_vs_rb,
    left_on=['opponent_id', 'season_id', 'opponent_def_week'],
    right_on=['team_id', 'season_id', 'week_id'],
    how='left',
    suffixes=('', '_def_rb')
)

print(f"   - After merging defensive stats: {plyr_rec_games.shape}")

# Calculate per-game defensive metrics
plyr_rec_games['games_played'] = plyr_rec_games['week_id_def_pass']

# Overall pass defense per game
plyr_rec_games['opp_pass_yds_per_game'] = plyr_rec_games['tm_def_pass_yds'] / plyr_rec_games['games_played']
plyr_rec_games['opp_pass_yds_att'] = plyr_rec_games['tm_def_pass_yds_att']
plyr_rec_games['opp_pass_rtg'] = plyr_rec_games['tm_def_pass_rtg']
plyr_rec_games['opp_pass_cmp_pct'] = plyr_rec_games['tm_def_pass_cmp_pct']

# Position-specific defense per game
plyr_rec_games['opp_wr_yds_per_game'] = plyr_rec_games['tm_def_wr_yds'].fillna(0) / plyr_rec_games['games_played']
plyr_rec_games['opp_wr_rec_per_game'] = plyr_rec_games['tm_def_wr_rec'].fillna(0) / plyr_rec_games['games_played']
plyr_rec_games['opp_wr_td_per_game'] = plyr_rec_games['tm_def_wr_td'].fillna(0) / plyr_rec_games['games_played']

plyr_rec_games['opp_te_yds_per_game'] = plyr_rec_games['tm_def_te_yds'].fillna(0) / plyr_rec_games['games_played']
plyr_rec_games['opp_te_rec_per_game'] = plyr_rec_games['tm_def_te_rec'].fillna(0) / plyr_rec_games['games_played']
plyr_rec_games['opp_te_td_per_game'] = plyr_rec_games['tm_def_te_td'].fillna(0) / plyr_rec_games['games_played']

plyr_rec_games['opp_rb_yds_per_game'] = plyr_rec_games['tm_def_rb_rec_yds'].fillna(0) / plyr_rec_games['games_played']
plyr_rec_games['opp_rb_rec_per_game'] = plyr_rec_games['tm_def_rb_rec'].fillna(0) / plyr_rec_games['games_played']

print("   - Calculated per-game defensive metrics")

# ============================================================================
# PART 4: Position-Specific Correlation Analysis
# ============================================================================

print("\n" + "="*80)
print("POSITION-SPECIFIC vs OVERALL DEFENSE CORRELATION")
print("="*80)

# Define analysis for each position
positions = {
    'WR': {
        'overall_metrics': ['opp_pass_yds_per_game', 'opp_pass_yds_att', 'opp_pass_rtg', 'opp_pass_cmp_pct'],
        'position_metrics': ['opp_wr_yds_per_game', 'opp_wr_rec_per_game', 'opp_wr_td_per_game']
    },
    'TE': {
        'overall_metrics': ['opp_pass_yds_per_game', 'opp_pass_yds_att', 'opp_pass_rtg', 'opp_pass_cmp_pct'],
        'position_metrics': ['opp_te_yds_per_game', 'opp_te_rec_per_game', 'opp_te_td_per_game']
    },
    'RB': {
        'overall_metrics': ['opp_pass_yds_per_game', 'opp_pass_yds_att', 'opp_pass_rtg', 'opp_pass_cmp_pct'],
        'position_metrics': ['opp_rb_yds_per_game', 'opp_rb_rec_per_game']
    }
}

# Calculate correlations for each position
correlation_results = []

for pos, metrics in positions.items():
    pos_data = plyr_rec_games[plyr_rec_games['plyr_pos'] == pos].copy()

    print(f"\n{pos} Position - {len(pos_data)} games:")
    print("-" * 80)

    # Overall defense metrics
    print(f"\nOverall Pass Defense Metrics:")
    for metric in metrics['overall_metrics']:
        valid_data = pos_data[['plyr_gm_rec_yds', metric]].dropna()

        if len(valid_data) > 50:
            r, p = stats.pearsonr(valid_data['plyr_gm_rec_yds'], valid_data[metric])
            print(f"  {metric:30s}: r = {r:6.3f}, p = {p:.4f}, N = {len(valid_data)}")

            correlation_results.append({
                'Position': pos,
                'Metric_Type': 'Overall',
                'Metric': metric,
                'Correlation': r,
                'P_value': p,
                'N': len(valid_data)
            })

    # Position-specific metrics
    print(f"\nPosition-Specific Defense Metrics:")
    for metric in metrics['position_metrics']:
        valid_data = pos_data[['plyr_gm_rec_yds', metric]].dropna()

        if len(valid_data) > 50:
            r, p = stats.pearsonr(valid_data['plyr_gm_rec_yds'], valid_data[metric])
            print(f"  {metric:30s}: r = {r:6.3f}, p = {p:.4f}, N = {len(valid_data)}")

            correlation_results.append({
                'Position': pos,
                'Metric_Type': 'Position_Specific',
                'Metric': metric,
                'Correlation': r,
                'P_value': p,
                'N': len(valid_data)
            })

corr_results_df = pd.DataFrame(correlation_results)
corr_results_df['Abs_Correlation'] = corr_results_df['Correlation'].abs()

# Save results
corr_results_df.to_csv(PRED_DIR / 'position_specific_correlations.csv', index=False)
print(f"\nSaved correlation results to: {PRED_DIR / 'position_specific_correlations.csv'}")

# ============================================================================
# PART 5: Comparative Analysis
# ============================================================================

print("\n" + "="*80)
print("COMPARATIVE ANALYSIS: Position-Specific vs Overall Defense")
print("="*80)

# Calculate average absolute correlation for each type
summary_stats = corr_results_df.groupby(['Position', 'Metric_Type']).agg({
    'Abs_Correlation': ['mean', 'max'],
    'N': 'mean'
}).round(3)

print("\nAverage Correlation Strength by Position and Metric Type:")
print("-" * 80)
print(summary_stats)

# Determine winner for each position
for pos in ['WR', 'TE', 'RB']:
    pos_corr = corr_results_df[corr_results_df['Position'] == pos]

    overall_avg = pos_corr[pos_corr['Metric_Type'] == 'Overall']['Abs_Correlation'].mean()
    position_avg = pos_corr[pos_corr['Metric_Type'] == 'Position_Specific']['Abs_Correlation'].mean()

    print(f"\n{pos}:")
    print(f"  Overall Defense Avg |r|: {overall_avg:.3f}")
    print(f"  Position Defense Avg |r|: {position_avg:.3f}")

    if position_avg > overall_avg:
        improvement = ((position_avg - overall_avg) / overall_avg) * 100
        print(f"  Winner: Position-Specific (+{improvement:.1f}%)")
    else:
        print(f"  Winner: Overall Defense")

# ============================================================================
# PART 6: Position-Specific Rankings
# ============================================================================

print("\n" + "="*80)
print("POSITION-SPECIFIC DEFENSIVE RANKINGS")
print("="*80)

# Create rankings for each position
position_rankings = []

for season in tm_def_vs_wr['season'].unique():
    for week in tm_def_vs_wr[tm_def_vs_wr['season'] == season]['week'].unique():

        # WR Defense
        wr_def = tm_def_vs_wr[(tm_def_vs_wr['season'] == season) & (tm_def_vs_wr['week'] == week)].copy()
        wr_def['games_played'] = wr_def['week_id']
        wr_def['wr_yds_per_game'] = wr_def['tm_def_wr_yds'].fillna(0) / wr_def['games_played']
        wr_def['wr_rank'] = wr_def['wr_yds_per_game'].rank(method='min')
        wr_def['wr_percentile'] = (1 - wr_def['wr_yds_per_game'].rank(pct=True)) * 100

        # TE Defense
        te_def = tm_def_vs_te[(tm_def_vs_te['season'] == season) & (tm_def_vs_te['week'] == week)].copy()
        te_def['games_played'] = te_def['week_id']
        te_def['te_yds_per_game'] = te_def['tm_def_te_yds'].fillna(0) / te_def['games_played']
        te_def['te_rank'] = te_def['te_yds_per_game'].rank(method='min')
        te_def['te_percentile'] = (1 - te_def['te_yds_per_game'].rank(pct=True)) * 100

        # RB Defense
        rb_def = tm_def_vs_rb[(tm_def_vs_rb['season'] == season) & (tm_def_vs_rb['week'] == week)].copy()
        rb_def['games_played'] = rb_def['week_id']
        rb_def['rb_yds_per_game'] = rb_def['tm_def_rb_rec_yds'].fillna(0) / rb_def['games_played']
        rb_def['rb_rank'] = rb_def['rb_yds_per_game'].rank(method='min')
        rb_def['rb_percentile'] = (1 - rb_def['rb_yds_per_game'].rank(pct=True)) * 100

        # Merge all positions
        combined = wr_def[['team_id', 'season', 'week', 'wr_yds_per_game', 'wr_rank', 'wr_percentile']].merge(
            te_def[['team_id', 'season', 'week', 'te_yds_per_game', 'te_rank', 'te_percentile']],
            on=['team_id', 'season', 'week']
        ).merge(
            rb_def[['team_id', 'season', 'week', 'rb_yds_per_game', 'rb_rank', 'rb_percentile']],
            on=['team_id', 'season', 'week']
        )

        position_rankings.append(combined)

rankings_df = pd.concat(position_rankings, ignore_index=True)

# Define defensive tiers
def assign_tier(percentile):
    if percentile >= 90:
        return 'Elite'
    elif percentile >= 75:
        return 'Above Average'
    elif percentile >= 50:
        return 'Average'
    elif percentile >= 25:
        return 'Below Average'
    else:
        return 'Poor'

rankings_df['wr_tier'] = rankings_df['wr_percentile'].apply(assign_tier)
rankings_df['te_tier'] = rankings_df['te_percentile'].apply(assign_tier)
rankings_df['rb_tier'] = rankings_df['rb_percentile'].apply(assign_tier)

# Save rankings
rankings_df.to_csv(PRED_DIR / 'position_specific_rankings.csv', index=False)
print(f"\nSaved position-specific rankings to: {PRED_DIR / 'position_specific_rankings.csv'}")

# Display top and bottom defenses by position
print("\nTop 5 WR Defenses (2024 Season End):")
season_2024 = rankings_df[(rankings_df['season'] == 2024) & (rankings_df['week'] == 18)].sort_values('wr_rank')
if len(season_2024) > 0:
    print(season_2024[['team_id', 'wr_yds_per_game', 'wr_rank', 'wr_tier']].head().to_string(index=False))

print("\nTop 5 TE Defenses (2024 Season End):")
if len(season_2024) > 0:
    print(season_2024.sort_values('te_rank')[['team_id', 'te_yds_per_game', 'te_rank', 'te_tier']].head().to_string(index=False))

# ============================================================================
# PART 7: Matchup Advantage Analysis
# ============================================================================

print("\n" + "="*80)
print("MATCHUP ADVANTAGE ANALYSIS")
print("="*80)

# Identify defensive weaknesses (teams strong overall but weak vs specific position)
print("\nTeams with Position-Specific Weaknesses (2024):")
print("-" * 80)

season_2024 = rankings_df[(rankings_df['season'] == 2024) & (rankings_df['week'] == 18)]

if len(season_2024) > 0:
    # WR-friendly but overall good
    wr_friendly = season_2024[season_2024['wr_percentile'] < 30]  # Weak vs WR
    print(f"\nWR-Friendly Defenses (Weak vs WR):")
    print(wr_friendly[['team_id', 'wr_yds_per_game', 'wr_tier']].to_string(index=False))

    # TE-friendly
    te_friendly = season_2024[season_2024['te_percentile'] < 30]  # Weak vs TE
    print(f"\nTE-Friendly Defenses (Weak vs TE):")
    print(te_friendly[['team_id', 'te_yds_per_game', 'te_tier']].to_string(index=False))

# ============================================================================
# PART 8: Visualization - Position-Specific Rankings
# ============================================================================

print("\n" + "="*80)
print("CREATING POSITION-SPECIFIC RANKINGS VISUALIZATION")
print("="*80)

# Get latest week for each season
# Convert week to numeric to avoid categorical issues
rankings_df['week'] = rankings_df['week'].astype(int)
latest_weeks = rankings_df.groupby('season')['week'].max().reset_index()
latest_data = rankings_df.merge(latest_weeks, on=['season', 'week'])

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

positions_viz = [
    ('wr_yds_per_game', 'wr_rank', 'WR Defense', axes[0]),
    ('te_yds_per_game', 'te_rank', 'TE Defense', axes[1]),
    ('rb_yds_per_game', 'rb_rank', 'RB Defense', axes[2])
]

for metric, rank_col, title, ax in positions_viz:
    # Get 2024 data
    data_2024 = latest_data[latest_data['season'] == 2024].sort_values(rank_col)

    if len(data_2024) > 0:
        # Create horizontal bar chart
        y_pos = np.arange(len(data_2024))
        colors = sns.color_palette("RdYlGn_r", len(data_2024))

        ax.barh(y_pos, data_2024[metric], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(data_2024['team_id'], fontsize=8)
        ax.set_xlabel('Yards per Game Allowed', fontweight='bold')
        ax.set_title(f'{title} Rankings (2024)', fontweight='bold', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

        # Add value labels
        for i, (idx, row) in enumerate(data_2024.iterrows()):
            ax.text(row[metric] + 1, i, f"{row[metric]:.1f}",
                   va='center', fontsize=7)

plt.suptitle('Position-Specific Defensive Rankings - 2024 Season',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(VIZ_DIR / 'position_specific_defense_rankings.png', dpi=300, bbox_inches='tight')
print(f"Saved visualization to: {VIZ_DIR / 'position_specific_defense_rankings.png'}")
plt.close()

# ============================================================================
# PART 9: Matchup Quality Indicators
# ============================================================================

print("\n" + "="*80)
print("MATCHUP QUALITY INDICATORS")
print("="*80)

# Analyze performance by defensive tier
matchup_analysis = []

for pos in ['WR', 'TE', 'RB']:
    pos_data = plyr_rec_games[plyr_rec_games['plyr_pos'] == pos].copy()

    # Merge defensive rankings
    if pos == 'WR':
        pos_data = pos_data.merge(
            rankings_df[['team_id', 'season', 'week', 'wr_tier', 'wr_percentile']],
            left_on=['opponent_id', 'season', 'week'],
            right_on=['team_id', 'season', 'week'],
            how='left',
            suffixes=('', '_opp')
        )
        tier_col = 'wr_tier'
        pctile_col = 'wr_percentile'
    elif pos == 'TE':
        pos_data = pos_data.merge(
            rankings_df[['team_id', 'season', 'week', 'te_tier', 'te_percentile']],
            left_on=['opponent_id', 'season', 'week'],
            right_on=['team_id', 'season', 'week'],
            how='left',
            suffixes=('', '_opp')
        )
        tier_col = 'te_tier'
        pctile_col = 'te_percentile'
    else:  # RB
        pos_data = pos_data.merge(
            rankings_df[['team_id', 'season', 'week', 'rb_tier', 'rb_percentile']],
            left_on=['opponent_id', 'season', 'week'],
            right_on=['team_id', 'season', 'week'],
            how='left',
            suffixes=('', '_opp')
        )
        tier_col = 'rb_tier'
        pctile_col = 'rb_percentile'

    # Performance by tier
    tier_stats = pos_data.groupby(tier_col)['plyr_gm_rec_yds'].agg(['mean', 'median', 'std', 'count'])

    print(f"\n{pos} Performance by Opponent Defensive Tier:")
    print("-" * 80)
    print(tier_stats.to_string())

    for tier in tier_stats.index:
        matchup_analysis.append({
            'Position': pos,
            'Defensive_Tier': tier,
            'Avg_Yards': tier_stats.loc[tier, 'mean'],
            'Median_Yards': tier_stats.loc[tier, 'median'],
            'Std_Yards': tier_stats.loc[tier, 'std'],
            'N': tier_stats.loc[tier, 'count']
        })

matchup_df = pd.DataFrame(matchup_analysis)
matchup_df.to_csv(PRED_DIR / 'matchup_quality_by_tier.csv', index=False)
print(f"\nSaved matchup analysis to: {PRED_DIR / 'matchup_quality_by_tier.csv'}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80)

print("\nKey Findings:")
print("-" * 80)

# Compare position-specific vs overall for each position
for pos in ['WR', 'TE', 'RB']:
    pos_corr = corr_results_df[corr_results_df['Position'] == pos]

    overall_best = pos_corr[pos_corr['Metric_Type'] == 'Overall'].nlargest(1, 'Abs_Correlation')
    position_best = pos_corr[pos_corr['Metric_Type'] == 'Position_Specific'].nlargest(1, 'Abs_Correlation')

    print(f"\n{pos} Position:")
    if len(overall_best) > 0:
        print(f"  Best Overall Metric: {overall_best.iloc[0]['Metric']} (r = {overall_best.iloc[0]['Correlation']:.3f})")
    if len(position_best) > 0:
        print(f"  Best Position Metric: {position_best.iloc[0]['Metric']} (r = {position_best.iloc[0]['Correlation']:.3f})")

print("\n" + "="*80)
print("POSITION-SPECIFIC MATCHUP ANALYSIS COMPLETE")
print("="*80)
print(f"\nOutputs saved to:")
print(f"  - {PRED_DIR / 'position_specific_correlations.csv'}")
print(f"  - {PRED_DIR / 'position_specific_rankings.csv'}")
print(f"  - {PRED_DIR / 'matchup_quality_by_tier.csv'}")
print(f"  - {VIZ_DIR / 'position_specific_defense_rankings.png'}")
print("="*80)
