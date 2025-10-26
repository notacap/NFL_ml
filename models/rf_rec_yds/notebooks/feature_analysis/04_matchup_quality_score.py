"""
Matchup Quality Score Development
==================================
Develops a comprehensive matchup quality scoring system that combines multiple
defensive factors to predict receiver performance.

Objectives:
1. Create composite matchup quality scores
2. Optimize component weights
3. Validate score predictive power
4. Analyze player skill x matchup interactions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
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
print("MATCHUP QUALITY SCORE DEVELOPMENT")
print("="*80)

# ============================================================================
# PART 1: Load Preprocessed Data
# ============================================================================

print("\n1. Loading data...")

# Load player receiving games with opponent data
plyr_rec = pd.read_parquet(DATA_DIR / "plyr_gm" / "plyr_gm_rec")
games = pd.read_parquet(DATA_DIR / "gm_info" / "nfl_game")
plyr_info = pd.read_parquet(DATA_DIR / "players" / "plyr")

# Load defensive rankings
def_rankings = pd.read_csv(PRED_DIR / 'defensive_rankings_all.csv')
pos_rankings = pd.read_csv(PRED_DIR / 'position_specific_rankings.csv')

print(f"   - Player receiving games: {plyr_rec.shape}")
print(f"   - Games: {games.shape}")
print(f"   - Player info: {plyr_info.shape}")
print(f"   - Defensive rankings: {def_rankings.shape}")
print(f"   - Position rankings: {pos_rankings.shape}")

# ============================================================================
# PART 2: Build Comprehensive Dataset
# ============================================================================

print("\n2. Building comprehensive matchup dataset...")

# Merge player games with game info
plyr_rec_games = plyr_rec.merge(
    games[['game_id', 'home_team_id', 'away_team_id', 'season', 'week']],
    on='game_id',
    how='inner'
)

# Determine opponent
plyr_rec_games['opponent_id'] = plyr_rec_games.apply(
    lambda row: row['away_team_id'] if row['team_id'] == row['home_team_id'] else row['home_team_id'],
    axis=1
)

# Add player position
plyr_rec_games = plyr_rec_games.merge(
    plyr_info[['plyr_id', 'plyr_pos']],
    on='plyr_id',
    how='left'
)

# Filter to WR, TE, RB
plyr_rec_games = plyr_rec_games[plyr_rec_games['plyr_pos'].isin(['WR', 'TE', 'RB'])].copy()

print(f"   - Built dataset with {plyr_rec_games.shape[0]} player-games")

# ============================================================================
# PART 3: Merge Defensive Metrics
# ============================================================================

print("\n3. Merging opponent defensive metrics...")

# Merge overall defensive rankings (using previous week's stats)
plyr_rec_games['opponent_def_week'] = plyr_rec_games['week_id'] - 1

plyr_rec_games = plyr_rec_games.merge(
    def_rankings[['team_id', 'season', 'week', 'pass_defense_score', 'pressure_score', 'overall_defense_score',
                  'pass_yds_per_game_pctile', 'tm_def_pass_yds_att_pctile', 'tm_def_pass_rtg_pctile',
                  'tm_def_sk_pct_pctile', 'tm_def_prss_pct_pctile']],
    left_on=['opponent_id', 'season', 'opponent_def_week'],
    right_on=['team_id', 'season', 'week'],
    how='left',
    suffixes=('', '_opp_def')
)

# Merge position-specific rankings
plyr_rec_games = plyr_rec_games.merge(
    pos_rankings[['team_id', 'season', 'week', 'wr_percentile', 'te_percentile', 'rb_percentile',
                  'wr_tier', 'te_tier', 'rb_tier']],
    left_on=['opponent_id', 'season', 'opponent_def_week'],
    right_on=['team_id', 'season', 'week'],
    how='left',
    suffixes=('', '_pos_def')
)

print(f"   - After merging: {plyr_rec_games.shape}")
print(f"   - Non-null defensive scores: {plyr_rec_games['overall_defense_score'].notna().sum()}")

# ============================================================================
# PART 4: Create Position-Specific Matchup Scores
# ============================================================================

print("\n4. Creating position-specific matchup scores...")

# Assign the appropriate position-specific percentile
def get_position_percentile(row):
    if row['plyr_pos'] == 'WR':
        return row['wr_percentile']
    elif row['plyr_pos'] == 'TE':
        return row['te_percentile']
    elif row['plyr_pos'] == 'RB':
        return row['rb_percentile']
    else:
        return np.nan

plyr_rec_games['position_defense_pctile'] = plyr_rec_games.apply(get_position_percentile, axis=1)

# Calculate player historical average (as proxy for skill)
player_averages = plyr_rec_games.groupby('plyr_id')['plyr_gm_rec_yds'].agg(['mean', 'count']).reset_index()
player_averages.columns = ['plyr_id', 'player_avg_yards', 'player_games_played']

plyr_rec_games = plyr_rec_games.merge(player_averages, on='plyr_id', how='left')

# Classify player skill tiers
plyr_rec_games['player_skill_tier'] = pd.cut(
    plyr_rec_games['player_avg_yards'],
    bins=[0, 20, 40, 60, 1000],
    labels=['Low', 'Medium', 'High', 'Elite']
)

print("   - Created position-specific defensive percentiles")
print("   - Added player skill tiers")

# ============================================================================
# PART 5: Develop Matchup Quality Score v1.0 (Simple Weighted)
# ============================================================================

print("\n" + "="*80)
print("MATCHUP QUALITY SCORE v1.0 - Simple Weighted Approach")
print("="*80)

# Version 1: Simple weighted combination
# Higher score = Better matchup for the receiver (weaker defense)

# Component weights (to be optimized)
weights_v1 = {
    'overall_defense': 0.30,
    'position_specific': 0.50,
    'pressure': 0.20
}

plyr_rec_games['matchup_score_v1'] = (
    (100 - plyr_rec_games['overall_defense_score']) * weights_v1['overall_defense'] +
    (100 - plyr_rec_games['position_defense_pctile']) * weights_v1['position_specific'] +
    plyr_rec_games['pressure_score'] * weights_v1['pressure']
)

# Normalize to 0-100 scale
plyr_rec_games['matchup_score_v1'] = (
    (plyr_rec_games['matchup_score_v1'] - plyr_rec_games['matchup_score_v1'].min()) /
    (plyr_rec_games['matchup_score_v1'].max() - plyr_rec_games['matchup_score_v1'].min())
) * 100

print("\nMatchup Score v1.0 Statistics:")
print(plyr_rec_games['matchup_score_v1'].describe())

# Validate v1 score
valid_data_v1 = plyr_rec_games[['plyr_gm_rec_yds', 'matchup_score_v1']].dropna()
if len(valid_data_v1) > 100:
    r_v1, p_v1 = stats.pearsonr(valid_data_v1['plyr_gm_rec_yds'], valid_data_v1['matchup_score_v1'])
    print(f"\nCorrelation with actual yards: r = {r_v1:.3f}, p = {p_v1:.4f}, N = {len(valid_data_v1)}")

# ============================================================================
# PART 6: Develop Matchup Quality Score v2.0 (Optimized Weights)
# ============================================================================

print("\n" + "="*80)
print("MATCHUP QUALITY SCORE v2.0 - Optimized Weights")
print("="*80)

# Prepare data for optimization
feature_cols = ['pass_defense_score', 'position_defense_pctile', 'pressure_score',
                'pass_yds_per_game_pctile', 'tm_def_pass_rtg_pctile']

optimization_data = plyr_rec_games[feature_cols + ['plyr_gm_rec_yds']].dropna()

if len(optimization_data) > 1000:
    X_opt = optimization_data[feature_cols].values
    y_opt = optimization_data['plyr_gm_rec_yds'].values

    # Invert defensive scores (so higher = better matchup)
    X_opt[:, 0] = 100 - X_opt[:, 0]  # pass_defense_score
    X_opt[:, 1] = 100 - X_opt[:, 1]  # position_defense_pctile
    # pressure_score stays as is (higher = more pressure = worse for QB/receivers)
    X_opt[:, 3] = 100 - X_opt[:, 3]  # pass_yds_per_game_pctile
    X_opt[:, 4] = 100 - X_opt[:, 4]  # tm_def_pass_rtg_pctile

    # Standardize features
    scaler = StandardScaler()
    X_opt_scaled = scaler.fit_transform(X_opt)

    # Use Ridge regression to find optimal weights
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_opt_scaled, y_opt)

    # Extract coefficients as weights
    weights_v2 = ridge.coef_

    print("\nOptimized Feature Weights:")
    for i, col in enumerate(feature_cols):
        print(f"  {col:35s}: {weights_v2[i]:.3f}")

    # Cross-validation score
    cv_scores = cross_val_score(ridge, X_opt_scaled, y_opt, cv=5, scoring='r2')
    print(f"\nCross-Validation R² Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Apply optimized weights to create v2 score
    X_full = plyr_rec_games[feature_cols].copy()
    X_full.iloc[:, 0] = 100 - X_full.iloc[:, 0]
    X_full.iloc[:, 1] = 100 - X_full.iloc[:, 1]
    X_full.iloc[:, 3] = 100 - X_full.iloc[:, 3]
    X_full.iloc[:, 4] = 100 - X_full.iloc[:, 4]

    X_full_scaled = scaler.transform(X_full.fillna(0))
    plyr_rec_games['matchup_score_v2'] = ridge.predict(X_full_scaled)

    # Normalize to 0-100 scale
    plyr_rec_games['matchup_score_v2'] = (
        (plyr_rec_games['matchup_score_v2'] - plyr_rec_games['matchup_score_v2'].min()) /
        (plyr_rec_games['matchup_score_v2'].max() - plyr_rec_games['matchup_score_v2'].min())
    ) * 100

    print("\nMatchup Score v2.0 Statistics:")
    print(plyr_rec_games['matchup_score_v2'].describe())

else:
    print("\nInsufficient data for weight optimization")
    plyr_rec_games['matchup_score_v2'] = plyr_rec_games['matchup_score_v1']

# ============================================================================
# PART 7: Define Matchup Quality Tiers
# ============================================================================

print("\n" + "="*80)
print("MATCHUP QUALITY TIERS")
print("="*80)

# Define tiers based on percentile distribution
def assign_matchup_tier(score):
    if pd.isna(score):
        return 'Unknown'
    elif score >= 75:
        return 'Elite Matchup'
    elif score >= 60:
        return 'Favorable'
    elif score >= 40:
        return 'Neutral'
    elif score >= 25:
        return 'Difficult'
    else:
        return 'Avoid'

plyr_rec_games['matchup_tier_v1'] = plyr_rec_games['matchup_score_v1'].apply(assign_matchup_tier)
plyr_rec_games['matchup_tier_v2'] = plyr_rec_games['matchup_score_v2'].apply(assign_matchup_tier)

# Analyze performance by tier
print("\nPerformance by Matchup Tier (v1):")
print("-" * 80)
tier_stats_v1 = plyr_rec_games.groupby('matchup_tier_v1')['plyr_gm_rec_yds'].agg(['mean', 'median', 'std', 'count'])
print(tier_stats_v1.to_string())

print("\nPerformance by Matchup Tier (v2):")
print("-" * 80)
tier_stats_v2 = plyr_rec_games.groupby('matchup_tier_v2')['plyr_gm_rec_yds'].agg(['mean', 'median', 'std', 'count'])
print(tier_stats_v2.to_string())

# ============================================================================
# PART 8: Player Skill × Matchup Interaction Analysis
# ============================================================================

print("\n" + "="*80)
print("PLAYER SKILL × MATCHUP QUALITY INTERACTION")
print("="*80)

# Analyze how matchup quality affects players of different skill levels
interaction_data = plyr_rec_games[
    plyr_rec_games['player_skill_tier'].notna() &
    plyr_rec_games['matchup_tier_v2'].notna()
].copy()

if len(interaction_data) > 100:
    interaction_stats = interaction_data.groupby(['player_skill_tier', 'matchup_tier_v2'])['plyr_gm_rec_yds'].agg([
        'mean', 'median', 'count'
    ]).round(1)

    print("\nAverage Receiving Yards by Player Skill × Matchup Quality:")
    print("-" * 80)
    print(interaction_stats.to_string())

    # Pivot for easier reading
    pivot_mean = interaction_data.pivot_table(
        values='plyr_gm_rec_yds',
        index='player_skill_tier',
        columns='matchup_tier_v2',
        aggfunc='mean'
    )

    print("\nPivot Table - Average Yards:")
    print(pivot_mean.round(1).to_string())

    # Calculate matchup sensitivity (variance across matchup tiers)
    matchup_sensitivity = []
    for skill_tier in interaction_data['player_skill_tier'].unique():
        skill_data = interaction_data[interaction_data['player_skill_tier'] == skill_tier]
        tier_means = skill_data.groupby('matchup_tier_v2')['plyr_gm_rec_yds'].mean()

        if len(tier_means) > 1:
            sensitivity = tier_means.std()
            matchup_sensitivity.append({
                'Skill_Tier': skill_tier,
                'Matchup_Sensitivity': sensitivity,
                'Range': tier_means.max() - tier_means.min()
            })

    sensitivity_df = pd.DataFrame(matchup_sensitivity)
    print("\nMatchup Sensitivity by Skill Tier:")
    print("-" * 80)
    print(sensitivity_df.to_string(index=False))

# ============================================================================
# PART 9: Visualization - Matchup Quality Distribution
# ============================================================================

print("\n" + "="*80)
print("CREATING MATCHUP QUALITY VISUALIZATIONS")
print("="*80)

# Distribution of matchup scores
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Score distribution
ax1 = axes[0, 0]
plyr_rec_games['matchup_score_v2'].dropna().hist(bins=30, ax=ax1, edgecolor='black', alpha=0.7)
ax1.axvline(plyr_rec_games['matchup_score_v2'].median(), color='red', linestyle='--',
            label=f"Median: {plyr_rec_games['matchup_score_v2'].median():.1f}")
ax1.set_xlabel('Matchup Quality Score', fontweight='bold')
ax1.set_ylabel('Frequency', fontweight='bold')
ax1.set_title('Distribution of Matchup Quality Scores', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Performance by tier
ax2 = axes[0, 1]
tier_order = ['Elite Matchup', 'Favorable', 'Neutral', 'Difficult', 'Avoid']
tier_data = plyr_rec_games[plyr_rec_games['matchup_tier_v2'].isin(tier_order)]
tier_means = tier_data.groupby('matchup_tier_v2')['plyr_gm_rec_yds'].mean().reindex(tier_order)
tier_colors = ['darkgreen', 'lightgreen', 'gold', 'orange', 'red']

ax2.bar(range(len(tier_means)), tier_means.values, color=tier_colors, edgecolor='black')
ax2.set_xticks(range(len(tier_means)))
ax2.set_xticklabels(tier_means.index, rotation=45, ha='right')
ax2.set_ylabel('Average Receiving Yards', fontweight='bold')
ax2.set_title('Performance by Matchup Quality Tier', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for i, v in enumerate(tier_means.values):
    if not np.isnan(v):
        ax2.text(i, v + 1, f'{v:.1f}', ha='center', fontweight='bold')

# Scatter: Matchup score vs actual yards
ax3 = axes[1, 0]
scatter_data = plyr_rec_games[['matchup_score_v2', 'plyr_gm_rec_yds']].dropna()
if len(scatter_data) > 100:
    ax3.scatter(scatter_data['matchup_score_v2'], scatter_data['plyr_gm_rec_yds'],
               alpha=0.3, s=10)
    # Add regression line
    z = np.polyfit(scatter_data['matchup_score_v2'], scatter_data['plyr_gm_rec_yds'], 1)
    p = np.poly1d(z)
    ax3.plot(scatter_data['matchup_score_v2'].sort_values(),
             p(scatter_data['matchup_score_v2'].sort_values()),
             "r--", linewidth=2, label=f'Trend Line')

    r, _ = stats.pearsonr(scatter_data['matchup_score_v2'], scatter_data['plyr_gm_rec_yds'])
    ax3.set_xlabel('Matchup Quality Score', fontweight='bold')
    ax3.set_ylabel('Actual Receiving Yards', fontweight='bold')
    ax3.set_title(f'Matchup Score vs Actual Performance (r={r:.3f})', fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)

# Player skill × matchup interaction heatmap
ax4 = axes[1, 1]
if len(interaction_data) > 100:
    pivot_for_heatmap = interaction_data.pivot_table(
        values='plyr_gm_rec_yds',
        index='player_skill_tier',
        columns='matchup_tier_v2',
        aggfunc='mean'
    )

    # Reorder columns
    col_order = [col for col in tier_order if col in pivot_for_heatmap.columns]
    pivot_for_heatmap = pivot_for_heatmap[col_order]

    sns.heatmap(pivot_for_heatmap, annot=True, fmt='.1f', cmap='RdYlGn',
               ax=ax4, cbar_kws={'label': 'Avg Yards'}, linewidths=0.5)
    ax4.set_xlabel('Matchup Quality Tier', fontweight='bold')
    ax4.set_ylabel('Player Skill Tier', fontweight='bold')
    ax4.set_title('Player Skill × Matchup Quality Interaction', fontweight='bold')

plt.suptitle('Matchup Quality Score Analysis', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(VIZ_DIR / 'matchup_quality_distribution.png', dpi=300, bbox_inches='tight')
print(f"Saved visualization to: {VIZ_DIR / 'matchup_quality_distribution.png'}")
plt.close()

# ============================================================================
# PART 10: Save Matchup Quality Scores
# ============================================================================

print("\n" + "="*80)
print("SAVING MATCHUP QUALITY SCORES")
print("="*80)

# Save comprehensive matchup data
matchup_output = plyr_rec_games[[
    'plyr_id', 'game_id', 'season', 'week', 'plyr_pos', 'opponent_id',
    'plyr_gm_rec_yds', 'plyr_gm_rec_tgt',
    'overall_defense_score', 'pass_defense_score', 'pressure_score',
    'position_defense_pctile', 'player_avg_yards', 'player_skill_tier',
    'matchup_score_v1', 'matchup_tier_v1',
    'matchup_score_v2', 'matchup_tier_v2'
]].copy()

matchup_output.to_csv(PRED_DIR / 'matchup_quality_scores.csv', index=False)
print(f"Saved matchup scores to: {PRED_DIR / 'matchup_quality_scores.csv'}")

# Save tier performance summary
tier_summary = []
for tier in tier_order:
    tier_data = plyr_rec_games[plyr_rec_games['matchup_tier_v2'] == tier]
    if len(tier_data) > 0:
        tier_summary.append({
            'Matchup_Tier': tier,
            'Count': len(tier_data),
            'Avg_Yards': tier_data['plyr_gm_rec_yds'].mean(),
            'Median_Yards': tier_data['plyr_gm_rec_yds'].median(),
            'Std_Yards': tier_data['plyr_gm_rec_yds'].std(),
            'Min_Score': tier_data['matchup_score_v2'].min(),
            'Max_Score': tier_data['matchup_score_v2'].max()
        })

tier_summary_df = pd.DataFrame(tier_summary)
tier_summary_df.to_csv(PRED_DIR / 'matchup_tier_performance_summary.csv', index=False)
print(f"Saved tier summary to: {PRED_DIR / 'matchup_tier_performance_summary.csv'}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80)

print("\nMatchup Score Development:")
print("-" * 80)
print(f"v1.0 (Simple Weighted) - Correlation: r = {r_v1:.3f}")
if 'cv_scores' in locals():
    print(f"v2.0 (Optimized) - Cross-Val R²: {cv_scores.mean():.3f}")

print("\nTier Performance:")
print("-" * 80)
for tier in tier_order:
    tier_data = plyr_rec_games[plyr_rec_games['matchup_tier_v2'] == tier]
    if len(tier_data) > 0:
        print(f"{tier:20s}: {tier_data['plyr_gm_rec_yds'].mean():6.1f} yards (N={len(tier_data):,})")

print("\nKey Insights:")
print("-" * 80)
elite_avg = plyr_rec_games[plyr_rec_games['matchup_tier_v2'] == 'Elite Matchup']['plyr_gm_rec_yds'].mean()
avoid_avg = plyr_rec_games[plyr_rec_games['matchup_tier_v2'] == 'Avoid']['plyr_gm_rec_yds'].mean()

if not np.isnan(elite_avg) and not np.isnan(avoid_avg):
    improvement = elite_avg - avoid_avg
    pct_improvement = (improvement / avoid_avg) * 100
    print(f"Elite matchups average {improvement:.1f} more yards than Avoid matchups")
    print(f"That's a {pct_improvement:.1f}% improvement")

print("\n" + "="*80)
print("MATCHUP QUALITY SCORE DEVELOPMENT COMPLETE")
print("="*80)
print(f"\nOutputs saved to:")
print(f"  - {PRED_DIR / 'matchup_quality_scores.csv'}")
print(f"  - {PRED_DIR / 'matchup_tier_performance_summary.csv'}")
print(f"  - {VIZ_DIR / 'matchup_quality_distribution.png'}")
print("="*80)
