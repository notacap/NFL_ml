"""
Game Script & Situational Analysis for NFL Receiver Production
===============================================================

Objective: Understand how game flow, home/away splits, Vegas lines, and
score differentials affect passing volume and receiver production.

Analysis includes:
1. Home vs Away receiver production
2. Score differential impact (leading/trailing)
3. Vegas line integration (spread, over/under, implied totals)
4. Division game effects
5. Game script scenarios (blowouts, close games)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu
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
print("GAME SCRIPT & SITUATIONAL ANALYSIS")
print("=" * 80)

# ============================================================================
# SECTION 1: Load Data
# ============================================================================
print("\n[1/7] Loading data...")

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

# Load datasets
game_info = load_parquet_dataset(BASE_PATH / 'gm_info' / 'nfl_game')
game_betting = load_parquet_dataset(BASE_PATH / 'gm_info' / 'nfl_game_info')
plyr_gm_rec = pd.read_parquet(BASE_PATH / 'plyr_gm' / 'plyr_gm_rec')
tm_gm_stats = load_parquet_dataset(BASE_PATH / 'tm_gm' / 'tm_gm_stats')
team_info = pd.read_parquet(BASE_PATH / 'nfl_team.parquet')

print(f"  - Game info: {game_info.shape}")
print(f"  - Game betting lines: {game_betting.shape}")
print(f"  - Player receiving: {plyr_gm_rec.shape}")
print(f"  - Team game stats: {tm_gm_stats.shape}")

# ============================================================================
# SECTION 2: Prepare Game-Level Data
# ============================================================================
print("\n[2/7] Preparing game-level data...")

# Aggregate receiver production by team-game
rec_production = plyr_gm_rec.groupby(['team_id', 'game_id', 'season_id', 'week_id']).agg({
    'plyr_gm_rec_yds': 'sum',
    'plyr_gm_rec_tgt': 'sum',
    'plyr_gm_rec': 'sum'
}).reset_index()
rec_production.columns = ['team_id', 'game_id', 'season_id', 'week_id',
                          'total_rec_yds', 'total_targets', 'total_receptions']

# Create home and away records from game_info
# Home team records
home_games = game_info[['game_id', 'season_id', 'week_id', 'home_team_id', 'away_team_id',
                        'home_team_score', 'away_team_score']].copy()
home_games['team_id'] = home_games['home_team_id']
home_games['opponent_id'] = home_games['away_team_id']
home_games['is_home'] = 1
home_games['score'] = home_games['home_team_score']
home_games['opp_score'] = home_games['away_team_score']

# Away team records
away_games = game_info[['game_id', 'season_id', 'week_id', 'home_team_id', 'away_team_id',
                        'home_team_score', 'away_team_score']].copy()
away_games['team_id'] = away_games['away_team_id']
away_games['opponent_id'] = away_games['home_team_id']
away_games['is_home'] = 0
away_games['score'] = away_games['away_team_score']
away_games['opp_score'] = away_games['home_team_score']

# Combine
all_games = pd.concat([
    home_games[['game_id', 'season_id', 'week_id', 'team_id', 'opponent_id',
               'is_home', 'score', 'opp_score']],
    away_games[['game_id', 'season_id', 'week_id', 'team_id', 'opponent_id',
               'is_home', 'score', 'opp_score']]
], ignore_index=True)

# Calculate score differential
all_games['score_diff'] = all_games['score'] - all_games['opp_score']
all_games['is_leading'] = (all_games['score_diff'] > 0).astype(int)
all_games['is_trailing'] = (all_games['score_diff'] < 0).astype(int)
all_games['is_tied'] = (all_games['score_diff'] == 0).astype(int)

# Categorize game scripts
all_games['game_script'] = 'close'
all_games.loc[all_games['score_diff'] >= 14, 'game_script'] = 'blowout_win'
all_games.loc[all_games['score_diff'] <= -14, 'game_script'] = 'blowout_loss'

print(f"  - Combined game records: {all_games.shape}")

# Merge with receiver production
analysis_df = all_games.merge(rec_production,
                               on=['team_id', 'game_id', 'season_id', 'week_id'],
                               how='inner')

# Add team names
analysis_df = analysis_df.merge(team_info[['team_id', 'team_name', 'abrv', 'division']],
                                on='team_id', how='left')
analysis_df = analysis_df.merge(team_info[['team_id', 'division']].rename(
    columns={'team_id': 'opponent_id', 'division': 'opp_division'}),
    on='opponent_id', how='left')

# Check if division game
analysis_df['is_division_game'] = (analysis_df['division'] == analysis_df['opp_division']).astype(int)

# Merge with betting lines
if not game_betting.empty:
    analysis_df = analysis_df.merge(game_betting[['game_id', 'vegas_line_tm_id', 'vegas_line',
                                                   'over_under_line']],
                                    on='game_id', how='left')

    # Determine if team was favored
    analysis_df['is_favorite'] = (analysis_df['team_id'] == analysis_df['vegas_line_tm_id']).astype(int)
    analysis_df['spread_value'] = analysis_df.apply(
        lambda row: row['vegas_line'] if row['is_favorite'] else -row['vegas_line']
        if pd.notna(row['vegas_line']) else np.nan,
        axis=1
    )

    # Calculate implied totals (simple approximation)
    analysis_df['implied_total'] = np.where(
        pd.notna(analysis_df['over_under_line']),
        (analysis_df['over_under_line'] / 2) + (analysis_df['spread_value'] / 2),
        np.nan
    )

print(f"  - Final analysis dataset: {analysis_df.shape}")

# ============================================================================
# SECTION 3: Home vs Away Analysis
# ============================================================================
print("\n[3/7] Analyzing home vs away splits...")

home_away_stats = analysis_df.groupby('is_home').agg({
    'total_rec_yds': ['mean', 'std', 'median', 'count'],
    'total_targets': 'mean',
    'total_receptions': 'mean'
}).round(2)

print("\nHome vs Away Receiver Production:")
print(home_away_stats)

# Statistical test
home_data = analysis_df[analysis_df['is_home'] == 1]['total_rec_yds'].dropna()
away_data = analysis_df[analysis_df['is_home'] == 0]['total_rec_yds'].dropna()

t_stat, p_value = ttest_ind(home_data, away_data)
effect_size = (home_data.mean() - away_data.mean()) / np.sqrt(
    ((len(home_data) - 1) * home_data.std()**2 + (len(away_data) - 1) * away_data.std()**2) /
    (len(home_data) + len(away_data) - 2)
)

print(f"\nStatistical Test (Home vs Away):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.6f}")
print(f"  Cohen's d (effect size): {effect_size:.4f}")
print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")

# ============================================================================
# SECTION 4: Score Differential Impact
# ============================================================================
print("\n[4/7] Analyzing score differential impact...")

# Leading vs Trailing
leading_trailing_stats = analysis_df.groupby(['is_leading', 'is_trailing']).agg({
    'total_rec_yds': ['mean', 'std', 'count'],
    'total_targets': 'mean'
}).round(2)

print("\nLeading vs Trailing Team Receiver Production:")
print(leading_trailing_stats)

# Statistical test
leading_data = analysis_df[analysis_df['is_leading'] == 1]['total_rec_yds'].dropna()
trailing_data = analysis_df[analysis_df['is_trailing'] == 1]['total_rec_yds'].dropna()

t_stat2, p_value2 = ttest_ind(trailing_data, leading_data)
print(f"\nStatistical Test (Trailing vs Leading):")
print(f"  t-statistic: {t_stat2:.4f}")
print(f"  p-value: {p_value2:.6f}")
print(f"  Trailing boost: {trailing_data.mean() - leading_data.mean():.2f} yards")
print(f"  Significant: {'YES' if p_value2 < 0.05 else 'NO'}")

# Game script categories
game_script_stats = analysis_df.groupby('game_script').agg({
    'total_rec_yds': ['mean', 'std', 'count'],
    'total_targets': 'mean'
}).round(2)

print("\nGame Script Categories:")
print(game_script_stats)

# ============================================================================
# SECTION 5: Vegas Lines Analysis
# ============================================================================
print("\n[5/7] Analyzing Vegas lines predictive power...")

if 'over_under_line' in analysis_df.columns:
    # Filter for games with betting data
    betting_df = analysis_df[analysis_df['over_under_line'].notna()].copy()

    print(f"\nGames with betting data: {len(betting_df)}")

    # Correlation: over/under with actual receiving yards
    corr_ou = betting_df[['over_under_line', 'total_rec_yds']].corr().iloc[0, 1]
    print(f"  - Over/Under line correlation with receiver yards: {corr_ou:.4f}")

    # Correlation: spread with receiving yards
    if 'spread_value' in betting_df.columns:
        corr_spread = betting_df[['spread_value', 'total_rec_yds']].dropna().corr().iloc[0, 1]
        print(f"  - Spread correlation with receiver yards: {corr_spread:.4f}")

    # Correlation: implied total with receiving yards
    if 'implied_total' in betting_df.columns:
        corr_implied = betting_df[['implied_total', 'total_rec_yds']].dropna().corr().iloc[0, 1]
        print(f"  - Implied total correlation with receiver yards: {corr_implied:.4f}")

    # Favorite vs Underdog
    if 'is_favorite' in betting_df.columns:
        fav_underdog_stats = betting_df.groupby('is_favorite').agg({
            'total_rec_yds': ['mean', 'std', 'count']
        }).round(2)
        print("\nFavorite vs Underdog Receiver Production:")
        print(fav_underdog_stats)

# ============================================================================
# SECTION 6: Division Games
# ============================================================================
print("\n[6/7] Analyzing division game effects...")

division_stats = analysis_df.groupby('is_division_game').agg({
    'total_rec_yds': ['mean', 'std', 'count'],
    'total_targets': 'mean'
}).round(2)

print("\nDivision vs Non-Division Games:")
print(division_stats)

# Statistical test
div_data = analysis_df[analysis_df['is_division_game'] == 1]['total_rec_yds'].dropna()
non_div_data = analysis_df[analysis_df['is_division_game'] == 0]['total_rec_yds'].dropna()

if len(div_data) > 0 and len(non_div_data) > 0:
    t_stat3, p_value3 = ttest_ind(div_data, non_div_data)
    print(f"\nStatistical Test (Division vs Non-Division):")
    print(f"  t-statistic: {t_stat3:.4f}")
    print(f"  p-value: {p_value3:.6f}")
    print(f"  Difference: {div_data.mean() - non_div_data.mean():.2f} yards")
    print(f"  Significant: {'YES' if p_value3 < 0.05 else 'NO'}")

# ============================================================================
# SECTION 7: Create Visualizations
# ============================================================================
print("\n[7/7] Creating visualizations...")

# Figure 1: Home vs Away Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Home vs Away Receiver Production', fontsize=16, fontweight='bold')

# Box plot
ax1 = axes[0]
home_away_data = [home_data, away_data]
bp = ax1.boxplot(home_away_data, labels=['Home', 'Away'], patch_artist=True,
                 medianprops=dict(color='red', linewidth=2))
for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
    patch.set_facecolor(color)
ax1.set_ylabel('Total Receiving Yards', fontsize=12)
ax1.set_title(f'Distribution Comparison\n(p = {p_value:.4f})', fontsize=12)
ax1.grid(True, alpha=0.3, axis='y')

# Bar plot with error bars
ax2 = axes[1]
locations = ['Home', 'Away']
means = [home_data.mean(), away_data.mean()]
stds = [home_data.std(), away_data.std()]
bars = ax2.bar(locations, means, yerr=stds, capsize=5,
               color=['lightblue', 'lightcoral'], alpha=0.7, edgecolor='black')
ax2.set_ylabel('Average Receiving Yards', fontsize=12)
ax2.set_title(f'Mean Comparison (Effect Size: d={effect_size:.3f})', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (loc, mean) in enumerate(zip(locations, means)):
    ax2.text(i, mean + stds[i] + 5, f'{mean:.1f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_VIZ / 'home_away_comparison.png', dpi=300, bbox_inches='tight')
print(f"  - Saved: home_away_comparison.png")
plt.close()

# Figure 2: Game Script Scenarios
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Game Script Impact on Receiver Production', fontsize=16, fontweight='bold')

# Plot 1: Leading vs Trailing
ax1 = axes[0, 0]
game_situation_data = [
    analysis_df[analysis_df['is_leading'] == 1]['total_rec_yds'].dropna(),
    analysis_df[analysis_df['is_tied'] == 1]['total_rec_yds'].dropna(),
    analysis_df[analysis_df['is_trailing'] == 1]['total_rec_yds'].dropna()
]
bp1 = ax1.boxplot(game_situation_data, labels=['Leading', 'Tied', 'Trailing'],
                  patch_artist=True, medianprops=dict(color='red', linewidth=2))
colors1 = ['lightgreen', 'lightgray', 'lightcoral']
for patch, color in zip(bp1['boxes'], colors1):
    patch.set_facecolor(color)
ax1.set_ylabel('Total Receiving Yards', fontsize=11)
ax1.set_title('Game Situation Impact', fontsize=12)
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Blowouts vs Close Games
ax2 = axes[0, 1]
script_categories = analysis_df['game_script'].unique()
script_data = [analysis_df[analysis_df['game_script'] == cat]['total_rec_yds'].dropna()
               for cat in script_categories]
bp2 = ax2.boxplot(script_data, labels=script_categories, patch_artist=True,
                  medianprops=dict(color='red', linewidth=2))
for patch in bp2['boxes']:
    patch.set_facecolor('lightblue')
ax2.set_ylabel('Total Receiving Yards', fontsize=11)
ax2.set_title('Game Script Categories', fontsize=12)
ax2.set_xticklabels(script_categories, rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Score Differential Bins
ax3 = axes[1, 0]
analysis_df['score_diff_bin'] = pd.cut(analysis_df['score_diff'],
                                        bins=[-100, -14, -7, 0, 7, 14, 100],
                                        labels=['<-14', '-14 to -7', '-7 to 0',
                                               '0 to 7', '7 to 14', '>14'])
score_bin_stats = analysis_df.groupby('score_diff_bin')['total_rec_yds'].mean()
ax3.bar(range(len(score_bin_stats)), score_bin_stats.values, color='steelblue', alpha=0.7)
ax3.set_xticks(range(len(score_bin_stats)))
ax3.set_xticklabels(score_bin_stats.index, rotation=45, ha='right')
ax3.set_xlabel('Score Differential', fontsize=11)
ax3.set_ylabel('Average Receiving Yards', fontsize=11)
ax3.set_title('Receiver Production by Score Differential', fontsize=12)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Division vs Non-Division
ax4 = axes[1, 1]
div_labels = ['Non-Division', 'Division']
div_means = [non_div_data.mean(), div_data.mean()]
div_stds = [non_div_data.std(), div_data.std()]
bars4 = ax4.bar(div_labels, div_means, yerr=div_stds, capsize=5,
               color=['lightblue', 'lightcoral'], alpha=0.7, edgecolor='black')
ax4.set_ylabel('Average Receiving Yards', fontsize=11)
ax4.set_title('Division Game Effect', fontsize=12)
ax4.grid(True, alpha=0.3, axis='y')
for i, (label, mean) in enumerate(zip(div_labels, div_means)):
    ax4.text(i, mean + div_stds[i] + 5, f'{mean:.1f}', ha='center',
            fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_VIZ / 'game_script_scenarios.png', dpi=300, bbox_inches='tight')
print(f"  - Saved: game_script_scenarios.png")
plt.close()

# Figure 3: Vegas Line Correlations (if available)
if 'over_under_line' in analysis_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Vegas Betting Lines vs Receiver Production', fontsize=16, fontweight='bold')

    # Over/Under correlation
    ax1 = axes[0]
    valid_ou = betting_df[['over_under_line', 'total_rec_yds']].dropna()
    ax1.scatter(valid_ou['over_under_line'], valid_ou['total_rec_yds'],
               alpha=0.4, s=30)
    z = np.polyfit(valid_ou['over_under_line'], valid_ou['total_rec_yds'], 1)
    p = np.poly1d(z)
    ax1.plot(valid_ou['over_under_line'].sort_values(),
            p(valid_ou['over_under_line'].sort_values()),
            "r--", linewidth=2, label='Trend Line')
    ax1.set_xlabel('Over/Under Line', fontsize=11)
    ax1.set_ylabel('Total Receiving Yards', fontsize=11)
    ax1.set_title(f'O/U Line vs Receiver Production\n(r = {corr_ou:.3f})', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Implied Total correlation (if available)
    ax2 = axes[1]
    if 'implied_total' in betting_df.columns:
        valid_implied = betting_df[['implied_total', 'total_rec_yds']].dropna()
        ax2.scatter(valid_implied['implied_total'], valid_implied['total_rec_yds'],
                   alpha=0.4, s=30, color='orange')
        z2 = np.polyfit(valid_implied['implied_total'], valid_implied['total_rec_yds'], 1)
        p2 = np.poly1d(z2)
        ax2.plot(valid_implied['implied_total'].sort_values(),
                p2(valid_implied['implied_total'].sort_values()),
                "r--", linewidth=2, label='Trend Line')
        ax2.set_xlabel('Implied Team Total', fontsize=11)
        ax2.set_ylabel('Total Receiving Yards', fontsize=11)
        ax2.set_title(f'Implied Total vs Receiver Production\n(r = {corr_implied:.3f})',
                     fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_VIZ / 'vegas_total_correlation.png', dpi=300, bbox_inches='tight')
    print(f"  - Saved: vegas_total_correlation.png")
    plt.close()

# ============================================================================
# SECTION 8: Save Game Context Effects Summary
# ============================================================================
print("\nSaving game context effects summary...")

context_effects = pd.DataFrame([
    {
        'context_variable': 'Home Field Advantage',
        'effect_size_yards': home_data.mean() - away_data.mean(),
        'effect_size_cohen_d': effect_size,
        'p_value': p_value,
        'significant': 'Yes' if p_value < 0.05 else 'No',
        'sample_size': len(home_data) + len(away_data)
    },
    {
        'context_variable': 'Trailing vs Leading',
        'effect_size_yards': trailing_data.mean() - leading_data.mean(),
        'effect_size_cohen_d': (trailing_data.mean() - leading_data.mean()) / np.sqrt(
            ((len(trailing_data)-1)*trailing_data.std()**2 + (len(leading_data)-1)*leading_data.std()**2) /
            (len(trailing_data) + len(leading_data) - 2)
        ),
        'p_value': p_value2,
        'significant': 'Yes' if p_value2 < 0.05 else 'No',
        'sample_size': len(trailing_data) + len(leading_data)
    },
    {
        'context_variable': 'Division Game Effect',
        'effect_size_yards': div_data.mean() - non_div_data.mean() if len(div_data) > 0 else np.nan,
        'effect_size_cohen_d': (div_data.mean() - non_div_data.mean()) / np.sqrt(
            ((len(div_data)-1)*div_data.std()**2 + (len(non_div_data)-1)*non_div_data.std()**2) /
            (len(div_data) + len(non_div_data) - 2)
        ) if len(div_data) > 0 else np.nan,
        'p_value': p_value3 if len(div_data) > 0 else np.nan,
        'significant': 'Yes' if (len(div_data) > 0 and p_value3 < 0.05) else 'No',
        'sample_size': len(div_data) + len(non_div_data)
    }
])

context_effects.to_csv(OUTPUT_DATA / 'game_context_effects.csv', index=False)
print(f"  - Saved: game_context_effects.csv")

print("\n" + "=" * 80)
print("KEY FINDINGS:")
print("=" * 80)
print(f"\n1. HOME FIELD ADVANTAGE:")
print(f"   - Home avg: {home_data.mean():.1f} yards")
print(f"   - Away avg: {away_data.mean():.1f} yards")
print(f"   - Difference: {home_data.mean() - away_data.mean():+.1f} yards")
print(f"   - Effect size (Cohen's d): {effect_size:.3f}")
print(f"   - Statistical significance: {'YES (p < 0.05)' if p_value < 0.05 else 'NO'}")

print(f"\n2. GAME SCRIPT (Trailing Teams):")
print(f"   - Leading team avg: {leading_data.mean():.1f} yards")
print(f"   - Trailing team avg: {trailing_data.mean():.1f} yards")
print(f"   - Trailing boost: {trailing_data.mean() - leading_data.mean():+.1f} yards")
print(f"   - Statistical significance: {'YES (p < 0.05)' if p_value2 < 0.05 else 'NO'}")

if 'over_under_line' in analysis_df.columns and not betting_df.empty:
    print(f"\n3. VEGAS LINES PREDICTIVE POWER:")
    print(f"   - Over/Under correlation: r = {corr_ou:.3f}")
    if 'spread_value' in betting_df.columns:
        print(f"   - Spread correlation: r = {corr_spread:.3f}")
    if 'implied_total' in betting_df.columns:
        print(f"   - Implied total correlation: r = {corr_implied:.3f}")
    print(f"   => Vegas lines {'ARE' if abs(corr_ou) > 0.2 else 'are NOT'} strong predictors")

print(f"\n4. DIVISION GAMES:")
if len(div_data) > 0 and len(non_div_data) > 0:
    print(f"   - Division game avg: {div_data.mean():.1f} yards")
    print(f"   - Non-division avg: {non_div_data.mean():.1f} yards")
    print(f"   - Difference: {div_data.mean() - non_div_data.mean():+.1f} yards")
    print(f"   - Statistical significance: {'YES (p < 0.05)' if p_value3 < 0.05 else 'NO'}")
else:
    print(f"   - Insufficient data for comparison")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)
