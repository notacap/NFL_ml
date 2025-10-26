"""
Historical Matchup Analysis
============================
Analyzes whether past performance against specific opponents predicts future outcomes.
Implements Bayesian approach for handling small sample sizes.

Objectives:
1. Assess predictive value of historical matchup data
2. Handle small sample size challenges
3. Test coaching/scheme familiarity effects
4. Develop Bayesian priors for limited data scenarios
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
print("HISTORICAL MATCHUP ANALYSIS")
print("="*80)

# ============================================================================
# PART 1: Load and Prepare Data
# ============================================================================

print("\n1. Loading data...")

# Load player receiving games
plyr_rec = pd.read_parquet(DATA_DIR / "plyr_gm" / "plyr_gm_rec")
games = pd.read_parquet(DATA_DIR / "gm_info" / "nfl_game")
plyr_info = pd.read_parquet(DATA_DIR / "players" / "plyr")

print(f"   - Player receiving games: {plyr_rec.shape}")
print(f"   - Games: {games.shape}")

# Merge player games with game info
plyr_rec_games = plyr_rec.merge(
    games[['game_id', 'home_team_id', 'away_team_id', 'season', 'week', 'game_date']],
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

# Sort by date
plyr_rec_games['game_date'] = pd.to_datetime(plyr_rec_games['game_date'])
plyr_rec_games = plyr_rec_games.sort_values(['plyr_id', 'game_date'])

print(f"   - Built dataset with {plyr_rec_games.shape[0]} player-games")

# ============================================================================
# PART 2: Calculate Historical Matchup Performance
# ============================================================================

print("\n" + "="*80)
print("CALCULATING HISTORICAL MATCHUP PERFORMANCE")
print("="*80)

# For each player-game, calculate their historical performance vs that opponent
historical_matchup_data = []

for idx, row in plyr_rec_games.iterrows():
    player_id = row['plyr_id']
    opponent_id = row['opponent_id']
    game_date = row['game_date']

    # Get all previous games for this player vs this opponent
    historical_games = plyr_rec_games[
        (plyr_rec_games['plyr_id'] == player_id) &
        (plyr_rec_games['opponent_id'] == opponent_id) &
        (plyr_rec_games['game_date'] < game_date)
    ].copy()

    # Calculate historical stats
    if len(historical_games) > 0:
        # Overall history
        hist_mean = historical_games['plyr_gm_rec_yds'].mean()
        hist_median = historical_games['plyr_gm_rec_yds'].median()
        hist_std = historical_games['plyr_gm_rec_yds'].std()
        hist_count = len(historical_games)
        hist_max = historical_games['plyr_gm_rec_yds'].max()
        hist_min = historical_games['plyr_gm_rec_yds'].min()

        # Last 3 games vs opponent
        last_3 = historical_games.tail(3)
        hist_L3_mean = last_3['plyr_gm_rec_yds'].mean() if len(last_3) >= 3 else np.nan
        hist_L3_count = len(last_3) if len(last_3) >= 3 else 0

        # Last game vs opponent
        hist_last_game = historical_games.iloc[-1]['plyr_gm_rec_yds']

        # Days since last matchup
        days_since_last = (game_date - historical_games.iloc[-1]['game_date']).days
    else:
        hist_mean = np.nan
        hist_median = np.nan
        hist_std = np.nan
        hist_count = 0
        hist_max = np.nan
        hist_min = np.nan
        hist_L3_mean = np.nan
        hist_L3_count = 0
        hist_last_game = np.nan
        days_since_last = np.nan

    historical_matchup_data.append({
        'plyr_id': player_id,
        'game_id': row['game_id'],
        'season': row['season'],
        'week': row['week'],
        'opponent_id': opponent_id,
        'actual_yards': row['plyr_gm_rec_yds'],
        'hist_vs_opp_mean': hist_mean,
        'hist_vs_opp_median': hist_median,
        'hist_vs_opp_std': hist_std,
        'hist_vs_opp_count': hist_count,
        'hist_vs_opp_max': hist_max,
        'hist_vs_opp_min': hist_min,
        'hist_vs_opp_L3_mean': hist_L3_mean,
        'hist_vs_opp_L3_count': hist_L3_count,
        'hist_vs_opp_last_game': hist_last_game,
        'days_since_last_matchup': days_since_last
    })

    if idx % 1000 == 0:
        print(f"   Processed {idx:,} / {len(plyr_rec_games):,} games...", end='\r')

historical_df = pd.DataFrame(historical_matchup_data)
print(f"\n   Completed historical matchup calculations")

# ============================================================================
# PART 3: Predictive Value Assessment
# ============================================================================

print("\n" + "="*80)
print("PREDICTIVE VALUE ASSESSMENT")
print("="*80)

# Analyze correlation between historical performance and current game
print("\nCorrelations between historical matchup stats and actual performance:")
print("-" * 80)

predictors = {
    'hist_vs_opp_mean': 'Career Avg vs Opponent',
    'hist_vs_opp_median': 'Career Median vs Opponent',
    'hist_vs_opp_L3_mean': 'Last 3 vs Opponent',
    'hist_vs_opp_last_game': 'Last Game vs Opponent'
}

correlation_results = []

for pred_col, pred_name in predictors.items():
    valid_data = historical_df[['actual_yards', pred_col]].dropna()

    if len(valid_data) > 30:
        r, p = stats.pearsonr(valid_data['actual_yards'], valid_data[pred_col])

        print(f"{pred_name:35s}: r = {r:6.3f}, p = {p:.4f}, N = {len(valid_data):,}")

        correlation_results.append({
            'Predictor': pred_name,
            'Correlation': r,
            'P_value': p,
            'N': len(valid_data)
        })

corr_df = pd.DataFrame(correlation_results)

# Analyze by sample size
print("\n\nPredictive Power by Number of Historical Games:")
print("-" * 80)

for min_games in [1, 2, 3, 5]:
    subset = historical_df[historical_df['hist_vs_opp_count'] >= min_games]
    valid_data = subset[['actual_yards', 'hist_vs_opp_mean']].dropna()

    if len(valid_data) > 30:
        r, p = stats.pearsonr(valid_data['actual_yards'], valid_data['hist_vs_opp_mean'])
        print(f"At least {min_games} games: r = {r:6.3f}, p = {p:.4f}, N = {len(valid_data):,}")

# ============================================================================
# PART 4: Bayesian Prior Development
# ============================================================================

print("\n" + "="*80)
print("BAYESIAN PRIOR DEVELOPMENT")
print("="*80)

# Calculate player overall average (prior)
player_priors = plyr_rec_games.groupby('plyr_id')['plyr_gm_rec_yds'].agg(['mean', 'count', 'std']).reset_index()
player_priors.columns = ['plyr_id', 'player_overall_mean', 'player_total_games', 'player_overall_std']

# Merge with historical matchup data
historical_df = historical_df.merge(player_priors, on='plyr_id', how='left')

# Bayesian adjustment: weight historical matchup data by sample size
# Formula: bayesian_estimate = (n * hist_mean + k * overall_mean) / (n + k)
# where k is a "shrinkage" parameter (higher k = more regression to overall mean)

def bayesian_estimate(row, k=5):
    """
    Calculate Bayesian estimate combining historical matchup and overall performance.

    k: shrinkage parameter (higher = more weight to prior/overall mean)
    """
    if pd.isna(row['hist_vs_opp_mean']) or row['hist_vs_opp_count'] == 0:
        return row['player_overall_mean']

    n = row['hist_vs_opp_count']
    hist_mean = row['hist_vs_opp_mean']
    overall_mean = row['player_overall_mean']

    bayesian_est = (n * hist_mean + k * overall_mean) / (n + k)
    return bayesian_est

# Test different shrinkage parameters
shrinkage_params = [1, 3, 5, 10, 20]
bayesian_results = []

print("\nBayesian Estimate Performance (different shrinkage parameters):")
print("-" * 80)

for k in shrinkage_params:
    historical_df[f'bayesian_est_k{k}'] = historical_df.apply(lambda row: bayesian_estimate(row, k), axis=1)

    valid_data = historical_df[['actual_yards', f'bayesian_est_k{k}']].dropna()

    if len(valid_data) > 30:
        r, p = stats.pearsonr(valid_data['actual_yards'], valid_data[f'bayesian_est_k{k}'])
        print(f"k = {k:2d}: r = {r:6.3f}, p = {p:.4f}, N = {len(valid_data):,}")

        bayesian_results.append({
            'Shrinkage_K': k,
            'Correlation': r,
            'P_value': p,
            'N': len(valid_data)
        })

bayesian_df = pd.DataFrame(bayesian_results)

# Find optimal k
if len(bayesian_df) > 0:
    optimal_k = bayesian_df.loc[bayesian_df['Correlation'].idxmax(), 'Shrinkage_K']
    print(f"\nOptimal shrinkage parameter: k = {optimal_k}")

# ============================================================================
# PART 5: Temporal Analysis
# ============================================================================

print("\n" + "="*80)
print("TEMPORAL ANALYSIS - Recency Effects")
print("="*80)

# Analyze how predictive value changes with time since last matchup
print("\nPredictive power by days since last matchup:")
print("-" * 80)

time_bins = [(0, 180), (180, 365), (365, 730), (730, 10000)]
time_labels = ['< 6 months', '6-12 months', '1-2 years', '> 2 years']

for (min_days, max_days), label in zip(time_bins, time_labels):
    subset = historical_df[
        (historical_df['days_since_last_matchup'] >= min_days) &
        (historical_df['days_since_last_matchup'] < max_days)
    ]

    valid_data = subset[['actual_yards', 'hist_vs_opp_last_game']].dropna()

    if len(valid_data) > 30:
        r, p = stats.pearsonr(valid_data['actual_yards'], valid_data['hist_vs_opp_last_game'])
        print(f"{label:15s}: r = {r:6.3f}, p = {p:.4f}, N = {len(valid_data):,}")

# ============================================================================
# PART 6: Sample Size Analysis
# ============================================================================

print("\n" + "="*80)
print("SAMPLE SIZE ANALYSIS")
print("="*80)

# Distribution of historical matchup sample sizes
print("\nDistribution of historical matchup counts:")
print("-" * 80)
sample_size_dist = historical_df[historical_df['hist_vs_opp_count'] > 0]['hist_vs_opp_count'].value_counts().sort_index()
print(sample_size_dist.head(10).to_string())

# Calculate percentage with at least N games
for n in [1, 2, 3, 5]:
    pct = (historical_df['hist_vs_opp_count'] >= n).sum() / len(historical_df) * 100
    print(f"At least {n} historical game(s): {pct:.1f}%")

# ============================================================================
# PART 7: Visualization - Historical Matchup Predictive Value
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Correlation by predictor
ax1 = axes[0, 0]
corr_df_sorted = corr_df.sort_values('Correlation', ascending=False)
colors = ['green' if x > 0.2 else 'orange' if x > 0.1 else 'red' for x in corr_df_sorted['Correlation']]
ax1.barh(range(len(corr_df_sorted)), corr_df_sorted['Correlation'], color=colors, edgecolor='black')
ax1.set_yticks(range(len(corr_df_sorted)))
ax1.set_yticklabels(corr_df_sorted['Predictor'])
ax1.set_xlabel('Correlation with Actual Yards', fontweight='bold')
ax1.set_title('Historical Matchup Predictors', fontweight='bold')
ax1.axvline(0, color='black', linewidth=0.5)
ax1.grid(axis='x', alpha=0.3)

# Add value labels
for i, v in enumerate(corr_df_sorted['Correlation']):
    ax1.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

# 2. Bayesian performance by shrinkage parameter
ax2 = axes[0, 1]
ax2.plot(bayesian_df['Shrinkage_K'], bayesian_df['Correlation'], marker='o', linewidth=2, markersize=8)
ax2.set_xlabel('Shrinkage Parameter (k)', fontweight='bold')
ax2.set_ylabel('Correlation', fontweight='bold')
ax2.set_title('Bayesian Estimate Performance', fontweight='bold')
ax2.grid(alpha=0.3)
if len(bayesian_df) > 0:
    ax2.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
    ax2.legend()

# 3. Sample size distribution
ax3 = axes[1, 0]
hist_counts = historical_df[historical_df['hist_vs_opp_count'] > 0]['hist_vs_opp_count']
ax3.hist(hist_counts, bins=range(0, min(hist_counts.max() + 2, 15)), edgecolor='black', alpha=0.7)
ax3.set_xlabel('Number of Historical Games vs Opponent', fontweight='bold')
ax3.set_ylabel('Frequency', fontweight='bold')
ax3.set_title('Distribution of Historical Matchup Sample Sizes', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# 4. Predictive power by sample size
ax4 = axes[1, 1]
sample_sizes = [1, 2, 3, 4, 5]
correlations = []
sample_counts = []

for n in sample_sizes:
    subset = historical_df[historical_df['hist_vs_opp_count'] >= n]
    valid_data = subset[['actual_yards', 'hist_vs_opp_mean']].dropna()

    if len(valid_data) > 30:
        r, _ = stats.pearsonr(valid_data['actual_yards'], valid_data['hist_vs_opp_mean'])
        correlations.append(r)
        sample_counts.append(len(valid_data))
    else:
        correlations.append(np.nan)
        sample_counts.append(0)

ax4_twin = ax4.twinx()
ax4.plot(sample_sizes, correlations, marker='o', color='blue', linewidth=2, markersize=8, label='Correlation')
ax4_twin.bar(sample_sizes, sample_counts, alpha=0.3, color='gray', label='Sample Count')

ax4.set_xlabel('Minimum Historical Games', fontweight='bold')
ax4.set_ylabel('Correlation', fontweight='bold', color='blue')
ax4_twin.set_ylabel('Sample Count', fontweight='bold', color='gray')
ax4.set_title('Predictive Power by Sample Size', fontweight='bold')
ax4.grid(alpha=0.3)
ax4.legend(loc='upper left')
ax4_twin.legend(loc='upper right')

plt.suptitle('Historical Matchup Analysis', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(VIZ_DIR / 'historical_matchup_analysis.png', dpi=300, bbox_inches='tight')
print(f"Saved visualization to: {VIZ_DIR / 'historical_matchup_analysis.png'}")
plt.close()

# ============================================================================
# PART 8: Save Results
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save historical matchup data with best Bayesian estimate
historical_output = historical_df[[
    'plyr_id', 'game_id', 'season', 'week', 'opponent_id', 'actual_yards',
    'hist_vs_opp_count', 'hist_vs_opp_mean', 'hist_vs_opp_median',
    'hist_vs_opp_L3_mean', 'hist_vs_opp_last_game',
    'player_overall_mean', f'bayesian_est_k{optimal_k if len(bayesian_df) > 0 else 5}',
    'days_since_last_matchup'
]].copy()

if len(bayesian_df) > 0:
    historical_output.rename(columns={f'bayesian_est_k{optimal_k}': 'bayesian_estimate'}, inplace=True)
else:
    historical_output.rename(columns={'bayesian_est_k5': 'bayesian_estimate'}, inplace=True)

historical_output.to_csv(PRED_DIR / 'historical_matchup_performance.csv', index=False)
print(f"Saved historical matchup data to: {PRED_DIR / 'historical_matchup_performance.csv'}")

# Save correlation summary
corr_df.to_csv(PRED_DIR / 'historical_matchup_correlations.csv', index=False)
print(f"Saved correlations to: {PRED_DIR / 'historical_matchup_correlations.csv'}")

# Save Bayesian results
bayesian_df.to_csv(PRED_DIR / 'bayesian_shrinkage_analysis.csv', index=False)
print(f"Saved Bayesian analysis to: {PRED_DIR / 'bayesian_shrinkage_analysis.csv'}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80)

print("\nHistorical Matchup Data Availability:")
print("-" * 80)
total_games = len(historical_df)
with_history = (historical_df['hist_vs_opp_count'] > 0).sum()
print(f"Total player-games: {total_games:,}")
print(f"With matchup history: {with_history:,} ({with_history/total_games*100:.1f}%)")
print(f"Without history: {total_games - with_history:,} ({(total_games-with_history)/total_games*100:.1f}%)")

print("\nPredictive Value:")
print("-" * 80)
if len(corr_df) > 0:
    best_predictor = corr_df.loc[corr_df['Correlation'].idxmax()]
    print(f"Best predictor: {best_predictor['Predictor']}")
    print(f"Correlation: r = {best_predictor['Correlation']:.3f}")
    print(f"Sample size: N = {best_predictor['N']:,}")

print("\nBayesian Approach:")
print("-" * 80)
if len(bayesian_df) > 0:
    optimal_result = bayesian_df[bayesian_df['Shrinkage_K'] == optimal_k].iloc[0]
    print(f"Optimal shrinkage: k = {optimal_k}")
    print(f"Correlation: r = {optimal_result['Correlation']:.3f}")
    print(f"Interpretation: Weight {optimal_k} prior observations equally with historical matchup data")

print("\nRecommendation:")
print("-" * 80)
if with_history / total_games < 0.5:
    print("With limited historical matchup data available (<50% of games),")
    print("Bayesian approach is RECOMMENDED to avoid overfitting to small samples.")
else:
    print("Historical matchup data available for majority of games.")
    print("Direct historical average can be used, with Bayesian adjustment for small samples.")

print("\n" + "="*80)
print("HISTORICAL MATCHUP ANALYSIS COMPLETE")
print("="*80)
print(f"\nOutputs saved to:")
print(f"  - {PRED_DIR / 'historical_matchup_performance.csv'}")
print(f"  - {PRED_DIR / 'historical_matchup_correlations.csv'}")
print(f"  - {PRED_DIR / 'bayesian_shrinkage_analysis.csv'}")
print(f"  - {VIZ_DIR / 'historical_matchup_analysis.png'}")
print("="*80)
