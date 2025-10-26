"""
Momentum & Trend Analysis
==========================
Tests for "hot hand" effects, momentum patterns, and trend-based features.

Tasks:
1. Calculate short-term momentum indicators
2. Identify trend directions (improving, declining, stable)
3. Detect breakout performances
4. Test hot hand hypothesis
5. Analyze cold streak recovery patterns
6. Measure mean reversion strength and timing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, linregress
import warnings
warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = Path(r"C:\Users\nocap\Desktop\code\NFL_ml")
DATA_DIR = BASE_DIR / "parquet_files" / "clean"
OUTPUT_DIR = Path(r"C:\Users\nocap\Desktop\code\NFL_ml\models\rf_rec_yds\outputs")
VIZ_DIR = OUTPUT_DIR / "visualizations"
PRED_DIR = OUTPUT_DIR / "predictions"

print("=" * 80)
print("MOMENTUM & TREND ANALYSIS")
print("=" * 80)


def load_data(seasons=[2022, 2023, 2024]):
    """Load player game receiving data"""
    print(f"\n[1/7] Loading data for seasons: {seasons}")

    dfs_rec = []
    for season in seasons:
        season_dir = DATA_DIR / "plyr_gm" / "plyr_gm_rec" / f"season={season}"
        if not season_dir.exists():
            continue

        for week_dir in sorted([d for d in season_dir.iterdir() if d.is_dir()]):
            week_num = int(week_dir.name.split('=')[1])
            parquet_file = week_dir / "data.parquet"
            if parquet_file.exists():
                df_week = pd.read_parquet(parquet_file)
                df_week['season'] = season
                df_week['week'] = week_num
                dfs_rec.append(df_week)

    df_rec = pd.concat(dfs_rec, ignore_index=True)

    # Load player info
    dfs_player = []
    for season in seasons:
        player_dir = DATA_DIR / "players" / "plyr" / f"season={season}"
        if player_dir.exists():
            parquet_files = list(player_dir.glob("*.parquet"))
            if parquet_files:
                df_player = pd.read_parquet(parquet_files[0])
                df_player['season'] = season
                dfs_player.append(df_player)

    if not dfs_player:
        df_players = pd.DataFrame(columns=['plyr_id', 'plyr_name', 'plyr_pos', 'season'])
    else:
        df_players = pd.concat(dfs_player, ignore_index=True)
        df_players = df_players[['plyr_id', 'plyr_name', 'plyr_pos', 'season']].drop_duplicates()

    # Merge
    df = df_rec.merge(
        df_players[['plyr_id', 'plyr_name', 'plyr_pos', 'season']],
        on=['plyr_id', 'season'],
        how='left'
    )

    # Filter to relevant positions
    df = df[df['plyr_pos'].isin(['WR', 'TE', 'RB'])].copy()

    # Sort by player and game
    df['game_order'] = df['season'] * 100 + df['week']
    df = df.sort_values(['plyr_id', 'game_order']).reset_index(drop=True)

    print(f"  Loaded {len(df):,} player-game records")
    print(f"  Unique players: {df['plyr_id'].nunique():,}")

    return df


def calculate_momentum_features(df):
    """Calculate momentum and trend indicators"""
    print(f"\n[2/7] Calculating momentum features")

    # Calculate rolling averages for comparison
    for window in [3, 5, 10]:
        df[f'rolling_avg_{window}'] = df.groupby('plyr_id')['plyr_gm_rec_yds'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )

    # Momentum score: last game vs rolling averages
    df['momentum_vs_3game'] = df['plyr_gm_rec_yds'] - df['rolling_avg_3']
    df['momentum_vs_5game'] = df['plyr_gm_rec_yds'] - df['rolling_avg_5']
    df['momentum_vs_10game'] = df['plyr_gm_rec_yds'] - df['rolling_avg_10']

    # Shift momentum scores to use as predictive features
    df['momentum_vs_3game_lag1'] = df.groupby('plyr_id')['momentum_vs_3game'].shift(1)
    df['momentum_vs_5game_lag1'] = df.groupby('plyr_id')['momentum_vs_5game'].shift(1)

    # Calculate trend (linear regression slope over last N games)
    def calculate_trend_slope(series, window=5):
        """Calculate linear regression slope for last N values"""
        if len(series) < 2:
            return np.nan

        x = np.arange(len(series))
        y = series.values

        # Remove NaN values
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan

        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 2:
            return np.nan

        slope, _, _, _, _ = linregress(x_clean, y_clean)
        return slope

    # Calculate trend for each game (looking back)
    for window in [3, 5]:
        df[f'trend_slope_{window}'] = df.groupby('plyr_id')['plyr_gm_rec_yds'].transform(
            lambda x: x.rolling(window=window, min_periods=2).apply(
                lambda vals: calculate_trend_slope(pd.Series(vals), window=len(vals))
            ).shift(1)
        )

    # Moving average crossovers
    df['ma_crossover_3_5'] = (df['rolling_avg_3'] > df['rolling_avg_5']).astype(int)

    # Calculate percentage change from last game
    df['pct_change_last_game'] = df.groupby('plyr_id')['plyr_gm_rec_yds'].pct_change()

    # Shift for prediction
    df['pct_change_last_game_lag1'] = df.groupby('plyr_id')['pct_change_last_game'].shift(1)

    print(f"  Created momentum and trend features")

    return df


def detect_breakouts(df):
    """Identify breakout performances"""
    print(f"\n[3/7] Detecting breakout performances")

    # Calculate season average for each player
    season_avg = df.groupby(['plyr_id', 'season'])['plyr_gm_rec_yds'].transform('mean')

    # Position average
    position_avg = df.groupby(['plyr_pos', 'season', 'week'])['plyr_gm_rec_yds'].transform('mean')

    # Breakout flags
    df['breakout_1.5x_season'] = (df['plyr_gm_rec_yds'] > season_avg * 1.5).astype(int)
    df['breakout_2x_season'] = (df['plyr_gm_rec_yds'] > season_avg * 2).astype(int)
    df['breakout_2x_position'] = (df['plyr_gm_rec_yds'] > position_avg * 2).astype(int)

    # Lag for prediction
    df['breakout_1.5x_season_lag1'] = df.groupby('plyr_id')['breakout_1.5x_season'].shift(1)
    df['breakout_2x_season_lag1'] = df.groupby('plyr_id')['breakout_2x_season'].shift(1)

    # Count consecutive games above average
    def count_consecutive_above_avg(series, avg_series):
        """Count consecutive games above average"""
        above_avg = (series > avg_series).astype(int)
        counts = []
        current_count = 0

        for val in above_avg:
            if val == 1:
                current_count += 1
            else:
                current_count = 0
            counts.append(current_count)

        return pd.Series(counts, index=series.index)

    df['consecutive_above_avg'] = df.groupby('plyr_id').apply(
        lambda x: count_consecutive_above_avg(x['plyr_gm_rec_yds'], season_avg.loc[x.index])
    ).reset_index(level=0, drop=True)

    df['consecutive_above_avg_lag1'] = df.groupby('plyr_id')['consecutive_above_avg'].shift(1)

    print(f"  Identified breakout performances")

    return df


def test_hot_hand_hypothesis(df):
    """Test if strong recent performance predicts continued strong performance"""
    print(f"\n[4/7] Testing hot hand hypothesis")

    # Create target variable (next game performance)
    df['next_game_yards'] = df.groupby('plyr_id')['plyr_gm_rec_yds'].shift(-1)

    # Define "hot" as above player's season average
    player_season_avg = df.groupby(['plyr_id', 'season'])['plyr_gm_rec_yds'].transform('mean')

    df['is_hot'] = (df['plyr_gm_rec_yds'] > player_season_avg).astype(int)
    df['is_cold'] = (df['plyr_gm_rec_yds'] < player_season_avg * 0.5).astype(int)

    # Count recent hot/cold games
    df['hot_games_last_3'] = df.groupby('plyr_id')['is_hot'].transform(
        lambda x: x.rolling(window=3, min_periods=1).sum().shift(1)
    )

    df['cold_games_last_3'] = df.groupby('plyr_id')['is_cold'].transform(
        lambda x: x.rolling(window=3, min_periods=1).sum().shift(1)
    )

    # Test correlation
    results = []

    # Overall correlation: momentum vs next game
    momentum_features = [
        'momentum_vs_3game_lag1',
        'momentum_vs_5game_lag1',
        'trend_slope_3',
        'trend_slope_5',
        'hot_games_last_3',
        'pct_change_last_game_lag1'
    ]

    for feature in momentum_features:
        if feature not in df.columns:
            continue

        valid_mask = df[feature].notna() & df['next_game_yards'].notna()

        if valid_mask.sum() < 30:
            continue

        corr, pval = pearsonr(
            df.loc[valid_mask, feature],
            df.loc[valid_mask, 'next_game_yards']
        )

        results.append({
            'feature': feature,
            'position': 'ALL',
            'correlation': corr,
            'p_value': pval,
            'sample_size': valid_mask.sum()
        })

        # Position-specific
        for position in ['WR', 'TE', 'RB']:
            pos_mask = (df['plyr_pos'] == position) & valid_mask

            if pos_mask.sum() < 30:
                continue

            corr, pval = pearsonr(
                df.loc[pos_mask, feature],
                df.loc[pos_mask, 'next_game_yards']
            )

            results.append({
                'feature': feature,
                'position': position,
                'correlation': corr,
                'p_value': pval,
                'sample_size': pos_mask.sum()
            })

    df_hot_hand = pd.DataFrame(results)

    print(f"  Calculated {len(df_hot_hand):,} correlation tests")

    return df, df_hot_hand


def analyze_cold_streak_recovery(df):
    """Analyze recovery patterns from cold streaks"""
    print(f"\n[5/7] Analyzing cold streak recovery")

    recovery_results = []

    # For each player-season
    for (player_id, season), df_ps in df.groupby(['plyr_id', 'season']):

        if len(df_ps) < 8:
            continue

        player_name = df_ps['plyr_name'].iloc[0]
        position = df_ps['plyr_pos'].iloc[0]
        yards = df_ps['plyr_gm_rec_yds'].values
        player_mean = yards.mean()

        # Define cold streak as 2+ consecutive games below 50% of mean
        cold_threshold = player_mean * 0.5
        in_cold_streak = False
        cold_streak_length = 0

        for i in range(len(yards) - 1):
            if yards[i] < cold_threshold:
                if not in_cold_streak:
                    in_cold_streak = True
                    cold_streak_length = 1
                else:
                    cold_streak_length += 1
            else:
                if in_cold_streak and cold_streak_length >= 2:
                    # Exiting cold streak - record recovery
                    recovery_yards = yards[i]
                    recovery_results.append({
                        'plyr_id': player_id,
                        'player_name': player_name,
                        'position': position,
                        'cold_streak_length': cold_streak_length,
                        'recovery_yards': recovery_yards,
                        'player_mean': player_mean,
                        'recovery_vs_mean': recovery_yards - player_mean
                    })

                in_cold_streak = False
                cold_streak_length = 0

    df_recovery = pd.DataFrame(recovery_results)

    if len(df_recovery) > 0:
        print(f"  Identified {len(df_recovery):,} cold streak recovery instances")
        print(f"\n  Recovery Statistics:")
        print(f"    Mean recovery yards: {df_recovery['recovery_yards'].mean():.1f}")
        print(f"    Mean vs player average: {df_recovery['recovery_vs_mean'].mean():.1f}")
    else:
        print("  No cold streak recoveries found")

    return df_recovery


def calculate_reversion_strength(df):
    """Measure mean reversion strength and timing"""
    print(f"\n[6/7] Calculating mean reversion strength")

    reversion_results = []

    for (player_id, season), df_ps in df.groupby(['plyr_id', 'season']):

        if len(df_ps) < 10:
            continue

        player_name = df_ps['plyr_name'].iloc[0]
        position = df_ps['plyr_pos'].iloc[0]
        yards = df_ps['plyr_gm_rec_yds'].values
        player_mean = yards.mean()

        # For each outlier game, track how long until reversion
        for i in range(len(yards) - 3):  # Need at least 3 games ahead
            deviation = yards[i] - player_mean

            # Check if outlier (>1 std from mean)
            if abs(deviation) > yards.std():

                # Track next 3 games
                reverted = False
                games_to_revert = None

                for j in range(1, min(4, len(yards) - i)):
                    next_deviation = abs(yards[i + j] - player_mean)

                    # Check if reverted (within 0.5 std of mean)
                    if next_deviation < yards.std() * 0.5:
                        reverted = True
                        games_to_revert = j
                        break

                reversion_results.append({
                    'plyr_id': player_id,
                    'player_name': player_name,
                    'position': position,
                    'outlier_yards': yards[i],
                    'deviation': deviation,
                    'player_mean': player_mean,
                    'reverted': reverted,
                    'games_to_revert': games_to_revert if reverted else np.nan
                })

    df_reversion = pd.DataFrame(reversion_results)

    if len(df_reversion) > 0:
        print(f"  Analyzed {len(df_reversion):,} outlier performances")
        print(f"    Reversion rate: {df_reversion['reverted'].mean():.1%}")
        print(f"    Mean games to revert: {df_reversion['games_to_revert'].mean():.2f}")
    else:
        print("  No outlier performances found")

    return df_reversion


def create_visualizations(df, df_hot_hand, df_recovery, df_reversion):
    """Create comprehensive visualizations"""
    print(f"\n[7/7] Creating visualizations")

    # 1. Momentum Effect Analysis
    print("  Creating momentum effect plot...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Top left: Momentum vs Next Game (scatter)
    valid_mask = df['momentum_vs_3game_lag1'].notna() & df['next_game_yards'].notna()
    df_valid = df[valid_mask].copy()

    # Bin momentum scores for clearer visualization
    df_valid['momentum_bin'] = pd.qcut(df_valid['momentum_vs_3game_lag1'], q=10, duplicates='drop')
    momentum_summary = df_valid.groupby('momentum_bin').agg({
        'next_game_yards': ['mean', 'std', 'count'],
        'momentum_vs_3game_lag1': 'mean'
    }).reset_index()

    axes[0, 0].errorbar(
        momentum_summary[('momentum_vs_3game_lag1', 'mean')],
        momentum_summary[('next_game_yards', 'mean')],
        yerr=momentum_summary[('next_game_yards', 'std')] / np.sqrt(momentum_summary[('next_game_yards', 'count')]),
        fmt='o-',
        capsize=5,
        capthick=2,
        markersize=8,
        color='steelblue',
        ecolor='gray',
        linewidth=2
    )

    axes[0, 0].axhline(y=df_valid['next_game_yards'].mean(), color='red', linestyle='--', label='Overall Average')
    axes[0, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    axes[0, 0].set_xlabel('Momentum Score (last game vs 3-game avg)', fontsize=12)
    axes[0, 0].set_ylabel('Next Game Yards', fontsize=12)
    axes[0, 0].set_title('Momentum Effect on Next Game Performance', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Top right: Hot Hand correlation heatmap
    df_heatmap = df_hot_hand[df_hot_hand['position'].isin(['WR', 'TE', 'RB'])].pivot_table(
        index='feature',
        columns='position',
        values='correlation'
    )

    sns.heatmap(
        df_heatmap,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=0,
        vmin=-0.2,
        vmax=0.2,
        ax=axes[0, 1],
        cbar_kws={'label': 'Correlation'}
    )

    axes[0, 1].set_title('Hot Hand Correlations by Position', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Position', fontsize=12)
    axes[0, 1].set_ylabel('Momentum Feature', fontsize=12)

    # Bottom left: Trend slope distribution
    for position, color in zip(['WR', 'TE', 'RB'], ['#FF6B6B', '#4ECDC4', '#45B7D1']):
        df_pos = df[df['plyr_pos'] == position]
        df_pos_valid = df_pos[df_pos['trend_slope_5'].notna()]

        axes[1, 0].hist(
            df_pos_valid['trend_slope_5'],
            bins=50,
            alpha=0.5,
            label=position,
            color=color,
            edgecolor='black'
        )

    axes[1, 0].axvline(x=0, color='black', linestyle='--', linewidth=2, label='No Trend')
    axes[1, 0].set_xlabel('5-Game Trend Slope', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Distribution of Performance Trends', fontsize=13, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Bottom right: Cold streak recovery
    if len(df_recovery) > 0:
        recovery_bins = df_recovery.groupby('cold_streak_length').agg({
            'recovery_vs_mean': ['mean', 'std', 'count']
        }).reset_index()

        axes[1, 1].bar(
            recovery_bins['cold_streak_length'],
            recovery_bins[('recovery_vs_mean', 'mean')],
            color='steelblue',
            alpha=0.7,
            edgecolor='black'
        )

        axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Player Mean')
        axes[1, 1].set_xlabel('Cold Streak Length (games)', fontsize=12)
        axes[1, 1].set_ylabel('Recovery Performance vs Mean', fontsize=12)
        axes[1, 1].set_title('Cold Streak Recovery Patterns', fontsize=13, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient cold streak data', ha='center', va='center', fontsize=14)
        axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'momentum_effect_analysis.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: {VIZ_DIR / 'momentum_effect_analysis.png'}")
    plt.close()

    # 2. Mean Reversion Timing
    if len(df_reversion) > 0:
        print("  Creating mean reversion timing plot...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Reversion rate over time
        reversion_by_games = df_reversion[df_reversion['reverted']].groupby('games_to_revert').size()

        axes[0].bar(
            reversion_by_games.index,
            reversion_by_games.values,
            color='steelblue',
            alpha=0.7,
            edgecolor='black'
        )

        axes[0].set_xlabel('Games Until Reversion', fontsize=12)
        axes[0].set_ylabel('Number of Instances', fontsize=12)
        axes[0].set_title('Mean Reversion Timing', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        # Reversion by position
        position_reversion = df_reversion.groupby('position').agg({
            'reverted': 'mean',
            'games_to_revert': 'mean'
        })

        x = np.arange(len(position_reversion))
        width = 0.35

        axes[1].bar(
            x - width/2,
            position_reversion['reverted'],
            width,
            label='Reversion Rate',
            color='#2ECC71',
            alpha=0.7,
            edgecolor='black'
        )

        ax2 = axes[1].twinx()
        ax2.bar(
            x + width/2,
            position_reversion['games_to_revert'],
            width,
            label='Games to Revert',
            color='#E74C3C',
            alpha=0.7,
            edgecolor='black'
        )

        axes[1].set_xlabel('Position', fontsize=12)
        axes[1].set_ylabel('Reversion Rate', fontsize=12, color='#2ECC71')
        ax2.set_ylabel('Average Games to Revert', fontsize=12, color='#E74C3C')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(position_reversion.index)
        axes[1].set_title('Position-Specific Reversion Patterns', fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='y', labelcolor='#2ECC71')
        ax2.tick_params(axis='y', labelcolor='#E74C3C')

        # Add legends
        lines1, labels1 = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()
        plt.savefig(VIZ_DIR / 'mean_reversion_timing.png', dpi=300, bbox_inches='tight')
        print(f"    Saved: {VIZ_DIR / 'mean_reversion_timing.png'}")
        plt.close()


def save_results(df, df_hot_hand, df_recovery, df_reversion):
    """Save analysis results"""
    print(f"\nSaving results...")

    # Save hot hand correlations
    output_file = PRED_DIR / 'momentum_correlations.csv'
    df_hot_hand.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")

    # Save recovery analysis
    if len(df_recovery) > 0:
        output_file = PRED_DIR / 'cold_streak_recovery.csv'
        df_recovery.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")

    # Save reversion analysis
    if len(df_reversion) > 0:
        output_file = PRED_DIR / 'reversion_timing.csv'
        df_reversion.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("MOMENTUM & TREND ANALYSIS: KEY FINDINGS")
    print("=" * 80)

    print("\nHot Hand Evidence (correlation with next game yards):")
    top_momentum = df_hot_hand[df_hot_hand['position'] == 'ALL'].sort_values('correlation', ascending=False)
    print(top_momentum[['feature', 'correlation', 'p_value', 'sample_size']].to_string(index=False))

    if len(df_recovery) > 0:
        print("\n\nCold Streak Recovery:")
        print(f"  Total cold streak exits: {len(df_recovery):,}")
        print(f"  Average recovery performance vs mean: {df_recovery['recovery_vs_mean'].mean():.1f} yards")

    if len(df_reversion) > 0:
        print("\n\nMean Reversion Patterns:")
        print(f"  Outlier performances analyzed: {len(df_reversion):,}")
        print(f"  Reversion rate (within 3 games): {df_reversion['reverted'].mean():.1%}")
        print(f"  Average games to revert: {df_reversion['games_to_revert'].mean():.2f}")


def main():
    """Main execution function"""

    # Load data
    df = load_data(seasons=[2022, 2023, 2024])

    # Calculate momentum features
    df = calculate_momentum_features(df)

    # Detect breakouts
    df = detect_breakouts(df)

    # Test hot hand hypothesis
    df, df_hot_hand = test_hot_hand_hypothesis(df)

    # Analyze cold streak recovery
    df_recovery = analyze_cold_streak_recovery(df)

    # Calculate reversion strength
    df_reversion = calculate_reversion_strength(df)

    # Create visualizations
    create_visualizations(df, df_hot_hand, df_recovery, df_reversion)

    # Save results
    save_results(df, df_hot_hand, df_recovery, df_reversion)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
