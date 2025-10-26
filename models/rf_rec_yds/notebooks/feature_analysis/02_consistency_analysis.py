"""
Consistency vs Volatility Analysis
===================================
Quantifies player consistency and its impact on predictability.

Tasks:
1. Calculate variance measures (CV, standard deviation, IQR)
2. Identify performance buckets (boom/bust rates, floor/ceiling)
3. Analyze streak patterns
4. Test if past consistency predicts future consistency
5. Examine mean reversion patterns
6. Position-specific consistency profiles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = Path(r"C:\Users\nocap\Desktop\code\NFL_ml")
DATA_DIR = BASE_DIR / "parquet_files" / "clean"
OUTPUT_DIR = Path(r"C:\Users\nocap\Desktop\code\NFL_ml\models\rf_rec_yds\outputs")
VIZ_DIR = OUTPUT_DIR / "visualizations"
PRED_DIR = OUTPUT_DIR / "predictions"

print("=" * 80)
print("CONSISTENCY VS VOLATILITY ANALYSIS")
print("=" * 80)


def load_data(seasons=[2022, 2023, 2024]):
    """Load player game receiving data"""
    print(f"\n[1/6] Loading data for seasons: {seasons}")

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


def calculate_consistency_metrics(df, min_games=8):
    """Calculate comprehensive consistency metrics for each player-season"""
    print(f"\n[2/6] Calculating consistency metrics (min {min_games} games)")

    results = []

    # Group by player-season
    for (player_id, season), df_player_season in df.groupby(['plyr_id', 'season']):

        if len(df_player_season) < min_games:
            continue

        player_name = df_player_season['plyr_name'].iloc[0]
        position = df_player_season['plyr_pos'].iloc[0]
        yards = df_player_season['plyr_gm_rec_yds'].values

        # Basic statistics
        mean_yards = yards.mean()
        median_yards = np.median(yards)
        std_yards = yards.std()
        min_yards = yards.min()
        max_yards = yards.max()

        # Coefficient of variation
        cv = std_yards / mean_yards if mean_yards > 0 else np.nan

        # Interquartile range
        q25 = np.percentile(yards, 25)
        q75 = np.percentile(yards, 75)
        iqr = q75 - q25

        # Performance buckets
        boom_rate = (yards >= 100).mean()  # % games >= 100 yards
        bust_rate = (yards < 20).mean()    # % games < 20 yards
        solid_rate = ((yards >= 50) & (yards < 100)).mean()  # 50-99 yards

        # Floor and ceiling
        floor = q25  # 25th percentile
        ceiling = q75  # 75th percentile

        # Longest streaks
        def longest_streak(condition):
            """Find longest consecutive streak meeting condition"""
            max_streak = 0
            current_streak = 0
            for val in condition:
                if val:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0
            return max_streak

        streak_good = longest_streak(yards >= 50)
        streak_bad = longest_streak(yards < 30)

        # Game-to-game variance (absolute differences)
        yard_diffs = np.abs(np.diff(yards))
        avg_game_to_game_change = yard_diffs.mean()
        max_game_to_game_change = yard_diffs.max()

        # Autocorrelation (does performance predict next game?)
        if len(yards) > 2:
            autocorr = np.corrcoef(yards[:-1], yards[1:])[0, 1] if len(yards) > 1 else np.nan
        else:
            autocorr = np.nan

        results.append({
            'plyr_id': player_id,
            'player_name': player_name,
            'position': position,
            'season': season,
            'games_played': len(df_player_season),

            # Central tendency
            'mean_yards': mean_yards,
            'median_yards': median_yards,

            # Variability
            'std_yards': std_yards,
            'cv': cv,
            'iqr': iqr,
            'min_yards': min_yards,
            'max_yards': max_yards,
            'range': max_yards - min_yards,

            # Performance buckets
            'boom_rate': boom_rate,
            'bust_rate': bust_rate,
            'solid_rate': solid_rate,
            'floor': floor,
            'ceiling': ceiling,

            # Streaks
            'longest_good_streak': streak_good,
            'longest_bad_streak': streak_bad,

            # Game-to-game
            'avg_gtg_change': avg_game_to_game_change,
            'max_gtg_change': max_game_to_game_change,
            'autocorrelation': autocorr
        })

    df_consistency = pd.DataFrame(results)

    print(f"  Calculated consistency for {len(df_consistency):,} player-seasons")

    return df_consistency


def analyze_predictability(df_consistency):
    """Test if consistent players are more predictable"""
    print(f"\n[3/6] Analyzing relationship between consistency and predictability")

    # Group players by consistency level
    df_consistency['consistency_tier'] = pd.qcut(
        df_consistency['cv'],
        q=4,
        labels=['Very Consistent', 'Consistent', 'Volatile', 'Very Volatile']
    )

    # Analyze by tier
    print("\n  Consistency Tiers:")
    tier_summary = df_consistency.groupby('consistency_tier').agg({
        'cv': ['mean', 'min', 'max'],
        'mean_yards': 'mean',
        'std_yards': 'mean',
        'boom_rate': 'mean',
        'bust_rate': 'mean',
        'autocorrelation': 'mean'
    }).round(3)
    print(tier_summary)

    # Position-specific analysis
    print("\n  Position-Specific Consistency:")
    pos_summary = df_consistency.groupby('position').agg({
        'cv': ['mean', 'median', 'std'],
        'autocorrelation': 'mean',
        'boom_rate': 'mean',
        'bust_rate': 'mean'
    }).round(3)
    print(pos_summary)

    return df_consistency


def analyze_mean_reversion(df):
    """Test for mean reversion patterns"""
    print(f"\n[4/6] Analyzing mean reversion patterns")

    reversion_results = []

    # For each player-season with sufficient games
    for (player_id, season), df_ps in df.groupby(['plyr_id', 'season']):

        if len(df_ps) < 10:
            continue

        player_name = df_ps['plyr_name'].iloc[0]
        position = df_ps['plyr_pos'].iloc[0]
        yards = df_ps['plyr_gm_rec_yds'].values

        # Calculate player's mean
        player_mean = yards.mean()

        # For each game, calculate deviation from mean and next game's change
        for i in range(len(yards) - 1):
            deviation_from_mean = yards[i] - player_mean
            next_game_yards = yards[i + 1]
            change_to_next = next_game_yards - yards[i]

            # Categorize deviation
            if deviation_from_mean > player_mean * 0.5:  # Boom game
                deviation_category = 'Boom'
            elif deviation_from_mean < -player_mean * 0.5:  # Bust game
                deviation_category = 'Bust'
            else:
                deviation_category = 'Average'

            reversion_results.append({
                'plyr_id': player_id,
                'player_name': player_name,
                'position': position,
                'current_yards': yards[i],
                'next_yards': next_game_yards,
                'deviation_from_mean': deviation_from_mean,
                'change_to_next': change_to_next,
                'deviation_category': deviation_category,
                'player_mean': player_mean
            })

    df_reversion = pd.DataFrame(reversion_results)

    # Analyze reversion by deviation category
    print("\n  Mean Reversion by Performance Category:")
    reversion_summary = df_reversion.groupby('deviation_category').agg({
        'current_yards': 'mean',
        'next_yards': 'mean',
        'change_to_next': 'mean'
    }).round(2)
    print(reversion_summary)

    # Calculate correlation between deviation and next game change
    if len(df_reversion) > 30:
        corr, pval = pearsonr(
            df_reversion['deviation_from_mean'],
            df_reversion['change_to_next']
        )
        print(f"\n  Correlation (deviation vs next game change): {corr:.3f} (p={pval:.4f})")
        print(f"  Negative correlation indicates mean reversion")

    return df_reversion


def calculate_rolling_consistency(df):
    """Calculate rolling consistency metrics"""
    print(f"\n[5/6] Calculating rolling consistency features")

    # Calculate rolling CV for last 5 games
    df['rolling_cv_5'] = df.groupby('plyr_id')['plyr_gm_rec_yds'].transform(
        lambda x: x.rolling(window=5, min_periods=3).std() / x.rolling(window=5, min_periods=3).mean()
    )

    # Shift to avoid data leakage
    df['rolling_cv_5'] = df.groupby('plyr_id')['rolling_cv_5'].shift(1)

    # Calculate if this creates a predictive feature
    valid_mask = df['rolling_cv_5'].notna() & df['plyr_gm_rec_yds'].notna()

    if valid_mask.sum() > 30:
        corr, pval = pearsonr(
            df.loc[valid_mask, 'rolling_cv_5'],
            df.loc[valid_mask, 'plyr_gm_rec_yds']
        )
        print(f"  Correlation (rolling CV vs current yards): {corr:.3f} (p={pval:.4f})")

    return df


def create_visualizations(df_consistency, df_reversion, df):
    """Create comprehensive visualizations"""
    print(f"\n[6/6] Creating visualizations")

    # 1. Player Volatility Distribution
    print("  Creating volatility distribution plot...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    positions = ['WR', 'TE', 'RB']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for idx, (position, color) in enumerate(zip(positions, colors)):
        df_pos = df_consistency[df_consistency['position'] == position]

        axes[idx].hist(df_pos['cv'], bins=30, color=color, alpha=0.7, edgecolor='black')
        axes[idx].axvline(df_pos['cv'].median(), color='red', linestyle='--', linewidth=2, label=f'Median: {df_pos["cv"].median():.2f}')
        axes[idx].set_xlabel('Coefficient of Variation', fontsize=12)
        axes[idx].set_ylabel('Frequency', fontsize=12)
        axes[idx].set_title(f'{position} Position', fontsize=14, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(axis='y', alpha=0.3)

    plt.suptitle('Player Volatility Distribution by Position', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'player_volatility_distribution.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: {VIZ_DIR / 'player_volatility_distribution.png'}")
    plt.close()

    # 2. Consistency vs Performance
    print("  Creating consistency vs performance plot...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # CV vs Mean Yards
    for position, color in zip(['WR', 'TE', 'RB'], ['#FF6B6B', '#4ECDC4', '#45B7D1']):
        df_pos = df_consistency[df_consistency['position'] == position]
        axes[0].scatter(
            df_pos['cv'],
            df_pos['mean_yards'],
            label=position,
            alpha=0.6,
            s=50,
            c=color,
            edgecolors='black',
            linewidth=0.5
        )

    axes[0].set_xlabel('Coefficient of Variation (Consistency)', fontsize=12)
    axes[0].set_ylabel('Mean Yards per Game', fontsize=12)
    axes[0].set_title('Consistency vs Average Performance', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Boom Rate vs Bust Rate
    for position, color in zip(['WR', 'TE', 'RB'], ['#FF6B6B', '#4ECDC4', '#45B7D1']):
        df_pos = df_consistency[df_consistency['position'] == position]
        axes[1].scatter(
            df_pos['boom_rate'],
            df_pos['bust_rate'],
            label=position,
            alpha=0.6,
            s=50,
            c=color,
            edgecolors='black',
            linewidth=0.5
        )

    axes[1].set_xlabel('Boom Rate (% games >= 100 yds)', fontsize=12)
    axes[1].set_ylabel('Bust Rate (% games < 20 yds)', fontsize=12)
    axes[1].set_title('Boom vs Bust Rates', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'consistency_vs_performance.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: {VIZ_DIR / 'consistency_vs_performance.png'}")
    plt.close()

    # 3. Mean Reversion Analysis
    print("  Creating mean reversion plot...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Deviation vs Next Game Change
    axes[0].scatter(
        df_reversion['deviation_from_mean'],
        df_reversion['change_to_next'],
        alpha=0.3,
        s=20,
        c='steelblue'
    )

    # Add regression line
    z = np.polyfit(df_reversion['deviation_from_mean'], df_reversion['change_to_next'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_reversion['deviation_from_mean'].min(), df_reversion['deviation_from_mean'].max(), 100)
    axes[0].plot(x_line, p(x_line), "r--", linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.1f}')

    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    axes[0].set_xlabel('Deviation from Player Mean (current game)', fontsize=12)
    axes[0].set_ylabel('Change to Next Game', fontsize=12)
    axes[0].set_title('Mean Reversion Pattern', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot by deviation category
    categories = ['Bust', 'Average', 'Boom']
    data_to_plot = [df_reversion[df_reversion['deviation_category'] == cat]['change_to_next'].values
                    for cat in categories]

    bp = axes[1].boxplot(data_to_plot, labels=categories, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#E74C3C', '#95A5A6', '#2ECC71']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Performance Category', fontsize=12)
    axes[1].set_ylabel('Yards Change to Next Game', fontsize=12)
    axes[1].set_title('Mean Reversion by Performance Category', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'mean_reversion_analysis.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: {VIZ_DIR / 'mean_reversion_analysis.png'}")
    plt.close()

    # 4. Autocorrelation Distribution
    print("  Creating autocorrelation distribution plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    for position, color in zip(['WR', 'TE', 'RB'], ['#FF6B6B', '#4ECDC4', '#45B7D1']):
        df_pos = df_consistency[df_consistency['position'] == position]
        df_pos_valid = df_pos[df_pos['autocorrelation'].notna()]

        ax.hist(df_pos_valid['autocorrelation'], bins=30, alpha=0.5, label=position, color=color, edgecolor='black')

    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='No Autocorrelation')
    ax.set_xlabel('Autocorrelation (game t vs game t+1)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Game-to-Game Autocorrelation', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'autocorrelation_distribution.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: {VIZ_DIR / 'autocorrelation_distribution.png'}")
    plt.close()


def save_results(df_consistency, df_reversion):
    """Save analysis results"""
    print(f"\nSaving results...")

    # Save consistency scores
    output_file = PRED_DIR / 'consistency_scores.csv'
    df_consistency.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")

    # Save reversion analysis
    output_file = PRED_DIR / 'mean_reversion_analysis.csv'
    df_reversion.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("CONSISTENCY ANALYSIS: KEY FINDINGS")
    print("=" * 80)

    print("\nConsistency by Position:")
    print(df_consistency.groupby('position')['cv'].describe().round(3))

    print("\n\nMost Consistent Players (Lowest CV):")
    top_consistent = df_consistency.nsmallest(10, 'cv')[
        ['player_name', 'position', 'season', 'cv', 'mean_yards', 'games_played']
    ]
    print(top_consistent.to_string(index=False))

    print("\n\nMost Volatile Players (Highest CV):")
    top_volatile = df_consistency.nlargest(10, 'cv')[
        ['player_name', 'position', 'season', 'cv', 'mean_yards', 'boom_rate', 'bust_rate']
    ]
    print(top_volatile.to_string(index=False))

    print("\n\nAutocorrelation Analysis:")
    print(f"Mean autocorrelation: {df_consistency['autocorrelation'].mean():.3f}")
    print(f"Players with positive autocorrelation: {(df_consistency['autocorrelation'] > 0.1).sum()}")
    print(f"Players with negative autocorrelation: {(df_consistency['autocorrelation'] < -0.1).sum()}")


def main():
    """Main execution function"""

    # Load data
    df = load_data(seasons=[2022, 2023, 2024])

    # Calculate consistency metrics
    df_consistency = calculate_consistency_metrics(df)

    # Analyze predictability
    df_consistency = analyze_predictability(df_consistency)

    # Analyze mean reversion
    df_reversion = analyze_mean_reversion(df)

    # Calculate rolling consistency
    df = calculate_rolling_consistency(df)

    # Create visualizations
    create_visualizations(df_consistency, df_reversion, df)

    # Save results
    save_results(df_consistency, df_reversion)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
