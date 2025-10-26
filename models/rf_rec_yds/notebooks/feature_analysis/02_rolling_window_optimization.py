"""
Rolling Window Optimization Analysis
=====================================
Determines optimal historical lookback periods for predicting receiving yards.

Tasks:
1. Test multiple window sizes (2, 3, 4, 5, 7, 10 games)
2. Calculate rolling statistics (mean, median, max, min, std)
3. Test equal weight vs exponentially weighted schemes
4. Analyze correlation with next-game receiving yards
5. Position-specific optimization (WR, TE, RB)
6. Handle edge cases (insufficient history early in season)
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

# Create output directories
VIZ_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ROLLING WINDOW OPTIMIZATION ANALYSIS")
print("=" * 80)


def load_player_game_receiving_data(seasons=[2022, 2023, 2024]):
    """Load and combine player game receiving data across seasons"""
    print(f"\n[1/8] Loading player game receiving data for seasons: {seasons}")

    dfs = []
    for season in seasons:
        season_dir = DATA_DIR / "plyr_gm" / "plyr_gm_rec" / f"season={season}"

        if not season_dir.exists():
            print(f"  Warning: Season {season} directory not found")
            continue

        # Load all weeks for this season
        week_dirs = sorted([d for d in season_dir.iterdir() if d.is_dir()])

        for week_dir in week_dirs:
            week_num = int(week_dir.name.split('=')[1])
            parquet_file = week_dir / "data.parquet"

            if parquet_file.exists():
                df_week = pd.read_parquet(parquet_file)
                df_week['season'] = season
                df_week['week'] = week_num
                dfs.append(df_week)

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df):,} player-game records")
    print(f"  Columns: {list(df.columns)}")

    return df


def load_player_info(seasons=[2022, 2023, 2024]):
    """Load player information including position"""
    print(f"\n[2/8] Loading player info for position data")

    dfs = []
    for season in seasons:
        player_dir = DATA_DIR / "players" / "plyr" / f"season={season}"

        if player_dir.exists():
            # Find the first parquet file in the directory
            parquet_files = list(player_dir.glob("*.parquet"))
            if parquet_files:
                df_player = pd.read_parquet(parquet_files[0])
                df_player['season'] = season
                dfs.append(df_player)

    if not dfs:
        print("  Warning: No player data found")
        return pd.DataFrame(columns=['plyr_id', 'plyr_name', 'plyr_pos', 'season'])

    df = pd.concat(dfs, ignore_index=True)

    # Keep only relevant columns
    if 'plyr_pos' in df.columns:
        df = df[['plyr_id', 'plyr_name', 'plyr_pos', 'season']].drop_duplicates()

    print(f"  Loaded {len(df):,} player-season records")

    return df


def load_snap_counts(seasons=[2022, 2023, 2024]):
    """Load snap count data"""
    print(f"\n[3/8] Loading snap count data")

    dfs = []
    for season in seasons:
        season_dir = DATA_DIR / "plyr_gm" / "plyr_gm_snap_ct" / f"season={season}"

        if not season_dir.exists():
            continue

        week_dirs = sorted([d for d in season_dir.iterdir() if d.is_dir()])

        for week_dir in week_dirs:
            week_num = int(week_dir.name.split('=')[1])
            parquet_file = week_dir / "data.parquet"

            if parquet_file.exists():
                df_week = pd.read_parquet(parquet_file)
                df_week['season'] = season
                df_week['week'] = week_num
                dfs.append(df_week)

    if not dfs:
        print("  Warning: No snap count data found")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df):,} snap count records")

    return df


def prepare_analysis_dataset(df_rec, df_players, df_snaps):
    """Prepare combined dataset for analysis"""
    print(f"\n[4/8] Preparing analysis dataset")

    # Merge with player info
    df = df_rec.merge(
        df_players[['plyr_id', 'plyr_name', 'plyr_pos', 'season']],
        on=['plyr_id', 'season'],
        how='left'
    )

    # Merge with snap counts
    if not df_snaps.empty:
        snap_cols = ['plyr_id', 'gm_id', 'season', 'week', 'plyr_gm_off_snap_ct', 'plyr_gm_off_snap_ct_pct']
        available_snap_cols = [col for col in snap_cols if col in df_snaps.columns]

        df = df.merge(
            df_snaps[available_snap_cols],
            on=['plyr_id', 'gm_id', 'season', 'week'],
            how='left'
        )

    # Filter to relevant positions (WR, TE, RB)
    relevant_positions = ['WR', 'TE', 'RB']
    df = df[df['plyr_pos'].isin(relevant_positions)].copy()

    # Create game date for proper sorting
    df['game_order'] = df['season'] * 100 + df['week']

    # Sort by player and game order
    df = df.sort_values(['plyr_id', 'game_order']).reset_index(drop=True)

    print(f"  Final dataset: {len(df):,} records")
    print(f"  Unique players: {df['plyr_id'].nunique():,}")
    print(f"  Position breakdown:")
    for pos in relevant_positions:
        count = (df['plyr_pos'] == pos).sum()
        print(f"    {pos}: {count:,} records")

    return df


def calculate_rolling_features(df, windows=[2, 3, 4, 5, 7, 10], metrics=['mean', 'median', 'std', 'min', 'max']):
    """Calculate rolling statistics for various window sizes"""
    print(f"\n[5/8] Calculating rolling features")
    print(f"  Windows to test: {windows}")
    print(f"  Metrics to calculate: {metrics}")

    # Core receiving stats to roll
    rolling_cols = [
        'plyr_gm_rec_yds',
        'plyr_gm_rec',
        'plyr_gm_rec_tgt',
        'plyr_gm_rec_td'
    ]

    # Add snap count if available
    if 'plyr_gm_off_snap_ct' in df.columns:
        rolling_cols.append('plyr_gm_off_snap_ct')

    # Calculate efficiency metrics
    df['yards_per_reception'] = df['plyr_gm_rec_yds'] / df['plyr_gm_rec'].replace(0, np.nan)
    df['catch_rate'] = df['plyr_gm_rec'] / df['plyr_gm_rec_tgt'].replace(0, np.nan)
    df['yards_per_target'] = df['plyr_gm_rec_yds'] / df['plyr_gm_rec_tgt'].replace(0, np.nan)

    efficiency_cols = ['yards_per_reception', 'catch_rate', 'yards_per_target']

    all_rolling_cols = rolling_cols + efficiency_cols

    # Calculate rolling features for each player
    feature_cols = []

    for window in windows:
        print(f"  Processing window size: {window}")

        for col in all_rolling_cols:
            if col not in df.columns:
                continue

            for metric in metrics:
                feature_name = f'{col}_rolling_{window}_{metric}'

                if metric == 'mean':
                    df[feature_name] = df.groupby('plyr_id')[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                    )
                elif metric == 'median':
                    df[feature_name] = df.groupby('plyr_id')[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).median().shift(1)
                    )
                elif metric == 'std':
                    df[feature_name] = df.groupby('plyr_id')[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std().shift(1)
                    )
                elif metric == 'min':
                    df[feature_name] = df.groupby('plyr_id')[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).min().shift(1)
                    )
                elif metric == 'max':
                    df[feature_name] = df.groupby('plyr_id')[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).max().shift(1)
                    )

                feature_cols.append(feature_name)

    # Calculate exponentially weighted moving averages for key windows
    for window in [3, 5, 10]:
        for col in ['plyr_gm_rec_yds', 'plyr_gm_rec_tgt', 'yards_per_target']:
            if col not in df.columns:
                continue

            feature_name = f'{col}_ewm_{window}'
            df[feature_name] = df.groupby('plyr_id')[col].transform(
                lambda x: x.ewm(span=window, min_periods=1).mean().shift(1)
            )
            feature_cols.append(feature_name)

    print(f"  Created {len(feature_cols)} rolling features")

    return df, feature_cols


def calculate_correlations(df, feature_cols, target='plyr_gm_rec_yds'):
    """Calculate correlations between rolling features and next-game yards"""
    print(f"\n[6/8] Calculating feature correlations with {target}")

    results = []

    for feature in feature_cols:
        if feature not in df.columns:
            continue

        # Overall correlation
        valid_mask = df[feature].notna() & df[target].notna()

        if valid_mask.sum() < 30:  # Need minimum sample size
            continue

        pearson_corr, pearson_pval = pearsonr(
            df.loc[valid_mask, feature],
            df.loc[valid_mask, target]
        )

        spearman_corr, spearman_pval = spearmanr(
            df.loc[valid_mask, feature],
            df.loc[valid_mask, target]
        )

        results.append({
            'feature': feature,
            'position': 'ALL',
            'pearson_corr': pearson_corr,
            'pearson_pval': pearson_pval,
            'spearman_corr': spearman_corr,
            'spearman_pval': spearman_pval,
            'sample_size': valid_mask.sum()
        })

        # Position-specific correlations
        for position in ['WR', 'TE', 'RB']:
            pos_mask = (df['plyr_pos'] == position) & valid_mask

            if pos_mask.sum() < 30:
                continue

            pearson_corr, pearson_pval = pearsonr(
                df.loc[pos_mask, feature],
                df.loc[pos_mask, target]
            )

            spearman_corr, spearman_pval = spearmanr(
                df.loc[pos_mask, feature],
                df.loc[pos_mask, target]
            )

            results.append({
                'feature': feature,
                'position': position,
                'pearson_corr': pearson_corr,
                'pearson_pval': pearson_pval,
                'spearman_corr': spearman_corr,
                'spearman_pval': spearman_pval,
                'sample_size': pos_mask.sum()
            })

    df_corr = pd.DataFrame(results)

    # Parse feature components
    df_corr['base_metric'] = df_corr['feature'].str.split('_rolling_|_ewm_').str[0]
    df_corr['window_size'] = df_corr['feature'].str.extract(r'(?:rolling|ewm)_(\d+)')[0].astype(float)
    df_corr['stat_type'] = df_corr['feature'].str.extract(r'_(\w+)$')[0]

    # Flag exponentially weighted
    df_corr['is_ewm'] = df_corr['feature'].str.contains('_ewm_')

    print(f"  Calculated correlations for {len(df_corr):,} feature-position combinations")

    return df_corr


def find_optimal_windows(df_corr):
    """Identify optimal window sizes for each metric and position"""
    print(f"\n[7/8] Finding optimal window sizes")

    # Focus on mean statistics for primary recommendation
    df_means = df_corr[
        (df_corr['stat_type'] == 'mean') |
        (df_corr['is_ewm'] == True)
    ].copy()

    # Find best window for each base metric and position
    optimal_windows = []

    for position in ['ALL', 'WR', 'TE', 'RB']:
        for base_metric in df_means['base_metric'].unique():
            df_subset = df_means[
                (df_means['position'] == position) &
                (df_means['base_metric'] == base_metric)
            ]

            if len(df_subset) == 0:
                continue

            # Find window with highest absolute correlation
            best_idx = df_subset['pearson_corr'].abs().idxmax()
            best_row = df_subset.loc[best_idx]

            optimal_windows.append({
                'position': position,
                'base_metric': base_metric,
                'optimal_window': best_row['window_size'],
                'method': 'EWM' if best_row['is_ewm'] else 'Simple',
                'correlation': best_row['pearson_corr'],
                'p_value': best_row['pearson_pval'],
                'sample_size': best_row['sample_size']
            })

    df_optimal = pd.DataFrame(optimal_windows)
    df_optimal = df_optimal.sort_values(['position', 'correlation'], ascending=[True, False])

    print(f"  Identified {len(df_optimal)} optimal window configurations")

    return df_optimal


def create_visualizations(df, df_corr, df_optimal):
    """Create comprehensive visualizations"""
    print(f"\n[8/8] Creating visualizations")

    # 1. Heatmap: Window Size x Metric x Correlation by Position
    print("  Creating rolling window correlations heatmap...")

    # Filter to mean statistics for clarity
    df_heatmap = df_corr[df_corr['stat_type'] == 'mean'].copy()

    # Create pivot table for each position
    positions = ['WR', 'TE', 'RB', 'ALL']
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    for idx, position in enumerate(positions):
        df_pos = df_heatmap[df_heatmap['position'] == position]

        # Pivot: base_metric vs window_size
        pivot = df_pos.pivot_table(
            index='base_metric',
            columns='window_size',
            values='pearson_corr',
            aggfunc='first'
        )

        # Sort by average correlation
        pivot = pivot.loc[pivot.abs().mean(axis=1).sort_values(ascending=False).index]

        # Create heatmap
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0,
            vmin=-0.3,
            vmax=0.8,
            ax=axes[idx],
            cbar_kws={'label': 'Pearson Correlation'}
        )

        axes[idx].set_title(f'{position} - Rolling Window Correlations', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Window Size (games)', fontsize=12)
        axes[idx].set_ylabel('Metric', fontsize=12)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'rolling_window_correlations_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: {VIZ_DIR / 'rolling_window_correlations_heatmap.png'}")
    plt.close()

    # 2. Line plot: Correlation vs Window Size for key metrics
    print("  Creating correlation vs window size plot...")

    key_metrics = ['plyr_gm_rec_yds', 'plyr_gm_rec_tgt', 'yards_per_target']
    df_lines = df_corr[
        (df_corr['stat_type'] == 'mean') &
        (df_corr['base_metric'].isin(key_metrics))
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for idx, position in enumerate(['WR', 'TE', 'RB']):
        df_pos = df_lines[df_lines['position'] == position]

        for metric in key_metrics:
            df_metric = df_pos[df_pos['base_metric'] == metric].sort_values('window_size')

            axes[idx].plot(
                df_metric['window_size'],
                df_metric['pearson_corr'],
                marker='o',
                linewidth=2,
                label=metric.replace('plyr_gm_', '').replace('_', ' ').title()
            )

        axes[idx].set_title(f'{position} Position', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Rolling Window Size (games)', fontsize=12)
        axes[idx].set_ylabel('Pearson Correlation', fontsize=12)
        axes[idx].legend(fontsize=10)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axhline(y=0, color='black', linestyle='--', linewidth=0.8)

    plt.suptitle('Correlation vs Window Size by Position', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'correlation_vs_window_size.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: {VIZ_DIR / 'correlation_vs_window_size.png'}")
    plt.close()

    # 3. Bar plot: Top features overall
    print("  Creating top features bar plot...")

    df_top = df_corr[df_corr['position'] == 'ALL'].nlargest(20, 'pearson_corr')

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(df_top)), df_top['pearson_corr'], color='steelblue')
    plt.yticks(range(len(df_top)), df_top['feature'], fontsize=9)
    plt.xlabel('Pearson Correlation', fontsize=12)
    plt.title('Top 20 Rolling Features by Correlation (All Positions)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'top_rolling_features.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: {VIZ_DIR / 'top_rolling_features.png'}")
    plt.close()


def save_results(df_corr, df_optimal):
    """Save analysis results"""
    print(f"\nSaving results...")

    # Save full correlation matrix
    output_file = PRED_DIR / 'rolling_window_correlations_full.csv'
    df_corr.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")

    # Save optimal windows
    output_file = PRED_DIR / 'optimal_rolling_windows.csv'
    df_optimal.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")

    # Create summary report
    print("\n" + "=" * 80)
    print("OPTIMAL ROLLING WINDOW RECOMMENDATIONS")
    print("=" * 80)

    for position in ['ALL', 'WR', 'TE', 'RB']:
        df_pos = df_optimal[df_optimal['position'] == position].head(10)

        if len(df_pos) == 0:
            continue

        print(f"\n{position} Position - Top 10 Features:")
        print("-" * 80)
        for _, row in df_pos.iterrows():
            print(f"  {row['base_metric']:40s} | Window: {row['optimal_window']:2.0f} | "
                  f"Method: {row['method']:6s} | Corr: {row['correlation']:6.3f} | "
                  f"p={row['p_value']:.4f} | n={row['sample_size']:,}")


def main():
    """Main execution function"""

    # Load data
    df_rec = load_player_game_receiving_data(seasons=[2022, 2023, 2024])
    df_players = load_player_info(seasons=[2022, 2023, 2024])
    df_snaps = load_snap_counts(seasons=[2022, 2023, 2024])

    # Prepare dataset
    df = prepare_analysis_dataset(df_rec, df_players, df_snaps)

    # Calculate rolling features
    df, feature_cols = calculate_rolling_features(df)

    # Calculate correlations
    df_corr = calculate_correlations(df, feature_cols)

    # Find optimal windows
    df_optimal = find_optimal_windows(df_corr)

    # Create visualizations
    create_visualizations(df, df_corr, df_optimal)

    # Save results
    save_results(df_corr, df_optimal)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
