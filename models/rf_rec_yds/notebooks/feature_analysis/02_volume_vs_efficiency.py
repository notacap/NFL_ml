"""
Volume vs Efficiency Analysis
==============================
Determines whether opportunity (volume) or efficiency metrics better predict receiving yards.

Tasks:
1. Calculate volume metrics (targets, receptions, snap counts, shares)
2. Calculate efficiency metrics (yards/reception, yards/target, catch rate)
3. Assess stability and variance of each metric type
4. Compare predictive power through correlations
5. Identify player archetypes (high-volume/low-efficiency, etc.)
6. Position-specific volume vs efficiency patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = Path(r"C:\Users\nocap\Desktop\code\NFL_ml")
DATA_DIR = BASE_DIR / "parquet_files" / "clean"
OUTPUT_DIR = Path(r"C:\Users\nocap\Desktop\code\NFL_ml\models\rf_rec_yds\outputs")
VIZ_DIR = OUTPUT_DIR / "visualizations"
PRED_DIR = OUTPUT_DIR / "predictions"

print("=" * 80)
print("VOLUME VS EFFICIENCY ANALYSIS")
print("=" * 80)


def load_data(seasons=[2022, 2023, 2024]):
    """Load all necessary data"""
    print(f"\n[1/7] Loading data for seasons: {seasons}")

    # Load player game receiving data
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

    # Load snap counts
    dfs_snap = []
    for season in seasons:
        season_dir = DATA_DIR / "plyr_gm" / "plyr_gm_snap_ct" / f"season={season}"
        if not season_dir.exists():
            continue

        for week_dir in sorted([d for d in season_dir.iterdir() if d.is_dir()]):
            week_num = int(week_dir.name.split('=')[1])
            parquet_file = week_dir / "data.parquet"
            if parquet_file.exists():
                df_week = pd.read_parquet(parquet_file)
                df_week['season'] = season
                df_week['week'] = week_num
                dfs_snap.append(df_week)

    df_snaps = pd.concat(dfs_snap, ignore_index=True) if dfs_snap else pd.DataFrame()

    print(f"  Receiving records: {len(df_rec):,}")
    print(f"  Player records: {len(df_players):,}")
    print(f"  Snap count records: {len(df_snaps):,}")

    return df_rec, df_players, df_snaps


def calculate_team_aggregates(df):
    """Calculate team-level aggregates for computing shares"""
    print(f"\n[2/7] Calculating team aggregates")

    # Team totals per game
    team_totals = df.groupby(['team_id', 'game_id', 'season', 'week']).agg({
        'plyr_gm_rec_tgt': 'sum',
        'plyr_gm_rec': 'sum',
        'plyr_gm_rec_yds': 'sum'
    }).reset_index()

    team_totals.columns = ['team_id', 'game_id', 'season', 'week',
                           'tm_total_tgt', 'tm_total_rec', 'tm_total_rec_yds']

    print(f"  Calculated team totals for {len(team_totals):,} team-games")

    return team_totals


def prepare_analysis_dataset(df_rec, df_players, df_snaps, team_totals):
    """Prepare comprehensive dataset with all metrics"""
    print(f"\n[3/7] Preparing analysis dataset")

    # Merge with player info
    df = df_rec.merge(
        df_players[['plyr_id', 'plyr_name', 'plyr_pos', 'season']],
        on=['plyr_id', 'season'],
        how='left'
    )

    # Merge with snap counts
    if not df_snaps.empty:
        snap_cols = ['plyr_id', 'game_id', 'season', 'week', 'plyr_gm_off_snap_ct', 'plyr_gm_off_snap_ct_pct']
        available_snap_cols = [col for col in snap_cols if col in df_snaps.columns]
        df = df.merge(df_snaps[available_snap_cols], on=['plyr_id', 'game_id', 'season', 'week'], how='left')

    # Merge with team totals
    df = df.merge(team_totals, on=['team_id', 'game_id', 'season', 'week'], how='left')

    # Filter to relevant positions
    df = df[df['plyr_pos'].isin(['WR', 'TE', 'RB'])].copy()

    # Calculate VOLUME metrics
    df['raw_targets'] = df['plyr_gm_rec_tgt']
    df['raw_receptions'] = df['plyr_gm_rec']
    df['raw_snap_count'] = df.get('plyr_gm_off_snap_ct', np.nan)

    # Relative volume (shares)
    df['target_share'] = df['plyr_gm_rec_tgt'] / df['tm_total_tgt'].replace(0, np.nan)
    df['reception_share'] = df['plyr_gm_rec'] / df['tm_total_rec'].replace(0, np.nan)
    df['yards_share'] = df['plyr_gm_rec_yds'] / df['tm_total_rec_yds'].replace(0, np.nan)
    df['snap_share'] = df.get('plyr_gm_off_snap_ct_pct', np.nan)

    # Calculate EFFICIENCY metrics
    df['yards_per_reception'] = df['plyr_gm_rec_yds'] / df['plyr_gm_rec'].replace(0, np.nan)
    df['yards_per_target'] = df['plyr_gm_rec_yds'] / df['plyr_gm_rec_tgt'].replace(0, np.nan)
    df['catch_rate'] = df['plyr_gm_rec'] / df['plyr_gm_rec_tgt'].replace(0, np.nan)

    if 'plyr_gm_off_snap_ct' in df.columns:
        df['yards_per_snap'] = df['plyr_gm_rec_yds'] / df['plyr_gm_off_snap_ct'].replace(0, np.nan)
        df['targets_per_snap'] = df['plyr_gm_rec_tgt'] / df['plyr_gm_off_snap_ct'].replace(0, np.nan)

    # Sort by player and game
    df['game_order'] = df['season'] * 100 + df['week']
    df = df.sort_values(['plyr_id', 'game_order']).reset_index(drop=True)

    # Create lagged target variable (next game yards)
    df['next_game_yards'] = df.groupby('plyr_id')['plyr_gm_rec_yds'].shift(-1)

    print(f"  Final dataset: {len(df):,} records")
    print(f"  Unique players: {df['plyr_id'].nunique():,}")

    return df


def calculate_stability_metrics(df):
    """Calculate stability/variance metrics for volume and efficiency"""
    print(f"\n[4/7] Calculating stability metrics")

    # Filter to players with sufficient games
    min_games = 5
    player_game_counts = df.groupby('plyr_id').size()
    qualified_players = player_game_counts[player_game_counts >= min_games].index

    df_qualified = df[df['plyr_id'].isin(qualified_players)].copy()

    # Calculate player-level statistics
    volume_metrics = ['raw_targets', 'target_share', 'raw_receptions', 'snap_share']
    efficiency_metrics = ['yards_per_reception', 'yards_per_target', 'catch_rate', 'yards_per_snap']

    stability_results = []

    for player_id in qualified_players:
        df_player = df_qualified[df_qualified['plyr_id'] == player_id]

        if len(df_player) < min_games:
            continue

        player_name = df_player['plyr_name'].iloc[0]
        position = df_player['plyr_pos'].iloc[0]

        result = {
            'plyr_id': player_id,
            'player_name': player_name,
            'position': position,
            'games_played': len(df_player)
        }

        # Volume stability
        for metric in volume_metrics:
            if metric in df_player.columns and df_player[metric].notna().sum() > 0:
                values = df_player[metric].dropna()
                result[f'{metric}_mean'] = values.mean()
                result[f'{metric}_std'] = values.std()
                result[f'{metric}_cv'] = values.std() / values.mean() if values.mean() != 0 else np.nan

        # Efficiency stability
        for metric in efficiency_metrics:
            if metric in df_player.columns and df_player[metric].notna().sum() > 0:
                values = df_player[metric].dropna()
                result[f'{metric}_mean'] = values.mean()
                result[f'{metric}_std'] = values.std()
                result[f'{metric}_cv'] = values.std() / values.mean() if values.mean() != 0 else np.nan

        stability_results.append(result)

    df_stability = pd.DataFrame(stability_results)

    print(f"  Calculated stability for {len(df_stability):,} players")

    return df_stability


def compare_predictive_power(df):
    """Compare correlation of volume vs efficiency metrics with next game yards"""
    print(f"\n[5/7] Comparing predictive power")

    # Define metric groups
    volume_metrics = {
        'Raw Volume': ['raw_targets', 'raw_receptions', 'raw_snap_count'],
        'Relative Volume': ['target_share', 'reception_share', 'snap_share']
    }

    efficiency_metrics = {
        'Reception Efficiency': ['yards_per_reception', 'yards_per_target', 'catch_rate'],
        'Snap Efficiency': ['yards_per_snap', 'targets_per_snap']
    }

    results = []

    # Overall correlations
    for category, metrics in {**volume_metrics, **efficiency_metrics}.items():
        for metric in metrics:
            if metric not in df.columns:
                continue

            valid_mask = df[metric].notna() & df['next_game_yards'].notna()

            if valid_mask.sum() < 30:
                continue

            pearson_corr, pearson_pval = pearsonr(
                df.loc[valid_mask, metric],
                df.loc[valid_mask, 'next_game_yards']
            )

            results.append({
                'category': category,
                'metric': metric,
                'position': 'ALL',
                'correlation': pearson_corr,
                'p_value': pearson_pval,
                'sample_size': valid_mask.sum(),
                'metric_type': 'Volume' if category in volume_metrics else 'Efficiency'
            })

            # Position-specific
            for position in ['WR', 'TE', 'RB']:
                pos_mask = (df['plyr_pos'] == position) & valid_mask

                if pos_mask.sum() < 30:
                    continue

                pearson_corr, pearson_pval = pearsonr(
                    df.loc[pos_mask, metric],
                    df.loc[pos_mask, 'next_game_yards']
                )

                results.append({
                    'category': category,
                    'metric': metric,
                    'position': position,
                    'correlation': pearson_corr,
                    'p_value': pearson_pval,
                    'sample_size': pos_mask.sum(),
                    'metric_type': 'Volume' if category in volume_metrics else 'Efficiency'
                })

    df_corr = pd.DataFrame(results)

    print(f"  Calculated {len(df_corr):,} correlation coefficients")

    # Summary statistics
    print("\n  Volume vs Efficiency Summary (ALL positions):")
    summary = df_corr[df_corr['position'] == 'ALL'].groupby('metric_type')['correlation'].agg(['mean', 'median', 'max'])
    print(summary)

    return df_corr


def identify_player_archetypes(df_stability):
    """Cluster players into archetypes based on volume and efficiency"""
    print(f"\n[6/7] Identifying player archetypes")

    # Use mean values for clustering
    cluster_features = [
        'target_share_mean',
        'yards_per_target_mean'
    ]

    df_cluster = df_stability[cluster_features + ['plyr_id', 'player_name', 'position']].dropna()

    if len(df_cluster) < 10:
        print("  Insufficient data for clustering")
        return pd.DataFrame()

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster[cluster_features])

    # K-means clustering (4 archetypes)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_cluster['archetype'] = kmeans.fit_predict(X_scaled)

    # Calculate cluster centers in original scale
    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=cluster_features
    )

    # Assign archetype labels based on characteristics
    def assign_archetype_label(row):
        target_share = row['target_share_mean']
        yds_per_target = row['yards_per_target_mean']

        if target_share > df_cluster['target_share_mean'].median():
            if yds_per_target > df_cluster['yards_per_target_mean'].median():
                return 'Elite (High Volume, High Efficiency)'
            else:
                return 'Possession (High Volume, Low Efficiency)'
        else:
            if yds_per_target > df_cluster['yards_per_target_mean'].median():
                return 'Deep Threat (Low Volume, High Efficiency)'
            else:
                return 'Role Player (Low Volume, Low Efficiency)'

    df_cluster['archetype_label'] = df_cluster.apply(assign_archetype_label, axis=1)

    print(f"  Clustered {len(df_cluster):,} players into archetypes")
    print("\n  Archetype distribution:")
    print(df_cluster['archetype_label'].value_counts())

    return df_cluster


def create_visualizations(df, df_corr, df_cluster):
    """Create comprehensive visualizations"""
    print(f"\n[7/7] Creating visualizations")

    # 1. Volume vs Efficiency Scatter
    print("  Creating volume vs efficiency scatter plot...")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    positions = ['WR', 'TE', 'RB']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for idx, (position, color) in enumerate(zip(positions, colors)):
        df_pos = df[
            (df['plyr_pos'] == position) &
            (df['target_share'].notna()) &
            (df['yards_per_target'].notna()) &
            (df['plyr_gm_rec_yds'].notna())
        ]

        scatter = axes[idx].scatter(
            df_pos['target_share'],
            df_pos['yards_per_target'],
            c=df_pos['plyr_gm_rec_yds'],
            s=50,
            alpha=0.6,
            cmap='viridis',
            edgecolors='black',
            linewidth=0.5
        )

        axes[idx].set_title(f'{position} Position', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Target Share', fontsize=12)
        axes[idx].set_ylabel('Yards per Target', fontsize=12)
        axes[idx].grid(True, alpha=0.3)

        # Add median lines
        axes[idx].axvline(df_pos['target_share'].median(), color='red', linestyle='--', alpha=0.5)
        axes[idx].axhline(df_pos['yards_per_target'].median(), color='red', linestyle='--', alpha=0.5)

        plt.colorbar(scatter, ax=axes[idx], label='Receiving Yards')

    plt.suptitle('Volume vs Efficiency: Target Share vs Yards per Target', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'volume_efficiency_scatter.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: {VIZ_DIR / 'volume_efficiency_scatter.png'}")
    plt.close()

    # 2. Predictive Power Comparison
    print("  Creating predictive power comparison plot...")

    df_all = df_corr[df_corr['position'] == 'ALL'].sort_values('correlation', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 10))

    colors_map = {'Volume': '#FF6B6B', 'Efficiency': '#4ECDC4'}
    colors_list = [colors_map[mt] for mt in df_all['metric_type']]

    ax.barh(range(len(df_all)), df_all['correlation'], color=colors_list, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(df_all)))
    ax.set_yticklabels(df_all['metric'].str.replace('_', ' ').str.title(), fontsize=10)
    ax.set_xlabel('Pearson Correlation with Next Game Yards', fontsize=12)
    ax.set_title('Predictive Power: Volume vs Efficiency Metrics', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', alpha=0.7, label='Volume'),
        Patch(facecolor='#4ECDC4', alpha=0.7, label='Efficiency')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'volume_vs_efficiency_predictive_power.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: {VIZ_DIR / 'volume_vs_efficiency_predictive_power.png'}")
    plt.close()

    # 3. Player Archetypes Visualization
    if not df_cluster.empty:
        print("  Creating player archetypes plot...")

        fig, ax = plt.subplots(figsize=(14, 10))

        archetype_colors = {
            'Elite (High Volume, High Efficiency)': '#2ECC71',
            'Possession (High Volume, Low Efficiency)': '#3498DB',
            'Deep Threat (Low Volume, High Efficiency)': '#E74C3C',
            'Role Player (Low Volume, Low Efficiency)': '#95A5A6'
        }

        for archetype, color in archetype_colors.items():
            df_arch = df_cluster[df_cluster['archetype_label'] == archetype]

            ax.scatter(
                df_arch['target_share_mean'],
                df_arch['yards_per_target_mean'],
                label=archetype,
                s=100,
                alpha=0.6,
                c=color,
                edgecolors='black',
                linewidth=1
            )

        ax.set_xlabel('Average Target Share', fontsize=12)
        ax.set_ylabel('Average Yards per Target', fontsize=12)
        ax.set_title('Player Archetypes: Volume vs Efficiency', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add quadrant lines
        ax.axvline(df_cluster['target_share_mean'].median(), color='black', linestyle='--', alpha=0.3)
        ax.axhline(df_cluster['yards_per_target_mean'].median(), color='black', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(VIZ_DIR / 'player_archetypes.png', dpi=300, bbox_inches='tight')
        print(f"    Saved: {VIZ_DIR / 'player_archetypes.png'}")
        plt.close()

    # 4. Position-specific correlation comparison
    print("  Creating position-specific comparison plot...")

    df_pos_corr = df_corr[df_corr['position'].isin(['WR', 'TE', 'RB'])]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, position in enumerate(['WR', 'TE', 'RB']):
        df_p = df_pos_corr[df_pos_corr['position'] == position].sort_values('correlation', ascending=True)

        colors_list = [colors_map[mt] for mt in df_p['metric_type']]

        axes[idx].barh(range(len(df_p)), df_p['correlation'], color=colors_list, alpha=0.7, edgecolor='black')
        axes[idx].set_yticks(range(len(df_p)))
        axes[idx].set_yticklabels(df_p['metric'].str.replace('_', ' ').str.title(), fontsize=8)
        axes[idx].set_xlabel('Correlation', fontsize=10)
        axes[idx].set_title(f'{position} Position', fontsize=12, fontweight='bold')
        axes[idx].grid(axis='x', alpha=0.3)
        axes[idx].axvline(x=0, color='black', linestyle='-', linewidth=0.8)

    plt.suptitle('Position-Specific: Volume vs Efficiency Predictive Power', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'position_volume_efficiency_comparison.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: {VIZ_DIR / 'position_volume_efficiency_comparison.png'}")
    plt.close()


def save_results(df_corr, df_cluster):
    """Save analysis results"""
    print(f"\nSaving results...")

    # Save correlations
    output_file = PRED_DIR / 'volume_efficiency_correlations.csv'
    df_corr.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")

    # Save archetypes
    if not df_cluster.empty:
        output_file = PRED_DIR / 'player_archetypes.csv'
        df_cluster.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("VOLUME VS EFFICIENCY: KEY FINDINGS")
    print("=" * 80)

    print("\nOverall Predictive Power (ALL positions):")
    summary = df_corr[df_corr['position'] == 'ALL'].groupby('metric_type').agg({
        'correlation': ['mean', 'median', 'max']
    }).round(3)
    print(summary)

    print("\n\nTop 5 Volume Metrics:")
    top_volume = df_corr[
        (df_corr['position'] == 'ALL') &
        (df_corr['metric_type'] == 'Volume')
    ].nlargest(5, 'correlation')[['metric', 'correlation', 'p_value']]
    print(top_volume.to_string(index=False))

    print("\n\nTop 5 Efficiency Metrics:")
    top_efficiency = df_corr[
        (df_corr['position'] == 'ALL') &
        (df_corr['metric_type'] == 'Efficiency')
    ].nlargest(5, 'correlation')[['metric', 'correlation', 'p_value']]
    print(top_efficiency.to_string(index=False))


def main():
    """Main execution function"""

    # Load data
    df_rec, df_players, df_snaps = load_data(seasons=[2022, 2023, 2024])

    # Calculate team aggregates
    team_totals = calculate_team_aggregates(df_rec)

    # Prepare dataset
    df = prepare_analysis_dataset(df_rec, df_players, df_snaps, team_totals)

    # Calculate stability metrics
    df_stability = calculate_stability_metrics(df)

    # Compare predictive power
    df_corr = compare_predictive_power(df)

    # Identify archetypes
    df_cluster = identify_player_archetypes(df_stability)

    # Create visualizations
    create_visualizations(df, df_corr, df_cluster)

    # Save results
    save_results(df_corr, df_cluster)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
