"""
Create Temporal Train/Validation/Test Splits for WR Receiving Yards Prediction

This script creates temporally-separated dataset splits:
- TRAIN: 2022-2023 seasons (historical data)
- VALIDATION: 2024 season (for hyperparameter tuning)
- TEST: 2025 season (for final model evaluation)

CRITICAL: No temporal overlap between splits. Test on future data only.

Input: data/processed/player_features/wr_baseline_features.parquet
Outputs:
  - data/splits/temporal/wr_train.parquet
  - data/splits/temporal/wr_validation.parquet
  - data/splits/temporal/wr_test.parquet
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import json

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from utils.path_manager import PathManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / '03_create_train_val_split.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_feature_data(input_path: Path) -> pd.DataFrame:
    """
    Load feature dataset from parquet.

    Args:
        input_path: Path to feature parquet file

    Returns:
        DataFrame with engineered features
    """
    logger.info(f"Loading feature data from: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Feature data not found: {input_path}")

    df = pd.read_parquet(input_path)

    logger.info(f"  Loaded {len(df):,} records")
    logger.info(f"  Columns: {len(df.columns)}")
    logger.info(f"  Seasons: {sorted(df['season'].unique())}")

    return df


def create_temporal_splits(
    df: pd.DataFrame,
    train_seasons: list = [2022, 2023],
    val_seasons: list = [2024],
    test_seasons: list = [2025]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test sets based on season.

    Args:
        df: Input dataframe with all data
        train_seasons: Seasons for training
        val_seasons: Seasons for validation
        test_seasons: Seasons for testing

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("Creating temporal splits...")
    logger.info(f"  Train seasons: {train_seasons}")
    logger.info(f"  Validation seasons: {val_seasons}")
    logger.info(f"  Test seasons: {test_seasons}")

    # Create splits
    train_df = df[df['season'].isin(train_seasons)].copy()
    val_df = df[df['season'].isin(val_seasons)].copy()
    test_df = df[df['season'].isin(test_seasons)].copy()

    logger.info(f"\nSplit sizes:")
    logger.info(f"  Train: {len(train_df):,} records ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"  Validation: {len(val_df):,} records ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"  Test: {len(test_df):,} records ({len(test_df)/len(df)*100:.1f}%)")

    # Verify no overlap
    train_games = set(train_df['game_id'].unique())
    val_games = set(val_df['game_id'].unique())
    test_games = set(test_df['game_id'].unique())

    overlap_train_val = train_games & val_games
    overlap_val_test = val_games & test_games
    overlap_train_test = train_games & test_games

    if overlap_train_val or overlap_val_test or overlap_train_test:
        logger.error("CRITICAL: Temporal overlap detected between splits!")
        if overlap_train_val:
            logger.error(f"  Train-Val overlap: {len(overlap_train_val)} games")
        if overlap_val_test:
            logger.error(f"  Val-Test overlap: {len(overlap_val_test)} games")
        if overlap_train_test:
            logger.error(f"  Train-Test overlap: {len(overlap_train_test)} games")
        raise ValueError("Temporal overlap between splits detected!")

    logger.info("  No temporal overlap between splits - VERIFIED")

    return train_df, val_df, test_df


def analyze_target_distribution(
    df: pd.DataFrame,
    split_name: str,
    target_col: str = 'plyr_gm_rec_yds'
) -> Dict[str, Any]:
    """
    Analyze target variable distribution for a split.

    Args:
        df: DataFrame for a specific split
        split_name: Name of the split (for logging)
        target_col: Name of target column

    Returns:
        Dictionary with distribution statistics
    """
    logger.info(f"\nAnalyzing {split_name} target distribution...")

    stats = {}

    if target_col not in df.columns:
        logger.warning(f"  Target column '{target_col}' not found")
        return stats

    target = df[target_col]

    # Basic statistics
    stats['count'] = int(target.notna().sum())
    stats['null_count'] = int(target.isna().sum())
    stats['null_rate'] = float(target.isna().mean())
    stats['mean'] = float(target.mean())
    stats['std'] = float(target.std())
    stats['min'] = float(target.min())
    stats['max'] = float(target.max())

    # Percentiles
    percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    for p in percentiles:
        stats[f'p{p}'] = float(target.quantile(p/100))

    # Log statistics
    logger.info(f"  Records: {stats['count']:,}")
    logger.info(f"  Null rate: {stats['null_rate']*100:.2f}%")
    logger.info(f"  Mean: {stats['mean']:.2f} yards")
    logger.info(f"  Std: {stats['std']:.2f} yards")
    logger.info(f"  Min: {stats['min']:.0f} yards")
    logger.info(f"  Max: {stats['max']:.0f} yards")
    logger.info(f"  Median (p50): {stats['p50']:.0f} yards")
    logger.info(f"  P25-P75: [{stats['p25']:.0f}, {stats['p75']:.0f}] yards")

    # Distribution shape analysis
    zero_count = (target == 0).sum()
    stats['zero_games'] = int(zero_count)
    stats['zero_rate'] = float(zero_count / len(target))

    logger.info(f"  Zero-yard games: {stats['zero_games']:,} ({stats['zero_rate']*100:.1f}%)")

    # High performers (>100 yards)
    high_perf = (target > 100).sum()
    stats['games_over_100yds'] = int(high_perf)
    stats['rate_over_100yds'] = float(high_perf / len(target))

    logger.info(f"  Games >100 yards: {stats['games_over_100yds']:,} ({stats['rate_over_100yds']*100:.1f}%)")

    return stats


def analyze_feature_coverage(
    df: pd.DataFrame,
    split_name: str
) -> Dict[str, Any]:
    """
    Analyze feature availability and null rates.

    Args:
        df: DataFrame for a specific split
        split_name: Name of the split (for logging)

    Returns:
        Dictionary with feature coverage statistics
    """
    logger.info(f"\nAnalyzing {split_name} feature coverage...")

    stats = {}

    # Identify feature columns
    metadata_cols = [
        'plyr_id', 'game_id', 'season', 'week', 'season_id', 'week_id',
        'plyr_name', 'team_id', 'opponent_id', 'game_date', 'plyr_pos'
    ]
    target_col = 'plyr_gm_rec_yds'

    feature_cols = [col for col in df.columns
                   if col not in metadata_cols and col != target_col]

    stats['feature_count'] = len(feature_cols)

    # Null rates
    null_rates = {}
    high_null_features = []

    for col in feature_cols:
        null_rate = df[col].isna().mean()
        null_rates[col] = float(null_rate)

        if null_rate > 0.50:
            high_null_features.append((col, null_rate))

    stats['null_rates'] = null_rates
    stats['high_null_features'] = [feat for feat, _ in high_null_features]
    stats['high_null_count'] = len(high_null_features)

    logger.info(f"  Total features: {stats['feature_count']}")
    logger.info(f"  Features with >50% nulls: {stats['high_null_count']}")

    if len(high_null_features) > 0:
        logger.warning(f"  High-null features:")
        for feat, rate in high_null_features[:10]:
            logger.warning(f"    {feat}: {rate*100:.1f}% null")

    # Average null rate across all features
    avg_null_rate = np.mean(list(null_rates.values()))
    stats['avg_null_rate'] = float(avg_null_rate)

    logger.info(f"  Average null rate across features: {avg_null_rate*100:.2f}%")

    return stats


def analyze_temporal_boundaries(
    df: pd.DataFrame,
    split_name: str
) -> Dict[str, Any]:
    """
    Analyze temporal boundaries of a split.

    Args:
        df: DataFrame for a specific split
        split_name: Name of the split

    Returns:
        Dictionary with temporal statistics
    """
    logger.info(f"\nAnalyzing {split_name} temporal boundaries...")

    stats = {}

    # Season coverage
    seasons = sorted(df['season'].unique())
    stats['seasons'] = seasons
    stats['season_range'] = f"{seasons[0]}-{seasons[-1]}"

    # Week coverage
    weeks = sorted(df['week'].unique())
    stats['weeks'] = weeks
    stats['week_range'] = f"{weeks[0]}-{weeks[-1]}"

    # Game date range (if available)
    if 'game_date' in df.columns:
        min_date = df['game_date'].min()
        max_date = df['game_date'].max()
        stats['date_range'] = f"{min_date} to {max_date}"

        logger.info(f"  Date range: {stats['date_range']}")

    logger.info(f"  Seasons: {stats['seasons']}")
    logger.info(f"  Weeks: {stats['week_range']}")

    # Games per season
    games_per_season = df.groupby('season')['game_id'].nunique()
    stats['games_per_season'] = games_per_season.to_dict()

    for season, game_count in games_per_season.items():
        logger.info(f"  Season {season}: {game_count} unique games")

    # Players per season
    players_per_season = df.groupby('season')['plyr_id'].nunique()
    stats['players_per_season'] = players_per_season.to_dict()

    for season, player_count in players_per_season.items():
        logger.info(f"  Season {season}: {player_count} unique players")

    return stats


def compare_distributions(
    train_stats: Dict[str, Any],
    val_stats: Dict[str, Any],
    test_stats: Dict[str, Any]
) -> None:
    """
    Compare distributions across splits to detect distributional shift.

    Args:
        train_stats: Training set statistics
        val_stats: Validation set statistics
        test_stats: Test set statistics
    """
    logger.info("\n" + "="*80)
    logger.info("DISTRIBUTION COMPARISON ACROSS SPLITS")
    logger.info("="*80)

    # Compare target distributions
    logger.info("\nTarget Variable (plyr_gm_rec_yds) Comparison:")
    logger.info(f"  {'Metric':<20} {'Train':>12} {'Validation':>12} {'Test':>12}")
    logger.info("  " + "-"*60)

    metrics = ['mean', 'std', 'p50', 'p25', 'p75', 'zero_rate', 'rate_over_100yds']

    for metric in metrics:
        train_val = train_stats.get(metric, np.nan)
        val_val = val_stats.get(metric, np.nan)
        test_val = test_stats.get(metric, np.nan)

        if metric in ['zero_rate', 'rate_over_100yds']:
            # Format as percentage
            logger.info(f"  {metric:<20} {train_val*100:>11.1f}% {val_val*100:>11.1f}% {test_val*100:>11.1f}%")
        else:
            # Format as float
            logger.info(f"  {metric:<20} {train_val:>12.2f} {val_val:>12.2f} {test_val:>12.2f}")

    # Flag significant distribution shifts
    logger.info("\nDistribution Shift Analysis:")

    # Check mean difference
    train_mean = train_stats.get('mean', 0)
    val_mean = val_stats.get('mean', 0)
    test_mean = test_stats.get('mean', 0)

    if train_mean > 0:
        val_shift = abs((val_mean - train_mean) / train_mean)
        test_shift = abs((test_mean - train_mean) / train_mean)

        logger.info(f"  Train-Val mean shift: {val_shift*100:.1f}%")
        logger.info(f"  Train-Test mean shift: {test_shift*100:.1f}%")

        if val_shift > 0.10:
            logger.warning(f"  WARNING: Validation set shows >{10}% mean shift from training")
        if test_shift > 0.10:
            logger.warning(f"  WARNING: Test set shows >{10}% mean shift from training")

    # Compare feature null rates
    logger.info("\nFeature Coverage Comparison:")
    logger.info(f"  {'Split':<15} {'Avg Null Rate':>15} {'High-Null Features':>20}")
    logger.info("  " + "-"*52)

    for split_name, stats in [('Train', train_stats), ('Validation', val_stats), ('Test', test_stats)]:
        avg_null = stats.get('avg_null_rate', 0)
        high_null_count = stats.get('high_null_count', 0)
        logger.info(f"  {split_name:<15} {avg_null*100:>14.2f}% {high_null_count:>20}")


def save_split(df: pd.DataFrame, output_path: Path, split_name: str) -> None:
    """
    Save a data split to parquet.

    Args:
        df: DataFrame to save
        output_path: Path to save parquet file
        split_name: Name of the split (for logging)
    """
    logger.info(f"\nSaving {split_name} split to: {output_path}")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to parquet
    df.to_parquet(output_path, index=False, engine='pyarrow')

    # Verify file was created
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"  File saved successfully: {file_size_mb:.2f} MB")


def save_split_statistics(
    stats: Dict[str, Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Save split statistics to JSON file.

    Args:
        stats: Dictionary with statistics for all splits
        output_path: Path to save JSON file
    """
    logger.info(f"\nSaving split statistics to: {output_path}")

    # Convert numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj

    stats_clean = convert_types(stats)

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(stats_clean, f, indent=2)

    logger.info(f"  Statistics saved successfully")


def main():
    """Main execution function."""
    logger.info("="*80)
    logger.info("CREATE TEMPORAL TRAIN/VALIDATION/TEST SPLITS")
    logger.info("="*80)

    try:
        # Initialize paths
        logger.info("\n1. Initializing paths...")
        paths = PathManager()

        # Load feature data
        logger.info("\n2. Loading feature data...")
        input_path = paths.get('data', 'processed') / 'player_features' / 'wr_baseline_features.parquet'
        df = load_feature_data(input_path)

        # Create temporal splits
        logger.info("\n3. Creating temporal splits...")
        train_df, val_df, test_df = create_temporal_splits(
            df,
            train_seasons=[2022, 2023],
            val_seasons=[2024],
            test_seasons=[2025]
        )

        # Analyze each split
        logger.info("\n4. Analyzing splits...")

        # Target distribution
        train_target_stats = analyze_target_distribution(train_df, "TRAIN")
        val_target_stats = analyze_target_distribution(val_df, "VALIDATION")
        test_target_stats = analyze_target_distribution(test_df, "TEST")

        # Feature coverage
        train_feature_stats = analyze_feature_coverage(train_df, "TRAIN")
        val_feature_stats = analyze_feature_coverage(val_df, "VALIDATION")
        test_feature_stats = analyze_feature_coverage(test_df, "TEST")

        # Temporal boundaries
        train_temporal_stats = analyze_temporal_boundaries(train_df, "TRAIN")
        val_temporal_stats = analyze_temporal_boundaries(val_df, "VALIDATION")
        test_temporal_stats = analyze_temporal_boundaries(test_df, "TEST")

        # Compare distributions
        logger.info("\n5. Comparing distributions across splits...")
        compare_distributions(train_target_stats, val_target_stats, test_target_stats)

        # Combine all statistics
        all_stats = {
            'train': {
                'target': train_target_stats,
                'features': train_feature_stats,
                'temporal': train_temporal_stats
            },
            'validation': {
                'target': val_target_stats,
                'features': val_feature_stats,
                'temporal': val_temporal_stats
            },
            'test': {
                'target': test_target_stats,
                'features': test_feature_stats,
                'temporal': test_temporal_stats
            }
        }

        # Save splits
        logger.info("\n6. Saving data splits...")
        output_dir = paths.get('data', 'splits_temporal')

        save_split(train_df, output_dir / 'wr_train.parquet', 'TRAIN')
        save_split(val_df, output_dir / 'wr_validation.parquet', 'VALIDATION')
        save_split(test_df, output_dir / 'wr_test.parquet', 'TEST')

        # Save statistics
        stats_path = output_dir / 'split_statistics.json'
        save_split_statistics(all_stats, stats_path)

        logger.info("\n" + "="*80)
        logger.info("TEMPORAL SPLIT CREATION COMPLETE")
        logger.info("="*80)

        logger.info(f"\nOutput files:")
        logger.info(f"  Train: {output_dir / 'wr_train.parquet'}")
        logger.info(f"  Validation: {output_dir / 'wr_validation.parquet'}")
        logger.info(f"  Test: {output_dir / 'wr_test.parquet'}")
        logger.info(f"  Statistics: {stats_path}")

        logger.info(f"\nFinal dataset sizes:")
        logger.info(f"  Train: {len(train_df):,} records")
        logger.info(f"  Validation: {len(val_df):,} records")
        logger.info(f"  Test: {len(test_df):,} records")
        logger.info(f"  Total: {len(df):,} records")

        return 0

    except Exception as e:
        logger.error(f"\nError in split creation: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
