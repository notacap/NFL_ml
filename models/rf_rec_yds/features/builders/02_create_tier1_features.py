"""
Create Tier 1 Features for WR Receiving Yards Prediction

This script engineers time-series features WITHOUT data leakage:
- Rolling performance metrics (3-game, 5-game windows)
- Target share metrics
- Opponent strength indicators
- Game context features

CRITICAL: For week N predictions, use ONLY data from weeks 1 to N-1.
Week 1 games have no historical data (features = 0 or NaN).

Input: data/processed/player_features/wr_baseline_raw.parquet
Output: data/processed/player_features/wr_baseline_features.parquet
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from utils.path_manager import PathManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / '02_create_tier1_features.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_baseline_data(input_path: Path) -> pd.DataFrame:
    """
    Load baseline data from parquet.

    Args:
        input_path: Path to baseline parquet file

    Returns:
        DataFrame with baseline data
    """
    logger.info(f"Loading baseline data from: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Baseline data not found: {input_path}")

    df = pd.read_parquet(input_path)

    logger.info(f"  Loaded {len(df):,} records")
    logger.info(f"  Columns: {len(df.columns)}")
    logger.info(f"  Date range: {df['season'].min()}-{df['season'].max()}, "
                f"weeks {df['week'].min()}-{df['week'].max()}")

    return df


def create_temporal_sort_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a temporal sorting key for proper time-series ordering.

    This ensures features are computed in chronological order within each season.

    Args:
        df: Input dataframe

    Returns:
        DataFrame with temporal sort key added
    """
    logger.info("Creating temporal sort key...")

    # Sort by player, season, and week to ensure proper temporal ordering
    df = df.sort_values(['plyr_id', 'season', 'week']).reset_index(drop=True)

    logger.info("  Data sorted by player, season, week")

    return df


def create_rolling_receiving_features(
    df: pd.DataFrame,
    windows: List[int] = [3, 5]
) -> pd.DataFrame:
    """
    Create rolling window features for receiving performance.

    CRITICAL: Uses expanding window to prevent leakage. For week N,
    only data from weeks 1 to N-1 are used.

    Args:
        df: Input dataframe (must be temporally sorted)
        windows: List of window sizes for rolling calculations

    Returns:
        DataFrame with rolling features added
    """
    logger.info(f"Creating rolling receiving features (windows: {windows})...")

    # Key receiving metrics to compute rolling stats
    metrics = {
        'plyr_gm_rec_yds': 'receiving_yards',
        'plyr_gm_rec_tgt': 'targets',
        'plyr_gm_rec': 'receptions'
    }

    # Verify all metrics exist
    for col in metrics.keys():
        if col not in df.columns:
            logger.warning(f"  Metric '{col}' not found in dataframe, skipping")
            continue

        # Create rolling features for each window size
        for window in windows:
            # Use shift(1) to prevent leakage - only use PRIOR games
            # For week N, this computes stats from N-window to N-1
            for metric_col, metric_name in metrics.items():
                if metric_col not in df.columns:
                    continue

                feature_name = f'{metric_name}_last{window}'

                # Rolling mean over last N games (excluding current game)
                df[feature_name] = (
                    df.groupby('plyr_id')[metric_col]
                    .shift(1)  # Exclude current game
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )

                # Track null rate
                null_rate = df[feature_name].isna().mean()
                logger.info(f"  Created {feature_name} (null rate: {null_rate*100:.2f}%)")

    logger.info(f"  Rolling features created for {len(metrics)} metrics")

    return df


def create_target_share_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target share features (player % of team targets).

    CRITICAL: For week N, compute target share using ONLY prior games (weeks 1 to N-1).

    Args:
        df: Input dataframe with rolling features

    Returns:
        DataFrame with target share features added
    """
    logger.info("Creating target share features...")

    if 'plyr_gm_rec_tgt' not in df.columns:
        logger.warning("  Target column 'plyr_gm_rec_tgt' not found, skipping target share")
        return df

    # We need team-level target totals per game
    # This requires aggregating all players on the team for each game

    # First, compute team total targets per game
    team_totals = df.groupby(['game_id', 'team_id'])['plyr_gm_rec_tgt'].sum().reset_index()
    team_totals = team_totals.rename(columns={'plyr_gm_rec_tgt': 'team_total_targets'})

    # Join team totals back to main dataframe
    df = df.merge(team_totals, on=['game_id', 'team_id'], how='left')

    # For rolling target share, we need to:
    # 1. Compute target share for each game
    # 2. Take rolling average of target share (excluding current game)

    # Current game target share (will be shifted)
    df['target_share_current'] = df['plyr_gm_rec_tgt'] / df['team_total_targets'].replace(0, np.nan)

    # Rolling 3-game average of target share (excluding current game)
    df['target_share_last3'] = (
        df.groupby('plyr_id')['target_share_current']
        .shift(1)  # Exclude current game
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    null_rate = df['target_share_last3'].isna().mean()
    logger.info(f"  Created target_share_last3 (null rate: {null_rate*100:.2f}%)")

    # Drop temporary columns
    df = df.drop(columns=['target_share_current', 'team_total_targets'])

    return df


def create_opponent_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create opponent defensive strength features.

    These features are already from prior weeks (joined in baseline script),
    but we create additional derived features here.

    Args:
        df: Input dataframe

    Returns:
        DataFrame with opponent strength features added
    """
    logger.info("Creating opponent defensive strength features...")

    # Check if opponent defense stats are available
    opp_def_cols = [col for col in df.columns if col.startswith('opp_tm_def_')]

    if len(opp_def_cols) == 0:
        logger.warning("  No opponent defense columns found, skipping")
        return df

    logger.info(f"  Found {len(opp_def_cols)} opponent defense columns")

    # Key metric: opponent pass yards allowed per game
    if 'opp_tm_def_pass_ypg' in df.columns:
        # Create defensive rank within each week (1 = best defense, higher = worse)
        df['opp_defense_rank'] = (
            df.groupby(['season', 'week'])['opp_tm_def_pass_ypg']
            .rank(method='dense', ascending=True)  # Lower yards = better = lower rank
        )

        null_rate = df['opp_defense_rank'].isna().mean()
        logger.info(f"  Created opp_defense_rank (null rate: {null_rate*100:.2f}%)")

    # Rename key opponent metrics for clarity
    if 'opp_tm_def_pass_yds' in df.columns:
        df = df.rename(columns={'opp_tm_def_pass_yds': 'opp_pass_yards_allowed_cumulative'})

    if 'opp_tm_def_pass_ypg' in df.columns:
        df = df.rename(columns={'opp_tm_def_pass_ypg': 'opp_pass_yards_allowed_per_game'})

    # Create rolling average of opponent pass yards allowed
    # Note: This is already cumulative from team stats, but we can create
    # a simpler "last 5 games" metric by tracking it per opponent over time

    # For simplicity, use the cumulative average from opponent team stats
    if 'opp_pass_yards_allowed_per_game' in df.columns:
        df['opp_pass_yards_allowed_last5'] = df['opp_pass_yards_allowed_per_game']

    return df


def create_game_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create game context features.

    These are static features about the game itself.

    Args:
        df: Input dataframe

    Returns:
        DataFrame with game context features added
    """
    logger.info("Creating game context features...")

    # is_home should already exist from baseline
    if 'is_home' not in df.columns:
        logger.warning("  'is_home' column not found, cannot create game context features")
        return df

    # Verify is_home is binary
    home_values = df['is_home'].unique()
    logger.info(f"  is_home values: {sorted(home_values)}")

    # Create season progress feature (week / 18)
    if 'week' in df.columns:
        df['season_progress'] = df['week'] / 18.0
        logger.info(f"  Created season_progress (range: {df['season_progress'].min():.2f} - {df['season_progress'].max():.2f})")

    # Days since season start (for fatigue/injury accumulation)
    if 'game_date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['game_date']):
        # Get first game date per season per team
        season_start = df.groupby(['season', 'team_id'])['game_date'].min().reset_index()
        season_start = season_start.rename(columns={'game_date': 'season_start_date'})

        df = df.merge(season_start, on=['season', 'team_id'], how='left')
        df['days_since_season_start'] = (df['game_date'] - df['season_start_date']).dt.days

        logger.info(f"  Created days_since_season_start")

        # Drop temporary column
        df = df.drop(columns=['season_start_date'])

    return df


def create_player_consistency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features measuring player performance consistency.

    Args:
        df: Input dataframe

    Returns:
        DataFrame with consistency features added
    """
    logger.info("Creating player consistency features...")

    if 'plyr_gm_rec_yds' not in df.columns:
        logger.warning("  Target variable not found, skipping consistency features")
        return df

    # Rolling standard deviation of receiving yards (measure of consistency)
    df['receiving_yards_std_last5'] = (
        df.groupby('plyr_id')['plyr_gm_rec_yds']
        .shift(1)  # Exclude current game
        .rolling(window=5, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Coefficient of variation (std / mean) for relative consistency
    df['receiving_yards_cv_last5'] = (
        df['receiving_yards_std_last5'] /
        df['receiving_yards_last5'].replace(0, np.nan)
    )

    null_rate_std = df['receiving_yards_std_last5'].isna().mean()
    null_rate_cv = df['receiving_yards_cv_last5'].isna().mean()

    logger.info(f"  Created receiving_yards_std_last5 (null rate: {null_rate_std*100:.2f}%)")
    logger.info(f"  Created receiving_yards_cv_last5 (null rate: {null_rate_cv*100:.2f}%)")

    return df


def detect_leakage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Automated leakage detection - verify temporal integrity.

    For week 1 games, rolling features should be null or zero.
    For week 2+, features should exist.

    Args:
        df: Dataframe with engineered features

    Returns:
        Dictionary with leakage test results
    """
    logger.info("Running automated leakage detection...")

    results = {}

    # Test 1: Week 1 games should have null/zero rolling features
    week1_df = df[df['week'] == 1].copy()

    if len(week1_df) > 0:
        rolling_features = [col for col in df.columns if 'last' in col]

        week1_null_rates = {}
        for feat in rolling_features:
            if feat in week1_df.columns:
                # Check if values are null OR zero (both acceptable for week 1)
                null_or_zero = (week1_df[feat].isna() | (week1_df[feat] == 0)).mean()
                week1_null_rates[feat] = null_or_zero

        # Week 1 should have high null/zero rate (>80%)
        problematic_features = {k: v for k, v in week1_null_rates.items() if v < 0.80}

        results['week1_test'] = {
            'passed': len(problematic_features) == 0,
            'problematic_features': problematic_features
        }

        if len(problematic_features) > 0:
            logger.warning(f"  LEAKAGE WARNING: Week 1 features have low null/zero rates!")
            for feat, rate in problematic_features.items():
                logger.warning(f"    {feat}: {rate*100:.1f}% null/zero (expected >80%)")
        else:
            logger.info(f"  Week 1 leakage test PASSED: All rolling features properly null/zero")

    # Test 2: Verify temporal ordering
    # For each player, game dates should be monotonically increasing
    if 'game_date' in df.columns and 'plyr_id' in df.columns:
        temporal_violations = 0
        for plyr_id in df['plyr_id'].unique()[:100]:  # Sample check
            plyr_df = df[df['plyr_id'] == plyr_id].sort_values(['season', 'week'])
            if 'game_date' in plyr_df.columns:
                if not plyr_df['game_date'].is_monotonic_increasing:
                    temporal_violations += 1

        results['temporal_order_test'] = {
            'passed': temporal_violations == 0,
            'violations': temporal_violations
        }

        if temporal_violations > 0:
            logger.warning(f"  LEAKAGE WARNING: Found {temporal_violations} temporal ordering violations!")
        else:
            logger.info(f"  Temporal ordering test PASSED")

    # Test 3: No future data in rolling features
    # For week N, rolling features should only use data from weeks 1 to N-1
    # This is hard to test directly, but we can check if week 2 has different values than week 1
    week2_df = df[df['week'] == 2].copy()
    if len(week2_df) > 0 and len(week1_df) > 0:
        # Week 2 should have MORE non-null rolling features than week 1
        rolling_features = [col for col in df.columns if 'last' in col and col in week2_df.columns]

        if len(rolling_features) > 0:
            week1_avg_null = week1_df[rolling_features].isna().mean().mean()
            week2_avg_null = week2_df[rolling_features].isna().mean().mean()

            results['progressive_feature_test'] = {
                'passed': week2_avg_null < week1_avg_null,
                'week1_null_rate': float(week1_avg_null),
                'week2_null_rate': float(week2_avg_null)
            }

            if week2_avg_null >= week1_avg_null:
                logger.warning(f"  LEAKAGE WARNING: Week 2 null rate >= Week 1 null rate!")
            else:
                logger.info(f"  Progressive feature test PASSED")

    logger.info("Leakage detection complete")

    return results


def validate_features(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate engineered features and log statistics.

    Args:
        df: Dataframe with engineered features

    Returns:
        Dictionary with validation statistics
    """
    logger.info("Validating engineered features...")

    stats = {}

    # Identify feature columns (exclude metadata and target)
    metadata_cols = [
        'plyr_id', 'game_id', 'season', 'week', 'season_id', 'week_id',
        'plyr_name', 'team_id', 'opponent_id', 'game_date', 'plyr_pos'
    ]

    target_col = 'plyr_gm_rec_yds'

    feature_cols = [col for col in df.columns
                   if col not in metadata_cols and col != target_col]

    stats['feature_count'] = len(feature_cols)
    stats['total_columns'] = len(df.columns)

    logger.info(f"  Total features: {stats['feature_count']}")

    # Null rates for each feature
    null_rates = {}
    high_null_features = []

    for col in feature_cols:
        null_rate = df[col].isna().mean()
        null_rates[col] = float(null_rate)

        if null_rate > 0.50:  # Flag features with >50% nulls
            high_null_features.append((col, null_rate))

    stats['null_rates'] = null_rates
    stats['high_null_features'] = high_null_features

    if len(high_null_features) > 0:
        logger.warning(f"  Found {len(high_null_features)} features with >50% null values:")
        for feat, rate in high_null_features[:10]:  # Show first 10
            logger.warning(f"    {feat}: {rate*100:.1f}%")
    else:
        logger.info(f"  No features with >50% null values")

    # Feature value ranges (for numeric features)
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    stats['numeric_feature_count'] = len(numeric_features)

    logger.info(f"  Numeric features: {len(numeric_features)}")

    # Check for infinite values
    inf_counts = {}
    for col in numeric_features:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = int(inf_count)

    stats['infinite_value_features'] = inf_counts

    if len(inf_counts) > 0:
        logger.warning(f"  Found {len(inf_counts)} features with infinite values:")
        for feat, count in inf_counts.items():
            logger.warning(f"    {feat}: {count} infinite values")
    else:
        logger.info(f"  No features with infinite values")

    # Distribution summary for key features
    key_features = [
        'receiving_yards_last3', 'receiving_yards_last5',
        'targets_last3', 'target_share_last3', 'opp_defense_rank'
    ]

    for feat in key_features:
        if feat in df.columns and df[feat].notna().sum() > 0:
            stats[f'{feat}_mean'] = float(df[feat].mean())
            stats[f'{feat}_std'] = float(df[feat].std())
            stats[f'{feat}_min'] = float(df[feat].min())
            stats[f'{feat}_max'] = float(df[feat].max())

    logger.info("\nKey Feature Statistics:")
    for feat in key_features:
        if feat in df.columns:
            mean = df[feat].mean()
            std = df[feat].std()
            logger.info(f"  {feat}: mean={mean:.2f}, std={std:.2f}")

    return stats


def save_features(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save feature dataset to parquet.

    Args:
        df: Feature dataframe
        output_path: Path to save parquet file
    """
    logger.info(f"Saving feature dataset to: {output_path}")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to parquet
    df.to_parquet(output_path, index=False, engine='pyarrow')

    # Verify file was created
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"  File saved successfully: {file_size_mb:.2f} MB")


def main():
    """Main execution function."""
    logger.info("="*80)
    logger.info("CREATE TIER 1 FEATURES - WR RECEIVING YARDS PREDICTION")
    logger.info("="*80)

    try:
        # Initialize paths
        logger.info("\n1. Initializing paths...")
        paths = PathManager()

        # Load baseline data
        logger.info("\n2. Loading baseline data...")
        input_path = paths.get('data', 'processed') / 'player_features' / 'wr_baseline_raw.parquet'
        df = load_baseline_data(input_path)

        # Prepare for feature engineering
        logger.info("\n3. Preparing temporal ordering...")
        df = create_temporal_sort_key(df)

        # Create features (ORDER MATTERS - features may depend on each other)
        logger.info("\n4. Creating rolling performance features...")
        df = create_rolling_receiving_features(df, windows=[3, 5])

        logger.info("\n5. Creating target share features...")
        df = create_target_share_features(df)

        logger.info("\n6. Creating opponent strength features...")
        df = create_opponent_strength_features(df)

        logger.info("\n7. Creating game context features...")
        df = create_game_context_features(df)

        logger.info("\n8. Creating player consistency features...")
        df = create_player_consistency_features(df)

        # Validate features
        logger.info("\n9. Validating features...")
        validation_stats = validate_features(df)

        # Leakage detection
        logger.info("\n10. Running leakage detection...")
        leakage_results = detect_leakage(df)

        # Save features
        logger.info("\n11. Saving feature dataset...")
        output_path = paths.get('data', 'processed') / 'player_features' / 'wr_baseline_features.parquet'
        save_features(df, output_path)

        logger.info("\n" + "="*80)
        logger.info("TIER 1 FEATURE ENGINEERING COMPLETE")
        logger.info("="*80)
        logger.info(f"\nOutput file: {output_path}")
        logger.info(f"Total records: {len(df):,}")
        logger.info(f"Total columns: {len(df.columns)}")
        logger.info(f"Feature columns: {validation_stats['feature_count']}")

        # Summary of leakage tests
        logger.info("\nLeakage Detection Summary:")
        for test_name, result in leakage_results.items():
            status = "PASSED" if result.get('passed', False) else "FAILED"
            logger.info(f"  {test_name}: {status}")

        return 0

    except Exception as e:
        logger.error(f"\nError in feature engineering: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
