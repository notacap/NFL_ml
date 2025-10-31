"""
Load and Join Baseline Data for WR Receiving Yards Prediction

This script loads and joins multiple data sources to create a baseline dataset
for predicting wide receiver receiving yards. It performs the following:

1. Load game-level receiving stats (contains target: plyr_gm_rec_yds)
2. Join player reference data (filter to WR position)
3. Join game information (home/away, opponent)
4. Join team defensive passing stats (opponent strength)
5. Join player season cumulative stats (prior season performance)

Output: data/processed/player_features/wr_baseline_raw.parquet

Temporal Range: 2022-2025 seasons
Position Filter: WR (Wide Receivers only)
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
from utils.data_loader import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / '01_load_baseline_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_receiving_game_stats(
    loader: DataLoader,
    seasons: List[int]
) -> pd.DataFrame:
    """
    Load player game-level receiving statistics.

    This contains the target variable (plyr_gm_rec_yds) and base receiving stats.

    Args:
        loader: DataLoader instance
        seasons: List of seasons to load

    Returns:
        DataFrame with game-level receiving stats
    """
    logger.info(f"Loading player game receiving stats for seasons: {seasons}")

    df = loader.load_table('player_game_receiving', seasons=seasons)

    logger.info(f"  Loaded {len(df):,} game records")
    logger.info(f"  Columns: {len(df.columns)}")
    logger.info(f"  Seasons present: {sorted(df['season'].unique())}")
    logger.info(f"  Weeks range: {df['week'].min()} to {df['week'].max()}")

    # Validate target variable exists
    if 'plyr_gm_rec_yds' not in df.columns:
        raise ValueError("Target variable 'plyr_gm_rec_yds' not found in receiving stats")

    return df


def load_player_reference(
    loader: DataLoader,
    seasons: List[int]
) -> pd.DataFrame:
    """
    Load player reference data and filter to Wide Receivers only.

    Args:
        loader: DataLoader instance
        seasons: List of seasons to load

    Returns:
        DataFrame with player info for WRs only
    """
    logger.info(f"Loading player reference data for seasons: {seasons}")

    df = loader.load_table('players', seasons=seasons)

    logger.info(f"  Loaded {len(df):,} player records (all positions)")

    # Filter to WR position only
    df_wr = df[df['plyr_pos'] == 'WR'].copy()

    logger.info(f"  Filtered to {len(df_wr):,} WR records")
    logger.info(f"  Unique WRs: {df_wr['plyr_id'].nunique()}")

    if len(df_wr) == 0:
        raise ValueError("No Wide Receiver records found in player data")

    # Select relevant columns
    cols_to_keep = [
        'plyr_id', 'season_id', 'plyr_name', 'plyr_pos', 'plyr_age',
        'plyr_height', 'plyr_weight', 'plyr_yrs_played', 'season'
    ]

    # Only keep columns that exist
    cols_to_keep = [col for col in cols_to_keep if col in df_wr.columns]
    df_wr = df_wr[cols_to_keep]

    return df_wr


def load_game_info(
    loader: DataLoader,
    seasons: List[int]
) -> pd.DataFrame:
    """
    Load game information (teams, home/away, scores).

    Args:
        loader: DataLoader instance
        seasons: List of seasons to load

    Returns:
        DataFrame with game information
    """
    logger.info(f"Loading game information for seasons: {seasons}")

    df = loader.load_table('game', seasons=seasons)

    logger.info(f"  Loaded {len(df):,} games")
    logger.info(f"  Seasons: {sorted(df['season'].unique())}")

    # Select relevant columns
    cols_to_keep = [
        'game_id', 'season_id', 'week_id', 'home_team_id', 'away_team_id',
        'home_team_score', 'away_team_score', 'game_date', 'season', 'week'
    ]

    cols_to_keep = [col for col in cols_to_keep if col in df.columns]
    df = df[cols_to_keep]

    return df


def load_team_defense_passing(
    loader: DataLoader,
    seasons: List[int]
) -> pd.DataFrame:
    """
    Load team season defensive passing stats (cumulative through each week).

    This represents opponent defensive strength against the pass.

    Args:
        loader: DataLoader instance
        seasons: List of seasons to load

    Returns:
        DataFrame with team defensive passing stats
    """
    logger.info(f"Loading team defensive passing stats for seasons: {seasons}")

    df = loader.load_table('team_season_defense_passing', seasons=seasons)

    logger.info(f"  Loaded {len(df):,} team-week records")
    logger.info(f"  Unique teams: {df['team_id'].nunique()}")

    # Select key defensive metrics
    cols_to_keep = [
        'team_id', 'season_id', 'week_id', 'season', 'week',
        'tm_def_pass_yds', 'tm_def_pass_td', 'tm_def_pass_cmp_pct',
        'tm_def_pass_yds_att', 'tm_def_pass_ypg', 'tm_def_pass_rtg',
        'tm_def_int', 'tm_def_sk'
    ]

    cols_to_keep = [col for col in cols_to_keep if col in df.columns]
    df = df[cols_to_keep]

    return df


def load_player_season_receiving(
    loader: DataLoader,
    seasons: List[int]
) -> pd.DataFrame:
    """
    Load player season cumulative receiving stats.

    These are cumulative totals through each week of the season.

    Args:
        loader: DataLoader instance
        seasons: List of seasons to load

    Returns:
        DataFrame with season cumulative receiving stats
    """
    logger.info(f"Loading player season receiving stats for seasons: {seasons}")

    # Need to handle potential schema issues with this table
    try:
        df = loader.load_table('player_season_receiving', seasons=seasons)

        logger.info(f"  Loaded {len(df):,} player-week records")
        logger.info(f"  Unique players: {df['plyr_id'].nunique()}")

        # Select relevant cumulative stats
        cols_to_keep = [
            'plyr_id', 'season_id', 'week_id', 'season', 'week',
            'plyr_rec_tgt', 'plyr_rec', 'plyr_rec_yds', 'plyr_rec_td',
            'plyr_rec_catch_pct', 'plyr_rec_yds_rec', 'plyr_rec_ypg',
            'game_count'
        ]

        cols_to_keep = [col for col in cols_to_keep if col in df.columns]
        df = df[cols_to_keep]

        return df

    except Exception as e:
        logger.warning(f"  Could not load player season receiving stats: {e}")
        logger.warning("  Continuing without season cumulative stats")
        return None


def join_player_info(
    game_stats: pd.DataFrame,
    player_info: pd.DataFrame
) -> pd.DataFrame:
    """
    Join player information to game stats and filter to WRs only.

    Args:
        game_stats: Game-level receiving stats
        player_info: Player reference data (WR only)

    Returns:
        DataFrame with player info joined
    """
    logger.info("Joining player information to game stats...")

    initial_count = len(game_stats)

    # Join on plyr_id and season
    df = game_stats.merge(
        player_info,
        on=['plyr_id', 'season'],
        how='inner',  # Only keep WRs
        suffixes=('', '_plyr')
    )

    logger.info(f"  Records before join: {initial_count:,}")
    logger.info(f"  Records after join: {len(df):,}")
    logger.info(f"  Records lost (non-WR): {initial_count - len(df):,}")
    logger.info(f"  Unique WRs in dataset: {df['plyr_id'].nunique()}")

    # Drop duplicate columns from join
    df = df.loc[:, ~df.columns.duplicated()]

    return df


def filter_week_one(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out week 1 games from the dataset.

    Week 1 games are excluded because there are no prior opponent defensive stats
    available for feature engineering. Week 1 defensive stats are still used as
    features for predicting week 2+ games.

    Args:
        df: DataFrame with all games including week 1

    Returns:
        DataFrame with week 1 games removed
    """
    logger.info("Filtering out week 1 games...")

    initial_count = len(df)
    initial_weeks = sorted(df['week'].unique())

    # Filter out week 1
    df_filtered = df[df['week'] > 1].copy()

    filtered_count = len(df_filtered)
    removed_count = initial_count - filtered_count
    filtered_weeks = sorted(df_filtered['week'].unique())

    logger.info(f"  Records before filtering: {initial_count:,}")
    logger.info(f"  Records after filtering: {filtered_count:,}")
    logger.info(f"  Records removed (week 1): {removed_count:,} ({removed_count/initial_count*100:.1f}%)")
    logger.info(f"  Week range before: {min(initial_weeks)} to {max(initial_weeks)}")
    logger.info(f"  Week range after: {min(filtered_weeks)} to {max(filtered_weeks)}")
    logger.info(f"  Reason: No prior opponent defense stats available for week 1")

    return df_filtered


def join_game_info(
    df: pd.DataFrame,
    game_info: pd.DataFrame
) -> pd.DataFrame:
    """
    Join game information and determine home/away status.

    Args:
        df: Current dataframe with player game stats
        game_info: Game information

    Returns:
        DataFrame with game context added
    """
    logger.info("Joining game information...")

    initial_count = len(df)

    # Join on game_id
    df = df.merge(
        game_info,
        on=['game_id', 'season', 'week'],
        how='left',
        suffixes=('', '_gm')
    )

    logger.info(f"  Records before join: {initial_count:,}")
    logger.info(f"  Records after join: {len(df):,}")

    # Determine if player's team is home or away
    df['is_home'] = (df['team_id'] == df['home_team_id']).astype(int)

    # Determine opponent team
    df['opponent_id'] = np.where(
        df['is_home'] == 1,
        df['away_team_id'],
        df['home_team_id']
    )

    logger.info(f"  Home games: {df['is_home'].sum():,} ({df['is_home'].mean()*100:.1f}%)")
    logger.info(f"  Away games: {(1-df['is_home']).sum():,} ({(1-df['is_home']).mean()*100:.1f}%)")

    # Drop duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    return df


def join_opponent_defense(
    df: pd.DataFrame,
    team_def: pd.DataFrame
) -> pd.DataFrame:
    """
    Join opponent team defensive statistics.

    Note: We join opponent defense stats from PRIOR week to avoid leakage.
    For week N prediction, use opponent's defensive stats through week N-1.

    IMPORTANT: Week 1 games should be filtered out BEFORE calling this function,
    since there are no prior week stats available. Week 1 defensive stats are
    still used as features for predicting week 2 games.

    Args:
        df: Current dataframe with player and game info (week 1 should be filtered out)
        team_def: Team defensive passing stats

    Returns:
        DataFrame with opponent defensive stats added
    """
    logger.info("Joining opponent defensive statistics...")

    initial_count = len(df)

    # For week N, use opponent defense stats through week N-1
    # This requires shifting the week
    team_def_shifted = team_def.copy()
    team_def_shifted['week_for_join'] = team_def_shifted['week'] + 1

    # Rename columns to indicate they are opponent stats
    opp_cols = {col: f'opp_{col}' for col in team_def_shifted.columns
                if col not in ['team_id', 'season', 'week', 'week_for_join', 'season_id']}
    team_def_shifted = team_def_shifted.rename(columns=opp_cols)

    # Join opponent defense from prior week
    # Week 1 games should already be filtered out, so all games should have valid joins
    df = df.merge(
        team_def_shifted,
        left_on=['opponent_id', 'season', 'week'],
        right_on=['team_id', 'season', 'week_for_join'],
        how='left',
        suffixes=('', '_oppdef')
    )

    logger.info(f"  Records before join: {initial_count:,}")
    logger.info(f"  Records after join: {len(df):,}")

    # Validate that opponent defense stats are complete (no nulls expected)
    if 'opp_tm_def_pass_yds' in df.columns:
        null_count = df['opp_tm_def_pass_yds'].isna().sum()
        null_rate = df['opp_tm_def_pass_yds'].isna().mean()
        logger.info(f"  Opponent defense null count: {null_count:,} ({null_rate*100:.2f}%)")

        if null_count > 0:
            logger.warning(f"  WARNING: Found {null_count:,} records with null opponent defense!")
            logger.warning(f"  This should be 0 if week 1 was properly filtered out")
            # Show which weeks have nulls
            null_weeks = df[df['opp_tm_def_pass_yds'].isna()]['week'].value_counts().sort_index()
            logger.warning(f"  Weeks with null opponent defense: {null_weeks.to_dict()}")
        else:
            logger.info(f"  [OK] All opponent defense stats are complete (no nulls)")

    # Drop duplicate/unnecessary columns
    cols_to_drop = ['team_id_oppdef', 'week_for_join']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    df = df.loc[:, ~df.columns.duplicated()]

    return df


def join_season_cumulative_stats(
    df: pd.DataFrame,
    season_stats: pd.DataFrame
) -> pd.DataFrame:
    """
    Join player season cumulative receiving stats.

    Note: For week N, use cumulative stats through week N-1 to avoid leakage.

    Args:
        df: Current dataframe
        season_stats: Player season cumulative stats

    Returns:
        DataFrame with season cumulative stats added
    """
    if season_stats is None or len(season_stats) == 0:
        logger.info("Skipping season cumulative stats (not available)")
        return df

    logger.info("Joining player season cumulative statistics...")

    initial_count = len(df)

    # For week N, use cumulative stats through week N-1
    season_stats_shifted = season_stats.copy()
    season_stats_shifted['week_for_join'] = season_stats_shifted['week'] + 1

    # Rename columns to indicate they are season cumulative
    szn_cols = {col: f'szn_cum_{col}' for col in season_stats_shifted.columns
                if col not in ['plyr_id', 'season', 'week', 'week_for_join', 'season_id']}
    season_stats_shifted = season_stats_shifted.rename(columns=szn_cols)

    # Join season cumulative from prior week
    df = df.merge(
        season_stats_shifted,
        left_on=['plyr_id', 'season', 'week'],
        right_on=['plyr_id', 'season', 'week_for_join'],
        how='left',
        suffixes=('', '_szn')
    )

    logger.info(f"  Records before join: {initial_count:,}")
    logger.info(f"  Records after join: {len(df):,}")

    # Calculate null rate (expected for week 1)
    if 'szn_cum_plyr_rec_yds' in df.columns:
        null_rate = df['szn_cum_plyr_rec_yds'].isna().mean()
        logger.info(f"  Season cumulative null rate: {null_rate*100:.1f}%")

    # Drop unnecessary columns
    df = df.drop(columns=['week_for_join'], errors='ignore')
    df = df.loc[:, ~df.columns.duplicated()]

    return df


def validate_week_filtering(df: pd.DataFrame) -> None:
    """
    Validate that week 1 has been properly filtered out and opponent defense is complete.

    Args:
        df: DataFrame after opponent defense join

    Raises:
        AssertionError: If week 1 games are still present
    """
    logger.info("Validating week filtering and opponent defense completeness...")

    # Check minimum week
    min_week = df['week'].min()
    max_week = df['week'].max()
    logger.info(f"  Week range: {min_week} to {max_week}")

    # Assert that week 1 is filtered out
    assert min_week >= 2, f"Week 1 should be filtered out! Found minimum week: {min_week}"
    logger.info(f"  [OK] Week 1 properly filtered out (min week = {min_week})")

    # Check for nulls in opponent defense columns
    opp_def_cols = [col for col in df.columns if col.startswith('opp_tm_def')]

    if len(opp_def_cols) > 0:
        logger.info(f"  Checking {len(opp_def_cols)} opponent defense columns for nulls...")

        total_nulls = 0
        null_columns = []

        for col in opp_def_cols:
            null_count = df[col].isna().sum()
            if null_count > 0:
                total_nulls += null_count
                null_columns.append((col, null_count))

        if null_columns:
            logger.error(f"  ERROR: Found nulls in opponent defense columns:")
            for col, count in null_columns:
                logger.error(f"    {col}: {count:,} nulls")
            raise ValueError(f"Found {total_nulls:,} total nulls in {len(null_columns)} opponent defense columns")
        else:
            logger.info(f"  [OK] All opponent defense columns are complete (zero nulls)")
    else:
        logger.warning(f"  WARNING: No opponent defense columns found to validate")

    logger.info(f"  Validation complete: Dataset ready for training")


def validate_baseline_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the baseline dataset and return statistics.

    Args:
        df: Baseline dataframe

    Returns:
        Dictionary with validation statistics
    """
    logger.info("Validating baseline dataset...")

    stats = {}

    # Basic shape
    stats['total_records'] = len(df)
    stats['total_columns'] = len(df.columns)
    stats['unique_players'] = df['plyr_id'].nunique()
    stats['unique_games'] = df['game_id'].nunique()

    # Temporal coverage
    stats['seasons'] = sorted(df['season'].unique().tolist())
    stats['weeks_range'] = (int(df['week'].min()), int(df['week'].max()))

    # Position verification
    if 'plyr_pos' in df.columns:
        positions = df['plyr_pos'].unique()
        stats['positions'] = positions.tolist()
        if len(positions) > 1 or 'WR' not in positions:
            logger.warning(f"Unexpected positions found: {positions}")

    # Target variable statistics
    if 'plyr_gm_rec_yds' in df.columns:
        stats['target_mean'] = float(df['plyr_gm_rec_yds'].mean())
        stats['target_std'] = float(df['plyr_gm_rec_yds'].std())
        stats['target_min'] = float(df['plyr_gm_rec_yds'].min())
        stats['target_max'] = float(df['plyr_gm_rec_yds'].max())
        stats['target_null_rate'] = float(df['plyr_gm_rec_yds'].isna().mean())

        logger.info(f"  Target (plyr_gm_rec_yds) statistics:")
        logger.info(f"    Mean: {stats['target_mean']:.2f} yards")
        logger.info(f"    Std: {stats['target_std']:.2f} yards")
        logger.info(f"    Range: [{stats['target_min']:.0f}, {stats['target_max']:.0f}]")
        logger.info(f"    Null rate: {stats['target_null_rate']*100:.2f}%")

    # Base feature availability
    base_features = ['plyr_gm_rec_tgt', 'plyr_gm_rec', 'is_home']
    for feat in base_features:
        if feat in df.columns:
            null_rate = df[feat].isna().mean()
            stats[f'{feat}_null_rate'] = float(null_rate)
            logger.info(f"  {feat} null rate: {null_rate*100:.2f}%")

    # Check for duplicate records
    duplicate_count = df.duplicated(subset=['plyr_id', 'game_id']).sum()
    stats['duplicate_records'] = int(duplicate_count)
    if duplicate_count > 0:
        logger.warning(f"  Found {duplicate_count} duplicate player-game records!")

    logger.info(f"\nBaseline Dataset Summary:")
    logger.info(f"  Total records: {stats['total_records']:,}")
    logger.info(f"  Unique players: {stats['unique_players']:,}")
    logger.info(f"  Unique games: {stats['unique_games']:,}")
    logger.info(f"  Seasons: {stats['seasons']}")
    logger.info(f"  Week range: {stats['weeks_range']}")

    return stats


def save_baseline_data(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save baseline dataset to parquet.

    Args:
        df: Baseline dataframe
        output_path: Path to save parquet file
    """
    logger.info(f"Saving baseline dataset to: {output_path}")

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
    logger.info("LOAD BASELINE DATA - WR RECEIVING YARDS PREDICTION")
    logger.info("="*80)

    try:
        # Initialize path manager and data loader
        logger.info("\n1. Initializing data loader...")
        paths = PathManager()
        loader = DataLoader(paths)

        # Define temporal scope
        seasons = [2022, 2023, 2024, 2025]
        logger.info(f"   Target seasons: {seasons}")

        # Load all required tables
        logger.info("\n2. Loading data tables...")

        # Core tables
        game_stats = load_receiving_game_stats(loader, seasons)
        player_info = load_player_reference(loader, seasons)
        game_info = load_game_info(loader, seasons)
        team_defense = load_team_defense_passing(loader, seasons)
        season_stats = load_player_season_receiving(loader, seasons)

        # Join all tables
        logger.info("\n3. Joining data tables...")

        df = join_player_info(game_stats, player_info)
        df = join_game_info(df, game_info)

        # Filter out week 1 games before joining opponent defense
        logger.info("\n3a. Filtering week 1 games...")
        df = filter_week_one(df)

        df = join_opponent_defense(df, team_defense)

        # Validate week filtering and opponent defense completeness
        logger.info("\n3b. Validating week filtering...")
        validate_week_filtering(df)

        df = join_season_cumulative_stats(df, season_stats)

        # Validate dataset
        logger.info("\n4. Validating baseline dataset...")
        stats = validate_baseline_data(df)

        # Save to parquet
        logger.info("\n5. Saving baseline dataset...")
        output_path = paths.get('data', 'processed') / 'player_features' / 'wr_baseline_raw.parquet'
        save_baseline_data(df, output_path)

        logger.info("\n" + "="*80)
        logger.info("BASELINE DATA LOADING COMPLETE")
        logger.info("="*80)
        logger.info(f"\nOutput file: {output_path}")
        logger.info(f"Total records: {len(df):,}")
        logger.info(f"Total features: {len(df.columns)}")

        return 0

    except Exception as e:
        logger.error(f"\nError in baseline data loading: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
