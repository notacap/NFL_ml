"""
Build weekly cumulative receiving statistics from game-level data.

This script processes player game-level receiving data for seasons 2022-2024
and generates weekly cumulative season statistics (weeks 1-17).

Usage:
    python receiving.py
    python receiving.py --test  # Test mode: Season 2024, Week 1 only
"""

import os
import sys
import logging
import argparse
from typing import Dict, Callable
import pandas as pd
import numpy as np

# Import common utilities
from common_utils import (
    load_week_mapping,
    load_game_level_data,
    calculate_cumulative_stats,
    save_cumulative_data,
    validate_output,
    safe_divide,
    logger
)

# ============================================================================
# Configuration
# ============================================================================

SEASONS = [2022, 2023, 2024]
WEEKS = range(1, 18)  # Weeks 1-17
SOURCE_ROOT = "C:/Users/nocap/Desktop/code/NFL_ml/parquet_files/raw"
OUTPUT_ROOT = SOURCE_ROOT

GAME_TABLE = "plyr_gm/plyr_gm_rec"
SEASON_TABLE = "plyr_szn/plyr_rec"

# Column mapping from game-level to season-level (remove '_gm_')
GAME_TO_SEASON_PREFIX = {
    'plyr_gm_rec_': 'plyr_rec_'
}

# ============================================================================
# Column Definitions
# ============================================================================

# Columns to sum cumulatively (using game-level names)
# These are volume/counting stats that should accumulate over the season
GAME_SUM_COLUMNS = [
    'plyr_gm_rec_tgt',        # Targets
    'plyr_gm_rec',            # Receptions
    'plyr_gm_rec_yds',        # Receiving yards
    'plyr_gm_rec_td',         # Receiving touchdowns
    'plyr_gm_rec_first_dwn',  # First downs
    'plyr_gm_rec_aybc',       # Air yards before catch (total, used to calculate ADOT)
    'plyr_gm_rec_yac',        # Yards after catch (total)
    'plyr_gm_rec_brkn_tkl',   # Broken tackles (total count)
    'plyr_gm_rec_drp',        # Drops (total count)
    'plyr_gm_rec_int',        # Interceptions (targets intercepted)
    'plyr_gm_rec_fmbl',       # Fumbles (joined from plyr_gm_fmbl table)
]

# Columns to take maximum (using game-level names)
GAME_MAX_COLUMNS = [
    'plyr_gm_rec_lng',        # Longest reception
]

# Identity columns to preserve
IDENTITY_COLUMNS = [
    'plyr_id',
    'season_id',
    'week_id',
]

# Columns that exist in game data but should NOT be summed
# These are rate/average stats calculated per-game that need recalculation
GAME_CALCULATED_DO_NOT_SUM = [
    'plyr_gm_rec_adot',           # Average depth of target (air yards / targets)
    'plyr_gm_rec_aybc_route',     # Air yards before catch per route
    'plyr_gm_rec_yac_route',      # Yards after catch per route
    'plyr_gm_rec_brkn_tkl_rec',   # Broken tackles per reception
    'plyr_gm_rec_drp_pct',        # Drop percentage (drops / targets)
    'plyr_gm_rec_pass_rtg',       # Passer rating when targeting this player
]


def rename_game_to_season_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns from game-level naming to season-level naming.

    Removes '_gm_' from column names.

    Args:
        df: DataFrame with game-level column names

    Returns:
        DataFrame with renamed columns
    """
    rename_dict = {}

    for col in df.columns:
        # Handle the base column 'plyr_gm_rec' (without underscore)
        if col == 'plyr_gm_rec':
            rename_dict[col] = 'plyr_rec'
        # Handle all other columns with the prefix 'plyr_gm_rec_'
        elif 'plyr_gm_rec_' in col:
            new_col = col.replace('plyr_gm_rec_', 'plyr_rec_')
            rename_dict[col] = new_col

    if rename_dict:
        df = df.rename(columns=rename_dict)
        logger.debug(f"Renamed {len(rename_dict)} columns")

    return df


def calculate_passer_rating_series(
    attempts: pd.Series,
    completions: pd.Series,
    yards: pd.Series,
    touchdowns: pd.Series,
    interceptions: pd.Series
) -> pd.Series:
    """
    Calculate NFL passer rating for targeting a specific receiver.

    Returns NaN when attempts = 0.
    """
    rating = pd.Series(np.nan, index=attempts.index)
    has_attempts = attempts > 0

    if not has_attempts.any():
        return rating

    att = attempts[has_attempts]
    cmp = completions[has_attempts]
    yds = yards[has_attempts]
    td = touchdowns[has_attempts]
    int_thrown = interceptions[has_attempts]

    # Four components (each capped 0-2.375)
    a = ((cmp / att) - 0.3) * 5
    a = a.clip(lower=0, upper=2.375)

    b = ((yds / att) - 3) * 0.25
    b = b.clip(lower=0, upper=2.375)

    c = (td / att) * 20
    c = c.clip(lower=0, upper=2.375)

    d = 2.375 - ((int_thrown / att) * 25)
    d = d.clip(lower=0, upper=2.375)

    rating[has_attempts] = ((a + b + c + d) / 6) * 100

    return rating


def get_calculated_columns() -> Dict[str, Callable]:
    """
    Define calculated columns for season cumulative stats.

    Each function receives the cumulative DataFrame and returns a Series.
    These are rate/average statistics that must be recalculated from
    cumulative totals, not summed from game-level averages.

    Returns:
        Dictionary mapping column name to calculation function
    """
    calculated_cols = {
        # ADOT (Average Depth of Target): total air yards / total targets
        # FIXED: Previously summed game-level ADOTs, now calculated from totals
        'plyr_rec_adot': lambda df: safe_divide(
            df['plyr_rec_aybc'],
            df['plyr_rec_tgt']
        ),

        # Catch percentage: receptions / targets
        'plyr_rec_catch_pct': lambda df: safe_divide(
            df['plyr_rec'],
            df['plyr_rec_tgt']
        ),

        # Yards per reception: yards / receptions
        'plyr_rec_yds_rec': lambda df: safe_divide(
            df['plyr_rec_yds'],
            df['plyr_rec']
        ),

        # Yards per target: yards / targets
        'plyr_rec_yds_tgt': lambda df: safe_divide(
            df['plyr_rec_yds'],
            df['plyr_rec_tgt']
        ),

        # Yards per game: yards / games played
        'plyr_rec_ypg': lambda df: safe_divide(
            df['plyr_rec_yds'],
            df['game_count']
        ),

        # Air yards before catch per reception
        'plyr_rec_aybc_rec': lambda df: safe_divide(
            df['plyr_rec_aybc'],
            df['plyr_rec']
        ),

        # Yards after catch per reception
        'plyr_rec_yac_rec': lambda df: safe_divide(
            df['plyr_rec_yac'],
            df['plyr_rec']
        ),

        # Broken tackles per reception
        # Recalculated from cumulative totals (game data has this pre-calculated)
        'plyr_rec_brkn_tkl_rec': lambda df: safe_divide(
            df['plyr_rec_brkn_tkl'],
            df['plyr_rec']
        ),

        # Drop percentage: drops / targets
        # Recalculated from cumulative totals (game data has this pre-calculated)
        'plyr_rec_drp_pct': lambda df: safe_divide(
            df['plyr_rec_drp'],
            df['plyr_rec_tgt']
        ),

        # Passer rating when targeting this receiver
        # Calculated from cumulative components
        'plyr_rec_pass_rtg': lambda df: calculate_passer_rating_series(
            attempts=df['plyr_rec_tgt'],
            completions=df['plyr_rec'],
            yards=df['plyr_rec_yds'],
            touchdowns=df['plyr_rec_td'],
            interceptions=df['plyr_rec_int']
        ),
    }

    return calculated_cols


def add_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns that don't exist in game-level data but may be needed in output.

    This handles columns that either:
    1. Are derived from game_count (like games played)
    2. Require data from other tables (documented as NULL)
    3. Cannot be calculated from receiving data alone (documented as NULL)

    Args:
        df: Cumulative DataFrame

    Returns:
        DataFrame with additional columns added
    """
    # plyr_rec_gm: Number of games played (derived from game_count)
    if 'game_count' in df.columns:
        df['plyr_rec_gm'] = df['game_count']

    # plyr_rec_succ_rt: Success rate
    # NOT AVAILABLE: Success rate definition unclear and not in game data
    # Common definition: successful plays / total plays (where successful = gain > expected)
    # Cannot be calculated without down/distance context - setting to NULL
    if 'plyr_rec_succ_rt' not in df.columns:
        df['plyr_rec_succ_rt'] = np.nan
        logger.debug("plyr_rec_succ_rt set to NULL - requires down/distance data")

    # plyr_rec_aybc_route: Air yards before catch per route
    # AVAILABLE IN GAME DATA but NOT CALCULABLE cumulatively
    # Would need total routes run across all games, which is not summed
    # Setting to NULL
    if 'plyr_rec_aybc_route' not in df.columns:
        df['plyr_rec_aybc_route'] = np.nan
        logger.debug("plyr_rec_aybc_route set to NULL - routes run not tracked")

    # plyr_rec_yac_route: Yards after catch per route
    # AVAILABLE IN GAME DATA but NOT CALCULABLE cumulatively
    # Would need total routes run across all games, which is not summed
    # Setting to NULL
    if 'plyr_rec_yac_route' not in df.columns:
        df['plyr_rec_yac_route'] = np.nan
        logger.debug("plyr_rec_yac_route set to NULL - routes run not tracked")

    return df


def log_column_summary():
    """
    Log a summary of which columns are being processed and how.

    This helps verify that all columns are handled correctly.
    """
    logger.info("=" * 80)
    logger.info("COLUMN PROCESSING SUMMARY")
    logger.info("=" * 80)

    logger.info("\nColumns being SUMMED (volume stats):")
    for col in GAME_SUM_COLUMNS:
        season_col = col.replace('plyr_gm_rec_', 'plyr_rec_')
        logger.info(f"  • {col} → {season_col}")

    logger.info(f"\nColumns being MAXED:")
    for col in GAME_MAX_COLUMNS:
        season_col = col.replace('plyr_gm_rec_', 'plyr_rec_')
        logger.info(f"  • {col} → {season_col}")

    logger.info(f"\nColumns being CALCULATED (rate/average stats):")
    for col_name in get_calculated_columns().keys():
        logger.info(f"  • {col_name}")

    logger.info(f"\nColumns from game data NOT summed (recalculated or unavailable):")
    for col in GAME_CALCULATED_DO_NOT_SUM:
        logger.info(f"  • {col} - pre-calculated in game data, recalculated cumulatively")

    logger.info("\nColumns set to NULL (not available in receiving data):")
    logger.info("  • plyr_rec_succ_rt - requires down/distance data")
    logger.info("  • plyr_rec_aybc_route - requires routes run data")
    logger.info("  • plyr_rec_yac_route - requires routes run data")

    logger.info("=" * 80)


def process_season_week(
    season: int,
    week: int,
    week_mapping: pd.DataFrame
) -> bool:
    """
    Process cumulative stats for a specific season through a specific week.

    Args:
        season: NFL season year
        week: Week number (cumulative through this week)
        week_mapping: DataFrame with week_id to week_num mapping

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load all game data from week 1 through target week
        weeks_to_load = list(range(1, week + 1))

        game_df = load_game_level_data(
            source_root=SOURCE_ROOT,
            table_path=GAME_TABLE,
            season=season,
            weeks=weeks_to_load,
            week_mapping=week_mapping
        )

        if game_df.empty:
            logger.warning(f"No game data found for Season {season}, Week {week}")
            return False

        # Load fumble data and join with receiving data
        try:
            fumble_df = load_game_level_data(
                source_root=SOURCE_ROOT,
                table_path="plyr_gm/plyr_gm_fmbl",
                season=season,
                weeks=weeks_to_load,
                week_mapping=week_mapping
            )

            if not fumble_df.empty:
                # Select only the columns we need from fumble data
                fumble_cols = ['plyr_id', 'season_id', 'week_id', 'plyr_gm_rec_fmbl']
                fumble_df = fumble_df[fumble_cols]

                # Left join fumble data to receiving data (not all receivers fumble)
                game_df = game_df.merge(
                    fumble_df,
                    on=['plyr_id', 'season_id', 'week_id'],
                    how='left'
                )

                # Fill NaN fumbles with 0 (player didn't fumble that week)
                game_df['plyr_gm_rec_fmbl'] = game_df['plyr_gm_rec_fmbl'].fillna(0)
                logger.debug(f"Joined fumble data: {len(fumble_df)} fumble records")
            else:
                # No fumble data available, set to 0
                game_df['plyr_gm_rec_fmbl'] = 0
                logger.debug("No fumble data found, setting plyr_gm_rec_fmbl to 0")

        except Exception as e:
            logger.warning(f"Could not load fumble data: {e}. Setting plyr_gm_rec_fmbl to 0")
            game_df['plyr_gm_rec_fmbl'] = 0

        # Calculate cumulative stats through this week
        cumulative_df = calculate_cumulative_stats(
            game_df=game_df,
            group_cols=['plyr_id', 'season_id'],
            sum_cols=GAME_SUM_COLUMNS,
            max_cols=GAME_MAX_COLUMNS,
            calculated_cols=get_calculated_columns(),
            target_week=week,
            rename_func=rename_game_to_season_columns
        )

        if cumulative_df.empty:
            logger.warning(f"No cumulative data generated for Season {season}, Week {week}")
            return False

        # Add missing columns
        cumulative_df = add_missing_columns(cumulative_df)

        # Get week_id for this week from the mapping
        week_info = week_mapping[week_mapping['week_num'] == week]
        if not week_info.empty:
            week_id = week_info['week_id'].iloc[0]
            cumulative_df['week_id'] = week_id
        else:
            logger.error(f"Could not find week_id for season {season}, week {week}")
            # Use the week number as fallback
            cumulative_df['week_id'] = week

        # Validate output
        is_valid, warnings = validate_output(
            source_df=game_df,
            output_df=cumulative_df,
            season=season,
            week=week,
            sum_cols=[col.replace('plyr_gm_rec_', 'plyr_rec_') for col in GAME_SUM_COLUMNS]
        )

        if warnings:
            for warning in warnings:
                logger.warning(warning)

        # Save to partitioned parquet
        save_cumulative_data(
            df=cumulative_df,
            output_root=OUTPUT_ROOT,
            table_path=SEASON_TABLE,
            season=season,
            week=week,
            id_column='plyr_rec_id'
        )

        logger.info(f"✓ Season {season}, Week {week}: {len(cumulative_df)} players processed")
        return True

    except Exception as e:
        logger.error(f"✗ Season {season}, Week {week}: Error - {e}", exc_info=True)
        return False


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Build weekly cumulative receiving statistics from game-level data"
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: process only Season 2024, Week 1'
    )
    args = parser.parse_args()

    # Determine which seasons and weeks to process
    if args.test:
        seasons_to_process = [2024]
        weeks_to_process = range(1, 2)  # Just week 1
        logger.info("=" * 80)
        logger.info("TEST MODE: Processing Season 2024, Week 1 only")
        logger.info("=" * 80)
    else:
        seasons_to_process = SEASONS
        weeks_to_process = WEEKS
        logger.info("=" * 80)
        logger.info("Building Weekly Cumulative Receiving Statistics")
        logger.info("=" * 80)

    logger.info(f"Seasons: {seasons_to_process}")
    logger.info(f"Weeks: {list(weeks_to_process)}")
    logger.info(f"Source: {SOURCE_ROOT}")
    logger.info(f"Output: {OUTPUT_ROOT}/{SEASON_TABLE}")

    # Log column processing summary
    log_column_summary()

    total_processed = 0
    total_failed = 0

    for season in seasons_to_process:
        logger.info(f"\nProcessing Season {season}...")

        try:
            # Load week mapping for this season
            week_mapping = load_week_mapping(SOURCE_ROOT, season)

        except Exception as e:
            logger.error(f"Failed to load week mapping for season {season}: {e}")
            logger.error(f"Skipping season {season}")
            total_failed += len(weeks_to_process)
            continue

        # Process each week cumulatively
        for week in weeks_to_process:
            success = process_season_week(season, week, week_mapping)

            if success:
                total_processed += 1
            else:
                total_failed += 1

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Processing Complete!")
    logger.info(f"Successfully processed: {total_processed} season-weeks")
    logger.info(f"Failed: {total_failed} season-weeks")
    logger.info(f"Total expected: {len(seasons_to_process) * len(weeks_to_process)}")
    logger.info("=" * 80)

    if total_failed > 0:
        logger.warning(f"⚠ {total_failed} season-weeks failed to process")
        sys.exit(1)
    else:
        logger.info("✓ All season-weeks processed successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
