"""
Build weekly cumulative receiving statistics from game-level data.

This script processes player game-level receiving data for seasons 2022-2024
and generates weekly cumulative season statistics (weeks 1-17).

Usage:
    python receiving.py
"""

import os
import sys
import logging
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
GAME_SUM_COLUMNS = [
    'plyr_gm_rec_tgt',
    'plyr_gm_rec',
    'plyr_gm_rec_yds',
    'plyr_gm_rec_td',
    'plyr_gm_rec_first_dwn',
    'plyr_gm_rec_aybc',
    'plyr_gm_rec_yac',
    'plyr_gm_rec_adot',
    'plyr_gm_rec_brkn_tkl',
    'plyr_gm_rec_drp',
    'plyr_gm_rec_int',
]

# Columns to take maximum (using game-level names)
GAME_MAX_COLUMNS = [
    'plyr_gm_rec_lng',
]

# Identity columns to preserve
IDENTITY_COLUMNS = [
    'plyr_id',
    'season_id',
    'week_id',
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
        if 'plyr_gm_rec_' in col:
            new_col = col.replace('plyr_gm_rec_', 'plyr_rec_')
            rename_dict[col] = new_col

    if rename_dict:
        df = df.rename(columns=rename_dict)
        logger.debug(f"Renamed {len(rename_dict)} columns")

    return df


def get_calculated_columns() -> Dict[str, Callable]:
    """
    Define calculated columns for season cumulative stats.

    Each function receives the cumulative DataFrame and returns a Series.

    Returns:
        Dictionary mapping column name to calculation function
    """
    calculated_cols = {
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
        'plyr_rec_brkn_tkl_rec': lambda df: safe_divide(
            df['plyr_rec_brkn_tkl'],
            df['plyr_rec']
        ),

        # Drop percentage: drops / targets
        'plyr_rec_drp_pct': lambda df: safe_divide(
            df['plyr_rec_drp'],
            df['plyr_rec_tgt']
        ),
    }

    return calculated_cols


def add_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns that don't exist in game-level data but are needed in output.

    Args:
        df: Cumulative DataFrame

    Returns:
        DataFrame with additional columns added
    """
    # Rename game_count to plyr_rec_gm (number of games)
    if 'game_count' in df.columns:
        df['plyr_rec_gm'] = df['game_count']

    # plyr_rec_fmbl - fumbles (not in receiving game data, default to 0)
    if 'plyr_rec_fmbl' not in df.columns:
        df['plyr_rec_fmbl'] = 0

    # plyr_rec_succ_rt - success rate (not in game data, calculate or default)
    # Success rate typically = (first_downs + TDs) / receptions
    # But this is speculative - may need to investigate actual calculation
    if 'plyr_rec_succ_rt' not in df.columns:
        if 'plyr_rec_first_dwn' in df.columns and 'plyr_rec_td' in df.columns and 'plyr_rec' in df.columns:
            df['plyr_rec_succ_rt'] = safe_divide(
                df['plyr_rec_first_dwn'] + df['plyr_rec_td'],
                df['plyr_rec'],
                default=0.0
            )
        else:
            df['plyr_rec_succ_rt'] = 0.0

    # plyr_rec_pass_rtg - passer rating when targeting this player
    # This is complex to calculate from scratch, may need raw pass stats
    # For now, use weighted average from game data if available
    if 'plyr_rec_pass_rtg' not in df.columns:
        df['plyr_rec_pass_rtg'] = np.nan

    return df


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
    logger.info("=" * 80)
    logger.info("Building Weekly Cumulative Receiving Statistics")
    logger.info("=" * 80)
    logger.info(f"Seasons: {SEASONS}")
    logger.info(f"Weeks: {list(WEEKS)}")
    logger.info(f"Source: {SOURCE_ROOT}")
    logger.info(f"Output: {OUTPUT_ROOT}/{SEASON_TABLE}")
    logger.info("=" * 80)

    total_processed = 0
    total_failed = 0

    for season in SEASONS:
        logger.info(f"\nProcessing Season {season}...")

        try:
            # Load week mapping for this season
            week_mapping = load_week_mapping(SOURCE_ROOT, season)

        except Exception as e:
            logger.error(f"Failed to load week mapping for season {season}: {e}")
            logger.error(f"Skipping season {season}")
            total_failed += len(WEEKS)
            continue

        # Process each week cumulatively
        for week in WEEKS:
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
    logger.info(f"Total expected: {len(SEASONS) * len(WEEKS)}")
    logger.info("=" * 80)

    if total_failed > 0:
        logger.warning(f"⚠ {total_failed} season-weeks failed to process")
        sys.exit(1)
    else:
        logger.info("✓ All season-weeks processed successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
