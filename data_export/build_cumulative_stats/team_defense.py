"""
Build weekly cumulative team defensive statistics from player game-level data.

This script processes player game-level defensive data, aggregates individual
player stats to team level, and generates weekly cumulative season statistics.

Available seasons and weeks:
- 2022: weeks 1-18
- 2023: weeks 1-18
- 2024: weeks 1-18
- 2025: weeks 1-11 (in progress)

Key difference from receiving.py:
- Source: plyr_gm/plyr_gm_def (player defensive game stats)
- Output: tm_szn/tm_def_plyr_agg (team defensive stats aggregated from players)
- Groups by team_id instead of plyr_id
- Builds a new table (not filling gaps in existing data)

Usage:
    python team_defense.py                    # Process all historical data
    python team_defense.py --test             # Test mode: Season 2024, Week 1 only
    python team_defense.py --week-id 47       # Process only rows with week_id=47 (weekly update)
    python team_defense.py --week-id 47,48    # Process multiple week_ids
"""

import os
import sys
import logging
import argparse
from typing import Dict, Callable, List, Optional, Tuple
import pandas as pd
import numpy as np

# Import common utilities
from common_utils import (
    load_week_mapping,
    load_game_level_data,
    save_cumulative_data,
    safe_divide,
    logger
)

# ============================================================================
# Configuration
# ============================================================================

# Season -> max weeks available
SEASON_WEEKS = {
    2022: 18,
    2023: 18,
    2024: 18,
    2025: 11,  # In progress
}

SOURCE_ROOT = "C:/Users/nocap/Desktop/code/NFL_ml/parquet_files/raw"
OUTPUT_ROOT = SOURCE_ROOT

GAME_TABLE = "plyr_gm/plyr_gm_def"
SEASON_TABLE = "tm_szn/tm_def_plyr_agg"  # Team defense aggregated from player data

# ============================================================================
# Column Definitions
# ============================================================================

# Columns to sum cumulatively (using game-level names)
# These are volume/counting stats that should accumulate over the season
GAME_SUM_COLUMNS = [
    'plyr_gm_def_int',           # Interceptions
    'plyr_gm_def_int_yds',       # Interception return yards
    'plyr_gm_def_int_td',        # Interception return TDs
    'plyr_gm_pass_def',          # Passes defended
    'plyr_gm_def_comb_tkl',      # Combined tackles
    'plyr_gm_def_solo_tkl',      # Solo tackles
    'plyr_gm_def_tkl_assist',    # Tackle assists
    'plyr_gm_def_tfl',           # Tackles for loss
    'plyr_gm_def_qb_hit',        # QB hits
    'plyr_gm_def_fmble_rec',     # Fumbles recovered
    'plyr_gm_def_fmbl_ret_yds',  # Fumble return yards
    'plyr_gm_def_fmbl_td',       # Fumble return TDs
    'plyr_gm_def_force_fmbl',    # Forced fumbles
    'plyr_gm_def_tgt',           # Times targeted in coverage
    'plyr_gm_def_cmp',           # Completions allowed in coverage
    'plyr_gm_def_pass_yds',      # Pass yards allowed in coverage
    'plyr_gm_def_pass_td',       # Pass TDs allowed in coverage
    'plyr_gm_def_ay',            # Air yards allowed (total, for ADOT calculation)
    'plyr_gm_def_yac',           # Yards after catch allowed
    'plyr_gm_def_bltz',          # Blitzes
    'plyr_gm_def_hrry',          # Hurries
    'plyr_gm_def_qbkd',          # QB knockdowns
    'plyr_gm_def_sk',            # Sacks
    'plyr_gm_def_prss',          # Pressures
    'plyr_gm_def_mtkl',          # Missed tackles
]

# Columns to take maximum (using game-level names)
GAME_MAX_COLUMNS = [
    'plyr_gm_def_int_lng',       # Longest interception return
]

# Identity columns for team grouping
IDENTITY_COLUMNS = [
    'team_id',
    'season_id',
    'week_id',
]

# Columns that exist in game data but should NOT be summed
# These are rate/average stats calculated per-game that need recalculation
GAME_CALCULATED_DO_NOT_SUM = [
    'plyr_gm_def_cmp_pct',       # Completion percentage allowed
    'plyr_gm_def_pass_yds_cmp',  # Yards per completion allowed
    'plyr_gm_def_pass_yds_tgt',  # Yards per target allowed
    'plyr_gm_def_pass_rtg',      # Passer rating allowed
    'plyr_gm_def_adot',          # Average depth of target allowed
    'plyr_gm_def_mtkl_pct',      # Missed tackle percentage
]


def rename_game_to_season_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns from game-level player naming to team season-level naming.

    Transforms 'plyr_gm_def_' prefix to 'tm_def_' prefix.

    Args:
        df: DataFrame with game-level column names

    Returns:
        DataFrame with renamed columns
    """
    rename_dict = {}

    for col in df.columns:
        # Handle columns with the prefix 'plyr_gm_def_'
        if 'plyr_gm_def_' in col:
            new_col = col.replace('plyr_gm_def_', 'tm_def_')
            rename_dict[col] = new_col
        # Handle 'plyr_gm_pass_def' -> 'tm_def_pass_def'
        elif col == 'plyr_gm_pass_def':
            rename_dict[col] = 'tm_def_pass_def'

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
    Calculate NFL passer rating allowed by the defense.

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
    int_caught = interceptions[has_attempts]

    # Four components (each capped 0-2.375)
    a = ((cmp / att) - 0.3) * 5
    a = a.clip(lower=0, upper=2.375)

    b = ((yds / att) - 3) * 0.25
    b = b.clip(lower=0, upper=2.375)

    c = (td / att) * 20
    c = c.clip(lower=0, upper=2.375)

    d = 2.375 - ((int_caught / att) * 25)
    d = d.clip(lower=0, upper=2.375)

    rating[has_attempts] = ((a + b + c + d) / 6) * 100

    return rating


def get_calculated_columns() -> Dict[str, Callable]:
    """
    Define calculated columns for team season cumulative defensive stats.

    Each function receives the cumulative DataFrame and returns a Series.
    These are rate/average statistics that must be recalculated from
    cumulative totals, not summed from game-level averages.

    Returns:
        Dictionary mapping column name to calculation function
    """
    calculated_cols = {
        # Completion percentage allowed: completions / targets
        'tm_def_cmp_pct': lambda df: safe_divide(
            df['tm_def_cmp'],
            df['tm_def_tgt']
        ),

        # Yards per completion allowed: yards / completions
        'tm_def_pass_yds_cmp': lambda df: safe_divide(
            df['tm_def_pass_yds'],
            df['tm_def_cmp']
        ),

        # Yards per target allowed: yards / targets
        'tm_def_pass_yds_tgt': lambda df: safe_divide(
            df['tm_def_pass_yds'],
            df['tm_def_tgt']
        ),

        # ADOT (Average Depth of Target) allowed: air yards / targets
        'tm_def_adot': lambda df: safe_divide(
            df['tm_def_ay'],
            df['tm_def_tgt']
        ),

        # YAC per completion allowed
        'tm_def_yac_cmp': lambda df: safe_divide(
            df['tm_def_yac'],
            df['tm_def_cmp']
        ),

        # Missed tackle percentage: missed / (combined + missed)
        'tm_def_mtkl_pct': lambda df: safe_divide(
            df['tm_def_mtkl'],
            df['tm_def_comb_tkl'] + df['tm_def_mtkl']
        ),

        # Passer rating allowed (when targeted in coverage)
        'tm_def_pass_rtg': lambda df: calculate_passer_rating_series(
            attempts=df['tm_def_tgt'],
            completions=df['tm_def_cmp'],
            yards=df['tm_def_pass_yds'],
            touchdowns=df['tm_def_pass_td'],
            interceptions=df['tm_def_int']
        ),

        # Sack rate (sacks per pressure)
        'tm_def_sk_pct': lambda df: safe_divide(
            df['tm_def_sk'],
            df['tm_def_prss']
        ),

        # Interception rate (INTs per target)
        'tm_def_int_pct': lambda df: safe_divide(
            df['tm_def_int'],
            df['tm_def_tgt']
        ),

        # Tackles per game
        'tm_def_tkl_pg': lambda df: safe_divide(
            df['tm_def_comb_tkl'],
            df['game_count']
        ),

        # Sacks per game
        'tm_def_sk_pg': lambda df: safe_divide(
            df['tm_def_sk'],
            df['game_count']
        ),

        # Pressures per game
        'tm_def_prss_pg': lambda df: safe_divide(
            df['tm_def_prss'],
            df['game_count']
        ),

        # Turnovers (INTs + fumble recoveries)
        'tm_def_to': lambda df: df['tm_def_int'] + df['tm_def_fmble_rec'],

        # Turnovers per game
        'tm_def_to_pg': lambda df: safe_divide(
            df['tm_def_int'] + df['tm_def_fmble_rec'],
            df['game_count']
        ),
    }

    return calculated_cols


def add_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns that don't exist in game-level data but may be needed in output.

    Args:
        df: Cumulative DataFrame

    Returns:
        DataFrame with additional columns added
    """
    # tm_def_gm: Number of games played (derived from game_count)
    if 'game_count' in df.columns:
        df['tm_def_gm'] = df['game_count']

    return df


def log_column_summary():
    """
    Log a summary of which columns are being processed and how.
    """
    logger.info("=" * 80)
    logger.info("COLUMN PROCESSING SUMMARY - TEAM DEFENSE")
    logger.info("=" * 80)

    logger.info("\nColumns being SUMMED (volume stats - aggregated across all players):")
    for col in GAME_SUM_COLUMNS:
        season_col = col.replace('plyr_gm_def_', 'tm_def_').replace('plyr_gm_pass_def', 'tm_def_pass_def')
        logger.info(f"  * {col} -> {season_col}")

    logger.info(f"\nColumns being MAXED:")
    for col in GAME_MAX_COLUMNS:
        season_col = col.replace('plyr_gm_def_', 'tm_def_')
        logger.info(f"  * {col} -> {season_col}")

    logger.info(f"\nColumns being CALCULATED (rate/average stats):")
    for col_name in get_calculated_columns().keys():
        logger.info(f"  * {col_name}")

    logger.info(f"\nColumns from game data NOT summed (recalculated cumulatively):")
    for col in GAME_CALCULATED_DO_NOT_SUM:
        logger.info(f"  * {col} - pre-calculated in game data, recalculated cumulatively")

    logger.info("=" * 80)


def calculate_team_cumulative_stats(
    game_df: pd.DataFrame,
    group_cols: List[str],
    sum_cols: List[str],
    max_cols: List[str],
    calculated_cols: Dict[str, Callable],
    target_week: int,
    rename_func: Callable = None
) -> pd.DataFrame:
    """
    Calculate cumulative team statistics from player game-level data.

    This differs from the player version in that it:
    1. First aggregates player stats to team-game level
    2. Then calculates cumulative stats through target week

    Args:
        game_df: Player game-level DataFrame with week_num column
        group_cols: Columns to group by for final output (e.g., ['team_id', 'season_id'])
        sum_cols: Columns to sum cumulatively
        max_cols: Columns to take maximum value
        calculated_cols: Dict mapping column name to calculation function
        target_week: Calculate cumulative stats through this week number
        rename_func: Optional function to rename columns before calculating derived columns

    Returns:
        DataFrame with one row per team containing cumulative stats through target_week
    """
    # Filter to only include games through target_week
    filtered_df = game_df[game_df['week_num'] <= target_week].copy()

    if len(filtered_df) == 0:
        logger.warning(f"No data found for week <= {target_week}")
        return pd.DataFrame()

    # Step 1: Aggregate player stats to team-game level
    # This gives us team stats per game (across all team's defensive players)
    team_game_group_cols = ['team_id', 'season_id', 'game_id', 'week_num']

    # Build aggregation for team-game level
    team_game_agg = {}
    for col in sum_cols:
        if col in filtered_df.columns:
            team_game_agg[col] = 'sum'

    for col in max_cols:
        if col in filtered_df.columns:
            team_game_agg[col] = 'max'

    # Aggregate to team-game level
    team_game_df = filtered_df.groupby(team_game_group_cols).agg(team_game_agg).reset_index()
    logger.debug(f"Aggregated to {len(team_game_df)} team-game records")

    # Step 2: Calculate cumulative stats across games for each team
    # Group by team and season (final grouping)
    grouped = team_game_df.groupby(group_cols)

    # Build aggregation dictionary for cumulative stats
    agg_dict = {}

    # Sum columns (sum across all games for the team)
    for col in sum_cols:
        if col in team_game_df.columns:
            agg_dict[col] = 'sum'

    # Max columns
    for col in max_cols:
        if col in team_game_df.columns:
            agg_dict[col] = 'max'

    # Perform aggregation
    cumulative_df = grouped.agg(agg_dict).reset_index()

    # Add game count (number of team games)
    game_count = grouped.size().to_frame('game_count').reset_index()
    cumulative_df = cumulative_df.merge(game_count, on=group_cols, how='left')

    # Rename columns if function provided (before calculating derived columns)
    if rename_func:
        cumulative_df = rename_func(cumulative_df)

    # Calculate derived columns
    for col_name, calc_func in calculated_cols.items():
        try:
            cumulative_df[col_name] = calc_func(cumulative_df)
        except Exception as e:
            logger.error(f"Error calculating {col_name}: {e}")
            cumulative_df[col_name] = np.nan

    logger.info(f"Calculated cumulative stats for {len(cumulative_df)} teams through week {target_week}")

    return cumulative_df


def validate_team_output(
    source_df: pd.DataFrame,
    output_df: pd.DataFrame,
    season: int,
    week: int,
    sum_cols: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validate cumulative team output data for consistency.

    Args:
        source_df: Source player game-level DataFrame
        output_df: Output team cumulative DataFrame
        season: Season year
        week: Week number
        sum_cols: List of summed columns to validate

    Returns:
        Tuple of (is_valid, list of warning messages)
    """
    warnings = []
    is_valid = True

    # Check for empty output
    if output_df.empty:
        warnings.append(f"Season {season}, Week {week}: Output DataFrame is empty")
        return False, warnings

    # Check unique teams
    unique_teams_source = source_df['team_id'].nunique()
    unique_teams_output = output_df['team_id'].nunique()

    if unique_teams_output != unique_teams_source:
        warnings.append(
            f"Season {season}, Week {week}: Team count mismatch - "
            f"Source: {unique_teams_source}, Output: {unique_teams_output}"
        )

    # Check for nulls in critical columns
    critical_cols = ['team_id', 'season_id', 'week_id']
    for col in critical_cols:
        if col in output_df.columns:
            null_count = output_df[col].isna().sum()
            if null_count > 0:
                warnings.append(f"Season {season}, Week {week}: Found {null_count} nulls in {col}")
                is_valid = False

    # Check for negative values in sum columns (where inappropriate)
    for col in sum_cols:
        if col in output_df.columns:
            negative_count = (output_df[col] < 0).sum()
            if negative_count > 0:
                warnings.append(
                    f"Season {season}, Week {week}: Found {negative_count} negative values in {col}"
                )

    # Log summary statistics
    logger.info(f"Season {season}, Week {week} validation:")
    logger.info(f"  - Output rows: {len(output_df)}")
    logger.info(f"  - Unique teams: {unique_teams_output}")
    logger.info(f"  - Warnings: {len(warnings)}")

    return is_valid, warnings


def load_team_game_level_data(
    source_root: str,
    table_path: str,
    season: int,
    weeks: List[int],
    week_mapping: pd.DataFrame
) -> pd.DataFrame:
    """
    Load game-level parquet data for specified season and weeks.

    Modified from common_utils version to handle team_id sorting instead of plyr_id.

    Args:
        source_root: Root directory of parquet files
        table_path: Relative path to table
        season: NFL season year
        weeks: List of week numbers to load
        week_mapping: DataFrame with week_id to week_num mapping

    Returns:
        DataFrame with game-level data sorted by week_num
    """
    # Use the common loader
    game_df = load_game_level_data(
        source_root=source_root,
        table_path=table_path,
        season=season,
        weeks=weeks,
        week_mapping=week_mapping
    )

    # Re-sort by team_id and week_num for team-based processing
    if not game_df.empty:
        game_df = game_df.sort_values(['team_id', 'week_num']).reset_index(drop=True)

    return game_df


def process_season_week(
    season: int,
    week: int,
    week_mapping: pd.DataFrame
) -> bool:
    """
    Process cumulative team defensive stats for a specific season through a specific week.

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

        game_df = load_team_game_level_data(
            source_root=SOURCE_ROOT,
            table_path=GAME_TABLE,
            season=season,
            weeks=weeks_to_load,
            week_mapping=week_mapping
        )

        if game_df.empty:
            logger.warning(f"No game data found for Season {season}, Week {week}")
            return False

        # Calculate cumulative team stats through this week
        cumulative_df = calculate_team_cumulative_stats(
            game_df=game_df,
            group_cols=['team_id', 'season_id'],
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
            cumulative_df['week_id'] = week

        # Validate output
        # Get season-level column names for validation
        season_sum_cols = [
            col.replace('plyr_gm_def_', 'tm_def_').replace('plyr_gm_pass_def', 'tm_def_pass_def')
            for col in GAME_SUM_COLUMNS
        ]

        is_valid, warnings = validate_team_output(
            source_df=game_df,
            output_df=cumulative_df,
            season=season,
            week=week,
            sum_cols=season_sum_cols
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
            id_column='tm_def_plyr_agg_id'
        )

        logger.info(f"+ Season {season}, Week {week}: {len(cumulative_df)} teams processed")
        return True

    except Exception as e:
        logger.error(f"x Season {season}, Week {week}: Error - {e}", exc_info=True)
        return False


def load_all_week_mappings() -> pd.DataFrame:
    """
    Load week mappings for all seasons and combine into a single DataFrame.

    Returns:
        DataFrame with week_id, week_num, season_id, and season columns
    """
    all_mappings = []

    for season in SEASON_WEEKS.keys():
        try:
            mapping = load_week_mapping(SOURCE_ROOT, season)
            mapping['season'] = season
            all_mappings.append(mapping)
        except Exception as e:
            logger.warning(f"Could not load week mapping for season {season}: {e}")

    if not all_mappings:
        raise ValueError("Could not load any week mappings")

    return pd.concat(all_mappings, ignore_index=True)


def get_season_week_from_week_id(week_id: int, all_mappings: pd.DataFrame) -> Optional[Tuple[int, int]]:
    """
    Look up the season and week_num for a given week_id.

    Args:
        week_id: The week_id to look up
        all_mappings: DataFrame with all week mappings

    Returns:
        Tuple of (season, week_num) or None if not found
    """
    row = all_mappings[all_mappings['week_id'] == week_id]
    if row.empty:
        return None
    return (int(row['season'].iloc[0]), int(row['week_num'].iloc[0]))


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Build weekly cumulative team defensive statistics from player game-level data"
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: process only Season 2024, Week 1'
    )
    parser.add_argument(
        '--week-id',
        type=str,
        default=None,
        help='Process only specific week_id(s). Comma-separated for multiple (e.g., 47 or 47,48)'
    )
    args = parser.parse_args()

    # Build list of (season, week) tuples to process
    season_weeks_to_process = []

    if args.test:
        # Test mode: just 2024 week 1
        season_weeks_to_process = [(2024, 1)]
        logger.info("=" * 80)
        logger.info("TEST MODE: Processing Season 2024, Week 1 only")
        logger.info("=" * 80)

    elif args.week_id:
        # Weekly update mode: process specific week_id(s)
        logger.info("=" * 80)
        logger.info("WEEKLY UPDATE MODE: Processing specific week_id(s)")
        logger.info("=" * 80)

        # Parse week_ids
        week_ids = [int(w.strip()) for w in args.week_id.split(',')]
        logger.info(f"Target week_ids: {week_ids}")

        # Load all week mappings to look up season/week from week_id
        all_mappings = load_all_week_mappings()

        for week_id in week_ids:
            result = get_season_week_from_week_id(week_id, all_mappings)
            if result:
                season, week_num = result
                season_weeks_to_process.append((season, week_num))
                logger.info(f"  week_id {week_id} -> Season {season}, Week {week_num}")
            else:
                logger.error(f"  week_id {week_id} not found in week mappings!")

    else:
        # Full historical mode: process all available seasons and weeks
        logger.info("=" * 80)
        logger.info("FULL HISTORICAL MODE: Processing all available data")
        logger.info("=" * 80)

        for season, max_week in SEASON_WEEKS.items():
            for week in range(1, max_week + 1):
                season_weeks_to_process.append((season, week))

    if not season_weeks_to_process:
        logger.error("No season-weeks to process!")
        sys.exit(1)

    logger.info(f"Total season-weeks to process: {len(season_weeks_to_process)}")
    logger.info(f"Source: {SOURCE_ROOT}/{GAME_TABLE}")
    logger.info(f"Output: {OUTPUT_ROOT}/{SEASON_TABLE}")

    # Log column processing summary
    log_column_summary()

    total_processed = 0
    total_failed = 0

    # Group by season to load week mapping once per season
    seasons_in_queue = sorted(set(s for s, w in season_weeks_to_process))

    for season in seasons_in_queue:
        weeks_for_season = sorted([w for s, w in season_weeks_to_process if s == season])

        logger.info(f"\nProcessing Season {season}, Weeks {weeks_for_season}...")

        try:
            # Load week mapping for this season
            week_mapping = load_week_mapping(SOURCE_ROOT, season)

        except Exception as e:
            logger.error(f"Failed to load week mapping for season {season}: {e}")
            logger.error(f"Skipping season {season}")
            total_failed += len(weeks_for_season)
            continue

        # Process each week cumulatively
        for week in weeks_for_season:
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
    logger.info(f"Total expected: {len(season_weeks_to_process)}")
    logger.info("=" * 80)

    if total_failed > 0:
        logger.warning(f"! {total_failed} season-weeks failed to process")
        sys.exit(1)
    else:
        logger.info("+ All season-weeks processed successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
