"""
Build weekly cumulative team defensive passing statistics.

Three-stage aggregation:
1. Calculate defensive sack yards from opponent's offensive sack yards
2. Aggregate player defensive stats to team game level
3. Calculate cumulative season stats through each week (1-17)

Data Sources:
- plyr_gm_def: Player defensive coverage/pressure stats
- plyr_gm_pass: QB sack yards (used to calculate opponent's defensive sack yards)
- nfl_game: Game info for matching opponents

Note: tm_def_pass_exp is set to NULL as expected points are not
calculable from basic defensive statistics.

Usage:
    python tm_def_pass.py
    python tm_def_pass.py --test  # Test mode: Season 2024, Week 1 only
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

# Table paths
PLAYER_GAME_DEF_TABLE = "plyr_gm/plyr_gm_def"
PLAYER_GAME_PASS_TABLE = "plyr_gm/plyr_gm_pass"
GAME_INFO_TABLE = "gm_info/nfl_game"
TEAM_SEASON_TABLE = "tm_szn/tm_def_pass"

# ============================================================================
# Column Definitions
# ============================================================================

# Columns to sum cumulatively at team game level
TEAM_GAME_SUM_COLUMNS = [
    'tm_def_pass_cmp',
    'tm_def_pass_att',
    'tm_def_pass_yds',
    'tm_def_pass_td',
    'tm_def_int',
    'tm_def_pass_def',
    'tm_def_sk',
    'tm_def_sk_yds',      # Calculated from opponent QB sack yards
    'tm_def_qb_hit',
    'tm_def_tfl'
]

# NULL columns (data not available)
NULL_COLUMNS = {
    'tm_def_pass_exp': "Expected points not calculable from basic stats"
}

# ============================================================================
# Helper Functions
# ============================================================================

def calculate_passer_rating_series(
    attempts: pd.Series,
    completions: pd.Series,
    yards: pd.Series,
    touchdowns: pd.Series,
    interceptions: pd.Series
) -> pd.Series:
    """
    Calculate NFL passer rating.

    This is used to calculate defensive passer rating allowed.
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


def get_opponent_sack_yards(
    season: int,
    weeks: list,
    week_mapping: pd.DataFrame,
    source_root: str
) -> pd.DataFrame:
    """
    Calculate defensive sack yards from opponent's offensive sack yards.

    A defense's sack yards = opponent's QB sack yards taken in the same game.

    Returns DataFrame with columns: [game_id, team_id, tm_def_sk_yds]
    """
    # Load QB passing data
    pass_df = load_game_level_data(
        source_root=source_root,
        table_path=PLAYER_GAME_PASS_TABLE,
        season=season,
        weeks=weeks,
        week_mapping=week_mapping
    )

    if pass_df.empty:
        logger.warning(f"No passing data found for sack yards calculation")
        return pd.DataFrame(columns=['game_id', 'team_id', 'tm_def_sk_yds'])

    # Aggregate QB sack yards by team per game (offensive sack yards taken)
    off_sack_yds = pass_df.groupby(['game_id', 'team_id'])['plyr_gm_pass_sk_yds'].sum().reset_index()
    off_sack_yds = off_sack_yds.rename(columns={'plyr_gm_pass_sk_yds': 'off_sk_yds_taken'})

    # Load game info to get opponents - load manually since it's not player-level data
    game_dfs = []
    for week in weeks:
        week_path = f"{source_root}/{GAME_INFO_TABLE}/season={season}/week={week}"
        if os.path.exists(week_path):
            # Read all parquet files in the directory
            df = pd.read_parquet(week_path)
            game_dfs.append(df)
        else:
            logger.debug(f"No game data found at {week_path}")

    if not game_dfs:
        logger.error("No game data found for opponent matching")
        return pd.DataFrame(columns=['game_id', 'team_id', 'tm_def_sk_yds'])

    game_df = pd.concat(game_dfs, ignore_index=True)

    # Create mapping of game_id to home/away teams
    game_teams = game_df[['game_id', 'home_team_id', 'away_team_id']].drop_duplicates()

    # Merge offensive sack yards with game info to get both teams' data
    # Add home team's offensive sack yards
    home_off = off_sack_yds.merge(
        game_teams,
        left_on=['game_id', 'team_id'],
        right_on=['game_id', 'home_team_id'],
        how='inner'
    )
    home_off['opponent_id'] = home_off['away_team_id']

    # Add away team's offensive sack yards
    away_off = off_sack_yds.merge(
        game_teams,
        left_on=['game_id', 'team_id'],
        right_on=['game_id', 'away_team_id'],
        how='inner'
    )
    away_off['opponent_id'] = away_off['home_team_id']

    # Combine both
    all_off = pd.concat([home_off, away_off], ignore_index=True)

    # Now flip: opponent gets credit for defensive sack yards
    # The defense is the opponent, and they get the offensive sack yards as their defensive stat
    def_sack_yds = all_off[['game_id', 'opponent_id', 'off_sk_yds_taken']].copy()
    def_sack_yds = def_sack_yds.rename(columns={
        'opponent_id': 'team_id',
        'off_sk_yds_taken': 'tm_def_sk_yds'
    })

    # Remove duplicates (if any)
    def_sack_yds = def_sack_yds.drop_duplicates()

    # Fill NaN with 0 (opponent didn't take sacks)
    def_sack_yds['tm_def_sk_yds'] = def_sack_yds['tm_def_sk_yds'].fillna(0)

    logger.debug(f"Calculated defensive sack yards for {len(def_sack_yds)} team-games")

    return def_sack_yds


def aggregate_player_to_team(
    def_df: pd.DataFrame,
    sack_yds_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate player game stats to team game stats.

    Args:
        def_df: Player defensive stats
        sack_yds_df: Defensive sack yards (from opponent's offense)
    """
    # Map player columns to team columns
    rename_map = {
        'plyr_gm_def_tgt': 'tm_def_pass_att',
        'plyr_gm_def_cmp': 'tm_def_pass_cmp',
        'plyr_gm_def_pass_yds': 'tm_def_pass_yds',
        'plyr_gm_def_pass_td': 'tm_def_pass_td',
        'plyr_gm_def_int': 'tm_def_int',
        'plyr_gm_pass_def': 'tm_def_pass_def',
        'plyr_gm_def_sk': 'tm_def_sk',
        'plyr_gm_def_qb_hit': 'tm_def_qb_hit',
        'plyr_gm_def_tfl': 'tm_def_tfl'
    }

    # First check which columns exist in def_df
    existing_cols = [col for col in rename_map.keys() if col in def_df.columns]

    if not existing_cols:
        logger.error("None of the expected defensive columns found in player data")
        return pd.DataFrame()

    # Group and sum only existing columns
    group_cols = ['team_id', 'season_id', 'week_id', 'game_id']
    team_df = def_df.groupby(group_cols)[existing_cols].sum().reset_index()

    # Rename columns
    team_df = team_df.rename(columns=rename_map)

    # Merge defensive sack yards from opponent's offense
    team_df = team_df.merge(
        sack_yds_df,
        on=['game_id', 'team_id'],
        how='left'
    )

    # Fill NaN with 0 (no sack yards if data missing)
    team_df['tm_def_sk_yds'] = team_df['tm_def_sk_yds'].fillna(0)

    logger.debug(f"Aggregated to {len(team_df)} team-games with sack yards")

    return team_df


def get_calculated_columns() -> Dict[str, Callable]:
    """
    Define calculated columns for team season cumulative defensive stats.

    Each function receives the cumulative DataFrame and returns a Series.
    """
    calculated_cols = {
        # Completion percentage allowed
        'tm_def_pass_cmp_pct': lambda df: safe_divide(
            df['tm_def_pass_cmp'],
            df['tm_def_pass_att']
        ),

        # TD percentage allowed
        'tm_def_pass_td_pct': lambda df: safe_divide(
            df['tm_def_pass_td'],
            df['tm_def_pass_att']
        ),

        # Interception percentage (defensive interceptions per pass attempt)
        'tm_def_int_pct': lambda df: safe_divide(
            df['tm_def_int'],
            df['tm_def_pass_att']
        ),

        # Yards per attempt allowed
        'tm_def_pass_yds_att': lambda df: safe_divide(
            df['tm_def_pass_yds'],
            df['tm_def_pass_att']
        ),

        # Adjusted yards per attempt allowed
        'tm_def_pass_yds_att_adj': lambda df: safe_divide(
            df['tm_def_pass_yds'] + 20*df['tm_def_pass_td'] - 45*df['tm_def_int'],
            df['tm_def_pass_att']
        ),

        # Yards per completion allowed
        'tm_def_pass_ypc': lambda df: safe_divide(
            df['tm_def_pass_yds'],
            df['tm_def_pass_cmp']
        ),

        # Yards per game allowed
        'tm_def_pass_ypg': lambda df: safe_divide(
            df['tm_def_pass_yds'],
            df['game_count']
        ),

        # Defensive passer rating allowed
        'tm_def_pass_rtg': lambda df: calculate_passer_rating_series(
            df['tm_def_pass_att'],
            df['tm_def_pass_cmp'],
            df['tm_def_pass_yds'],
            df['tm_def_pass_td'],
            df['tm_def_int']
        ),

        # Sack percentage
        'tm_def_sk_pct': lambda df: safe_divide(
            df['tm_def_sk'],
            df['tm_def_pass_att'] + df['tm_def_sk']
        ),

        # Net yards per attempt
        'tm_def_pass_net_yds_att': lambda df: safe_divide(
            df['tm_def_pass_yds'] - df['tm_def_sk_yds'],
            df['tm_def_pass_att'] + df['tm_def_sk']
        ),

        # Adjusted net yards per attempt
        'tm_def_pass_net_yds_att_adj': lambda df: safe_divide(
            df['tm_def_pass_yds'] - df['tm_def_sk_yds'] + 20*df['tm_def_pass_td'] - 45*df['tm_def_int'],
            df['tm_def_pass_att'] + df['tm_def_sk']
        )
    }

    return calculated_cols


def add_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add NULL columns with documentation."""
    for col, reason in NULL_COLUMNS.items():
        df[col] = np.nan
        logger.debug(f"{col} set to NULL - {reason}")
    return df


def log_column_summary():
    """Log summary of column processing."""
    logger.info("=" * 80)
    logger.info("COLUMN PROCESSING SUMMARY")
    logger.info("=" * 80)

    logger.info("\nStage 1a - Player Defensive Stats (SUM to team level):")
    logger.info("  • plyr_gm_def_tgt → tm_def_pass_att")
    logger.info("  • plyr_gm_def_cmp → tm_def_pass_cmp")
    logger.info("  • plyr_gm_def_pass_yds → tm_def_pass_yds")
    logger.info("  • plyr_gm_def_pass_td → tm_def_pass_td")
    logger.info("  • plyr_gm_def_int → tm_def_int")
    logger.info("  • plyr_gm_pass_def → tm_def_pass_def")
    logger.info("  • plyr_gm_def_sk → tm_def_sk")
    logger.info("  • plyr_gm_def_qb_hit → tm_def_qb_hit")
    logger.info("  • plyr_gm_def_tfl → tm_def_tfl")

    logger.info("\nStage 1b - Defensive Sack Yards (from opponent's offense):")
    logger.info("  • opponent's QB plyr_gm_pass_sk_yds → tm_def_sk_yds")
    logger.info("  • Logic: Defense sack yards = opponent's offensive sack yards")

    logger.info("\nStage 2 - Cumulative Season Totals (SUM through weeks):")
    for col in TEAM_GAME_SUM_COLUMNS:
        logger.info(f"  • {col}")

    logger.info("\nStage 2 - Calculated Columns (from cumulative totals):")
    for col in get_calculated_columns().keys():
        logger.info(f"  • {col}")

    logger.info("\nNULL Columns:")
    for col, reason in NULL_COLUMNS.items():
        logger.info(f"  • {col} - {reason}")

    logger.info("=" * 80)


def process_season_week(season: int, week: int, week_mapping: pd.DataFrame) -> bool:
    """
    Two-stage aggregation process.

    Args:
        season: NFL season year
        week: Week number (cumulative through this week)
        week_mapping: DataFrame with week_id to week_num mapping

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load weeks 1 through target week
        weeks_to_load = list(range(1, week + 1))

        # Stage 1a: Load player defensive data
        logger.debug(f"Loading player defensive stats for weeks 1-{week}...")
        player_def_df = load_game_level_data(
            source_root=SOURCE_ROOT,
            table_path=PLAYER_GAME_DEF_TABLE,
            season=season,
            weeks=weeks_to_load,
            week_mapping=week_mapping
        )

        if player_def_df.empty:
            logger.warning(f"No defensive data found for Season {season}, Week {week}")
            return False

        # Stage 1b: Calculate defensive sack yards from opponent's offense
        logger.debug("Calculating defensive sack yards from opponent QB sack yards...")
        sack_yds_df = get_opponent_sack_yards(
            season=season,
            weeks=weeks_to_load,
            week_mapping=week_mapping,
            source_root=SOURCE_ROOT
        )

        # Stage 1c: Aggregate player stats to team game level
        logger.debug("Aggregating player defensive stats to team game level...")
        team_game_df = aggregate_player_to_team(player_def_df, sack_yds_df)

        if team_game_df.empty:
            logger.warning(f"No team game data generated for Season {season}, Week {week}")
            return False

        # Add week_num from week_mapping for cumulative calculation
        team_game_df = team_game_df.merge(
            week_mapping[['week_id', 'week_num']],
            on='week_id',
            how='left'
        )

        # Stage 2: Calculate cumulative season stats
        logger.debug("Calculating cumulative season defensive stats...")
        cumulative_df = calculate_cumulative_stats(
            game_df=team_game_df,
            group_cols=['team_id', 'season_id'],
            sum_cols=TEAM_GAME_SUM_COLUMNS,
            max_cols=[],  # No max columns for team defensive stats
            calculated_cols=get_calculated_columns(),
            target_week=week
        )

        if cumulative_df.empty:
            logger.warning(f"No cumulative data generated for Season {season}, Week {week}")
            return False

        # Add NULL columns (just tm_def_pass_exp)
        cumulative_df = add_missing_columns(cumulative_df)

        # Get week_id for this week
        week_info = week_mapping[week_mapping['week_num'] == week]
        if not week_info.empty:
            week_id = week_info['week_id'].iloc[0]
            cumulative_df['week_id'] = week_id
        else:
            logger.error(f"Could not find week_id for season {season}, week {week}")
            cumulative_df['week_id'] = week

        # Simple validation for team data
        unique_teams = cumulative_df['team_id'].nunique()
        logger.info(f"Season {season}, Week {week} validation:")
        logger.info(f"  - Output rows: {len(cumulative_df)}")
        logger.info(f"  - Unique teams: {unique_teams}")

        # Check for nulls in critical columns
        for col in ['team_id', 'season_id', 'week_id']:
            if col in cumulative_df.columns:
                null_count = cumulative_df[col].isna().sum()
                if null_count > 0:
                    logger.warning(f"Found {null_count} nulls in {col}")

        # Check for negative values in counting stats
        for col in TEAM_GAME_SUM_COLUMNS:
            if col in cumulative_df.columns:
                negative_count = (cumulative_df[col] < 0).sum()
                if negative_count > 0:
                    logger.warning(f"Found {negative_count} negative values in {col}")

        # Save to partitioned parquet
        save_cumulative_data(
            df=cumulative_df,
            output_root=OUTPUT_ROOT,
            table_path=TEAM_SEASON_TABLE,
            season=season,
            week=week,
            id_column='tm_def_pass_id'
        )

        logger.info(f"✓ Season {season}, Week {week}: {len(cumulative_df)} teams processed")
        return True

    except Exception as e:
        logger.error(f"✗ Season {season}, Week {week}: Error - {e}", exc_info=True)
        return False


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Build weekly cumulative team defensive passing statistics"
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
        logger.info("Building Weekly Cumulative Team Defensive Passing Statistics")
        logger.info("=" * 80)

    logger.info(f"Seasons: {seasons_to_process}")
    logger.info(f"Weeks: {list(weeks_to_process)}")
    logger.info(f"Source: {SOURCE_ROOT}")
    logger.info(f"Output: {OUTPUT_ROOT}/{TEAM_SEASON_TABLE}")

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
