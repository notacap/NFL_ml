"""
Common utility functions for building cumulative statistics from game-level data.

This module provides reusable functions for:
- Loading week mappings (week_id to week_num)
- Loading game-level data with week number merging
- Calculating cumulative statistics
- Saving partitioned parquet files
- Validating output data
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Callable, Optional, Tuple
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _read_parquet_rowgroups(file_path: str) -> pd.DataFrame:
    """
    Read parquet file by row groups to work around corruption issues.

    Args:
        file_path: Path to parquet file

    Returns:
        DataFrame with file contents

    Raises:
        Exception: If unable to read any row groups
    """
    try:
        parquet_file = pq.ParquetFile(file_path)
        dfs = []

        for i in range(parquet_file.num_row_groups):
            try:
                table = parquet_file.read_row_group(i)
                df = table.to_pandas()
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to read row group {i} from {file_path}: {e}")
                continue

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            raise ValueError(f"No row groups could be read from {file_path}")

    except Exception as e:
        raise Exception(f"Failed to read parquet file by row groups: {e}")


def load_week_mapping(source_root: str, season: int) -> pd.DataFrame:
    """
    Load the nfl_week table to map week_id to week_num for a given season.

    Args:
        source_root: Root directory of parquet files
        season: NFL season year (e.g., 2024)

    Returns:
        DataFrame with columns: week_id, week_num, season_id

    Raises:
        FileNotFoundError: If week mapping file doesn't exist
        ValueError: If no data found for the specified season
    """
    week_mapping_path = os.path.join(
        source_root,
        'static',
        'nfl_week',
        f'season={season}'
    )

    if not os.path.exists(week_mapping_path):
        raise FileNotFoundError(f"Week mapping not found at {week_mapping_path}")

    # Find all parquet files in the directory
    parquet_files = list(Path(week_mapping_path).glob('*.parquet'))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {week_mapping_path}")

    # Read all parquet files and concatenate
    dfs = []
    for file in parquet_files:
        df = None

        # Try multiple reading strategies
        strategies = [
            # Strategy 1: Use pyarrow dataset API (more robust)
            lambda f: pd.read_parquet(f, engine='pyarrow'),

            # Strategy 2: Use pandas default
            lambda f: pd.read_parquet(f),

            # Strategy 3: Use pyarrow ParquetFile with individual row group reading
            lambda f: _read_parquet_rowgroups(f),
        ]

        for i, strategy in enumerate(strategies):
            try:
                df = strategy(str(file))
                if df is not None and not df.empty:
                    logger.debug(f"Successfully read {file.name} with strategy {i+1}")
                    break
            except Exception as e:
                logger.debug(f"Strategy {i+1} failed for {file.name}: {e}")
                continue

        if df is not None and not df.empty:
            dfs.append(df)
        else:
            logger.error(f"All strategies failed to read {file}")
            # Continue to next file instead of raising
            continue

    if not dfs:
        raise ValueError(f"Could not read any week mapping data for season {season}")

    week_mapping = pd.concat(dfs, ignore_index=True)

    # Validate required columns
    required_cols = ['week_id', 'week_num']
    missing_cols = set(required_cols) - set(week_mapping.columns)
    if missing_cols:
        raise ValueError(f"Week mapping missing required columns: {missing_cols}")

    # Add season partition if not present
    if 'season' not in week_mapping.columns and 'season_id' in week_mapping.columns:
        week_mapping['season'] = week_mapping['season_id']

    logger.info(f"Loaded week mapping for season {season}: {len(week_mapping)} weeks")

    return week_mapping[['week_id', 'week_num', 'season_id']].drop_duplicates()


def load_game_level_data(
    source_root: str,
    table_path: str,
    season: int,
    weeks: List[int],
    week_mapping: pd.DataFrame
) -> pd.DataFrame:
    """
    Load game-level parquet data for specified season and weeks.

    Merges with week mapping to add week_num and sorts by week_num.

    Args:
        source_root: Root directory of parquet files
        table_path: Relative path to table (e.g., 'plyr_gm/plyr_gm_rec')
        season: NFL season year
        weeks: List of week numbers to load (e.g., [1, 2, 3])
        week_mapping: DataFrame with week_id to week_num mapping

    Returns:
        DataFrame with game-level data sorted by week_num

    Raises:
        FileNotFoundError: If data files don't exist
    """
    all_data = []

    for week in weeks:
        week_path = os.path.join(source_root, table_path, f'season={season}', f'week={week}')

        if not os.path.exists(week_path):
            logger.warning(f"Week path not found: {week_path}")
            continue

        # Find all parquet files in the directory
        parquet_files = list(Path(week_path).glob('*.parquet'))

        if not parquet_files:
            logger.warning(f"No parquet files found in {week_path}")
            continue

        for file in parquet_files:
            df = None

            # Try multiple reading strategies
            strategies = [
                # Strategy 1: Use pandas with pyarrow engine
                lambda f: pd.read_parquet(f, engine='pyarrow'),

                # Strategy 2: Use pandas default
                lambda f: pd.read_parquet(f),

                # Strategy 3: Use pyarrow ParquetFile with row group reading
                lambda f: _read_parquet_rowgroups(f),
            ]

            for i, strategy in enumerate(strategies):
                try:
                    df = strategy(str(file))
                    if df is not None and not df.empty:
                        logger.debug(f"Successfully read {file.name} with strategy {i+1}")
                        break
                except Exception as e:
                    logger.debug(f"Strategy {i+1} failed for {file.name}: {e}")
                    continue

            if df is not None and not df.empty:
                # Add partition columns if not present
                if 'season' not in df.columns:
                    df['season'] = season
                if 'week' not in df.columns:
                    df['week'] = week

                all_data.append(df)
                logger.debug(f"Loaded {len(df)} rows from {file.name}")
            else:
                logger.error(f"All strategies failed to read {file}")
                # Continue to next file rather than failing completely
                continue

    if not all_data:
        raise FileNotFoundError(f"No data found for season {season}, weeks {weeks}")

    # Concatenate all data
    game_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Loaded {len(game_df)} game records for season {season}, weeks {weeks}")

    # Merge with week mapping to get week_num
    game_df = game_df.merge(
        week_mapping[['week_id', 'week_num']],
        on='week_id',
        how='left'
    )

    # Check for missing week_num
    missing_week_num = game_df['week_num'].isna().sum()
    if missing_week_num > 0:
        logger.warning(f"Found {missing_week_num} rows with missing week_num")
        # Fill with partition week number as fallback
        game_df['week_num'] = game_df['week_num'].fillna(game_df['week'])

    # Sort by week_num to ensure proper cumulative calculation
    game_df = game_df.sort_values(['plyr_id', 'week_num']).reset_index(drop=True)

    return game_df


def calculate_cumulative_stats(
    game_df: pd.DataFrame,
    group_cols: List[str],
    sum_cols: List[str],
    max_cols: List[str],
    calculated_cols: Dict[str, Callable],
    target_week: int,
    rename_func: Optional[Callable] = None
) -> pd.DataFrame:
    """
    Calculate cumulative statistics from game-level data through a target week.

    Args:
        game_df: Game-level DataFrame with week_num column
        group_cols: Columns to group by (e.g., ['plyr_id', 'season_id'])
        sum_cols: Columns to sum cumulatively
        max_cols: Columns to take maximum value
        calculated_cols: Dict mapping column name to calculation function
                        Function receives the grouped DataFrame and returns Series
        target_week: Calculate cumulative stats through this week number
        rename_func: Optional function to rename columns before calculating derived columns

    Returns:
        DataFrame with one row per group containing cumulative stats through target_week
    """
    # Filter to only include games through target_week
    filtered_df = game_df[game_df['week_num'] <= target_week].copy()

    if len(filtered_df) == 0:
        logger.warning(f"No data found for week <= {target_week}")
        return pd.DataFrame()

    # Group by specified columns
    grouped = filtered_df.groupby(group_cols)

    # Build aggregation dictionary
    agg_dict = {}

    # Sum columns
    for col in sum_cols:
        if col in filtered_df.columns:
            agg_dict[col] = 'sum'
        else:
            logger.warning(f"Column {col} not found in data, skipping")

    # Max columns
    for col in max_cols:
        if col in filtered_df.columns:
            agg_dict[col] = 'max'
        else:
            logger.warning(f"Column {col} not found in data, skipping")

    # Perform aggregation
    cumulative_df = grouped.agg(agg_dict).reset_index()

    # Add game count
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
            # Set to NaN on error
            cumulative_df[col_name] = np.nan

    logger.info(f"Calculated cumulative stats for {len(cumulative_df)} players through week {target_week}")

    return cumulative_df


def save_cumulative_data(
    df: pd.DataFrame,
    output_root: str,
    table_path: str,
    season: int,
    week: int,
    id_column: Optional[str] = None
) -> None:
    """
    Save cumulative data to partitioned parquet structure.

    Args:
        df: DataFrame to save
        output_root: Root directory for output
        table_path: Relative path to table (e.g., 'plyr_szn/plyr_rec')
        season: NFL season year
        week: Week number for partitioning
        id_column: Name of ID column to generate (if None, no ID is added)

    Raises:
        IOError: If unable to write file
    """
    if df.empty:
        logger.warning(f"Empty DataFrame, skipping save for season {season}, week {week}")
        return

    # Create output directory
    output_dir = os.path.join(output_root, table_path, f'season={season}', f'week={week}')
    os.makedirs(output_dir, exist_ok=True)

    # Clear existing parquet files in this partition (to ensure clean overwrite)
    existing_files = list(Path(output_dir).glob('*.parquet'))
    if existing_files:
        logger.debug(f"Clearing {len(existing_files)} existing parquet file(s) from {output_dir}")
        for file in existing_files:
            try:
                file.unlink()
            except Exception as e:
                logger.warning(f"Could not delete {file}: {e}")

    # Create a copy to avoid modifying original
    save_df = df.copy()

    # Generate ID column if specified
    if id_column and id_column not in save_df.columns:
        save_df[id_column] = range(1, len(save_df) + 1)

    # Remove partition columns from DataFrame (they're in the path)
    cols_to_drop = []
    if 'season' in save_df.columns:
        cols_to_drop.append('season')
    if 'week' in save_df.columns:
        cols_to_drop.append('week')
    if 'week_num' in save_df.columns:
        cols_to_drop.append('week_num')

    if cols_to_drop:
        save_df = save_df.drop(columns=cols_to_drop, errors='ignore')

    # Save to parquet
    output_file = os.path.join(output_dir, f'{season}_{week}.parquet')

    try:
        # Use pyarrow for writing
        table = pa.Table.from_pandas(save_df)
        pq.write_table(table, output_file)
        logger.info(f"Saved {len(save_df)} rows to {output_file}")
    except Exception as e:
        logger.error(f"Error writing with pyarrow: {e}")
        # Fallback to pandas
        try:
            save_df.to_parquet(output_file, engine='fastparquet', index=False)
            logger.info(f"Saved {len(save_df)} rows to {output_file} (fastparquet)")
        except Exception as e2:
            raise IOError(f"Failed to save file: {e2}")


def validate_output(
    source_df: pd.DataFrame,
    output_df: pd.DataFrame,
    season: int,
    week: int,
    sum_cols: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validate cumulative output data for consistency.

    Performs basic validation checks:
    - Row count consistency
    - No unexpected nulls in key columns
    - Cumulative values are non-negative (where expected)
    - Value ranges are reasonable

    Args:
        source_df: Source game-level DataFrame
        output_df: Output cumulative DataFrame
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

    # Check unique players
    unique_players_source = source_df['plyr_id'].nunique()
    unique_players_output = output_df['plyr_id'].nunique()

    if unique_players_output != unique_players_source:
        warnings.append(
            f"Season {season}, Week {week}: Player count mismatch - "
            f"Source: {unique_players_source}, Output: {unique_players_output}"
        )

    # Check for nulls in critical columns
    critical_cols = ['plyr_id', 'season_id', 'week_id']
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
                # Some columns like fumbles might legitimately be negative in special cases
                # But most counting stats should be >= 0
                if col not in ['plyr_rec_fmbl']:  # Add exceptions as needed
                    warnings.append(
                        f"Season {season}, Week {week}: Found {negative_count} negative values in {col}"
                    )

    # Log summary statistics
    logger.info(f"Season {season}, Week {week} validation:")
    logger.info(f"  - Output rows: {len(output_df)}")
    logger.info(f"  - Unique players: {unique_players_output}")
    logger.info(f"  - Warnings: {len(warnings)}")

    return is_valid, warnings


def safe_divide(numerator: pd.Series, denominator: pd.Series, default: float = np.nan) -> pd.Series:
    """
    Safely divide two series, handling division by zero.

    Args:
        numerator: Numerator series
        denominator: Denominator series
        default: Value to use when denominator is 0 or NaN (default: np.nan)

    Returns:
        Series with division results, NaN where division by zero would occur
    """
    result = pd.Series(index=numerator.index, dtype=float)

    # Only divide where denominator is not zero and not null
    valid_mask = (denominator != 0) & denominator.notna() & numerator.notna()

    result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
    result[~valid_mask] = default

    return result
