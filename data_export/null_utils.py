import pandas as pd
import pyarrow.parquet as pq
import os
import logging
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime

# Audit configuration - set to True to export CSV audit files
AUDIT = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_season_filter(season_arg: str) -> List[int]:
    """Parse season argument into list of years.

    Args:
        season_arg: Season filter string (e.g., "2024", "2023,2024", "2022-2024")

    Returns:
        List of season years
    """
    if not season_arg:
        return None

    # Check if it's a range (e.g., "2022-2024")
    if '-' in season_arg:
        parts = season_arg.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid season range format: {season_arg}. Use format: YYYY-YYYY")
        try:
            start = int(parts[0])
            end = int(parts[1])
            return list(range(start, end + 1))
        except ValueError:
            raise ValueError(f"Invalid season range: {season_arg}. Years must be integers.")

    # Check if it's a comma-separated list (e.g., "2023,2024")
    elif ',' in season_arg:
        try:
            return [int(s.strip()) for s in season_arg.split(',')]
        except ValueError:
            raise ValueError(f"Invalid season list: {season_arg}. All values must be integers.")

    # Single season
    else:
        try:
            return [int(season_arg)]
        except ValueError:
            raise ValueError(f"Invalid season: {season_arg}. Must be an integer.")

def parse_week_filter(week_arg: str) -> List[int]:
    """Parse week argument into list of weeks.

    Args:
        week_arg: Week filter string (e.g., "5", "1,3,8", "1-8")

    Returns:
        List of week numbers
    """
    if not week_arg:
        return None

    # Check if it's a range (e.g., "1-8")
    if '-' in week_arg:
        parts = week_arg.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid week range format: {week_arg}. Use format: N-N")
        try:
            start = int(parts[0])
            end = int(parts[1])
            return list(range(start, end + 1))
        except ValueError:
            raise ValueError(f"Invalid week range: {week_arg}. Weeks must be integers.")

    # Check if it's a comma-separated list (e.g., "1,3,8")
    elif ',' in week_arg:
        try:
            return [int(w.strip()) for w in week_arg.split(',')]
        except ValueError:
            raise ValueError(f"Invalid week list: {week_arg}. All values must be integers.")

    # Single week
    else:
        try:
            return [int(week_arg)]
        except ValueError:
            raise ValueError(f"Invalid week: {week_arg}. Must be an integer.")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for season and week filtering.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Process NFL player statistics and handle null values',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s                           # Process all data (default)
  %(prog)s --season 2024            # Single season
  %(prog)s --season 2023,2024       # Multiple seasons (comma-separated)
  %(prog)s --season 2022-2024       # Season range (inclusive)
  %(prog)s --season 2024 --week 5   # Single season and week
  %(prog)s --season 2024 --week 1,3,8  # Specific weeks
  %(prog)s --season 2024 --week 1-8    # Week range
        '''
    )

    parser.add_argument(
        '--season',
        type=str,
        help='Season filter: single year (2024), comma-separated list (2023,2024), or range (2022-2024)'
    )

    parser.add_argument(
        '--week',
        type=str,
        help='Week filter: single week (5), comma-separated list (1,3,8), or range (1-8). Requires --season.'
    )

    args = parser.parse_args()

    # Validate that week requires season
    if args.week and not args.season:
        parser.error("--week requires --season to be specified")

    return args

def get_filtered_partitions(table_path: Path, seasons: Optional[List[int]] = None,
                           weeks: Optional[List[int]] = None) -> List[Tuple[Path, int, int]]:
    """Get list of partition paths that match the season/week filters.

    Args:
        table_path: Path to the partitioned table directory
        seasons: List of season years to include (None = all)
        weeks: List of week numbers to include (None = all)

    Returns:
        List of tuples (partition_path, season, week)
    """
    partitions = []
    table_path = Path(table_path)

    if not table_path.exists():
        logger.warning(f"Table path does not exist: {table_path}")
        return partitions

    # Iterate through season partitions
    for season_dir in sorted(table_path.glob("season=*")):
        season = int(season_dir.name.split("=")[1])

        # Filter by season if specified
        if seasons and season not in seasons:
            continue

        # Iterate through week partitions
        for week_dir in sorted(season_dir.glob("week=*")):
            week = int(week_dir.name.split("=")[1])

            # Filter by week if specified
            if weeks and week not in weeks:
                continue

            partitions.append((week_dir, season, week))

    return partitions

class BaseNullHandler:
    def __init__(self, raw_dir: str, output_dir: str = None):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir) if output_dir else None
        self.null_columns_summary = {}

    def read_partitioned_table(self, table_path: str) -> pd.DataFrame:
        """Read a partitioned parquet table into a DataFrame"""
        try:
            df = pd.read_parquet(table_path)
            logger.info(f"Successfully read table from {table_path}")
            logger.info(f"Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error reading table from {table_path}: {e}")
            raise

    def get_null_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of columns with at least one NULL value"""
        null_cols = df.columns[df.isnull().any()].tolist()
        return null_cols

    def export_null_rows_to_csv(self, df: pd.DataFrame, table_name: str, null_cols_after: List[str]) -> None:
        """Export rows with remaining nulls to CSV"""
        if null_cols_after:
            rows_with_nulls = df[df[null_cols_after].isnull().any(axis=1)]
            if len(rows_with_nulls) > 0:
                csv_dir = Path(r"C:\Users\nocap\Desktop\code\NFL_ml\data_export\null_values")
                csv_dir.mkdir(parents=True, exist_ok=True)
                csv_filename = f"{table_name}_rows_with_nulls.csv"
                csv_path = csv_dir / csv_filename
                rows_with_nulls.to_csv(csv_path, index=False)
                logger.info(f"\nExported {len(rows_with_nulls)} rows with NULL values to {csv_filename}")
                logger.info(f"  Full path: {csv_path.absolute()}")

    def export_audit_csv(self, df: pd.DataFrame, table_name: str) -> None:
        """Export processed dataframe to audit CSV if AUDIT is True"""
        if AUDIT:
            # Create audit directory structure
            audit_base_dir = Path(r"C:\Users\nocap\Desktop\code\NFL_ml\data_export\audit")
            table_dir = audit_base_dir / table_name
            table_dir.mkdir(parents=True, exist_ok=True)

            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"{table_name}_{timestamp}.csv"
            csv_path = table_dir / csv_filename

            # Export to CSV
            df.to_csv(csv_path, index=False)
            logger.info(f"\n[AUDIT] Exported processed data to audit CSV:")
            logger.info(f"  Filename: {csv_filename}")
            logger.info(f"  Full path: {csv_path.absolute()}")
            logger.info(f"  Shape: {df.shape}")

    def process_table(self, table_name: str, table_path: str, handler_func) -> None:
        """Process a single table for null value handling"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing table: {table_name}")
        logger.info(f"Path: {table_path}")
        logger.info(f"{'='*60}")

        # Read the table
        df = self.read_partitioned_table(table_path)

        # Get columns with nulls before processing
        null_cols_before = self.get_null_columns(df)
        logger.info(f"Columns with NULL values BEFORE imputation: {len(null_cols_before)}")
        if null_cols_before:
            logger.info(f"  {null_cols_before}")

        # Apply table-specific null handling
        df = handler_func(df)

        # Get columns with nulls after processing
        null_cols_after = self.get_null_columns(df)
        logger.info(f"\nColumns with NULL values AFTER imputation: {len(null_cols_after)}")
        if null_cols_after:
            logger.info(f"  {null_cols_after}")

        # Store summary
        self.null_columns_summary[table_name] = {
            'before': null_cols_before,
            'after': null_cols_after,
            'shape': df.shape
        }

        # Log detailed summary
        logger.info(f"\nSummary for {table_name}:")
        logger.info(f"  Original NULL columns: {len(null_cols_before)}")
        logger.info(f"  Remaining NULL columns: {len(null_cols_after)}")
        logger.info(f"  DataFrame shape: {df.shape}")

        if null_cols_after:
            logger.info(f"\nColumns still containing NULL values:")
            for col in null_cols_after:
                null_count = df[col].isnull().sum()
                logger.info(f"    {col}: {null_count} NULLs")

            # Export rows with remaining nulls to CSV
            self.export_null_rows_to_csv(df, table_name, null_cols_after)

        # Export audit CSV if enabled
        self.export_audit_csv(df, table_name)

        return df

    def write_partitioned_parquet(self, df: pd.DataFrame, table_name: str, category: str,
                                  season: int, week: int) -> None:
        """Write processed DataFrame to partitioned Parquet file in clean directory.

        Args:
            df: Processed DataFrame
            table_name: Name of the table (e.g., 'plyr_def', 'plyr_gm_def')
            category: Category directory (e.g., 'plyr_szn', 'plyr_gm')
            season: Season year
            week: Week number
        """
        if not self.output_dir:
            logger.warning("No output directory specified. Skipping Parquet write.")
            return

        # Build output path maintaining partition structure
        output_path = self.output_dir / category / table_name / f"season={season}" / f"week={week}"
        output_path.mkdir(parents=True, exist_ok=True)

        # Write to Parquet
        output_file = output_path / "data.parquet"
        df.to_parquet(output_file, engine='pyarrow', index=False)

        logger.info(f"Written partition to: {output_file}")
        logger.info(f"  Shape: {df.shape}")

    def process_partitioned_table(self, table_name: str, table_path: str, category: str,
                                  handler_func, seasons: Optional[List[int]] = None,
                                  weeks: Optional[List[int]] = None) -> int:
        """Process a partitioned table with season/week filtering.

        Args:
            table_name: Name of the table (e.g., 'plyr_def', 'plyr_gm_def')
            table_path: Path to the raw partitioned table directory
            category: Category directory (e.g., 'plyr_szn', 'plyr_gm')
            handler_func: Function to handle null values for this table
            seasons: List of seasons to process (None = all)
            weeks: List of weeks to process (None = all)

        Returns:
            Number of partitions processed
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing table: {table_name}")
        logger.info(f"Path: {table_path}")
        if seasons:
            logger.info(f"Season filter: {seasons}")
        if weeks:
            logger.info(f"Week filter: {weeks}")
        logger.info(f"{'='*60}")

        # Get filtered partitions
        partitions = get_filtered_partitions(Path(table_path), seasons, weeks)

        if not partitions:
            logger.warning(f"No partitions found matching filters for {table_name}")
            return 0

        logger.info(f"Found {len(partitions)} partition(s) to process")

        # Track aggregated stats
        total_rows = 0
        null_cols_before_all = set()
        null_cols_after_all = set()

        # Process each partition
        for partition_path, season, week in partitions:
            logger.info(f"\nProcessing partition: season={season}, week={week}")

            # Read partition
            try:
                df = pd.read_parquet(partition_path)
                logger.info(f"  Read {len(df)} rows")
                total_rows += len(df)
            except Exception as e:
                logger.error(f"  Error reading partition {partition_path}: {e}")
                continue

            # Get columns with nulls before processing
            null_cols_before = self.get_null_columns(df)
            null_cols_before_all.update(null_cols_before)
            logger.info(f"  Columns with NULL values BEFORE imputation: {len(null_cols_before)}")

            # Apply table-specific null handling
            df = handler_func(df)

            # Get columns with nulls after processing
            null_cols_after = self.get_null_columns(df)
            null_cols_after_all.update(null_cols_after)
            logger.info(f"  Columns with NULL values AFTER imputation: {len(null_cols_after)}")

            # Write to clean directory if output_dir is specified
            if self.output_dir:
                self.write_partitioned_parquet(df, table_name, category, season, week)

        # Store summary
        self.null_columns_summary[table_name] = {
            'before': sorted(list(null_cols_before_all)),
            'after': sorted(list(null_cols_after_all)),
            'partitions': len(partitions),
            'total_rows': total_rows
        }

        # Log final summary for this table
        logger.info(f"\nSummary for {table_name}:")
        logger.info(f"  Partitions processed: {len(partitions)}")
        logger.info(f"  Total rows: {total_rows}")
        logger.info(f"  Unique NULL columns before: {len(null_cols_before_all)}")
        logger.info(f"  Unique NULL columns after: {len(null_cols_after_all)}")

        if null_cols_after_all:
            logger.info(f"\nColumns still containing NULL values across all partitions:")
            logger.info(f"  {sorted(list(null_cols_after_all))}")

        return len(partitions)

    def print_final_summary(self) -> None:
        """Print final summary of all processed tables"""
        logger.info(f"\n{'='*60}")
        logger.info("FINAL SUMMARY - All Tables")
        logger.info(f"{'='*60}")

        for table_name, summary in self.null_columns_summary.items():
            logger.info(f"\nTable: {table_name}")

            # Handle both old format (shape) and new format (partitions/total_rows)
            if 'shape' in summary:
                logger.info(f"  Shape: {summary['shape']}")
            elif 'partitions' in summary:
                logger.info(f"  Partitions processed: {summary['partitions']}")
                logger.info(f"  Total rows: {summary['total_rows']}")

            logger.info(f"  NULL columns before: {len(summary['before'])}")
            logger.info(f"  NULL columns after: {len(summary['after'])}")
            if summary['after']:
                logger.info(f"  Remaining NULL columns: {summary['after']}")


# Utility functions for null_master.py orchestration

def copy_partition(source_path: Path, dest_path: Path) -> None:
    """Copy a single partition preserving structure.

    Args:
        source_path: Path to source partition directory
        dest_path: Path to destination partition directory
    """
    try:
        # Create destination directory
        dest_path.mkdir(parents=True, exist_ok=True)

        # Copy all parquet files in the partition
        for parquet_file in source_path.glob("*.parquet"):
            dest_file = dest_path / parquet_file.name
            shutil.copy2(parquet_file, dest_file)
            logger.debug(f"Copied: {parquet_file.name} to {dest_path}")
    except Exception as e:
        logger.error(f"Error copying partition from {source_path} to {dest_path}: {e}")
        raise


def copy_partitioned_table(table_name: str, source_root: Path, dest_root: Path,
                           relative_path: str, seasons: Optional[List[int]] = None,
                           weeks: Optional[List[int]] = None) -> int:
    """Copy entire table with filter support.

    Args:
        table_name: Name of the table
        source_root: Root directory of source (raw) data
        dest_root: Root directory of destination (clean) data
        relative_path: Relative path to table (e.g., 'plyr_gm/plyr_gm_fmbl')
        seasons: List of seasons to include (None = all)
        weeks: List of weeks to include (None = all)

    Returns:
        Number of partitions copied
    """
    logger.info(f"\nCopying table: {table_name}")
    logger.info(f"  From: {source_root / relative_path}")
    logger.info(f"  To: {dest_root / relative_path}")

    source_table_path = source_root / relative_path
    dest_table_path = dest_root / relative_path

    if not source_table_path.exists():
        logger.warning(f"Source table does not exist: {source_table_path}")
        return 0

    # Get filtered partitions
    partitions = get_filtered_partitions(source_table_path, seasons, weeks)

    if not partitions:
        logger.warning(f"No partitions found matching filters for {table_name}")
        return 0

    logger.info(f"Found {len(partitions)} partition(s) to copy")

    # Copy each partition
    partitions_copied = 0
    for partition_path, season, week in partitions:
        # Build destination path preserving partition structure
        relative_partition = partition_path.relative_to(source_table_path)
        dest_partition_path = dest_table_path / relative_partition

        try:
            copy_partition(partition_path, dest_partition_path)
            partitions_copied += 1
            logger.info(f"  Copied partition: season={season}, week={week}")
        except Exception as e:
            logger.error(f"  Failed to copy partition season={season}, week={week}: {e}")

    logger.info(f"Successfully copied {partitions_copied}/{len(partitions)} partitions")
    return partitions_copied


def copy_non_partitioned_file(source_root: Path, dest_root: Path, filename: str) -> bool:
    """Copy a non-partitioned parquet file.

    Args:
        source_root: Root directory of source (raw) data
        dest_root: Root directory of destination (clean) data
        filename: Name of the file (e.g., 'nfl_team.parquet')

    Returns:
        True if copied successfully, False otherwise
    """
    source_file = source_root / filename
    dest_file = dest_root / filename

    if not source_file.exists():
        logger.warning(f"Source file does not exist: {source_file}")
        return False

    try:
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, dest_file)
        logger.info(f"Copied static file: {filename}")
        return True
    except Exception as e:
        logger.error(f"Error copying {filename}: {e}")
        return False


def validate_directory_nulls(directory: Path, table_name: str = None) -> Dict[str, List[str]]:
    """Check directory for NULL values in parquet files.

    Args:
        directory: Directory to scan for parquet files
        table_name: Optional table name for filtering

    Returns:
        Dictionary mapping table names to list of columns with nulls
    """
    null_report = {}

    # Find all parquet files
    parquet_files = list(directory.rglob("*.parquet"))

    if not parquet_files:
        logger.warning(f"No parquet files found in {directory}")
        return null_report

    logger.info(f"Validating {len(parquet_files)} parquet file(s) for NULL values...")

    # Group files by table
    table_files = {}
    for file_path in parquet_files:
        # Extract table name from path
        # Assuming structure: .../category/table_name/season=X/week=Y/file.parquet
        parts = file_path.parts

        # Find table name (directory before season=X or the file itself for non-partitioned)
        if file_path.name in ['nfl_team.parquet', 'nfl_season.parquet', 'nfl_week.parquet', 'nfl_gm_quarter.parquet']:
            tbl_name = file_path.stem
        else:
            # Find the part before 'season=' partition
            for i, part in enumerate(parts):
                if part.startswith('season='):
                    tbl_name = parts[i-1]
                    break
            else:
                # No season partition, might be in players/ directory
                if 'players' in parts:
                    tbl_name = parts[parts.index('players') + 1]
                else:
                    tbl_name = file_path.parent.name

        if table_name and tbl_name != table_name:
            continue

        if tbl_name not in table_files:
            table_files[tbl_name] = []
        table_files[tbl_name].append(file_path)

    # Check each table for nulls
    for tbl_name, files in table_files.items():
        null_cols = set()

        for file_path in files:
            try:
                df = pd.read_parquet(file_path)
                file_null_cols = df.columns[df.isnull().any()].tolist()
                null_cols.update(file_null_cols)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

        if null_cols:
            null_report[tbl_name] = sorted(list(null_cols))

    return null_report


def run_script_with_args(script_path: Path, args: List[str]) -> bool:
    """Execute a Python script with inherited arguments.

    Args:
        script_path: Path to the script to execute
        args: List of command-line arguments to pass

    Returns:
        True if script executed successfully, False otherwise
    """
    try:
        # Build command
        cmd = [sys.executable, str(script_path)] + args

        logger.info(f"Executing: {script_path.name}")
        logger.debug(f"Command: {' '.join(cmd)}")

        # Run script and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=script_path.parent
        )

        # Log output
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"  {line}")

        if result.stderr and result.returncode != 0:
            for line in result.stderr.splitlines():
                logger.error(f"  {line}")

        # Check return code
        if result.returncode == 0:
            logger.info(f"Successfully completed: {script_path.name}")
            return True
        else:
            logger.error(f"Script failed with return code {result.returncode}: {script_path.name}")
            return False

    except Exception as e:
        logger.error(f"Error executing {script_path}: {e}")
        return False