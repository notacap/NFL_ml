#!/usr/bin/env python3
"""
NFL Data Null Handling Master Orchestration Script

This script orchestrates the complete null handling pipeline for NFL data:
1. Executes all null handling scripts from handle_null_values directory
2. Copies unprocessed parquet files from raw to clean directory
3. Validates final dataset for remaining NULL values
4. Generates comprehensive summary report

Author: Claude Code
Date: 2025-10-14
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime

# Import utility functions from null_utils
from null_utils import (
    parse_season_filter,
    parse_week_filter,
    copy_partitioned_table,
    copy_non_partitioned_file,
    validate_directory_nulls,
    run_script_with_args
)

# Configure logging
log_dir = Path(r"C:\Users\nocap\Desktop\code\NFL_ml\data_export\logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"null_master_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Data directories
RAW_ROOT = Path(r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw")
CLEAN_ROOT = Path(r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\clean")
SCRIPT_DIR = Path(r"C:\Users\nocap\Desktop\code\NFL_ml\data_export\handle_null_values")

# Script registry - defines all null handling scripts and their metadata
SCRIPT_REGISTRY = [
    {'name': 'plyr_def.py', 'table': 'plyr_def', 'category': 'plyr_szn'},
    {'name': 'plyr_gm_def.py', 'table': 'plyr_gm_def', 'category': 'plyr_gm'},
    {'name': 'plyr_gm_pass.py', 'table': 'plyr_gm_pass', 'category': 'plyr_gm'},
    {'name': 'plyr_gm_rec.py', 'table': 'plyr_gm_rec', 'category': 'plyr_gm'},
    {'name': 'plyr_gm_rush.py', 'table': 'plyr_gm_rush', 'category': 'plyr_gm'},
    {'name': 'plyr_pass.py', 'table': 'plyr_pass', 'category': 'plyr_szn'},
    {'name': 'plyr_rec.py', 'table': 'plyr_rec', 'category': 'plyr_szn'},
    {'name': 'plyr_rush.py', 'table': 'plyr_rush', 'category': 'plyr_szn'},
    {'name': 'plyr_rz_pass.py', 'table': 'plyr_rz_pass', 'category': 'plyr_szn'},
    {'name': 'plyr_rz_rec.py', 'table': 'plyr_rz_rec', 'category': 'plyr_rz_rec'},
    {'name': 'plyr_rz_rush.py', 'table': 'plyr_rz_rush', 'category': 'plyr_szn'},
    {'name': 'tm_conv.py', 'table': 'tm_conv', 'category': 'tm_szn'},
    {'name': 'tm_def.py', 'table': 'tm_def', 'category': 'tm_szn'},
    {'name': 'tm_def_conv.py', 'table': 'tm_def_conv', 'category': 'tm_szn'},
    {'name': 'tm_def_pass.py', 'table': 'tm_def_pass', 'category': 'tm_szn'},
    {'name': 'tm_gm_drive.py', 'table': 'tm_gm_drive', 'category': 'tm_gm'},
]

# Tables that will be processed by null handler scripts
PROCESSED_TABLES = {script['table'] for script in SCRIPT_REGISTRY}

# Unprocessed tables - will be copied directly to clean directory
UNPROCESSED_TABLES = {
    # Player game-level
    'plyr_gm_fmbl': 'plyr_gm/plyr_gm_fmbl',
    'plyr_gm_snap_ct': 'plyr_gm/plyr_gm_snap_ct',
    'plyr_gm_starters': 'plyr_gm/plyr_gm_starters',

    # Player season cumulative
    'plyr_scoring': 'plyr_szn/plyr_scoring',

    # Team game-level
    'tm_gm_stats': 'tm_gm/tm_gm_stats',
    'tm_gm_exp_pts': 'tm_gm/tm_gm_exp_pts',

    # Team season cumulative
    'tm_pass': 'tm_szn/tm_pass',
    'tm_rush': 'tm_szn/tm_rush',
    'tm_rec': 'tm_szn/tm_rec',
    'tm_def_rush': 'tm_szn/tm_def_rush',
    'tm_def_dr_against_avg': 'tm_szn/tm_def_dr_against_avg',
    'tm_def_vs_qb': 'tm_szn/tm_def_vs_qb',
    'tm_def_vs_rb': 'tm_szn/tm_def_vs_rb',
    'tm_def_vs_te': 'tm_szn/tm_def_vs_te',
    'tm_def_vs_wr': 'tm_szn/tm_def_vs_wr',
    'nfl_standings': 'tm_szn/nfl_standings',

    # Game info
    'nfl_game': 'gm_info/nfl_game',
    'nfl_game_info': 'gm_info/nfl_game_info',
    'nfl_gm_weather': 'gm_info/nfl_gm_weather',
    'nfl_game_pbp': 'gm_info/nfl_game_pbp',
    'injury_report': 'gm_info/injury_report',

    # Reference tables (season partitioned only)
    'plyr': 'players/plyr',
    'multi_tm_plyr': 'players/multi_tm_plyr',
}

# Static (non-partitioned) files
STATIC_FILES = [
    'nfl_team.parquet',
    'nfl_season.parquet',
    'nfl_week.parquet',
    'nfl_gm_quarter.parquet',
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Master orchestration script for NFL data null handling pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s                                    # Process all data
  %(prog)s --season 2024                     # Process 2024 season only
  %(prog)s --season 2024 --week 1-8          # Process weeks 1-8 of 2024
  %(prog)s --plyr_def                        # Process only plyr_def table
  %(prog)s --exclude plyr_def.py             # Exclude plyr_def script
  %(prog)s --validate-only                   # Only validate clean directory
  %(prog)s --season 2023,2024 --tm_gm_stats  # Copy tm_gm_stats for 2023-2024
        '''
    )

    # Season and week filters
    parser.add_argument(
        '--season',
        type=str,
        help='Season filter: single year (2024), comma-separated (2023,2024), or range (2022-2024)'
    )

    parser.add_argument(
        '--week',
        type=str,
        help='Week filter: single week (5), comma-separated (1,3,8), or range (1-8). Requires --season.'
    )

    # Table-specific flags (dynamic)
    all_tables = PROCESSED_TABLES.union(set(UNPROCESSED_TABLES.keys()))
    for table in sorted(all_tables):
        parser.add_argument(
            f'--{table}',
            action='store_true',
            help=f'Process only {table} table'
        )

    # Exclusion list
    parser.add_argument(
        '--exclude',
        type=str,
        help='Comma-separated list of scripts to exclude (e.g., plyr_def.py,plyr_pass.py)'
    )

    # Validation only mode
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only run NULL validation on clean directory without processing'
    )

    args = parser.parse_args()

    # Validate that week requires season
    if args.week and not args.season:
        parser.error("--week requires --season to be specified")

    return args


def get_scripts_to_execute(args: argparse.Namespace, exclude_list: Set[str]) -> List[Dict]:
    """Determine which null handling scripts to execute based on arguments.

    Args:
        args: Parsed command-line arguments
        exclude_list: Set of script names to exclude

    Returns:
        List of script metadata dictionaries to execute
    """
    scripts = []

    # Check if any specific table flags are set
    specific_tables = []
    for script in SCRIPT_REGISTRY:
        table_name = script['table']
        if hasattr(args, table_name.replace('-', '_')) and getattr(args, table_name.replace('-', '_')):
            specific_tables.append(table_name)

    # If specific tables requested, only include those
    if specific_tables:
        for script in SCRIPT_REGISTRY:
            if script['table'] in specific_tables and script['name'] not in exclude_list:
                scripts.append(script)
    else:
        # Include all scripts except those in exclude list
        for script in SCRIPT_REGISTRY:
            if script['name'] not in exclude_list:
                scripts.append(script)

    return scripts


def get_tables_to_copy(args: argparse.Namespace) -> List[tuple]:
    """Determine which unprocessed tables to copy based on arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        List of (table_name, relative_path) tuples to copy
    """
    tables_to_copy = []

    # Check if any specific table flags are set
    specific_tables = []
    for table_name in UNPROCESSED_TABLES.keys():
        # Handle table names with underscores in argparse
        if hasattr(args, table_name) and getattr(args, table_name):
            specific_tables.append(table_name)

    # If specific tables requested, only include those
    if specific_tables:
        for table_name in specific_tables:
            if table_name in UNPROCESSED_TABLES:
                tables_to_copy.append((table_name, UNPROCESSED_TABLES[table_name]))
    else:
        # Copy all unprocessed tables
        for table_name, relative_path in UNPROCESSED_TABLES.items():
            tables_to_copy.append((table_name, relative_path))

    return tables_to_copy


def execute_null_handlers(scripts: List[Dict], args: argparse.Namespace,
                          seasons: Optional[List[int]], weeks: Optional[List[int]]) -> Dict[str, bool]:
    """Execute null handling scripts with inherited filters.

    Args:
        scripts: List of script metadata dictionaries
        args: Parsed command-line arguments
        seasons: List of seasons to process
        weeks: List of weeks to process

    Returns:
        Dictionary mapping script names to success status
    """
    logger.info(f"\n{'='*80}")
    logger.info("PHASE 1: EXECUTING NULL HANDLING SCRIPTS")
    logger.info(f"{'='*80}")

    if not scripts:
        logger.info("No null handling scripts to execute.")
        return {}

    logger.info(f"Scripts to execute: {len(scripts)}")
    for script in scripts:
        logger.info(f"  - {script['name']} ({script['table']})")

    # Build command-line arguments to pass to scripts
    script_args = []
    if seasons:
        if len(seasons) == 1:
            script_args.extend(['--season', str(seasons[0])])
        else:
            # Format as comma-separated or range
            script_args.extend(['--season', ','.join(map(str, seasons))])

    if weeks:
        if len(weeks) == 1:
            script_args.extend(['--week', str(weeks[0])])
        else:
            # Format as comma-separated or range
            script_args.extend(['--week', ','.join(map(str, weeks))])

    # Execute each script
    results = {}
    for script in scripts:
        script_path = SCRIPT_DIR / script['name']

        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            results[script['name']] = False
            continue

        logger.info(f"\n{'-'*80}")
        success = run_script_with_args(script_path, script_args)
        results[script['name']] = success
        logger.info(f"{'-'*80}")

    return results


def copy_unprocessed_tables(tables: List[tuple], seasons: Optional[List[int]],
                            weeks: Optional[List[int]]) -> Dict[str, int]:
    """Copy unprocessed tables from raw to clean directory.

    Args:
        tables: List of (table_name, relative_path) tuples
        seasons: List of seasons to copy
        weeks: List of weeks to copy

    Returns:
        Dictionary mapping table names to number of partitions copied
    """
    logger.info(f"\n{'='*80}")
    logger.info("PHASE 2: COPYING UNPROCESSED TABLES")
    logger.info(f"{'='*80}")

    if not tables:
        logger.info("No unprocessed tables to copy.")
        return {}

    logger.info(f"Tables to copy: {len(tables)}")
    for table_name, _ in tables:
        logger.info(f"  - {table_name}")

    results = {}
    for table_name, relative_path in tables:
        partitions_copied = copy_partitioned_table(
            table_name=table_name,
            source_root=RAW_ROOT,
            dest_root=CLEAN_ROOT,
            relative_path=relative_path,
            seasons=seasons,
            weeks=weeks
        )
        results[table_name] = partitions_copied

    return results


def copy_static_files() -> Dict[str, bool]:
    """Copy static (non-partitioned) files from raw to clean directory.

    Returns:
        Dictionary mapping file names to success status
    """
    logger.info(f"\n{'='*80}")
    logger.info("PHASE 3: COPYING STATIC FILES")
    logger.info(f"{'='*80}")

    results = {}
    for filename in STATIC_FILES:
        success = copy_non_partitioned_file(RAW_ROOT, CLEAN_ROOT, filename)
        results[filename] = success

    return results


def validate_clean_directory() -> Dict[str, List[str]]:
    """Validate clean directory for remaining NULL values.

    Returns:
        Dictionary mapping table names to list of columns with NULLs
    """
    logger.info(f"\n{'='*80}")
    logger.info("PHASE 4: VALIDATING CLEAN DIRECTORY FOR NULL VALUES")
    logger.info(f"{'='*80}")

    if not CLEAN_ROOT.exists():
        logger.warning(f"Clean directory does not exist: {CLEAN_ROOT}")
        return {}

    null_report = validate_directory_nulls(CLEAN_ROOT)

    return null_report


def print_final_summary(script_results: Dict[str, bool], copy_results: Dict[str, int],
                       static_results: Dict[str, bool], null_report: Dict[str, List[str]]) -> None:
    """Print comprehensive final summary report.

    Args:
        script_results: Results from null handler script execution
        copy_results: Results from copying unprocessed tables
        static_results: Results from copying static files
        null_report: NULL validation report
    """
    logger.info(f"\n{'='*80}")
    logger.info("FINAL SUMMARY REPORT")
    logger.info(f"{'='*80}")

    # Script execution summary
    if script_results:
        logger.info(f"\nNull Handler Scripts ({len(script_results)} total):")
        successful = sum(1 for success in script_results.values() if success)
        failed = len(script_results) - successful
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")

        if failed > 0:
            logger.info(f"\n  Failed scripts:")
            for script_name, success in script_results.items():
                if not success:
                    logger.info(f"    - {script_name}")

    # Table copy summary
    if copy_results:
        logger.info(f"\nUnprocessed Tables Copied ({len(copy_results)} total):")
        total_partitions = sum(copy_results.values())
        logger.info(f"  Total partitions copied: {total_partitions}")

        for table_name, partitions in copy_results.items():
            logger.info(f"  - {table_name}: {partitions} partition(s)")

    # Static files summary
    if static_results:
        logger.info(f"\nStatic Files ({len(static_results)} total):")
        successful = sum(1 for success in static_results.values() if success)
        logger.info(f"  Successfully copied: {successful}/{len(static_results)}")

        failed_files = [name for name, success in static_results.items() if not success]
        if failed_files:
            logger.info(f"  Failed files: {', '.join(failed_files)}")

    # NULL validation summary
    logger.info(f"\nNULL Validation Results:")
    if null_report:
        logger.info(f"  Tables with NULL values: {len(null_report)}")
        logger.info(f"\n  Details:")
        for table_name, null_cols in sorted(null_report.items()):
            logger.info(f"    {table_name}:")
            logger.info(f"      Columns with NULLs: {len(null_cols)}")
            logger.info(f"      {null_cols}")
    else:
        logger.info(f"  No NULL values found in clean directory!")

    # Log file location
    logger.info(f"\nDetailed logs saved to: {log_file}")
    logger.info(f"{'='*80}")


def main():
    """Main orchestration function."""
    start_time = datetime.now()

    logger.info(f"{'='*80}")
    logger.info("NFL DATA NULL HANDLING MASTER ORCHESTRATION")
    logger.info(f"{'='*80}")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Raw directory: {RAW_ROOT}")
    logger.info(f"Clean directory: {CLEAN_ROOT}")
    logger.info(f"Log file: {log_file}")

    # Parse arguments
    args = parse_args()

    # Parse season and week filters
    seasons = parse_season_filter(args.season) if args.season else None
    weeks = parse_week_filter(args.week) if args.week else None

    if seasons:
        logger.info(f"Season filter: {seasons}")
    if weeks:
        logger.info(f"Week filter: {weeks}")

    # Parse exclusion list
    exclude_list = set()
    if args.exclude:
        exclude_list = set(s.strip() for s in args.exclude.split(','))
        logger.info(f"Excluding scripts: {exclude_list}")

    # Validation-only mode
    if args.validate_only:
        logger.info("\nRunning in VALIDATION-ONLY mode")
        null_report = validate_clean_directory()
        print_final_summary({}, {}, {}, null_report)
        return

    # Determine what to process
    scripts_to_execute = get_scripts_to_execute(args, exclude_list)
    tables_to_copy = get_tables_to_copy(args)

    # Execute pipeline
    script_results = execute_null_handlers(scripts_to_execute, args, seasons, weeks)
    copy_results = copy_unprocessed_tables(tables_to_copy, seasons, weeks)
    static_results = copy_static_files()
    null_report = validate_clean_directory()

    # Print final summary
    print_final_summary(script_results, copy_results, static_results, null_report)

    # Calculate and log execution time
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"\nExecution completed in {duration.total_seconds():.2f} seconds")
    logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nFatal error: {e}", exc_info=True)
        sys.exit(1)
