#!/usr/bin/env python3
"""
NFL Data Pipeline Master Script

Orchestrates the execution of all data pipeline scripts in the correct order.
Supports exclusions and provides detailed logging of the pipeline execution.
"""

import os
import sys
import subprocess
import time
import argparse
from datetime import datetime
from typing import List, Tuple, Dict

# Pipeline scripts to run in order
PIPELINE_SCRIPTS = [
    # (subdirectory, script_name, timeout_minutes)
    ('web_scrape', 'scrape_master.py', 480),  # 8 hours for scraping
    ('clean_data', 'clean_master.py', 60),    # 1 hour for cleaning
    ('database', 'db_master.py', 60),          # 1 hour for database
    ('web_scrape\\scripts\\api', 'weather.py', 30),  # 30 minutes for weather API
    ('database\\insert_scripts', 'nfl_game_weather.py', 30),  # 30 minutes for insert
    ('data_export', 'export_raw_db_data.py', 60),  # 1 hour for data export
]

# Scripts to exclude from execution
# Add script names here to skip them (e.g., ['scrape_master.py', 'weather.py'])
EXCLUSIONS = ['scrape_master.py']

def log_message(level: str, message: str):
    """Print formatted log messages with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {level}: {message}")

def run_script(script_path: str, script_name: str, timeout_minutes: int = 30, extra_args: List[str] = None) -> Tuple[bool, str]:
    """
    Execute a Python script and return success status and output.
    
    Args:
        script_path: Full path to the script
        script_name: Display name for the script
        timeout_minutes: Timeout in minutes for script execution
        
    Returns:
        Tuple of (success: bool, output: str)
    """
    print(f"\n{'='*70}")
    log_message("INFO", f"Running: {script_name}")
    log_message("INFO", f"Path: {script_path}")
    log_message("INFO", f"Timeout: {timeout_minutes} minutes")
    print(f"{'='*70}")
    
    try:
        # Change to the script's directory and run it
        script_dir = os.path.dirname(script_path)
        # Build command with extra arguments if provided
        command = [sys.executable, script_path]
        if extra_args:
            command.extend(extra_args)

        result = subprocess.run(
            command,
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=timeout_minutes * 60  # Convert minutes to seconds
        )
        # Print stdout if available
        if result.stdout:
            print(result.stdout)
        
        if result.returncode == 0:
            log_message("SUCCESS", f"{script_name} completed successfully")
            return True, result.stdout
        else:
            log_message("ERROR", f"{script_name} failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error output: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        log_message("ERROR", f"{script_name} timed out after {timeout_minutes} minutes")
        return False, "Script execution timed out"
    except Exception as e:
        log_message("ERROR", f"Failed to run {script_name}: {e}")
        return False, str(e)

def get_script_path(base_dir: str, subdirectory: str, script_name: str) -> str:
    """
    Build the full path to a script.
    
    Args:
        base_dir: Base directory of the project
        subdirectory: Subdirectory containing the script
        script_name: Name of the script file
        
    Returns:
        Full path to the script
    """
    return os.path.join(base_dir, subdirectory, script_name)

def validate_script_exists(script_path: str) -> bool:
    """
    Check if a script file exists.
    
    Args:
        script_path: Path to the script
        
    Returns:
        True if the script exists, False otherwise
    """
    return os.path.exists(script_path) and os.path.isfile(script_path)

def build_export_args(seasons: str = None, weeks: str = None, tables: str = None, categories: str = None) -> List[str]:
    """Build command line arguments for the export script."""
    args = []

    if seasons:
        args.extend(['--seasons', seasons])
    if weeks:
        args.extend(['--weeks', weeks])
    if tables:
        args.extend(['--tables', tables])
    if categories:
        args.extend(['--categories', categories])

    return args

def main():
    """Main execution function for the data pipeline"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='NFL Data Pipeline Master Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Export Arguments (applied only to data export step):
  These arguments will be passed to the export_raw_db_data.py script:

  --export-seasons SEASONS    Season filter for export (e.g., "2024" or "2023,2024" or "2023-2024")
  --export-weeks WEEKS        Week filter for export (e.g., "1" or "1,2,3" or "1-8")
  --export-tables TABLES      Comma-separated list of specific tables to export
  --export-categories CATS    Comma-separated list of categories to export (static,reference,game_level,season_cumulative)

Examples:
  # Run full pipeline
  python data_pipeline.py

  # Run pipeline with export filtering
  python data_pipeline.py --export-seasons 2024 --export-weeks 1-8
  python data_pipeline.py --export-categories game_level,season_cumulative
        """
    )

    parser.add_argument(
        '--export-seasons',
        type=str,
        help='Season filter for export (e.g., "2024" or "2023,2024" or "2023-2024")'
    )

    parser.add_argument(
        '--export-weeks',
        type=str,
        help='Week filter for export (e.g., "1" or "1,2,3" or "1-8")'
    )

    parser.add_argument(
        '--export-tables',
        type=str,
        help='Comma-separated list of specific tables to export'
    )

    parser.add_argument(
        '--export-categories',
        type=str,
        help='Comma-separated list of categories to export (static,reference,game_level,season_cumulative)'
    )

    args = parser.parse_args()

    # Build export arguments
    export_args = build_export_args(
        seasons=args.export_seasons,
        weeks=args.export_weeks,
        tables=args.export_tables,
        categories=args.export_categories
    )

    if export_args:
        log_message("INFO", f"Export arguments: {' '.join(export_args)}")
    
    # Header
    print("\n" + "#"*70)
    print("#" + " "*20 + "NFL DATA PIPELINE MASTER" + " "*20 + "#")
    print("#"*70)
    
    start_time = time.time()
    
    # Get the base directory (NFL_ml)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    log_message("INFO", f"Base directory: {base_dir}")
    log_message("INFO", f"Total pipeline scripts: {len(PIPELINE_SCRIPTS)}")
    
    if EXCLUSIONS:
        log_message("INFO", f"Excluded scripts: {', '.join(EXCLUSIONS)}")
    
    # Build list of scripts to execute
    scripts_to_execute = []
    
    for item in PIPELINE_SCRIPTS:
        # Handle both old format (subdirectory, script_name) and new format (subdirectory, script_name, timeout)
        if len(item) == 3:
            subdirectory, script_name, timeout = item
        else:
            subdirectory, script_name = item
            timeout = 30  # Default timeout
        
        # Check if script is excluded
        if script_name in EXCLUSIONS:
            log_message("INFO", f"Skipping excluded script: {script_name}")
            continue
        
        script_path = get_script_path(base_dir, subdirectory, script_name)
        
        # Validate script exists
        if not validate_script_exists(script_path):
            log_message("WARNING", f"Script not found: {script_path}")
            continue
        
        scripts_to_execute.append((script_path, f"{subdirectory}/{script_name}", timeout))
    
    # Display execution plan
    print("\n" + "-"*70)
    log_message("INFO", "EXECUTION PLAN:")
    for i, (_, display_name, timeout) in enumerate(scripts_to_execute, 1):
        print(f"  {i}. {display_name} (timeout: {timeout} minutes)")
    print("-"*70)
    
    # Confirm execution
    response = input("\nProceed with pipeline execution? (y/n): ")
    if response.lower() != 'y':
        log_message("INFO", "Pipeline execution cancelled by user")
        return 0
    
    # Execute scripts
    successful_scripts = []
    failed_scripts = []
    execution_results = []
    
    for script_path, display_name, timeout in scripts_to_execute:
        script_name = os.path.basename(script_path)

        # Check if this is the export script and add export arguments if provided
        extra_args = None
        if script_name == 'export_raw_db_data.py' and export_args:
            extra_args = export_args
            log_message("INFO", f"Running {display_name} with export filters")

        # Run the script
        success, output = run_script(script_path, display_name, timeout, extra_args)
        
        if success:
            successful_scripts.append(display_name)
            execution_results.append((display_name, "SUCCESS"))
        else:
            failed_scripts.append(display_name)
            execution_results.append((display_name, "FAILED"))
            
            # Ask if user wants to continue after failure
            response = input(f"\n{display_name} failed. Continue with remaining scripts? (y/n): ")
            if response.lower() != 'y':
                log_message("INFO", "Pipeline execution stopped by user")
                break
    
    # Calculate execution time
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    # Print summary
    print("\n" + "#"*70)
    print("#" + " "*24 + "PIPELINE SUMMARY" + " "*24 + "#")
    print("#"*70)
    
    log_message("INFO", f"Total execution time: {minutes}m {seconds}s")
    log_message("INFO", f"Scripts executed: {len(execution_results)}")
    log_message("INFO", f"Successful: {len(successful_scripts)}")
    log_message("INFO", f"Failed: {len(failed_scripts)}")
    
    # Detailed results
    print("\n" + "-"*70)
    print("DETAILED RESULTS:")
    print("-"*70)
    
    if successful_scripts:
        print("\n✅ Successful scripts:")
        for script in successful_scripts:
            print(f"   - {script}")
    
    if failed_scripts:
        print("\n❌ Failed scripts:")
        for script in failed_scripts:
            print(f"   - {script}")
    
    # Pipeline status
    print("\n" + "="*70)
    if failed_scripts:
        log_message("WARNING", "PIPELINE COMPLETED WITH ERRORS")
        print("="*70)
        return 1
    else:
        log_message("SUCCESS", "PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n")
        log_message("INFO", "Pipeline execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        log_message("ERROR", f"Unexpected error: {e}")
        sys.exit(1)