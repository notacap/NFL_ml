"""
Master script to run all NFL data scrapers.

This script executes all scraper scripts in the scripts directory and api subdirectory,
with options to exclude specific files.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

# Failed scraper tracking
failed_scrapers = {}  # script_path -> {'script_name': str, 'error': str, 'attempts': int}


def get_scraper_files(scripts_dir, api_dir):
    """Get list of all Python scraper files, excluding utility files."""
    scraper_files = []
    schedule_file = None
    
    # Files to always exclude
    excluded_files = {
        'scraper_utils.py',
        'view_logs.py',
        '__init__.py',
        'scrape_master.py',
        'games.py',
        'all_active_player.py',
        'weather.py',
        'temp_defense_advanced.py',
        'nfl_schedule.py'
    }
    
    # Get files from main scripts directory
    if os.path.exists(scripts_dir):
        for file in os.listdir(scripts_dir):
            if file.endswith('.py') and file not in excluded_files:
                file_path = os.path.join(scripts_dir, file)
                # Check if this is schedule.py
                if file == 'schedule.py':
                    schedule_file = file_path
                else:
                    scraper_files.append(file_path)
    
    # Get files from api subdirectory
    if os.path.exists(api_dir):
        for file in os.listdir(api_dir):
            if file.endswith('.py') and file not in excluded_files:
                scraper_files.append(os.path.join(api_dir, file))
    
    # Sort the files (excluding schedule.py)
    scraper_files = sorted(scraper_files)
    
    # Add schedule.py at the beginning if it exists
    if schedule_file:
        scraper_files.insert(0, schedule_file)
    
    return scraper_files


def create_failed_scrapers_directory():
    """
    Creates the directory for failed scraper CSV files.
    
    Returns:
        str: Path to the failed scrapers directory
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    failed_dir = os.path.join(current_dir, "failed_scrapers")
    if not os.path.exists(failed_dir):
        os.makedirs(failed_dir, exist_ok=True)
    return failed_dir


def track_failed_scraper(script_path, error_msg):
    """
    Track a failed scraper for later retry.
    
    Args:
        script_path (str): Path to the script that failed
        error_msg (str): Error message from the failure
    """
    global failed_scrapers
    
    script_name = os.path.basename(script_path)
    
    if script_path not in failed_scrapers:
        failed_scrapers[script_path] = {
            'script_name': script_name,
            'error': error_msg,
            'attempts': 1
        }
    else:
        failed_scrapers[script_path]['attempts'] += 1
        failed_scrapers[script_path]['error'] = error_msg


def save_failed_scrapers_csv():
    """
    Saves current failed scrapers to CSV file.
    
    Returns:
        str: Path to saved CSV file, or None if no failures
    """
    global failed_scrapers
    
    if not failed_scrapers:
        return None
    
    # Create failed scrapers directory
    failed_dir = create_failed_scrapers_directory()
    
    # Create CSV data
    csv_data = []
    for script_path, data in failed_scrapers.items():
        csv_data.append({
            'script_name': data['script_name'],
            'script_path': script_path,
            'error': data['error'],
            'attempts': data['attempts']
        })
    
    # Sort by script name for better organization
    csv_data.sort(key=lambda x: x['script_name'])
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"failed_scrapers_{timestamp}.csv"
    csv_path = os.path.join(failed_dir, filename)
    
    try:
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        print(f"\nFailed scrapers saved to: {csv_path}")
        print(f"Total failed scrapers: {len(failed_scrapers)}")
        return csv_path
    except Exception as e:
        print(f"Error saving failed scrapers CSV: {e}")
        return None


def clear_failed_scrapers():
    """
    Clears the failed scrapers tracking.
    """
    global failed_scrapers
    failed_scrapers = {}


def run_scraper(script_path):
    """Run a single scraper script."""
    script_name = os.path.basename(script_path)
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")
    
    try:
        # Change to the script's directory and run it
        script_dir = os.path.dirname(script_path)
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=script_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully")
            # Print stdout if available
            if result.stdout:
                print(result.stdout)
            return True
        else:
            error_msg = f"Return code {result.returncode}"
            if result.stderr:
                error_msg += f": {result.stderr}"
            print(f"❌ {script_name} failed with {error_msg}")
            track_failed_scraper(script_path, error_msg)
            return False
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error running {script_name}: {error_msg}")
        track_failed_scraper(script_path, error_msg)
        return False


def retry_failed_scrapers(max_retries=3):
    """
    Attempts to re-run all failed scrapers up to max_retries times.
    
    Args:
        max_retries (int): Maximum number of retry attempts (default: 3)
    
    Returns:
        bool: True if all retries completed, False if interrupted
    """
    global failed_scrapers
    
    retry_count = 0
    while retry_count < max_retries and failed_scrapers:
        retry_count += 1
        print(f"\n=== RETRY ATTEMPT {retry_count}/{max_retries} ===")
        print(f"Retrying {len(failed_scrapers)} failed scrapers...")
        
        # Save current failed scrapers to CSV
        save_failed_scrapers_csv()
        
        # Create a copy to iterate over (will be modified during iteration)
        current_failures = dict(failed_scrapers)
        clear_failed_scrapers()  # Clear for fresh tracking
        
        retry_success_count = 0
        retry_failure_count = 0
        
        for script_path, failure_data in current_failures.items():
            try:
                script_name = failure_data['script_name']
                attempts = failure_data['attempts']
                
                print(f"\nRetrying {script_name} (attempt #{attempts + 1})...")
                
                # Retry running the scraper
                success = run_scraper(script_path)
                
                if success:
                    retry_success_count += 1
                    print(f"    Retry successful: {script_name}")
                else:
                    retry_failure_count += 1
                    print(f"    Retry failed: {script_name}")
                
            except KeyboardInterrupt:
                print(f"\nRetry interrupted by user")
                return False
            except Exception as e:
                print(f"Error during retry: {e}")
                retry_failure_count += 1
        
        print(f"\nRetry attempt {retry_count} completed:")
        print(f"  Successful scrapers: {retry_success_count}")
        print(f"  Failed scrapers: {retry_failure_count}")
        print(f"  Remaining failures: {len(failed_scrapers)}")
        
        if not failed_scrapers:
            print(f"\nAll failed scrapers successfully retried!")
            break
    
    if failed_scrapers and retry_count >= max_retries:
        print(f"\nMaximum retries ({max_retries}) reached. {len(failed_scrapers)} scrapers still failing.")
        save_failed_scrapers_csv()  # Save final failures
    
    return True


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run all NFL data scrapers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python master.py                           # Run all scrapers
  python master.py --exclude games.py       # Exclude specific file
  python master.py --exclude games.py roster_details.py  # Exclude multiple files
  python master.py --list                   # List all scraper files
        """
    )
    
    parser.add_argument(
        '--exclude', 
        nargs='*', 
        default=[],
        help='List of script filenames to exclude (e.g., games.py roster_details.py)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available scraper files without running them'
    )
    
    args = parser.parse_args()
    
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(current_dir, 'scripts')
    api_dir = os.path.join(scripts_dir, 'api')
    
    # Get all scraper files
    scraper_files = get_scraper_files(scripts_dir, api_dir)
    
    if not scraper_files:
        print("No scraper files found!")
        return
    
    # Apply exclusions
    if args.exclude:
        excluded_set = set(args.exclude)
        scraper_files = [
            f for f in scraper_files 
            if os.path.basename(f) not in excluded_set
        ]
    
    # List mode
    if args.list:
        print("Available scraper files:")
        print("-" * 40)
        for script_path in scraper_files:
            rel_path = os.path.relpath(script_path, current_dir)
            print(f"  {rel_path}")
        
        if args.exclude:
            print(f"\nExcluded files: {', '.join(args.exclude)}")
        
        print(f"\nTotal: {len(scraper_files)} files")
        return
    
    # Run scrapers
    print(f"Starting master scraping session")
    print(f"Found {len(scraper_files)} scraper files to run")
    
    if args.exclude:
        print(f"Excluding: {', '.join(args.exclude)}")
    
    # Track results
    successful = []
    failed = []
    
    # Run each scraper
    for script_path in scraper_files:
        script_name = os.path.basename(script_path)
        
        if run_scraper(script_path):
            successful.append(script_name)
        else:
            failed.append(script_name)
    
    # After all scrapers are processed, attempt to retry failed scrapers
    if failed_scrapers:
        print(f"\n{'='*60}")
        print("FAILED SCRAPER RETRY PHASE")
        print(f"{'='*60}")
        print(f"Found {len(failed_scrapers)} failed scrapers")
        print("Starting automatic retry process...")
        
        try:
            retry_successful = retry_failed_scrapers(max_retries=3)
            if not retry_successful:
                print("Retry process was interrupted by user")
        except Exception as e:
            print(f"Error during retry process: {e}")
    else:
        print(f"\nNo failed scrapers detected - all scripts ran successfully!")
    
    # Final summary
    print(f"\n{'='*60}")
    print("MASTER SCRAPING SESSION SUMMARY")
    print(f"{'='*60}")
    print(f"Total scripts: {len(scraper_files)}")
    print(f"Initial successful: {len(successful)}")
    print(f"Initial failed: {len(failed)}")
    print(f"Final remaining failures: {len(failed_scrapers)}")
    
    if successful:
        print(f"\n✅ Initially successful scrapers:")
        for script in successful:
            print(f"   - {script}")
    
    if failed_scrapers:
        print(f"\n❌ Final failed scrapers (after retries):")
        for script_path, data in failed_scrapers.items():
            script_name = data['script_name']
            attempts = data['attempts']
            print(f"   - {script_name} (failed {attempts} times)")
        print(f"\nFailed scraper CSVs saved to: failed_scrapers/")
    elif failed:
        print(f"\n✅ All initially failed scrapers were successfully retried!")
    
    # Exit with appropriate code - only exit with error if there are still failures after retries
    sys.exit(0 if not failed_scrapers else 1)


if __name__ == "__main__":
    main()