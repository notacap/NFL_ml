import os
import sys
import subprocess
from typing import List, Dict
import time

def run_script(script_path: str, script_name: str) -> bool:
    """
    Run a Python script and return success status.
    
    Args:
        script_path: Full path to the script
        script_name: Display name for the script
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"Path: {script_path}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout:
            print(result.stdout)
        
        print(f"✓ Successfully completed: {script_name}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {script_name}")
        print(f"Error code: {e.returncode}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error running {script_name}: {str(e)}")
        return False

def main():
    """
    Master script to run all data cleaning scripts in order.
    """
    
    # Get the base directory (clean_data)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define scripts to run in order
    # Format: (subdirectory, script_name)
    scripts_to_run = [
        ('csv_format', 'consolidate_headers.py'),
        ('csv_format', 'consolidate_season_headers.py'),
        ('players', 'clean_team_rosters.py'),
        ('players', 'merge_weekly_snap_count_and_rosters.py'),
        ('players', 'identify_multi_tm_plyr.py'),
    ]
    
    # EXCLUSIONS LIST - Add script names here to exclude them from execution
    # Example: exclusions = ['consolidate_headers.py', 'clean_team_rosters.py']
    exclusions = []
    
    # Track execution results
    results = []
    start_time = time.time()
    
    print("\n" + "#"*60)
    print("#" + " "*18 + "MASTER SCRIPT RUNNER" + " "*18 + "#")
    print("#"*60)
    print(f"\nBase directory: {base_dir}")
    print(f"Total scripts to run: {len(scripts_to_run)}")
    
    if exclusions:
        print(f"\nExcluded scripts:")
        for excluded in exclusions:
            print(f"  - {excluded}")
    
    # Run each script
    for subdirectory, script_name in scripts_to_run:
        # Check if script is excluded
        if script_name in exclusions:
            print(f"\n⊘ Skipping (excluded): {script_name}")
            results.append((script_name, 'SKIPPED'))
            continue
        
        # Build full path
        script_path = os.path.join(base_dir, subdirectory, script_name)
        
        # Check if script exists
        if not os.path.exists(script_path):
            print(f"\n✗ Script not found: {script_path}")
            results.append((script_name, 'NOT FOUND'))
            continue
        
        # Run the script
        display_name = f"{subdirectory}/{script_name}"
        success = run_script(script_path, display_name)
        
        if success:
            results.append((script_name, 'SUCCESS'))
        else:
            results.append((script_name, 'FAILED'))
            # Ask if user wants to continue after failure
            response = input("\nContinue with remaining scripts? (y/n): ")
            if response.lower() != 'y':
                print("\nStopping execution...")
                break
    
    # Print summary
    elapsed_time = time.time() - start_time
    print("\n" + "#"*60)
    print("#" + " "*22 + "SUMMARY" + " "*23 + "#")
    print("#"*60)
    
    success_count = sum(1 for _, status in results if status == 'SUCCESS')
    failed_count = sum(1 for _, status in results if status == 'FAILED')
    skipped_count = sum(1 for _, status in results if status == 'SKIPPED')
    not_found_count = sum(1 for _, status in results if status == 'NOT FOUND')
    
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
    print(f"\nResults:")
    print(f"  ✓ Successful: {success_count}")
    print(f"  ✗ Failed: {failed_count}")
    print(f"  ⊘ Skipped: {skipped_count}")
    print(f"  ? Not Found: {not_found_count}")
    
    print(f"\nDetailed results:")
    for script_name, status in results:
        status_symbol = {
            'SUCCESS': '✓',
            'FAILED': '✗',
            'SKIPPED': '⊘',
            'NOT FOUND': '?'
        }.get(status, ' ')
        print(f"  {status_symbol} {script_name}: {status}")
    
    # Return exit code based on results
    if failed_count > 0 or not_found_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()