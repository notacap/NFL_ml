#!/usr/bin/env python3
"""
Database Master Script

Orchestrates the execution of insert scripts in the correct order,
with support for exclusions and dependency management.
"""

import os
import sys
import subprocess
import glob
from datetime import datetime
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Scripts to exclude from execution
EXCLUSIONS = [
    'nfl_season.py',
    'nfl_team.py',
    'nfl_game_weather.py',
    'nfl_gm_quarter.py',
    'nfl_week.py'
]

# Scripts that require interactive user input
INTERACTIVE_SCRIPTS = [
    'plyr.py',
    'plyr_def.py',
    'plyr_gm_fmbl.py',
    'plyr_gm_pass.py'
]

# Priority scripts that must run first, in order
PRIORITY_SCRIPTS = [
    'plyr.py',
    'multi_tm_plyr.py',
    'nfl_game.py'
]

def log_message(level: str, message: str):
    """Print formatted log messages"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {level}: {message}")

def run_script(script_path: str) -> Tuple[bool, str]:
    """
    Execute a Python script and return success status and output.

    Args:
        script_path: Path to the Python script to execute

    Returns:
        Tuple of (success: bool, output: str)
    """
    script_name = os.path.basename(script_path)
    log_message("INFO", f"Running {script_name}...")

    # Check if this script requires interactive input
    is_interactive = script_name in INTERACTIVE_SCRIPTS

    try:
        if is_interactive:
            # For interactive scripts, don't capture output so user prompts work
            log_message("INFO", f"{script_name} requires user interaction - running in interactive mode")
            result = subprocess.run(
                [sys.executable, script_path],
                timeout=600  # 10 minute timeout per script
            )

            if result.returncode == 0:
                log_message("SUCCESS", f"{script_name} completed successfully")
                return True, "Script completed successfully (interactive mode)"
            else:
                log_message("ERROR", f"{script_name} failed with return code {result.returncode}")
                return False, f"Script failed with return code {result.returncode}"
        else:
            # For non-interactive scripts, capture output for logging
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout per script
            )

            if result.returncode == 0:
                log_message("SUCCESS", f"{script_name} completed successfully")
                return True, result.stdout
            else:
                log_message("ERROR", f"{script_name} failed with return code {result.returncode}")
                if result.stderr:
                    print(f"Error output: {result.stderr}")
                return False, result.stderr

    except subprocess.TimeoutExpired:
        log_message("ERROR", f"{script_name} timed out after 10 minutes")
        return False, "Script execution timed out"
    except Exception as e:
        log_message("ERROR", f"Failed to run {script_name}: {e}")
        return False, str(e)

def get_insert_scripts() -> List[str]:
    """
    Get all Python scripts from the insert_scripts directory,
    excluding those in the exclusions list.
    
    Returns:
        List of script paths
    """
    insert_scripts_dir = os.path.join(os.path.dirname(__file__), 'insert_scripts')
    
    if not os.path.exists(insert_scripts_dir):
        log_message("ERROR", f"Insert scripts directory not found: {insert_scripts_dir}")
        return []
    
    # Get all Python files
    all_scripts = glob.glob(os.path.join(insert_scripts_dir, '*.py'))
    
    # Filter out excluded scripts
    filtered_scripts = []
    for script_path in all_scripts:
        script_name = os.path.basename(script_path)
        if script_name not in EXCLUSIONS:
            filtered_scripts.append(script_path)
    
    return filtered_scripts

def organize_scripts(scripts: List[str]) -> List[str]:
    """
    Organize scripts with priority scripts first, then remaining scripts.
    
    Args:
        scripts: List of all script paths to organize
        
    Returns:
        Ordered list of script paths
    """
    insert_scripts_dir = os.path.join(os.path.dirname(__file__), 'insert_scripts')
    ordered_scripts = []
    remaining_scripts = scripts.copy()
    
    # Add priority scripts in order
    for priority_script in PRIORITY_SCRIPTS:
        priority_path = os.path.join(insert_scripts_dir, priority_script)
        if priority_path in remaining_scripts:
            ordered_scripts.append(priority_path)
            remaining_scripts.remove(priority_path)
        else:
            log_message("WARNING", f"Priority script not found: {priority_script}")
    
    # Add remaining scripts
    ordered_scripts.extend(sorted(remaining_scripts))
    
    return ordered_scripts

def main():
    """Main execution function"""
    log_message("INFO", "=" * 60)
    log_message("INFO", "Database Master Script Starting")
    log_message("INFO", "=" * 60)
    
    # Get all insert scripts
    scripts = get_insert_scripts()
    
    if not scripts:
        log_message("ERROR", "No scripts found to execute")
        return 1
    
    # Organize scripts with priority order
    ordered_scripts = organize_scripts(scripts)
    
    log_message("INFO", f"Found {len(ordered_scripts)} scripts to execute")
    log_message("INFO", f"Excluded {len(EXCLUSIONS)} scripts")
    
    # Display execution plan
    log_message("INFO", "\nExecution Plan:")
    for i, script_path in enumerate(ordered_scripts, 1):
        script_name = os.path.basename(script_path)
        tags = []
        if script_name in PRIORITY_SCRIPTS:
            tags.append("PRIORITY")
        if script_name in INTERACTIVE_SCRIPTS:
            tags.append("INTERACTIVE")

        if tags:
            log_message("INFO", f"  {i}. {script_name} [{', '.join(tags)}]")
        else:
            log_message("INFO", f"  {i}. {script_name}")
    
    print("\n" + "=" * 60)
    
    # Execute scripts
    successful_scripts = []
    failed_scripts = []
    
    for script_path in ordered_scripts:
        script_name = os.path.basename(script_path)
        print("\n" + "-" * 40)
        
        success, output = run_script(script_path)
        
        if success:
            successful_scripts.append(script_name)
        else:
            failed_scripts.append(script_name)
            
            # Ask if user wants to continue after failure
            response = input(f"\n{script_name} failed. Continue with remaining scripts? (y/n): ")
            if response.lower() != 'y':
                log_message("INFO", "Execution stopped by user")
                break
    
    # Print summary
    print("\n" + "=" * 60)
    log_message("INFO", "EXECUTION SUMMARY")
    log_message("INFO", "=" * 60)
    log_message("INFO", f"Total scripts executed: {len(successful_scripts) + len(failed_scripts)}")
    log_message("INFO", f"Successful: {len(successful_scripts)}")
    log_message("INFO", f"Failed: {len(failed_scripts)}")
    
    if successful_scripts:
        print("\nSuccessful scripts:")
        for script in successful_scripts:
            print(f"  ✓ {script}")
    
    if failed_scripts:
        print("\nFailed scripts:")
        for script in failed_scripts:
            print(f"  ✗ {script}")
    
    print("\n" + "=" * 60)
    
    # Return exit code based on failures
    return 0 if not failed_scripts else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log_message("INFO", "\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        log_message("ERROR", f"Unexpected error: {e}")
        sys.exit(1)