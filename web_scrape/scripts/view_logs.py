"""
View Scraper Logs

Standalone utility to view failed table scrapes and session summaries from plain text logs.
"""

import os
import sys
from datetime import datetime, timedelta
from scraper_utils import SCRAPER_LOG_FILE

def show_recent_failures(hours=24):
    """Shows failed tables from recent scraper sessions."""
    if not os.path.exists(SCRAPER_LOG_FILE):
        print("No log file found.")
        return
    
    cutoff_time = datetime.now() - timedelta(hours=hours)
    failures = []
    
    try:
        with open(SCRAPER_LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or 'FAILED:' not in line:
                    continue
                
                # Parse timestamp
                if line.startswith('[') and ']' in line:
                    timestamp_str = line[1:line.index(']')]
                    try:
                        log_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                        if log_time >= cutoff_time:
                            failures.append(line)
                    except ValueError:
                        continue
    
    except FileNotFoundError:
        print("No log file found.")
        return
    
    if not failures:
        print(f"No failed tables in the last {hours} hours.")
        return
    
    print(f"\n=== FAILED TABLES (Last {hours} hours) ===")
    for failure in failures:
        print(failure)
    print("-" * 70)

def show_recent_successes(hours=24):
    """Shows successful tables from recent scraper sessions."""
    if not os.path.exists(SCRAPER_LOG_FILE):
        print("No log file found.")
        return
    
    cutoff_time = datetime.now() - timedelta(hours=hours)
    successes = []
    
    try:
        with open(SCRAPER_LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or 'SUCCESS:' not in line:
                    continue
                
                # Parse timestamp
                if line.startswith('[') and ']' in line:
                    timestamp_str = line[1:line.index(']')]
                    try:
                        log_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                        if log_time >= cutoff_time:
                            successes.append(line)
                    except ValueError:
                        continue
    
    except FileNotFoundError:
        print("No log file found.")
        return
    
    if not successes:
        print(f"No successful tables in the last {hours} hours.")
        return
    
    print(f"\n=== SUCCESSFUL TABLES (Last {hours} hours) ===")
    for success in successes:
        print(success)
    print("-" * 70)

def show_all_sessions():
    """Shows all scraper sessions from the log file."""
    if not os.path.exists(SCRAPER_LOG_FILE):
        print("No log file found.")
        return
    
    sessions = []
    
    try:
        with open(SCRAPER_LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if 'STARTED:' in line or 'COMPLETED:' in line:
                    sessions.append(line)
    
    except FileNotFoundError:
        print("No log file found.")
        return
    
    if not sessions:
        print("No sessions found.")
        return
    
    print("=== ALL SCRAPER SESSIONS ===")
    for session in sessions:
        print(session)
    print("-" * 70)

def show_tail(lines=50):
    """Shows the last N lines of the log file."""
    if not os.path.exists(SCRAPER_LOG_FILE):
        print("No log file found.")
        return
    
    try:
        with open(SCRAPER_LOG_FILE, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            
        if not all_lines:
            print("Log file is empty.")
            return
        
        tail_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        print(f"\n=== LAST {len(tail_lines)} LOG ENTRIES ===")
        for line in tail_lines:
            print(line.rstrip())
        print("-" * 70)
    
    except FileNotFoundError:
        print("No log file found.")

def show_usage():
    """Shows usage information."""
    print("NFL Scraper Log Viewer")
    print("Usage: python view_logs.py [option]")
    print("")
    print("Options:")
    print("  failures [hours]  Show failed tables from last N hours (default: 24)")
    print("  success [hours]   Show successful tables from last N hours (default: 24)")
    print("  sessions          Show all scraper session start/end entries")
    print("  tail [lines]      Show last N log entries (default: 50)")
    print("  help              Show this help message")
    print("")
    print("Examples:")
    print("  python view_logs.py failures     # Show failures from last 24 hours")
    print("  python view_logs.py failures 6   # Show failures from last 6 hours")
    print("  python view_logs.py success       # Show successes from last 24 hours")
    print("  python view_logs.py sessions      # Show all sessions")
    print("  python view_logs.py tail 100      # Show last 100 log entries")

def main():
    """Main function."""
    if len(sys.argv) == 1:
        show_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == "help":
        show_usage()
    elif command == "failures":
        hours = 24
        if len(sys.argv) > 2:
            try:
                hours = int(sys.argv[2])
            except ValueError:
                print("Invalid hours value. Using default of 24.")
        show_recent_failures(hours)
    elif command == "success":
        hours = 24
        if len(sys.argv) > 2:
            try:
                hours = int(sys.argv[2])
            except ValueError:
                print("Invalid hours value. Using default of 24.")
        show_recent_successes(hours)
    elif command == "sessions":
        show_all_sessions()
    elif command == "tail":
        lines = 50
        if len(sys.argv) > 2:
            try:
                lines = int(sys.argv[2])
            except ValueError:
                print("Invalid lines value. Using default of 50.")
        show_tail(lines)
    else:
        print(f"Unknown command: {command}")
        show_usage()

if __name__ == "__main__":
    main()