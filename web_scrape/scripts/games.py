"""
NFL Game-Level Data Web Scraper

Scrapes detailed game data from boxscore URLs found in schedule CSV files.
Creates week-based subdirectories and scrapes multiple tables per game.
"""

from scraper_utils import *
import glob
import time
import random
import signal
import sys
import threading
import gc
import psutil
import subprocess
import argparse
import json
from datetime import datetime
from selenium.webdriver.common.by import By

# Rate limiting configuration (seconds) - INCREASED due to 429 errors and driver crashes
DELAY_BETWEEN_GAMES = 15.0  # Delay between scraping different games (increased for stability)
DELAY_BETWEEN_TABLES = 8.0  # Delay between scraping tables within the same game (increased for stability)
USE_RANDOM_DELAYS = True  # Add randomization to delays (±50%)

# Timeout configuration (seconds)
DRIVER_TIMEOUT = 20  # Maximum time to wait for any driver operation (reduced to fail faster)
PAGE_LOAD_TIMEOUT = 30  # Maximum time to wait for page loads (reduced to fail faster)

# Driver refresh configuration
REFRESH_DRIVER_AFTER_GAMES = 3  # Refresh driver after this many games to prevent crashes (more frequent)
REFRESH_DRIVER_AFTER_TABLES = 10  # Force new driver after this many tables within a game

# Global shutdown flag for graceful interruption
shutdown_requested = False

# Failed scrape tracking
failed_scrapes = {}  # URL -> {'week': int, 'tables': set, 'row_data': dict}

# Checkpoint tracking for resume functionality
checkpoint_data = {
    'last_completed_game_index': -1,  # -1 means no games completed yet
    'total_games': 0,
    'total_tables_scraped': 0,
    'games_completed': 0,
    'game_id_counter': 1,
    'timestamp': None
}

# Table configuration: name for files and table_id for scraping
GAME_TABLES = {
    "gm_scoring": "scoring",
    "gm_info": "game_info", 
    "gm_expected_pts": "expected_points",
    "gm_tm_stats": "team_stats",
    "gm_plyr_def": "player_defense",
    "gm_plyr_adv_def": "defense_advanced",
    "gm_plyr_adv_passing": "passing_advanced",
    "gm_plyr_adv_rushing": "rushing_advanced", 
    "gm_plyr_adv_receiving": "receiving_advanced",
    "gm_home_starters": "home_starters",
    "gm_away_starters": "vis_starters",
    "gm_home_snap_counts": "home_snap_counts",
    "gm_away_snap_counts": "vis_snap_counts",
    "gm_home_drives": "home_drives",
    "gm_away_drives": "vis_drives",
    "gm_play_by_play": "pbp"
}

def get_latest_schedule_file():
    """
    Finds the most recently created schedule CSV file.
    
    Returns:
        str: Path to the latest schedule CSV file, or None if not found
    """
    schedule_dir = os.path.join(ROOT_DATA_DIR, str(YEAR), "schedule", f"week_{WEEK_NUMBER}")
    
    if not os.path.exists(schedule_dir):
        print(f"Schedule directory not found: {schedule_dir}")
        return None
    
    # Find all schedule CSV files
    schedule_pattern = os.path.join(schedule_dir, "schedule*.csv")
    schedule_files = glob.glob(schedule_pattern)
    
    if not schedule_files:
        print(f"No schedule files found in: {schedule_dir}")
        return None
    
    # Get the most recent file by modification time
    latest_file = max(schedule_files, key=os.path.getmtime)
    print(f"Using schedule file: {latest_file}")
    
    return latest_file

def load_schedule_data():
    """
    Loads schedule data from the latest CSV file.
    
    Returns:
        pd.DataFrame: Schedule DataFrame with filtered weeks, or None if failed
    """
    schedule_file = get_latest_schedule_file()
    
    if not schedule_file:
        return None
    
    try:
        df = pd.read_csv(schedule_file)
        
        # Convert Week column to numeric to handle filtering
        df['Week'] = pd.to_numeric(df['Week'], errors='coerce')
        
        # Filter for the specified week range
        df_filtered = df[
            (df['Week'] >= START_WEEK) & 
            (df['Week'] <= END_WEEK)
        ].copy()
        
        print(f"Loaded {len(df_filtered)} games from weeks {START_WEEK}-{END_WEEK}")
        
        # Filter out games with null/empty Boxscore URLs
        initial_count = len(df_filtered)
        df_filtered = df_filtered.dropna(subset=['Boxscore']).copy()
        df_filtered = df_filtered[df_filtered['Boxscore'].str.strip() != ''].copy()
        
        filtered_count = initial_count - len(df_filtered)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} games with missing Boxscore URLs")
        
        print(f"Final dataset: {len(df_filtered)} games with valid Boxscore URLs")
        return df_filtered
        
    except Exception as e:
        print(f"Error loading schedule data: {e}")
        return None

def create_week_directory(week_num):
    """
    Creates directory structure for a specific week.

    Args:
        week_num (int): Week number

    Returns:
        str: Path to the week directory
    """
    games_dir = os.path.join(ROOT_DATA_DIR, str(YEAR), "games")
    week_dir = os.path.join(games_dir, f"week_{week_num}.0")

    if not os.path.exists(week_dir):
        os.makedirs(week_dir, exist_ok=True)

    return week_dir

def apply_rate_limit(base_delay):
    """
    Applies rate limiting with optional randomization.
    
    Args:
        base_delay (float): Base delay in seconds
    """
    if USE_RANDOM_DELAYS:
        # Add ±50% randomization to the delay
        random_factor = random.uniform(0.5, 1.5)
        delay = base_delay * random_factor
    else:
        delay = base_delay
    
    time.sleep(delay)

def signal_handler(sig, frame):
    """
    Handle Ctrl+C interruption gracefully.
    """
    global shutdown_requested
    print(f"\n\nReceived interrupt signal. Attempting graceful shutdown...")
    
    # Save failed scrapes first (if any exist)
    if failed_scrapes:
        print("Saving failed scrapes for retry capability...")
        save_failed_scrapes_csv()
    
    # Save checkpoint for resume capability
    print("Saving checkpoint for resume capability...")
    save_checkpoint()
    
    print("Killing any hanging Chrome processes...")
    
    # Force kill Chrome processes to ensure clean exit
    kill_chrome_processes()
    
    print("Setting shutdown flag...")
    shutdown_requested = True

def check_shutdown():
    """
    Check if shutdown was requested and raise KeyboardInterrupt if so.
    """
    if shutdown_requested:
        raise KeyboardInterrupt("Shutdown requested by user")

def create_failed_scrapes_directory():
    """
    Creates the directory for failed scrape CSV files.
    
    Returns:
        str: Path to the failed scrapes directory
    """
    failed_dir = os.path.join(ROOT_DATA_DIR, str(YEAR), "games", "failed")
    if not os.path.exists(failed_dir):
        os.makedirs(failed_dir, exist_ok=True)
    return failed_dir

def track_failed_table(table_name, url, week, row_data):
    """
    Track a failed table scrape for later retry.
    
    Args:
        table_name (str): Name of the table that failed
        url (str): URL that was being scraped
        week (int): Week number
        row_data (dict): Row data from schedule DataFrame
    """
    global failed_scrapes
    
    if url not in failed_scrapes:
        failed_scrapes[url] = {
            'week': week,
            'tables': set(),
            'row_data': row_data
        }
    
    failed_scrapes[url]['tables'].add(table_name)

def save_failed_scrapes_csv():
    """
    Saves current failed scrapes to CSV file.

    Returns:
        str: Path to saved CSV file, or None if no failures
    """
    global failed_scrapes

    if not failed_scrapes:
        return None

    # Create failed scrapes directory
    failed_dir = create_failed_scrapes_directory()

    # Create CSV data
    csv_data = []
    for url, data in failed_scrapes.items():
        row_data = data.get('row_data', {})
        csv_data.append({
            'week': data['week'],
            'tables': ','.join(sorted(data['tables'])),
            'url': url,
            'home_team': row_data.get('Home', 'Unknown'),
            'away_team': row_data.get('Away', 'Unknown')
        })

    # Sort by week for better organization
    csv_data.sort(key=lambda x: x['week'])

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"failed_scrape_{timestamp}.csv"
    csv_path = os.path.join(failed_dir, filename)

    try:
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        print(f"\nFailed scrapes saved to: {csv_path}")
        print(f"Total failed games: {len(failed_scrapes)}")
        return csv_path
    except Exception as e:
        print(f"Error saving failed scrapes CSV: {e}")
        return None

def clear_failed_scrapes():
    """
    Clears the failed scrapes tracking.
    """
    global failed_scrapes
    failed_scrapes = {}

def get_checkpoint_path():
    """
    Gets the path for the checkpoint file.
    
    Returns:
        str: Path to the checkpoint JSON file
    """
    checkpoint_dir = os.path.join(ROOT_DATA_DIR, str(YEAR), "games")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, "scraping_checkpoint.json")

def save_checkpoint():
    """
    Saves current scraping progress to checkpoint file.
    """
    global checkpoint_data
    
    try:
        checkpoint_data['timestamp'] = datetime.now().isoformat()
        checkpoint_path = get_checkpoint_path()
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"    Checkpoint saved: Game {checkpoint_data['games_completed']}/{checkpoint_data['total_games']}")
        
    except Exception as e:
        print(f"    Warning: Could not save checkpoint: {e}")

def load_checkpoint():
    """
    Loads scraping progress from checkpoint file and any existing failed scrapes.
    
    Returns:
        bool: True if checkpoint was loaded successfully, False otherwise
    """
    global checkpoint_data
    
    try:
        checkpoint_path = get_checkpoint_path()
        
        if not os.path.exists(checkpoint_path):
            print("No checkpoint file found - starting from beginning")
            return False
        
        with open(checkpoint_path, 'r') as f:
            loaded_data = json.load(f)
        
        # Validate checkpoint data structure
        required_keys = ['last_completed_game_index', 'total_games', 'total_tables_scraped', 
                        'games_completed', 'game_id_counter', 'timestamp']
        
        if all(key in loaded_data for key in required_keys):
            checkpoint_data.update(loaded_data)
            print(f"Checkpoint loaded: Resuming from game {checkpoint_data['games_completed'] + 1}/{checkpoint_data['total_games']}")
            print(f"Previous session: {checkpoint_data['total_tables_scraped']} tables scraped")
            print(f"Checkpoint timestamp: {checkpoint_data['timestamp']}")
            
            # Also load any existing failed scrapes from the most recent CSV
            # This ensures that failed tables from the interrupted session are included in retries
            print("Checking for failed tables from previous session...")
            if load_failed_scrape_data():
                print(f"Loaded {len(failed_scrapes)} failed games from previous session for retry")
            else:
                print("No failed tables found from previous session")
            
            return True
        else:
            print("Invalid checkpoint file format - starting from beginning")
            return False
            
    except Exception as e:
        print(f"Error loading checkpoint: {e} - starting from beginning")
        return False

def clear_checkpoint():
    """
    Removes the checkpoint file (called when scraping completes successfully).
    """
    try:
        checkpoint_path = get_checkpoint_path()
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print("Checkpoint file cleared - scraping completed successfully")
    except Exception as e:
        print(f"Warning: Could not clear checkpoint file: {e}")

def update_checkpoint(game_index, games_completed, total_tables_scraped, game_id_counter):
    """
    Updates checkpoint data with current progress.
    
    Args:
        game_index (int): Current game index in the DataFrame
        games_completed (int): Number of games completed
        total_tables_scraped (int): Total tables scraped so far
        game_id_counter (int): Current game ID counter value
    """
    global checkpoint_data
    
    checkpoint_data['last_completed_game_index'] = game_index
    checkpoint_data['games_completed'] = games_completed
    checkpoint_data['total_tables_scraped'] = total_tables_scraped
    checkpoint_data['game_id_counter'] = game_id_counter

def get_latest_failed_scrape_file():
    """
    Finds the most recently created failed scrape CSV file.
    
    Returns:
        str: Path to the latest failed scrape CSV file, or None if not found
    """
    failed_dir = create_failed_scrapes_directory()
    
    # Find all failed scrape CSV files
    failed_pattern = os.path.join(failed_dir, "failed_scrape_*.csv")
    failed_files = glob.glob(failed_pattern)
    
    if not failed_files:
        print(f"No failed scrape files found in: {failed_dir}")
        return None
    
    # Get the most recent file by modification time
    latest_file = max(failed_files, key=os.path.getmtime)
    print(f"Using failed scrape file: {latest_file}")
    
    return latest_file

def load_failed_scrape_data(merge_with_existing=True):
    """
    Loads failed scrape data from the latest CSV file and converts it to the format
    expected by the retry system.

    Args:
        merge_with_existing (bool): If True, merge with existing failed_scrapes rather than overwriting

    Returns:
        bool: True if failed data was loaded successfully, False otherwise
    """
    global failed_scrapes

    failed_file = get_latest_failed_scrape_file()

    if not failed_file:
        return False

    try:
        df = pd.read_csv(failed_file)

        # Validate required columns
        required_columns = ['week', 'tables', 'url']
        if not all(col in df.columns for col in required_columns):
            print(f"Invalid failed scrape file format. Required columns: {required_columns}")
            return False

        print(f"Loading {len(df)} failed games from: {os.path.basename(failed_file)}")

        # Track how many were loaded/merged
        loaded_count = 0
        merged_count = 0

        # Convert CSV data back to failed_scrapes format
        for _, row in df.iterrows():
            url = row['url']
            week = row['week']
            tables_str = row['tables']

            # Parse tables string back to set
            csv_tables_set = set(tables_str.split(',')) if tables_str else set()

            # Use team names from CSV if available, otherwise use 'Unknown'
            home_team = row.get('home_team', 'Unknown')
            away_team = row.get('away_team', 'Unknown')

            # Create row data with team information
            row_data = {
                'Week': week,
                'Boxscore': url,
                'Home': home_team,
                'Away': away_team
            }

            if merge_with_existing and url in failed_scrapes:
                # Merge tables with existing entry
                existing_tables = failed_scrapes[url]['tables']
                merged_tables = existing_tables.union(csv_tables_set)
                failed_scrapes[url]['tables'] = merged_tables
                merged_count += 1
            else:
                # Create new entry
                failed_scrapes[url] = {
                    'week': week,
                    'tables': csv_tables_set,
                    'row_data': row_data
                }
                loaded_count += 1

        if merge_with_existing and merged_count > 0:
            print(f"Merged failed tables for {merged_count} existing games")
        if loaded_count > 0:
            print(f"Added {loaded_count} new failed games")

        print(f"Total failed games available for retry: {len(failed_scrapes)}")
        return True

    except Exception as e:
        print(f"Error loading failed scrape data: {e}")
        return False

def scrape_from_failed_csv():
    """
    Main function to scrape tables from failed scrape CSV file.
    
    Returns:
        dict: Summary of scraping results
    """
    print(f"=== FAILED SCRAPE MODE ===")
    print(f"Attempting to scrape tables from most recent failed scrape CSV...")
    
    # Load failed scrape data (don't merge since this is primary --failed mode)
    if not load_failed_scrape_data(merge_with_existing=False):
        return {"error": "Could not load failed scrape data"}
    
    if not failed_scrapes:
        print("No failed scrapes found to retry")
        return {"error": "No failed scrapes to retry"}
    
    print(f"Starting retry of {len(failed_scrapes)} failed games...")
    
    # Use the existing retry mechanism
    try:
        retry_successful = retry_failed_scrapes(max_retries=3)
        
        if retry_successful:
            total_attempts = len(failed_scrapes) if failed_scrapes else 0
            results = {
                "failed_games_attempted": len(failed_scrapes) if failed_scrapes else 0,
                "final_failed_games": len(failed_scrapes),
                "retry_successful": retry_successful
            }
        else:
            results = {"error": "Retry process was interrupted"}
            
    except Exception as e:
        print(f"Error during failed scrape retry: {e}")
        results = {"error": f"Retry failed: {e}"}
    
    return results

def retry_failed_scrapes(max_retries=3):
    """
    Attempts to re-scrape all failed tables up to max_retries times.
    
    Args:
        max_retries (int): Maximum number of retry attempts (default: 3)
    
    Returns:
        bool: True if all retries completed, False if interrupted
    """
    global failed_scrapes
    
    retry_count = 0
    while retry_count < max_retries and failed_scrapes:
        retry_count += 1
        print(f"\n=== RETRY ATTEMPT {retry_count}/{max_retries} ===")
        print(f"Retrying {len(failed_scrapes)} failed games...")
        
        # Save current failed scrapes to CSV
        save_failed_scrapes_csv()
        
        # Create a copy to iterate over (will be modified during iteration)
        current_failures = dict(failed_scrapes)
        clear_failed_scrapes()  # Clear for fresh tracking
        
        retry_success_count = 0
        retry_failure_count = 0
        
        for url, failure_data in current_failures.items():
            try:
                check_shutdown()
                
                week = failure_data['week']
                failed_tables = failure_data['tables']
                row_data = failure_data['row_data']
                
                print(f"\nRetrying Week {week}: {failed_tables} tables")
                print(f"URL: {url}")
                
                # Create a row-like object from stored data
                class MockRow:
                    def __init__(self, data):
                        for key, value in data.items():
                            setattr(self, key, value)
                    
                    def __getitem__(self, key):
                        return getattr(self, key)
                
                row = MockRow(row_data)
                
                # Retry scraping for this game, but only for the failed tables
                tables_scraped = scrape_single_game_retry(row, retry_count * 1000 + hash(url) % 1000, failed_tables)
                
                if tables_scraped > 0:
                    retry_success_count += 1
                    print(f"    Retry successful: {tables_scraped} tables scraped")
                else:
                    retry_failure_count += 1
                    print(f"    Retry failed: no tables scraped")
                
                # Apply rate limiting between retry attempts
                if len(current_failures) > 1:
                    apply_rate_limit(DELAY_BETWEEN_GAMES)
                
            except KeyboardInterrupt:
                print(f"\nRetry interrupted by user")
                return False
            except Exception as e:
                print(f"Error during retry: {e}")
                retry_failure_count += 1
        
        print(f"\nRetry attempt {retry_count} completed:")
        print(f"  Successful games: {retry_success_count}")
        print(f"  Failed games: {retry_failure_count}")
        print(f"  Remaining failures: {len(failed_scrapes)}")
        
        if not failed_scrapes:
            print(f"\nAll failed scrapes successfully retried!")
            break
    
    if failed_scrapes and retry_count >= max_retries:
        print(f"\nMaximum retries ({max_retries}) reached. {len(failed_scrapes)} games still have failed tables.")
        save_failed_scrapes_csv()  # Save final failures
    
    return True

def kill_chrome_processes_fast():
    """
    Force kills Chrome processes quickly without hanging.
    Uses subprocess for more reliable process killing.
    """
    try:
        print(f"    Killing Chrome processes...")
        
        # Use taskkill command for more reliable process termination on Windows
        commands = [
            ["taskkill", "/f", "/im", "chrome.exe", "/t"],
            ["taskkill", "/f", "/im", "chromedriver.exe", "/t"]
        ]
        
        for cmd in commands:
            try:
                # Run with short timeout to prevent hanging
                result = subprocess.run(cmd, capture_output=True, timeout=3, text=True)
                if result.returncode != 0 and "not found" not in result.stderr.lower():
                    print(f"    Warning: {' '.join(cmd)} returned {result.returncode}")
            except subprocess.TimeoutExpired:
                print(f"    Warning: {' '.join(cmd)} timed out")
            except Exception as e:
                print(f"    Warning: {' '.join(cmd)} failed: {e}")
        
        # Brief wait for processes to die
        time.sleep(1)
        print(f"    Process cleanup completed")
        
    except Exception as e:
        print(f"    Warning: Process cleanup failed: {e}")

def kill_chrome_processes():
    """
    Force kills all Chrome processes that might be hanging.
    Fallback method using psutil with timeout protection.
    """
    try:
        print(f"    Attempting psutil process cleanup...")
        start_time = time.time()
        killed_count = 0
        
        # Kill Chrome processes with timeout protection
        for proc in psutil.process_iter(['pid', 'name']):
            # Check timeout to prevent infinite loops
            if time.time() - start_time > 5:
                print(f"    Process cleanup timeout after 5s")
                break
                
            if proc.info['name'] and 'chrome' in proc.info['name'].lower():
                try:
                    proc.kill()
                    killed_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        
        # Kill ChromeDriver processes with timeout protection  
        for proc in psutil.process_iter(['pid', 'name']):
            # Check timeout to prevent infinite loops
            if time.time() - start_time > 10:
                print(f"    Process cleanup timeout after 10s")
                break
                
            if proc.info['name'] and 'chromedriver' in proc.info['name'].lower():
                try:
                    proc.kill()
                    killed_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        
        print(f"    Killed {killed_count} processes in {time.time() - start_time:.1f}s")
        time.sleep(1)  # Brief wait
        
    except Exception as e:
        print(f"    Warning: psutil cleanup failed: {e}")
        # Fallback to fast method
        kill_chrome_processes_fast()

def is_driver_alive(driver):
    """
    Check if the WebDriver is still alive and responsive.
    
    Args:
        driver: Selenium WebDriver instance
        
    Returns:
        bool: True if driver is alive, False otherwise
    """
    if driver is None:
        return False
        
    try:
        # Quick check with short timeout
        # Just check if we can get the current URL
        driver.current_url
        return True
    except Exception as check_error:
        # Any exception means driver is dead
        return False

def create_fresh_driver_with_timeout(timeout=30, use_minimal=False):
    """
    Creates a fresh WebDriver instance with timeout protection.
    
    Args:
        timeout (int): Maximum seconds to wait for driver creation
        use_minimal (bool): Use minimal Chrome configuration
        
    Returns:
        WebDriver: New driver instance or None if timeout
    """
    try:
        if use_minimal:
            driver = setup_minimal_driver()
        else:
            driver = setup_driver()
            
        driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
        driver.implicitly_wait(5)
        return driver
        
    except Exception as e:
        print(f"    Driver creation failed: {e}")
        return None

def create_fresh_driver():
    """
    Creates a fresh WebDriver instance with proper timeouts.
    
    Returns:
        WebDriver: New driver instance
    """
    return create_fresh_driver_with_timeout(30)

def scrape_with_simple_timeout(driver, url, table_id, timeout=DRIVER_TIMEOUT):
    """
    Scrapes a table with a simple timeout approach - just sets WebDriver timeouts.
    Does NOT kill the driver on timeout to allow reuse.
    
    Args:
        driver: Selenium WebDriver instance
        url (str): URL to scrape
        table_id (str): HTML table ID
        timeout (int): Timeout in seconds
        
    Returns:
        tuple: (WebElement or None, driver_is_dead: bool)
    """
    driver_died = False
    start_time = time.time()
    
    try:
        # Check if driver is still alive before using it
        if not is_driver_alive(driver):
            print(f"    Driver is dead - marking for recreation")
            return None, True
        
        # Only navigate if we're not already on the right page
        current_url = driver.current_url if driver.current_url else ""
        if current_url != url:
            # Set shorter timeouts on the driver itself
            driver.set_page_load_timeout(timeout)
            driver.implicitly_wait(5)
            
            # Navigate with retry logic
            retry_count = 0
            max_retries = 2
            while retry_count <= max_retries:
                try:
                    driver.get(url)
                    break  # Success
                except TimeoutException:
                    retry_count += 1
                    if retry_count > max_retries:
                        raise
                    print(f"    Page load timeout, retry {retry_count}/{max_retries}")
                    time.sleep(2)
        
        # Wait a moment for page to stabilize
        time.sleep(0.5)
        
        # Check for rate limiting
        try:
            page_title = driver.title
            page_source_snippet = driver.page_source[:500].lower()
            if ("rate limited" in page_title.lower() or "429" in page_title or 
                "rate limit" in page_source_snippet or "too many requests" in page_source_snippet):
                print(f"    RATE LIMITED! Waiting 60 seconds...")
                time.sleep(60)
                raise Exception(f"Rate limited - need longer delays")
        except Exception as title_error:
            if "rate limited" in str(title_error).lower():
                raise title_error
        
        # Use explicit wait for the table instead of just find_element
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        
        wait = WebDriverWait(driver, 10)  # 10 second wait for element
        table = wait.until(EC.presence_of_element_located((By.ID, table_id)))
        
        # Verify table has content
        if table and table.text:
            return table, False
        else:
            print(f"    Table found but empty")
            return None, False
        
    except TimeoutException as te:
        elapsed = time.time() - start_time
        print(f"    Timeout after {elapsed:.1f}s waiting for table")
        # Check if driver is still alive
        if not is_driver_alive(driver):
            driver_died = True
        return None, driver_died
        
    except Exception as e:
        elapsed = time.time() - start_time
        
        # Check if it's a timeout or driver crash
        error_str = str(e).lower()
        if ("timeout" in error_str or 
            "chrome not reachable" in error_str or
            "session" in error_str or
            "no such window" in error_str or
            "target window already closed" in error_str or
            "unknown error" in error_str or
            "gethandleverifier" in error_str or
            len(error_str.strip()) == 0):  # Empty error message usually means crash
            
            print(f"    Chrome crash/error after {elapsed:.1f}s - driver needs recreation")
            driver_died = True
        else:
            print(f"    Table scrape error: {str(e)[:100]}")
            # Still check if driver died
            if not is_driver_alive(driver):
                driver_died = True
        
        return None, driver_died

def scrape_snap_counts_table_with_driver(driver_ref, url, table_id, table_name, week, year, game_id, failure_counter=None):
    """
    Scrapes a snap counts table and adds required columns (home_team, team, week, year, game_id).
    
    Args:
        driver_ref: List containing driver instance [driver] (mutable reference)
        url (str): Boxscore URL
        table_id (str): HTML table ID to scrape ('home_snap_counts' or 'vis_snap_counts')
        table_name (str): Name for the table (for logging)
        week (int): Week number from schedule data
        year (int): Year from global configuration
        game_id (int): Unique identifier for this game/URL
        failure_counter: List containing failure count [count] (mutable reference)
    
    Returns:
        pd.DataFrame: Scraped table data with additional columns, or None if failed
    """
    try:
        check_shutdown()  # Check for user interruption
        print(f"    Scraping {table_name} table...")
        
        # Use timeout-protected scraping
        table, driver_died = scrape_with_simple_timeout(driver_ref[0], url, table_id, DRIVER_TIMEOUT)
        
        # If driver died, recreate it and try once more
        if driver_died:
            print(f"    Driver died - recreating and retrying {table_name}...")
            
            # Update failure counter if provided
            if failure_counter is not None:
                failure_counter[0] += 1
            
            # Clean up old driver
            try:
                driver_ref[0].quit()
            except Exception as quit_error:
                print(f"    Warning: Could not quit old driver: {quit_error}")
            
            # Kill any hanging Chrome processes (with timeout)
            try:
                kill_chrome_processes_fast()  # Use fast method first
            except Exception as kill_error:
                print(f"    Warning: Process cleanup failed: {kill_error}")
            
            # Create fresh driver (with timeout protection)
            fresh_driver = create_fresh_driver_with_timeout(20)  # Shorter timeout
            
            if fresh_driver is None:
                print(f"    Could not create fresh driver for {table_name}")
                log_failed_table(table_name, "Driver creation failed", url)
                driver_ref[0] = None  # Mark driver as dead
                return None
            
            driver_ref[0] = fresh_driver
            
            # Retry once with fresh driver
            try:
                table, driver_died = scrape_with_simple_timeout(driver_ref[0], url, table_id, DRIVER_TIMEOUT)
                
                if driver_died:
                    print(f"    Fresh driver also failed for {table_name}")
                    log_failed_table(table_name, "Driver recreation failed", url)
                    if failure_counter is not None:
                        failure_counter[0] += 1  # Count this as another failure
                    return None
                else:
                    # Success - reset failure counter
                    if failure_counter is not None:
                        failure_counter[0] = 0
            except Exception as retry_error:
                print(f"    Retry with fresh driver failed: {retry_error}")
                log_failed_table(table_name, f"Retry failed: {retry_error}", url)
                if failure_counter is not None:
                    failure_counter[0] += 1
                return None
        
        if table is None:
            log_failed_table(table_name, "No table found", url)
            return None
        
        # Extract team name from table caption
        team_name = None
        try:
            caption = table.find_element(By.TAG_NAME, "caption")
            caption_text = caption.text.strip()
            
            # If caption text is empty, try to get text from innerHTML
            if not caption_text:
                inner_html = caption.get_attribute('innerHTML')
                # Strip HTML tags to get plain text
                import re
                caption_text = re.sub(r'<[^>]+>', '', inner_html).strip()
            
            # Remove "Snap Counts Table" from the caption to get just the team name
            if "Snap Counts Table" in caption_text:
                team_name = caption_text.replace("Snap Counts Table", "").strip()
            else:
                team_name = caption_text.strip()
                
        except Exception as e:
            print(f"    Warning: Could not extract team name from caption: {e}")
            team_name = "Unknown"
        
        # Convert table to DataFrame
        df = table_to_dataframe(table)
        
        # Basic filtering - remove header rows
        if not df.empty:
            # Remove common header patterns
            df = df[df.iloc[:, 0] != 'Rk'].copy() if 'Rk' in df.iloc[:, 0].values else df
            df = df[df.iloc[:, 0] != 'Player'].copy() if len(df.columns) > 1 and 'Player' in df.iloc[:, 0].values else df
            df = df.dropna(how='all').copy()  # Remove completely empty rows
            
            # Add required columns
            if not df.empty:
                # home_team: 1 if home_snap_counts, 0 if vis_snap_counts
                df['home_team'] = 1 if table_id == 'home_snap_counts' else 0
                df['team'] = team_name
                df['week'] = week
                df['year'] = year
                df['game_id'] = game_id
        
        return df
        
    except KeyboardInterrupt:
        print(f"    Interrupted while scraping {table_name}")
        raise
    except Exception as e:
        print(f"    Error scraping {table_name} table: {e}")
        log_failed_table(table_name, f"Error scraping data: {e}", url)
        return None

def scrape_drives_starters_table_with_driver(driver_ref, url, table_id, table_name, week, year, game_id, failure_counter=None):
    """
    Scrapes drives and starters tables and adds required columns (home_team, team, week, year, game_id).
    
    Args:
        driver_ref: List containing driver instance [driver] (mutable reference)
        url (str): Boxscore URL
        table_id (str): HTML table ID to scrape ('home_drives', 'vis_drives', 'home_starters', 'vis_starters')
        table_name (str): Name for the table (for logging)
        week (int): Week number from schedule data
        year (int): Year from global configuration
        game_id (int): Unique identifier for this game/URL
        failure_counter: List containing failure count [count] (mutable reference)
    
    Returns:
        pd.DataFrame: Scraped table data with additional columns, or None if failed
    """
    try:
        check_shutdown()  # Check for user interruption
        print(f"    Scraping {table_name} table...")
        
        # Use timeout-protected scraping
        table, driver_died = scrape_with_simple_timeout(driver_ref[0], url, table_id, DRIVER_TIMEOUT)
        
        # If driver died, recreate it and try once more
        if driver_died:
            print(f"    Driver died - recreating and retrying {table_name}...")
            
            # Update failure counter if provided
            if failure_counter is not None:
                failure_counter[0] += 1
            
            # Clean up old driver
            try:
                driver_ref[0].quit()
            except Exception as quit_error:
                print(f"    Warning: Could not quit old driver: {quit_error}")
            
            # Kill any hanging Chrome processes (with timeout)
            try:
                kill_chrome_processes_fast()  # Use fast method first
            except Exception as kill_error:
                print(f"    Warning: Process cleanup failed: {kill_error}")
            
            # Create fresh driver (with timeout protection)
            fresh_driver = create_fresh_driver_with_timeout(20)  # Shorter timeout
            
            if fresh_driver is None:
                print(f"    Could not create fresh driver for {table_name}")
                log_failed_table(table_name, "Driver creation failed", url)
                driver_ref[0] = None  # Mark driver as dead
                return None
            
            driver_ref[0] = fresh_driver
            
            # Retry once with fresh driver
            try:
                table, driver_died = scrape_with_simple_timeout(driver_ref[0], url, table_id, DRIVER_TIMEOUT)
                
                if driver_died:
                    print(f"    Fresh driver also failed for {table_name}")
                    log_failed_table(table_name, "Driver recreation failed", url)
                    if failure_counter is not None:
                        failure_counter[0] += 1  # Count this as another failure
                    return None
                else:
                    # Success - reset failure counter
                    if failure_counter is not None:
                        failure_counter[0] = 0
            except Exception as retry_error:
                print(f"    Retry with fresh driver failed: {retry_error}")
                log_failed_table(table_name, f"Retry failed: {retry_error}", url)
                if failure_counter is not None:
                    failure_counter[0] += 1
                return None
        
        if table is None:
            log_failed_table(table_name, "No table found", url)
            return None
        
        # Extract team name from table caption
        team_name = None
        try:
            caption = table.find_element(By.TAG_NAME, "caption")
            caption_text = caption.text.strip()
            
            # If caption text is empty, try to get text from innerHTML
            if not caption_text:
                inner_html = caption.get_attribute('innerHTML')
                # Strip HTML tags to get plain text
                import re
                caption_text = re.sub(r'<[^>]+>', '', inner_html).strip()
            
            # Remove "Drives Table" or "Starters Table" from the caption to get just the team name
            if "Drives Table" in caption_text:
                team_name = caption_text.replace("Drives Table", "").strip()
            elif "Starters Table" in caption_text:
                team_name = caption_text.replace("Starters Table", "").strip()
            else:
                team_name = caption_text.strip()
                
        except Exception as e:
            print(f"    Warning: Could not extract team name from caption: {e}")
            team_name = "Unknown"
        
        # Convert table to DataFrame
        df = table_to_dataframe(table)
        
        # Basic filtering - remove header rows
        if not df.empty:
            # Remove common header patterns
            df = df[df.iloc[:, 0] != 'Rk'].copy() if 'Rk' in df.iloc[:, 0].values else df
            df = df[df.iloc[:, 0] != 'Player'].copy() if len(df.columns) > 1 and 'Player' in df.iloc[:, 0].values else df
            df = df.dropna(how='all').copy()  # Remove completely empty rows
            
            # Add required columns
            if not df.empty:
                # home_team: 1 if home table, 0 if vis table
                df['home_team'] = 1 if table_id.startswith('home_') else 0
                df['team'] = team_name
                df['week'] = week
                df['year'] = year
                df['game_id'] = game_id
        
        return df
        
    except KeyboardInterrupt:
        print(f"    Interrupted while scraping {table_name}")
        raise
    except Exception as e:
        print(f"    Error scraping {table_name} table: {e}")
        log_failed_table(table_name, f"Error scraping data: {e}", url)
        return None

def scrape_game_table_with_driver(driver_ref, url, table_id, table_name, week, year, game_id, failure_counter=None):
    """
    Scrapes a single table using an existing driver instance.
    Handles driver recreation if needed.
    
    Args:
        driver_ref: List containing driver instance [driver] (mutable reference)
        url (str): Boxscore URL
        table_id (str): HTML table ID to scrape
        table_name (str): Name for the table (for logging)
        week (int): Week number from schedule data
        year (int): Year from global configuration
        game_id (int): Unique identifier for this game/URL
        failure_counter: List containing failure count [count] (mutable reference)
    
    Returns:
        pd.DataFrame: Scraped table data with week, year, and game_id columns, or None if failed
    """
    try:
        check_shutdown()  # Check for user interruption
        print(f"    Scraping {table_name} table...")
        
        # Use timeout-protected scraping
        table, driver_died = scrape_with_simple_timeout(driver_ref[0], url, table_id, DRIVER_TIMEOUT)
        
        # If driver died, recreate it and try once more
        if driver_died:
            print(f"    Driver died - recreating and retrying {table_name}...")
            
            # Update failure counter if provided
            if failure_counter is not None:
                failure_counter[0] += 1
            
            # Clean up old driver
            try:
                driver_ref[0].quit()
            except Exception as quit_error:
                print(f"    Warning: Could not quit old driver: {quit_error}")
            
            # Kill any hanging Chrome processes (with timeout)
            try:
                kill_chrome_processes_fast()  # Use fast method first
            except Exception as kill_error:
                print(f"    Warning: Process cleanup failed: {kill_error}")
            
            # Create fresh driver (with timeout protection)
            fresh_driver = create_fresh_driver_with_timeout(20)  # Shorter timeout
            
            if fresh_driver is None:
                print(f"    Could not create fresh driver for {table_name}")
                log_failed_table(table_name, "Driver creation failed", url)
                driver_ref[0] = None  # Mark driver as dead
                return None
            
            driver_ref[0] = fresh_driver
            
            # Retry once with fresh driver
            try:
                table, driver_died = scrape_with_simple_timeout(driver_ref[0], url, table_id, DRIVER_TIMEOUT)
                
                if driver_died:
                    print(f"    Fresh driver also failed for {table_name}")
                    log_failed_table(table_name, "Driver recreation failed", url)
                    if failure_counter is not None:
                        failure_counter[0] += 1  # Count this as another failure
                    return None
                else:
                    # Success - reset failure counter
                    if failure_counter is not None:
                        failure_counter[0] = 0
            except Exception as retry_error:
                print(f"    Retry with fresh driver failed: {retry_error}")
                log_failed_table(table_name, f"Retry failed: {retry_error}", url)
                if failure_counter is not None:
                    failure_counter[0] += 1
                return None
        
        if table is None:
            log_failed_table(table_name, "No table found", url)
            return None
        
        # Convert table to DataFrame
        df = table_to_dataframe(table)
        
        # Basic filtering - remove header rows
        if not df.empty:
            # Remove common header patterns
            df = df[df.iloc[:, 0] != 'Rk'].copy() if 'Rk' in df.iloc[:, 0].values else df
            df = df[df.iloc[:, 0] != 'Player'].copy() if len(df.columns) > 1 and 'Player' in df.iloc[:, 0].values else df
            df = df.dropna(how='all').copy()  # Remove completely empty rows
            
            # Add week, year, and game_id columns to all tables
            if not df.empty:
                df['week'] = week
                df['year'] = year
                df['game_id'] = game_id
        
        return df
        
    except KeyboardInterrupt:
        print(f"    Interrupted while scraping {table_name}")
        raise
    except Exception as e:
        print(f"    Error scraping {table_name} table: {e}")
        log_failed_table(table_name, f"Error scraping data: {e}", url)
        return None

def scrape_game_table(url, table_id, table_name, week=None, year=None, game_id=1):
    """
    Scrapes a single table from a game boxscore URL (legacy wrapper).
    
    Args:
        url (str): Boxscore URL
        table_id (str): HTML table ID to scrape
        table_name (str): Name for the table (for logging)
        week (int, optional): Week number
        year (int, optional): Year
        game_id (int, optional): Game ID (defaults to 1)
    
    Returns:
        pd.DataFrame: Scraped table data, or None if failed
    """
    driver = setup_driver()
    
    try:
        # Set page load timeout
        driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
        # Use current year if not provided
        if year is None:
            year = YEAR
        return scrape_game_table_with_driver([driver], url, table_id, table_name, week, year, game_id)
    finally:
        try:
            driver.quit()
        except:
            pass

def scrape_player_offense_table_with_driver(driver_ref, url, week, year, game_id):
    """
    Scrapes the special player_offense table using existing driver and splits it into 4 separate DataFrames
    based on header data-stat attributes.
    
    Args:
        driver_ref: List containing driver instance [driver] (mutable reference)
        url (str): Boxscore URL
        week (int): Week number from schedule data
        year (int): Year from global configuration
        game_id (int): Unique identifier for this game/URL
    
    Returns:
        dict: Dictionary with keys 'gm_plyr_passing', 'gm_plyr_rushing', 'gm_plyr_receiving', 'gm_plyr_fumbles'
              Each value is a DataFrame with week, year, and game_id columns, or None if failed
    """
    try:
        print(f"    Scraping player_offense table (multi-split)...")
        
        # Use timeout-protected scraping
        table, driver_died = scrape_with_simple_timeout(driver_ref[0], url, "player_offense", DRIVER_TIMEOUT)
        
        # If driver died, recreate it and try once more
        if driver_died:
            print(f"    Driver died - recreating and retrying player_offense...")
            
            # Clean up old driver
            try:
                driver_ref[0].quit()
            except Exception as quit_error:
                print(f"    Warning: Could not quit old driver: {quit_error}")
            
            # Kill any hanging Chrome processes (with timeout)
            try:
                kill_chrome_processes_fast()  # Use fast method first
            except Exception as kill_error:
                print(f"    Warning: Process cleanup failed: {kill_error}")
            
            # Create fresh driver (with timeout protection)
            fresh_driver = create_fresh_driver_with_timeout(20)  # Shorter timeout
            
            if fresh_driver is None:
                print(f"    Could not create fresh driver for player_offense")
                log_failed_table("player_offense", "Driver creation failed", url)
                driver_ref[0] = None  # Mark driver as dead
                return {"gm_plyr_passing": None, "gm_plyr_rushing": None, "gm_plyr_receiving": None, "gm_plyr_fumbles": None}
            
            driver_ref[0] = fresh_driver
            
            # Retry once with fresh driver
            try:
                table, driver_died = scrape_with_simple_timeout(driver_ref[0], url, "player_offense", DRIVER_TIMEOUT)
                
                if driver_died:
                    print(f"    Fresh driver also failed for player_offense")
                    log_failed_table("player_offense", "Driver recreation failed", url)
                    return {"gm_plyr_passing": None, "gm_plyr_rushing": None, "gm_plyr_receiving": None, "gm_plyr_fumbles": None}
            except Exception as retry_error:
                print(f"    Retry with fresh driver failed: {retry_error}")
                log_failed_table("player_offense", f"Retry failed: {retry_error}", url)
                return {"gm_plyr_passing": None, "gm_plyr_rushing": None, "gm_plyr_receiving": None, "gm_plyr_fumbles": None}
        
        if table is None:
            log_failed_table("player_offense", "No table found", url)
            return {"gm_plyr_passing": None, "gm_plyr_rushing": None, "gm_plyr_receiving": None, "gm_plyr_fumbles": None}
        
        # Find the over_header row to identify column ranges
        header_mapping = {}
        try:
            over_header_row = table.find_element(By.CSS_SELECTOR, "thead tr.over_header")
            th_elements = over_header_row.find_elements(By.TAG_NAME, "th")
            
            current_col = 0
            for th in th_elements:
                data_stat = th.get_attribute("data-stat")
                colspan = int(th.get_attribute("colspan") or 1)
                
                if data_stat == "header_pass":
                    header_mapping["gm_plyr_passing"] = (current_col, current_col + colspan)
                elif data_stat == "header_rush":
                    header_mapping["gm_plyr_rushing"] = (current_col, current_col + colspan)
                elif data_stat == "header_rec":
                    header_mapping["gm_plyr_receiving"] = (current_col, current_col + colspan)
                elif data_stat == "header_fumbles":
                    header_mapping["gm_plyr_fumbles"] = (current_col, current_col + colspan)
                
                current_col += colspan
                
        except Exception as e:
            print(f"    Error parsing header structure: {e}")
            return {"gm_plyr_passing": None, "gm_plyr_rushing": None, "gm_plyr_receiving": None, "gm_plyr_fumbles": None}
        
        # Convert full table to DataFrame
        df = table_to_dataframe(table)
        
        if df.empty:
            return {"gm_plyr_passing": None, "gm_plyr_rushing": None, "gm_plyr_receiving": None, "gm_plyr_fumbles": None}
        
        # Basic filtering
        df = df[df.iloc[:, 0] != 'Rk'].copy() if 'Rk' in df.iloc[:, 0].values else df
        df = df[df.iloc[:, 0] != 'Player'].copy() if len(df.columns) > 1 and 'Player' in df.iloc[:, 0].values else df
        df = df.dropna(how='all').copy()
        
        # Find team column index by looking for data-stat="team" in the table header
        team_col_idx = None
        try:
            # Look for the second <tr> element in thead (the one without class)
            thead = table.find_element(By.TAG_NAME, "thead")
            tr_elements = thead.find_elements(By.TAG_NAME, "tr")
            
            if len(tr_elements) >= 2:
                # Get the second <tr> element (index 1)
                second_tr = tr_elements[1]
                th_elements = second_tr.find_elements(By.TAG_NAME, "th")
                
                # Find th with data-stat="team"
                for idx, th in enumerate(th_elements):
                    data_stat = th.get_attribute("data-stat")
                    if data_stat == "team":
                        team_col_idx = idx
                        print(f"    Found team column at index {idx}")
                        break
            
            if team_col_idx is None:
                print(f"    Warning: Could not find team column with data-stat='team'")
                
        except Exception as e:
            print(f"    Warning: Could not find team column: {e}")
            # Fallback to old TM column search
            try:
                for idx, col in enumerate(df.columns):
                    if isinstance(col, tuple):
                        # Handle multi-level columns
                        if any('TM' in str(level) for level in col):
                            team_col_idx = idx
                            break
                    elif 'TM' in str(col):
                        team_col_idx = idx
                        break
            except Exception as fallback_e:
                print(f"    Warning: Fallback TM column search also failed: {fallback_e}")
        
        # Split into separate DataFrames based on column ranges
        result = {}
        for table_name, (start_col, end_col) in header_mapping.items():
            try:
                # Collect columns to include: player name (0), TM column (if found), and stat columns
                columns_to_include = []
                
                if start_col == 0:
                    # If the stat section starts at column 0, take the range but also add team if not included
                    columns_to_include = list(range(start_col, end_col))
                    if team_col_idx is not None and team_col_idx not in columns_to_include:
                        columns_to_include.append(team_col_idx)
                else:
                    # Include player name column (0), team column (if found), and stat columns
                    columns_to_include = [0]  # Player name column
                    if team_col_idx is not None:
                        columns_to_include.append(team_col_idx)  # Team column
                    columns_to_include.extend(range(start_col, end_col))  # Stat columns
                
                # Remove duplicates and sort
                columns_to_include = sorted(list(set(columns_to_include)))
                
                # Create subset DataFrame
                subset_df = df.iloc[:, columns_to_include].copy()
                
                # Add week, year, and game_id columns to each split table
                if subset_df is not None and not subset_df.empty:
                    subset_df['week'] = week
                    subset_df['year'] = year
                    subset_df['game_id'] = game_id
                
                result[table_name] = subset_df if not subset_df.empty else None
            except Exception as e:
                print(f"    Error creating {table_name} subset: {e}")
                result[table_name] = None
        
        return result
        
    except KeyboardInterrupt:
        print(f"    Interrupted while scraping player_offense")
        raise
    except Exception as e:
        print(f"    Error scraping player_offense table: {e}")
        log_failed_table("player_offense", f"Error scraping data: {e}", url)
        return {"gm_plyr_passing": None, "gm_plyr_rushing": None, "gm_plyr_receiving": None, "gm_plyr_fumbles": None}

def scrape_player_offense_table(url, week=None, year=None, game_id=1):
    """
    Scrapes the special player_offense table (legacy wrapper).
    
    Args:
        url (str): Boxscore URL
        week (int, optional): Week number
        year (int, optional): Year
        game_id (int, optional): Game ID (defaults to 1)
    
    Returns:
        dict: Dictionary with keys 'gm_plyr_passing', 'gm_plyr_rushing', 'gm_plyr_receiving', 'gm_plyr_fumbles'
    """
    driver = setup_driver()
    
    try:
        driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
        # Use current year if not provided
        if year is None:
            year = YEAR
        return scrape_player_offense_table_with_driver([driver], url, week, year, game_id)
    finally:
        try:
            driver.quit()
        except:
            pass

def generate_game_filename(home_team, away_team, week, table_name):
    """
    Generates filename for game data CSV.
    
    Args:
        home_team (str): Home team name
        away_team (str): Away team name  
        week (int): Week number
        table_name (str): Table name identifier
    
    Returns:
        str: Formatted filename
    """
    # Clean team names (replace spaces with underscores)
    home_clean = home_team.replace(" ", "_")
    away_clean = away_team.replace(" ", "_")
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"{home_clean}_{away_clean}_wk{week}_{YEAR}_{table_name}_{timestamp}.csv"
    return filename

def scrape_single_game(row, game_id):
    """
    Scrapes all tables for a single game using a shared driver instance.
    
    Args:
        row (pd.Series): Row from schedule DataFrame containing game info
        game_id (int): Unique identifier for this game/URL
    
    Returns:
        int: Number of tables successfully scraped
    """
    week = row['Week']
    boxscore_url = row['Boxscore'] 
    home_team = row['Home']
    away_team = row['Away']
    
    # Store row data for potential retry tracking
    row_data = dict(row)
    
    print(f"\nScraping Week {week}: {away_team} @ {home_team}")
    print(f"URL: {boxscore_url}")
    
    # Check for shutdown request
    check_shutdown()
    
    # Create week directory
    week_dir = create_week_directory(week)
    
    successful_scrapes = 0
    driver = None
    driver_recreation_failures = 0  # Track consecutive driver failures
    MAX_DRIVER_FAILURES = 3  # Circuit breaker threshold
    tables_since_driver_refresh = 0  # Track tables scraped with current driver
    
    try:
        # Create single driver instance for this game - use list for mutable reference
        driver = create_fresh_driver()
        driver_ref = [driver]  # Mutable reference for driver recreation
        failure_counter = [driver_recreation_failures]  # Mutable reference for failure tracking
        
        # Scrape regular tables
        table_count = 0
        for table_name, table_id in GAME_TABLES.items():
            check_shutdown()  # Check before each table
            
            # Force driver refresh if we've scraped too many tables with same driver
            if tables_since_driver_refresh >= REFRESH_DRIVER_AFTER_TABLES:
                print(f"    Proactive driver refresh after {tables_since_driver_refresh} tables")
                try:
                    if driver_ref[0]:
                        driver_ref[0].quit()
                except:
                    pass
                
                kill_chrome_processes_fast()
                time.sleep(2)  # Give system time to clean up
                
                fresh_driver = create_fresh_driver_with_timeout(20)
                if fresh_driver:
                    driver_ref[0] = fresh_driver
                    tables_since_driver_refresh = 0
                    failure_counter[0] = 0  # Reset failure counter too
                    print(f"    Driver refreshed successfully")
                else:
                    print(f"    Warning: Could not refresh driver")
            
            # Skip if driver is dead and couldn't be recreated
            if driver_ref[0] is None:
                print(f"    Skipping {table_name} - no working driver available")
                log_failed_table(table_name, "No working driver available", boxscore_url)
                track_failed_table(table_name, boxscore_url, week, row_data)
                continue
            
            # Proactive health check before scraping
            if not is_driver_alive(driver_ref[0]):
                print(f"    Driver appears dead before {table_name} - attempting recreation...")
                
                # Check circuit breaker
                if driver_recreation_failures >= MAX_DRIVER_FAILURES:
                    print(f"    Circuit breaker triggered - too many driver failures ({driver_recreation_failures})")
                    print(f"    Skipping remaining tables for this game to prevent infinite loop")
                    log_failed_table(table_name, "Circuit breaker - too many driver failures", boxscore_url)
                    track_failed_table(table_name, boxscore_url, week, row_data)
                    driver_ref[0] = None
                    break  # Skip all remaining tables for this game
                
                # Try to recreate driver
                try:
                    driver_ref[0].quit()
                except:
                    pass
                
                kill_chrome_processes_fast()
                
                # Try minimal config if we've had recent failures
                use_minimal = driver_recreation_failures >= 1
                fresh_driver = create_fresh_driver_with_timeout(20, use_minimal)
                
                # If minimal config also fails, try standard config once
                if fresh_driver is None and use_minimal:
                    print(f"    Minimal config failed, trying standard config...")
                    fresh_driver = create_fresh_driver_with_timeout(15, use_minimal=False)
                
                if fresh_driver is None:
                    driver_recreation_failures += 1
                    print(f"    Could not recreate driver ({driver_recreation_failures}/{MAX_DRIVER_FAILURES}) - skipping {table_name}")
                    log_failed_table(table_name, "Driver recreation failed", boxscore_url)
                    track_failed_table(table_name, boxscore_url, week, row_data)
                    driver_ref[0] = None
                    continue
                
                driver_ref[0] = fresh_driver
                driver_recreation_failures = 0  # Reset counter on success
                print(f"    Driver recreated successfully")
            
            # Use specialized functions for different table types
            if table_id in ['home_snap_counts', 'vis_snap_counts']:
                df = scrape_snap_counts_table_with_driver(driver_ref, boxscore_url, table_id, table_name, week, YEAR, game_id, failure_counter)
            elif table_id in ['home_drives', 'vis_drives', 'home_starters', 'vis_starters']:
                df = scrape_drives_starters_table_with_driver(driver_ref, boxscore_url, table_id, table_name, week, YEAR, game_id, failure_counter)
            else:
                df = scrape_game_table_with_driver(driver_ref, boxscore_url, table_id, table_name, week, YEAR, game_id, failure_counter)
            
            # Update local failure counter from mutable reference
            driver_recreation_failures = failure_counter[0]
            
            if df is not None and not df.empty:
                # Generate filename and save
                filename = generate_game_filename(home_team, away_team, week, table_name)
                csv_path = os.path.join(week_dir, filename)
                
                try:
                    df.to_csv(csv_path, index=False)
                    print(f"    Saved: {filename}")
                    log_successful_table(filename, csv_path, boxscore_url)
                    successful_scrapes += 1
                    tables_since_driver_refresh += 1  # Increment counter
                except Exception as e:
                    print(f"    Error saving {table_name}: {e}")
                    log_failed_table(filename, f"Error saving CSV: {e}", boxscore_url)
                    track_failed_table(table_name, boxscore_url, week, row_data)
            else:
                print(f"    No data for {table_name}")
                log_failed_table(table_name, "No data scraped", boxscore_url)
                track_failed_table(table_name, boxscore_url, week, row_data)
                tables_since_driver_refresh += 1  # Count failed attempts too
            
            # Apply rate limiting between tables (except for the last table)
            table_count += 1
            if table_count < len(GAME_TABLES):
                apply_rate_limit(DELAY_BETWEEN_TABLES)
        
        # Add delay before scraping special player_offense table
        print(f"    Preparing to scrape player offense tables...")
        apply_rate_limit(DELAY_BETWEEN_TABLES)  # Always add delay before player offense section
        
        # Scrape special player_offense table (creates 4 separate files)
        check_shutdown()
        
        # Skip if driver is dead and couldn't be recreated
        if driver_ref[0] is None:
            print(f"    Skipping player_offense - no working driver available")
            log_failed_table("player_offense", "No working driver available", boxscore_url)
            for offense_table in ["gm_plyr_passing", "gm_plyr_rushing", "gm_plyr_receiving", "gm_plyr_fumbles"]:
                track_failed_table(offense_table, boxscore_url, week, row_data)
            offense_tables = {"gm_plyr_passing": None, "gm_plyr_rushing": None, "gm_plyr_receiving": None, "gm_plyr_fumbles": None}
        else:
            # Proactive health check before player_offense scraping
            if not is_driver_alive(driver_ref[0]):
                print(f"    Driver appears dead before player_offense - attempting recreation...")
                
                # Try to recreate driver
                try:
                    driver_ref[0].quit()
                except:
                    pass
                
                kill_chrome_processes_fast()
                fresh_driver = create_fresh_driver_with_timeout(20)
                
                if fresh_driver is None:
                    print(f"    Could not recreate driver - skipping player_offense")
                    log_failed_table("player_offense", "Driver recreation failed", boxscore_url)
                    for offense_table in ["gm_plyr_passing", "gm_plyr_rushing", "gm_plyr_receiving", "gm_plyr_fumbles"]:
                        track_failed_table(offense_table, boxscore_url, week, row_data)
                    driver_ref[0] = None
                    offense_tables = {"gm_plyr_passing": None, "gm_plyr_rushing": None, "gm_plyr_receiving": None, "gm_plyr_fumbles": None}
                else:
                    driver_ref[0] = fresh_driver
                    print(f"    Driver recreated successfully")
                    offense_tables = scrape_player_offense_table_with_driver(driver_ref, boxscore_url, week, YEAR, game_id)
            else:
                offense_tables = scrape_player_offense_table_with_driver(driver_ref, boxscore_url, week, YEAR, game_id)
        
        offense_table_count = 0
        for table_name, df in offense_tables.items():
            if df is not None and not df.empty:
                # Generate filename and save
                filename = generate_game_filename(home_team, away_team, week, table_name)
                csv_path = os.path.join(week_dir, filename)
                
                try:
                    df.to_csv(csv_path, index=False)
                    print(f"    Saved: {filename}")
                    log_successful_table(filename, csv_path, boxscore_url)
                    successful_scrapes += 1
                    tables_since_driver_refresh += 1  # Increment counter for offense tables too
                except Exception as e:
                    print(f"    Error saving {table_name}: {e}")
                    log_failed_table(filename, f"Error saving CSV: {e}", boxscore_url)
                    track_failed_table(table_name, boxscore_url, week, row_data)
            else:
                print(f"    No data for {table_name}")
                log_failed_table(table_name, "No data scraped", boxscore_url)
                track_failed_table(table_name, boxscore_url, week, row_data)
                tables_since_driver_refresh += 1  # Count failed attempts too
            
            # Apply rate limiting between ALL offense tables (including after the last one to prevent issues with next game)
            offense_table_count += 1
            apply_rate_limit(DELAY_BETWEEN_TABLES)
    
    except KeyboardInterrupt:
        print(f"\nInterrupted while scraping game: {away_team} @ {home_team}")
        raise
    except Exception as e:
        print(f"Error in scrape_single_game: {e}")
        # If it's a timeout error, do aggressive cleanup
        if "timeout" in str(e).lower():
            print(f"    Detected timeout error - performing aggressive cleanup")
            kill_chrome_processes()
    finally:
        # Clean up driver
        if 'driver_ref' in locals() and driver_ref and driver_ref[0]:
            try:
                driver_ref[0].quit()
            except Exception as cleanup_error:
                print(f"    Warning: Driver cleanup failed: {cleanup_error}")
                # If normal cleanup fails, force kill processes
                kill_chrome_processes()
        elif driver:
            try:
                driver.quit()
            except Exception as cleanup_error:
                print(f"    Warning: Driver cleanup failed: {cleanup_error}")
                kill_chrome_processes()
    
    return successful_scrapes

def scrape_single_game_retry(row, game_id, failed_tables_to_retry):
    """
    Retries scraping only the specified failed tables for a single game.
    
    Args:
        row (pd.Series or MockRow): Row from schedule DataFrame containing game info
        game_id (int): Unique identifier for this game/URL
        failed_tables_to_retry (set): Set of table names to retry
    
    Returns:
        int: Number of tables successfully scraped
    """
    week = row['Week']
    boxscore_url = row['Boxscore'] 
    home_team = row['Home']
    away_team = row['Away']
    
    print(f"    Retrying {len(failed_tables_to_retry)} failed tables for {away_team} @ {home_team}")
    
    # Check for shutdown request
    check_shutdown()
    
    # Create week directory
    week_dir = create_week_directory(week)
    
    successful_scrapes = 0
    driver = None
    driver_recreation_failures = 0
    MAX_DRIVER_FAILURES = 3
    
    try:
        # Create single driver instance for this retry - use list for mutable reference
        driver = create_fresh_driver()
        driver_ref = [driver]
        failure_counter = [driver_recreation_failures]
        
        # Retry regular tables (only those that failed)
        regular_tables_to_retry = {table_name: table_id for table_name, table_id in GAME_TABLES.items() 
                                  if table_name in failed_tables_to_retry}
        
        for table_name, table_id in regular_tables_to_retry.items():
            check_shutdown()
            
            # Skip if driver is dead and couldn't be recreated
            if driver_ref[0] is None:
                print(f"      Skipping {table_name} - no working driver available")
                track_failed_table(table_name, boxscore_url, week, row.__dict__ if hasattr(row, '__dict__') else dict(row))
                continue
            
            # Use specialized functions for different table types
            if table_id in ['home_snap_counts', 'vis_snap_counts']:
                df = scrape_snap_counts_table_with_driver(driver_ref, boxscore_url, table_id, table_name, week, YEAR, game_id, failure_counter)
            elif table_id in ['home_drives', 'vis_drives', 'home_starters', 'vis_starters']:
                df = scrape_drives_starters_table_with_driver(driver_ref, boxscore_url, table_id, table_name, week, YEAR, game_id, failure_counter)
            else:
                df = scrape_game_table_with_driver(driver_ref, boxscore_url, table_id, table_name, week, YEAR, game_id, failure_counter)
            
            if df is not None and not df.empty:
                # Generate filename and save
                filename = generate_game_filename(home_team, away_team, week, table_name)
                csv_path = os.path.join(week_dir, filename)
                
                try:
                    df.to_csv(csv_path, index=False)
                    print(f"      Retry success: {filename}")
                    log_successful_table(filename, csv_path, boxscore_url)
                    successful_scrapes += 1
                except Exception as e:
                    print(f"      Error saving {table_name}: {e}")
                    track_failed_table(table_name, boxscore_url, week, row.__dict__ if hasattr(row, '__dict__') else dict(row))
            else:
                print(f"      Retry failed: {table_name}")
                track_failed_table(table_name, boxscore_url, week, row.__dict__ if hasattr(row, '__dict__') else dict(row))
            
            # Apply rate limiting between ALL tables during retry
            apply_rate_limit(DELAY_BETWEEN_TABLES)
        
        # Retry player offense tables if any of them failed
        offense_tables_to_retry = {'gm_plyr_passing', 'gm_plyr_rushing', 'gm_plyr_receiving', 'gm_plyr_fumbles'}
        offense_failed = offense_tables_to_retry.intersection(failed_tables_to_retry)
        
        if offense_failed and driver_ref[0] is not None:
            print(f"      Retrying player_offense tables: {offense_failed}")
            apply_rate_limit(DELAY_BETWEEN_TABLES)
            
            offense_tables = scrape_player_offense_table_with_driver(driver_ref, boxscore_url, week, YEAR, game_id)
            
            for table_name in offense_failed:
                df = offense_tables.get(table_name)
                if df is not None and not df.empty:
                    # Generate filename and save
                    filename = generate_game_filename(home_team, away_team, week, table_name)
                    csv_path = os.path.join(week_dir, filename)
                    
                    try:
                        df.to_csv(csv_path, index=False)
                        print(f"      Retry success: {filename}")
                        log_successful_table(filename, csv_path, boxscore_url)
                        successful_scrapes += 1
                    except Exception as e:
                        print(f"      Error saving {table_name}: {e}")
                        track_failed_table(table_name, boxscore_url, week, row.__dict__ if hasattr(row, '__dict__') else dict(row))
                else:
                    print(f"      Retry failed: {table_name}")
                    track_failed_table(table_name, boxscore_url, week, row.__dict__ if hasattr(row, '__dict__') else dict(row))
                
                # Apply rate limiting after each offense table retry
                apply_rate_limit(DELAY_BETWEEN_TABLES)
    
    except KeyboardInterrupt:
        print(f"      Interrupted while retrying game: {away_team} @ {home_team}")
        raise
    except Exception as e:
        print(f"      Error in retry: {e}")
    finally:
        # Clean up driver
        if 'driver_ref' in locals() and driver_ref and driver_ref[0]:
            try:
                driver_ref[0].quit()
            except Exception as cleanup_error:
                print(f"      Warning: Driver cleanup failed: {cleanup_error}")
                kill_chrome_processes()
        elif driver:
            try:
                driver.quit()
            except Exception as cleanup_error:
                print(f"      Warning: Driver cleanup failed: {cleanup_error}")
                kill_chrome_processes()
    
    return successful_scrapes

def scrape_all_games(resume_from_checkpoint=False):
    """
    Main function to scrape all games in the specified week range with driver refresh.
    
    Args:
        resume_from_checkpoint (bool): Whether to resume from saved checkpoint
    
    Returns:
        dict: Summary of scraping results
    """
    global checkpoint_data
    # Load schedule data
    schedule_df = load_schedule_data()
    
    if schedule_df is None or schedule_df.empty:
        print("No schedule data available")
        return {"error": "No schedule data"}
    
    total_games = len(schedule_df)
    total_tables_scraped = 0
    games_completed = 0
    game_id = 1  # Initialize game_id counter
    start_index = 0  # Start from beginning by default
    
    # Initialize checkpoint data for this session
    checkpoint_data['total_games'] = total_games
    
    # Handle resume from checkpoint
    if resume_from_checkpoint:
        if load_checkpoint():
            # Resume from where we left off
            start_index = checkpoint_data['last_completed_game_index'] + 1
            games_completed = checkpoint_data['games_completed']
            total_tables_scraped = checkpoint_data['total_tables_scraped']
            game_id = checkpoint_data['game_id_counter']
            
            if start_index >= total_games:
                print("All games already completed according to checkpoint!")
                clear_checkpoint()
                return {
                    "total_games": total_games,
                    "games_completed": games_completed,
                    "total_tables_scraped": total_tables_scraped,
                    "expected_tables": total_games * (len(GAME_TABLES) + 4),
                    "final_failed_games": len(failed_scrapes)
                }
            
            print(f"Resuming scraping: {total_games - start_index} games remaining")
        else:
            print("Failed to load checkpoint - starting from beginning")
    else:
        print(f"Starting game data scraping for {total_games} games")
    
    print(f"Driver will refresh every {REFRESH_DRIVER_AFTER_GAMES} games for stability")
    
    # Scrape each game (starting from checkpoint if resuming)
    for index, row in schedule_df.iloc[start_index:].iterrows():
        try:
            check_shutdown()  # Check before each game
            
            # Check if we need to trigger garbage collection
            if games_completed > 0 and games_completed % REFRESH_DRIVER_AFTER_GAMES == 0:
                print(f"\n=== DRIVER REFRESH ({games_completed} games completed) ===")
                print("Running garbage collection to free memory...")
                gc.collect()
                time.sleep(1)  # Brief pause to allow cleanup
            
            tables_scraped = scrape_single_game(row, game_id)
            total_tables_scraped += tables_scraped
            games_completed += 1
            game_id += 1  # Increment game_id for next game/URL
            
            # Update checkpoint after each completed game
            update_checkpoint(index, games_completed, total_tables_scraped, game_id)
            
            print(f"Game {games_completed}/{total_games} completed - {tables_scraped}/{len(GAME_TABLES)+4} tables scraped")
            
            # Apply rate limiting between games (except for the last game)
            if games_completed < total_games:
                print(f"    Waiting {DELAY_BETWEEN_GAMES}s before next game...")
                apply_rate_limit(DELAY_BETWEEN_GAMES)
            
        except KeyboardInterrupt:
            print(f"\n\nScraping interrupted by user after {games_completed} games")
            print("Progress has been saved to checkpoint file")
            break
        except Exception as e:
            print(f"Error processing game {games_completed + 1}: {e}")
    
    # After all games are processed, attempt to retry failed scrapes
    if failed_scrapes:
        print(f"\n=== FAILED SCRAPE RETRY PHASE ===")
        print(f"Found {len(failed_scrapes)} games with failed table scrapes")
        print("Starting automatic retry process...")
        
        try:
            retry_successful = retry_failed_scrapes(max_retries=3)
            if not retry_successful:
                print("Retry process was interrupted by user")
        except Exception as e:
            print(f"Error during retry process: {e}")
    else:
        print(f"\nNo failed scrapes detected - all tables successfully scraped!")
    
    # Clear checkpoint if scraping completed successfully
    if games_completed >= total_games:
        clear_checkpoint()
    
    results = {
        "total_games": total_games,
        "games_completed": games_completed, 
        "total_tables_scraped": total_tables_scraped,
        "expected_tables": total_games * (len(GAME_TABLES) + 4),  # +4 for player_offense split tables
        "final_failed_games": len(failed_scrapes)
    }
    
    return results

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='NFL Game Data Scraper')
    parser.add_argument('--continue', dest='continue_flag', action='store_true',
                       help='Resume scraping from the last saved checkpoint')
    parser.add_argument('--failed', dest='failed_flag', action='store_true',
                       help='Retry scraping tables from the most recent failed scrape CSV file')
    args = parser.parse_args()
    
    # Validate arguments
    if args.continue_flag and args.failed_flag:
        print("Error: Cannot use --continue and --failed flags together")
        print("Use --continue to resume normal scraping or --failed to retry failed tables")
        return
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start logging session
    start_scraper_session("games")
    
    try:
        print(f"NFL Game Data Scraper")
        print(f"Year: {YEAR}")
        
        # Handle failed scrape mode
        if args.failed_flag:
            print(f"\n*** FAILED SCRAPE RETRY MODE ENABLED ***")
            print(f"Will retry tables from most recent failed scrape CSV")
            print(f"Rate limiting will be applied between all retry attempts")
            print(f"\nPress Ctrl+C to interrupt gracefully at any time...")
            
            # Run failed scrape retry
            results = scrape_from_failed_csv()
        else:
            # Normal scraping mode
            print(f"Week Range: {START_WEEK} - {END_WEEK}")
            print(f"Regular tables per game: {len(GAME_TABLES)}")
            print(f"Special player_offense tables: 4 (passing, rushing, receiving, fumbles)")
            print(f"Total tables per game: {len(GAME_TABLES) + 4}")
            
            # Show resume mode if applicable
            if args.continue_flag:
                print(f"\n*** RESUME MODE ENABLED ***")
                print(f"Will attempt to resume from last checkpoint")
            
            print(f"\nTimeout Settings:")
            print(f"  Driver operation timeout: {DRIVER_TIMEOUT}s")
            print(f"  Page load timeout: {PAGE_LOAD_TIMEOUT}s")
            print(f"\nStability Settings:")
            print(f"  Driver refresh interval: Every {REFRESH_DRIVER_AFTER_GAMES} games")
            print(f"  Memory cleanup: Garbage collection enabled")
            print(f"\nRate Limiting Settings (Enhanced for Stability):")
            print(f"  Delay between games: {DELAY_BETWEEN_GAMES}s")
            print(f"  Delay between tables: {DELAY_BETWEEN_TABLES}s (applied after EVERY table)")
            print(f"  Random delays: {'Enabled' if USE_RANDOM_DELAYS else 'Disabled'} (±50% variation)")
            print(f"  Total requests per game: {len(GAME_TABLES) + 4} tables + 1 page load")
            print(f"\nCheckpoint System:")
            print(f"  Auto-save: Progress saved after each completed game")
            print(f"  Resume: Use --continue flag to resume from checkpoint")
            print(f"  Location: scraping_checkpoint.json in games directory")
            print(f"\nFailed Scrape Retry:")
            print(f"  CSV files saved to: failed/ directory")
            print(f"  Use --failed flag to retry from most recent CSV")
            print(f"\nPress Ctrl+C to interrupt gracefully at any time...")
            
            # Start scraping (with resume capability)
            results = scrape_all_games(resume_from_checkpoint=args.continue_flag)
        
        # Print summary
        if "error" not in results:
            if args.failed_flag:
                # Failed scrape retry mode summary
                print(f"\n=== FAILED SCRAPE RETRY SUMMARY ===")
                print(f"Failed games attempted: {results.get('failed_games_attempted', 0)}")
                print(f"Final failed games remaining: {results.get('final_failed_games', 0)}")
                if results.get('final_failed_games', 0) > 0:
                    print(f"Some tables still failed - check failed/ directory for updated CSV")
                else:
                    print(f"All failed tables successfully scraped!")
            else:
                # Normal scraping mode summary
                print(f"\n=== SCRAPING SUMMARY ===")
                print(f"Games processed: {results['games_completed']}/{results['total_games']}")
                print(f"Tables scraped: {results['total_tables_scraped']}/{results['expected_tables']}")
                if results['expected_tables'] > 0:
                    print(f"Success rate: {results['total_tables_scraped']/results['expected_tables']*100:.1f}%")
                if results.get('final_failed_games', 0) > 0:
                    print(f"Games with remaining failed tables: {results['final_failed_games']}")
                    print(f"Failed scrape CSVs saved to: {ROOT_DATA_DIR}\\{YEAR}\\games\\failed\\")
                    print(f"Use --failed flag to retry failed tables")
                else:
                    print(f"All tables successfully scraped!")
        else:
            print(f"Scraping failed: {results['error']}")
    
    except KeyboardInterrupt:
        print(f"\n\nScript interrupted by user. Logging session and exiting...")
    except Exception as e:
        print(f"\n\nUnexpected error in main: {e}")
    finally:
        # End logging session
        end_scraper_session()

if __name__ == "__main__":
    main()