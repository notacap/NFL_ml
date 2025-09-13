"""
Temporary script to scrape ONLY the defense_advanced table from NFL games.
This script is based on games.py but simplified to only handle the missing defense_advanced table.
Delete this file after recovering the missing CSV files.
"""

from scraper_utils import *
import glob
import time
import random
import signal
import sys
import gc
import psutil
import subprocess
from datetime import datetime
from selenium.webdriver.common.by import By

# Rate limiting configuration (seconds) - INCREASED due to 429 errors
DELAY_BETWEEN_GAMES = 8.0  # Delay between scraping different games (increased from 2.0)
DELAY_BETWEEN_TABLES = 3.0  # Delay between scraping tables within the same game (increased from 0.5)
USE_RANDOM_DELAYS = True  # Add randomization to delays (±50%)

# Timeout configuration (seconds)
DRIVER_TIMEOUT = 30  # Maximum time to wait for any driver operation
PAGE_LOAD_TIMEOUT = 60  # Maximum time to wait for page loads

# Global shutdown flag for graceful interruption
shutdown_requested = False

# Table configuration: ONLY defense_advanced table
DEFENSE_TABLE = {
    "gm_plyr_adv_def": "defense_advanced"
}

def get_latest_schedule_file():
    """
    Finds the most recently created schedule CSV file.
    
    Returns:
        str: Path to the latest schedule CSV file, or None if not found
    """
    schedule_dir = os.path.join(ROOT_DATA_DIR, str(YEAR), "schedule")
    
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
    week_dir = os.path.join(games_dir, f"week_{week_num}")
    
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

def kill_chrome_processes():
    """
    Force kills all Chrome processes that might be hanging.
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
        # Multiple checks to ensure driver is truly alive
        # 1. Check current URL
        url = driver.current_url
        
        # 2. Check window handles (will fail if Chrome crashed)
        handles = driver.window_handles
        
        # 3. Try a simple command that requires Chrome to respond
        driver.execute_script("return document.readyState;")
        
        return True
    except Exception as check_error:
        return False

def create_fresh_driver():
    """
    Creates a fresh WebDriver instance with proper timeouts.
    
    Returns:
        WebDriver: New driver instance
    """
    try:
        driver = setup_driver()
        driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
        driver.implicitly_wait(5)
        return driver
    except Exception as e:
        print(f"    Driver creation failed: {e}")
        return None

def scrape_defense_advanced_table(driver, url, week, year, game_id):
    """
    Scrapes the defense_advanced table.
    
    Args:
        driver: Selenium WebDriver instance
        url (str): Boxscore URL
        week (int): Week number
        year (int): Year
        game_id (int): Game ID
        
    Returns:
        pd.DataFrame: Scraped table data, or None if failed
    """
    try:
        check_shutdown()
        print(f"    Scraping defense_advanced table...")
        
        # Set page load timeout
        driver.set_page_load_timeout(DRIVER_TIMEOUT)
        driver.get(url)
        
        # Check for rate limiting
        try:
            page_title = driver.title
            if "rate limited" in page_title.lower() or "429" in page_title:
                print(f"    RATE LIMITED! Waiting 30 seconds...")
                time.sleep(30)
                raise Exception(f"Rate limited - need longer delays")
        except Exception as title_error:
            if "rate limited" in str(title_error).lower():
                raise title_error
        
        # Find the defense_advanced table
        table = driver.find_element(By.ID, "defense_advanced")
        
        if table is None:
            log_failed_table("gm_plyr_adv_def", "No table found", url)
            return None
        
        # Convert table to DataFrame
        df = table_to_dataframe(table)
        
        # Basic filtering - remove header rows
        if not df.empty:
            # Remove common header patterns
            df = df[df.iloc[:, 0] != 'Rk'].copy() if 'Rk' in df.iloc[:, 0].values else df
            df = df[df.iloc[:, 0] != 'Player'].copy() if len(df.columns) > 1 and 'Player' in df.iloc[:, 0].values else df
            df = df.dropna(how='all').copy()  # Remove completely empty rows
            
            # Add week, year, and game_id columns
            if not df.empty:
                df['week'] = week
                df['year'] = year
                df['game_id'] = game_id
        
        return df
        
    except KeyboardInterrupt:
        print(f"    Interrupted while scraping defense_advanced")
        raise
    except Exception as e:
        print(f"    Error scraping defense_advanced table: {e}")
        log_failed_table("gm_plyr_adv_def", f"Error scraping data: {e}", url)
        return None

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

def scrape_single_game_defense_advanced(row, game_id):
    """
    Scrapes ONLY the defense_advanced table for a single game.
    
    Args:
        row (pd.Series): Row from schedule DataFrame containing game info
        game_id (int): Unique identifier for this game/URL
    
    Returns:
        int: Number of tables successfully scraped (0 or 1)
    """
    week = row['Week']
    boxscore_url = row['Boxscore'] 
    home_team = row['Home']
    away_team = row['Away']
    
    print(f"\nScraping Week {week}: {away_team} @ {home_team} (defense_advanced only)")
    print(f"URL: {boxscore_url}")
    
    # Check for shutdown request
    check_shutdown()
    
    # Create week directory
    week_dir = create_week_directory(week)
    
    successful_scrapes = 0
    driver = None
    
    try:
        # Create driver instance for this game
        driver = create_fresh_driver()
        
        if driver is None:
            print(f"    Could not create driver - skipping game")
            log_failed_table("gm_plyr_adv_def", "Driver creation failed", boxscore_url)
            return 0
        
        # Scrape defense_advanced table
        df = scrape_defense_advanced_table(driver, boxscore_url, week, YEAR, game_id)
        
        if df is not None and not df.empty:
            # Generate filename and save
            filename = generate_game_filename(home_team, away_team, week, "gm_plyr_adv_def")
            csv_path = os.path.join(week_dir, filename)
            
            try:
                df.to_csv(csv_path, index=False)
                print(f"    Saved: {filename}")
                log_successful_table(filename, csv_path, boxscore_url)
                successful_scrapes = 1
            except Exception as e:
                print(f"    Error saving gm_plyr_adv_def: {e}")
                log_failed_table("gm_plyr_adv_def", f"Error saving CSV: {e}", boxscore_url)
        else:
            print(f"    No data for gm_plyr_adv_def")
            log_failed_table("gm_plyr_adv_def", "No data scraped", boxscore_url)
    
    except KeyboardInterrupt:
        print(f"\nInterrupted while scraping game: {away_team} @ {home_team}")
        raise
    except Exception as e:
        print(f"Error in scrape_single_game_defense_advanced: {e}")
        log_failed_table("gm_plyr_adv_def", f"Unexpected error: {e}", boxscore_url)
    finally:
        # Clean up driver
        if driver:
            try:
                driver.quit()
            except Exception as cleanup_error:
                print(f"    Warning: Driver cleanup failed: {cleanup_error}")
                kill_chrome_processes()
    
    return successful_scrapes

def scrape_all_games_defense_advanced():
    """
    Main function to scrape defense_advanced table for all games in the specified week range.
    
    Returns:
        dict: Summary of scraping results
    """
    # Load schedule data
    schedule_df = load_schedule_data()
    
    if schedule_df is None or schedule_df.empty:
        print("No schedule data available")
        return {"error": "No schedule data"}
    
    print(f"Starting defense_advanced table scraping for {len(schedule_df)} games")
    
    total_games = len(schedule_df)
    total_tables_scraped = 0
    games_completed = 0
    game_id = 1  # Initialize game_id counter
    
    # Scrape each game
    for index, row in schedule_df.iterrows():
        try:
            check_shutdown()  # Check before each game
            
            tables_scraped = scrape_single_game_defense_advanced(row, game_id)
            total_tables_scraped += tables_scraped
            games_completed += 1
            game_id += 1  # Increment game_id for next game/URL
            
            print(f"Game {games_completed}/{total_games} completed - {tables_scraped}/1 tables scraped")
            
            # Apply rate limiting between games (except for the last game)
            if games_completed < total_games:
                print(f"    Waiting {DELAY_BETWEEN_GAMES}s before next game...")
                apply_rate_limit(DELAY_BETWEEN_GAMES)
            
        except KeyboardInterrupt:
            print(f"\n\nScraping interrupted by user after {games_completed} games")
            break
        except Exception as e:
            print(f"Error processing game {games_completed + 1}: {e}")
    
    results = {
        "total_games": total_games,
        "games_completed": games_completed, 
        "total_tables_scraped": total_tables_scraped,
        "expected_tables": total_games,  # Only 1 table per game
        "success_rate": (total_tables_scraped / total_games * 100) if total_games > 0 else 0
    }
    
    return results

def main():
    """Main execution function."""
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start logging session
    start_scraper_session("temp_defense_advanced")
    
    try:
        print(f"NFL Defense Advanced Table Scraper (TEMPORARY)")
        print(f"Year: {YEAR}")
        print(f"Week Range: {START_WEEK} - {END_WEEK}")
        print(f"Table: defense_advanced -> gm_plyr_adv_def")
        print(f"\nTimeout Settings:")
        print(f"  Driver operation timeout: {DRIVER_TIMEOUT}s")
        print(f"  Page load timeout: {PAGE_LOAD_TIMEOUT}s")
        print(f"\nRate Limiting Settings:")
        print(f"  Delay between games: {DELAY_BETWEEN_GAMES}s")
        print(f"  Random delays: {'Enabled' if USE_RANDOM_DELAYS else 'Disabled'}")
        print(f"\nPress Ctrl+C to interrupt gracefully at any time...")
        print(f"NOTE: This is a temporary script - delete after recovering missing CSVs")
        
        # Start scraping
        results = scrape_all_games_defense_advanced()
        
        # Print summary
        if "error" not in results:
            print(f"\n=== SCRAPING SUMMARY ===")
            print(f"Games processed: {results['games_completed']}/{results['total_games']}")
            print(f"Tables scraped: {results['total_tables_scraped']}/{results['expected_tables']}")
            print(f"Success rate: {results['success_rate']:.1f}%")
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