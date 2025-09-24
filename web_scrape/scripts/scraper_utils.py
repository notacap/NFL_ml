"""
Shared utilities for NFL web scraping scripts.
Contains common functions, configurations, and constants.
"""

import pandas as pd
import os
import warnings
import logging
import json
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from io import StringIO

# Suppress warnings
warnings.filterwarnings("ignore")

# Common constants
BASE_URL = "https://www.pro-football-reference.com"
ROOT_DATA_DIR = r"C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data"
DEFAULT_TIMEOUT = 20

# =============================================================================
# CONFIGURATION: Change this year to scrape data for different seasons
# =============================================================================
YEAR = 2025

# =============================================================================
# WEEK NUMBER CONFIGURATION: Set the week number for directory structure
# =============================================================================
WEEK_NUMBER = 1

# =============================================================================
# WEEK RANGE CONFIGURATION: Set the range of weeks for game-level scraping
# =============================================================================
START_WEEK = 1
END_WEEK = 1

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_DIR = os.path.join(ROOT_DATA_DIR, "logs")
SCRAPER_LOG_FILE = os.path.join(LOG_DIR, "scraper.log")

# Global variable to track current scraper session
_current_session = {
    "session_id": None,
    "start_time": None,
    "scraper_name": None,
    "failed_count": 0,
    "successful_count": 0
}

def setup_driver():
    """
    Creates and configures a Chrome WebDriver with comprehensive error suppression.
    
    Returns:
        webdriver.Chrome: Configured Chrome WebDriver instance
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-logging")
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--disable-features=VizDisplayCompositor")
    chrome_options.add_argument("--disable-gpu-sandbox")
    chrome_options.add_argument("--disable-software-rasterizer")
    chrome_options.add_argument("--disable-background-timer-throttling")
    chrome_options.add_argument("--disable-renderer-backgrounding")
    chrome_options.add_argument("--disable-backgrounding-occluded-windows")
    chrome_options.add_argument("--disable-ipc-flooding-protection")
    chrome_options.add_argument("--mute-audio")
    chrome_options.add_argument("--disable-client-side-phishing-detection")
    chrome_options.add_argument("--disable-default-apps")
    chrome_options.add_argument("--disable-hang-monitor")
    chrome_options.add_argument("--disable-prompt-on-repost")
    chrome_options.add_argument("--disable-sync")
    chrome_options.add_argument("--disable-translate")
    chrome_options.add_argument("--disable-webgl")
    chrome_options.add_argument("--disable-webgl2")
    
    # Conservative stability options to prevent crashes
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-domain-reliability")
    chrome_options.add_argument("--disable-component-update")
    chrome_options.add_argument("--disable-crash-reporter")
    chrome_options.add_argument("--disable-background-networking")
    chrome_options.add_argument("--memory-pressure-off")
    
    # CRITICAL: Memory and connection pool management
    chrome_options.add_argument("--max_old_space_size=4096")  # Increase JS heap size
    chrome_options.add_argument("--memory-model=low")  # Use low memory model
    chrome_options.add_argument("--max-connections-per-host=6")  # Limit connections
    chrome_options.add_argument("--aggressive-cache-discard")  # Aggressively discard cache
    chrome_options.add_argument("--disable-features=CalculateNativeWinOcclusion")  # Reduce GPU overhead on Windows
    
    # Page loading optimization
    chrome_options.add_argument("--disable-images")  # Don't load images
    chrome_options.add_argument("--disable-javascript")  # Disable JS if tables are static HTML
    chrome_options.page_load_strategy = 'eager'  # Don't wait for all resources
    
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging', 'enable-automation'])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # Prefs to reduce memory usage
    prefs = {
        "profile.default_content_setting_values": {
            "images": 2,  # Block images
            "plugins": 2,  # Block plugins
            "popups": 2,  # Block popups
            "geolocation": 2,  # Block location
            "notifications": 2,  # Block notifications
            "media_stream": 2,  # Block media stream
        },
        "profile.managed_default_content_settings": {
            "images": 2
        }
    }
    chrome_options.add_experimental_option("prefs", prefs)
    
    service = Service(ChromeDriverManager().install())
    service.log_path = os.devnull
    # Suppress service logs
    service.creation_flags = 0x08000000  # CREATE_NO_WINDOW flag for Windows
    
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Set aggressive timeouts to prevent hanging
    driver.set_page_load_timeout(30)
    driver.implicitly_wait(5)
    
    return driver

def setup_minimal_driver():
    """
    Creates a minimal Chrome WebDriver configuration for when standard setup fails.
    Uses fewer flags to avoid conflicts that might cause crashes.
    
    Returns:
        webdriver.Chrome: Configured Chrome WebDriver instance
    """
    print("    Trying minimal Chrome configuration...")
    chrome_options = Options()
    
    # Only essential options to reduce crash potential
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-logging")
    chrome_options.add_argument("--log-level=3")
    
    # Minimal stability options
    chrome_options.add_argument("--disable-crash-reporter")
    chrome_options.add_argument("--memory-pressure-off")
    chrome_options.add_argument("--disable-images")  # Don't load images
    chrome_options.add_argument("--max-connections-per-host=2")  # Very limited connections
    
    # Use eager page load strategy
    chrome_options.page_load_strategy = 'eager'
    
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    service = Service(ChromeDriverManager().install())
    service.log_path = os.devnull
    service.creation_flags = 0x08000000  # CREATE_NO_WINDOW flag for Windows
    
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Set timeouts
    driver.set_page_load_timeout(20)
    driver.implicitly_wait(3)
    
    return driver

def create_directories(year, data_type, week_number=None):
    """
    Creates directory structure for storing scraped data.
    
    Args:
        year (int): The year for the data
        data_type (str): Type of data (e.g., 'schedule', 'stats', etc.)
        week_number (int, optional): Week number for creating week subdirectory
    
    Returns:
        str: Path to the created directory
    """
    year_dir = os.path.join(ROOT_DATA_DIR, str(year))
    data_dir = os.path.join(year_dir, data_type)
    
    # Add week subdirectory if week_number is provided
    if week_number is not None:
        data_dir = os.path.join(data_dir, f"week_{week_number}")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    
    return data_dir

def wait_for_table(driver, table_id, timeout=DEFAULT_TIMEOUT):
    """
    Waits for a table element to be present and returns it.
    
    Args:
        driver: WebDriver instance
        table_id (str): ID of the table element
        timeout (int): Maximum time to wait
    
    Returns:
        WebElement: The table element
    
    Raises:
        TimeoutException: If table is not found within timeout
    """
    wait = WebDriverWait(driver, timeout)
    return wait.until(EC.presence_of_element_located((By.ID, table_id)))

def filter_table_html(table_html):
    """
    Removes rows with header classes from table HTML.
    
    Args:
        table_html (str): Raw HTML string of the table
    
    Returns:
        str: Filtered HTML string
    """
    import re
    # Remove tr elements that contain class="over_header thead"
    pattern1 = r'<tr[^>]*class="[^"]*over_header[^"]*thead[^"]*"[^>]*>.*?</tr>'
    filtered_html = re.sub(pattern1, '', table_html, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove tr elements that contain class="thead"
    pattern2 = r'<tr[^>]*class="[^"]*thead[^"]*"[^>]*>.*?</tr>'
    filtered_html = re.sub(pattern2, '', filtered_html, flags=re.DOTALL | re.IGNORECASE)
    
    return filtered_html

def table_to_dataframe(table):
    """
    Converts a table WebElement to a pandas DataFrame.
    
    Args:
        table: WebElement representing the table
    
    Returns:
        pd.DataFrame: DataFrame containing table data
    """
    table_html = table.get_attribute('outerHTML')
    # Filter out over_header thead rows before pandas processing
    filtered_html = filter_table_html(table_html)
    df = pd.read_html(StringIO(filtered_html))[0]
    return df

def extract_links_from_table(table, data_stat_attr):
    """
    Extracts href links from table cells with specific data-stat attribute.
    
    Args:
        table: WebElement representing the table
        data_stat_attr (str): The data-stat attribute value to look for
    
    Returns:
        list: List of extracted URLs
    """
    links = []
    rows = table.find_elements(By.TAG_NAME, "tr")
    
    for i, row in enumerate(rows):
        if i == 0:  # Skip header row
            continue
        
        # Skip header rows (thead class and over_header thead class)
        row_class = row.get_attribute("class") or ""
        if "thead" in row_class or "over_header" in row_class:
            continue
            
        cells = row.find_elements(By.TAG_NAME, "td")
        
        # Skip if no td cells (likely a header row)
        if len(cells) == 0:
            continue
        
        # Find the cell with specified data-stat attribute
        link_url = ""
        for cell in cells:
            if cell.get_attribute("data-stat") == data_stat_attr:
                try:
                    link_element = cell.find_element(By.TAG_NAME, "a")
                    link_url = link_element.get_attribute("href")
                except:
                    pass
                break
        
        links.append(link_url)
    
    return links

def filter_dataframe(df, exclude_conditions=None):
    """
    Filters DataFrame based on exclusion conditions.
    
    Args:
        df (pd.DataFrame): DataFrame to filter
        exclude_conditions (list): List of conditions to exclude
    
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    filtered_df = df.copy()
    
    # Default exclusion: header rows
    filtered_df = filtered_df[filtered_df.iloc[:, 0] != 'Week'].copy()
    
    if exclude_conditions:
        for condition in exclude_conditions:
            column_idx = condition.get('column', 0)
            contains_text = condition.get('contains')
            if contains_text:
                filtered_df = filtered_df[
                    ~filtered_df.iloc[:, column_idx].astype(str).str.contains(contains_text, na=False)
                ].copy()
    
    return filtered_df

def setup_logging():
    """
    Sets up logging directory and creates log file if it doesn't exist.
    """
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)

def write_log_entry(message):
    """
    Writes a log entry to the scraper log file.
    
    Args:
        message (str): Message to log
    """
    setup_logging()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    
    with open(SCRAPER_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_entry)

def start_scraper_session(scraper_name):
    """
    Starts a new scraper session for logging.
    
    Args:
        scraper_name (str): Name of the scraper being run
    """
    global _current_session
    
    session_id = f"{scraper_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _current_session = {
        "session_id": session_id,
        "start_time": datetime.now(),
        "scraper_name": scraper_name,
        "failed_count": 0,
        "successful_count": 0
    }
    
    write_log_entry(f"STARTED: {session_id}")
    print(f"Started logging session: {session_id}")

def log_failed_table(table_name, error_reason, url=None):
    """
    Logs a failed table scrape.
    
    Args:
        table_name (str): Name/identifier of the table that failed
        error_reason (str): Reason why the table failed
        url (str, optional): URL that was being scraped
    """
    global _current_session
    
    if _current_session["session_id"] is None:
        return
    
    _current_session["failed_count"] += 1
    
    url_part = f" | URL: {url}" if url else ""
    log_message = f"FAILED: {_current_session['scraper_name']} | {table_name} | {error_reason}{url_part}"
    write_log_entry(log_message)
    print(f"FAILED: {table_name} - {error_reason}")

def log_successful_table(table_name, csv_path, url=None):
    """
    Logs a successful table scrape.
    
    Args:
        table_name (str): Name/identifier of the table that succeeded
        csv_path (str): Path to the saved CSV file
        url (str, optional): URL that was scraped
    """
    global _current_session
    
    if _current_session["session_id"] is None:
        return
    
    _current_session["successful_count"] += 1
    
    url_part = f" | URL: {url}" if url else ""
    log_message = f"SUCCESS: {_current_session['scraper_name']} | {table_name} | {csv_path}{url_part}"
    write_log_entry(log_message)

def end_scraper_session():
    """
    Ends the current scraper session and logs summary.
    """
    global _current_session
    
    if _current_session["session_id"] is None:
        return
    
    # Calculate duration
    end_time = datetime.now()
    duration = end_time - _current_session["start_time"]
    duration_str = str(duration).split('.')[0]  # Remove microseconds
    
    # Log session end
    session_summary = f"COMPLETED: {_current_session['session_id']} | Duration: {duration_str} | Success: {_current_session['successful_count']} | Failed: {_current_session['failed_count']}"
    write_log_entry(session_summary)
    
    # Print session summary
    print(f"\n=== SCRAPER SESSION SUMMARY ===")
    print(f"Session: {_current_session['session_id']}")
    print(f"Duration: {duration_str}")
    print(f"Successful tables: {_current_session['successful_count']}")
    print(f"Failed tables: {_current_session['failed_count']}")
    print(f"Log saved to: {SCRAPER_LOG_FILE}")
    print("=" * 35)
    
    # Reset session
    _current_session = {
        "session_id": None,
        "start_time": None,
        "scraper_name": None,
        "failed_count": 0,
        "successful_count": 0
    }

def show_recent_failures(hours=24):
    """
    Shows failed tables from recent scraper sessions.
    
    Args:
        hours (int): Number of hours to look back (default: 24)
    """
    setup_logging()
    
    if not os.path.exists(SCRAPER_LOG_FILE):
        print("No log file found.")
        return
    
    from datetime import timedelta
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
    print("-" * 50)

def save_data_to_csv(df, directory, filename):
    """
    Saves DataFrame to CSV file with automatic timestamp appending.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        directory (str): Directory to save the file
        filename (str): Name of the CSV file (timestamp will be automatically added)
    
    Returns:
        str or None: Path to saved file, or None if failed
    """
    if df is not None and not df.empty:
        try:
            # Add timestamp to filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name, ext = os.path.splitext(filename)
            timestamped_filename = f"{name}_{timestamp}{ext}"
            
            csv_path = os.path.join(directory, timestamped_filename)
            df.to_csv(csv_path, index=False)
            print(f"Data saved to: {csv_path}")
            log_successful_table(filename, csv_path)
            return csv_path
        except Exception as e:
            print(f"Error saving CSV file: {e}")
            log_failed_table(filename, f"Error saving CSV: {e}")
            return None
    else:
        print("No data to save")
        log_failed_table(filename or "unknown_table", "No data scraped")
        return None

def scrape_table_with_links(url, table_id, data_stat_attr, exclude_conditions=None, timeout=DEFAULT_TIMEOUT):
    """
    Generic function to scrape a table and extract links from specified cells.
    
    Args:
        url (str): URL to scrape
        table_id (str): ID of the table element
        data_stat_attr (str): data-stat attribute to extract links from
        exclude_conditions (list): Conditions to exclude rows
        timeout (int): Timeout for waiting
    
    Returns:
        tuple: (DataFrame, list of links) or (None, None) if failed
    """
    driver = setup_driver()
    
    try:
        driver.get(url)
        table = wait_for_table(driver, table_id, timeout)
        
        # Convert table to DataFrame
        df = table_to_dataframe(table)
        
        # Extract links
        links = extract_links_from_table(table, data_stat_attr)
        
        # Filter DataFrame
        df = filter_dataframe(df, exclude_conditions)
        
        return df, links
        
    except TimeoutException:
        print(f"Timeout waiting for table to load: {url}")
        log_failed_table(table_id, f"Timeout waiting for table to load", url)
        return None, None
    except Exception as e:
        print(f"Error scraping data: {e}")
        log_failed_table(table_id, f"Error scraping data: {e}", url)
        return None, None
    finally:
        driver.quit()