"""
NFL Roster Details Web Scraper

Scrapes NFL roster data for all teams from pro-football-reference.com
Extracts detailed roster information and saves individual CSV files per team.
"""

from scraper_utils import *
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Team abbreviations for URL construction
TEAM_ABBREVIATIONS = [
    'buf', 'mia', 'nyj', 'nwe', 'rav', 'pit', 'cin', 'cle',
    'htx', 'clt', 'jax', 'oti', 'kan', 'sdg', 'den', 'rai',
    'phi', 'was', 'dal', 'nyg', 'det', 'min', 'gnb', 'chi',
    'tam', 'atl', 'car', 'nor', 'ram', 'sea', 'crd', 'sfo'
]

def extract_team_name(driver):
    """
    Extracts team name from the h1 element containing year, team name, and "Roster & Players".
    
    Args:
        driver: WebDriver instance
    
    Returns:
        str: Team name or empty string if not found
    """
    try:
        h1_element = driver.find_element(By.TAG_NAME, "h1")
        spans = h1_element.find_elements(By.TAG_NAME, "span")
        
        # Second span should contain the team name (index 1)
        if len(spans) >= 2:
            team_name = spans[1].text.strip()
            return team_name
        else:
            return ""
    except Exception as e:
        print(f"Error extracting team name: {e}")
        return ""

def scrape_team_roster(team_abbr, year):
    """
    Scrapes roster data for a specific team and year.
    
    Args:
        team_abbr (str): Team abbreviation
        year (int): Year to scrape
    
    Returns:
        tuple: (DataFrame, team_name) or (None, None) if failed
    """
    url = f"{BASE_URL}/teams/{team_abbr}/{year}_roster.htm"
    driver = setup_driver()
    
    try:
        driver.get(url)
        
        # Extract team name from h1 element
        team_name = extract_team_name(driver)
        
        # Wait for roster table to load
        table = wait_for_table(driver, "roster", DEFAULT_TIMEOUT)
        
        # Convert table to DataFrame
        df = table_to_dataframe(table)
        
        return df, team_name
        
    except TimeoutException:
        print(f"Timeout waiting for roster table to load: {url}")
        return None, None
    except Exception as e:
        print(f"Error scraping roster for {team_abbr}: {e}")
        return None, None
    finally:
        driver.quit()

def process_roster_data(df, team_name, team_abbr):
    """
    Processes the roster DataFrame.
    
    Args:
        df (pd.DataFrame): Raw roster DataFrame
        team_name (str): Team name
        team_abbr (str): Team abbreviation
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    if df is None or df.empty:
        return None
    
    # Filter out header rows
    filtered_df = df[df.iloc[:, 0] != 'No.'].copy()
    
    return filtered_df

def scrape_all_rosters(year):
    """
    Scrapes roster data for all teams.
    
    Args:
        year (int): Year to scrape
    
    Returns:
        dict: Dictionary mapping team abbreviations to (DataFrame, team_name) tuples
    """
    results = {}
    
    for i, team_abbr in enumerate(TEAM_ABBREVIATIONS, 1):
        print(f"Scraping roster for {team_abbr} ({i}/{len(TEAM_ABBREVIATIONS)})")
        
        df, team_name = scrape_team_roster(team_abbr, year)
        
        if df is not None and not df.empty and team_name:
            processed_df = process_roster_data(df, team_name, team_abbr)
            if processed_df is not None and not processed_df.empty:
                results[team_abbr] = (processed_df, team_name)
                print(f"Successfully scraped {len(processed_df)} players for {team_name}")
            else:
                print(f"Failed to process roster data for {team_abbr}")
        else:
            print(f"Failed to scrape roster for {team_abbr}")
        
        # Add small delay between requests to avoid rate limiting
        if i < len(TEAM_ABBREVIATIONS):
            time.sleep(1)
    
    return results

def main():
    """Main execution function."""
    print(f"Scraping NFL roster details for year: {YEAR}")
    
    # Create directory structure
    roster_dir = create_directories(YEAR, "roster_details", WEEK_NUMBER + 1)
    print(f"Data will be saved to: {roster_dir}")
    
    # Scrape all team rosters
    results = scrape_all_rosters(YEAR)
    
    # Save individual CSV files for each team
    successful_saves = 0
    for team_abbr, (df, team_name) in results.items():
        if df is not None and not df.empty:
            # Create filename: {team_name}_{year}.csv
            filename = f"{team_name.replace(' ', '_')}_{YEAR}.csv"
            csv_path = save_data_to_csv(df, roster_dir, filename)
            if csv_path:
                successful_saves += 1
    
    print(f"\nCompleted: {successful_saves}/{len(TEAM_ABBREVIATIONS)} teams successfully scraped and saved")

if __name__ == "__main__":
    main()