"""
NFL Scoring Statistics Web Scraper

Scrapes NFL scoring statistics data from pro-football-reference.com
Extracts player scoring stats including touchdowns, field goals, and points for a given season.
"""

from scraper_utils import *

def process_scoring_data(df):
    """
    Processes the scoring statistics DataFrame.
    
    Args:
        df (pd.DataFrame): Raw scoring stats DataFrame
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    if df is None or df.empty:
        return None
    
    # Remove any header rows that might have been captured
    df = df[df.iloc[:, 0] != 'Rk'].copy()  # Remove header rows with 'Rk' in first column
    df = df[df.iloc[:, 1] != 'Player'].copy()  # Remove header rows with 'Player' in second column
    
    # Reset index after filtering
    df.reset_index(drop=True, inplace=True)
    
    return df

def scrape_scoring_stats(year):
    """
    Scrapes NFL scoring statistics for a given year.
    
    Args:
        year (int): The year to scrape data for
    
    Returns:
        pd.DataFrame: Processed scoring stats DataFrame
    """
    url = f"{BASE_URL}/years/{year}/scoring.htm"
    
    print(f"Scraping scoring stats from: {url}")
    
    driver = setup_driver()
    
    try:
        driver.get(url)
        table = wait_for_table(driver, "scoring")
        
        # Convert table to DataFrame
        df = table_to_dataframe(table)
        
        # Process the data
        df = process_scoring_data(df)
        
        return df
        
    except TimeoutException:
        print(f"Timeout waiting for scoring table to load: {url}")
        return None
    except Exception as e:
        print(f"Error scraping scoring data: {e}")
        return None
    finally:
        driver.quit()

def main():
    """Main execution function."""
    print(f"Scraping NFL scoring statistics for year: {YEAR}")
    
    # Create directory structure
    scoring_dir = create_directories(YEAR, "plyr_scoring", WEEK_NUMBER)
    print(f"Data will be saved to: {scoring_dir}")
    
    # Scrape the data
    df = scrape_scoring_stats(YEAR)
    
    # Save the data
    if df is not None and not df.empty:
        print(f"Successfully scraped scoring stats for {len(df)} players")
        save_data_to_csv(df, scoring_dir, "plyr_scoring.csv")
    else:
        print("Failed to scrape scoring statistics data")

if __name__ == "__main__":
    main()