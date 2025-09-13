"""
NFL Scrimmage Statistics Web Scraper

Scrapes NFL scrimmage statistics data from pro-football-reference.com
Extracts combined rushing and receiving stats for a given season.
"""

from scraper_utils import *

def process_scrimmage_data(df):
    """
    Processes the scrimmage statistics DataFrame.
    
    Args:
        df (pd.DataFrame): Raw scrimmage stats DataFrame
    
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

def scrape_scrimmage_stats(year):
    """
    Scrapes NFL scrimmage statistics for a given year.
    
    Args:
        year (int): The year to scrape data for
    
    Returns:
        pd.DataFrame: Processed scrimmage stats DataFrame
    """
    url = f"{BASE_URL}/years/{year}/scrimmage.htm"
    
    print(f"Scraping scrimmage stats from: {url}")
    
    driver = setup_driver()
    
    try:
        driver.get(url)
        table = wait_for_table(driver, "scrimmage")
        
        # Convert table to DataFrame
        df = table_to_dataframe(table)
        
        # Process the data
        df = process_scrimmage_data(df)
        
        return df
        
    except TimeoutException:
        print(f"Timeout waiting for scrimmage table to load: {url}")
        return None
    except Exception as e:
        print(f"Error scraping scrimmage data: {e}")
        return None
    finally:
        driver.quit()

def main():
    """Main execution function."""
    print(f"Scraping NFL scrimmage statistics for year: {YEAR}")
    
    # Create directory structure
    scrimmage_dir = create_directories(YEAR, "plyr_scrimmage", WEEK_NUMBER)
    print(f"Data will be saved to: {scrimmage_dir}")
    
    # Scrape the data
    df = scrape_scrimmage_stats(YEAR)
    
    # Save the data
    if df is not None and not df.empty:
        print(f"Successfully scraped scrimmage stats for {len(df)} players")
        save_data_to_csv(df, scrimmage_dir, "plyr_scrimmage.csv")
    else:
        print("Failed to scrape scrimmage statistics data")

if __name__ == "__main__":
    main()