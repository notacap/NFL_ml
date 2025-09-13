"""
NFL Advanced Receiving Statistics Web Scraper

Scrapes NFL advanced receiving statistics data from pro-football-reference.com
Extracts advanced receiving metrics for a given season.
"""

from scraper_utils import *

def process_receiving_advanced_data(df):
    """
    Processes the advanced receiving statistics DataFrame.
    
    Args:
        df (pd.DataFrame): Raw advanced receiving stats DataFrame
    
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

def scrape_receiving_advanced_stats(year):
    """
    Scrapes NFL advanced receiving statistics for a given year.
    
    Args:
        year (int): The year to scrape data for
    
    Returns:
        pd.DataFrame: Processed advanced receiving stats DataFrame
    """
    url = f"{BASE_URL}/years/{year}/receiving_advanced.htm"
    
    print(f"Scraping advanced receiving stats from: {url}")
    
    driver = setup_driver()
    
    try:
        driver.get(url)
        table = wait_for_table(driver, "adv_receiving")
        
        # Convert table to DataFrame
        df = table_to_dataframe(table)
        
        # Process the data
        df = process_receiving_advanced_data(df)
        
        return df
        
    except TimeoutException:
        print(f"Timeout waiting for advanced receiving table to load: {url}")
        return None
    except Exception as e:
        print(f"Error scraping advanced receiving data: {e}")
        return None
    finally:
        driver.quit()

def main():
    """Main execution function."""
    print(f"Scraping NFL advanced receiving statistics for year: {YEAR}")
    
    # Create directory structure
    receiving_advanced_dir = create_directories(YEAR, "plyr_adv_receiving", WEEK_NUMBER)
    print(f"Data will be saved to: {receiving_advanced_dir}")
    
    # Scrape the data
    df = scrape_receiving_advanced_stats(YEAR)
    
    # Save the data
    if df is not None and not df.empty:
        print(f"Successfully scraped advanced receiving stats for {len(df)} players")
        save_data_to_csv(df, receiving_advanced_dir, "plyr_adv_receiving.csv")
    else:
        print("Failed to scrape advanced receiving statistics data")

if __name__ == "__main__":
    main()