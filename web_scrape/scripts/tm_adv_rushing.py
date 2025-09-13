"""
NFL Advanced Rushing Statistics Web Scraper

Scrapes NFL advanced rushing statistics data from pro-football-reference.com
Extracts advanced rushing metrics for a given season.
"""

from scraper_utils import *

def process_advanced_rushing_data(df):
    """
    Processes the advanced rushing statistics DataFrame.
    
    Args:
        df (pd.DataFrame): Raw advanced rushing stats DataFrame
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    if df is None or df.empty:
        return None
    
    # Remove any header rows that might have been captured
    df = df[df.iloc[:, 0] != 'Rk'].copy()  # Remove header rows with 'Rk' in first column
    df = df[df.iloc[:, 1] != 'Team'].copy()  # Remove header rows with 'Team' in second column
    
    # Reset index after filtering
    df.reset_index(drop=True, inplace=True)
    
    return df

def scrape_advanced_rushing_stats(year):
    """
    Scrapes NFL advanced rushing statistics for a given year.
    
    Args:
        year (int): The year to scrape data for
    
    Returns:
        pd.DataFrame: Processed advanced rushing stats DataFrame
    """
    url = f"{BASE_URL}/years/{year}/advanced.htm"
    
    print(f"Scraping advanced rushing stats from: {url}")
    
    driver = setup_driver()
    
    try:
        driver.get(url)
        table = wait_for_table(driver, "advanced_rushing")
        
        # Convert table to DataFrame
        df = table_to_dataframe(table)
        
        # Process the data
        df = process_advanced_rushing_data(df)
        
        return df
        
    except TimeoutException:
        print(f"Timeout waiting for advanced rushing table to load: {url}")
        return None
    except Exception as e:
        print(f"Error scraping advanced rushing data: {e}")
        return None
    finally:
        driver.quit()

def main():
    """Main execution function."""
    print(f"Scraping NFL advanced rushing statistics for year: {YEAR}")
    
    # Create directory structure
    advanced_rushing_dir = create_directories(YEAR, "tm_adv_rushing", WEEK_NUMBER)
    print(f"Data will be saved to: {advanced_rushing_dir}")
    
    # Scrape the data
    df = scrape_advanced_rushing_stats(YEAR)
    
    # Save the data
    if df is not None and not df.empty:
        print(f"Successfully scraped advanced rushing stats for {len(df)} teams")
        save_data_to_csv(df, advanced_rushing_dir, "tm_adv_rushing.csv")
    else:
        print("Failed to scrape advanced rushing statistics data")

if __name__ == "__main__":
    main()