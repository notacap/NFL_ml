"""
NFL Rushing Statistics Web Scraper

Scrapes NFL rushing statistics data from pro-football-reference.com
Extracts running back and player rushing stats for a given season.
"""

from scraper_utils import *

def process_rushing_data(df):
    """
    Processes the rushing statistics DataFrame.
    
    Args:
        df (pd.DataFrame): Raw rushing stats DataFrame
    
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

def scrape_rushing_stats(year):
    """
    Scrapes NFL rushing statistics for a given year.
    
    Args:
        year (int): The year to scrape data for
    
    Returns:
        pd.DataFrame: Processed rushing stats DataFrame
    """
    url = f"{BASE_URL}/years/{year}/rushing.htm"
    
    print(f"Scraping rushing stats from: {url}")
    
    driver = setup_driver()
    
    try:
        driver.get(url)
        table = wait_for_table(driver, "rushing")
        
        # Convert table to DataFrame
        df = table_to_dataframe(table)
        
        # Process the data
        df = process_rushing_data(df)
        
        return df
        
    except TimeoutException:
        print(f"Timeout waiting for rushing table to load: {url}")
        return None
    except Exception as e:
        print(f"Error scraping rushing data: {e}")
        return None
    finally:
        driver.quit()

def main():
    """Main execution function."""
    # Start logging session
    start_scraper_session("plyr_rushing")
    
    try:
        print(f"Scraping NFL rushing statistics for year: {YEAR}")
        
        # Create directory structure
        rushing_dir = create_directories(YEAR, "plyr_rushing", WEEK_NUMBER)
        print(f"Data will be saved to: {rushing_dir}")
        
        # Scrape the data
        df = scrape_rushing_stats(YEAR)
        
        # Save the data
        if df is not None and not df.empty:
            print(f"Successfully scraped rushing stats for {len(df)} players")
            save_data_to_csv(df, rushing_dir, "plyr_rushing.csv")
        else:
            print("Failed to scrape rushing statistics data")
    
    finally:
        # End logging session
        end_scraper_session()

if __name__ == "__main__":
    main()