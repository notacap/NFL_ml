"""
NFL Passing Statistics Web Scraper

Scrapes NFL passing statistics data from pro-football-reference.com
Extracts quarterback passing stats for a given season.
"""

from scraper_utils import *

def process_passing_data(df):
    """
    Processes the passing statistics DataFrame.
    
    Args:
        df (pd.DataFrame): Raw passing stats DataFrame
    
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

def scrape_passing_stats(year):
    """
    Scrapes NFL passing statistics for a given year.
    
    Args:
        year (int): The year to scrape data for
    
    Returns:
        pd.DataFrame: Processed passing stats DataFrame
    """
    url = f"{BASE_URL}/years/{year}/passing.htm"
    
    print(f"Scraping passing stats from: {url}")
    
    driver = setup_driver()
    
    try:
        driver.get(url)
        table = wait_for_table(driver, "passing")
        
        # Convert table to DataFrame
        df = table_to_dataframe(table)
        
        # Process the data
        df = process_passing_data(df)
        
        return df
        
    except TimeoutException:
        print(f"Timeout waiting for passing table to load: {url}")
        return None
    except Exception as e:
        print(f"Error scraping passing data: {e}")
        return None
    finally:
        driver.quit()

def main():
    """Main execution function."""
    # Start logging session
    start_scraper_session("plyr_passing")
    
    try:
        print(f"Scraping NFL passing statistics for year: {YEAR}")
        
        # Create directory structure
        passing_dir = create_directories(YEAR, "plyr_passing", WEEK_NUMBER)
        print(f"Data will be saved to: {passing_dir}")
        
        # Scrape the data
        df = scrape_passing_stats(YEAR)
        
        # Save the data
        if df is not None and not df.empty:
            print(f"Successfully scraped passing stats for {len(df)} players")
            save_data_to_csv(df, passing_dir, "plyr_passing.csv")
        else:
            print("Failed to scrape passing statistics data")
    
    finally:
        # End logging session
        end_scraper_session()

if __name__ == "__main__":
    main()