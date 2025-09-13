"""
NFL Red Zone Rushing Statistics Web Scraper

Scrapes NFL red zone rushing statistics data from pro-football-reference.com
Extracts red zone rushing metrics for a given season.
"""

from scraper_utils import *

def process_redzone_rushing_data(df):
    """
    Processes the red zone rushing statistics DataFrame.
    
    Args:
        df (pd.DataFrame): Raw red zone rushing stats DataFrame
    
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

def scrape_redzone_rushing_stats(year):
    """
    Scrapes NFL red zone rushing statistics for a given year.
    
    Args:
        year (int): The year to scrape data for
    
    Returns:
        pd.DataFrame: Processed red zone rushing stats DataFrame
    """
    url = f"{BASE_URL}/years/{year}/redzone-rushing.htm"
    
    print(f"Scraping red zone rushing stats from: {url}")
    
    driver = setup_driver()
    
    try:
        driver.get(url)
        table = wait_for_table(driver, "fantasy_rz")
        
        # Convert table to DataFrame
        df = table_to_dataframe(table)
        
        # Process the data
        df = process_redzone_rushing_data(df)
        
        return df
        
    except TimeoutException:
        print(f"Timeout waiting for red zone rushing table to load: {url}")
        return None
    except Exception as e:
        print(f"Error scraping red zone rushing data: {e}")
        return None
    finally:
        driver.quit()

def main():
    """Main execution function."""
    print(f"Scraping NFL red zone rushing statistics for year: {YEAR}")
    
    # Create directory structure
    redzone_rushing_dir = create_directories(YEAR, "plyr_rz_rushing", WEEK_NUMBER)
    print(f"Data will be saved to: {redzone_rushing_dir}")
    
    # Scrape the data
    df = scrape_redzone_rushing_stats(YEAR)
    
    # Save the data
    if df is not None and not df.empty:
        print(f"Successfully scraped red zone rushing stats for {len(df)} players")
        save_data_to_csv(df, redzone_rushing_dir, "plyr_rz_rushing.csv")
    else:
        print("Failed to scrape red zone rushing statistics data")

if __name__ == "__main__":
    main()