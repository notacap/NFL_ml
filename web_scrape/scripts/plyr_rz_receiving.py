"""
NFL Red Zone Receiving Statistics Web Scraper

Scrapes NFL red zone receiving statistics data from pro-football-reference.com
Extracts red zone receiving metrics for a given season.
"""

from scraper_utils import *

def process_redzone_receiving_data(df):
    """
    Processes the red zone receiving statistics DataFrame.
    
    Args:
        df (pd.DataFrame): Raw red zone receiving stats DataFrame
    
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

def scrape_redzone_receiving_stats(year):
    """
    Scrapes NFL red zone receiving statistics for a given year.
    
    Args:
        year (int): The year to scrape data for
    
    Returns:
        pd.DataFrame: Processed red zone receiving stats DataFrame
    """
    url = f"{BASE_URL}/years/{year}/redzone-receiving.htm"
    
    print(f"Scraping red zone receiving stats from: {url}")
    
    driver = setup_driver()
    
    try:
        driver.get(url)
        table = wait_for_table(driver, "fantasy_rz")
        
        # Convert table to DataFrame
        df = table_to_dataframe(table)
        
        # Process the data
        df = process_redzone_receiving_data(df)
        
        return df
        
    except TimeoutException:
        print(f"Timeout waiting for red zone receiving table to load: {url}")
        return None
    except Exception as e:
        print(f"Error scraping red zone receiving data: {e}")
        return None
    finally:
        driver.quit()

def main():
    """Main execution function."""
    print(f"Scraping NFL red zone receiving statistics for year: {YEAR}")
    
    # Create directory structure
    redzone_receiving_dir = create_directories(YEAR, "plyr_rz_receiving", WEEK_NUMBER)
    print(f"Data will be saved to: {redzone_receiving_dir}")
    
    # Scrape the data
    df = scrape_redzone_receiving_stats(YEAR)
    
    # Save the data
    if df is not None and not df.empty:
        print(f"Successfully scraped red zone receiving stats for {len(df)} players")
        save_data_to_csv(df, redzone_receiving_dir, "plyr_rz_receiving.csv")
    else:
        print("Failed to scrape red zone receiving statistics data")

if __name__ == "__main__":
    main()