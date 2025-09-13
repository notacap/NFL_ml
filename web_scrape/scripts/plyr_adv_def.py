"""
NFL Advanced Defense Statistics Web Scraper

Scrapes NFL advanced defense statistics data from pro-football-reference.com
Extracts advanced defensive metrics for a given season.
"""

from scraper_utils import *

def process_defense_advanced_data(df):
    """
    Processes the advanced defense statistics DataFrame.
    
    Args:
        df (pd.DataFrame): Raw advanced defense stats DataFrame
    
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

def scrape_defense_advanced_stats(year):
    """
    Scrapes NFL advanced defense statistics for a given year.
    
    Args:
        year (int): The year to scrape data for
    
    Returns:
        pd.DataFrame: Processed advanced defense stats DataFrame
    """
    url = f"{BASE_URL}/years/{year}/defense_advanced.htm"
    
    print(f"Scraping advanced defense stats from: {url}")
    
    driver = setup_driver()
    
    try:
        driver.get(url)
        table = wait_for_table(driver, "defense_advanced")
        
        # Convert table to DataFrame
        df = table_to_dataframe(table)
        
        # Process the data
        df = process_defense_advanced_data(df)
        
        return df
        
    except TimeoutException:
        print(f"Timeout waiting for advanced defense table to load: {url}")
        return None
    except Exception as e:
        print(f"Error scraping advanced defense data: {e}")
        return None
    finally:
        driver.quit()

def main():
    """Main execution function."""
    print(f"Scraping NFL advanced defense statistics for year: {YEAR}")
    
    # Create directory structure
    defense_advanced_dir = create_directories(YEAR, "plyr_adv_def", WEEK_NUMBER)
    print(f"Data will be saved to: {defense_advanced_dir}")
    
    # Scrape the data
    df = scrape_defense_advanced_stats(YEAR)
    
    # Save the data
    if df is not None and not df.empty:
        print(f"Successfully scraped advanced defense stats for {len(df)} players")
        save_data_to_csv(df, defense_advanced_dir, "plyr_adv_def.csv")
    else:
        print("Failed to scrape advanced defense statistics data")

if __name__ == "__main__":
    main()