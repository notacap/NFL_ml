"""
Team Defense Drives Against Scraper

Scrapes team defense drives against statistics from pro-football-reference.com.
Extracts data from the drives table on the team opponent stats page.
"""

from scraper_utils import *

def scrape_drives_against_data(year):
    """
    Scrapes team defense drives against data for a given year.
    
    Args:
        year (int): The year to scrape data for
    
    Returns:
        pd.DataFrame: DataFrame containing drives against stats or None if failed
    """
    url = f"{BASE_URL}/years/{year}/opp.htm"
    table_id = "drives"
    
    print(f"Scraping drives against data for {year} from: {url}")
    
    # Use the shared scraping function
    df, _ = scrape_table_with_links(url, table_id, "", exclude_conditions=None)
    
    return df

def main():
    """Main execution function."""
    # REQUIRED: Start logging session
    start_scraper_session("tm_def_drives_against")
    
    try:
        print(f"Scraping team defense drives against data for year: {YEAR}")
        
        # Create directories
        data_dir = create_directories(YEAR, "tm_def_drives_against", WEEK_NUMBER)
        
        # Scrape and save data
        df = scrape_drives_against_data(YEAR)
        if df is not None:
            save_data_to_csv(df, data_dir, "tm_def_drives_against.csv")
        else:
            print("Failed to scrape drives against data")
    
    finally:
        # REQUIRED: End logging session (shows summary)
        end_scraper_session()

if __name__ == "__main__":
    main()