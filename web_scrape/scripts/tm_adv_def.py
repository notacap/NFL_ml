"""
Team Advanced Defense Stats Scraper

Scrapes team advanced defense statistics from pro-football-reference.com.
Extracts data from the advanced_defense table on the team opponent stats page.
"""

from scraper_utils import *

def scrape_advanced_defense_data(year):
    """
    Scrapes team advanced defense data for a given year.
    
    Args:
        year (int): The year to scrape data for
    
    Returns:
        pd.DataFrame: DataFrame containing advanced defense stats or None if failed
    """
    url = f"{BASE_URL}/years/{year}/opp.htm"
    table_id = "advanced_defense"
    
    print(f"Scraping advanced defense data for {year} from: {url}")
    
    # Use the shared scraping function
    df, _ = scrape_table_with_links(url, table_id, "", exclude_conditions=None)
    
    return df

def main():
    """Main execution function."""
    # REQUIRED: Start logging session
    start_scraper_session("tm_adv_def")
    
    try:
        print(f"Scraping team advanced defense data for year: {YEAR}")
        
        # Create directories
        data_dir = create_directories(YEAR, "tm_adv_def", WEEK_NUMBER)
        
        # Scrape and save data
        df = scrape_advanced_defense_data(YEAR)
        if df is not None:
            save_data_to_csv(df, data_dir, "tm_adv_def.csv")
        else:
            print("Failed to scrape advanced defense data")
    
    finally:
        # REQUIRED: End logging session (shows summary)
        end_scraper_session()

if __name__ == "__main__":
    main()