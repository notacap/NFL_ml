"""
Team Defense Scoring Stats Scraper

Scrapes team defense scoring statistics from pro-football-reference.com.
Extracts data from the team_scoring table on the team opponent stats page.
"""

from scraper_utils import *

def scrape_team_scoring_defense_data(year):
    """
    Scrapes team defense scoring data for a given year.
    
    Args:
        year (int): The year to scrape data for
    
    Returns:
        pd.DataFrame: DataFrame containing team scoring defense stats or None if failed
    """
    url = f"{BASE_URL}/years/{year}/opp.htm"
    table_id = "team_scoring"
    
    print(f"Scraping team scoring defense data for {year} from: {url}")
    
    # Use the shared scraping function
    df, _ = scrape_table_with_links(url, table_id, "", exclude_conditions=None)
    
    return df

def main():
    """Main execution function."""
    # REQUIRED: Start logging session
    start_scraper_session("tm_def_scoring")
    
    try:
        print(f"Scraping team defense scoring data for year: {YEAR}")
        
        # Create directories
        data_dir = create_directories(YEAR, "tm_def_scoring", WEEK_NUMBER)
        
        # Scrape and save data
        df = scrape_team_scoring_defense_data(YEAR)
        if df is not None:
            save_data_to_csv(df, data_dir, "tm_def_scoring.csv")
        else:
            print("Failed to scrape team scoring defense data")
    
    finally:
        # REQUIRED: End logging session (shows summary)
        end_scraper_session()

if __name__ == "__main__":
    main()