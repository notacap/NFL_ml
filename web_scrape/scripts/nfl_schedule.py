"""
NFL Schedule Web Scraper

Scrapes NFL game schedule data from pro-football-reference.com
Extracts comprehensive game schedule information.
"""

from scraper_utils import *

def process_schedule_data(df):
    """
    Processes the schedule DataFrame by cleaning and standardizing the data.
    
    Args:
        df (pd.DataFrame): Raw schedule DataFrame
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    if df is None or df.empty:
        return None
    
    # Basic filtering - remove common header patterns
    df = df[df.iloc[:, 0] != 'Week'].copy() if 'Week' in df.iloc[:, 0].values else df
    df = df[df.iloc[:, 0] != 'Wk'].copy() if 'Wk' in df.iloc[:, 0].values else df
    df = df.dropna(how='all').copy()  # Remove completely empty rows
    
    # Add year column for consistency
    if not df.empty:
        df['year'] = YEAR
        df['week'] = WEEK_NUMBER
    
    return df

def scrape_nfl_schedule(year):
    """
    Scrapes NFL schedule for a given year.
    
    Args:
        year (int): The year to scrape data for
    
    Returns:
        pd.DataFrame: Processed schedule DataFrame
    """
    url = f"{BASE_URL}/years/{year}/games.htm"
    
    # Scrape table - no links needed for schedule
    df, _ = scrape_table_with_links(
        url=url,
        table_id="games",  # The schedule table has id="games"
        data_stat_attr="boxscore_word",  # Use boxscore_word column for any link extraction
        exclude_conditions=None  # No specific exclusions needed
    )
    
    # Process the data
    df = process_schedule_data(df)
    
    return df

def main():
    """Main execution function."""
    # Start logging session
    start_scraper_session("nfl_schedule")
    
    try:
        print(f"Scraping NFL schedule for year: {YEAR}")
        
        # Create directory structure
        schedule_dir = create_directories(YEAR, "nfl_schedule", WEEK_NUMBER)
        print(f"Data will be saved to: {schedule_dir}")
        
        # Scrape the data
        df = scrape_nfl_schedule(YEAR)
        
        # Save the data
        if df is not None:
            print(f"Successfully scraped schedule with {len(df)} games")
            save_data_to_csv(df, schedule_dir, "nfl_schedule.csv")
        else:
            print("Failed to scrape NFL schedule")
    
    finally:
        # End logging session
        end_scraper_session()

if __name__ == "__main__":
    main()