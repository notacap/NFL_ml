"""
NFL Team Defense Stats Scraper

Scrapes team defense stats data from pro-football-reference.com
Extracts comprehensive defensive statistics for all teams.
"""

from scraper_utils import *

def process_team_def_data(df):
    """
    Processes the team defense stats DataFrame.
    
    Args:
        df (pd.DataFrame): Raw team defense stats DataFrame
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    if df is None or df.empty:
        return None
    
    # Basic cleaning - remove any rows that might be headers
    if len(df) > 0:
        # Remove rows where the first column contains 'Tm' (team header)
        df = df[df.iloc[:, 0] != 'Tm'].copy()
        # Remove rows where the first column contains 'Rk' (rank header)
        df = df[df.iloc[:, 0] != 'Rk'].copy()
    
    return df

def scrape_team_def(year):
    """
    Scrapes team defense stats data for a given year.
    
    Args:
        year (int): The year to scrape data for
    
    Returns:
        pd.DataFrame: Processed team defense stats DataFrame
    """
    url = f"{BASE_URL}/years/{year}/opp.htm"
    
    # Define exclusion conditions
    exclude_conditions = [
        {'column': 0, 'contains': 'Tm'},  # Skip header rows
        {'column': 0, 'contains': 'Rk'}   # Skip rank header rows
    ]
    
    # Scrape table (no links needed for this table)
    df, _ = scrape_table_with_links(
        url=url,
        table_id="team_stats",
        data_stat_attr="team",  # Using team as data-stat for consistency
        exclude_conditions=exclude_conditions
    )
    
    # Process the data
    df = process_team_def_data(df)
    
    return df

def main():
    """Main execution function."""
    # Start logging session
    start_scraper_session("tm_def")
    
    try:
        print(f"Scraping team defense stats data for year: {YEAR}")
        
        # Create directory structure
        data_dir = create_directories(YEAR, "tm_def", WEEK_NUMBER)
        print(f"Data will be saved to: {data_dir}")
        
        # Scrape the data
        df = scrape_team_def(YEAR)
        
        # Save the data
        if df is not None:
            print(f"Successfully scraped team defense stats for {len(df)} teams")
            save_data_to_csv(df, data_dir, "tm_def.csv")
        else:
            print("Failed to scrape team defense stats data")
    
    finally:
        # End logging session
        end_scraper_session()

if __name__ == "__main__":
    main()