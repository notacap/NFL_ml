"""
NFL Team Scoring Stats Web Scraper

Scrapes NFL team scoring statistics from pro-football-reference.com
Extracts comprehensive team scoring data.
"""

from scraper_utils import *

def process_team_stats_data(df):
    """
    Processes the team stats DataFrame by cleaning and standardizing the data.
    
    Args:
        df (pd.DataFrame): Raw team stats DataFrame
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    if df is None or df.empty:
        return None
    
    # Basic filtering - remove common header patterns
    df = df[df.iloc[:, 0] != 'Rk'].copy() if 'Rk' in df.iloc[:, 0].values else df
    df = df[df.iloc[:, 0] != 'Tm'].copy() if 'Tm' in df.iloc[:, 0].values else df
    df = df.dropna(how='all').copy()  # Remove completely empty rows
    
    # Add year column for consistency
    if not df.empty:
        df['year'] = YEAR
        df['week'] = WEEK_NUMBER
    
    return df

def scrape_team_scoring_stats(year):
    """
    Scrapes NFL team scoring statistics for a given year.
    
    Args:
        year (int): The year to scrape data for
    
    Returns:
        pd.DataFrame: Processed team scoring stats DataFrame
    """
    url = f"{BASE_URL}/years/{year}/"
    
    # Scrape table - no links needed for team stats
    df, _ = scrape_table_with_links(
        url=url,
        table_id="team_scoring",
        data_stat_attr="team",  # Use team column for any link extraction
        exclude_conditions=None  # No specific exclusions needed
    )
    
    # Process the data
    df = process_team_stats_data(df)
    
    return df

def main():
    """Main execution function."""
    # Start logging session
    start_scraper_session("tm_scoring")
    
    try:
        print(f"Scraping NFL team scoring stats for year: {YEAR}")
        
        # Create directory structure
        team_stats_dir = create_directories(YEAR, "tm_scoring", WEEK_NUMBER)
        print(f"Data will be saved to: {team_stats_dir}")
        
        # Scrape the data
        df = scrape_team_scoring_stats(YEAR)
        
        # Save the data
        if df is not None:
            print(f"Successfully scraped team scoring stats for {len(df)} teams")
            save_data_to_csv(df, team_stats_dir, "tm_scoring.csv")
        else:
            print("Failed to scrape team scoring stats")
    
    finally:
        # End logging session
        end_scraper_session()

if __name__ == "__main__":
    main()