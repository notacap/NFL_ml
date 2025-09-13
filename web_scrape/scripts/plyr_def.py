"""
NFL Defense Stats Web Scraper

Scrapes NFL defensive statistics data from pro-football-reference.com
Extracts player defensive stats and creates standardized CSV output.
"""

from scraper_utils import *

def process_defense_data(df, links):
    """
    Processes the defense DataFrame by cleaning data and adding player links.
    
    Args:
        df (pd.DataFrame): Raw defense DataFrame
        links (list): List of player URLs
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    if df is None or df.empty:
        return None
    
    # Remove rows where Player column contains header text
    df = df[df.iloc[:, 0] != 'Player'].copy()
    df = df[df.iloc[:, 0] != 'Rk'].copy()
    
    # Remove rows with NaN in player column
    df = df.dropna(subset=[df.columns[0]]).copy()
    
    # Add player links column
    if len(links) >= len(df):
        links = links[:len(df)]
        df['Player_URL'] = links
    
    return df

def scrape_defense_stats(year):
    """
    Scrapes NFL defensive statistics for a given year.
    
    Args:
        year (int): The year to scrape data for
    
    Returns:
        pd.DataFrame: Processed defense stats DataFrame
    """
    url = f"{BASE_URL}/years/{year}/defense.htm"
    
    # Scrape table and extract player links
    df, player_links = scrape_table_with_links(
        url=url,
        table_id="defense",
        data_stat_attr="player"
    )
    
    # Process the data
    df = process_defense_data(df, player_links)
    
    return df

def main():
    """Main execution function."""
    # Start logging session
    start_scraper_session("plyr_def")
    
    try:
        print(f"Scraping NFL defensive statistics for year: {YEAR}")
        
        # Create directory structure
        defense_dir = create_directories(YEAR, "plyr_def", WEEK_NUMBER)
        print(f"Data will be saved to: {defense_dir}")
        
        # Scrape the data
        df = scrape_defense_stats(YEAR)
        
        # Save the data
        if df is not None:
            print(f"Successfully scraped defensive stats for {len(df)} players")
            save_data_to_csv(df, defense_dir, "plyr_def.csv")
        else:
            print("Failed to scrape defensive stats data")
    
    finally:
        # End logging session
        end_scraper_session()

if __name__ == "__main__":
    main()