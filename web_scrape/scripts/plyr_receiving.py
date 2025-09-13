"""
NFL Receiving Stats Web Scraper

Scrapes NFL receiving statistics data from pro-football-reference.com
Extracts player receiving stats and creates standardized CSV output.
"""

from scraper_utils import *

def process_receiving_data(df, links):
    """
    Processes the receiving DataFrame by cleaning data and adding player links.
    
    Args:
        df (pd.DataFrame): Raw receiving DataFrame
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

def scrape_receiving_stats(year):
    """
    Scrapes NFL receiving statistics for a given year.
    
    Args:
        year (int): The year to scrape data for
    
    Returns:
        pd.DataFrame: Processed receiving stats DataFrame
    """
    url = f"{BASE_URL}/years/{year}/receiving.htm"
    
    # Scrape table and extract player links
    df, player_links = scrape_table_with_links(
        url=url,
        table_id="receiving",
        data_stat_attr="player"
    )
    
    # Process the data
    df = process_receiving_data(df, player_links)
    
    return df

def main():
    """Main execution function."""
    # Start logging session
    start_scraper_session("plyr_receiving")
    
    try:
        print(f"Scraping NFL receiving statistics for year: {YEAR}")
        
        # Create directory structure
        receiving_dir = create_directories(YEAR, "plyr_receiving", WEEK_NUMBER)
        print(f"Data will be saved to: {receiving_dir}")
        
        # Scrape the data
        df = scrape_receiving_stats(YEAR)
        
        # Save the data
        if df is not None:
            print(f"Successfully scraped receiving stats for {len(df)} players")
            save_data_to_csv(df, receiving_dir, "plyr_receiving.csv")
        else:
            print("Failed to scrape receiving stats data")
    
    finally:
        # End logging session
        end_scraper_session()

if __name__ == "__main__":
    main()