"""
NFL Advanced Rushing Stats Web Scraper

Scrapes NFL advanced rushing statistics data from pro-football-reference.com
Extracts player advanced rushing stats and creates standardized CSV output.
"""

from scraper_utils import *

def process_rushing_advanced_data(df, links):
    """
    Processes the advanced rushing DataFrame by cleaning data and adding player links.
    
    Args:
        df (pd.DataFrame): Raw advanced rushing DataFrame
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

def scrape_rushing_advanced_stats(year):
    """
    Scrapes NFL advanced rushing statistics for a given year.
    
    Args:
        year (int): The year to scrape data for
    
    Returns:
        pd.DataFrame: Processed advanced rushing stats DataFrame
    """
    url = f"{BASE_URL}/years/{year}/rushing_advanced.htm"
    
    # Scrape table and extract player links
    df, player_links = scrape_table_with_links(
        url=url,
        table_id="adv_rushing",
        data_stat_attr="player"
    )
    
    # Process the data
    df = process_rushing_advanced_data(df, player_links)
    
    return df

def main():
    """Main execution function."""
    print(f"Scraping NFL advanced rushing statistics for year: {YEAR}")
    
    # Create directory structure
    rushing_advanced_dir = create_directories(YEAR, "plyr_adv_rushing", WEEK_NUMBER)
    print(f"Data will be saved to: {rushing_advanced_dir}")
    
    # Scrape the data
    df = scrape_rushing_advanced_stats(YEAR)
    
    # Save the data
    if df is not None:
        print(f"Successfully scraped advanced rushing stats for {len(df)} players")
        save_data_to_csv(df, rushing_advanced_dir, "plyr_adv_rushing.csv")
    else:
        print("Failed to scrape advanced rushing stats data")

if __name__ == "__main__":
    main()