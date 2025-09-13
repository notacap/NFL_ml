"""
NFL Advanced Passing Stats Web Scraper

Scrapes NFL advanced passing statistics data from pro-football-reference.com
Extracts multiple advanced passing tables: air_yards, accuracy, pressure, play_type
Creates separate CSV files for each table within the advanced_passing_stats directory.
"""

from scraper_utils import *

def process_advanced_passing_data(df, links):
    """
    Processes advanced passing DataFrame by cleaning data and adding player links.
    
    Args:
        df (pd.DataFrame): Raw advanced passing DataFrame
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

def scrape_advanced_passing_table(year, table_id, table_name):
    """
    Scrapes a specific advanced passing table for a given year.
    
    Args:
        year (int): The year to scrape data for
        table_id (str): The ID of the table element
        table_name (str): Name for the table (for logging)
    
    Returns:
        pd.DataFrame: Processed advanced passing DataFrame
    """
    url = f"{BASE_URL}/years/{year}/advanced.htm"
    
    print(f"  Scraping {table_name} table...")
    
    # Scrape table and extract player links
    df, player_links = scrape_table_with_links(
        url=url,
        table_id=table_id,
        data_stat_attr="player"
    )
    
    # Process the data
    df = process_advanced_passing_data(df, player_links)
    
    return df

def scrape_all_advanced_passing_stats(year):
    """
    Scrapes all advanced passing statistics tables for a given year.
    
    Args:
        year (int): The year to scrape data for
    
    Returns:
        dict: Dictionary containing all processed DataFrames
    """
    tables = {
        "air_yards": "Air Yards",
        "accuracy": "Accuracy", 
        "pressure": "Pressure",
        "play_type": "Play Type"
    }
    
    results = {}
    
    for table_id, table_name in tables.items():
        df = scrape_advanced_passing_table(year, table_id, table_name)
        results[table_id] = df
    
    return results

def main():
    """Main execution function."""
    # Start logging session
    start_scraper_session("tm_adv_passing")
    
    try:
        print(f"Scraping NFL advanced passing statistics for year: {YEAR}")
        
        # Create directory structure
        advanced_dir = create_directories(YEAR, "tm_adv_passing", WEEK_NUMBER)
        print(f"Data will be saved to: {advanced_dir}")
        
        # Scrape all tables
        all_tables = scrape_all_advanced_passing_stats(YEAR)
        
        # Save each table to separate CSV files in their own subdirectories
        subdirectory_names = {
            "air_yards": "tm_airyards",
            "accuracy": "tm_accuracy", 
            "pressure": "tm_pass_pressure",
            "play_type": "tm_pass_playtype"
        }
        
        for table_id, df in all_tables.items():
            if df is not None:
                # Create subdirectory for each table
                subdir_name = subdirectory_names[table_id]
                table_dir = create_directories(YEAR, f"tm_adv_passing/{subdir_name}", WEEK_NUMBER)
                print(f"Successfully scraped {table_id} stats for {len(df)} players")
                save_data_to_csv(df, table_dir, f"{subdir_name}.csv")
            else:
                print(f"Failed to scrape {table_id} stats data")
    
    finally:
        # End logging session
        end_scraper_session()

if __name__ == "__main__":
    main()