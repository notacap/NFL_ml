"""
NFL Team Advanced Receiving Stats Web Scraper

Scrapes NFL team advanced receiving statistics data from pro-football-reference.com
Extracts team advanced receiving stats and creates standardized CSV output.
"""

from scraper_utils import *

def process_tm_adv_receiving_data(df, links):
    """
    Processes the team advanced receiving DataFrame by cleaning data and adding team links.
    
    Args:
        df (pd.DataFrame): Raw team advanced receiving DataFrame
        links (list): List of team URLs
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    if df is None or df.empty:
        return None
    
    # Remove rows where Team column contains header text
    df = df[df.iloc[:, 0] != 'Team'].copy()
    df = df[df.iloc[:, 0] != 'Rk'].copy()
    
    # Remove rows with NaN in team column
    df = df.dropna(subset=[df.columns[0]]).copy()
    
    # Add team links column
    if len(links) >= len(df):
        links = links[:len(df)]
        df['Team_URL'] = links
    
    return df

def scrape_tm_adv_receiving_stats(year):
    """
    Scrapes NFL team advanced receiving statistics for a given year.
    
    Args:
        year (int): The year to scrape data for
    
    Returns:
        pd.DataFrame: Processed team advanced receiving stats DataFrame
    """
    url = f"{BASE_URL}/years/{year}/advanced.htm"
    
    # Scrape table and extract team links
    df, team_links = scrape_table_with_links(
        url=url,
        table_id="advanced_receiving",
        data_stat_attr="team"
    )
    
    # Process the data
    df = process_tm_adv_receiving_data(df, team_links)
    
    return df

def main():
    """Main execution function."""
    print(f"Scraping NFL team advanced receiving statistics for year: {YEAR}")
    
    # Create directory structure
    tm_adv_receiving_dir = create_directories(YEAR, "tm_adv_receiving", WEEK_NUMBER)
    print(f"Data will be saved to: {tm_adv_receiving_dir}")
    
    # Scrape the data
    df = scrape_tm_adv_receiving_stats(YEAR)
    
    # Save the data
    if df is not None:
        print(f"Successfully scraped team advanced receiving stats for {len(df)} teams")
        save_data_to_csv(df, tm_adv_receiving_dir, "tm_adv_receiving.csv")
    else:
        print("Failed to scrape team advanced receiving stats data")

if __name__ == "__main__":
    main()