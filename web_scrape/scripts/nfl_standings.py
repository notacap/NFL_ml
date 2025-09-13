"""
NFL Standings Web Scraper

Scrapes NFL standings data (AFC and NFC) from pro-football-reference.com
Extracts comprehensive standings information for both conferences.
"""

from scraper_utils import *

def process_standings_data(df, conference):
    """
    Processes the standings DataFrame by cleaning and standardizing the data.
    
    Args:
        df (pd.DataFrame): Raw standings DataFrame
        conference (str): Conference name (AFC or NFC)
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    if df is None or df.empty:
        return None
    
    # Basic filtering - remove common header patterns
    df = df[df.iloc[:, 0] != 'Tm'].copy() if 'Tm' in df.iloc[:, 0].values else df
    df = df[df.iloc[:, 0] != 'Team'].copy() if 'Team' in df.iloc[:, 0].values else df
    df = df.dropna(how='all').copy()  # Remove completely empty rows
    
    # Add metadata columns
    if not df.empty:
        df['conference'] = conference
        df['year'] = YEAR
        df['week'] = WEEK_NUMBER
    
    return df

def scrape_nfl_standings(year):
    """
    Scrapes NFL standings for both AFC and NFC conferences for a given year.
    
    Args:
        year (int): The year to scrape data for
    
    Returns:
        pd.DataFrame: Combined processed standings DataFrame for both conferences
    """
    url = f"{BASE_URL}/years/{year}/"
    
    # Scrape AFC table
    print("  Scraping AFC standings...")
    afc_df, _ = scrape_table_with_links(
        url=url,
        table_id="AFC",
        data_stat_attr="team",  # Use team column for any link extraction
        exclude_conditions=None
    )
    
    # Process AFC data
    afc_df = process_standings_data(afc_df, "AFC")
    
    # Scrape NFC table
    print("  Scraping NFC standings...")
    nfc_df, _ = scrape_table_with_links(
        url=url,
        table_id="NFC",
        data_stat_attr="team",  # Use team column for any link extraction
        exclude_conditions=None
    )
    
    # Process NFC data
    nfc_df = process_standings_data(nfc_df, "NFC")
    
    # Combine both dataframes
    if afc_df is not None and nfc_df is not None:
        combined_df = pd.concat([afc_df, nfc_df], ignore_index=True)
        return combined_df
    elif afc_df is not None:
        return afc_df
    elif nfc_df is not None:
        return nfc_df
    else:
        return None

def main():
    """Main execution function."""
    # Start logging session
    start_scraper_session("nfl_standings")
    
    try:
        print(f"Scraping NFL standings for year: {YEAR}, week: {WEEK_NUMBER}")
        
        # Create directory structure
        standings_dir = create_directories(YEAR, "nfl_standings", WEEK_NUMBER)
        print(f"Data will be saved to: {standings_dir}")
        
        # Scrape the data
        df = scrape_nfl_standings(YEAR)
        
        # Save the data
        if df is not None:
            print(f"Successfully scraped standings for {len(df)} teams")
            save_data_to_csv(df, standings_dir, "nfl_standings.csv")
        else:
            print("Failed to scrape NFL standings")
    
    finally:
        # End logging session
        end_scraper_session()

if __name__ == "__main__":
    main()