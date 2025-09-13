"""
NFL Team Defense vs Wide Receivers Fantasy Points Scraper

Scrapes team defense vs wide receivers fantasy points data from pro-football-reference.com
Extracts defensive stats showing how many fantasy points each team allows to WRs.
"""

from scraper_utils import *

def process_def_vs_wr_data(df):
    """
    Processes the team defense vs WR DataFrame.
    
    Args:
        df (pd.DataFrame): Raw defense vs WR DataFrame
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    if df is None or df.empty:
        return None
    
    # Basic cleaning - remove any rows that might be headers
    if len(df) > 0:
        # Remove rows where the first column contains 'Tm' (team header)
        df = df[df.iloc[:, 0] != 'Tm'].copy()
    
    return df

def scrape_def_vs_wr(year):
    """
    Scrapes team defense vs WR fantasy points data for a given year.
    
    Args:
        year (int): The year to scrape data for
    
    Returns:
        pd.DataFrame: Processed defense vs WR DataFrame
    """
    url = f"{BASE_URL}/years/{year}/fantasy-points-against-WR.htm"
    
    # Define exclusion conditions
    exclude_conditions = [
        {'column': 0, 'contains': 'Tm'}  # Skip header rows
    ]
    
    # Scrape table (no links needed for this table)
    df, _ = scrape_table_with_links(
        url=url,
        table_id="fantasy_def",
        data_stat_attr="team",  # Using team as data-stat for consistency
        exclude_conditions=exclude_conditions
    )
    
    # Process the data
    df = process_def_vs_wr_data(df)
    
    return df

def main():
    """Main execution function."""
    # Start logging session
    start_scraper_session("tm_def_vs_wr")
    
    try:
        print(f"Scraping team defense vs WR fantasy points data for year: {YEAR}")
        
        # Create directory structure
        data_dir = create_directories(YEAR, "tm_def_vs_wr", WEEK_NUMBER)
        print(f"Data will be saved to: {data_dir}")
        
        # Scrape the data
        df = scrape_def_vs_wr(YEAR)
        
        # Save the data
        if df is not None:
            print(f"Successfully scraped defense vs WR data for {len(df)} teams")
            save_data_to_csv(df, data_dir, "tm_def_vs_wr.csv")
        else:
            print("Failed to scrape team defense vs WR data")
    
    finally:
        # End logging session
        end_scraper_session()

if __name__ == "__main__":
    main()