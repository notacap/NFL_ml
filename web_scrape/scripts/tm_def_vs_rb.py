"""
NFL Team Defense vs Running Backs Fantasy Points Scraper

Scrapes team defense vs running backs fantasy points data from pro-football-reference.com
Extracts defensive stats showing how many fantasy points each team allows to RBs.
"""

from scraper_utils import *

def process_def_vs_rb_data(df):
    """
    Processes the team defense vs RB DataFrame.
    
    Args:
        df (pd.DataFrame): Raw defense vs RB DataFrame
    
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

def scrape_def_vs_rb(year):
    """
    Scrapes team defense vs RB fantasy points data for a given year.
    
    Args:
        year (int): The year to scrape data for
    
    Returns:
        pd.DataFrame: Processed defense vs RB DataFrame
    """
    url = f"{BASE_URL}/years/{year}/fantasy-points-against-RB.htm"
    
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
    df = process_def_vs_rb_data(df)
    
    return df

def main():
    """Main execution function."""
    # Start logging session
    start_scraper_session("tm_def_vs_rb")
    
    try:
        print(f"Scraping team defense vs RB fantasy points data for year: {YEAR}")
        
        # Create directory structure
        data_dir = create_directories(YEAR, "tm_defense_vs_rb", WEEK_NUMBER)
        print(f"Data will be saved to: {data_dir}")
        
        # Scrape the data
        df = scrape_def_vs_rb(YEAR)
        
        # Save the data
        if df is not None:
            print(f"Successfully scraped defense vs RB data for {len(df)} teams")
            save_data_to_csv(df, data_dir, "tm_def_vs_rb.csv")
        else:
            print("Failed to scrape team defense vs RB data")
    
    finally:
        # End logging session
        end_scraper_session()

if __name__ == "__main__":
    main()