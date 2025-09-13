"""
NFL Schedule Web Scraper

Scrapes NFL game schedule data from pro-football-reference.com
Extracts game details, boxscore links, and creates Home/Away columns.
"""

from scraper_utils import *

def add_home_away_columns(df):
    """
    Adds Home, Away, Home Score, and Away Score columns based on the '@' symbol logic.
    
    Args:
        df (pd.DataFrame): DataFrame with game data
    
    Returns:
        pd.DataFrame: DataFrame with added Home, Away, Home Score, and Away Score columns
    """
    # Detect which column naming convention is used
    if 'PtsW' in df.columns and 'PtsL' in df.columns:
        winner_score_col = 'PtsW'
        loser_score_col = 'PtsL'
    elif 'Pts' in df.columns and 'Pts.1' in df.columns:
        winner_score_col = 'Pts'
        loser_score_col = 'Pts.1'
    else:
        raise ValueError("Could not find score columns. Expected either 'PtsW'/'PtsL' or 'Pts'/'Pts.1'")
    
    home_teams = []
    away_teams = []
    home_scores = []
    away_scores = []
    
    for _, row in df.iterrows():
        winner_tie = row.iloc[4]  # Winner/tie column (index 4)
        at_symbol = row.iloc[5]   # 5th indexed column (index 5)
        loser_tie = row.iloc[6]   # Loser/tie column (index 6)
        winner_score = row[winner_score_col]   # Winner/tie team score
        loser_score = row[loser_score_col]     # Loser/tie team score
        
        if at_symbol == "@":
            # Away team is Winner/tie, Home team is Loser/tie
            away_teams.append(winner_tie)
            home_teams.append(loser_tie)
            away_scores.append(winner_score)
            home_scores.append(loser_score)
        else:
            # Home team is Winner/tie, Away team is Loser/tie
            home_teams.append(winner_tie)
            away_teams.append(loser_tie)
            home_scores.append(winner_score)
            away_scores.append(loser_score)
    
    # Add the new columns
    df['Home'] = home_teams
    df['Away'] = away_teams
    df['Home Score'] = home_scores
    df['Away Score'] = away_scores
    
    return df

def process_schedule_data(df, boxscore_links):
    """
    Processes the schedule DataFrame by adding boxscore links and Home/Away columns.
    
    Args:
        df (pd.DataFrame): Raw schedule DataFrame
        boxscore_links (list): List of boxscore URLs
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    if df is None or df.empty:
        return None
    
    # Find the column that contains "boxscore" text
    boxscore_col_idx = None
    for idx, col in enumerate(df.columns):
        if len(df) > 0 and df.iloc[0, idx] == "boxscore":
            boxscore_col_idx = idx
            break
    
    # If we didn't find it by value, use column 7 (Unnamed: 7)
    if boxscore_col_idx is None:
        boxscore_col_idx = 7
    
    # Rename the column to "Boxscore"
    cols = list(df.columns)
    cols[boxscore_col_idx] = "Boxscore"
    df.columns = cols
    
    # Update the column with URLs
    if len(boxscore_links) >= len(df):
        boxscore_links = boxscore_links[:len(df)]
        df.iloc[:, boxscore_col_idx] = boxscore_links
    
    # Add Home and Away columns
    df = add_home_away_columns(df)
    
    # Remove the 5th indexed column (column 5)
    df = df.drop(df.columns[5], axis=1)
    
    return df

def scrape_schedule(year):
    """
    Scrapes NFL schedule data for a given year.
    
    Args:
        year (int): The year to scrape data for
    
    Returns:
        pd.DataFrame: Processed schedule DataFrame
    """
    url = f"{BASE_URL}/years/{year}/games.htm"
    
    # Define exclusion conditions
    exclude_conditions = [
        {'column': 0, 'contains': 'Pre'}  # Skip preseason games
    ]
    
    # Scrape table and extract boxscore links
    df, boxscore_links = scrape_table_with_links(
        url=url,
        table_id="games",
        data_stat_attr="boxscore_word",
        exclude_conditions=exclude_conditions
    )
    
    # Process the data
    df = process_schedule_data(df, boxscore_links)
    
    return df

def main():
    """Main execution function."""
    # Start logging session
    start_scraper_session("schedule")
    
    try:
        print(f"Scraping NFL schedule data for year: {YEAR}")
        
        # Create directory structure
        schedule_dir = create_directories(YEAR, "schedule", WEEK_NUMBER)
        print(f"Data will be saved to: {schedule_dir}")
        
        # Scrape the data
        df = scrape_schedule(YEAR)
        
        # Save the data
        if df is not None:
            print(f"Successfully scraped {len(df)} games")
            save_data_to_csv(df, schedule_dir, "schedule.csv")
        else:
            print("Failed to scrape schedule data")
    
    finally:
        # End logging session
        end_scraper_session()

if __name__ == "__main__":
    main()