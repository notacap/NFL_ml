"""
Player Insert Script

Inserts player data into the plyr table from CSV files.
Finds the most recently created file in the source directory and handles date format conversion.
"""

import sys
import os
import glob
from datetime import datetime
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_utils import DatabaseConnector, YEAR, WEEK, load_csv_data, clean_column_names, handle_null_values, batch_upsert_data

def create_plyr_table(db: DatabaseConnector) -> bool:
    """Create the plyr table if it doesn't exist"""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS plyr (
        plyr_id INT AUTO_INCREMENT PRIMARY KEY,
        team_id INT,
        season_id INT,
        plyr_name VARCHAR(255) NOT NULL,
        plyr_age INT,
        plyr_pos VARCHAR(10),
        plyr_gm_played INT,
        plyr_gm_started INT,
        plyr_weight INT,
        plyr_height INT,
        plyr_yrs_played INT,
        plyr_college VARCHAR(255),
        plyr_birthday DATE,
        plyr_avg_value DECIMAL(5,2),
        plyr_draft_tm VARCHAR(255),
        plyr_draft_rd INT,
        plyr_draft_pick INT,
        plyr_draft_yr INT,
        FOREIGN KEY (team_id) REFERENCES nfl_team(team_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        UNIQUE KEY uk_player_season_pos_exp (plyr_name, season_id, plyr_pos, plyr_weight, plyr_yrs_played)
    )
    """
    
    if db.execute_query(create_table_query):
        print("plyr table created/verified successfully")
        return True
    else:
        print("Failed to create plyr table")
        return False

def get_most_recent_csv_file(directory_path: str) -> str:
    """Find the most recently created CSV file in the directory"""
    try:
        csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {directory_path}")
            return None
        
        # Get the most recent file based on creation time
        most_recent_file = max(csv_files, key=os.path.getctime)
        print(f"Found most recent CSV file: {most_recent_file}")
        return most_recent_file
        
    except Exception as e:
        print(f"Error finding CSV files in {directory_path}: {e}")
        return None

def convert_birthdate_format(date_str):
    """Convert MM/DD/YYYY format to YYYY-MM-DD format for database"""
    if pd.isna(date_str) or date_str is None or date_str == '':
        return None
    
    try:
        # Parse MM/DD/YYYY format
        date_obj = datetime.strptime(str(date_str), "%m/%d/%Y")
        # Return in YYYY-MM-DD format for database
        return date_obj.strftime("%Y-%m-%d")
    except ValueError as e:
        print(f"Warning: Could not parse date '{date_str}': {e}")
        return None

def get_season_id(db: DatabaseConnector, year: int) -> int:
    """Get season_id for the given year"""
    result = db.fetch_all("SELECT season_id FROM nfl_season WHERE year = %s", (year,))
    if result:
        return result[0][0]
    else:
        print(f"Warning: No season found for year {year}")
        return None

def get_team_id_by_name(db: DatabaseConnector, team_name: str) -> int:
    """Get team_id for the given team name"""
    result = db.fetch_all("SELECT team_id FROM nfl_team WHERE team_name = %s", (team_name,))
    if result:
        return result[0][0]
    else:
        print(f"Warning: No team found for name '{team_name}'")
        return None


def preprocess_player_data(db: DatabaseConnector, df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess player data for database insertion"""
    print("Preprocessing player data...")
    
    # Get season_id for the year
    season_id = get_season_id(db, YEAR)
    if season_id is None:
        print(f"Error: Could not find season_id for year {YEAR}")
        return pd.DataFrame()
    
    # Map CSV columns to database columns
    column_mapping = {
        'plyr_name': 'plyr_name',
        'current_team': 'current_team',  # Will be processed to team_id
        'pos': 'plyr_pos',
        'age': 'plyr_age',
        'gm_played': 'plyr_gm_played',
        'gm_started': 'plyr_gm_started',
        'weight': 'plyr_weight',
        'height': 'plyr_height',
        'yrs_played': 'plyr_yrs_played',
        'plyr_college': 'plyr_college',
        'plyr_birthdate': 'plyr_birthday',  # CSV has plyr_birthdate, DB expects plyr_birthday
        'plyr_avg_value': 'plyr_avg_value',
        'plyr_draft_tm': 'plyr_draft_tm',
        'plyr_draft_rd': 'plyr_draft_rd',
        'plyr_draft_pick': 'plyr_draft_pick',
        'plyr_draft_yr': 'plyr_draft_yr'
    }
    
    # Select and rename columns that exist in the CSV
    available_columns = {}
    for csv_col, db_col in column_mapping.items():
        if csv_col in df.columns:
            available_columns[csv_col] = db_col
    
    # Select only the columns we need
    df_processed = df[list(available_columns.keys())].copy()
    df_processed = df_processed.rename(columns=available_columns)
    
    # Add season_id to all rows
    df_processed['season_id'] = season_id
    
    # Convert birthdate format if the column exists
    if 'plyr_birthday' in df_processed.columns:
        print("Converting birthdate format...")
        df_processed['plyr_birthday'] = df_processed['plyr_birthday'].apply(convert_birthdate_format)
    
    # Handle team_id mapping
    if 'current_team' in df_processed.columns:
        print("Mapping team names to team_ids...")
        # Map team names directly to team_ids using team_name column in nfl_team table
        df_processed['team_id'] = df_processed['current_team'].apply(
            lambda x: get_team_id_by_name(db, x) if pd.notna(x) else None
        )
        # Drop the current_team column as we now have team_id
        df_processed = df_processed.drop(columns=['current_team'])
    else:
        print("Warning: No current_team column found. Setting team_id to NULL")
        df_processed['team_id'] = None
    
    # Handle null values
    df_processed = handle_null_values(df_processed)
    
    print(f"Preprocessed {len(df_processed)} player records")
    return df_processed

def insert_player_data(db: DatabaseConnector, df: pd.DataFrame) -> bool:
    """Insert/Update player data into the plyr table using UPSERT"""
    print("Processing player data (inserting new/updating existing)...")
    
    # Use batch upsert from db_utils
    success = batch_upsert_data(db, 'plyr', df, batch_size=500)
    
    if success:
        print("Player data processed successfully")
    else:
        print("Failed to process player data")
    
    return success

def main():
    """Main function to insert player data"""
    print("Player Insert Script")
    print("=" * 40)
    print(f"Processing data for Year: {YEAR}, Week: {WEEK}")
    
    # Construct the source directory path
    source_dir = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\plyr\\plyr_clean\\{WEEK}"
    print(f"Source directory: {source_dir}")
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory does not exist: {source_dir}")
        return
    
    # Find the most recent CSV file
    csv_file_path = get_most_recent_csv_file(source_dir)
    if not csv_file_path:
        print("No CSV file found to process")
        return
    
    db = DatabaseConnector()
    
    try:
        if not db.connect():
            print("Failed to connect to database")
            return
        
        # Create table if it doesn't exist
        if not create_plyr_table(db):
            return
        
        # Load data from CSV file
        print(f"Loading data from: {csv_file_path}")
        df = load_csv_data(csv_file_path)
        
        if df.empty:
            print("No data loaded from CSV file")
            return
        
        # Preprocess the data
        processed_df = preprocess_player_data(db, df)
        
        if processed_df.empty:
            print("No valid data after preprocessing")
            return
        
        # Insert player data
        if insert_player_data(db, processed_df):
            print("All player data processed successfully")
        else:
            print("Some player data failed to insert")
    
    except Exception as e:
        print(f"Error in main execution: {e}")
    
    finally:
        db.disconnect()

if __name__ == "__main__":
    main()