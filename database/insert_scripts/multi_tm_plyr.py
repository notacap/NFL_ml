"""
Multi-Team Player Insert Script

Inserts multi-team player data into the multi_tm_plyr table from CSV files.
Processes players who have played for multiple teams in a season.
"""

import sys
import os
import glob
from datetime import datetime
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_utils import DatabaseConnector, YEAR, WEEK, load_csv_data, clean_column_names, handle_null_values, batch_upsert_data, get_or_create_player_guid

def create_multi_tm_plyr_table(db: DatabaseConnector) -> bool:
    """Create the multi_tm_plyr table if it doesn't exist"""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS multi_tm_plyr (
        multi_tm_plyr_id INT AUTO_INCREMENT PRIMARY KEY,
        plyr_id INT,
        season_id INT,
        current_tm_id INT,
        former_tm_id INT,
        first_tm_id INT,
        current_tm_week_id INT,
        former_tm_lst_wk_id INT,
        first_tm_lst_wk_id INT,
        second_tm_week_id INT,
        plyr_name VARCHAR(255) NOT NULL,
        plyr_pos VARCHAR(10),
        plyr_age INT,
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
        FOREIGN KEY (plyr_id) REFERENCES plyr(plyr_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        FOREIGN KEY (current_tm_id) REFERENCES nfl_team(team_id),
        FOREIGN KEY (former_tm_id) REFERENCES nfl_team(team_id),
        FOREIGN KEY (first_tm_id) REFERENCES nfl_team(team_id),
        FOREIGN KEY (current_tm_week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (former_tm_lst_wk_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (second_tm_week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (first_tm_lst_wk_id) REFERENCES nfl_week(week_id),
        UNIQUE KEY uk_player_identity (plyr_name, plyr_birthday, plyr_draft_tm, season_id)
    )
    """
    
    if db.execute_query(create_table_query):
        print("multi_tm_plyr table created/verified successfully")
        return True
    else:
        print("Failed to create multi_tm_plyr table")
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
    if not team_name or pd.isna(team_name):
        return None
    result = db.fetch_all("SELECT team_id FROM nfl_team WHERE team_name = %s", (team_name,))
    if result:
        return result[0][0]
    else:
        print(f"Warning: No team found for name '{team_name}'")
        return None

def get_week_id(db: DatabaseConnector, season_id: int, week_num: str) -> int:
    """Get week_id for the given season and week number"""
    if not week_num or pd.isna(week_num):
        return None
    result = db.fetch_all("SELECT week_id FROM nfl_week WHERE season_id = %s AND week_num = %s", (season_id, str(week_num)))
    if result:
        return result[0][0]
    else:
        print(f"Warning: No week found for season {season_id}, week {week_num}")
        return None

def get_player_id_from_guid(db: DatabaseConnector, player_guid: str, season_id: int) -> int:
    """Get plyr_id using player_guid.
    
    Returns:
        int: plyr_id for this player in this season
    """
    result = db.fetch_all(
        "SELECT plyr_id FROM plyr WHERE player_guid = %s AND season_id = %s",
        (player_guid, season_id)
    )
    
    if result:
        return result[0][0]
    else:
        raise ValueError(f"No plyr record found for GUID {player_guid[:8]}... in season {season_id}")

def is_multi_team_player(row) -> bool:
    """Check if a player has multi-team data based on temp.py logic"""
    multi_team_columns = [
        'former_team', 'current_team_week', 'former_team_last_week', 
        'first_team', 'former_team_first_week', 'first_team_last_week'
    ]
    
    # Check if any of the multi-team columns have values
    for col in multi_team_columns:
        if col in row and row[col] and not pd.isna(row[col]) and str(row[col]).strip():
            return True
    return False

def parse_weeks_string(weeks_str):
    """Parse the weeks string to extract week numbers"""
    if not weeks_str or pd.isna(weeks_str):
        return []
    
    try:
        # Remove quotes and split by comma
        weeks_str = str(weeks_str).strip().strip('"')
        if not weeks_str:
            return []
        
        week_numbers = []
        for week in weeks_str.split(','):
            week = week.strip()
            if week:
                week_numbers.append(int(week))
        return sorted(week_numbers)
    except Exception as e:
        print(f"Warning: Could not parse weeks string '{weeks_str}': {e}")
        return []

def preprocess_multi_team_data(db: DatabaseConnector, df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess multi-team player data for database insertion"""
    print("Preprocessing multi-team player data...")
    
    # Get season_id for the year
    season_id = get_season_id(db, YEAR)
    if season_id is None:
        print(f"Error: Could not find season_id for year {YEAR}")
        return pd.DataFrame()
    
    # Filter for multi-team players only
    multi_team_rows = []
    for idx, row in df.iterrows():
        if is_multi_team_player(row):
            multi_team_rows.append(row)
    
    if not multi_team_rows:
        print("No multi-team players found in the data")
        return pd.DataFrame()
    
    print(f"Found {len(multi_team_rows)} multi-team players")
    
    # Convert to DataFrame
    df_multi = pd.DataFrame(multi_team_rows)
    
    # Map CSV columns to database columns
    column_mapping = {
        'plyr_name': 'plyr_name',
        'current_team': 'current_team',  # Will be processed to current_tm_id
        'former_team': 'former_team',    # Will be processed to former_tm_id
        'first_team': 'first_team',      # Will be processed to first_tm_id
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
        'plyr_draft_yr': 'plyr_draft_yr',
        'current_team_week': 'current_team_week',
        'former_team_last_week': 'former_team_last_week',
        'first_team_last_week': 'first_team_last_week',
        'former_team_first_week': 'former_team_first_week'
    }
    
    # Select and rename columns that exist in the CSV
    available_columns = {}
    for csv_col, db_col in column_mapping.items():
        if csv_col in df_multi.columns:
            available_columns[csv_col] = db_col
    
    # Select only the columns we need
    df_processed = df_multi[list(available_columns.keys())].copy()
    df_processed = df_processed.rename(columns=available_columns)
    
    # Add season_id to all rows
    df_processed['season_id'] = season_id
    
    # Convert birthdate format if the column exists
    if 'plyr_birthday' in df_processed.columns:
        print("Converting birthdate format...")
        df_processed['plyr_birthday'] = df_processed['plyr_birthday'].apply(convert_birthdate_format)
    
    # Generate player_guids
    print("Generating player GUIDs...")
    player_guids = []
    for idx, row in df_processed.iterrows():
        player_guid = get_or_create_player_guid(
            db,
            plyr_name=row['plyr_name'],
            plyr_birthday=row.get('plyr_birthday'),
            plyr_draft_tm=row.get('plyr_draft_tm'),
            plyr_height=row.get('plyr_height'),
            plyr_college=row.get('plyr_college'),
            plyr_draft_rd=row.get('plyr_draft_rd'),
            plyr_draft_pick=row.get('plyr_draft_pick'),
            plyr_draft_yr=row.get('plyr_draft_yr'),
            primary_pos=row.get('plyr_pos')
        )
        player_guids.append(player_guid)
    
    df_processed['player_guid'] = player_guids
    
    # Get plyr_id for each player
    print("Looking up player IDs...")
    plyr_ids = []
    for idx, row in df_processed.iterrows():
        plyr_id = get_player_id_from_guid(db, row['player_guid'], season_id)
        plyr_ids.append(plyr_id)
    
    df_processed['plyr_id'] = plyr_ids
    
    # Remove rows where we couldn't find the player_id
    initial_count = len(df_processed)
    df_processed = df_processed[df_processed['plyr_id'].notna()]
    final_count = len(df_processed)
    
    if initial_count > final_count:
        print(f"Removed {initial_count - final_count} rows due to missing player IDs")
    
    # Drop columns now in plyr_master
    columns_to_drop = ['plyr_name', 'plyr_birthday', 'plyr_pos', 'plyr_age', 
                       'plyr_weight', 'plyr_height', 'plyr_yrs_played', 'plyr_college',
                       'plyr_avg_value', 'plyr_draft_tm', 'plyr_draft_rd', 
                       'plyr_draft_pick', 'plyr_draft_yr']
    df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns])
    
    # Handle team_id mappings
    team_columns = [
        ('current_team', 'current_tm_id'),
        ('former_team', 'former_tm_id'),
        ('first_team', 'first_tm_id')
    ]
    
    for csv_col, db_col in team_columns:
        if csv_col in df_processed.columns:
            print(f"Mapping {csv_col} to {db_col}...")
            df_processed[db_col] = df_processed[csv_col].apply(
                lambda x: get_team_id_by_name(db, x) if pd.notna(x) and str(x).strip() else None
            )
            # Drop the original team column
            df_processed = df_processed.drop(columns=[csv_col])
    
    # Handle week_id mappings
    week_columns = [
        ('current_team_week', 'current_tm_week_id'),
        ('former_team_last_week', 'former_tm_lst_wk_id'),
        ('first_team_last_week', 'first_tm_lst_wk_id'),
        ('former_team_first_week', 'second_tm_week_id')
    ]
    
    for csv_col, db_col in week_columns:
        if csv_col in df_processed.columns:
            print(f"Mapping {csv_col} to {db_col}...")
            df_processed[db_col] = df_processed[csv_col].apply(
                lambda x: get_week_id(db, season_id, x) if pd.notna(x) and str(x).strip() else None
            )
            # Drop the original week column
            df_processed = df_processed.drop(columns=[csv_col])
    
    # Handle null values
    df_processed = handle_null_values(df_processed)
    
    print(f"Preprocessed {len(df_processed)} multi-team player records")
    return df_processed

def insert_multi_team_data(db: DatabaseConnector, df: pd.DataFrame) -> bool:
    """Insert/Update multi-team player data into the multi_tm_plyr table using UPSERT"""
    print("Processing multi-team player data (inserting new/updating existing)...")
    
    # Use batch upsert from db_utils
    success = batch_upsert_data(db, 'multi_tm_plyr', df, batch_size=100)
    
    if success:
        print("Multi-team player data processed successfully")
    else:
        print("Failed to process multi-team player data")
    
    return success

def main():
    """Main function to insert multi-team player data"""
    print("Multi-Team Player Insert Script")
    print("=" * 40)
    print(f"Processing data for Year: {YEAR}, Week: {WEEK}")
    
    # Construct the source directory path (same as plyr.py)
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
        if not create_multi_tm_plyr_table(db):
            return
        
        # Load data from CSV file
        print(f"Loading data from: {csv_file_path}")
        df = load_csv_data(csv_file_path)
        
        if df.empty:
            print("No data loaded from CSV file")
            return
        
        # Preprocess the data
        processed_df = preprocess_multi_team_data(db, df)
        
        if processed_df.empty:
            print("No valid multi-team data after preprocessing")
            return
        
        # Insert multi-team player data
        if insert_multi_team_data(db, processed_df):
            print("All multi-team player data processed successfully")
        else:
            print("Some multi-team player data failed to insert")
    
    except Exception as e:
        print(f"Error in main execution: {e}")
    
    finally:
        db.disconnect()

if __name__ == "__main__":
    main()