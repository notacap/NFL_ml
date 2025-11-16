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
from db_utils import DatabaseConnector, YEAR, WEEK, load_csv_data, clean_column_names, handle_null_values, batch_upsert_data, get_or_create_player_guid, check_birthday_tolerance, prompt_user_birthday_resolution

def create_plyr_master_table(db: DatabaseConnector) -> bool:
    """Create the plyr_master table if it doesn't exist"""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS plyr_master (
        plyr_guid VARCHAR(64) PRIMARY KEY,
        plyr_name VARCHAR(255) NOT NULL,
        plyr_birthday DATE,
        plyr_college VARCHAR(255),
        plyr_draft_tm VARCHAR(255),
        plyr_draft_rd TINYINT UNSIGNED,
        plyr_draft_pick SMALLINT UNSIGNED,
        plyr_draft_yr SMALLINT UNSIGNED,
        primary_pos VARCHAR(10),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_plyr_name (plyr_name),
        INDEX idx_plyr_draft_yr (plyr_draft_yr),
        INDEX idx_primary_pos (primary_pos)
    )
    """
    
    if db.execute_query(create_table_query):
        print("plyr_master table created/verified successfully")
        return True
    else:
        print("Failed to create plyr_master table")
        return False

def create_plyr_table(db: DatabaseConnector) -> bool:
    """Create the plyr table if it doesn't exist"""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS plyr (
        plyr_id INT AUTO_INCREMENT PRIMARY KEY,
        plyr_guid VARCHAR(64),
        team_id INT,
        season_id INT,
        plyr_name VARCHAR(255) NOT NULL,
        plyr_age TINYINT UNSIGNED,
        plyr_pos VARCHAR(10),
        plyr_alt_pos VARCHAR(10),
        plyr_gm_played TINYINT UNSIGNED,
        plyr_gm_started TINYINT UNSIGNED,
        plyr_weight SMALLINT UNSIGNED,
        plyr_height SMALLINT UNSIGNED,
        plyr_yrs_played TINYINT UNSIGNED,
        plyr_birthday DATE NOT NULL,
        plyr_avg_value TINYINT,
        plyr_draft_tm VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (plyr_guid) REFERENCES plyr_master(plyr_guid),
        FOREIGN KEY (team_id) REFERENCES nfl_team(team_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        INDEX idx_plyr_name (plyr_name),
        INDEX idx_team_season (team_id, season_id),
        INDEX idx_plyr_pos (plyr_pos),
        UNIQUE KEY uk_player_season_pos_exp (plyr_name, season_id, plyr_birthday, plyr_draft_tm)
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
    
    # Map CSV columns to database columns (includes plyr_master columns)
    column_mapping = {
        'plyr_name': 'plyr_name',
        'current_team': 'current_team',  
        'pos': 'plyr_pos',
        'age': 'plyr_age',
        'gm_played': 'plyr_gm_played',
        'gm_started': 'plyr_gm_started',
        'weight': 'plyr_weight',
        'height': 'plyr_height',
        'yrs_played': 'plyr_yrs_played',
        'plyr_birthdate': 'plyr_birthday',
        'plyr_avg_value': 'plyr_avg_value',
        'plyr_draft_tm': 'plyr_draft_tm',
        'plyr_college': 'plyr_college',
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
    
    # Check for birthday tolerance issues in plyr table and generate player_guids
    print("Checking for birthday discrepancies and generating player GUIDs...")
    player_guids = []
    
    for idx, row in df_processed.iterrows():
        plyr_name = row['plyr_name']
        plyr_birthday = None if pd.isna(row.get('plyr_birthday')) else row.get('plyr_birthday')
        plyr_draft_tm = None if pd.isna(row.get('plyr_draft_tm')) else row.get('plyr_draft_tm')
        
        # Check for birthday tolerance match in plyr table
        tolerance_match = check_birthday_tolerance(
            db, plyr_name, plyr_birthday, plyr_draft_tm,
            season_id=season_id, table_name='plyr'
        )
        
        if tolerance_match:
            csv_data = {
                'plyr_name': plyr_name,
                'plyr_birthday': plyr_birthday,
                'plyr_draft_tm': plyr_draft_tm,
                'season_id': season_id
            }
            
            is_same_player, correct_birthday = prompt_user_birthday_resolution(csv_data, tolerance_match)
            
            if is_same_player:
                df_processed.at[idx, 'plyr_birthday'] = correct_birthday
                plyr_birthday = correct_birthday
                
                if correct_birthday != tolerance_match['db_birthday']:
                    update_query = """
                        UPDATE plyr 
                        SET plyr_birthday = %s 
                        WHERE plyr_id = %s
                    """
                    db.execute_query(update_query, (correct_birthday, tolerance_match['plyr_id']))
                    print(f"Updated plyr table birthday to {correct_birthday}")
        
        # Get or create player_guid (this also checks plyr_master table)
        player_guid = get_or_create_player_guid(
            db,
            plyr_name=plyr_name,
            plyr_birthday=plyr_birthday,
            plyr_draft_tm=plyr_draft_tm,
            plyr_college=None if pd.isna(row.get('plyr_college')) else row.get('plyr_college'),
            plyr_draft_rd=None if pd.isna(row.get('plyr_draft_rd')) else row.get('plyr_draft_rd'),
            plyr_draft_pick=None if pd.isna(row.get('plyr_draft_pick')) else row.get('plyr_draft_pick'),
            plyr_draft_yr=None if pd.isna(row.get('plyr_draft_yr')) else row.get('plyr_draft_yr'),
            primary_pos=None if pd.isna(row.get('plyr_pos')) else row.get('plyr_pos')
        )
        player_guids.append(player_guid)
    
    df_processed['plyr_guid'] = player_guids
    
    print(f"Preprocessed {len(df_processed)} player records")
    return df_processed

def insert_player_data(db: DatabaseConnector, df: pd.DataFrame) -> bool:
    """Insert/Update player data into the plyr table.
    
    Uses batch upsert to update all mutable fields (pos, weight, yrs_played, etc.)
    while the unique key (plyr_name, plyr_birthday, plyr_draft_tm, season_id) remains immutable.
    """
    print("Processing player data (inserting new/updating existing)...")

    # Filter out columns that belong to plyr_master only (not plyr table)
    plyr_master_only_cols = ['plyr_college', 'plyr_draft_rd', 'plyr_draft_pick', 'plyr_draft_yr']
    df_plyr = df.drop(columns=[col for col in plyr_master_only_cols if col in df.columns])

    # Use batch upsert from db_utils
    success = batch_upsert_data(db, 'plyr', df_plyr, batch_size=500)

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
        
        # Create tables if they don't exist
        if not create_plyr_master_table(db):
            return
        
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