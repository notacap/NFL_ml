#!/usr/bin/env python3

import os
import sys
import pandas as pd
import glob
from pathlib import Path

# Add the parent directory to the Python path to import db_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_utils import (
    DatabaseConnector, YEAR, WEEK_START, WEEK_END, 
    batch_upsert_data, handle_null_values, 
    get_season_id, get_week_id, get_game_id, get_player_id,
    create_table_if_not_exists
)


def create_adv_plyr_gm_rec_table(db: DatabaseConnector) -> bool:
    """Create the adv_plyr_gm_rec table if it doesn't exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS adv_plyr_gm_rec (
        adv_plyr_gm_rec_id INT PRIMARY KEY AUTO_INCREMENT,
        plyr_id INT,
        week_id INT,
        game_id INT,
        season_id INT,
        plyr_gm_rec_first_dwn INT,
        plyr_gm_rec_aybc INT,
        plyr_gm_rec_aybc_route FLOAT(7,4),
        plyr_gm_rec_yac INT,
        plyr_gm_rec_yac_route FLOAT(7,4),
        plyr_gm_rec_adot FLOAT(7,4),
        plyr_gm_rec_brkn_tkl INT,
        plyr_gm_rec_brkn_tkl_rec FLOAT(7,4),
        plyr_gm_rec_drp INT,
        plyr_gm_rec_drp_pct FLOAT(7,4),
        plyr_gm_rec_int INT,
        plyr_gm_rec_pass_rtg FLOAT(7,4),
        FOREIGN KEY (plyr_id) REFERENCES plyr(plyr_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (game_id) REFERENCES nfl_game(game_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        UNIQUE KEY uk_player_game (plyr_id, game_id)
    );
    """
    
    return create_table_if_not_exists(db, 'adv_plyr_gm_rec', create_table_sql)


def get_csv_files() -> list:
    """Get list of CSV files to process based on WEEK_START and WEEK_END."""
    base_dir = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\games"
    csv_files = []
    
    for week in range(WEEK_START, WEEK_END + 1):
        week_dir = os.path.join(base_dir, f"week_{week}.0")
        clean_dir = os.path.join(week_dir, "clean")
        if os.path.exists(clean_dir):
            pattern = os.path.join(clean_dir, "cleaned_*_gm_plyr_adv_receiving_*.csv")
            week_files = glob.glob(pattern)
            for file_path in week_files:
                csv_files.append((week, file_path))
        else:
            print(f"[WARNING] Clean directory not found: {clean_dir}")
    
    return sorted(csv_files)


def convert_percentage_to_decimal(value) -> float:
    """Convert percentage string or decimal to decimal format."""
    if pd.isna(value) or value == '' or str(value).strip() == '':
        return None
    
    try:
        value_str = str(value).strip()
        if value_str.endswith('%'):
            # Remove % and convert to decimal (divide by 100)
            return float(value_str[:-1]) / 100.0
        else:
            # Assume it's already a decimal or percentage as number
            float_val = float(value_str)
            # If the value is > 1, assume it's a percentage (e.g., 75.0 means 75%)
            if float_val > 1.0:
                return float_val / 100.0
            return float_val
    except (ValueError, TypeError):
        return None


def handle_empty_ratio(value) -> float:
    """Handle empty strings and ratios that might be undefined."""
    if pd.isna(value) or value == '' or str(value).strip() == '':
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def process_csv_file(db: DatabaseConnector, file_path: str, season_id: int) -> pd.DataFrame:
    """Process a single CSV file and return processed DataFrame."""
    
    print(f"Processing file: {os.path.basename(file_path)}")
    
    # Read CSV with single header row
    df = pd.read_csv(file_path, header=0)
    
    # Get week number from the 'week' column
    if 'week' in df.columns:
        week_num = df['week'].iloc[0]
    else:
        raise ValueError(f"Week column not found in file {file_path}. Available columns: {list(df.columns)}")
    
    if pd.isna(week_num):
        raise ValueError(f"Week number is null in file {file_path}")
    
    # Convert to int (week_num might be 1.0, 2.0, etc.)
    week_num = int(float(week_num))
    
    week_id = get_week_id(db, season_id, week_num)
    
    # Get unique team abbreviations from the Tm column
    unique_teams = df['Tm'].dropna().unique()
    if len(unique_teams) != 2:
        raise ValueError(f"Expected 2 teams in CSV file, found {len(unique_teams)}: {unique_teams}")
    
    team1_abrv, team2_abrv = unique_teams[0], unique_teams[1]
    game_id = get_game_id(db, season_id, week_id, team1_abrv, team2_abrv)
    
    # Process each row
    processed_rows = []
    for index, row in df.iterrows():
        try:
            # Get player name and team from row
            player_name = row.get('Player', '')
            team_abrv = row.get('Tm', '')
            
            if not player_name or player_name.strip() == '':
                continue
                
            # Skip rows where Tm column is null, empty, or just whitespace
            if pd.isna(team_abrv) or not team_abrv or str(team_abrv).strip() == '':
                print(f"[INFO] Skipping player {player_name} - no team information (Tm column is empty)")
                continue
                
            # Get player_id
            plyr_id = get_player_id(db, player_name, team_abrv, season_id)
            
            # Create processed row with foreign keys
            processed_row = {
                'plyr_id': plyr_id,
                'week_id': week_id,
                'game_id': game_id,
                'season_id': season_id
            }
            
            # Map database columns to CSV columns based on actual schema
            stat_mapping = {
                'plyr_gm_rec_first_dwn': '1D',
                'plyr_gm_rec_aybc': 'YBC',  # Air Yards Before Catch
                'plyr_gm_rec_aybc_route': 'YBC/R',  # YBC per Route
                'plyr_gm_rec_yac': 'YAC',
                'plyr_gm_rec_yac_route': 'YAC/R',  # YAC per Route
                'plyr_gm_rec_adot': 'ADOT',
                'plyr_gm_rec_brkn_tkl': 'BrkTkl',
                'plyr_gm_rec_brkn_tkl_rec': 'Rec/Br',  # Special handling for empty strings
                'plyr_gm_rec_drp': 'Drop',
                'plyr_gm_rec_drp_pct': 'Drop%',  # Special percentage handling
                'plyr_gm_rec_int': 'Int',
                'plyr_gm_rec_pass_rtg': 'Rat'
            }
            
            # Process each statistical column
            for db_col, csv_col in stat_mapping.items():
                if csv_col in df.columns:
                    value = row.get(csv_col)
                    
                    # Handle percentage columns specially (convert from percentage to decimal)
                    if db_col in ['plyr_gm_rec_drp_pct']:
                        # Drop% comes as percentage (e.g., 16.7) but needs to be decimal (0.167)
                        if pd.notna(value) and value != '' and str(value).strip() != '':
                            try:
                                processed_row[db_col] = float(value) / 100.0
                            except (ValueError, TypeError):
                                processed_row[db_col] = None
                        else:
                            processed_row[db_col] = None
                    # Handle ratio/float columns that can be empty strings
                    elif db_col in ['plyr_gm_rec_brkn_tkl_rec', 'plyr_gm_rec_aybc_route', 'plyr_gm_rec_yac_route', 
                                   'plyr_gm_rec_adot', 'plyr_gm_rec_pass_rtg']:
                        processed_row[db_col] = handle_empty_ratio(value)
                    # Handle integer columns
                    elif db_col in ['plyr_gm_rec_first_dwn', 'plyr_gm_rec_aybc', 'plyr_gm_rec_yac', 
                                   'plyr_gm_rec_brkn_tkl', 'plyr_gm_rec_drp', 'plyr_gm_rec_int']:
                        if pd.notna(value) and value != '' and str(value).strip() != '':
                            try:
                                processed_row[db_col] = int(float(value))
                            except (ValueError, TypeError):
                                processed_row[db_col] = None
                        else:
                            processed_row[db_col] = None
                    # Handle other float columns
                    else:
                        if pd.notna(value) and value != '' and str(value).strip() != '':
                            try:
                                processed_row[db_col] = float(value)
                            except (ValueError, TypeError):
                                processed_row[db_col] = None
                        else:
                            processed_row[db_col] = None
            
            processed_rows.append(processed_row)
            
        except Exception as e:
            print(f"[WARNING] Error processing row {index} for player {row.get('Player', 'Unknown')}: {e}")
            continue
    
    if processed_rows:
        processed_df = pd.DataFrame(processed_rows)
        processed_df = handle_null_values(processed_df)
        
        # Ensure proper data types for foreign key columns (these must not be null)
        fk_columns = ['plyr_id', 'week_id', 'game_id', 'season_id']
        for col in fk_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').astype('int64')
        
        # Handle integer columns that can be null
        nullable_int_columns = ['plyr_gm_rec_first_dwn', 'plyr_gm_rec_aybc', 'plyr_gm_rec_yac', 
                               'plyr_gm_rec_brkn_tkl', 'plyr_gm_rec_drp', 'plyr_gm_rec_int']
        
        for col in nullable_int_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                processed_df[col] = processed_df[col].round().astype('Int64')
        
        # Handle decimal/float columns
        float_columns = ['plyr_gm_rec_aybc_route', 'plyr_gm_rec_yac_route', 'plyr_gm_rec_adot', 
                        'plyr_gm_rec_brkn_tkl_rec', 'plyr_gm_rec_drp_pct', 'plyr_gm_rec_pass_rtg']
        
        for col in float_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        return processed_df
    else:
        return pd.DataFrame()


def main():
    """Main function to process all CSV files."""
    
    print(f"Starting Advanced Player Game Receiving Data Import for {YEAR}")
    print(f"Processing weeks {WEEK_START} to {WEEK_END}")
    
    # Initialize database connection
    db = DatabaseConnector()
    if not db.connect():
        print("[FAIL] Failed to connect to database")
        return
    
    try:
        # Create table if it doesn't exist
        if not create_adv_plyr_gm_rec_table(db):
            return
        
        # Get season_id
        season_id = get_season_id(db, YEAR)
        print(f"Season ID for {YEAR}: {season_id}")
        
        # Get CSV files to process
        csv_files = get_csv_files()
        if not csv_files:
            print("[WARNING] No CSV files found to process")
            return
        
        print(f"Found {len(csv_files)} CSV files to process")
        
        total_processed = 0
        total_inserted = 0
        
        # Process each file
        for week, file_path in csv_files:
            try:
                processed_df = process_csv_file(db, file_path, season_id)
                
                if not processed_df.empty:
                    success = batch_upsert_data(db, 'adv_plyr_gm_rec', processed_df)
                    if success:
                        rows_in_file = len(processed_df)
                        total_processed += rows_in_file
                        total_inserted += rows_in_file
                        print(f"[OK] Week {week}: Processed {rows_in_file} player records")
                    else:
                        print(f"[FAIL] Week {week}: Failed to insert data")
                else:
                    print(f"[WARNING] Week {week}: No data to process")
                    
            except Exception as e:
                print(f"[ERROR] Error processing week {week} file {os.path.basename(file_path)}: {e}")
                continue
        
        print(f"\nImport Summary:")
        print(f"   Total files processed: {len(csv_files)}")
        print(f"   Total player records processed: {total_processed}")
        print(f"   Total records inserted/updated: {total_inserted}")
        
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
    finally:
        db.disconnect()


if __name__ == "__main__":
    main()