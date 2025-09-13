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


def create_adv_plyr_gm_rush_table(db: DatabaseConnector) -> bool:
    """Create the adv_plyr_gm_rush table if it doesn't exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS adv_plyr_gm_rush (
        adv_plyr_gm_rush_id INT PRIMARY KEY AUTO_INCREMENT,
        plyr_id INT,
        week_id INT,
        game_id INT,
        season_id INT,
        plyr_gm_rush_first_dwn INT,
        plyr_gm_rush_ybc INT,
        plyr_gm_rush_ybc_att FLOAT(7,4),
        plyr_gm_rush_yac INT,
        plyr_gm_rush_yac_att FLOAT(7,4),
        plyr_gm_rush_brkn_tkl INT,
        plyr_gm_rush_brkn_tkl_att FLOAT(7,4),
        FOREIGN KEY (plyr_id) REFERENCES plyr(plyr_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (game_id) REFERENCES nfl_game(game_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        UNIQUE KEY uk_plyr_gm_rush_adv (plyr_id, game_id)
    );
    """
    
    return create_table_if_not_exists(db, 'adv_plyr_gm_rush', create_table_sql)


def get_csv_files() -> list:
    """Get list of CSV files to process based on WEEK_START and WEEK_END."""
    base_dir = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\games"
    csv_files = []
    
    for week in range(WEEK_START, WEEK_END + 1):
        week_dir = os.path.join(base_dir, f"week_{week}.0")
        clean_dir = os.path.join(week_dir, "clean")
        if os.path.exists(clean_dir):
            pattern = os.path.join(clean_dir, "cleaned_*_gm_plyr_adv_rushing_*.csv")
            week_files = glob.glob(pattern)
            for file_path in week_files:
                csv_files.append((week, file_path))
        else:
            print(f"[WARNING] Clean directory not found: {clean_dir}")
    
    return sorted(csv_files)


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
            
            # Create processed row
            processed_row = {
                'plyr_id': plyr_id,
                'week_id': week_id,
                'game_id': game_id,
                'season_id': season_id
            }
            
            # Map database columns to CSV columns - confirmed mapping from database schema
            stat_mapping = {
                'plyr_gm_rush_first_dwn': '1D',           # First downs gained
                'plyr_gm_rush_ybc': 'YBC',                # Yards Before Contact
                'plyr_gm_rush_ybc_att': 'YBC/Att',        # YBC per Attempt
                'plyr_gm_rush_yac': 'YAC',                # Yards After Contact
                'plyr_gm_rush_yac_att': 'YAC/Att',        # YAC per Attempt
                'plyr_gm_rush_brkn_tkl': 'BrkTkl',        # Broken Tackles
                'plyr_gm_rush_brkn_tkl_att': 'Att/Br'     # Attempts per Broken Tackle
            }
            
            # NOTE: CSV columns 'Att', 'Yds', 'TD' are NOT mapped to this table
            # They belong in the basic plyr_gm_rush table
            
            # Add advanced rushing stats
            for db_col, csv_col in stat_mapping.items():
                if csv_col in df.columns:
                    value = row.get(csv_col)
                    
                    # Handle empty values specially - convert empty strings to None
                    if pd.notna(value) and value != '' and str(value).strip() != '':
                        try:
                            # Convert 1D from float to int (3.0 -> 3)
                            if db_col == 'plyr_gm_rush_first_dwn':
                                processed_row[db_col] = int(float(value))
                            # Handle float columns (ratios and averages)
                            elif db_col in ['plyr_gm_rush_ybc_att', 'plyr_gm_rush_yac_att', 'plyr_gm_rush_brkn_tkl_att']:
                                processed_row[db_col] = float(value)
                            # Handle integer columns (counts and yards)
                            else:
                                processed_row[db_col] = int(float(value))
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
        
        # Handle integer columns that can be null (using nullable integer type)
        nullable_int_columns = ['plyr_gm_rush_first_dwn', 'plyr_gm_rush_ybc', 
                               'plyr_gm_rush_yac', 'plyr_gm_rush_brkn_tkl']
        
        for col in nullable_int_columns:
            if col in processed_df.columns:
                # Convert to numeric, coercing errors to NaN
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                # For integer columns, round the values and convert to nullable int
                processed_df[col] = processed_df[col].round().astype('Int64')
        
        # Handle float columns (ratios that need precise decimal representation)
        float_columns = ['plyr_gm_rush_ybc_att', 'plyr_gm_rush_yac_att', 'plyr_gm_rush_brkn_tkl_att']
        
        for col in float_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        return processed_df
    else:
        return pd.DataFrame()


def main():
    """Main function to process all CSV files."""
    
    print(f"Starting Advanced Player Game Rushing Data Import for {YEAR}")
    print(f"Processing weeks {WEEK_START} to {WEEK_END}")
    
    # Initialize database connection
    db = DatabaseConnector()
    if not db.connect():
        print("[FAIL] Failed to connect to database")
        return
    
    try:
        # Create table if it doesn't exist
        if not create_adv_plyr_gm_rush_table(db):
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
                    success = batch_upsert_data(db, 'adv_plyr_gm_rush', processed_df)
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