#!/usr/bin/env python3
"""
NFL Player Red Zone Receiving Stats Insert Script (plyr_rz_rec.py)

This script processes season-level NFL player red zone receiving statistics.
The script inserts/upserts player red zone receiving stats into the plyr_rz_rec table.

Red zone is defined as inside the 20-yard line.
Target zone is defined as inside the 10-yard line (goal line area).
"""

import sys
import os
import glob
import pandas as pd
import argparse
from datetime import datetime

# Add parent directory to path to import db_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_utils import (
    DatabaseConnector, 
    YEAR, WEEK, 
    get_season_id, get_week_id, get_player_id, 
    create_table_if_not_exists, batch_upsert_data, handle_null_values
)

def create_plyr_rz_rec_table(db: DatabaseConnector) -> bool:
    """Create plyr_rz_rec table if it doesn't exist"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS plyr_rz_rec (
        plyr_rz_rec_id INT AUTO_INCREMENT PRIMARY KEY,
        plyr_id INT,
        season_id INT,
        week_id INT,
        plyr_rec_rz_tgt INT,
        plyr_rec_rz INT,
        plyr_rec_rz_catch_pct DECIMAL(5,4),
        plyr_rec_rz_yds INT,
        plyr_rec_rz_td INT,
        plyr_rec_rz_tgt_pct DECIMAL(5,4),
        plyr_rec_tz_tgt INT,
        plyr_rec_tz INT,
        plyr_rec_tz_catch_pct DECIMAL(5,4),
        plyr_rec_tz_yds INT,
        plyr_rec_tz_td INT,
        plyr_rec_tz_tgt_pct DECIMAL(5,4),
        UNIQUE KEY uk_plyr_season (plyr_id, season_id, week_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (plyr_id) REFERENCES plyr(plyr_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id)
    )
    """
    return create_table_if_not_exists(db, "plyr_rz_rec", create_table_sql)

def get_most_recent_csv_file(directory: str) -> str:
    """Get the most recently created CSV file from directory"""
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    
    # Get the most recent file based on creation time
    most_recent = max(csv_files, key=os.path.getctime)
    return most_recent

def load_red_zone_receiving_stats(year: int, week: int) -> pd.DataFrame:
    """Load red zone receiving stats CSV file"""
    directory = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{year}\\plyr_rz_receiving\\week_{week}\\clean"
    file_path = get_most_recent_csv_file(directory)
    
    print(f"[INFO] Loading red zone receiving stats from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Column mapping from CSV to our processing format
    column_mapping = {
        'Player': 'player_name',
        'Tm': 'team',
        # Red Zone (Inside 20) Statistics
        'Inside 20 Tgt': 'plyr_rec_rz_tgt',
        'Inside 20 Rec': 'plyr_rec_rz', 
        'Inside 20 Ctch%': 'plyr_rec_rz_catch_pct',
        'Inside 20 Yds': 'plyr_rec_rz_yds',
        'Inside 20 TD': 'plyr_rec_rz_td',
        'Inside 20 %Tgt': 'plyr_rec_rz_tgt_pct',
        # Target Zone (Inside 10) Statistics
        'Inside 10 Tgt': 'plyr_rec_tz_tgt',
        'Inside 10 Rec': 'plyr_rec_tz',
        'Inside 10 Ctch%': 'plyr_rec_tz_catch_pct',
        'Inside 10 Yds': 'plyr_rec_tz_yds',
        'Inside 10 TD': 'plyr_rec_tz_td',
        'Inside 10 %Tgt': 'plyr_rec_tz_tgt_pct'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Select only the columns we need
    required_columns = ['player_name', 'team'] + [col for col in column_mapping.values() if col not in ['player_name', 'team']]
    available_columns = [col for col in required_columns if col in df.columns]
    df = df[available_columns]
    
    # Convert percentages to decimals
    for pct_col in ['plyr_rec_rz_catch_pct', 'plyr_rec_rz_tgt_pct', 'plyr_rec_tz_catch_pct', 'plyr_rec_tz_tgt_pct']:
        if pct_col in df.columns:
            # Handle empty strings and convert percentage to decimal
            def convert_percentage(value):
                if pd.isna(value) or (isinstance(value, str) and value.strip() == ''):
                    return None
                if isinstance(value, str):
                    # Handle special case ".0%" 
                    if value == ".0%":
                        return 0.0
                    # Remove % and convert to decimal
                    try:
                        return float(value.rstrip('%')) / 100.0
                    except (ValueError, TypeError):
                        return None
                try:
                    return float(value) / 100.0 if value > 1 else float(value)
                except (ValueError, TypeError):
                    return None
            
            df[pct_col] = df[pct_col].apply(convert_percentage)
    
    print(f"[INFO] Loaded {len(df)} red zone receiving records")
    return df

def process_player_red_zone_data(db: DatabaseConnector, df: pd.DataFrame, season_id: int, week_id: int, interactive: bool = False) -> pd.DataFrame:
    """Process player red zone receiving data for database insertion"""
    
    processed_data = []
    error_count = 0
    success_count = 0
    skipped_count = 0
    
    for index, row in df.iterrows():
        try:
            # Skip League Average rows or any non-player entries
            if pd.isna(row['player_name']) or row['player_name'] == 'League Average':
                continue
            
            # Handle multi-team players (2TM, 3TM) by searching without team filter
            if row['team'] in ['2TM', '3TM']:
                # Get potential player matches
                try:
                    # For multi-team players, we can't use team to disambiguate
                    # If multiple matches found, will need manual selection
                    player_id = get_player_id(
                        db, 
                        row['player_name'], 
                        '', # No team filter for multi-team players
                        season_id
                    )
                except ValueError as e:
                    if "multiple matches" in str(e).lower():
                        # For multi-team players, we need manual selection since we can't use team
                        if interactive:
                            print(f"[INFO] Multiple matches found for multi-team player {row['player_name']}")
                            print(f"[INFO] Please run script in interactive mode to select correct player")
                        print(f"[ERROR] Multiple matches found for multi-team player {row['player_name']} - use interactive mode or check player lookup")
                        error_count += 1
                        continue
                    else:
                        print(f"[ERROR] Could not find player {row['player_name']} ({row['team']}): {e}")
                        error_count += 1
                        continue
            else:
                # Get player ID using team for disambiguation
                try:
                    player_id = get_player_id(
                        db, 
                        row['player_name'], 
                        row['team'], 
                        season_id
                    )
                except ValueError as e:
                    print(f"[ERROR] Could not find player {row['player_name']} ({row['team']}): {e}")
                    error_count += 1
                    continue
            
            # Prepare data row for database insertion
            data_row = {
                'plyr_id': player_id,
                'season_id': season_id,
                'week_id': week_id
            }
            
            # Map all red zone receiving stat columns
            stat_columns = [
                'plyr_rec_rz_tgt', 'plyr_rec_rz', 'plyr_rec_rz_catch_pct', 
                'plyr_rec_rz_yds', 'plyr_rec_rz_td', 'plyr_rec_rz_tgt_pct',
                'plyr_rec_tz_tgt', 'plyr_rec_tz', 'plyr_rec_tz_catch_pct',
                'plyr_rec_tz_yds', 'plyr_rec_tz_td', 'plyr_rec_tz_tgt_pct'
            ]
            
            for col in stat_columns:
                if col in row and pd.notnull(row[col]):
                    # Handle different data types
                    value = row[col]
                    if pd.isna(value):
                        data_row[col] = None
                    elif isinstance(value, str) and value.strip() == '':
                        data_row[col] = None
                    else:
                        # Convert to appropriate type
                        if col in ['plyr_rec_rz_catch_pct', 'plyr_rec_rz_tgt_pct', 'plyr_rec_tz_catch_pct', 'plyr_rec_tz_tgt_pct']:
                            # These are already converted to decimal in load function
                            data_row[col] = float(value) if value is not None else None
                        else:
                            # Integer columns
                            data_row[col] = int(float(value)) if value != '' else None
                else:
                    data_row[col] = None
            
            processed_data.append(data_row)
            success_count += 1
            
        except Exception as e:
            print(f"[ERROR] Failed to process player {row['player_name']} ({row['team']}): {e}")
            error_count += 1
            continue
    
    print(f"[INFO] Processed {success_count} players successfully, {error_count} errors, {skipped_count} skipped")
    
    # Convert to DataFrame
    final_df = pd.DataFrame(processed_data)
    return handle_null_values(final_df)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Insert NFL player red zone receiving statistics into database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plyr_rz_rec.py                    # Run in standard mode (auto-select first match)
  python plyr_rz_rec.py --interactive      # Run in interactive mode (manual player selection)  
        """
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Enable interactive mode for manual player selection when multiple matches are found'
    )
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()
    
    print(f"[INFO] Starting plyr_rz_rec.py script for {YEAR} week {WEEK}")
    print(f"[INFO] Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.interactive:
        print("[INFO] Interactive mode enabled - you will be prompted for player selection when multiple matches are found")
    else:
        print("[INFO] Standard mode - script will automatically select first match for multiple player matches")
    
    # Initialize database connection
    db = DatabaseConnector()
    
    try:
        # Connect to database
        if not db.connect():
            print("[ERROR] Failed to connect to database")
            return False
        
        print("[INFO] Connected to database successfully")
        
        # Create table if needed
        if not create_plyr_rz_rec_table(db):
            print("[ERROR] Failed to create plyr_rz_rec table")
            return False
        
        # Get foreign key IDs
        season_id = get_season_id(db, YEAR)
        week_id = get_week_id(db, season_id, WEEK)
        
        print(f"[INFO] Using season_id: {season_id}, week_id: {week_id}")
        
        # Load CSV data
        df = load_red_zone_receiving_stats(YEAR, WEEK)
        
        # Process data for database insertion
        processed_df = process_player_red_zone_data(db, df, season_id, week_id, args.interactive)
        
        if processed_df.empty:
            print("[WARNING] No data to insert")
            return False
        
        # Insert/upsert data
        success = batch_upsert_data(db, 'plyr_rz_rec', processed_df)
        
        if success:
            print(f"[SUCCESS] plyr_rz_rec.py completed successfully!")
            print(f"[INFO] Processed {len(processed_df)} player red zone receiving stat records")
            return True
        else:
            print("[ERROR] Failed to insert data into plyr_rz_rec table")
            return False
            
    except Exception as e:
        print(f"[ERROR] Script execution failed: {e}")
        return False
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)