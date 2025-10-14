#!/usr/bin/env python3
"""
NFL Player Receiving Stats Insert Script (plyr_rec.py)

This script processes season-level NFL player receiving statistics from two source files:
1. plyr_receiving CSV: Basic receiving stats (targets, receptions, yards, TDs, etc.)
2. plyr_adv_receiving CSV: Advanced receiving stats (AYBC, YAC, ADOT, drops, etc.)

The script consolidates player stats from both files and inserts/upserts them into the plyr_rec table.
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
    get_season_id, get_week_id, get_season_player_id, 
    create_table_if_not_exists, batch_upsert_data, batch_upsert_data_with_logging, handle_null_values,
    apply_position_mapping
)

def create_plyr_rec_table(db: DatabaseConnector) -> bool:
    """Create plyr_rec table if it doesn't exist"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS plyr_rec (
        plyr_rec_id INT AUTO_INCREMENT PRIMARY KEY,
        plyr_id INT,
        season_id INT,
        week_id INT,
        plyr_rec_tgt INT,
        plyr_rec INT,
        plyr_rec_catch_pct DECIMAL(6,4),
        plyr_rec_yds INT,
        plyr_rec_yds_rec DECIMAL(7,4),
        plyr_rec_td INT,
        plyr_rec_first_dwn INT,
        plyr_rec_succ_rt DECIMAL(7,4),
        plyr_rec_lng INT,
        plyr_rec_yds_tgt DECIMAL(7,4),
        plyr_rec_gm DECIMAL(7,4),
        plyr_rec_ypg DECIMAL(7,4),
        plyr_rec_fmbl INT,
        plyr_rec_aybc INT,
        plyr_rec_aybc_rec DECIMAL(7,4),
        plyr_rec_yac INT,
        plyr_rec_yac_rec DECIMAL(7,4),
        plyr_rec_adot DECIMAL(7,4),
        plyr_rec_brkn_tkl INT,
        plyr_rec_brkn_tkl_rec DECIMAL(7,4),
        plyr_rec_drp INT,
        plyr_rec_drp_pct DECIMAL(5,4),
        plyr_rec_int INT,
        plyr_rec_pass_rtg DECIMAL(7,4),
        UNIQUE KEY uk_plyr_season (plyr_id, season_id, week_id),
        FOREIGN KEY (plyr_id) REFERENCES plyr(plyr_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id)
    )
    """
    return create_table_if_not_exists(db, "plyr_rec", create_table_sql)


def get_most_recent_csv_file(directory: str) -> str:
    """Get the most recently created CSV file from directory"""
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    
    # Get the most recent file based on creation time
    most_recent = max(csv_files, key=os.path.getctime)
    return most_recent

def load_basic_receiving_stats(year: int, week: int) -> pd.DataFrame:
    """Load basic receiving stats CSV file"""
    directory = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{year}\\plyr_receiving\\week_{week}\\clean"
    file_path = get_most_recent_csv_file(directory)
    
    print(f"[INFO] Loading basic receiving stats from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Column mapping from CSV to our processing format
    column_mapping = {
        'Player': 'player_name',
        'Age': 'age',
        'Team': 'team',
        'Pos': 'position',
        'Receiving Tgt': 'plyr_rec_tgt',
        'Receiving Rec': 'plyr_rec',
        'Receiving Ctch%': 'plyr_rec_catch_pct',
        'Receiving Yds': 'plyr_rec_yds',
        'Receiving Y/R': 'plyr_rec_yds_rec',
        'Receiving TD': 'plyr_rec_td',
        'Receiving 1D': 'plyr_rec_first_dwn',
        'Receiving Succ%': 'plyr_rec_succ_rt',
        'Receiving Lng': 'plyr_rec_lng',
        'Receiving Y/Tgt': 'plyr_rec_yds_tgt',
        'Receiving R/G': 'plyr_rec_gm',
        'Receiving Y/G': 'plyr_rec_ypg',
        'Fmb': 'plyr_rec_fmbl'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Select only the columns we need
    required_columns = ['player_name', 'age', 'team', 'position'] + [col for col in column_mapping.values() if col not in ['player_name', 'age', 'team', 'position']]
    df = df[required_columns]
    
    print(f"[INFO] Loaded {len(df)} basic receiving records")
    return df

def load_advanced_receiving_stats(year: int, week: int) -> pd.DataFrame:
    """Load advanced receiving stats CSV file"""
    directory = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{year}\\plyr_adv_receiving\\week_{week}\\clean"
    file_path = get_most_recent_csv_file(directory)
    
    print(f"[INFO] Loading advanced receiving stats from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Column mapping from CSV to our processing format
    column_mapping = {
        'Player': 'player_name',
        'Age': 'age',
        'Team': 'team',
        'Pos': 'position',
        'Receiving YBC': 'plyr_rec_aybc',
        'Receiving YBC/R': 'plyr_rec_aybc_rec',
        'Receiving YAC': 'plyr_rec_yac',
        'Receiving YAC/R': 'plyr_rec_yac_rec',
        'Receiving ADOT': 'plyr_rec_adot',
        'Receiving BrkTkl': 'plyr_rec_brkn_tkl',
        'Receiving Rec/Br': 'plyr_rec_brkn_tkl_rec',
        'Receiving Drop': 'plyr_rec_drp',
        'Receiving Drop%': 'plyr_rec_drp_pct',
        'Receiving Int': 'plyr_rec_int',
        'Receiving Rat': 'plyr_rec_pass_rtg'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Select only the columns we need
    required_columns = ['player_name', 'age', 'team', 'position'] + [col for col in column_mapping.values() if col not in ['player_name', 'age', 'team', 'position']]
    available_columns = [col for col in required_columns if col in df.columns]
    df = df[available_columns]
    
    print(f"[INFO] Loaded {len(df)} advanced receiving records")
    return df

def create_player_key(row):
    """Create a unique key for player matching - excludes team to handle inconsistent 2TM/3TM records"""
    # Handle age conversion for matching (basic is int, advanced is float)
    age = int(float(row['age'])) if pd.notnull(row['age']) else None
    return (row['player_name'], age, row['position'])

def filter_season_totals_only(df: pd.DataFrame, file_type: str) -> pd.DataFrame:
    """Filter to keep only season total rows for each player.
    
    For multi-team players, keep only the 2TM/3TM row (full season stats).
    For single-team players, keep their regular team row.
    """
    
    print(f"[INFO] Filtering {file_type} - Original records: {len(df)}")
    
    # Group by player name to identify multi-team cases
    filtered_records = []
    
    for player_name, group in df.groupby('player_name'):
        # Check if this player has multi-team entries
        team_values = group['team'].tolist()
        
        if '2TM' in team_values or '3TM' in team_values:
            # Multi-team player: take only the 2TM/3TM row (aggregated season stats)
            season_total_row = group[group['team'].isin(['2TM', '3TM'])]
            if len(season_total_row) > 0:
                filtered_records.append(season_total_row.iloc[0])
                print(f"[INFO] Multi-team player {player_name}: Using {'2TM' if '2TM' in team_values else '3TM'} row")
        else:
            # Single-team player: take their regular team row
            # If multiple rows exist (shouldn't happen), take the first
            filtered_records.append(group.iloc[0])
    
    # Convert back to DataFrame
    filtered_df = pd.DataFrame(filtered_records).reset_index(drop=True)
    
    print(f"[INFO] Filtered {file_type} - Final records: {len(filtered_df)}")
    print(f"[INFO] Filtered out {len(df) - len(filtered_df)} individual team records for multi-team players")
    
    return filtered_df

def consolidate_player_stats(basic_df: pd.DataFrame, advanced_df: pd.DataFrame) -> pd.DataFrame:
    """Consolidate stats from both files for unique players"""
    
    # First, filter both dataframes to keep only season totals
    basic_filtered = filter_season_totals_only(basic_df, "basic receiving")
    advanced_filtered = filter_season_totals_only(advanced_df, "advanced receiving")
    
    # Create player keys for both dataframes
    basic_filtered['player_key'] = basic_filtered.apply(create_player_key, axis=1)
    advanced_filtered['player_key'] = advanced_filtered.apply(create_player_key, axis=1)
    
    # Merge dataframes on player key (excluding team to handle 2TM/3TM inconsistencies)
    # Use outer join to include players from both files
    merged_df = pd.merge(
        basic_filtered, 
        advanced_filtered, 
        on=['player_key', 'player_name', 'position'], 
        how='outer',
        suffixes=('', '_adv')
    )
    
    # Handle age column (take from either basic or advanced)
    merged_df['age'] = merged_df['age'].fillna(merged_df.get('age_adv', pd.NA))
    
    # Handle team conflicts - prioritize 2TM/3TM over individual team names
    if 'team_adv' in merged_df.columns:
        def resolve_team_conflict(row):
            team1 = row.get('team')
            team2 = row.get('team_adv')
            
            # If both exist, prioritize 2TM/3TM aggregated totals
            if pd.notnull(team1) and pd.notnull(team2):
                if team1 in ['2TM', '3TM']:
                    return team1
                elif team2 in ['2TM', '3TM']:
                    return team2
                else:
                    # Both are individual teams, use the first one
                    return team1
            # Use whichever one exists
            return team1 if pd.notnull(team1) else team2
        
        merged_df['team'] = merged_df.apply(resolve_team_conflict, axis=1)
    
    # Drop duplicate columns and player key
    columns_to_drop = [col for col in merged_df.columns if col.endswith('_adv') or col == 'player_key']
    merged_df = merged_df.drop(columns=columns_to_drop)
    
    print(f"[INFO] Consolidated stats for {len(merged_df)} unique players")
    print(f"[INFO] Players only in basic file: {len(basic_filtered) - len(set(basic_filtered['player_name']) & set(advanced_filtered['player_name']))}")
    print(f"[INFO] Players only in advanced file: {len(advanced_filtered) - len(set(basic_filtered['player_name']) & set(advanced_filtered['player_name']))}") 
    
    return merged_df

def process_player_receiving_data(db: DatabaseConnector, df: pd.DataFrame, season_id: int, week_id: int, interactive: bool = True) -> pd.DataFrame:
    """Process consolidated player receiving data for database insertion"""
    
    processed_data = []
    error_count = 0
    success_count = 0
    
    for index, row in df.iterrows():
        try:
            # Skip League Average rows or any non-player entries
            if pd.isna(row['player_name']) or row['player_name'] == 'League Average':
                continue
            
            # Apply position mapping
            mapped_position = apply_position_mapping(row['position'])
            
            # Convert age to integer for player lookup
            age = int(float(row['age'])) if pd.notnull(row['age']) else None
            
            # Handle multi-team players (2TM, 3TM) by searching without team filter
            if row['team'] in ['2TM', '3TM']:
                player_id = get_season_player_id(
                    db, 
                    row['player_name'], 
                    '', # No team filter for multi-team players
                    season_id,
                    age=age,
                    position=mapped_position,
                    interactive=interactive
                )
            else:
                # Get player ID using enhanced season lookup
                player_id = get_season_player_id(
                    db, 
                    row['player_name'], 
                    row['team'], 
                    season_id,
                    age=age,
                    position=mapped_position,
                    interactive=interactive
                )
            
            # Skip this player if user chose to skip in interactive mode
            if interactive and player_id == 0:
                print(f"[INFO] Skipping player {row['player_name']} - user selection")
                continue
            
            # Prepare data row for database insertion
            data_row = {
                'plyr_id': player_id,
                'season_id': season_id,
                'week_id': week_id
            }
            
            # Map all receiving stat columns
            stat_columns = [
                'plyr_rec_tgt', 'plyr_rec', 'plyr_rec_catch_pct', 'plyr_rec_yds', 'plyr_rec_yds_rec',
                'plyr_rec_td', 'plyr_rec_first_dwn', 'plyr_rec_succ_rt', 'plyr_rec_lng', 'plyr_rec_yds_tgt',
                'plyr_rec_gm', 'plyr_rec_ypg', 'plyr_rec_fmbl', 'plyr_rec_aybc', 'plyr_rec_aybc_rec',
                'plyr_rec_yac', 'plyr_rec_yac_rec', 'plyr_rec_adot', 'plyr_rec_brkn_tkl', 'plyr_rec_brkn_tkl_rec',
                'plyr_rec_drp', 'plyr_rec_drp_pct', 'plyr_rec_int', 'plyr_rec_pass_rtg'
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
                        # Convert percentages to decimals and handle other conversions
                        if col in ['plyr_rec_catch_pct', 'plyr_rec_succ_rt', 'plyr_rec_drp_pct']:
                            # Convert percentage to decimal (e.g., 72.6 -> 0.726)
                            data_row[col] = float(value) / 100.0 if value != '' else None
                        elif col in ['plyr_rec_yds_rec', 'plyr_rec_yds_tgt', 'plyr_rec_gm', 'plyr_rec_ypg',
                                   'plyr_rec_aybc_rec', 'plyr_rec_yac_rec', 'plyr_rec_adot', 'plyr_rec_brkn_tkl_rec',
                                   'plyr_rec_pass_rtg']:
                            data_row[col] = float(value) if value != '' else None
                        else:
                            data_row[col] = int(float(value)) if value != '' else None
                else:
                    data_row[col] = None
            
            processed_data.append(data_row)
            success_count += 1
            
        except Exception as e:
            print(f"[ERROR] Failed to process player {row['player_name']} ({row['team']}): {e}")
            error_count += 1
            continue
    
    print(f"[INFO] Processed {success_count} players successfully, {error_count} errors")
    
    # Convert to DataFrame
    final_df = pd.DataFrame(processed_data)
    return handle_null_values(final_df)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Insert NFL player receiving statistics into database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plyr_rec.py                    # Run with interactive player selection (default)
  python plyr_rec.py --debug            # Run with debug logging (show update details)
        """
    )

    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug logging to show which records are being updated vs inserted'
    )

    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()

    print(f"[INFO] Starting plyr_rec.py script for {YEAR} week {WEEK}")
    print(f"[INFO] Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("[INFO] Interactive mode enabled - you will be prompted for player selection when multiple matches are found")

    if args.debug:
        print("[INFO] Debug mode enabled - detailed logging of insert vs update operations")
    
    # Initialize database connection
    db = DatabaseConnector()
    
    try:
        # Connect to database
        if not db.connect():
            print("[ERROR] Failed to connect to database")
            return False
        
        print("[INFO] Connected to database successfully")
        
        # Create table if needed
        if not create_plyr_rec_table(db):
            print("[ERROR] Failed to create plyr_rec table")
            return False
        
        # Get foreign key IDs
        season_id = get_season_id(db, YEAR)
        week_id = get_week_id(db, season_id, WEEK)
        
        print(f"[INFO] Using season_id: {season_id}, week_id: {week_id}")
        
        # Load CSV data
        basic_df = load_basic_receiving_stats(YEAR, WEEK)
        advanced_df = load_advanced_receiving_stats(YEAR, WEEK)
        
        # Consolidate player stats
        consolidated_df = consolidate_player_stats(basic_df, advanced_df)

        # Process data for database insertion (interactive mode is now default)
        processed_df = process_player_receiving_data(db, consolidated_df, season_id, week_id)
        
        if processed_df.empty:
            print("[WARNING] No data to insert")
            return False
        
        # Insert/upsert data
        if args.debug:
            success = batch_upsert_data_with_logging(db, 'plyr_rec', processed_df, debug_updates=True)
        else:
            success = batch_upsert_data(db, 'plyr_rec', processed_df)
        
        if success:
            print(f"[SUCCESS] plyr_rec.py completed successfully!")
            print(f"[INFO] Processed {len(processed_df)} player receiving stat records")
            return True
        else:
            print("[ERROR] Failed to insert data into plyr_rec table")
            return False
            
    except Exception as e:
        print(f"[ERROR] Script execution failed: {e}")
        return False
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)