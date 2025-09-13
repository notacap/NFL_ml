#!/usr/bin/env python3
"""
NFL Player Scoring Stats Insert Script (plyr_scoring.py)

This script processes season-level NFL player scoring statistics.
The script inserts/upserts player scoring stats into the plyr_scoring table.

Includes all types of scoring: touchdowns, extra points, field goals, two-point conversions, and safeties.
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
    create_table_if_not_exists, batch_upsert_data, handle_null_values
)

def create_plyr_scoring_table(db: DatabaseConnector) -> bool:
    """Create plyr_scoring table if it doesn't exist"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS plyr_scoring (
        plyr_scoring_id INT AUTO_INCREMENT PRIMARY KEY,
        plyr_id INT,
        season_id INT,
        week_id INT,
        plyr_pr_td INT,
        plyr_kr_td INT,
        plyr_oth_td INT,
        plyr_tot_td INT,
        plyr_two_pt_conv INT,
        plyr_two_pt_conv_att INT,
        plyr_def_two_pt_conv INT,
        plyr_ex_pt_md INT,
        plyr_ex_pt_att INT,
        plyr_fg_md INT,
        plyr_fg_att INT,
        plyr_tot_pt INT,
        plyr_tot_pt_gm DECIMAL(7,4),
        UNIQUE KEY uk_plyr_season (plyr_id, season_id, week_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (plyr_id) REFERENCES plyr(plyr_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id)
    )
    """
    return create_table_if_not_exists(db, "plyr_scoring", create_table_sql)

def get_most_recent_csv_file(directory: str) -> str:
    """Get the most recently created CSV file from directory"""
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    
    # Get the most recent file based on creation time
    most_recent = max(csv_files, key=os.path.getctime)
    return most_recent

def filter_multi_team_players(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out individual team entries for multi-team players, keeping only summary rows."""
    
    print(f"[INFO] Starting multi-team player filtering with {len(df)} total records")
    
    # Create unique identifier for players (Player + Age + Position)
    df['player_unique_id'] = df['player_name'] + '_' + df['age'].astype(str) + '_' + df['position'].astype(str)
    
    # Find duplicated player unique IDs
    duplicate_counts = df['player_unique_id'].value_counts()
    duplicated_players = duplicate_counts[duplicate_counts > 1]
    
    print(f"[INFO] Found {len(duplicated_players)} players with multiple team entries")
    
    rows_to_keep = []
    filtered_count = 0
    
    for unique_id, count in duplicated_players.items():
        player_rows = df[df['player_unique_id'] == unique_id]
        
        # Check if there's a summary row (2TM, 3TM, etc.)
        multi_team_markers = ['2TM', '3TM', '4TM', '5TM']
        summary_rows = player_rows[player_rows['team'].isin(multi_team_markers)]
        
        if len(summary_rows) > 0:
            # Keep only the summary row, regardless of how many individual team entries exist
            summary_row = summary_rows.iloc[0]
            summary_team = summary_row['team']
            player_name = summary_row['player_name']
            
            rows_to_keep.append(summary_row)
            print(f"[INFO] Keeping summary row for {player_name} ({summary_team}) - filtered out {count-1} individual team entries")
            filtered_count += count - 1
        else:
            # No summary row found, keep all entries (these might be legitimate different players or data issues)
            print(f"[WARNING] Player {player_rows.iloc[0]['player_name']} has {count} entries but no summary row found - keeping all entries")
            rows_to_keep.extend(player_rows.to_dict('records'))
    
    # Add all non-duplicated players (single team players)
    non_duplicated_players = df[~df['player_unique_id'].isin(duplicated_players.index)]
    
    # Create a list to hold all rows to keep
    if len(rows_to_keep) > 0:
        # Convert Series objects to dictionaries
        rows_to_keep_clean = []
        for row in rows_to_keep:
            if hasattr(row, 'to_dict'):
                rows_to_keep_clean.append(row.to_dict())
            else:
                rows_to_keep_clean.append(row)
        rows_to_keep = rows_to_keep_clean
        
        # Add non-duplicated players
        rows_to_keep.extend(non_duplicated_players.to_dict('records'))
        
        # Convert back to DataFrame
        filtered_df = pd.DataFrame(rows_to_keep)
    else:
        # If no multi-team players to filter, just use the non-duplicated players
        filtered_df = non_duplicated_players.copy()
    
    # Remove the temporary unique ID column
    if 'player_unique_id' in filtered_df.columns:
        filtered_df = filtered_df.drop('player_unique_id', axis=1)
    
    print(f"[INFO] Multi-team filtering complete: {len(df)} -> {len(filtered_df)} records ({filtered_count} individual team entries filtered out)")
    
    return filtered_df

def load_scoring_stats(year: int, week: int) -> pd.DataFrame:
    """Load scoring stats CSV file"""
    directory = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{year}\\plyr_scoring\\week_{week}\\clean"
    file_path = get_most_recent_csv_file(directory)
    
    print(f"[INFO] Loading scoring stats from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Column mapping from CSV to our processing format
    column_mapping = {
        'Player': 'player_name',
        'Team': 'team',
        'Age': 'age',
        'Pos': 'position',
        # Touchdown statistics
        'Touchdowns PRTD': 'plyr_pr_td',
        'Touchdowns KRTD': 'plyr_kr_td', 
        'Touchdowns OthTD': 'plyr_oth_td',
        'Touchdowns AllTD': 'plyr_tot_td',
        'Touchdowns 2PM': 'plyr_two_pt_conv',
        'D2P': 'plyr_def_two_pt_conv',
        # Kicking statistics
        'PAT XPM': 'plyr_ex_pt_md',
        'PAT XPA': 'plyr_ex_pt_att',
        'FG FGM': 'plyr_fg_md',
        'FG FGA': 'plyr_fg_att',
        # Total points
        'Pts': 'plyr_tot_pt',
        'Pts/G': 'plyr_tot_pt_gm'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Select only the columns we need
    required_columns = ['player_name', 'team', 'age', 'position'] + [col for col in column_mapping.values() if col not in ['player_name', 'team', 'age', 'position']]
    available_columns = [col for col in required_columns if col in df.columns]
    df = df[available_columns]
    
    print(f"[INFO] Loaded {len(df)} scoring records")
    
    # Filter multi-team players to keep only summary rows
    df = filter_multi_team_players(df)
    
    return df

def process_player_scoring_data(db: DatabaseConnector, df: pd.DataFrame, season_id: int, week_id: int, interactive: bool = False) -> pd.DataFrame:
    """Process player scoring data for database insertion"""
    
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
                    player_id = get_season_player_id(
                        db, 
                        row['player_name'], 
                        '', # No team filter for multi-team players
                        season_id,
                        age=row.get('age'),
                        position=row.get('position'),
                        interactive=interactive
                    )
                except ValueError as e:
                    # For multi-team players, we need manual selection since we can't use team to disambiguate
                    if "multiple matches" in str(e).lower() or interactive:
                        # Use interactive mode to handle multiple matches
                        player_id = get_season_player_id(
                            db,
                            row['player_name'],
                            '',
                            season_id,
                            age=row.get('age'),
                            position=row.get('position'),
                            interactive=True
                        )
                    else:
                        print(f"[ERROR] Multiple matches found for multi-team player {row['player_name']} - use interactive mode")
                        error_count += 1
                        continue
            else:
                # Get player ID using team for disambiguation
                player_id = get_season_player_id(
                    db, 
                    row['player_name'], 
                    row['team'], 
                    season_id,
                    age=row.get('age'),
                    position=row.get('position'),
                    interactive=interactive
                )
            
            # Skip this player if user chose to skip in interactive mode
            if interactive and player_id == 0:
                print(f"[INFO] Skipping player {row['player_name']} - user selection")
                skipped_count += 1
                continue
            
            # Prepare data row for database insertion
            data_row = {
                'plyr_id': player_id,
                'season_id': season_id,
                'week_id': week_id
            }
            
            # Map all scoring stat columns
            stat_columns = [
                'plyr_pr_td', 'plyr_kr_td', 'plyr_oth_td', 'plyr_tot_td',
                'plyr_two_pt_conv', 'plyr_def_two_pt_conv',
                'plyr_ex_pt_md', 'plyr_ex_pt_att', 'plyr_fg_md', 'plyr_fg_att',
                'plyr_tot_pt', 'plyr_tot_pt_gm'
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
                        if col == 'plyr_tot_pt_gm':
                            # Points per game is decimal
                            data_row[col] = float(value) if value is not None else None
                        else:
                            # All other columns are integers
                            data_row[col] = int(float(value)) if value != '' else None
                else:
                    data_row[col] = None
            
            # Handle plyr_two_pt_conv_att which is not in CSV - set to None
            data_row['plyr_two_pt_conv_att'] = None
            
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
        description='Insert NFL player scoring statistics into database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plyr_scoring.py                    # Run in standard mode (auto-select first match)
  python plyr_scoring.py --interactive      # Run in interactive mode (manual player selection)  
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
    
    print(f"[INFO] Starting plyr_scoring.py script for {YEAR} week {WEEK}")
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
        if not create_plyr_scoring_table(db):
            print("[ERROR] Failed to create plyr_scoring table")
            return False
        
        # Get foreign key IDs
        season_id = get_season_id(db, YEAR)
        week_id = get_week_id(db, season_id, WEEK)
        
        print(f"[INFO] Using season_id: {season_id}, week_id: {week_id}")
        
        # Load CSV data
        df = load_scoring_stats(YEAR, WEEK)
        
        # Process data for database insertion
        processed_df = process_player_scoring_data(db, df, season_id, week_id, args.interactive)
        
        if processed_df.empty:
            print("[WARNING] No data to insert")
            return False
        
        # Insert/upsert data
        success = batch_upsert_data(db, 'plyr_scoring', processed_df)
        
        if success:
            print(f"[SUCCESS] plyr_scoring.py completed successfully!")
            print(f"[INFO] Processed {len(processed_df)} player scoring stat records")
            return True
        else:
            print("[ERROR] Failed to insert data into plyr_scoring table")
            return False
            
    except Exception as e:
        print(f"[ERROR] Script execution failed: {e}")
        return False
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)