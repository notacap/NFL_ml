#!/usr/bin/env python3
"""
NFL Player Defense Stats Insert Script (plyr_def.py)

This script processes season-level NFL player defensive statistics from two source files:
1. plyr_def CSV: Basic defensive stats (INT, tackles, sacks, etc.)
2. plyr_adv_def CSV: Advanced defensive stats (coverage, pass rush, etc.)

The script consolidates player stats from both files and inserts/upserts them into the plyr_def table.
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
    create_table_if_not_exists, batch_upsert_data, batch_upsert_data_with_logging, handle_null_values,
    apply_position_mapping
)

def create_plyr_def_table(db: DatabaseConnector) -> bool:
    """Create plyr_def table if it doesn't exist"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS plyr_def (
        plyr_def_id INT AUTO_INCREMENT PRIMARY KEY,
        plyr_id INT,
        season_id INT,
        week_id INT,
        plyr_def_int INT,
        plyr_def_int_yds INT,
        plyr_def_int_td INT,
        plyr_def_int_lng INT,
        plyr_pass_def INT,
        plyr_def_force_fmbl INT,
        plyr_def_fmbl_rec INT,
        plyr_def_fmbl_rec_yds INT,
        plyr_def_fmbl_rec_td INT,
        plyr_def_solo_tkl INT,
        plyr_def_tkl_assist INT,
        plyr_def_tfl INT,
        plyr_def_qb_hit INT,
        plyr_def_sfty INT,
        plyr_def_tgt INT,
        plyr_def_cmp INT,
        plyr_def_cmp_pct DECIMAL(7,4),
        plyr_def_pass_yds INT,
        plyr_def_pass_yds_cmp DECIMAL(7,4),
        plyr_def_pass_yds_tgt DECIMAL(7,4),
        plyr_def_pass_td INT,
        plyr_def_pass_rtg DECIMAL(7,4),
        plyr_def_adot DECIMAL(7,4),
        plyr_def_ay INT,
        plyr_def_yac INT,
        plyr_def_bltz INT,
        plyr_def_hrry INT,
        plyr_def_qbkd INT,
        plyr_def_sk DECIMAL(7,4),
        plyr_def_prss INT,
        plyr_def_comb_tkl INT,
        plyr_def_mtkl INT,
        plyr_def_mtkl_pct DECIMAL(7,4),
        UNIQUE KEY uk_plyr_season (plyr_id, season_id, week_id),
        FOREIGN KEY (plyr_id) REFERENCES plyr(plyr_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id)
    )
    """
    return create_table_if_not_exists(db, "plyr_def", create_table_sql)


def get_most_recent_csv_file(directory: str) -> str:
    """Get the most recently created CSV file from directory"""
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    
    # Get the most recent file based on creation time
    most_recent = max(csv_files, key=os.path.getctime)
    return most_recent

def load_basic_defense_stats(year: int, week: int) -> pd.DataFrame:
    """Load basic defense stats CSV file"""
    directory = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{year}\\plyr_def\\week_{week}\\clean"
    file_path = get_most_recent_csv_file(directory)
    
    print(f"[INFO] Loading basic defense stats from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Column mapping from CSV to our processing format
    column_mapping = {
        'Player': 'player_name',
        'Age': 'age',
        'Team': 'team',
        'Pos': 'position',
        'Def Interceptions Int': 'plyr_def_int',
        'Def Interceptions Yds': 'plyr_def_int_yds',
        'Def Interceptions IntTD': 'plyr_def_int_td',
        'Def Interceptions Lng': 'plyr_def_int_lng',
        'Def Interceptions PD': 'plyr_pass_def',
        'Fumbles FF': 'plyr_def_force_fmbl',
        'Fumbles FR': 'plyr_def_fmbl_rec',
        'Fumbles Yds': 'plyr_def_fmbl_rec_yds',
        'Fumbles FRTD': 'plyr_def_fmbl_rec_td',
        'Tackles Solo': 'plyr_def_solo_tkl',
        'Tackles Ast': 'plyr_def_tkl_assist',
        'Tackles TFL': 'plyr_def_tfl',
        'Tackles QBHits': 'plyr_def_qb_hit',
        'Sfty': 'plyr_def_sfty'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Select only the columns we need
    required_columns = ['player_name', 'age', 'team', 'position'] + [col for col in column_mapping.values() if col not in ['player_name', 'age', 'team', 'position']]
    df = df[required_columns]
    
    print(f"[INFO] Loaded {len(df)} basic defense records")
    return df

def load_advanced_defense_stats(year: int, week: int) -> pd.DataFrame:
    """Load advanced defense stats CSV file"""
    directory = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{year}\\plyr_adv_def\\week_{week}\\clean"
    file_path = get_most_recent_csv_file(directory)
    
    print(f"[INFO] Loading advanced defense stats from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Column mapping from CSV to our processing format
    column_mapping = {
        'Player': 'player_name',
        'Age': 'age',
        'Team': 'team',
        'Pos': 'position',
        'Pass Coverage Tgt': 'plyr_def_tgt',
        'Pass Coverage Cmp': 'plyr_def_cmp',
        'Pass Coverage Cmp%': 'plyr_def_cmp_pct',
        'Pass Coverage Yds': 'plyr_def_pass_yds',
        'Pass Coverage Yds/Cmp': 'plyr_def_pass_yds_cmp',
        'Pass Coverage Yds/Tgt': 'plyr_def_pass_yds_tgt',
        'Pass Coverage TD': 'plyr_def_pass_td',
        'Pass Coverage Rat': 'plyr_def_pass_rtg',
        'Pass Coverage DADOT': 'plyr_def_adot',
        'Pass Coverage Air': 'plyr_def_ay',
        'Pass Coverage YAC': 'plyr_def_yac',
        'Pass Rush Bltz': 'plyr_def_bltz',
        'Pass Rush Hrry': 'plyr_def_hrry',
        'Pass Rush QBKD': 'plyr_def_qbkd',
        'Pass Rush Sk': 'plyr_def_sk',
        'Pass Rush Prss': 'plyr_def_prss',
        'Tackles Comb': 'plyr_def_comb_tkl',
        'Tackles MTkl': 'plyr_def_mtkl',
        'Tackles MTkl%': 'plyr_def_mtkl_pct'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Select only the columns we need
    required_columns = ['player_name', 'age', 'team', 'position'] + [col for col in column_mapping.values() if col not in ['player_name', 'age', 'team', 'position']]
    available_columns = [col for col in required_columns if col in df.columns]
    df = df[available_columns]
    
    # Convert percentages to decimals
    if 'plyr_def_cmp_pct' in df.columns:
        df['plyr_def_cmp_pct'] = pd.to_numeric(df['plyr_def_cmp_pct'], errors='coerce') / 100.0
    
    if 'plyr_def_mtkl_pct' in df.columns:
        df['plyr_def_mtkl_pct'] = pd.to_numeric(df['plyr_def_mtkl_pct'], errors='coerce') / 100.0
    
    print(f"[INFO] Loaded {len(df)} advanced defense records")
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

def filter_defensive_players_only(df: pd.DataFrame, file_type: str) -> pd.DataFrame:
    """Filter out offensive players, keeping only defensive players.
    
    Apply position mapping first, then filter out QB, RB, WR, TE, OL positions.
    """
    
    print(f"[INFO] Filtering offensive players from {file_type} - Records before: {len(df)}")
    
    # Apply position mapping to standardize positions
    df['mapped_position'] = df['position'].apply(apply_position_mapping)
    
    # Define offensive positions to exclude
    offensive_positions = ['QB', 'RB', 'WR', 'TE', 'OL']
    
    # Filter out offensive players
    defensive_df = df[~df['mapped_position'].isin(offensive_positions)].copy()
    
    # Update the position column to use the mapped position
    defensive_df['position'] = defensive_df['mapped_position']
    defensive_df.drop('mapped_position', axis=1, inplace=True)
    
    offensive_filtered = len(df) - len(defensive_df)
    print(f"[INFO] Filtered out {offensive_filtered} offensive players from {file_type}")
    print(f"[INFO] Remaining defensive players: {len(defensive_df)}")
    
    if offensive_filtered > 0:
        offensive_players = df[df['mapped_position'].isin(offensive_positions)]
        sample_offensive = offensive_players[['player_name', 'mapped_position']].head(3)
        print(f"[INFO] Sample offensive players filtered: \n{sample_offensive.to_string(index=False)}")
    
    return defensive_df

def consolidate_player_stats(basic_df: pd.DataFrame, advanced_df: pd.DataFrame) -> pd.DataFrame:
    """Consolidate stats from both files for unique players"""
    
    # First, filter both dataframes to keep only season totals
    basic_filtered = filter_season_totals_only(basic_df, "basic defense")
    advanced_filtered = filter_season_totals_only(advanced_df, "advanced defense")
    
    # Second, filter out offensive players (apply position mapping and exclude QB, RB, WR, TE, OL)
    basic_defensive = filter_defensive_players_only(basic_filtered, "basic defense")
    advanced_defensive = filter_defensive_players_only(advanced_filtered, "advanced defense")
    
    # Create player keys for both dataframes
    basic_defensive['player_key'] = basic_defensive.apply(create_player_key, axis=1)
    advanced_defensive['player_key'] = advanced_defensive.apply(create_player_key, axis=1)
    
    # Merge dataframes on player key (excluding team to handle 2TM/3TM inconsistencies)
    # Use outer join to include players from both files
    merged_df = pd.merge(
        basic_defensive, 
        advanced_defensive, 
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
    
    print(f"[INFO] Consolidated stats for {len(merged_df)} unique defensive players")
    print(f"[INFO] Defensive players only in basic file: {len(basic_defensive) - len(set(basic_defensive['player_name']) & set(advanced_defensive['player_name']))}")
    print(f"[INFO] Defensive players only in advanced file: {len(advanced_defensive) - len(set(basic_defensive['player_name']) & set(advanced_defensive['player_name']))}")
    
    return merged_df

def process_player_defense_data(db: DatabaseConnector, df: pd.DataFrame, season_id: int, week_id: int, interactive: bool = True) -> pd.DataFrame:
    """Process consolidated player defense data for database insertion"""
    
    processed_data = []
    error_count = 0
    success_count = 0
    
    for index, row in df.iterrows():
        try:
            # Skip League Average rows or any non-player entries
            if pd.isna(row['player_name']) or row['player_name'] == 'League Average':
                continue
            
            # Position mapping was already applied during filtering
            mapped_position = row['position']
            
            # Convert age to integer for player lookup
            age = int(float(row['age'])) if pd.notnull(row['age']) else None
            
            # Handle multi-team players (2TM, 3TM) by searching without team filter
            if row['team'] in ['2TM', '3TM']:
                player_id = get_player_id(
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
                player_id = get_player_id(
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
            
            # Map all defensive stat columns
            stat_columns = [
                'plyr_def_int', 'plyr_def_int_yds', 'plyr_def_int_td', 'plyr_def_int_lng',
                'plyr_pass_def', 'plyr_def_force_fmbl', 'plyr_def_fmbl_rec', 'plyr_def_fmbl_rec_yds',
                'plyr_def_fmbl_rec_td', 'plyr_def_solo_tkl', 'plyr_def_tkl_assist', 'plyr_def_tfl',
                'plyr_def_qb_hit', 'plyr_def_sfty', 'plyr_def_tgt', 'plyr_def_cmp', 'plyr_def_cmp_pct',
                'plyr_def_pass_yds', 'plyr_def_pass_yds_cmp', 'plyr_def_pass_yds_tgt', 'plyr_def_pass_td',
                'plyr_def_pass_rtg', 'plyr_def_adot', 'plyr_def_ay', 'plyr_def_yac', 'plyr_def_bltz',
                'plyr_def_hrry', 'plyr_def_qbkd', 'plyr_def_sk', 'plyr_def_prss', 'plyr_def_comb_tkl',
                'plyr_def_mtkl', 'plyr_def_mtkl_pct'
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
                        if col in ['plyr_def_cmp_pct', 'plyr_def_pass_yds_cmp', 'plyr_def_pass_yds_tgt', 
                                  'plyr_def_pass_rtg', 'plyr_def_adot', 'plyr_def_sk', 'plyr_def_mtkl_pct']:
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
        description='Insert NFL player defensive statistics into database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plyr_def.py                    # Run with interactive player selection (default)
  python plyr_def.py --debug            # Run with debug logging (show update details)
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

    print(f"[INFO] Starting plyr_def.py script for {YEAR} week {WEEK}")
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
        if not create_plyr_def_table(db):
            print("[ERROR] Failed to create plyr_def table")
            return False
        
        # Get foreign key IDs
        season_id = get_season_id(db, YEAR)
        week_id = get_week_id(db, season_id, WEEK)
        
        print(f"[INFO] Using season_id: {season_id}, week_id: {week_id}")
        
        # Load CSV data
        basic_df = load_basic_defense_stats(YEAR, WEEK)
        advanced_df = load_advanced_defense_stats(YEAR, WEEK)
        
        # Consolidate player stats
        consolidated_df = consolidate_player_stats(basic_df, advanced_df)

        # Process data for database insertion (interactive mode is now default)
        processed_df = process_player_defense_data(db, consolidated_df, season_id, week_id)
        
        # Debug: Check for duplicate players within our DataFrame
        if args.debug:
            print(f"\n[DEBUG] Checking for duplicate player records in processed DataFrame...")
            duplicates = processed_df.groupby(['plyr_id', 'season_id', 'week_id']).size()
            internal_duplicates = duplicates[duplicates > 1]
            
            if len(internal_duplicates) > 0:
                print(f"[DEBUG] WARNING: Found {len(internal_duplicates)} duplicate player combinations within DataFrame:")
                for (plyr_id, season_id, week_id), count in internal_duplicates.items():
                    # Get player info for better debugging
                    player_info_query = "SELECT plyr_name, t.abrv FROM plyr p JOIN nfl_team t ON p.team_id = t.team_id WHERE p.plyr_id = %s"
                    player_info = db.fetch_all(player_info_query, (plyr_id,))
                    player_name = player_info[0][0] if player_info else "Unknown"
                    team_abrv = player_info[0][1] if player_info else "Unknown"
                    print(f"[DEBUG]   Player ID {plyr_id} ({player_name}, {team_abrv}) appears {count} times for season {season_id}, week {week_id}")
                
                print(f"[DEBUG] This suggests duplicate processing within the script - check consolidation logic")
            else:
                print(f"[DEBUG] No duplicate player records found within DataFrame - duplicates must be in database")
        
        if processed_df.empty:
            print("[WARNING] No data to insert")
            return False
        
        # Insert/upsert data
        if args.debug:
            success = batch_upsert_data_with_logging(db, 'plyr_def', processed_df, debug_updates=True)
        else:
            success = batch_upsert_data(db, 'plyr_def', processed_df)
        
        if success:
            print(f"[SUCCESS] plyr_def.py completed successfully!")
            print(f"[INFO] Processed {len(processed_df)} player defensive stat records")
            return True
        else:
            print("[ERROR] Failed to insert data into plyr_def table")
            return False
            
    except Exception as e:
        print(f"[ERROR] Script execution failed: {e}")
        return False
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)