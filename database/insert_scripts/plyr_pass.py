#!/usr/bin/env python3
"""
NFL Player Passing Stats Insert Script (plyr_pass.py)

This script processes season-level NFL player passing statistics from two source files:
1. plyr_passing CSV: Basic passing stats (completions, attempts, TDs, INTs, QBrec, etc.)
2. plyr_adv_passing CSV: Advanced passing stats (air yards, pressure, RPO, play action, etc.)

The script consolidates player stats from both files and inserts/upserts them into the plyr_pass table.
Special handling for QBrec column parsing (e.g., "12-3-1" -> qb_win=12, qb_loss=3, qb_tie=1).
"""

import sys
import os
import glob
import pandas as pd
import argparse
from datetime import datetime
import re

# Add parent directory to path to import db_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_utils import (
    DatabaseConnector, 
    YEAR, WEEK, 
    get_season_id, get_week_id, get_player_id, 
    create_table_if_not_exists, batch_upsert_data, handle_null_values,
    apply_position_mapping
)

def create_plyr_pass_table(db: DatabaseConnector) -> bool:
    """Create plyr_pass table if it doesn't exist"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS plyr_pass (
        plyr_pass_id INT AUTO_INCREMENT PRIMARY KEY,
        plyr_id INT,
        season_id INT,
        week_id INT,
        qb_win INT,
        qb_loss INT,
        qb_tie INT,
        plyr_pass_cmp INT,
        plyr_pass_att INT,
        plyr_pass_cmp_pct DECIMAL(6,4),
        plyr_pass_yds INT,
        plyr_pass_td INT,
        plyr_pass_td_pct DECIMAL(7,4),
        plyr_pass_int INT,
        plyr_pass_int_pct DECIMAL(6,4),
        plyr_pass_first_dwn INT,
        plyr_pass_succ_rt DECIMAL(7,4),
        plyr_pass_lng INT,
        plyr_pass_yds_att DECIMAL(7,4),
        plyr_pass_adj_yds_att DECIMAL(7,4),
        plyr_pass_yds_cmp DECIMAL(7,4),
        plyr_pass_yds_gm DECIMAL(7,4),
        plyr_pass_rtg DECIMAL(7,4),
        plyr_qbr DECIMAL(7,4),
        plyr_pass_sk INT,
        plyr_pass_sk_yds INT,
        plyr_pass_sk_pct DECIMAL(6,4),
        plyr_pass_net_yds_att DECIMAL(7,4),
        plyr_pass_adj_net_yds_att DECIMAL(7,4),
        plyr_fqc INT,
        plyr_gwd INT,
        plyr_pass_adv_yds INT,
        plyr_pass_iay INT,
        plyr_pass_iay_att DECIMAL(7,4),
        plyr_pass_cmp_ay INT,
        plyr_pass_cay_cmp DECIMAL(7,4),
        plyr_pass_cay_att DECIMAL(7,4),
        plyr_pass_yac INT,
        plyr_pass_yac_cmp DECIMAL(7,4),
        plyr_rpo_play INT,
        plyr_rpo_yds INT,
        plyr_rpo_pass_att INT,
        plyr_rpo_pass_yds INT,
        plyr_rpo_rush_att INT,
        plyr_rpo_rush_yds INT,
        plyr_pa_att INT,
        plyr_pa_yds INT,
        plyr_pass_pkt_time TIME(3),
        plyr_pass_bltz INT,
        plyr_pass_hrry INT,
        plyr_pass_hit INT,
        plyr_pass_prss INT,
        plyr_pass_prss_pct DECIMAL(6,4),
        plyr_pass_scrmbl INT,
        plyr_pass_yds_scrmbl DECIMAL(4,1),
        UNIQUE KEY uk_plyr_season (plyr_id, week_id, season_id),
        FOREIGN KEY (plyr_id) REFERENCES plyr(plyr_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id)
    )
    """
    return create_table_if_not_exists(db, "plyr_pass", create_table_sql)


def parse_qbrec(qbrec_str: str) -> tuple:
    """Parse QBrec column (e.g., '12-3-1') into qb_win, qb_loss, qb_tie"""
    if not qbrec_str or pd.isna(qbrec_str) or qbrec_str.strip() == '':
        return None, None, None
    
    try:
        # Split on dash
        parts = str(qbrec_str).strip().split('-')
        if len(parts) == 3:
            qb_win = int(float(parts[0])) if parts[0] else None
            qb_loss = int(float(parts[1])) if parts[1] else None
            qb_tie = int(float(parts[2])) if parts[2] else None
            return qb_win, qb_loss, qb_tie
        else:
            print(f"[WARNING] Invalid QBrec format: {qbrec_str}")
            return None, None, None
    except (ValueError, TypeError) as e:
        print(f"[WARNING] Error parsing QBrec '{qbrec_str}': {e}")
        return None, None, None

def get_most_recent_csv_file(directory: str) -> str:
    """Get the most recently created CSV file from directory"""
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    
    # Get the most recent file based on creation time
    most_recent = max(csv_files, key=os.path.getctime)
    return most_recent

def load_basic_passing_stats(year: int, week: int) -> pd.DataFrame:
    """Load basic passing stats CSV file"""
    directory = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{year}\\plyr_passing\\week_{week}"
    file_path = get_most_recent_csv_file(directory)
    
    print(f"[INFO] Loading basic passing stats from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Column mapping from CSV to our processing format
    column_mapping = {
        'Player': 'player_name',
        'Age': 'age',
        'Team': 'team',
        'Pos': 'position',
        'QBrec': 'qbrec',
        'Cmp': 'plyr_pass_cmp',
        'Att': 'plyr_pass_att',
        'Cmp%': 'plyr_pass_cmp_pct',
        'Yds': 'plyr_pass_yds',
        'TD': 'plyr_pass_td',
        'TD%': 'plyr_pass_td_pct',
        'Int': 'plyr_pass_int',
        'Int%': 'plyr_pass_int_pct',
        '1D': 'plyr_pass_first_dwn',
        'Succ%': 'plyr_pass_succ_rt',
        'Lng': 'plyr_pass_lng',
        'Y/A': 'plyr_pass_yds_att',
        'AY/A': 'plyr_pass_adj_yds_att',
        'Y/C': 'plyr_pass_yds_cmp',
        'Y/G': 'plyr_pass_yds_gm',
        'Rate': 'plyr_pass_rtg',
        'QBR': 'plyr_qbr',
        'Sk': 'plyr_pass_sk',
        'Yds.1': 'plyr_pass_sk_yds',  # Second Yds column is sack yards
        'Sk%': 'plyr_pass_sk_pct',
        'NY/A': 'plyr_pass_net_yds_att',
        'ANY/A': 'plyr_pass_adj_net_yds_att',
        '4QC': 'plyr_fqc',
        'GWD': 'plyr_gwd'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Select only the columns we need
    required_columns = ['player_name', 'age', 'team', 'position', 'qbrec'] + [col for col in column_mapping.values() if col not in ['player_name', 'age', 'team', 'position', 'qbrec']]
    available_columns = [col for col in required_columns if col in df.columns]
    df = df[available_columns]
    
    # Convert percentages to decimals
    pct_columns = ['plyr_pass_cmp_pct', 'plyr_pass_td_pct', 'plyr_pass_int_pct', 'plyr_pass_succ_rt', 'plyr_pass_sk_pct']
    for col in pct_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0
    
    print(f"[INFO] Loaded {len(df)} basic passing records")
    return df

def load_advanced_passing_stats(year: int, week: int) -> pd.DataFrame:
    """Load advanced passing stats CSV file"""
    directory = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{year}\\plyr_adv_passing\\week_{week}\\clean"
    file_path = get_most_recent_csv_file(directory)
    
    print(f"[INFO] Loading advanced passing stats from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Column mapping from CSV to our processing format
    column_mapping = {
        'Player': 'player_name',
        'Age': 'age',
        'Team': 'team',
        'Pos': 'position',
        'Passing Att': 'plyr_pass_adv_yds',  # Advanced passing yards - map from Passing Att for now
        'Air Yards IAY': 'plyr_pass_iay',
        'Air Yards IAY/PA': 'plyr_pass_iay_att',
        'Air Yards CAY': 'plyr_pass_cmp_ay',
        'Air Yards CAY/Cmp': 'plyr_pass_cay_cmp',
        'Air Yards CAY/PA': 'plyr_pass_cay_att',
        'Air Yards YAC': 'plyr_pass_yac',
        'Air Yards YAC/Cmp': 'plyr_pass_yac_cmp',
        'RPO Plays': 'plyr_rpo_play',
        'RPO Yds': 'plyr_rpo_yds',
        'RPO PassAtt': 'plyr_rpo_pass_att',
        'RPO PassYds': 'plyr_rpo_pass_yds',
        'RPO RushAtt': 'plyr_rpo_rush_att',
        'RPO RushYds': 'plyr_rpo_rush_yds',
        'PlayAction PassAtt': 'plyr_pa_att',
        'PlayAction PassYds': 'plyr_pa_yds',
        'Pressure PktTime': 'plyr_pass_pkt_time',
        'Pressure Bltz': 'plyr_pass_bltz',
        'Pressure Hrry': 'plyr_pass_hrry',
        'Pressure Hits': 'plyr_pass_hit',
        'Pressure Prss': 'plyr_pass_prss',
        'Pressure Prss%': 'plyr_pass_prss_pct',
        'Pressure Scrm': 'plyr_pass_scrmbl',
        'Pressure Yds/Scr': 'plyr_pass_yds_scrmbl'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Select only the columns we need
    required_columns = ['player_name', 'age', 'team', 'position'] + [col for col in column_mapping.values() if col not in ['player_name', 'age', 'team', 'position']]
    available_columns = [col for col in required_columns if col in df.columns]
    df = df[available_columns]
    
    # Convert percentages to decimals
    if 'plyr_pass_prss_pct' in df.columns:
        df['plyr_pass_prss_pct'] = pd.to_numeric(df['plyr_pass_prss_pct'], errors='coerce') / 100.0
    
    print(f"[INFO] Loaded {len(df)} advanced passing records")
    return df

def create_player_key(row):
    """Create a unique key for player matching using Player, Age, Pos (as specified in prompt)"""
    # Handle age conversion for matching
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
    basic_filtered = filter_season_totals_only(basic_df, "basic passing")
    advanced_filtered = filter_season_totals_only(advanced_df, "advanced passing")
    
    # Apply position mapping to both dataframes
    basic_filtered['position'] = basic_filtered['position'].apply(apply_position_mapping)
    advanced_filtered['position'] = advanced_filtered['position'].apply(apply_position_mapping)
    
    # Create player keys for both dataframes using Player, Age, Pos as specified
    basic_filtered['player_key'] = basic_filtered.apply(create_player_key, axis=1)
    advanced_filtered['player_key'] = advanced_filtered.apply(create_player_key, axis=1)
    
    # Merge dataframes on player key (Player, Age, Pos)
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

def process_player_passing_data(db: DatabaseConnector, df: pd.DataFrame, season_id: int, week_id: int, interactive: bool = True) -> pd.DataFrame:
    """Process consolidated player passing data for database insertion"""
    
    processed_data = []
    error_count = 0
    success_count = 0
    
    for index, row in df.iterrows():
        try:
            # Skip League Average rows or any non-player entries
            if pd.isna(row['player_name']) or row['player_name'] == 'League Average':
                continue
            
            # Position mapping was already applied during consolidation
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
            
            # Parse QBrec column
            qb_win, qb_loss, qb_tie = parse_qbrec(row.get('qbrec'))
            
            # Prepare data row for database insertion
            data_row = {
                'plyr_id': player_id,
                'season_id': season_id,
                'week_id': week_id,
                'qb_win': qb_win,
                'qb_loss': qb_loss,
                'qb_tie': qb_tie
            }
            
            # Map all passing stat columns
            stat_columns = [
                'plyr_pass_cmp', 'plyr_pass_att', 'plyr_pass_cmp_pct', 'plyr_pass_yds', 'plyr_pass_td',
                'plyr_pass_td_pct', 'plyr_pass_int', 'plyr_pass_int_pct', 'plyr_pass_first_dwn', 'plyr_pass_succ_rt',
                'plyr_pass_lng', 'plyr_pass_yds_att', 'plyr_pass_adj_yds_att', 'plyr_pass_yds_cmp', 'plyr_pass_yds_gm',
                'plyr_pass_rtg', 'plyr_qbr', 'plyr_pass_sk', 'plyr_pass_sk_yds', 'plyr_pass_sk_pct',
                'plyr_pass_net_yds_att', 'plyr_pass_adj_net_yds_att', 'plyr_fqc', 'plyr_gwd', 'plyr_pass_adv_yds',
                'plyr_pass_iay', 'plyr_pass_iay_att', 'plyr_pass_cmp_ay', 'plyr_pass_cay_cmp', 'plyr_pass_cay_att',
                'plyr_pass_yac', 'plyr_pass_yac_cmp', 'plyr_rpo_play', 'plyr_rpo_yds', 'plyr_rpo_pass_att',
                'plyr_rpo_pass_yds', 'plyr_rpo_rush_att', 'plyr_rpo_rush_yds', 'plyr_pa_att', 'plyr_pa_yds',
                'plyr_pass_pkt_time', 'plyr_pass_bltz', 'plyr_pass_hrry', 'plyr_pass_hit', 'plyr_pass_prss',
                'plyr_pass_prss_pct', 'plyr_pass_scrmbl', 'plyr_pass_yds_scrmbl'
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
                        # Special handling for TIME column
                        if col == 'plyr_pass_pkt_time':
                            # Convert seconds to TIME format (e.g., 2.5 seconds -> 00:00:02.500)
                            try:
                                seconds = float(value)
                                data_row[col] = f"00:00:{seconds:06.3f}"
                            except (ValueError, TypeError):
                                data_row[col] = None
                        # Convert to appropriate type
                        elif col in ['plyr_pass_cmp_pct', 'plyr_pass_td_pct', 'plyr_pass_int_pct', 'plyr_pass_succ_rt',
                                   'plyr_pass_yds_att', 'plyr_pass_adj_yds_att', 'plyr_pass_yds_cmp', 'plyr_pass_yds_gm',
                                   'plyr_pass_rtg', 'plyr_qbr', 'plyr_pass_sk_pct', 'plyr_pass_net_yds_att', 
                                   'plyr_pass_adj_net_yds_att', 'plyr_pass_iay_att', 'plyr_pass_cay_cmp', 'plyr_pass_cay_att',
                                   'plyr_pass_yac_cmp', 'plyr_pass_prss_pct', 'plyr_pass_yds_scrmbl']:
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
        description='Insert NFL player passing statistics into database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plyr_pass.py                    # Run with interactive player selection (default)
        """
    )

    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()

    print(f"[INFO] Starting plyr_pass.py script for {YEAR} week {WEEK}")
    print(f"[INFO] Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("[INFO] Interactive mode enabled - you will be prompted for player selection when multiple matches are found")
    
    # Initialize database connection
    db = DatabaseConnector()
    
    try:
        # Connect to database
        if not db.connect():
            print("[ERROR] Failed to connect to database")
            return False
        
        print("[INFO] Connected to database successfully")
        
        # Create table if needed
        if not create_plyr_pass_table(db):
            print("[ERROR] Failed to create plyr_pass table")
            return False
        
        # Get foreign key IDs
        season_id = get_season_id(db, YEAR)
        week_id = get_week_id(db, season_id, WEEK)
        
        print(f"[INFO] Using season_id: {season_id}, week_id: {week_id}")
        
        # Load CSV data
        basic_df = load_basic_passing_stats(YEAR, WEEK)
        advanced_df = load_advanced_passing_stats(YEAR, WEEK)
        
        # Consolidate player stats
        consolidated_df = consolidate_player_stats(basic_df, advanced_df)

        # Process data for database insertion (interactive mode is now default)
        processed_df = process_player_passing_data(db, consolidated_df, season_id, week_id)
        
        if processed_df.empty:
            print("[WARNING] No data to insert")
            return False
        
        # Insert/upsert data
        success = batch_upsert_data(db, 'plyr_pass', processed_df)
        
        if success:
            print(f"[SUCCESS] plyr_pass.py completed successfully!")
            print(f"[INFO] Processed {len(processed_df)} player passing stat records")
            return True
        else:
            print("[ERROR] Failed to insert data into plyr_pass table")
            return False
            
    except Exception as e:
        print(f"[ERROR] Script execution failed: {e}")
        return False
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)