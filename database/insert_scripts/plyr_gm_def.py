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
    get_season_id, get_week_id, get_game_id, get_player_id, get_team_id,
    create_table_if_not_exists
)


def create_plyr_gm_def_table(db: DatabaseConnector) -> bool:
    """Create the plyr_gm_def table if it doesn't exist (from restructured_db.sql)."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS plyr_gm_def (
        plyr_gm_def_id INT PRIMARY KEY AUTO_INCREMENT,
        plyr_id INT,
        week_id INT,
        game_id INT,
        season_id INT,
        team_id INT,
        plyr_gm_def_int INT,
        plyr_gm_def_int_yds INT,
        plyr_gm_def_int_td INT,
        plyr_gm_def_int_lng INT,
        plyr_gm_pass_def INT,
        plyr_gm_def_comb_tkl INT,
        plyr_gm_def_solo_tkl INT,
        plyr_gm_def_tkl_assist INT,
        plyr_gm_def_tfl INT,
        plyr_gm_def_qb_hit INT,
        plyr_gm_def_fmble_rec INT,
        plyr_gm_def_fmbl_ret_yds INT,
        plyr_gm_def_fmbl_td INT,
        plyr_gm_def_force_fmbl INT,
        plyr_gm_def_tgt INT,
        plyr_gm_def_cmp INT,
        plyr_gm_def_cmp_pct FLOAT(5,4),
        plyr_gm_def_pass_yds INT,
        plyr_gm_def_pass_yds_cmp FLOAT(7,4),
        plyr_gm_def_pass_yds_tgt FLOAT(7,4),
        plyr_gm_def_pass_td INT,
        plyr_gm_def_pass_rtg FLOAT(7,4),
        plyr_gm_def_adot FLOAT(7,4),
        plyr_gm_def_ay INT,
        plyr_gm_def_yac INT,
        plyr_gm_def_bltz INT,
        plyr_gm_def_hrry INT,
        plyr_gm_def_qbkd INT,
        plyr_gm_def_sk INT,
        plyr_gm_def_prss INT,
        plyr_gm_def_mtkl INT,
        plyr_gm_def_mtkl_pct FLOAT(5,4),
        FOREIGN KEY (plyr_id) REFERENCES plyr(plyr_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (game_id) REFERENCES nfl_game(game_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        FOREIGN KEY (team_id) REFERENCES nfl_team(team_id),
        UNIQUE KEY uk_player_game (plyr_id, game_id)
    );
    """
    
    return create_table_if_not_exists(db, 'plyr_gm_def', create_table_sql)


def convert_percentage_to_float(value) -> float:
    """Convert percentage string (e.g. '66.7%') to float (0.667)."""
    if pd.isna(value) or value == '' or str(value).strip() == '':
        return None
    
    try:
        value_str = str(value).strip()
        if value_str.endswith('%'):
            # Remove % and convert to decimal (divide by 100)
            return float(value_str[:-1]) / 100.0
        else:
            # Assume it's already a decimal
            return float(value_str)
    except (ValueError, TypeError):
        return None


def get_csv_files() -> list:
    """Get list of CSV files to process based on WEEK_START and WEEK_END."""
    base_dir = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\games"
    csv_files = []
    
    for week in range(WEEK_START, WEEK_END + 1):
        week_dir = os.path.join(base_dir, f"week_{week}.0")
        clean_dir = os.path.join(week_dir, "clean")
        if os.path.exists(clean_dir):
            # Get both basic and advanced defense CSV files
            basic_pattern = os.path.join(clean_dir, "cleaned_*_gm_plyr_def_*.csv")
            adv_pattern = os.path.join(clean_dir, "cleaned_*_gm_plyr_adv_def_*.csv")
            
            basic_files = glob.glob(basic_pattern)
            adv_files = glob.glob(adv_pattern)
            
            # Group files by game (match by extracting game identifier without timestamps)
            games = {}
            
            for basic_file in basic_files:
                # Extract game identifier: everything before the timestamp
                basename = os.path.basename(basic_file)
                # Pattern: cleaned_Team1_Team2_wkX.0_2024_gm_plyr_def_TIMESTAMP.csv
                parts = basename.split('_gm_plyr_def_')
                if len(parts) == 2:
                    game_key = parts[0]  # Everything before '_gm_plyr_def_'
                    if game_key not in games:
                        games[game_key] = {}
                    games[game_key]['basic'] = basic_file
            
            for adv_file in adv_files:
                # Extract game identifier: everything before the timestamp
                basename = os.path.basename(adv_file)
                # Pattern: cleaned_Team1_Team2_wkX.0_2024_gm_plyr_adv_def_TIMESTAMP.csv
                parts = basename.split('_gm_plyr_adv_def_')
                if len(parts) == 2:
                    game_key = parts[0]  # Everything before '_gm_plyr_adv_def_'
                    if game_key not in games:
                        games[game_key] = {}
                    games[game_key]['advanced'] = adv_file
            
            # Only process games that have both file types
            for game_key, files in games.items():
                if 'basic' in files and 'advanced' in files:
                    csv_files.append((week, files['basic'], files['advanced']))
                else:
                    print(f"[WARNING] Week {week}: Missing file for game {game_key}")
                    
        else:
            print(f"[WARNING] Clean directory not found: {clean_dir}")
    
    return sorted(csv_files)


def process_csv_files(db: DatabaseConnector, basic_file: str, advanced_file: str, season_id: int) -> pd.DataFrame:
    """Process both basic and advanced CSV files and return consolidated DataFrame."""
    
    print(f"Processing basic file: {os.path.basename(basic_file)}")
    print(f"Processing advanced file: {os.path.basename(advanced_file)}")
    
    # Read both CSV files
    basic_df = pd.read_csv(basic_file, header=0)
    advanced_df = pd.read_csv(advanced_file, header=0)
    
    # Get week number from either file (they should be the same)
    if 'week' in basic_df.columns:
        week_num = basic_df['week'].iloc[0]
    elif 'week' in advanced_df.columns:
        week_num = advanced_df['week'].iloc[0]
    else:
        raise ValueError(f"Week column not found in either file")
    
    if pd.isna(week_num):
        raise ValueError(f"Week number is null")
    
    # Convert to int (week_num might be 1.0, 2.0, etc.)
    week_num = int(float(week_num))
    
    week_id = get_week_id(db, season_id, week_num)
    
    # Get unique team abbreviations (use basic file as primary)
    unique_teams = basic_df['Tm'].dropna().unique()
    if len(unique_teams) != 2:
        raise ValueError(f"Expected 2 teams in CSV file, found {len(unique_teams)}: {unique_teams}")
    
    team1_abrv, team2_abrv = unique_teams[0], unique_teams[1]
    game_id = get_game_id(db, season_id, week_id, team1_abrv, team2_abrv)
    
    # Process each player from basic file (primary source)
    processed_rows = []
    for basic_index, basic_row in basic_df.iterrows():
        try:
            # Get player info from basic file
            player_name = basic_row.get('Player', '')
            team_abrv = basic_row.get('Tm', '')
            
            if not player_name or player_name.strip() == '':
                continue
                
            # Skip rows where Tm column is null, empty, or just whitespace
            if pd.isna(team_abrv) or not team_abrv or str(team_abrv).strip() == '':
                print(f"[INFO] Skipping player {player_name} - no team information (Tm column is empty)")
                continue
                
            # Get player_id and team_id
            plyr_id = get_player_id(db, player_name, team_abrv, season_id)
            team_id = get_team_id(db, team_abrv)
            
            # Create processed row with foreign keys
            processed_row = {
                'plyr_id': plyr_id,
                'week_id': week_id,
                'game_id': game_id,
                'season_id': season_id,
                'team_id': team_id
            }
            
            # Map basic defensive stats from basic CSV
            basic_stat_mapping = {
                'plyr_gm_def_int': 'Def Interceptions Int',
                'plyr_gm_def_int_yds': 'Def Interceptions Yds', 
                'plyr_gm_def_int_td': 'Def Interceptions TD',
                'plyr_gm_def_int_lng': 'Def Interceptions Lng',
                'plyr_gm_pass_def': 'Def Interceptions PD',
                'plyr_gm_def_comb_tkl': 'Tackles Comb',
                'plyr_gm_def_solo_tkl': 'Tackles Solo',
                'plyr_gm_def_tkl_assist': 'Tackles Ast',
                'plyr_gm_def_tfl': 'Tackles TFL',
                'plyr_gm_def_qb_hit': 'Tackles QBHits',
                'plyr_gm_def_fmble_rec': 'Fumbles FR',
                'plyr_gm_def_fmbl_ret_yds': 'Fumbles Yds',
                'plyr_gm_def_fmbl_td': 'Fumbles TD',
                'plyr_gm_def_force_fmbl': 'Fumbles FF'
            }
            
            # Add basic defensive stats
            for db_col, csv_col in basic_stat_mapping.items():
                if csv_col in basic_df.columns:
                    value = basic_row.get(csv_col)
                    if pd.notna(value) and value != '' and str(value).strip() != '':
                        try:
                            processed_row[db_col] = int(float(value))
                        except (ValueError, TypeError):
                            processed_row[db_col] = None
                    else:
                        processed_row[db_col] = None
            
            # Get sacks from basic file if available (use float for half-sacks)
            if 'Sk' in basic_df.columns:
                sk_value = basic_row.get('Sk')
                if pd.notna(sk_value) and sk_value != '' and str(sk_value).strip() != '':
                    try:
                        processed_row['plyr_gm_def_sk'] = float(sk_value)
                    except (ValueError, TypeError):
                        processed_row['plyr_gm_def_sk'] = None
                else:
                    processed_row['plyr_gm_def_sk'] = None
            
            # Now find matching player in advanced file
            adv_match = advanced_df[
                (advanced_df['Player'] == player_name) & 
                (advanced_df['Tm'] == team_abrv)
            ]
            
            if not adv_match.empty:
                adv_row = adv_match.iloc[0]
                
                # Map advanced defensive stats
                advanced_stat_mapping = {
                    'plyr_gm_def_tgt': 'Tgt',
                    'plyr_gm_def_cmp': 'Cmp',
                    'plyr_gm_def_cmp_pct': 'Cmp%',  # Special handling for percentage
                    'plyr_gm_def_pass_yds': 'Yds',
                    'plyr_gm_def_pass_yds_cmp': 'Yds/Cmp',
                    'plyr_gm_def_pass_yds_tgt': 'Yds/Tgt',
                    'plyr_gm_def_pass_td': 'TD',
                    'plyr_gm_def_pass_rtg': 'Rat',
                    'plyr_gm_def_adot': 'DADOT',
                    'plyr_gm_def_ay': 'Air',
                    'plyr_gm_def_yac': 'YAC',
                    'plyr_gm_def_bltz': 'Bltz',
                    'plyr_gm_def_hrry': 'Hrry', 
                    'plyr_gm_def_qbkd': 'QBKD',
                    'plyr_gm_def_prss': 'Prss',
                    'plyr_gm_def_mtkl': 'MTkl',
                    'plyr_gm_def_mtkl_pct': 'MTkl%'  # Special handling for percentage
                }
                
                # Add advanced stats
                for db_col, csv_col in advanced_stat_mapping.items():
                    if csv_col in advanced_df.columns:
                        value = adv_row.get(csv_col)
                        
                        # Handle percentage columns specially
                        if db_col in ['plyr_gm_def_cmp_pct', 'plyr_gm_def_mtkl_pct']:
                            processed_row[db_col] = convert_percentage_to_float(value)
                        else:
                            # Handle regular numeric values
                            if pd.notna(value) and value != '' and str(value).strip() != '':
                                try:
                                    if db_col in ['plyr_gm_def_pass_yds_cmp', 'plyr_gm_def_pass_yds_tgt', 
                                                  'plyr_gm_def_pass_rtg', 'plyr_gm_def_adot']:
                                        processed_row[db_col] = float(value)  # Keep as float
                                    else:
                                        processed_row[db_col] = int(float(value))  # Convert to int
                                except (ValueError, TypeError):
                                    processed_row[db_col] = None
                            else:
                                processed_row[db_col] = None
                
                # Override sacks from advanced file if available and basic file didn't have it
                if 'Sk' in advanced_df.columns and 'plyr_gm_def_sk' not in processed_row:
                    sk_value = adv_row.get('Sk')
                    if pd.notna(sk_value) and sk_value != '' and str(sk_value).strip() != '':
                        try:
                            processed_row['plyr_gm_def_sk'] = float(sk_value)
                        except (ValueError, TypeError):
                            processed_row['plyr_gm_def_sk'] = None
            
            processed_rows.append(processed_row)
            
        except Exception as e:
            print(f"[WARNING] Error processing row {basic_index} for player {basic_row.get('Player', 'Unknown')}: {e}")
            continue
    
    if processed_rows:
        processed_df = pd.DataFrame(processed_rows)
        processed_df = handle_null_values(processed_df)
        
        # Ensure proper data types for foreign key columns
        fk_columns = ['plyr_id', 'week_id', 'game_id', 'season_id', 'team_id']
        for col in fk_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').astype('int64')
        
        # Handle integer columns that can be null
        nullable_int_columns = ['plyr_gm_def_int', 'plyr_gm_def_int_yds', 'plyr_gm_def_int_td', 
                               'plyr_gm_def_int_lng', 'plyr_gm_pass_def', 'plyr_gm_def_comb_tkl',
                               'plyr_gm_def_solo_tkl', 'plyr_gm_def_tkl_assist', 'plyr_gm_def_tfl',
                               'plyr_gm_def_qb_hit', 'plyr_gm_def_fmble_rec', 'plyr_gm_def_fmbl_ret_yds',
                               'plyr_gm_def_fmbl_td', 'plyr_gm_def_force_fmbl', 'plyr_gm_def_tgt',
                               'plyr_gm_def_cmp', 'plyr_gm_def_pass_yds', 'plyr_gm_def_pass_td',
                               'plyr_gm_def_ay', 'plyr_gm_def_yac', 'plyr_gm_def_bltz', 'plyr_gm_def_hrry',
                               'plyr_gm_def_qbkd', 'plyr_gm_def_prss', 'plyr_gm_def_mtkl']
        
        for col in nullable_int_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                processed_df[col] = processed_df[col].round().astype('Int64')
        
        # Handle float columns
        float_columns = ['plyr_gm_def_cmp_pct', 'plyr_gm_def_pass_yds_cmp', 'plyr_gm_def_pass_yds_tgt',
                        'plyr_gm_def_pass_rtg', 'plyr_gm_def_adot', 'plyr_gm_def_mtkl_pct', 'plyr_gm_def_sk']
        
        for col in float_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        return processed_df
    else:
        return pd.DataFrame()


def main():
    """Main function to process all CSV files."""
    
    print(f"Starting Consolidated Player Game Defense Data Import for {YEAR}")
    print(f"Processing weeks {WEEK_START} to {WEEK_END}")
    print("Consolidating basic and advanced defensive stats into single plyr_gm_def table")
    
    # Initialize database connection
    db = DatabaseConnector()
    if not db.connect():
        print("[FAIL] Failed to connect to database")
        return
    
    try:
        # Create table if it doesn't exist
        if not create_plyr_gm_def_table(db):
            return
        
        # Get season_id
        season_id = get_season_id(db, YEAR)
        print(f"Season ID for {YEAR}: {season_id}")
        
        # Get paired CSV files to process (basic + advanced)
        csv_file_pairs = get_csv_files()
        if not csv_file_pairs:
            print("[WARNING] No paired CSV files found to process")
            return
        
        print(f"Found {len(csv_file_pairs)} game pairs to process")
        
        total_processed = 0
        total_inserted = 0
        
        # Process each file pair
        for week, basic_file, advanced_file in csv_file_pairs:
            try:
                processed_df = process_csv_files(db, basic_file, advanced_file, season_id)
                
                if not processed_df.empty:
                    success = batch_upsert_data(db, 'plyr_gm_def', processed_df)
                    if success:
                        rows_in_file = len(processed_df)
                        total_processed += rows_in_file
                        total_inserted += rows_in_file
                        print(f"[OK] Week {week}: Processed {rows_in_file} consolidated player records")
                    else:
                        print(f"[FAIL] Week {week}: Failed to insert data")
                else:
                    print(f"[WARNING] Week {week}: No data to process")
                    
            except Exception as e:
                print(f"[ERROR] Error processing week {week} files: {e}")
                print(f"   Basic: {os.path.basename(basic_file)}")
                print(f"   Advanced: {os.path.basename(advanced_file)}")
                continue
        
        print(f"\nConsolidated Import Summary:")
        print(f"   Total game pairs processed: {len(csv_file_pairs)}")
        print(f"   Total player records processed: {total_processed}")
        print(f"   Total records inserted/updated: {total_inserted}")
        print("   Successfully consolidated basic and advanced defensive stats")
        
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
    finally:
        db.disconnect()


if __name__ == "__main__":
    main()