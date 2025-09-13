#!/usr/bin/env python3

import os
import sys
import pandas as pd
import glob
from pathlib import Path

# Add the parent directory to the Python path to import db_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_utils import DatabaseConnector, YEAR, WEEK_START, WEEK_END, batch_upsert_data, handle_null_values, get_season_id, get_week_id, get_player_id, get_game_id


def create_adv_plyr_gm_pass_table(db: DatabaseConnector) -> bool:
    """Create the adv_plyr_gm_pass table if it doesn't exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS adv_plyr_gm_pass (
        adv_plyr_gm_pass_id INT PRIMARY KEY AUTO_INCREMENT,
        plyr_id INT,
        week_id INT,
        game_id INT,
        season_id INT,
        plyr_gm_pass_first_dwn INT,
        plyr_gm_pass_first_dwn_pct DECIMAL(6,3),
        plyr_gm_pass_iay INT,
        plyr_gm_pass_iay_att DECIMAL(5,2),
        plyr_gm_pass_cay INT,
        plyr_gm_pass_cay_cmp DECIMAL(5,2),
        plyr_gm_pass_cay_att DECIMAL(5,2),
        plyr_gm_pass_yac INT,
        plyr_gm_pass_yac_cmp DECIMAL(5,2),
        plyr_gm_pass_drp INT,
        plyr_gm_pass_drp_pct DECIMAL(6,3),
        plyr_gm_pass_off_tgt INT,
        plyr_gm_pass_off_tgt_pct DECIMAL(6,3),
        plyr_gm_pass_bltz INT,
        plyr_gm_pass_hrry INT,
        plyr_gm_pass_hit INT,
        plyr_gm_pass_prss INT,
        plyr_gm_pass_prss_pct DECIMAL(6,3),
        plyr_gm_pass_scrmbl_tgt INT,
        plyr_gm_pass_yds_scrmbl DECIMAL(5,2),
        FOREIGN KEY (plyr_id) REFERENCES plyr(plyr_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (game_id) REFERENCES nfl_game(game_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        UNIQUE KEY uk_adv_plyr_gm_pass (plyr_id, game_id)
    );
    """
    
    try:
        success = db.execute_query(create_table_sql)
        if success:
            print("[OK] adv_plyr_gm_pass table created/verified successfully")
            return True
        else:
            print("[FAIL] Failed to create adv_plyr_gm_pass table")
            return False
    except Exception as e:
        print(f"[ERROR] Error creating adv_plyr_gm_pass table: {e}")
        return False


def convert_percentage(pct_str) -> float:
    """Convert percentage string (e.g., '10.0%') to decimal (0.100) with 3 decimal places."""
    if pd.isna(pct_str) or pct_str == '' or str(pct_str).strip() == '':
        return None
    try:
        # Remove % symbol and convert to decimal with 3 decimal places
        return round(float(str(pct_str).rstrip('%')) / 100.0, 3)
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
    
    # Convert to int (week_num might be 4.0, 12.0, etc.)
    week_num = int(float(week_num))
    
    week_id = get_week_id(db, season_id, week_num)
    
    # Get unique team abbreviations from the Tm column
    unique_teams = df['Tm'].unique()
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
                
            # Get player_id
            plyr_id = get_player_id(db, player_name, team_abrv, season_id)
            
            # Create processed row
            processed_row = {
                'plyr_id': plyr_id,
                'week_id': week_id,
                'game_id': game_id,
                'season_id': season_id
            }
            
            # Map database columns to CSV columns
            stat_mapping = {
                'plyr_gm_pass_first_dwn': '1D',
                'plyr_gm_pass_first_dwn_pct': '1D%',
                'plyr_gm_pass_iay': 'IAY',
                'plyr_gm_pass_iay_att': 'IAY/PA',
                'plyr_gm_pass_cay': 'CAY',
                'plyr_gm_pass_cay_cmp': 'CAY/Cmp',
                'plyr_gm_pass_cay_att': 'CAY/PA',
                'plyr_gm_pass_yac': 'YAC',
                'plyr_gm_pass_yac_cmp': 'YAC/Cmp',
                'plyr_gm_pass_drp': 'Drops',
                'plyr_gm_pass_drp_pct': 'Drop%',
                'plyr_gm_pass_off_tgt': 'BadTh',
                'plyr_gm_pass_off_tgt_pct': 'Bad%',
                'plyr_gm_pass_bltz': 'Bltz',
                'plyr_gm_pass_hrry': 'Hrry',
                'plyr_gm_pass_hit': 'Hits',
                'plyr_gm_pass_prss': 'Prss',
                'plyr_gm_pass_prss_pct': 'Prss%',
                'plyr_gm_pass_scrmbl_tgt': 'Scrm',
                'plyr_gm_pass_yds_scrmbl': 'Yds/Scr'
            }
            
            # Add advanced passing stats
            for db_col, csv_col in stat_mapping.items():
                if csv_col in df.columns:
                    value = row.get(csv_col)
                    
                    # Handle percentage columns specially
                    if db_col.endswith('_pct'):
                        processed_row[db_col] = convert_percentage(value)
                    else:
                        if pd.notna(value) and value != '' and str(value).strip() != '':
                            try:
                                if db_col in ['plyr_gm_pass_iay_att', 'plyr_gm_pass_cay_cmp', 'plyr_gm_pass_cay_att', 
                                             'plyr_gm_pass_yac_cmp', 'plyr_gm_pass_yds_scrmbl']:
                                    processed_row[db_col] = float(value)
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
        
        # Ensure proper data types
        int_columns = ['plyr_id', 'week_id', 'game_id', 'season_id', 'plyr_gm_pass_first_dwn',
                      'plyr_gm_pass_iay', 'plyr_gm_pass_cay', 'plyr_gm_pass_yac', 'plyr_gm_pass_drp',
                      'plyr_gm_pass_off_tgt', 'plyr_gm_pass_bltz', 'plyr_gm_pass_hrry', 
                      'plyr_gm_pass_hit', 'plyr_gm_pass_prss', 'plyr_gm_pass_scrmbl_tgt']
        
        float_columns = ['plyr_gm_pass_first_dwn_pct', 'plyr_gm_pass_iay_att', 'plyr_gm_pass_cay_cmp',
                        'plyr_gm_pass_cay_att', 'plyr_gm_pass_yac_cmp', 'plyr_gm_pass_drp_pct',
                        'plyr_gm_pass_off_tgt_pct', 'plyr_gm_pass_prss_pct', 'plyr_gm_pass_yds_scrmbl']
        
        for col in int_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                
        for col in float_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        return processed_df
    else:
        return pd.DataFrame()


def get_csv_files() -> list:
    """Get list of CSV files to process based on WEEK_START and WEEK_END."""
    base_dir = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\games"
    csv_files = []
    
    for week in range(WEEK_START, WEEK_END + 1):
        week_dir = os.path.join(base_dir, f"week_{week}.0")
        clean_dir = os.path.join(week_dir, "clean")
        if os.path.exists(clean_dir):
            pattern = os.path.join(clean_dir, "cleaned_*_gm_plyr_adv_passing_*.csv")
            week_files = glob.glob(pattern)
            for file_path in week_files:
                csv_files.append((week, file_path))
        else:
            print(f"[WARNING] Clean directory not found: {clean_dir}")
    
    return sorted(csv_files)


def main():
    """Main function to process all CSV files."""
    
    print(f"Starting Advanced Player Game Passing Data Import for {YEAR}")
    print(f"Processing weeks {WEEK_START} to {WEEK_END}")
    
    # Initialize database connection
    db = DatabaseConnector()
    if not db.connect():
        print("[FAIL] Failed to connect to database")
        return
    
    try:
        # Create table if it doesn't exist
        if not create_adv_plyr_gm_pass_table(db):
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
                    success = batch_upsert_data(db, 'adv_plyr_gm_pass', processed_df)
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