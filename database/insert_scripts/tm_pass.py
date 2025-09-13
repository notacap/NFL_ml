#!/usr/bin/env python3

import sys
import os
import pandas as pd
import glob
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_utils import (
    DatabaseConnector, 
    load_csv_data, 
    handle_null_values, 
    batch_upsert_data,
    get_season_id,
    get_week_id, 
    get_team_id,
    create_table_if_not_exists,
    YEAR, 
    WEEK
)

def create_tm_pass_table(db: DatabaseConnector) -> bool:
    """Create tm_pass table if it doesn't exist"""
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS tm_pass (
        tm_pass_id INT AUTO_INCREMENT PRIMARY KEY,
        team_id INT,
        season_id INT,
        week_id INT,
        tm_pass_cmp INT,
        tm_pass_att INT,
        tm_pass_cmp_pct DECIMAL(5,4),
        tm_pass_yds INT,
        tm_pass_td INT,
        tm_pass_td_pct DECIMAL(5,4),
        tm_pass_int INT,
        tm_pass_int_pct DECIMAL(5,4),
        tm_pass_lng INT,
        tm_pass_yds_att DECIMAL(7,4),
        tm_pass_adj_yds_att DECIMAL(7,4),
        tm_pass_yds_cmp DECIMAL(7,4),
        tm_pass_yds_gm DECIMAL(7,4),
        tm_pass_rtg DECIMAL(7,4),
        tm_pass_sk INT,
        tm_pass_sack_yds INT,
        tm_pass_sk_pct DECIMAL(5,4),
        tm_pass_net_yds_att DECIMAL(7,4),
        tm_pass_adj_net_yds_att DECIMAL(7,4),
        tm_fqc INT,
        tm_gwd INT,
        tm_pass_exp DECIMAL(7,4),
        tm_pass_bttd INT,
        tm_pass_thrwawy INT,
        tm_pass_spk INT,
        tm_pass_drp INT,
        tm_pass_drp_pct DECIMAL(5,4),
        tm_pass_off_tgt INT,
        tm_pass_off_tgt_pct DECIMAL(5,4),
        tm_pass_on_tgt INT,
        tm_pass_on_tgt_pct DECIMAL(5,4),
        tm_pass_adv_yds INT,
        tm_pass_iay INT,
        tm_pass_iay_att DECIMAL(7,4),
        tm_pass_cay INT,
        tm_pass_cay_cmp DECIMAL(7,4),
        tm_pass_cay_att DECIMAL(7,4),
        tm_pass_yac INT,
        tm_pass_yac_cmp DECIMAL(7,4),
        tm_rpo_play INT,
        tm_rpo_yds INT,
        tm_rpo_pass_att INT,
        tm_rpo_pass_yds INT,
        tm_rpo_rush_att INT,
        tm_rpo_rush_yds INT,
        tm_pa_att INT,
        tm_pa_yds INT,
        tm_pass_pkt_time TIME(3),
        tm_pass_bltz INT,
        tm_pass_hrry INT,
        tm_pass_hit INT,
        tm_pass_prss INT,
        tm_pass_prss_pct DECIMAL(5,4),
        tm_pass_scrmbl INT,
        tm_pass_yds_scrmbl DECIMAL(7,4),
        UNIQUE KEY uk_tm_season (team_id, season_id, week_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (team_id) REFERENCES nfl_team(team_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id)
    );
    """
    
    return create_table_if_not_exists(db, "tm_pass", create_table_sql)

def get_tm_pass_csv_files() -> tuple:
    """Get the most recent CSV files from all five required directories"""
    
    # Define base paths using YEAR and WEEK from db_utils
    base_dir = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}"
    
    tm_pass_path = os.path.join(base_dir, "tm_pass", f"week_{WEEK}")
    tm_accuracy_path = os.path.join(base_dir, "tm_adv_passing", "tm_accuracy", f"week_{WEEK}", "clean")
    tm_airyards_path = os.path.join(base_dir, "tm_adv_passing", "tm_airyards", f"week_{WEEK}", "clean")
    tm_playtype_path = os.path.join(base_dir, "tm_adv_passing", "tm_pass_playtype", f"week_{WEEK}", "clean")
    tm_pressure_path = os.path.join(base_dir, "tm_adv_passing", "tm_pass_pressure", f"week_{WEEK}", "clean")
    
    # Find files in each directory
    tm_pass_pattern = os.path.join(tm_pass_path, "tm_pass_*.csv")
    tm_accuracy_pattern = os.path.join(tm_accuracy_path, "cleaned_tm_accuracy_*.csv")
    tm_airyards_pattern = os.path.join(tm_airyards_path, "cleaned_tm_airyards_*.csv")
    tm_playtype_pattern = os.path.join(tm_playtype_path, "cleaned_tm_pass_playtype_*.csv")
    tm_pressure_pattern = os.path.join(tm_pressure_path, "cleaned_tm_pass_pressure_*.csv")
    
    # Get most recent files
    tm_pass_files = sorted(glob.glob(tm_pass_pattern), key=os.path.getmtime, reverse=True)
    tm_accuracy_files = sorted(glob.glob(tm_accuracy_pattern), key=os.path.getmtime, reverse=True)
    tm_airyards_files = sorted(glob.glob(tm_airyards_pattern), key=os.path.getmtime, reverse=True)
    tm_playtype_files = sorted(glob.glob(tm_playtype_pattern), key=os.path.getmtime, reverse=True)
    tm_pressure_files = sorted(glob.glob(tm_pressure_pattern), key=os.path.getmtime, reverse=True)
    
    # Validate all files exist
    if not tm_pass_files:
        raise FileNotFoundError(f"No tm_pass CSV files found in {tm_pass_path}")
    if not tm_accuracy_files:
        raise FileNotFoundError(f"No tm_accuracy CSV files found in {tm_accuracy_path}")
    if not tm_airyards_files:
        raise FileNotFoundError(f"No tm_airyards CSV files found in {tm_airyards_path}")
    if not tm_playtype_files:
        raise FileNotFoundError(f"No tm_pass_playtype CSV files found in {tm_playtype_path}")
    if not tm_pressure_files:
        raise FileNotFoundError(f"No tm_pass_pressure CSV files found in {tm_pressure_path}")
    
    files = (tm_pass_files[0], tm_accuracy_files[0], tm_airyards_files[0], 
             tm_playtype_files[0], tm_pressure_files[0])
    
    print(f"Using tm_pass file: {files[0]}")
    print(f"Using tm_accuracy file: {files[1]}")
    print(f"Using tm_airyards file: {files[2]}")
    print(f"Using tm_pass_playtype file: {files[3]}")
    print(f"Using tm_pass_pressure file: {files[4]}")
    
    return files

def process_tm_pass_data(db: DatabaseConnector, tm_pass_file: str, tm_accuracy_file: str, 
                        tm_airyards_file: str, tm_playtype_file: str, tm_pressure_file: str,
                        season_id: int, week_id: int) -> pd.DataFrame:
    """Process and consolidate team passing data from all five CSV files"""
    
    # Load all CSV files
    print(f"Loading basic team passing data from {tm_pass_file}...")
    tm_pass_df = load_csv_data(tm_pass_file)
    if tm_pass_df.empty:
        raise ValueError("Failed to load tm_pass CSV data")
    
    print(f"Loading team accuracy data from {tm_accuracy_file}...")
    tm_accuracy_df = load_csv_data(tm_accuracy_file)
    if tm_accuracy_df.empty:
        raise ValueError("Failed to load tm_accuracy CSV data")
    
    print(f"Loading team air yards data from {tm_airyards_file}...")
    tm_airyards_df = load_csv_data(tm_airyards_file)
    if tm_airyards_df.empty:
        raise ValueError("Failed to load tm_airyards CSV data")
    
    print(f"Loading team play type data from {tm_playtype_file}...")
    tm_playtype_df = load_csv_data(tm_playtype_file)
    if tm_playtype_df.empty:
        raise ValueError("Failed to load tm_playtype CSV data")
    
    print(f"Loading team pressure data from {tm_pressure_file}...")
    tm_pressure_df = load_csv_data(tm_pressure_file)
    if tm_pressure_df.empty:
        raise ValueError("Failed to load tm_pressure CSV data")
    
    # Filter out summary rows from tm_pass (keep only actual teams)
    tm_pass_df = tm_pass_df[~tm_pass_df['Tm'].str.contains('Avg|League|Total', na=False)]
    
    print(f"Loaded {len(tm_pass_df)} teams from basic passing file")
    print(f"Loaded {len(tm_accuracy_df)} teams from accuracy file")
    print(f"Loaded {len(tm_airyards_df)} teams from air yards file")
    print(f"Loaded {len(tm_playtype_df)} teams from play type file")
    print(f"Loaded {len(tm_pressure_df)} teams from pressure file")
    
    # Merge all dataframes on team name
    merged_df = tm_pass_df.merge(tm_accuracy_df, on='Tm', how='inner', suffixes=('', '_acc'))
    merged_df = merged_df.merge(tm_airyards_df, on='Tm', how='inner', suffixes=('', '_air'))
    merged_df = merged_df.merge(tm_playtype_df, on='Tm', how='inner', suffixes=('', '_play'))
    merged_df = merged_df.merge(tm_pressure_df, on='Tm', how='inner', suffixes=('', '_pres'))
    
    print(f"Merged data for {len(merged_df)} teams")
    
    processed_records = []
    
    for _, row in merged_df.iterrows():
        try:
            team_name = row['Tm']
            print(f"Processing team: {team_name}")
            
            # Get team_id using the utility function
            team_id = get_team_id(db, team_name)
            
            # Helper functions for data conversion
            def safe_convert_int(value):
                if pd.isna(value) or value == '':
                    return None
                try:
                    return int(float(value))
                except (ValueError, TypeError):
                    return None
            
            def safe_convert_float(value):
                if pd.isna(value) or value == '':
                    return None
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return None
            
            def safe_convert_percentage(value):
                if pd.isna(value) or value == '':
                    return None
                try:
                    # Handle percentage strings like "20.6%"
                    if isinstance(value, str) and value.endswith('%'):
                        return float(value.rstrip('%')) / 100.0
                    return float(value) / 100.0 if float(value) > 1 else float(value)
                except (ValueError, TypeError):
                    return None
            
            def safe_convert_time(value):
                if pd.isna(value) or value == '':
                    return None
                try:
                    # Convert decimal seconds to TIME format (HH:MM:SS.mmm)
                    seconds = float(value)
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    remaining_seconds = seconds % 60
                    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:06.3f}"
                except (ValueError, TypeError):
                    return None
            
            record = {
                'team_id': team_id,
                'season_id': season_id,
                'week_id': week_id,
                
                # Basic passing stats from tm_pass file
                'tm_pass_cmp': safe_convert_int(row['Cmp']),
                'tm_pass_att': safe_convert_int(row['Att']),
                'tm_pass_cmp_pct': safe_convert_percentage(row['Cmp%']),
                'tm_pass_yds': safe_convert_int(row['Yds']),
                'tm_pass_td': safe_convert_int(row['TD']),
                'tm_pass_td_pct': safe_convert_percentage(row['TD%']),
                'tm_pass_int': safe_convert_int(row['Int']),
                'tm_pass_int_pct': safe_convert_percentage(row['Int%']),
                'tm_pass_lng': safe_convert_int(row['Lng']),
                'tm_pass_yds_att': safe_convert_float(row['Y/A']),
                'tm_pass_adj_yds_att': safe_convert_float(row['AY/A']),
                'tm_pass_yds_cmp': safe_convert_float(row['Y/C']),
                'tm_pass_yds_gm': safe_convert_float(row['Y/G']),
                'tm_pass_rtg': safe_convert_float(row['Rate']),
                'tm_pass_sk': safe_convert_int(row['Sk']),
                'tm_pass_sack_yds': safe_convert_int(row['Yds.1']),  # Sack yards column
                'tm_pass_sk_pct': safe_convert_percentage(row['Sk%']),
                'tm_pass_net_yds_att': safe_convert_float(row['NY/A']),
                'tm_pass_adj_net_yds_att': safe_convert_float(row['ANY/A']),
                'tm_fqc': safe_convert_int(row['4QC']),
                'tm_gwd': safe_convert_int(row['GWD']),
                'tm_pass_exp': safe_convert_float(row['EXP']),
                
                # Accuracy stats from tm_accuracy file
                'tm_pass_bttd': safe_convert_int(row['Passing Bats']),
                'tm_pass_thrwawy': safe_convert_int(row['Passing ThAwy']),
                'tm_pass_spk': safe_convert_int(row['Passing Spikes']),
                'tm_pass_drp': safe_convert_int(row['Passing Drops']),
                'tm_pass_drp_pct': safe_convert_percentage(row['Passing Drop%']),
                'tm_pass_off_tgt': safe_convert_int(row['Passing BadTh']),  # Bad throws as off-target
                'tm_pass_off_tgt_pct': safe_convert_percentage(row['Passing Bad%']),
                'tm_pass_on_tgt': safe_convert_int(row['Passing OnTgt']),
                'tm_pass_on_tgt_pct': safe_convert_percentage(row['Passing OnTgt%']),
                
                # Air yards stats from tm_airyards file
                'tm_pass_adv_yds': safe_convert_int(row['Passing Yds_air']),  # Use passing yards from air yards file
                'tm_pass_iay': safe_convert_int(row['Passing IAY']),
                'tm_pass_iay_att': safe_convert_float(row['Passing IAY/PA']),
                'tm_pass_cay': safe_convert_int(row['Passing CAY']),
                'tm_pass_cay_cmp': safe_convert_float(row['Passing CAY/Cmp']),
                'tm_pass_cay_att': safe_convert_float(row['Passing CAY/PA']),
                'tm_pass_yac': safe_convert_int(row['Passing YAC']),
                'tm_pass_yac_cmp': safe_convert_float(row['Passing YAC/Cmp']),
                
                # Play type stats from tm_playtype file
                'tm_rpo_play': safe_convert_int(row['RPO Plays']),
                'tm_rpo_yds': safe_convert_int(row['RPO Yds']),
                'tm_rpo_pass_att': safe_convert_int(row['RPO PassAtt']),
                'tm_rpo_pass_yds': safe_convert_int(row['RPO PassYds']),
                'tm_rpo_rush_att': safe_convert_int(row['RPO RushAtt']),
                'tm_rpo_rush_yds': safe_convert_int(row['RPO RushYds']),
                'tm_pa_att': safe_convert_int(row['PlayAction PassAtt']),
                'tm_pa_yds': safe_convert_int(row['PlayAction PassYds']),
                
                # Pressure stats from tm_pressure file
                'tm_pass_pkt_time': safe_convert_time(row['Passing PktTime']),
                'tm_pass_bltz': safe_convert_int(row['Passing Bltz']),
                'tm_pass_hrry': safe_convert_int(row['Passing Hrry']),
                'tm_pass_hit': safe_convert_int(row['Passing Hits']),
                'tm_pass_prss': safe_convert_int(row['Passing Prss']),
                'tm_pass_prss_pct': safe_convert_percentage(row['Passing Prss%']),
                'tm_pass_scrmbl': safe_convert_int(row['Passing Scrm']),
                'tm_pass_yds_scrmbl': safe_convert_float(row['Passing Yds/Scr'])
            }
            
            processed_records.append(record)
            
        except Exception as e:
            print(f"Error processing team {team_name}: {e}")
            continue
    
    # Convert to DataFrame
    final_df = pd.DataFrame(processed_records)
    
    # Handle null values
    final_df = handle_null_values(final_df)
    
    print(f"Successfully processed {len(final_df)} team passing records")
    return final_df

def main():
    """Main execution function"""
    
    # Initialize database connection
    db = DatabaseConnector()
    
    try:
        # Connect to database
        if not db.connect():
            print("Failed to connect to database")
            return False
        
        print("Connected to database successfully")
        
        # Create table if it doesn't exist
        if not create_tm_pass_table(db):
            print("Failed to create tm_pass table")
            return False
        
        # Get foreign key values
        season_id = get_season_id(db, YEAR)
        week_id = get_week_id(db, season_id, str(WEEK))
        
        print(f"Processing data for season {YEAR} (ID: {season_id}), week {WEEK} (ID: {week_id})")
        
        # Get CSV files
        tm_pass_file, tm_accuracy_file, tm_airyards_file, tm_playtype_file, tm_pressure_file = get_tm_pass_csv_files()
        
        # Process data
        processed_df = process_tm_pass_data(db, tm_pass_file, tm_accuracy_file, tm_airyards_file, 
                                          tm_playtype_file, tm_pressure_file, season_id, week_id)
        
        if processed_df.empty:
            print("No data to insert")
            return False
        
        # Insert data using batch upsert
        print(f"Inserting {len(processed_df)} records into tm_pass table...")
        success = batch_upsert_data(db, 'tm_pass', processed_df)
        
        if success:
            print("[SUCCESS] Team passing data insertion completed successfully!")
            return True
        else:
            print("[FAILED] Team passing data insertion failed!")
            return False
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        return False
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    success = main()
    if success:
        print("\nSUCCESS: tm_pass.py script completed successfully!")
    else:
        print("\nFAILED: tm_pass.py script failed!")