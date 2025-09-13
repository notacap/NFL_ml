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

def create_tm_rush_table(db: DatabaseConnector) -> bool:
    """Create tm_rush table if it doesn't exist"""
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS tm_rush (
        tm_rush_id INT AUTO_INCREMENT PRIMARY KEY,
        team_id INT,
        season_id INT,
        week_id INT,
        tm_rush_att INT,
        tm_rush_yds INT,
        tm_rush_td INT,
        tm_rush_lng INT,
        tm_rush_yds_att DECIMAL(7,4),
        tm_rush_yds_gm DECIMAL(7,4),
        tm_rush_fmbl INT,
        tm_rush_exp DECIMAL(7,4),
        tm_rush_first_dwn INT,
        tm_rush_ybc INT,
        tm_rush_ybc_att DECIMAL(7,4),
        tm_rush_yac INT,
        tm_rush_yac_att DECIMAL(7,4),
        tm_rush_brkn_tkl INT,
        tm_rush_att_brkn_tkl DECIMAL(7,4),
        UNIQUE KEY uk_tm_season (team_id, season_id, week_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (team_id) REFERENCES nfl_team(team_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id)
    );
    """
    
    return create_table_if_not_exists(db, "tm_rush", create_table_sql)

def get_tm_rush_csv_files() -> tuple:
    """Get the most recent tm_rush and tm_adv_rushing CSV files"""
    
    # Define base paths using YEAR and WEEK from db_utils
    tm_rush_path = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\tm_rush\\week_{WEEK}"
    tm_adv_rush_path = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\tm_adv_rushing\\week_{WEEK}"
    
    # Find tm_rush files
    tm_rush_pattern = os.path.join(tm_rush_path, "tm_rush_*.csv")
    tm_rush_files = sorted(glob.glob(tm_rush_pattern), key=os.path.getmtime, reverse=True)
    
    # Find tm_adv_rushing files  
    tm_adv_rush_pattern = os.path.join(tm_adv_rush_path, "tm_adv_rushing_*.csv")
    tm_adv_rush_files = sorted(glob.glob(tm_adv_rush_pattern), key=os.path.getmtime, reverse=True)
    
    if not tm_rush_files:
        raise FileNotFoundError(f"No tm_rush CSV files found in {tm_rush_path}")
    
    if not tm_adv_rush_files:
        raise FileNotFoundError(f"No tm_adv_rushing CSV files found in {tm_adv_rush_path}")
    
    print(f"Using tm_rush file: {tm_rush_files[0]}")
    print(f"Using tm_adv_rushing file: {tm_adv_rush_files[0]}")
    
    return tm_rush_files[0], tm_adv_rush_files[0]

def process_tm_rush_data(db: DatabaseConnector, tm_rush_file: str, tm_adv_rush_file: str, season_id: int, week_id: int) -> pd.DataFrame:
    """Process and consolidate tm_rush and tm_adv_rushing data"""
    
    # Load tm_rush data
    print(f"Loading basic team rushing data from {tm_rush_file}...")
    tm_rush_df = load_csv_data(tm_rush_file)
    if tm_rush_df.empty:
        raise ValueError("Failed to load tm_rush CSV data")
    
    # Load tm_adv_rushing data
    print(f"Loading advanced team rushing data from {tm_adv_rush_file}...")
    tm_adv_rush_df = load_csv_data(tm_adv_rush_file)
    if tm_adv_rush_df.empty:
        raise ValueError("Failed to load tm_adv_rushing CSV data")
    
    # Filter out summary rows from tm_rush (keep only actual teams)
    tm_rush_df = tm_rush_df[~tm_rush_df['Tm'].str.contains('Avg|League|Total', na=False)]
    
    print(f"Loaded {len(tm_rush_df)} teams from basic rushing file")
    print(f"Loaded {len(tm_adv_rush_df)} teams from advanced rushing file")
    
    # Merge data on team name
    merged_df = pd.merge(tm_rush_df, tm_adv_rush_df, on='Tm', how='inner', suffixes=('_basic', '_adv'))
    print(f"Merged data for {len(merged_df)} teams")
    
    processed_records = []
    
    for _, row in merged_df.iterrows():
        try:
            team_name = row['Tm']
            print(f"Processing team: {team_name}")
            
            # Get team_id using the utility function
            team_id = get_team_id(db, team_name)
            
            # Process basic rushing stats (from tm_rush file)
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
            
            record = {
                'team_id': team_id,
                'season_id': season_id,
                'week_id': week_id,
                
                # Basic rushing stats from tm_rush file
                'tm_rush_att': safe_convert_int(row['Att_basic']),  # Attempts (from basic file)
                'tm_rush_yds': safe_convert_int(row['Yds_basic']),  # Yards (from basic file)
                'tm_rush_td': safe_convert_int(row['TD_basic']),  # Touchdowns (from basic file)
                'tm_rush_lng': safe_convert_int(row['Lng']),  # Longest rush
                'tm_rush_yds_att': safe_convert_float(row['Y/A']),  # Yards per attempt
                'tm_rush_yds_gm': safe_convert_float(row['Y/G']),  # Yards per game
                'tm_rush_fmbl': safe_convert_int(row['Fmb']),  # Fumbles
                'tm_rush_exp': safe_convert_float(row['EXP']),  # Expected points
                
                # Advanced rushing stats from tm_adv_rushing file
                'tm_rush_first_dwn': safe_convert_int(row['1D']),  # First downs
                'tm_rush_ybc': safe_convert_int(row['YBC']),  # Yards before contact
                'tm_rush_ybc_att': safe_convert_float(row['YBC/Att']),  # YBC per attempt
                'tm_rush_yac': safe_convert_int(row['YAC']),  # Yards after contact
                'tm_rush_yac_att': safe_convert_float(row['YAC/Att']),  # YAC per attempt
                'tm_rush_brkn_tkl': safe_convert_int(row['BrkTkl']),  # Broken tackles
                'tm_rush_att_brkn_tkl': safe_convert_float(row['Att/Br'])  # Attempts per broken tackle
            }
            
            processed_records.append(record)
            
        except Exception as e:
            print(f"Error processing team {team_name}: {e}")
            continue
    
    # Convert to DataFrame
    final_df = pd.DataFrame(processed_records)
    
    # Handle null values
    final_df = handle_null_values(final_df)
    
    print(f"Successfully processed {len(final_df)} team rushing records")
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
        if not create_tm_rush_table(db):
            print("Failed to create tm_rush table")
            return False
        
        # Get foreign key values
        season_id = get_season_id(db, YEAR)
        week_id = get_week_id(db, season_id, str(WEEK))
        
        print(f"Processing data for season {YEAR} (ID: {season_id}), week {WEEK} (ID: {week_id})")
        
        # Get CSV files
        tm_rush_file, tm_adv_rush_file = get_tm_rush_csv_files()
        
        # Process data
        processed_df = process_tm_rush_data(db, tm_rush_file, tm_adv_rush_file, season_id, week_id)
        
        if processed_df.empty:
            print("No data to insert")
            return False
        
        # Insert data using batch upsert
        print(f"Inserting {len(processed_df)} records into tm_rush table...")
        success = batch_upsert_data(db, 'tm_rush', processed_df)
        
        if success:
            print("[SUCCESS] Team rushing data insertion completed successfully!")
            return True
        else:
            print("[FAILED] Team rushing data insertion failed!")
            return False
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        return False
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    success = main()
    if success:
        print("\nSUCCESS: tm_rush.py script completed successfully!")
    else:
        print("\nFAILED: tm_rush.py script failed!")