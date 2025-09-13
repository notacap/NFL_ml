#!/usr/bin/env python3

import sys
import os
import pandas as pd
import glob
from pathlib import Path
import re
from datetime import time

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

def create_tm_def_dr_against_avg_table(db: DatabaseConnector) -> bool:
    """Create tm_def_dr_against_avg table if it doesn't exist"""
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS tm_def_dr_against_avg (
        tm_def_dr_against_avg_id INT AUTO_INCREMENT PRIMARY KEY,
        team_id INT,
        season_id INT,
        week_id INT,
        tm_def_dr INT,
        tm_def_opp_avg_play_dr DECIMAL(7,4),
        tm_def_opp_net_yds_dr DECIMAL(7,4),
        tm_def_avg_field_pos DECIMAL(7,4),
        tm_def_avg_dr_time TIME,
        tm_def_avg_pt_dr DECIMAL(7,4),
        UNIQUE KEY uk_tm_season (team_id, season_id, week_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (team_id) REFERENCES nfl_team(team_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id)
    );
    """
    
    return create_table_if_not_exists(db, "tm_def_dr_against_avg", create_table_sql)

def get_tm_def_dr_against_avg_csv_file() -> str:
    """Get the most recent tm_def_drives_against CSV file"""
    
    # Define base path using YEAR and WEEK from db_utils
    tm_def_dr_against_path = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\tm_def_drives_against\\week_{WEEK}\\clean"
    
    # Find tm_def_drives_against files
    tm_def_dr_against_pattern = os.path.join(tm_def_dr_against_path, "cleaned_tm_def_drives_against_*.csv")
    tm_def_dr_against_files = sorted(glob.glob(tm_def_dr_against_pattern), key=os.path.getmtime, reverse=True)
    
    if not tm_def_dr_against_files:
        raise FileNotFoundError(f"No tm_def_drives_against CSV files found in {tm_def_dr_against_path}")
    
    print(f"Using tm_def_drives_against file: {tm_def_dr_against_files[0]}")
    
    return tm_def_dr_against_files[0]

def parse_average_drive_start(drive_start_value: str) -> float:
    """Parse 'Own XX.X' format and return the float value"""
    if pd.isna(drive_start_value) or drive_start_value == '':
        return None
    
    try:
        # Extract the numeric value after "Own "
        if isinstance(drive_start_value, str) and drive_start_value.startswith('Own '):
            numeric_part = drive_start_value.replace('Own ', '').strip()
            return float(numeric_part)
        else:
            # If it doesn't match expected format, try to convert directly
            return float(drive_start_value)
    except (ValueError, TypeError):
        print(f"Warning: Could not parse drive start value: {drive_start_value}")
        return None

def convert_time_to_seconds(time_str: str) -> time:
    """Convert 'M:SS' or 'MM:SS' format to TIME object"""
    if pd.isna(time_str) or time_str == '':
        return None
    
    try:
        # Parse time string (format: M:SS or MM:SS)
        time_parts = time_str.split(':')
        if len(time_parts) == 2:
            minutes = int(time_parts[0])
            seconds = int(time_parts[1])
            return time(hour=0, minute=minutes, second=seconds)
        else:
            print(f"Warning: Unexpected time format: {time_str}")
            return None
    except (ValueError, TypeError):
        print(f"Warning: Could not parse time value: {time_str}")
        return None

def process_tm_def_dr_against_avg_data(db: DatabaseConnector, csv_file: str, season_id: int, week_id: int) -> pd.DataFrame:
    """Process tm_def_drives_against data"""
    
    # Load tm_def_drives_against data
    print(f"Loading team defense drives against data from {csv_file}...")
    tm_def_dr_against_df = load_csv_data(csv_file)
    if tm_def_dr_against_df.empty:
        raise ValueError("Failed to load tm_def_drives_against CSV data")
    
    # Filter out summary rows (keep only actual teams)
    # Skip League Total row which has empty Rk field
    tm_def_dr_against_df = tm_def_dr_against_df[~tm_def_dr_against_df['Tm'].str.contains('Avg|League|Total', na=False)]
    tm_def_dr_against_df = tm_def_dr_against_df[tm_def_dr_against_df['Rk'].notna()]
    
    print(f"Loaded {len(tm_def_dr_against_df)} teams from defense drives against file")
    
    processed_records = []
    
    for _, row in tm_def_dr_against_df.iterrows():
        try:
            team_name = row['Tm']
            print(f"Processing team: {team_name}")
            
            # Get team_id using the utility function
            team_id = get_team_id(db, team_name)
            
            # Process defense drives against stats
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
                
                # Defense drives against stats mapped from CSV columns
                'tm_def_dr': safe_convert_int(row['#Dr']),  # Number of drives
                'tm_def_opp_avg_play_dr': safe_convert_float(row['Average Drive Plays']),  # Average plays per drive
                'tm_def_opp_net_yds_dr': safe_convert_float(row['Average Drive Yds']),  # Average yards per drive
                'tm_def_avg_field_pos': parse_average_drive_start(row['Average Drive Start']),  # Parse "Own XX.X" format
                'tm_def_avg_dr_time': convert_time_to_seconds(row['Average Drive Time']),  # Convert time format
                'tm_def_avg_pt_dr': safe_convert_float(row['Average Drive Pts'])  # Average points per drive
            }
            
            processed_records.append(record)
            
        except Exception as e:
            print(f"Error processing team {team_name}: {e}")
            continue
    
    # Convert to DataFrame
    final_df = pd.DataFrame(processed_records)
    
    # Handle null values
    final_df = handle_null_values(final_df)
    
    print(f"Successfully processed {len(final_df)} team defense drives against records")
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
        if not create_tm_def_dr_against_avg_table(db):
            print("Failed to create tm_def_dr_against_avg table")
            return False
        
        # Get foreign key values
        season_id = get_season_id(db, YEAR)
        week_id = get_week_id(db, season_id, str(WEEK))
        
        print(f"Processing data for season {YEAR} (ID: {season_id}), week {WEEK} (ID: {week_id})")
        
        # Get CSV file
        csv_file = get_tm_def_dr_against_avg_csv_file()
        
        # Process data
        processed_df = process_tm_def_dr_against_avg_data(db, csv_file, season_id, week_id)
        
        if processed_df.empty:
            print("No data to insert")
            return False
        
        # Insert data using batch upsert
        print(f"Inserting {len(processed_df)} records into tm_def_dr_against_avg table...")
        success = batch_upsert_data(db, 'tm_def_dr_against_avg', processed_df)
        
        if success:
            print("[SUCCESS] Team defense drives against data insertion completed successfully!")
            return True
        else:
            print("[FAILED] Team defense drives against data insertion failed!")
            return False
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        return False
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    success = main()
    if success:
        print("\nSUCCESS: tm_def_dr_against_avg.py script completed successfully!")
    else:
        print("\nFAILED: tm_def_dr_against_avg.py script failed!")