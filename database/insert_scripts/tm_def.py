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

def create_tm_def_table(db: DatabaseConnector) -> bool:
    """Create tm_def table if it doesn't exist"""
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS tm_def (
        tm_def_id INT AUTO_INCREMENT PRIMARY KEY,
        team_id INT,
        season_id INT,
        week_id INT,
        tm_def_pts_allwd INT,
        tm_def_yds INT,
        tm_def_ply INT,
        tm_def_ypp DECIMAL(7,4),
        tm_def_tkawy INT,
        tm_def_first_dwn INT,
        tm_def_pass_first_dwn INT,
        tm_def_rush_first_dwn INT,
        tm_def_penalty INT,
        tm_def_penalty_yds INT,
        tm_def_penalty_first_dwn INT,
        tm_def_score_pct DECIMAL(5,4),
        tm_def_to_pct DECIMAL(5,4),
        tm_def_exp DECIMAL(7,4),
        tm_def_bltz INT,
        tm_def_bltz_pct DECIMAL(5,4),
        tm_def_hrry INT,
        tm_def_hrry_pct DECIMAL(5,4),
        tm_def_qbkd INT,
        tm_def_qbkd_pct DECIMAL(5,4),
        tm_def_prss INT,
        tm_def_prss_pct DECIMAL(5,4),
        tm_def_missed_tkl INT,
        UNIQUE KEY uk_tm_season (team_id, season_id, week_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (team_id) REFERENCES nfl_team(team_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id)
    );
    """
    
    return create_table_if_not_exists(db, "tm_def", create_table_sql)

def get_tm_def_csv_files() -> tuple:
    """Get the most recent tm_def and tm_adv_def CSV files"""
    
    # Define base paths using YEAR and WEEK from db_utils
    tm_def_path = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\tm_def\\week_{WEEK}\\clean"
    tm_adv_def_path = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\tm_adv_def\\week_{WEEK}"
    
    # Find tm_def files
    tm_def_pattern = os.path.join(tm_def_path, "cleaned_tm_def_*.csv")
    tm_def_files = sorted(glob.glob(tm_def_pattern), key=os.path.getmtime, reverse=True)
    
    # Find tm_adv_def files  
    tm_adv_def_pattern = os.path.join(tm_adv_def_path, "tm_adv_def_*.csv")
    tm_adv_def_files = sorted(glob.glob(tm_adv_def_pattern), key=os.path.getmtime, reverse=True)
    
    if not tm_def_files:
        raise FileNotFoundError(f"No tm_def CSV files found in {tm_def_path}")
    
    if not tm_adv_def_files:
        raise FileNotFoundError(f"No tm_adv_def CSV files found in {tm_adv_def_path}")
    
    print(f"Using tm_def file: {tm_def_files[0]}")
    print(f"Using tm_adv_def file: {tm_adv_def_files[0]}")
    
    return tm_def_files[0], tm_adv_def_files[0]

def process_tm_def_data(db: DatabaseConnector, tm_def_file: str, tm_adv_def_file: str, season_id: int, week_id: int) -> pd.DataFrame:
    """Process and consolidate tm_def and tm_adv_def data"""
    
    # Load tm_def data
    print(f"Loading basic team defense data from {tm_def_file}...")
    tm_def_df = load_csv_data(tm_def_file)
    if tm_def_df.empty:
        raise ValueError("Failed to load tm_def CSV data")
    
    # Load tm_adv_def data
    print(f"Loading advanced team defense data from {tm_adv_def_file}...")
    tm_adv_def_df = load_csv_data(tm_adv_def_file)
    if tm_adv_def_df.empty:
        raise ValueError("Failed to load tm_adv_def CSV data")
    
    # Filter out summary rows from tm_def (keep only actual teams)
    tm_def_df = tm_def_df[~tm_def_df['Tm'].str.contains('Avg|League|Total', na=False)]
    
    print(f"Loaded {len(tm_def_df)} teams from basic defense file")
    print(f"Loaded {len(tm_adv_def_df)} teams from advanced defense file")
    
    # Merge data on team name
    merged_df = pd.merge(tm_def_df, tm_adv_def_df, on='Tm', how='inner', suffixes=('_basic', '_adv'))
    print(f"Merged data for {len(merged_df)} teams")
    
    processed_records = []
    
    for _, row in merged_df.iterrows():
        try:
            team_name = row['Tm']
            print(f"Processing team: {team_name}")
            
            # Get team_id using the utility function
            team_id = get_team_id(db, team_name)
            
            # Process basic defense stats (from tm_def file)
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
            
            record = {
                'team_id': team_id,
                'season_id': season_id,
                'week_id': week_id,
                
                # Basic defense stats from tm_def file
                'tm_def_pts_allwd': safe_convert_int(row['PA']),  # Points Allowed
                'tm_def_yds': safe_convert_int(row['Yds_basic']),  # Total Yards (from basic file)
                'tm_def_ply': safe_convert_int(row['Tot Yds & TO Ply']),  # Plays
                'tm_def_ypp': safe_convert_float(row['Tot Yds & TO Y/P']),  # Yards per Play
                'tm_def_tkawy': safe_convert_int(row['Tot Yds & TO TO']),  # Takeaways
                'tm_def_first_dwn': safe_convert_int(row['1stD']),  # First Downs Allowed
                'tm_def_pass_first_dwn': safe_convert_int(row['Passing 1stD']),  # Passing First Downs
                'tm_def_rush_first_dwn': safe_convert_int(row['Rushing 1stD']),  # Rushing First Downs  
                'tm_def_penalty': safe_convert_int(row['Penalties Pen']),  # Penalties
                'tm_def_penalty_yds': safe_convert_int(row['Penalties Yds']),  # Penalty Yards
                'tm_def_penalty_first_dwn': safe_convert_int(row['Penalties 1stPy']),  # Penalty First Downs
                'tm_def_score_pct': safe_convert_percentage(row['Sc%']),  # Score Percentage
                'tm_def_to_pct': safe_convert_percentage(row['TO%']),  # Turnover Percentage
                'tm_def_exp': safe_convert_float(row['EXP']),  # Expected Points
                
                # Advanced defense stats from tm_adv_def file
                'tm_def_bltz': safe_convert_int(row['Bltz']),  # Blitzes
                'tm_def_bltz_pct': safe_convert_percentage(row['Bltz%']),  # Blitz Percentage
                'tm_def_hrry': safe_convert_int(row['Hrry']),  # Hurries
                'tm_def_hrry_pct': safe_convert_percentage(row['Hrry%']),  # Hurry Percentage
                'tm_def_qbkd': safe_convert_int(row['QBKD']),  # QB Knockdowns
                'tm_def_qbkd_pct': safe_convert_percentage(row['QBKD%']),  # QB Knockdown Percentage
                'tm_def_prss': safe_convert_int(row['Prss']),  # Pressures
                'tm_def_prss_pct': safe_convert_percentage(row['Prss%']),  # Pressure Percentage
                'tm_def_missed_tkl': safe_convert_int(row['MTkl'])  # Missed Tackles
            }
            
            processed_records.append(record)
            
        except Exception as e:
            print(f"Error processing team {team_name}: {e}")
            continue
    
    # Convert to DataFrame
    final_df = pd.DataFrame(processed_records)
    
    # Handle null values
    final_df = handle_null_values(final_df)
    
    print(f"Successfully processed {len(final_df)} team defense records")
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
        if not create_tm_def_table(db):
            print("Failed to create tm_def table")
            return False
        
        # Get foreign key values
        season_id = get_season_id(db, YEAR)
        week_id = get_week_id(db, season_id, str(WEEK))
        
        print(f"Processing data for season {YEAR} (ID: {season_id}), week {WEEK} (ID: {week_id})")
        
        # Get CSV files
        tm_def_file, tm_adv_def_file = get_tm_def_csv_files()
        
        # Process data
        processed_df = process_tm_def_data(db, tm_def_file, tm_adv_def_file, season_id, week_id)
        
        if processed_df.empty:
            print("No data to insert")
            return False
        
        # Insert data using batch upsert
        print(f"Inserting {len(processed_df)} records into tm_def table...")
        success = batch_upsert_data(db, 'tm_def', processed_df)
        
        if success:
            print("[SUCCESS] Team defense data insertion completed successfully!")
            return True
        else:
            print("[FAILED] Team defense data insertion failed!")
            return False
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        return False
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    success = main()
    if success:
        print("\nSUCCESS: tm_def.py script completed successfully!")
    else:
        print("\nFAILED: tm_def.py script failed!")