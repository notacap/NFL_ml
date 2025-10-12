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

def create_tm_def_pass_table(db: DatabaseConnector) -> bool:
    """Create tm_def_pass table if it doesn't exist"""
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS tm_def_pass (
        tm_def_pass_id INT AUTO_INCREMENT PRIMARY KEY,
        team_id INT,
        season_id INT,
        week_id INT,
        tm_def_pass_cmp INT,
        tm_def_pass_att INT,
        tm_def_pass_cmp_pct DECIMAL(7,4),
        tm_def_pass_yds INT,
        tm_def_pass_td INT,
        tm_def_pass_td_pct DECIMAL(5,4),
        tm_def_int INT,
        tm_def_pass_def INT,
        tm_def_int_pct DECIMAL(5,4),
        tm_def_pass_yds_att DECIMAL(7,4),
        tm_def_pass_yds_att_adj DECIMAL(7,4),
        tm_def_pass_ypc DECIMAL(7,4),
        tm_def_pass_ypg DECIMAL(7,4),
        tm_def_pass_rtg DECIMAL(7,4),
        tm_def_sk INT,
        tm_def_sk_yds INT,
        tm_def_qb_hit INT,
        tm_def_tfl INT,
        tm_def_sk_pct DECIMAL(5,4),
        tm_def_pass_net_yds_att DECIMAL(7,4),
        tm_def_pass_net_yds_att_adj DECIMAL(7,4),
        tm_def_pass_exp DECIMAL(7,4),
        UNIQUE KEY uk_tm_season (team_id, season_id, week_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (team_id) REFERENCES nfl_team(team_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id)
    );
    """
    
    return create_table_if_not_exists(db, "tm_def_pass", create_table_sql)

def get_tm_def_pass_csv_file() -> str:
    """Get the most recent tm_def_pass CSV file"""
    
    # Define base path using YEAR and WEEK from db_utils
    tm_def_pass_path = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\tm_def_pass\\week_{WEEK}"
    
    # Find tm_def_pass files
    tm_def_pass_pattern = os.path.join(tm_def_pass_path, "tm_def_pass_*.csv")
    tm_def_pass_files = sorted(glob.glob(tm_def_pass_pattern), key=os.path.getmtime, reverse=True)
    
    if not tm_def_pass_files:
        raise FileNotFoundError(f"No tm_def_pass CSV files found in {tm_def_pass_path}")
    
    print(f"Using tm_def_pass file: {tm_def_pass_files[0]}")
    
    return tm_def_pass_files[0]

def process_tm_def_pass_data(db: DatabaseConnector, csv_file: str, season_id: int, week_id: int) -> pd.DataFrame:
    """Process tm_def_pass data"""
    
    # Load tm_def_pass data
    print(f"Loading team defense passing data from {csv_file}...")
    tm_def_pass_df = load_csv_data(csv_file)
    if tm_def_pass_df.empty:
        raise ValueError("Failed to load tm_def_pass CSV data")
    
    # Filter out summary rows (keep only actual teams)
    tm_def_pass_df = tm_def_pass_df[~tm_def_pass_df['Tm'].str.contains('Avg|League|Total', na=False)]
    
    print(f"Loaded {len(tm_def_pass_df)} teams from defense passing file")
    
    processed_records = []
    
    for _, row in tm_def_pass_df.iterrows():
        try:
            team_name = row['Tm']
            print(f"Processing team: {team_name}")
            
            # Get team_id using the utility function
            team_id = get_team_id(db, team_name)
            
            # Helper functions for safe conversions
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
                """Convert percentage to decimal (62.2 -> 0.622)"""
                if pd.isna(value) or value == '':
                    return None
                try:
                    return float(value) / 100.0
                except (ValueError, TypeError):
                    return None
            
            record = {
                'team_id': team_id,
                'season_id': season_id,
                'week_id': week_id,
                
                # Team defense passing stats mapped from CSV columns
                'tm_def_pass_cmp': safe_convert_int(row['Cmp']),           # Completions allowed
                'tm_def_pass_att': safe_convert_int(row['Att']),           # Attempts allowed
                'tm_def_pass_cmp_pct': safe_convert_percentage(row['Cmp%']), # Completion percentage
                'tm_def_pass_yds': safe_convert_int(row['Yds']),           # Passing yards allowed
                'tm_def_pass_td': safe_convert_int(row['TD']),             # TDs allowed
                'tm_def_pass_td_pct': safe_convert_percentage(row['TD%']), # TD percentage
                'tm_def_int': safe_convert_int(row['Int']),                # Interceptions
                'tm_def_pass_def': safe_convert_int(row['PD']),            # Pass deflections
                'tm_def_int_pct': safe_convert_percentage(row['Int%']),    # Interception percentage
                'tm_def_pass_yds_att': safe_convert_float(row['Y/A']),     # Yards per attempt allowed
                'tm_def_pass_yds_att_adj': safe_convert_float(row['AY/A']), # Adjusted yards per attempt
                'tm_def_pass_ypc': safe_convert_float(row['Y/C']),         # Yards per completion allowed
                'tm_def_pass_ypg': safe_convert_float(row['Y/G']),         # Yards per game allowed
                'tm_def_pass_rtg': safe_convert_float(row['Rate']),        # Passer rating allowed
                'tm_def_sk': safe_convert_int(row['Sk']),                  # Sacks
                'tm_def_sk_yds': safe_convert_int(row['Yds.1']),           # Sack yards
                'tm_def_qb_hit': safe_convert_int(row['QBHits']),          # QB hits
                'tm_def_tfl': safe_convert_int(row['TFL']),                # Tackles for loss
                'tm_def_sk_pct': safe_convert_percentage(row['Sk%']),      # Sack percentage
                'tm_def_pass_net_yds_att': safe_convert_float(row['NY/A']), # Net yards per attempt
                'tm_def_pass_net_yds_att_adj': safe_convert_float(row['ANY/A']), # Adjusted net yards per attempt
                'tm_def_pass_exp': safe_convert_float(row['EXP'])          # Expected points
            }
            
            processed_records.append(record)
            
        except Exception as e:
            print(f"Error processing team {team_name}: {e}")
            continue
    
    # Convert to DataFrame
    final_df = pd.DataFrame(processed_records)
    
    # Handle null values
    final_df = handle_null_values(final_df)
    
    print(f"Successfully processed {len(final_df)} team defense passing records")
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
        if not create_tm_def_pass_table(db):
            print("Failed to create tm_def_pass table")
            return False
        
        # Get foreign key values
        season_id = get_season_id(db, YEAR)
        week_id = get_week_id(db, season_id, str(WEEK))
        
        print(f"Processing data for season {YEAR} (ID: {season_id}), week {WEEK} (ID: {week_id})")
        
        # Get CSV file
        csv_file = get_tm_def_pass_csv_file()
        
        # Process data
        processed_df = process_tm_def_pass_data(db, csv_file, season_id, week_id)
        
        if processed_df.empty:
            print("No data to insert")
            return False
        
        # Insert data using batch upsert
        print(f"Inserting {len(processed_df)} records into tm_def_pass table...")
        success = batch_upsert_data(db, 'tm_def_pass', processed_df)
        
        if success:
            print("[SUCCESS] Team defense passing data insertion completed successfully!")
            return True
        else:
            print("[FAILED] Team defense passing data insertion failed!")
            return False
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        return False
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    success = main()
    if success:
        print("\nSUCCESS: tm_def_pass.py script completed successfully!")
    else:
        print("\nFAILED: tm_def_pass.py script failed!")