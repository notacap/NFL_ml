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

def create_tm_def_vs_wr_table(db: DatabaseConnector) -> bool:
    """Create tm_def_vs_wr table if it doesn't exist"""
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS tm_def_vs_wr (
        tm_def_vs_wr_id INT AUTO_INCREMENT PRIMARY KEY,
        team_id INT,
        season_id INT,
        week_id INT,
        tm_def_wr_tgt INT,
        tm_def_wr_rec INT,
        tm_def_wr_yds INT,
        tm_def_wr_td INT,
        UNIQUE KEY uk_tm_season (team_id, season_id, week_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (team_id) REFERENCES nfl_team(team_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id)
    );
    """
    
    return create_table_if_not_exists(db, "tm_def_vs_wr", create_table_sql)

def get_tm_def_vs_wr_csv_file() -> str:
    """Get the most recent tm_def_vs_wr CSV file"""
    
    # Define base path using YEAR and WEEK from db_utils
    tm_def_vs_wr_path = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\tm_def_vs_wr\\week_{WEEK}\\clean"
    
    # Find tm_def_vs_wr files
    tm_def_vs_wr_pattern = os.path.join(tm_def_vs_wr_path, "cleaned_tm_def_vs_wr_*.csv")
    tm_def_vs_wr_files = sorted(glob.glob(tm_def_vs_wr_pattern), key=os.path.getmtime, reverse=True)
    
    if not tm_def_vs_wr_files:
        raise FileNotFoundError(f"No tm_def_vs_wr CSV files found in {tm_def_vs_wr_path}")
    
    print(f"Using tm_def_vs_wr file: {tm_def_vs_wr_files[0]}")
    
    return tm_def_vs_wr_files[0]

def process_tm_def_vs_wr_data(db: DatabaseConnector, csv_file: str, season_id: int, week_id: int) -> pd.DataFrame:
    """Process tm_def_vs_wr data"""
    
    # Load tm_def_vs_wr data
    print(f"Loading team defense vs WR data from {csv_file}...")
    tm_def_vs_wr_df = load_csv_data(csv_file)
    if tm_def_vs_wr_df.empty:
        raise ValueError("Failed to load tm_def_vs_wr CSV data")
    
    # Filter out summary rows (keep only actual teams)
    tm_def_vs_wr_df = tm_def_vs_wr_df[~tm_def_vs_wr_df['Tm'].str.contains('Avg|League|Total', na=False)]
    
    print(f"Loaded {len(tm_def_vs_wr_df)} teams from defense vs WR file")
    
    processed_records = []
    
    for _, row in tm_def_vs_wr_df.iterrows():
        try:
            team_name = row['Tm']
            print(f"Processing team: {team_name}")
            
            # Get team_id using the utility function
            team_id = get_team_id(db, team_name)
            
            # Process defense vs WR stats
            def safe_convert_int(value):
                if pd.isna(value) or value == '':
                    return None
                try:
                    return int(float(value))
                except (ValueError, TypeError):
                    return None
            
            record = {
                'team_id': team_id,
                'season_id': season_id,
                'week_id': week_id,
                
                # Defense vs WR stats mapped from CSV columns
                'tm_def_wr_tgt': safe_convert_int(row['Receiving Tgt']),  # WR targets allowed
                'tm_def_wr_rec': safe_convert_int(row['Receiving Rec']),  # WR receptions allowed
                'tm_def_wr_yds': safe_convert_int(row['Receiving Yds']),  # WR yards allowed
                'tm_def_wr_td': safe_convert_int(row['Receiving TD'])     # WR touchdowns allowed
            }
            
            processed_records.append(record)
            
        except Exception as e:
            print(f"Error processing team {team_name}: {e}")
            continue
    
    # Convert to DataFrame
    final_df = pd.DataFrame(processed_records)
    
    # Handle null values
    final_df = handle_null_values(final_df)
    
    print(f"Successfully processed {len(final_df)} team defense vs WR records")
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
        if not create_tm_def_vs_wr_table(db):
            print("Failed to create tm_def_vs_wr table")
            return False
        
        # Get foreign key values
        season_id = get_season_id(db, YEAR)
        week_id = get_week_id(db, season_id, str(WEEK))
        
        print(f"Processing data for season {YEAR} (ID: {season_id}), week {WEEK} (ID: {week_id})")
        
        # Get CSV file
        csv_file = get_tm_def_vs_wr_csv_file()
        
        # Process data
        processed_df = process_tm_def_vs_wr_data(db, csv_file, season_id, week_id)
        
        if processed_df.empty:
            print("No data to insert")
            return False
        
        # Insert data using batch upsert
        print(f"Inserting {len(processed_df)} records into tm_def_vs_wr table...")
        success = batch_upsert_data(db, 'tm_def_vs_wr', processed_df)
        
        if success:
            print("[SUCCESS] Team defense vs WR data insertion completed successfully!")
            return True
        else:
            print("[FAILED] Team defense vs WR data insertion failed!")
            return False
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        return False
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    success = main()
    if success:
        print("\nSUCCESS: tm_def_vs_wr.py script completed successfully!")
    else:
        print("\nFAILED: tm_def_vs_wr.py script failed!")