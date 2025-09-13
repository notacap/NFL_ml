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

def create_tm_conv_table(db: DatabaseConnector) -> bool:
    """Create tm_conv table if it doesn't exist"""
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS tm_conv (
        tm_conv_id INT AUTO_INCREMENT PRIMARY KEY,
        team_id INT,
        season_id INT,
        week_id INT,
        tm_third_dwn_att INT,
        tm_third_dwn_conv INT,
        tm_third_dwn_conv_pct DECIMAL(5,4),
        tm_fourth_dwn_att INT,
        tm_fourth_dwn_conv INT,
        tm_fourth_dwn_conv_pct DECIMAL(5,4),
        tm_rz_att INT,
        tm_rz_td INT,
        tm_rz_conv_pct DECIMAL(5,4),
        UNIQUE KEY uk_tm_season (team_id, season_id, week_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (team_id) REFERENCES nfl_team(team_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id)
    );
    """
    
    return create_table_if_not_exists(db, "tm_conv", create_table_sql)

def get_tm_conv_csv_file() -> str:
    """Get the most recent tm_conv CSV file"""
    
    # Define base path using YEAR and WEEK from db_utils
    tm_conv_path = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\tm_conv\\week_{WEEK}\\clean"
    
    # Find tm_conv files
    tm_conv_pattern = os.path.join(tm_conv_path, "cleaned_tm_conv_*.csv")
    tm_conv_files = sorted(glob.glob(tm_conv_pattern), key=os.path.getmtime, reverse=True)
    
    if not tm_conv_files:
        raise FileNotFoundError(f"No tm_conv CSV files found in {tm_conv_path}")
    
    print(f"Using tm_conv file: {tm_conv_files[0]}")
    
    return tm_conv_files[0]

def process_tm_conv_data(db: DatabaseConnector, tm_conv_file: str, season_id: int, week_id: int) -> pd.DataFrame:
    """Process team conversion data"""
    
    # Load tm_conv data
    print(f"Loading team conversion data from {tm_conv_file}...")
    tm_conv_df = load_csv_data(tm_conv_file)
    if tm_conv_df.empty:
        raise ValueError("Failed to load tm_conv CSV data")
    
    # Filter out summary rows (keep only actual teams)
    tm_conv_df = tm_conv_df[~tm_conv_df['Tm'].str.contains('League|Total|Avg', na=False)]
    
    print(f"Loaded {len(tm_conv_df)} teams from conversion data file")
    
    processed_records = []
    
    for _, row in tm_conv_df.iterrows():
        try:
            team_name = row['Tm']
            print(f"Processing team: {team_name}")
            
            # Get team_id using the utility function
            team_id = get_team_id(db, team_name)
            
            # Helper functions for data conversion
            def safe_convert_int(value):
                """Convert value to int, handling NaN and empty strings"""
                if pd.isna(value) or value == '':
                    return None
                try:
                    return int(float(value))
                except (ValueError, TypeError):
                    return None
            
            def safe_convert_percentage(value):
                """Convert percentage string to decimal"""
                if pd.isna(value) or value == '':
                    return None
                try:
                    # Handle percentage strings like "48.5%"
                    if isinstance(value, str) and value.endswith('%'):
                        return float(value.rstrip('%')) / 100.0
                    # Handle raw decimal values
                    return float(value) / 100.0 if float(value) > 1 else float(value)
                except (ValueError, TypeError):
                    return None
            
            record = {
                'team_id': team_id,
                'season_id': season_id,
                'week_id': week_id,
                
                # Third down conversion stats
                'tm_third_dwn_att': safe_convert_int(row['Downs 3DAtt']),
                'tm_third_dwn_conv': safe_convert_int(row['Downs 3DConv']),
                'tm_third_dwn_conv_pct': safe_convert_percentage(row['Downs 3D%']),
                
                # Fourth down conversion stats
                'tm_fourth_dwn_att': safe_convert_int(row['Downs 4DAtt']),
                'tm_fourth_dwn_conv': safe_convert_int(row['Downs 4DConv']),
                'tm_fourth_dwn_conv_pct': safe_convert_percentage(row['Downs 4D%']),
                
                # Red zone conversion stats
                'tm_rz_att': safe_convert_int(row['Red Zone RZAtt']),
                'tm_rz_td': safe_convert_int(row['Red Zone RZTD']),
                'tm_rz_conv_pct': safe_convert_percentage(row['Red Zone RZPct'])
            }
            
            processed_records.append(record)
            
        except Exception as e:
            print(f"Error processing team {team_name}: {e}")
            continue
    
    # Convert to DataFrame
    final_df = pd.DataFrame(processed_records)
    
    # Handle null values
    final_df = handle_null_values(final_df)
    
    print(f"Successfully processed {len(final_df)} team conversion records")
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
        if not create_tm_conv_table(db):
            print("Failed to create tm_conv table")
            return False
        
        # Get foreign key values
        season_id = get_season_id(db, YEAR)
        week_id = get_week_id(db, season_id, str(WEEK))
        
        print(f"Processing data for season {YEAR} (ID: {season_id}), week {WEEK} (ID: {week_id})")
        
        # Get CSV file
        tm_conv_file = get_tm_conv_csv_file()
        
        # Process data
        processed_df = process_tm_conv_data(db, tm_conv_file, season_id, week_id)
        
        if processed_df.empty:
            print("No data to insert")
            return False
        
        # Insert data using batch upsert
        print(f"Inserting {len(processed_df)} records into tm_conv table...")
        success = batch_upsert_data(db, 'tm_conv', processed_df)
        
        if success:
            print("[SUCCESS] Team conversion data insertion completed successfully!")
            return True
        else:
            print("[FAILED] Team conversion data insertion failed!")
            return False
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        return False
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    success = main()
    if success:
        print("\nSUCCESS: tm_conv.py script completed successfully!")
    else:
        print("\nFAILED: tm_conv.py script failed!")