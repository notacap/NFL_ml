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

def create_nfl_standings_table(db: DatabaseConnector) -> bool:
    """Create nfl_standings table if it doesn't exist"""
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS nfl_standings (
        nfl_standings_id INT AUTO_INCREMENT PRIMARY KEY,
        team_id INT,
        season_id INT,
        week_id INT,
        win INT,
        loss INT,
        win_loss_pct DECIMAL(7,4),
        pt_for INT,
        pt_against INT,
        pt_diff INT,
        margin_of_victory DECIMAL(7,4),
        strength_of_schedule DECIMAL(7,4),
        simple_rating_system DECIMAL(7,4),
        off_srs DECIMAL(7,4),
        def_srs DECIMAL(7,4),
        UNIQUE KEY uk_tm_season (team_id, season_id, week_id),
        FOREIGN KEY (team_id) REFERENCES nfl_team(team_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id)
    );
    """
    
    return create_table_if_not_exists(db, "nfl_standings", create_table_sql)

def get_nfl_standings_csv_file() -> str:
    """Get the most recent nfl_standings CSV file"""
    
    # Define base path using YEAR and WEEK from db_utils
    standings_path = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\nfl_standings\\week_{WEEK}"
    
    # Find nfl_standings files
    standings_pattern = os.path.join(standings_path, "nfl_standings_*.csv")
    standings_files = sorted(glob.glob(standings_pattern), key=os.path.getmtime, reverse=True)
    
    if not standings_files:
        raise FileNotFoundError(f"No nfl_standings CSV files found in {standings_path}")
    
    print(f"Using nfl_standings file: {standings_files[0]}")
    
    return standings_files[0]

def process_nfl_standings_data(db: DatabaseConnector, standings_file: str, season_id: int, week_id: int) -> pd.DataFrame:
    """Process nfl_standings data"""
    
    # Load standings data
    print(f"Loading NFL standings data from {standings_file}...")
    standings_df = load_csv_data(standings_file)
    if standings_df.empty:
        raise ValueError("Failed to load nfl_standings CSV data")
    
    print(f"Loaded {len(standings_df)} teams from standings file")
    
    processed_records = []
    
    for _, row in standings_df.iterrows():
        try:
            team_name = row['Tm']
            print(f"Processing team: {team_name}")
            
            # Strip playoff symbols (*, +, etc.) from the end of team names
            clean_team_name = team_name.rstrip('*+')
            
            # Get team_id using the utility function (handles full team names)
            team_id = get_team_id(db, clean_team_name)
            
            # Helper functions for type conversion
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
                
                # Map CSV columns to database columns
                'win': safe_convert_int(row['W']),
                'loss': safe_convert_int(row['L']),
                'win_loss_pct': safe_convert_float(row['W-L%']),
                'pt_for': safe_convert_int(row['PF']),
                'pt_against': safe_convert_int(row['PA']),
                'pt_diff': safe_convert_int(row['PD']),
                'margin_of_victory': safe_convert_float(row['MoV']),
                'strength_of_schedule': safe_convert_float(row['SoS']),
                'simple_rating_system': safe_convert_float(row['SRS']),
                'off_srs': safe_convert_float(row['OSRS']),
                'def_srs': safe_convert_float(row['DSRS'])
            }
            
            processed_records.append(record)
            
        except Exception as e:
            print(f"Error processing team {team_name}: {e}")
            continue
    
    # Convert to DataFrame
    final_df = pd.DataFrame(processed_records)
    
    # Handle null values
    final_df = handle_null_values(final_df)
    
    print(f"Successfully processed {len(final_df)} team standings records")
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
        if not create_nfl_standings_table(db):
            print("Failed to create nfl_standings table")
            return False
        
        # Get foreign key values
        season_id = get_season_id(db, YEAR)
        week_id = get_week_id(db, season_id, str(WEEK))
        
        print(f"Processing data for season {YEAR} (ID: {season_id}), week {WEEK} (ID: {week_id})")
        
        # Get CSV file
        standings_file = get_nfl_standings_csv_file()
        
        # Process data
        processed_df = process_nfl_standings_data(db, standings_file, season_id, week_id)
        
        if processed_df.empty:
            print("No data to insert")
            return False
        
        # Insert data using batch upsert
        print(f"Inserting {len(processed_df)} records into nfl_standings table...")
        success = batch_upsert_data(db, 'nfl_standings', processed_df)
        
        if success:
            print("[SUCCESS] NFL standings data insertion completed successfully!")
            return True
        else:
            print("[FAILED] NFL standings data insertion failed!")
            return False
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        return False
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    success = main()
    if success:
        print("\nSUCCESS: nfl_standings.py script completed successfully!")
    else:
        print("\nFAILED: nfl_standings.py script failed!")