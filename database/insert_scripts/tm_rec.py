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

def create_tm_rec_table(db: DatabaseConnector) -> bool:
    """Create tm_rec table if it doesn't exist"""
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS tm_rec (
        tm_rec_id INT AUTO_INCREMENT PRIMARY KEY,
        team_id INT,
        season_id INT,
        week_id INT,
        tm_rec_tgt INT,
        tm_rec INT,
        tm_rec_yds INT,
        tm_rec_td INT,
        tm_rec_first_dwn INT,
        tm_rec_ybc INT,
        tm_rec_ybc_rec DECIMAL(7,4),
        tm_rec_yac INT,
        tm_rec_yac_rec DECIMAL(7,4),
        tm_rec_adot DECIMAL(7,4),
        tm_rec_brkn_tkl INT,
        tm_rec_brkn_tkl_rec DECIMAL(7,4),
        tm_rec_drp INT,
        tm_rec_drp_pct DECIMAL(5,4),
        UNIQUE KEY uk_tm_season (team_id, season_id, week_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (team_id) REFERENCES nfl_team(team_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id)
    );
    """
    
    return create_table_if_not_exists(db, "tm_rec", create_table_sql)

def get_tm_rec_csv_files() -> str:
    """Get the most recent tm_adv_receiving CSV file"""
    
    # Define base path using YEAR and WEEK from db_utils
    tm_rec_path = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\tm_adv_receiving\\week_{WEEK}"
    
    # Find tm_adv_receiving files
    tm_rec_pattern = os.path.join(tm_rec_path, "tm_adv_receiving_*.csv")
    tm_rec_files = sorted(glob.glob(tm_rec_pattern), key=os.path.getmtime, reverse=True)
    
    if not tm_rec_files:
        raise FileNotFoundError(f"No tm_adv_receiving CSV files found in {tm_rec_path}")
    
    print(f"Using tm_adv_receiving file: {tm_rec_files[0]}")
    
    return tm_rec_files[0]

def process_tm_rec_data(db: DatabaseConnector, tm_rec_file: str, season_id: int, week_id: int) -> pd.DataFrame:
    """Process and consolidate tm_adv_receiving data"""
    
    # Load tm_adv_receiving data
    print(f"Loading team advanced receiving data from {tm_rec_file}...")
    tm_rec_df = load_csv_data(tm_rec_file)
    if tm_rec_df.empty:
        raise ValueError("Failed to load tm_adv_receiving CSV data")
    
    # Filter out empty rows and summary rows
    tm_rec_df = tm_rec_df.dropna(subset=['Tm'])
    tm_rec_df = tm_rec_df[tm_rec_df['Tm'].str.strip() != '']
    
    # Filter out any potential summary rows (though analysis shows there aren't any)
    tm_rec_df = tm_rec_df[~tm_rec_df['Tm'].str.contains('Avg|League|Total', na=False)]
    
    print(f"Loaded {len(tm_rec_df)} teams from advanced receiving file")
    
    processed_records = []
    
    for _, row in tm_rec_df.iterrows():
        try:
            team_name = row['Tm']
            print(f"Processing team: {team_name}")
            
            # Get team_id using the utility function
            team_id = get_team_id(db, team_name)
            
            # Helper functions for safe conversion
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
                    # Handle percentage strings like "4.1%"
                    if isinstance(value, str) and value.endswith('%'):
                        return float(value.rstrip('%')) / 100.0
                    return float(value) / 100.0 if float(value) > 1 else float(value)
                except (ValueError, TypeError):
                    return None
            
            record = {
                'team_id': team_id,
                'season_id': season_id,
                'week_id': week_id,
                
                # Team advanced receiving stats
                'tm_rec_tgt': safe_convert_int(row['Tgt']),  # Targets
                'tm_rec': safe_convert_int(row['Rec']),  # Receptions
                'tm_rec_yds': safe_convert_int(row['Yds']),  # Receiving Yards
                'tm_rec_td': safe_convert_int(row['TD']),  # Touchdowns
                'tm_rec_first_dwn': safe_convert_int(row['1D']),  # First Downs
                'tm_rec_ybc': safe_convert_int(row['YBC']),  # Yards Before Catch
                'tm_rec_ybc_rec': safe_convert_float(row['YBC/R']),  # YBC per Reception
                'tm_rec_yac': safe_convert_int(row['YAC']),  # Yards After Catch
                'tm_rec_yac_rec': safe_convert_float(row['YAC/R']),  # YAC per Reception
                'tm_rec_adot': safe_convert_float(row['ADOT']),  # Average Depth of Target
                'tm_rec_brkn_tkl': safe_convert_int(row['BrkTkl']),  # Broken Tackles
                'tm_rec_brkn_tkl_rec': safe_convert_float(row['Rec/Br']),  # Receptions per Broken Tackle
                'tm_rec_drp': safe_convert_int(row['Drop']),  # Drops
                'tm_rec_drp_pct': safe_convert_percentage(row['Drop%'])  # Drop Percentage
            }
            
            processed_records.append(record)
            
        except Exception as e:
            print(f"Error processing team {team_name}: {e}")
            continue
    
    # Convert to DataFrame
    final_df = pd.DataFrame(processed_records)
    
    # Handle null values
    final_df = handle_null_values(final_df)
    
    print(f"Successfully processed {len(final_df)} team receiving records")
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
        if not create_tm_rec_table(db):
            print("Failed to create tm_rec table")
            return False
        
        # Get foreign key values
        season_id = get_season_id(db, YEAR)
        week_id = get_week_id(db, season_id, str(WEEK))
        
        print(f"Processing data for season {YEAR} (ID: {season_id}), week {WEEK} (ID: {week_id})")
        
        # Get CSV file
        tm_rec_file = get_tm_rec_csv_files()
        
        # Process data
        processed_df = process_tm_rec_data(db, tm_rec_file, season_id, week_id)
        
        if processed_df.empty:
            print("No data to insert")
            return False
        
        # Insert data using batch upsert
        print(f"Inserting {len(processed_df)} records into tm_rec table...")
        success = batch_upsert_data(db, 'tm_rec', processed_df)
        
        if success:
            print("[SUCCESS] Team receiving data insertion completed successfully!")
            return True
        else:
            print("[FAILED] Team receiving data insertion failed!")
            return False
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        return False
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    success = main()
    if success:
        print("\nSUCCESS: tm_rec.py script completed successfully!")
    else:
        print("\nFAILED: tm_rec.py script failed!")