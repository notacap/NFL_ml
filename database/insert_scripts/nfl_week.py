"""
NFL Week Insert Script

Inserts week data into the nfl_week table.
No CSV source file - data is generated for each season (1-18 weeks).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_utils import DatabaseConnector

def create_nfl_week_table(db: DatabaseConnector) -> bool:
    """Create the nfl_week table if it doesn't exist"""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS nfl_week (
        week_id INT PRIMARY KEY AUTO_INCREMENT,
        season_id INT,
        week_num INT,
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        UNIQUE KEY uk_season_week (season_id, week_num)
    )
    """
    
    if db.execute_query(create_table_query):
        print("nfl_week table created/verified successfully")
        return True
    else:
        print("Failed to create nfl_week table")
        return False

def get_season_ids(db: DatabaseConnector) -> dict:
    """Get season_id mappings from nfl_season table"""
    query = "SELECT season_id, year FROM nfl_season ORDER BY year"
    results = db.fetch_all(query)
    
    season_mapping = {}
    for season_id, year in results:
        season_mapping[year] = season_id
    
    return season_mapping

def week_exists(db: DatabaseConnector, season_id: int, week_num: int) -> bool:
    """Check if week already exists for the season"""
    result = db.fetch_all(
        "SELECT COUNT(*) FROM nfl_week WHERE season_id = %s AND week_num = %s", 
        (season_id, week_num)
    )
    return result[0][0] > 0 if result else False

def insert_week_data(db: DatabaseConnector, season_id: int, week_num: int) -> bool:
    """Insert week data into nfl_week table"""
    
    if week_exists(db, season_id, week_num):
        return True  # Already exists, skip silently
    
    insert_query = """
    INSERT INTO nfl_week (season_id, week_num) 
    VALUES (%s, %s)
    """
    
    if db.execute_query(insert_query, (season_id, week_num)):
        return True
    else:
        print(f"Failed to insert week {week_num} for season_id {season_id}")
        return False

def insert_weeks_for_season(db: DatabaseConnector, season_id: int, year: int) -> bool:
    """Insert all 18 weeks for a given season"""
    print(f"Inserting weeks for {year} season (season_id: {season_id})")
    
    success_count = 0
    for week_num in range(1, 19):  # Weeks 1-18
        if insert_week_data(db, season_id, week_num):
            success_count += 1
    
    print(f"Successfully inserted {success_count}/18 weeks for {year}")
    return success_count == 18

def main():
    """Main function to insert NFL week data"""
    print("NFL Week Insert Script")
    print("=" * 40)
    
    db = DatabaseConnector()
    
    try:
        if not db.connect():
            print("Failed to connect to database")
            return
        
        # Create table if it doesn't exist
        if not create_nfl_week_table(db):
            return
        
        # Get season IDs from the database
        season_mapping = get_season_ids(db)
        if not season_mapping:
            print("No seasons found in nfl_season table. Please run nfl_season.py first.")
            return
        
        print(f"Found seasons: {season_mapping}")
        
        # Insert weeks for all seasons
        total_success = 0
        for year, season_id in season_mapping.items():
            if insert_weeks_for_season(db, season_id, year):
                total_success += 1
            print("-" * 30)
        
        print(f"Week data insertion completed for {total_success}/{len(season_mapping)} seasons")
    
    except Exception as e:
        print(f"Error in main execution: {e}")
    
    finally:
        db.disconnect()

if __name__ == "__main__":
    main()