"""
NFL Season Insert Script

Inserts season data into the nfl_season table.
No CSV source file - data is hardcoded for the configured year.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_utils import DatabaseConnector, YEAR
from datetime import datetime

def create_nfl_season_table(db: DatabaseConnector) -> bool:
    """Create the nfl_season table if it doesn't exist"""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS nfl_season (
        season_id INT AUTO_INCREMENT PRIMARY KEY,
        year INT NOT NULL UNIQUE,
        start_date DATE,
        end_date DATE
    )
    """
    
    if db.execute_query(create_table_query):
        print("nfl_season table created/verified successfully")
        return True
    else:
        print("Failed to create nfl_season table")
        return False

def season_exists(db: DatabaseConnector, year: int) -> bool:
    """Check if season already exists in database"""
    result = db.fetch_all("SELECT COUNT(*) FROM nfl_season WHERE year = %s", (year,))
    return result[0][0] > 0 if result else False

def insert_season_data(db: DatabaseConnector, year: int, start_date: str, end_date: str) -> bool:
    """Insert season data into nfl_season table"""
    
    if season_exists(db, year):
        print(f"Season {year} already exists in database")
        return True
    
    insert_query = """
    INSERT INTO nfl_season (year, start_date, end_date) 
    VALUES (%s, %s, %s)
    """
    
    # Convert date strings to proper format
    start_date_obj = datetime.strptime(start_date, "%m/%d/%Y").date()
    end_date_obj = datetime.strptime(end_date, "%m/%d/%Y").date()
    
    if db.execute_query(insert_query, (year, start_date_obj, end_date_obj)):
        print(f"Successfully inserted season {year} data")
        return True
    else:
        print(f"Failed to insert season {year} data")
        return False

def get_season_data(year: int) -> tuple:
    """Get season data for the given year"""
    season_data = {
        2022: ("9/8/2022", "1/8/2023"),
        2023: ("9/7/2023", "1/7/2024"),
        2024: ("9/5/2024", "1/5/2025")
    }
    
    if year in season_data:
        return season_data[year]
    else:
        print(f"No season data defined for year {year}")
        return None

def main():
    """Main function to insert NFL season data"""
    print("NFL Season Insert Script")
    print("=" * 40)
    
    db = DatabaseConnector()
    
    try:
        if not db.connect():
            print("Failed to connect to database")
            return
        
        # Create table if it doesn't exist
        if not create_nfl_season_table(db):
            return
        
        # Insert all available seasons
        all_seasons = [2022, 2023, 2024]
        
        for year in all_seasons:
            season_data = get_season_data(year)
            if not season_data:
                continue
            
            start_date, end_date = season_data
            
            print(f"Inserting season data for {year}")
            print(f"Start Date: {start_date}")
            print(f"End Date: {end_date}")
            
            if insert_season_data(db, year, start_date, end_date):
                print(f"Season {year} insertion completed successfully")
            else:
                print(f"Season {year} insertion failed")
            print("-" * 30)
        
        print("All season data processing completed")
    
    except Exception as e:
        print(f"Error in main execution: {e}")
    
    finally:
        db.disconnect()

if __name__ == "__main__":
    main()