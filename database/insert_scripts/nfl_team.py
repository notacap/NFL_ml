"""
NFL Team Insert Script

Inserts team data into the nfl_team table from CSV files.
Source files: nfl_team.csv and team_location.csv
"""

import sys
import os
import csv
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_utils import DatabaseConnector

def create_nfl_team_table(db: DatabaseConnector) -> bool:
    """Create the nfl_team table if it doesn't exist"""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS nfl_team (
        team_id INT AUTO_INCREMENT PRIMARY KEY,
        team_name VARCHAR(255) NOT NULL,
        abrv VARCHAR(3) NOT NULL,
        alt_abrv VARCHAR(3),
        conference ENUM('AFC', 'NFC') NOT NULL,
        division ENUM('North', 'South', 'East', 'West') NOT NULL,
        indoor_stadium TINYINT(1) NOT NULL,
        warm_climate TINYINT(1) NOT NULL,
        stadium_address VARCHAR(255),
        stadium_latitude DECIMAL(8, 6),
        stadium_latitude_direction ENUM('N', 'S'),
        stadium_longitude DECIMAL(9, 6),
        stadium_longitude_direction ENUM('E', 'W')
    )
    """
    
    if db.execute_query(create_table_query):
        print("nfl_team table created/verified successfully")
        return True
    else:
        print("Failed to create nfl_team table")
        return False

def team_exists(db: DatabaseConnector, team_name: str) -> bool:
    """Check if team already exists in database"""
    result = db.fetch_all("SELECT COUNT(*) FROM nfl_team WHERE team_name = %s", (team_name,))
    return result[0][0] > 0 if result else False

def parse_coordinate(coord_str: str) -> tuple:
    """Parse coordinate string like '-84.4008° W' into (value, direction)"""
    # Remove degree symbol and any other unwanted characters
    coord_str = coord_str.strip().replace('°', '').replace('�', '')
    match = re.match(r'(-?\d+\.?\d*)\s*([NSEW])', coord_str)
    
    if match:
        value = float(match.group(1))
        direction = match.group(2)
        # Keep the original sign - negative for West/South, positive for North/East
        return value, direction
    else:
        print(f"Warning: Could not parse coordinate: {coord_str}")
        return None, None

def load_team_data(team_csv_path: str) -> dict:
    """Load team data from nfl_team.csv"""
    teams = {}
    
    try:
        with open(team_csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                team_name = row['name'].strip()
                teams[team_name] = {
                    'team_name': team_name,
                    'abrv': row['abbreviation'].strip(),
                    'conference': row['conference'].strip(),
                    'division': row['division'].strip(),
                    'indoor_stadium': 1 if row['indoor_stadium'].strip().upper() == 'TRUE' else 0,
                    'warm_climate': 1 if row['warm_climate'].strip().upper() == 'TRUE' else 0
                }
        print(f"Loaded {len(teams)} teams from {team_csv_path}")
        
    except FileNotFoundError:
        print(f"Error: Could not find file {team_csv_path}")
    except Exception as e:
        print(f"Error reading team CSV: {e}")
    
    return teams

def load_location_data(location_csv_path: str) -> dict:
    """Load location data from team_location.csv"""
    locations = {}
    
    try:
        with open(location_csv_path, 'r', encoding='utf-8', errors='ignore') as file:
            reader = csv.DictReader(file)
            for row in reader:
                team_name = row['Team Name'].strip()
                
                # Parse latitude and longitude
                lat_value, lat_direction = parse_coordinate(row['Stadium Latitude'])
                lng_value, lng_direction = parse_coordinate(row['Stadium Longitude'])
                
                locations[team_name] = {
                    'stadium_address': row['Stadium Address'].strip(),
                    'stadium_latitude': lat_value,
                    'stadium_latitude_direction': lat_direction,
                    'stadium_longitude': lng_value,
                    'stadium_longitude_direction': lng_direction
                }
        
        print(f"Loaded {len(locations)} team locations from {location_csv_path}")
        
    except FileNotFoundError:
        print(f"Error: Could not find file {location_csv_path}")
    except Exception as e:
        print(f"Error reading location CSV: {e}")
    
    return locations

def insert_team_data(db: DatabaseConnector, team_data: dict, location_data: dict) -> bool:
    """Insert team data into nfl_team table"""
    
    insert_query = """
    INSERT INTO nfl_team (
        team_name, abrv, alt_abrv, conference, division, 
        indoor_stadium, warm_climate, stadium_address, 
        stadium_latitude, stadium_latitude_direction, 
        stadium_longitude, stadium_longitude_direction
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    success_count = 0
    total_count = len(team_data)
    
    for team_name, team_info in team_data.items():
        if team_exists(db, team_name):
            print(f"Team {team_name} already exists in database")
            success_count += 1
            continue
        
        # Get corresponding location data
        location_info = location_data.get(team_name, {})
        
        if not location_info:
            print(f"Warning: No location data found for {team_name}")
        
        # Prepare data for insertion
        values = (
            team_info['team_name'],
            team_info['abrv'],
            None,  # alt_abrv - not provided in CSV
            team_info['conference'],
            team_info['division'],
            team_info['indoor_stadium'],
            team_info['warm_climate'],
            location_info.get('stadium_address'),
            location_info.get('stadium_latitude'),
            location_info.get('stadium_latitude_direction'),
            location_info.get('stadium_longitude'),
            location_info.get('stadium_longitude_direction')
        )
        
        if db.execute_query(insert_query, values):
            print(f"Successfully inserted: {team_name}")
            success_count += 1
        else:
            print(f"Failed to insert: {team_name}")
    
    print(f"Insertion completed: {success_count}/{total_count} teams processed")
    return success_count == total_count

def main():
    """Main function to insert NFL team data"""
    print("NFL Team Insert Script")
    print("=" * 40)
    
    # Define CSV file paths
    csv_dir = r"C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\nfl_teams"
    team_csv_path = os.path.join(csv_dir, "nfl_team.csv")
    location_csv_path = os.path.join(csv_dir, "team_location.csv")
    
    db = DatabaseConnector()
    
    try:
        if not db.connect():
            print("Failed to connect to database")
            return
        
        # Create table if it doesn't exist
        if not create_nfl_team_table(db):
            return
        
        # Load data from CSV files
        team_data = load_team_data(team_csv_path)
        location_data = load_location_data(location_csv_path)
        
        if not team_data:
            print("No team data loaded. Exiting.")
            return
        
        # Insert team data
        print("\nInserting team data...")
        if insert_team_data(db, team_data, location_data):
            print("All team data inserted successfully")
        else:
            print("Some teams failed to insert")
    
    except Exception as e:
        print(f"Error in main execution: {e}")
    
    finally:
        db.disconnect()

if __name__ == "__main__":
    main()