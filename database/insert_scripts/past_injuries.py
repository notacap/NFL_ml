"""
Past Injuries Script - NFL Player Playing Status Tracker

Inserts/upserts NFL player playing status information into the injury_report table 
for games that have already been played (historical data).
Ingests the same source CSV file as the plyr.py script.
"""

import sys
import os
import glob
from datetime import datetime
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_utils import (DatabaseConnector, YEAR, WEEK, get_season_id, get_week_id, 
                     get_team_id, get_game_id, get_player_id, create_table_if_not_exists,
                     standardize_team_name)

# Configuration

def create_injury_report_table(db: DatabaseConnector) -> bool:
    """Create the injury_report table if it doesn't exist"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS injury_report (
        injury_report_id INT AUTO_INCREMENT PRIMARY KEY,
        season_id INT,
        plyr_id INT,
        plyr_name VARCHAR(255) NOT NULL,
        team_id INT,
        game_id INT,
        week_id INT,
        was_active TINYINT,
        is_active TINYINT,
        practice_status VARCHAR(20),
        game_status VARCHAR(20),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        FOREIGN KEY (plyr_id) REFERENCES plyr(plyr_id),
        FOREIGN KEY (game_id) REFERENCES nfl_game(game_id),
        FOREIGN KEY (team_id) REFERENCES nfl_team(team_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        UNIQUE KEY uk_plyr_season_game (plyr_id, game_id)
    )
    """
    
    return create_table_if_not_exists(db, 'injury_report', create_table_sql)

def get_most_recent_csv_file(directory_path: str) -> str:
    """Find the most recently created CSV file with 'cleaned_players' in the name"""
    try:
        pattern = os.path.join(directory_path, "*cleaned_players*.csv")
        csv_files = glob.glob(pattern)
        
        if not csv_files:
            print(f"No CSV files with 'cleaned_players' found in {directory_path}")
            return None
        
        # Get the most recent file based on creation time
        most_recent_file = max(csv_files, key=os.path.getctime)
        print(f"Found most recent CSV file: {most_recent_file}")
        return most_recent_file
        
    except Exception as e:
        print(f"Error finding CSV files in {directory_path}: {e}")
        return None

def parse_weeks(weeks_string):
    """Parse comma-separated weeks string into a set of integers"""
    if pd.isna(weeks_string) or weeks_string == '':
        return set()
    # Handle quoted strings
    weeks_clean = str(weeks_string).strip('"').strip("'")
    return set(map(int, weeks_clean.split(',')))

def determine_max_week_from_data(df: pd.DataFrame) -> int:
    """Determine the maximum week value from all rows in the weeks column"""
    max_week = 0
    for _, row in df.iterrows():
        weeks_played = parse_weeks(row['weeks'])
        if weeks_played:
            row_max = max(weeks_played)
            max_week = max(max_week, row_max)
    
    print(f"Maximum week found in data: {max_week}")
    return max_week

def determine_team_for_week(row, week, weeks_played):
    """Determine which team a player was on for a specific week"""
    current_team = row['current_team']
    former_team = row['former_team'] if pd.notna(row['former_team']) and row['former_team'] != '' else None
    first_team = row['first_team'] if pd.notna(row['first_team']) and row['first_team'] != '' else None
    
    if not former_team and not first_team:
        return current_team  # Single-team player
    
    if former_team and not first_team:  # Two-team player
        try:
            former_team_first_week = int(row['former_team_first_week']) if pd.notna(row['former_team_first_week']) else 1
            former_team_last_week = int(row['former_team_last_week']) if pd.notna(row['former_team_last_week']) else 18
            
            if former_team_first_week == min(weeks_played):
                if week <= former_team_last_week:
                    return former_team
                else:
                    return current_team
            else:
                if week < former_team_first_week:
                    return current_team
                elif former_team_first_week <= week <= former_team_last_week:
                    return former_team
                else:
                    return current_team
        except (ValueError, TypeError):
            return current_team
    
    if former_team and first_team:  # Three-team player
        try:
            first_team_last_week = int(row['first_team_last_week']) if pd.notna(row['first_team_last_week']) else 6
            former_team_last_week = int(row['former_team_last_week']) if pd.notna(row['former_team_last_week']) else 12
            
            if week <= first_team_last_week:
                return first_team
            elif first_team_last_week < week <= former_team_last_week:
                return former_team
            else:
                return current_team
        except (ValueError, TypeError):
            return current_team
    
    return current_team  # Fallback to current team

def process_csv_row(db: DatabaseConnector, row, season_id, max_week):
    """Process a single CSV row and generate injury report data for all weeks"""
    try:
        # Convert full team name to abbreviation for player lookup
        team_abbrev = standardize_team_name(row['current_team'])
        player_id = get_player_id(db, row['plyr_name'], team_abbrev, season_id)
    except ValueError as e:
        print(f"Warning: {e}")
        return []

    weeks_played = parse_weeks(row['weeks'])
    processed_data = []

    for week in range(1, max_week + 1):  # Dynamic range based on data
        team_name = determine_team_for_week(row, week, weeks_played)
        
        try:
            # Convert full team name to abbreviation for database lookup
            team_abbrev = standardize_team_name(team_name)
            team_id = get_team_id(db, team_abbrev)
            week_id = get_week_id(db, season_id, week)
            
            # Get game_id for this team and week
            # For injury reports, we need to find the game where this team played
            games = db.fetch_all(
                "SELECT game_id FROM nfl_game WHERE (home_team_id = %s OR away_team_id = %s) AND week_id = %s AND season_id = %s",
                (team_id, team_id, week_id, season_id)
            )
            
            if games:
                game_id = games[0][0]
                
                processed_data.append({
                    'season_id': season_id,
                    'plyr_id': player_id,
                    'plyr_name': row['plyr_name'],
                    'team_id': team_id,
                    'game_id': game_id,
                    'week_id': week_id,
                    'was_active': 1 if week in weeks_played else 0  # TINYINT: 1 = Active, 0 = Not Active
                })
            else:
                print(f"Warning: No game found for {team_name} in week {week}")
                
        except ValueError as e:
            print(f"Warning: {e}")
            continue

    return processed_data

def upsert_injury_data(db: DatabaseConnector, data_list):
    """Upsert injury report data using batch processing"""
    if not data_list:
        return 0, 0
    
    query = """
    INSERT INTO injury_report (season_id, plyr_id, plyr_name, team_id, game_id, week_id, was_active)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
    plyr_name = VALUES(plyr_name),
    team_id = VALUES(team_id),
    game_id = VALUES(game_id),
    was_active = VALUES(was_active)
    """
    
    # Convert data to tuples for batch insert
    batch_data = []
    for data in data_list:
        batch_data.append((
            data['season_id'],
            data['plyr_id'],
            data['plyr_name'],
            data['team_id'],
            data['game_id'],
            data['week_id'],
            data['was_active']
        ))
    
    success, rows_affected = db.execute_many(query, batch_data)
    
    if success:
        # Estimate inserts vs updates based on MySQL behavior
        # rows_affected = 1 per insert, 2 per update
        estimated_inserts = max(0, len(batch_data) * 2 - rows_affected)
        estimated_updates = len(batch_data) - estimated_inserts
        return estimated_inserts, estimated_updates
    else:
        return 0, 0

def main():
    """Main function to process injury report data"""
    print("Past Injuries Script - NFL Player Playing Status Tracker")
    print("=" * 60)
    print(f"Processing data for Year: {YEAR}, Week: {WEEK}")
    
    # Construct the source directory path using YEAR and WEEK from db_utils
    source_dir = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\plyr\\plyr_clean\\{WEEK}"
    print(f"Source directory: {source_dir}")
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory does not exist: {source_dir}")
        return
    
    # Find the most recent CSV file with 'cleaned_players' in name
    csv_file_path = get_most_recent_csv_file(source_dir)
    if not csv_file_path:
        print("No CSV file found to process")
        return
    
    db = DatabaseConnector()
    
    try:
        if not db.connect():
            print("Failed to connect to database")
            return
        
        # Create injury_report table if it doesn't exist
        if not create_injury_report_table(db):
            print("Failed to create injury_report table")
            return
        
        # Get season_id using db_utils function
        try:
            season_id = get_season_id(db, YEAR)
            print(f"Using season_id: {season_id} for year {YEAR}")
        except ValueError as e:
            print(f"Error: {e}")
            return
        
        # Load and process CSV data
        print(f"Loading data from: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        
        if df.empty:
            print("No data loaded from CSV file")
            return
        
        # Determine maximum week from the data
        max_week = determine_max_week_from_data(df)
        if max_week == 0:
            print("No valid week data found in CSV")
            return
        
        print(f"Processing data for weeks 1-{max_week} of season {season_id}")
        
        total_players = 0
        total_rows_inserted = 0
        total_rows_updated = 0
        all_processed_data = []
        
        # Process each player row
        for _, row in df.iterrows():
            total_players += 1
            processed_data = process_csv_row(db, row, season_id, max_week)
            all_processed_data.extend(processed_data)
            
            if total_players % 50 == 0:
                print(f"Processed {total_players} players...")
        
        # Batch upsert all data
        if all_processed_data:
            print(f"Upserting {len(all_processed_data)} injury report records...")
            batch_inserted, batch_updated = upsert_injury_data(db, all_processed_data)
            total_rows_inserted += batch_inserted
            total_rows_updated += batch_updated
        
        # Summary
        print("\nProcessing Summary:")
        print(f"Total players processed: {total_players}")
        print(f"Total rows inserted: {total_rows_inserted}")
        print(f"Total rows updated: {total_rows_updated}")
        print(f"File processed: {os.path.basename(csv_file_path)}")
        print("Script execution completed successfully")
    
    except Exception as e:
        print(f"Error in main execution: {e}")
    
    finally:
        db.disconnect()

if __name__ == "__main__":
    main()
