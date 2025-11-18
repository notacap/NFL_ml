"""
Weeks Played Script - NFL Player Playing Status Tracker

Inserts/upserts NFL player playing status information into the injury_report table 
for games that have already been played (historical data).
Ingests the same source CSV file as the plyr.py script.
"""

import sys
import os
import glob
from datetime import datetime, timedelta
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_utils import (DatabaseConnector, YEAR, WEEK, get_season_id, get_week_id, 
                     get_team_id, create_table_if_not_exists, standardize_team_name)

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
        was_inactive TINYINT,
        wed_practice_status VARCHAR(20),
        thurs_practice_status VARCHAR(20),
        fri_practice_status VARCHAR(20),
        sat_practice_status VARCHAR(20),
        game_status VARCHAR(20),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
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

def get_player_id_by_exact_match(db: DatabaseConnector, plyr_name: str, plyr_birthday_str: str, plyr_draft_tm: str) -> int:
    """
    Get player_id by exact matching on plyr_name, plyr_birthday (±1 day), and plyr_draft_tm
    
    Args:
        db: Database connector instance
        plyr_name: Player's full name
        plyr_birthday_str: Birthday string in format MM/DD/YYYY or YYYY-MM-DD
        plyr_draft_tm: Draft team abbreviation
    
    Returns:
        plyr_id if found, raises ValueError otherwise
    """
    try:
        csv_birthday = pd.to_datetime(plyr_birthday_str)
    except:
        raise ValueError(f"Invalid birthday format for {plyr_name}: {plyr_birthday_str}")
    
    date_min = csv_birthday - timedelta(days=1)
    date_max = csv_birthday + timedelta(days=1)
    
    query = """
    SELECT plyr_id FROM plyr 
    WHERE plyr_name = %s 
    AND plyr_draft_tm = %s
    AND plyr_birthday BETWEEN %s AND %s
    """
    
    result = db.fetch_all(query, (plyr_name, plyr_draft_tm, date_min.date(), date_max.date()))
    
    if result and len(result) > 0:
        return result[0][0]
    else:
        raise ValueError(f"Player not found: {plyr_name}, Birthday: {plyr_birthday_str}, Draft Team: {plyr_draft_tm}")

def determine_team_for_week(row, week, weeks_played):
    """Determine which team a player was on for a specific week"""
    current_team = row['current_team']
    former_team = row['former_team'] if pd.notna(row['former_team']) and row['former_team'] != '' else None
    first_team = row['first_team'] if pd.notna(row['first_team']) and row['first_team'] != '' else None
    
    if not former_team and not first_team:
        return current_team
    
    if former_team and not first_team:
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
    
    if former_team and first_team:
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
    
    return current_team

def process_csv_row(db: DatabaseConnector, row, season_id, max_week):
    """Process a single CSV row and generate injury report data for all weeks"""
    try:
        player_id = get_player_id_by_exact_match(
            db, 
            row['plyr_name'], 
            row['plyr_birthdate'], 
            row['plyr_draft_tm']
        )
    except ValueError as e:
        print(f"Warning: {e}")
        return []

    weeks_played = parse_weeks(row['weeks'])
    processed_data = []

    for week in range(1, max_week + 1):
        team_name = determine_team_for_week(row, week, weeks_played)
        
        try:
            team_abbrev = standardize_team_name(team_name)
            team_id = get_team_id(db, team_abbrev)
            week_id = get_week_id(db, season_id, week)
            
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
                    'was_inactive': 0 if week in weeks_played else 1
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
    INSERT INTO injury_report (season_id, plyr_id, plyr_name, team_id, game_id, week_id, was_inactive)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
    plyr_name = VALUES(plyr_name),
    team_id = VALUES(team_id),
    game_id = VALUES(game_id),
    was_inactive = VALUES(was_inactive)
    """
    
    batch_data = []
    for data in data_list:
        batch_data.append((
            data['season_id'],
            data['plyr_id'],
            data['plyr_name'],
            data['team_id'],
            data['game_id'],
            data['week_id'],
            data['was_inactive']
        ))
    
    success, rows_affected = db.execute_many(query, batch_data)
    
    if success:
        estimated_inserts = max(0, len(batch_data) * 2 - rows_affected)
        estimated_updates = len(batch_data) - estimated_inserts
        return estimated_inserts, estimated_updates
    else:
        return 0, 0

def main():
    """Main function to process injury report data"""
    print("Weeks Played Script - NFL Player Playing Status Tracker")
    print("=" * 60)
    print(f"Processing data for Year: {YEAR}, Week: {WEEK}")
    
    source_dir = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\plyr\\plyr_clean\\{WEEK}"
    print(f"Source directory: {source_dir}")
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory does not exist: {source_dir}")
        return
    
    csv_file_path = get_most_recent_csv_file(source_dir)
    if not csv_file_path:
        print("No CSV file found to process")
        return
    
    db = DatabaseConnector()
    
    try:
        if not db.connect():
            print("Failed to connect to database")
            return
        
        if not create_injury_report_table(db):
            print("Failed to create injury_report table")
            return
        
        try:
            season_id = get_season_id(db, YEAR)
            print(f"Using season_id: {season_id} for year {YEAR}")
        except ValueError as e:
            print(f"Error: {e}")
            return
        
        print(f"Loading data from: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        
        if df.empty:
            print("No data loaded from CSV file")
            return
        
        max_week = determine_max_week_from_data(df)
        if max_week == 0:
            print("No valid week data found in CSV")
            return
        
        print(f"Processing data for weeks 1-{max_week} of season {season_id}")
        
        total_players = 0
        total_rows_inserted = 0
        total_rows_updated = 0
        all_processed_data = []
        
        for _, row in df.iterrows():
            total_players += 1
            processed_data = process_csv_row(db, row, season_id, max_week)
            all_processed_data.extend(processed_data)
            
            if total_players % 50 == 0:
                print(f"Processed {total_players} players...")
        
        if all_processed_data:
            print(f"Upserting {len(all_processed_data)} injury report records...")
            batch_inserted, batch_updated = upsert_injury_data(db, all_processed_data)
            total_rows_inserted += batch_inserted
            total_rows_updated += batch_updated
        
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
