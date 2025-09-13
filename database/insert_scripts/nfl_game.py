import csv
import os
import glob
from datetime import datetime
import sys

# Add the parent directory to sys.path to import db_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_utils import DatabaseConnector, YEAR, WEEK

def create_nfl_game_table_if_not_exists(db_connector):
    """Create nfl_game table if it doesn't exist"""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS nfl_game (
        game_id INT PRIMARY KEY AUTO_INCREMENT,
        week_id INT,
        season_id INT,
        home_team_id INT,
        away_team_id INT,
        home_team_score INT,
        away_team_score INT,
        game_date DATE,
        game_time TIME,
        game_result ENUM('home_win', 'away_win', 'tie'),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        FOREIGN KEY (home_team_id) REFERENCES nfl_team(team_id),
        FOREIGN KEY (away_team_id) REFERENCES nfl_team(team_id),
        UNIQUE KEY (season_id, week_id, home_team_id, away_team_id)
    )
    """
    try:
        success = db_connector.execute_query(create_table_query)
        if success:
            log_message("INFO", "nfl_game table created or already exists")
            return True
        else:
            log_message("ERROR", "Failed to create nfl_game table")
            return False
    except Exception as e:
        log_message("ERROR", f"Error creating nfl_game table: {e}")
        return False

def log_message(message_type, message):
    """Print messages to console with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message_type}: {message}")

def get_latest_csv(directory_path):
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
    if not csv_files:
        return None
    return max(csv_files, key=os.path.getctime)

def safe_int(value):
    try:
        return int(float(value)) if value and str(value) != 'nan' else None
    except (ValueError, TypeError):
        return None

def safe_float(value):
    try:
        return float(value) if value and str(value) != 'nan' else None
    except (ValueError, TypeError):
        return None

def get_season_id(db_connector, year):
    """Get season_id from nfl_season table based on year"""
    query = "SELECT season_id FROM nfl_season WHERE year = %s"
    result = db_connector.fetch_all(query, (year,))
    return result[0][0] if result else None

def get_week_id(db_connector, season_id, week_num):
    """Get week_id from nfl_week table based on season_id and week number"""
    query = "SELECT week_id FROM nfl_week WHERE season_id = %s AND week_num = %s"
    result = db_connector.fetch_all(query, (season_id, str(week_num)))
    return result[0][0] if result else None

def get_team_id(db_connector, team_name):
    """Get team_id from nfl_team table based on team name"""
    # First try exact team name match
    query = "SELECT team_id FROM nfl_team WHERE team_name = %s"
    result = db_connector.fetch_all(query, (team_name,))
    if result:
        return result[0][0]
    
    # If no exact match, try abbreviation match
    query = "SELECT team_id FROM nfl_team WHERE abrv = %s OR alt_abrv = %s"
    result = db_connector.fetch_all(query, (team_name, team_name))
    return result[0][0] if result else None

def convert_time_to_24h(time_str):
    """Convert time from format like '8:20PM' to '20:20:00'"""
    try:
        return datetime.strptime(time_str, '%I:%M%p').strftime('%H:%M:%S')
    except ValueError:
        try:
            # Try alternative format without minutes
            return datetime.strptime(time_str, '%I%p').strftime('%H:%M:%S')
        except ValueError:
            log_message("ERROR", f"Unable to parse time: {time_str}")
            return None

def convert_date_format(date_str):
    """Convert date from YYYY-MM-DD format to database format"""
    try:
        # Parse the date string in YYYY-MM-DD format
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        # Return in YYYY-MM-DD format (already correct for database)
        return date_obj.strftime('%Y-%m-%d')
    except ValueError as e:
        log_message("ERROR", f"Error converting date: {date_str}. Error: {e}")
        return None

def process_csv_row(db_connector, row, season_id):
    """Process a single CSV row and return game data"""
    week_num = safe_int(row['Week'])
    if not week_num:
        log_message("ERROR", f"Invalid week number: {row['Week']}")
        return None

    week_id = get_week_id(db_connector, season_id, week_num)
    if not week_id:
        log_message("ERROR", f"Week not found: Season {season_id}, Week {week_num}")
        return None

    # Get team information from Home and Away columns
    home_team_name = row.get('Home', '').strip()
    away_team_name = row.get('Away', '').strip()
    
    if not home_team_name or not away_team_name:
        log_message("ERROR", f"Missing team names: Home='{home_team_name}', Away='{away_team_name}'")
        return None
    
    home_team_id = get_team_id(db_connector, home_team_name)
    away_team_id = get_team_id(db_connector, away_team_name)

    if not home_team_id or not away_team_id:
        log_message("ERROR", f"Team not found: Home={home_team_name}, Away={away_team_name}")
        return None

    # Get scores from Home Score and Away Score columns
    home_team_score = safe_int(row['Home Score'])
    away_team_score = safe_int(row['Away Score'])

    if home_team_score is None or away_team_score is None:
        log_message("ERROR", f"Invalid scores: Home={row['Home Score']}, Away={row['Away Score']}")
        return None

    # Determine game result
    if home_team_score > away_team_score:
        game_result = 'home_win'
    elif home_team_score < away_team_score:
        game_result = 'away_win'
    else:
        game_result = 'tie'

    game_date = convert_date_format(row['Date'])
    if not game_date:
        log_message("ERROR", f"Invalid date format: {row['Date']}")
        return None

    game_time = convert_time_to_24h(row['Time'])
    if not game_time:
        log_message("ERROR", f"Invalid time format: {row['Time']}")
        return None

    return {
        'week_id': week_id,
        'season_id': season_id,
        'home_team_id': home_team_id,
        'away_team_id': away_team_id,
        'home_team_score': home_team_score,
        'away_team_score': away_team_score,
        'game_date': game_date,
        'game_time': game_time,
        'game_result': game_result
    }

def upsert_data(db_connector, data):
    """Insert or update game data"""
    query = """
    INSERT INTO nfl_game (week_id, season_id, home_team_id, away_team_id, home_team_score, away_team_score, game_date, game_time, game_result)
    VALUES (%(week_id)s, %(season_id)s, %(home_team_id)s, %(away_team_id)s, %(home_team_score)s, %(away_team_score)s, %(game_date)s, %(game_time)s, %(game_result)s)
    ON DUPLICATE KEY UPDATE
    home_team_score = VALUES(home_team_score),
    away_team_score = VALUES(away_team_score),
    game_date = VALUES(game_date),
    game_time = VALUES(game_time),
    game_result = VALUES(game_result)
    """
    try:
        success = db_connector.execute_query(query, data)
        if success:
            return True, db_connector.cursor.rowcount
        else:
            return False, 0
    except Exception as e:
        log_message("ERROR", f"Error in upsert_data: {e}")
        return False, 0

def main():
    db_connector = DatabaseConnector()
    if not db_connector.connect():
        log_message("ERROR", "Failed to connect to database")
        return

    try:
        # Create nfl_game table if it doesn't exist
        if not create_nfl_game_table_if_not_exists(db_connector):
            log_message("ERROR", "Failed to ensure nfl_game table exists")
            return

        # Get season_id based on YEAR from db_utils
        season_id = get_season_id(db_connector, YEAR)
        if not season_id:
            log_message("ERROR", f"Season not found for year: {YEAR}")
            return

        # Construct directory path using YEAR and WEEK variables
        # First try the week_{WEEK} subdirectory for the new format
        directory_path = rf'C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\{YEAR}\schedule\week_{WEEK}'
        csv_file_path = get_latest_csv(directory_path)
        
        # If no CSV found in week_{WEEK}, try the parent schedule directory
        if not csv_file_path:
            directory_path = rf'C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\{YEAR}\schedule'
            csv_file_path = get_latest_csv(directory_path)

        if not csv_file_path:
            log_message("ERROR", f"Error: No CSV file found in the directory {directory_path}.")
            return

        file_name = os.path.basename(csv_file_path)
        log_message("INFO", f"Processing file: {file_name}")

        total_rows = 0
        processed_rows = 0
        upserted_rows = 0
        updated_rows = 0
        inserted_rows = 0
        skipped_rows = 0

        with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                total_rows += 1
                log_message("INFO", f"Processing row {total_rows}: {row.get('Home', 'N/A')} vs {row.get('Away', 'N/A')}")
                processed_data = process_csv_row(db_connector, row, season_id)
                if processed_data:
                    processed_rows += 1
                    success, affected_rows = upsert_data(db_connector, processed_data)
                    if success:
                        upserted_rows += 1
                        if affected_rows == 2:  # MySQL returns 2 for an update operation
                            updated_rows += 1
                        else:
                            inserted_rows += 1
                    else:
                        log_message("ERROR", f"Failed to upsert data for game: {row['Home']} vs {row['Away']}")
                else:
                    skipped_rows += 1
                    log_message("WARNING", f"Skipped row {total_rows}: {row['Home']} vs {row['Away']}")

        # Log processing summary
        summary = f"""
        Processing Summary:
        Processed file: {file_name}
        Season Year: {YEAR}
        Season ID: {season_id}
        Total rows in CSV: {total_rows}
        Processed rows: {processed_rows}
        Upserted rows: {upserted_rows}
        Updated rows: {updated_rows}
        Inserted rows: {inserted_rows}
        Skipped rows: {skipped_rows}
        """
        log_message("INFO", summary)
        log_message("INFO", "_" * 80)

    except FileNotFoundError:
        log_message("ERROR", f"Error: The file {csv_file_path} was not found.")
    except PermissionError:
        log_message("ERROR", f"Error: Permission denied when trying to open {csv_file_path}.")
    except Exception as e:
        log_message("ERROR", f"Error while processing CSV file: {e}")
    finally:
        db_connector.disconnect()

if __name__ == "__main__":
    main()