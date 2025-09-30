"""
Injury Report Script - NFL Player Injury Status Tracker

Inserts/upserts NFL player injury status information into the injury_report table
for upcoming games. Processes two types of CSV files:
1. team_injuries_*.csv - Practice and game status for injured players
2. {TEAM_ABBREV}_injury_report_*.csv - Historical playing status by week
"""

import sys
import os
import glob
from datetime import datetime
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_utils import (DatabaseConnector, YEAR, WEEK, get_season_id, get_week_id,
                     get_team_id, get_player_id, create_table_if_not_exists,
                     standardize_team_name)

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

def get_most_recent_file_pair(team_dir: str) -> tuple:
    """Find the most recent pair of injury report files for a team

    Returns:
        tuple: (team_injuries_file, abbrev_injury_report_file) or (None, None) if not found
    """
    try:
        # Find team_injuries files (exclude team_injuries_totals)
        team_injuries_pattern = os.path.join(team_dir, "team_injuries_*.csv")
        team_injuries_files = [f for f in glob.glob(team_injuries_pattern) if 'totals' not in f]

        # Find team abbreviation injury report files
        abbrev_pattern = os.path.join(team_dir, "*_injury_report_*.csv")
        abbrev_files = glob.glob(abbrev_pattern)

        if not team_injuries_files or not abbrev_files:
            return None, None

        # Get most recent of each type based on timestamp in filename
        most_recent_team_injuries = max(team_injuries_files, key=os.path.getctime)
        most_recent_abbrev = max(abbrev_files, key=os.path.getctime)

        return most_recent_team_injuries, most_recent_abbrev

    except Exception as e:
        print(f"Error finding file pair in {team_dir}: {e}")
        return None, None

def process_team_injuries_file(db: DatabaseConnector, file_path: str, season_id: int,
                               target_week_id: int, team_id: int, game_id: int) -> list:
    """Process team_injuries CSV file and return list of injury records

    Args:
        file_path: Path to team_injuries CSV file
        season_id: Season ID
        target_week_id: Week ID for the upcoming game (WEEK + 1)
        team_id: Team ID
        game_id: Game ID for the upcoming game

    Returns:
        list: List of dictionaries containing injury report data
    """
    try:
        df = pd.read_csv(file_path)

        if df.empty:
            return []

        injury_records = []

        for _, row in df.iterrows():
            player_name = row['Player']
            practice_status = row['Practice Status'] if pd.notna(row['Practice Status']) else None
            game_status = row['Status'] if pd.notna(row['Status']) else None
            team_name = row['Team']

            # Convert team name to abbreviation
            team_abbrev = standardize_team_name(team_name)

            try:
                # Get player ID
                player_id = get_player_id(db, player_name, team_abbrev, season_id)

                # Determine is_active based on game status
                # If game_status is "Injured Reserve" or "Out", player is not active
                is_active = 0 if game_status in ['Injured Reserve', 'Out'] else 1

                injury_records.append({
                    'season_id': season_id,
                    'plyr_id': player_id,
                    'plyr_name': player_name,
                    'team_id': team_id,
                    'game_id': game_id,
                    'week_id': target_week_id,
                    'practice_status': practice_status,
                    'game_status': game_status,
                    'is_active': is_active
                })

            except ValueError as e:
                print(f"Warning: Could not find player {player_name} ({team_abbrev}): {e}")
                continue

        return injury_records

    except Exception as e:
        print(f"Error processing team_injuries file {file_path}: {e}")
        return []

def process_abbrev_injury_file(db: DatabaseConnector, file_path: str, season_id: int,
                               target_week_num: int, target_week_id: int,
                               team_id: int, game_id: int) -> list:
    """Process team abbreviation injury report CSV file

    Args:
        file_path: Path to {TEAM_ABBREV}_injury_report CSV file
        season_id: Season ID
        target_week_num: Week number to process (WEEK + 1)
        target_week_id: Week ID for the game
        team_id: Team ID
        game_id: Game ID for the game

    Returns:
        list: List of dictionaries containing injury report data
    """
    try:
        df = pd.read_csv(file_path)

        if df.empty:
            return []

        # Column name for the target week
        playing_status_col = f'playing_status_week_{target_week_num}'

        if playing_status_col not in df.columns:
            print(f"Warning: Column {playing_status_col} not found in {file_path}")
            return []

        injury_records = []

        for _, row in df.iterrows():
            player_name = row['Player']
            team_name = row['Team']
            was_active = row[playing_status_col]

            # Convert team name to abbreviation
            team_abbrev = standardize_team_name(team_name)

            try:
                # Get player ID
                player_id = get_player_id(db, player_name, team_abbrev, season_id)

                # Convert was_active to int (should already be 0 or 1)
                was_active = int(was_active) if pd.notna(was_active) else 0

                injury_records.append({
                    'season_id': season_id,
                    'plyr_id': player_id,
                    'plyr_name': player_name,
                    'team_id': team_id,
                    'game_id': game_id,
                    'week_id': target_week_id,
                    'was_active': was_active
                })

            except ValueError as e:
                print(f"Warning: Could not find player {player_name} ({team_abbrev}): {e}")
                continue

        return injury_records

    except Exception as e:
        print(f"Error processing abbrev injury file {file_path}: {e}")
        return []

def merge_injury_records(team_injuries_records: list, abbrev_records: list) -> list:
    """Merge records from both CSV files, combining data for the same players

    Args:
        team_injuries_records: Records from team_injuries file
        abbrev_records: Records from {TEAM_ABBREV}_injury_report file

    Returns:
        list: Merged list of injury records
    """
    # Create a dict keyed by plyr_id for efficient merging
    merged_dict = {}

    # Add team_injuries records
    for record in team_injuries_records:
        merged_dict[record['plyr_id']] = record.copy()

    # Merge in abbrev records
    for record in abbrev_records:
        plyr_id = record['plyr_id']
        if plyr_id in merged_dict:
            # Add was_active to existing record
            merged_dict[plyr_id]['was_active'] = record['was_active']
        else:
            # Add new record (player not in team_injuries file)
            merged_dict[plyr_id] = record.copy()

    return list(merged_dict.values())

def set_all_other_players_active(db: DatabaseConnector, processed_player_ids: set,
                                 season_id: int, week_id: int) -> list:
    """Set is_active=1 for all players not in the injury report

    Args:
        db: Database connector
        processed_player_ids: Set of player IDs already processed
        season_id: Season ID
        week_id: Week ID

    Returns:
        list: List of active player records
    """
    try:
        # Get all players for this season
        query = """
        SELECT p.plyr_id, p.plyr_name, p.team_id
        FROM plyr p
        WHERE p.season_id = %s
        """
        all_players = db.fetch_all(query, (season_id,))

        active_records = []

        for player_id, player_name, team_id in all_players:
            if player_id not in processed_player_ids:
                # Get this team's game for the week
                game_query = """
                SELECT game_id
                FROM nfl_game
                WHERE season_id = %s
                AND week_id = %s
                AND (home_team_id = %s OR away_team_id = %s)
                """
                games = db.fetch_all(game_query, (season_id, week_id, team_id, team_id))

                if games:
                    game_id = games[0][0]
                    active_records.append({
                        'season_id': season_id,
                        'plyr_id': player_id,
                        'plyr_name': player_name,
                        'team_id': team_id,
                        'game_id': game_id,
                        'week_id': week_id,
                        'is_active': 1
                    })

        return active_records

    except Exception as e:
        print(f"Error setting active players: {e}")
        return []

def upsert_injury_data(db: DatabaseConnector, data_list: list):
    """Upsert injury report data using batch processing"""
    if not data_list:
        return 0, 0

    query = """
    INSERT INTO injury_report (season_id, plyr_id, plyr_name, team_id, game_id, week_id,
                              was_active, is_active, practice_status, game_status)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
    plyr_name = VALUES(plyr_name),
    team_id = VALUES(team_id),
    game_id = VALUES(game_id),
    was_active = COALESCE(VALUES(was_active), was_active),
    is_active = COALESCE(VALUES(is_active), is_active),
    practice_status = COALESCE(VALUES(practice_status), practice_status),
    game_status = COALESCE(VALUES(game_status), game_status)
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
            data.get('was_active', None),
            data.get('is_active', None),
            data.get('practice_status', None),
            data.get('game_status', None)
        ))

    success, rows_affected = db.execute_many(query, batch_data)

    if success:
        # Estimate inserts vs updates
        estimated_inserts = max(0, len(batch_data) * 2 - rows_affected)
        estimated_updates = len(batch_data) - estimated_inserts
        return estimated_inserts, estimated_updates
    else:
        return 0, 0

def main():
    """Main function to process injury report data"""
    print("Injury Report Script - NFL Player Injury Status Tracker")
    print("=" * 60)
    print(f"Processing data for Year: {YEAR}, Week: {WEEK}")

    # Target week is WEEK + 1 (upcoming game)
    target_week_num = WEEK + 1
    print(f"Target week for injury reports: {target_week_num}")

    # Construct the source directory path
    source_dir = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\injury_report\\week_{WEEK}"
    print(f"Source directory: {source_dir}")

    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory does not exist: {source_dir}")
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

        # Get season_id and target_week_id
        try:
            season_id = get_season_id(db, YEAR)
            target_week_id = get_week_id(db, season_id, target_week_num)
            print(f"Using season_id: {season_id}, target_week_id: {target_week_id}")
        except ValueError as e:
            print(f"Error: {e}")
            return

        # Get all team directories
        team_dirs = [d for d in glob.glob(os.path.join(source_dir, "*")) if os.path.isdir(d)]

        if not team_dirs:
            print(f"No team directories found in {source_dir}")
            return

        print(f"Found {len(team_dirs)} team directories")

        all_injury_records = []
        processed_player_ids = set()
        total_teams_processed = 0

        for team_dir in team_dirs:
            team_name = os.path.basename(team_dir).replace('_', ' ')
            print(f"\nProcessing {team_name}...")

            # Get most recent file pair
            team_injuries_file, abbrev_file = get_most_recent_file_pair(team_dir)

            if not team_injuries_file or not abbrev_file:
                print(f"  Warning: Could not find file pair for {team_name}")
                continue

            # Convert team name to abbreviation
            team_abbrev = standardize_team_name(team_name)

            try:
                team_id = get_team_id(db, team_abbrev)
            except ValueError as e:
                print(f"  Warning: {e}")
                continue

            # Get game_id for this team and week
            game_query = """
            SELECT game_id
            FROM nfl_game
            WHERE season_id = %s
            AND week_id = %s
            AND (home_team_id = %s OR away_team_id = %s)
            """
            games = db.fetch_all(game_query, (season_id, target_week_id, team_id, team_id))

            if not games:
                print(f"  Warning: No game found for {team_name} in week {target_week_num}")
                continue

            game_id = games[0][0]

            # Process both files
            team_injuries_records = process_team_injuries_file(
                db, team_injuries_file, season_id, target_week_id, team_id, game_id
            )

            abbrev_records = process_abbrev_injury_file(
                db, abbrev_file, season_id, target_week_num, target_week_id, team_id, game_id
            )

            # Merge records
            merged_records = merge_injury_records(team_injuries_records, abbrev_records)

            print(f"  Processed {len(merged_records)} player records")

            all_injury_records.extend(merged_records)
            processed_player_ids.update([r['plyr_id'] for r in merged_records])
            total_teams_processed += 1

        # Set all other players as active
        print(f"\nSetting active status for players not in injury reports...")
        active_records = set_all_other_players_active(db, processed_player_ids, season_id, target_week_id)
        print(f"Found {len(active_records)} players to mark as active")

        all_injury_records.extend(active_records)

        # Upsert all data
        if all_injury_records:
            print(f"\nUpserting {len(all_injury_records)} total injury report records...")
            total_inserted, total_updated = upsert_injury_data(db, all_injury_records)

            # Summary
            print("\nProcessing Summary:")
            print(f"Total teams processed: {total_teams_processed}")
            print(f"Total records upserted: {len(all_injury_records)}")
            print(f"Estimated inserts: {total_inserted}")
            print(f"Estimated updates: {total_updated}")
            print("Script execution completed successfully")
        else:
            print("\nNo injury records to upsert")

    except Exception as e:
        print(f"Error in main execution: {e}")

    finally:
        db.disconnect()

if __name__ == "__main__":
    main()