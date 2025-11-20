"""
Injured Reserve Script - NFL Player IR Status Tracker

Inserts/upserts NFL player injured reserve status information into the injured_reserve table.
Ingests the same source CSV file as the past_injuries.py script but only processes
plyr_name and is_on_ir columns for the specified week.
"""

import sys
import os
import glob
from datetime import datetime
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_utils import (DatabaseConnector, YEAR, WEEK, get_season_id, get_week_id,
                     get_team_id, get_player_id, create_table_if_not_exists,
                     standardize_team_name, interactive_player_selection)

def create_injured_reserve_table(db: DatabaseConnector) -> bool:
    """Create the injured_reserve table if it doesn't exist"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS injured_reserve (
        injury_report_id INT AUTO_INCREMENT PRIMARY KEY,
        season_id INT,
        plyr_id INT,
        plyr_name VARCHAR(255) NOT NULL,
        team_id INT,
        week_id INT,
        is_on_ir TINYINT,
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        FOREIGN KEY (plyr_id) REFERENCES plyr(plyr_id),
        FOREIGN KEY (team_id) REFERENCES nfl_team(team_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        UNIQUE KEY uk_plyr_season_game (plyr_id, season_id)
    )
    """

    return create_table_if_not_exists(db, 'injured_reserve', create_table_sql)

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

def get_player_id_interactive(db: DatabaseConnector, player_name: str, team_abrv: str, season_id: int) -> int:
    """Get player_id with interactive selection for multiple matches"""
    # Generate name variations to handle suffixes
    suffixes = ["II", "III", "IV", "Jr.", "Sr."]
    name_variations = [player_name] + [f"{player_name} {suffix}" for suffix in suffixes]
    placeholders = ', '.join(['%s'] * len(name_variations))

    if team_abrv and team_abrv.strip():
        # Search with team filtering for better accuracy
        query = f"""
        SELECT p.plyr_id, p.plyr_name, 'plyr' AS source, p.team_id, t.abrv, p.plyr_pos, p.plyr_age
        FROM plyr p
        JOIN nfl_team t ON p.team_id = t.team_id
        WHERE p.plyr_name IN ({placeholders}) AND (t.abrv = %s OR t.alt_abrv = %s) AND p.season_id = %s
        UNION
        SELECT mtp.plyr_id, mtp.plyr_name, 'multi_tm_plyr' AS source,
               COALESCE(mtp.former_tm_id, mtp.first_tm_id) AS team_id,
               COALESCE(t1.abrv, t2.abrv) AS abrv,
               mtp.plyr_pos, mtp.plyr_age
        FROM multi_tm_plyr mtp
        LEFT JOIN nfl_team t1 ON mtp.former_tm_id = t1.team_id
        LEFT JOIN nfl_team t2 ON mtp.first_tm_id = t2.team_id
        WHERE mtp.plyr_name IN ({placeholders}) AND (t1.abrv = %s OR t1.alt_abrv = %s OR t2.abrv = %s OR t2.alt_abrv = %s) AND mtp.season_id = %s
        """
        params = name_variations + [team_abrv, team_abrv, season_id] + name_variations + [team_abrv, team_abrv, team_abrv, team_abrv, season_id]
    else:
        # Search without team filtering if team not available
        query = f"""
        SELECT p.plyr_id, p.plyr_name, 'plyr' AS source, p.team_id, t.abrv, p.plyr_pos, p.plyr_age
        FROM plyr p
        JOIN nfl_team t ON p.team_id = t.team_id
        WHERE p.plyr_name IN ({placeholders}) AND p.season_id = %s
        """
        params = name_variations + [season_id]

    results = db.fetch_all(query, params)

    if len(results) == 1:
        return results[0][0]
    elif len(results) > 1:
        # Multiple matches found - use interactive selection
        print(f"\n[WARNING] Multiple matches found for player: {player_name} (Team: {team_abrv})")
        selected_id = interactive_player_selection(
            player_name, team_abrv, None, None, results
        )
        if selected_id == 0:
            return None  # User chose to skip this player
        return selected_id
    else:
        print(f"[WARNING] No player found for {player_name} ({team_abrv}) in season {season_id}")
        return None

def process_csv_row(db: DatabaseConnector, row, season_id: int, week_id: int):
    """Process a single CSV row and generate injured reserve data"""
    try:
        # Skip rows where is_on_ir is not set or is null
        if pd.isna(row.get('is_on_ir')) or row.get('is_on_ir') == '':
            return None

        # Convert full team name to abbreviation for player lookup
        team_abbrev = standardize_team_name(row['current_team'])

        # Get player ID using interactive selection
        player_id = get_player_id_interactive(db, row['plyr_name'], team_abbrev, season_id)

        if player_id is None:
            print(f"[WARNING] Skipping player {row['plyr_name']} - no valid player ID found")
            return None

        # Get team_id
        team_id = get_team_id(db, team_abbrev)

        # Parse is_on_ir value (should be 1 for IR, 0 for not on IR)
        is_on_ir = 1 if str(row['is_on_ir']).strip().lower() in ['1', 'true', 'yes', 'ir'] else 0

        return {
            'season_id': season_id,
            'plyr_id': player_id,
            'plyr_name': row['plyr_name'],
            'team_id': team_id,
            'week_id': week_id,
            'is_on_ir': is_on_ir
        }

    except ValueError as e:
        print(f"[WARNING] Error processing player {row['plyr_name']}: {e}")
        return None

def upsert_injured_reserve_data(db: DatabaseConnector, data_list):
    """Upsert injured reserve data using batch processing"""
    if not data_list:
        return 0, 0

    query = """
    INSERT INTO injured_reserve (season_id, plyr_id, plyr_name, team_id, week_id, is_on_ir)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
    plyr_name = VALUES(plyr_name),
    team_id = VALUES(team_id),
    week_id = VALUES(week_id),
    is_on_ir = VALUES(is_on_ir)
    """

    # Convert data to tuples for batch insert
    batch_data = []
    for data in data_list:
        batch_data.append((
            data['season_id'],
            data['plyr_id'],
            data['plyr_name'],
            data['team_id'],
            data['week_id'],
            data['is_on_ir']
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
    """Main function to process injured reserve data"""
    print("Injured Reserve Script - NFL Player IR Status Tracker")
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

        # Create injured_reserve table if it doesn't exist
        if not create_injured_reserve_table(db):
            print("Failed to create injured_reserve table")
            return

        # Get foreign key IDs using db_utils functions
        try:
            season_id = get_season_id(db, YEAR)
            week_id = get_week_id(db, season_id, WEEK)
            print(f"Using season_id: {season_id}, week_id: {week_id} for year {YEAR}, week {WEEK}")
        except ValueError as e:
            print(f"Error: {e}")
            return

        # Load and process CSV data
        print(f"Loading data from: {csv_file_path}")
        df = pd.read_csv(csv_file_path)

        if df.empty:
            print("No data loaded from CSV file")
            return

        print(f"Processing injured reserve data for week {WEEK} of season {season_id}")

        total_players = 0
        total_rows_inserted = 0
        total_rows_updated = 0
        processed_data = []

        # Process each player row
        for _, row in df.iterrows():
            total_players += 1
            result = process_csv_row(db, row, season_id, week_id)

            if result:
                processed_data.append(result)

            if total_players % 50 == 0:
                print(f"Processed {total_players} players...")

        # Batch upsert all data
        if processed_data:
            print(f"Upserting {len(processed_data)} injured reserve records...")
            batch_inserted, batch_updated = upsert_injured_reserve_data(db, processed_data)
            total_rows_inserted += batch_inserted
            total_rows_updated += batch_updated

        # Summary
        print("\nProcessing Summary:")
        print(f"Total players processed: {total_players}")
        print(f"Valid IR records found: {len(processed_data)}")
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