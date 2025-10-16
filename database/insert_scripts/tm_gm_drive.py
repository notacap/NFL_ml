#!/usr/bin/env python3

import os
import sys
import pandas as pd
import glob
from pathlib import Path
import re

# Add the parent directory to the Python path to import db_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_utils import DatabaseConnector, YEAR, WEEK_START, WEEK_END, batch_upsert_data, handle_null_values, get_season_id, get_week_id, get_team_id, get_game_id, create_table_if_not_exists


def create_tm_gm_drive_table(db: DatabaseConnector) -> bool:
    """Create the tm_gm_drive table if it doesn't exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS tm_gm_drive (
        tm_gm_drive_id INT PRIMARY KEY AUTO_INCREMENT,
        team_id INT,
        week_id INT,
        game_id INT,
        season_id INT,
        tm_gm_dr_num INT,
        tm_gm_dr_qtr_id INT,
        tm_gm_dr_strt_fld_pos INT,
        tm_gm_dr_ply INT,
        tm_gm_dr_time TIME,
        tm_gm_dr_net_yds INT,
        tm_gm_dr_res ENUM('touchdown', 'field_goal', 'interception', 'fumble', 'punt', 'missed_field_goal', 'safety', 'downs', 'time', 'penalty', 'blocked_punt', 'blocked_field_goal'),
        tm_gm_dr_res_secondary ENUM('touchdown', 'field_goal', 'interception', 'fumble', 'punt', 'missed_field_goal', 'safety', 'downs', 'time', 'penalty', 'blocked_punt', 'blocked_field_goal'),
        FOREIGN KEY (team_id) REFERENCES nfl_team(team_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (game_id) REFERENCES nfl_game(game_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        FOREIGN KEY (tm_gm_dr_qtr_id) REFERENCES nfl_gm_quarter(nfl_quarter_id),
        UNIQUE KEY uk_team_game_drive (team_id, game_id, tm_gm_dr_num)
    );
    """
    
    return create_table_if_not_exists(db, 'tm_gm_drive', create_table_sql)


def parse_field_position(los_value):
    """Parse field position from 'TEAM ##' format to extract yard line number."""
    if pd.isna(los_value) or not los_value or str(los_value).strip() == '':
        return None
    
    los_str = str(los_value).strip()
    # Pattern: "TEAM ##" or "OWN ##" or "MID ##"
    match = re.search(r'(\w+)\s+(\d+)', los_str)
    if match:
        return int(match.group(2))
    
    # Handle edge cases like just a number
    if los_str.isdigit():
        return int(los_str)
    
    return None


def parse_time_to_time_format(time_value):
    """Convert time string like '4:29' to TIME format '00:04:29'."""
    if pd.isna(time_value) or not time_value or str(time_value).strip() == '':
        return None
    
    time_str = str(time_value).strip()
    # Pattern: "MM:SS"
    if ':' in time_str:
        parts = time_str.split(':')
        if len(parts) == 2:
            try:
                minutes = int(parts[0])
                seconds = int(parts[1])
                # Convert to TIME format: 00:MM:SS
                return f"00:{minutes:02d}:{seconds:02d}"
            except ValueError:
                return None
    
    return None


def map_drive_result(result_value):
    """Map CSV drive result to database ENUM values.

    Handles comma-separated values by returning both primary and secondary outcomes.
    Example: 'Fumble, Safety' -> ('fumble', 'safety')
    Example: 'Touchdown' -> ('touchdown', None)

    Returns:
        tuple: (primary_result, secondary_result) where secondary_result can be None
    """
    if pd.isna(result_value) or not result_value or str(result_value).strip() == '':
        return (None, None)

    result_str = str(result_value).strip()

    # Mapping from CSV values to database ENUM values
    result_mapping = {
        'Field Goal': 'field_goal',
        'Touchdown': 'touchdown',
        'Punt': 'punt',
        'Interception': 'interception',
        'Fumble': 'fumble',
        'Downs': 'downs',
        'End of Game': 'time',
        'End of Half': 'time',
        'Safety': 'safety',
        'Missed FG': 'missed_field_goal',
        'Blocked Punt': 'blocked_punt',
        'Blocked FG': 'blocked_field_goal'
    }

    # Handle comma-separated values
    if ',' in result_str:
        parts = [part.strip() for part in result_str.split(',')]
        primary = result_mapping.get(parts[0], None)
        secondary = result_mapping.get(parts[1], None) if len(parts) > 1 else None
        return (primary, secondary)
    else:
        # Single value - no secondary result
        return (result_mapping.get(result_str, None), None)


def get_quarter_id(db: DatabaseConnector, quarter_value) -> int:
    """Get nfl_quarter_id from nfl_gm_quarter table based on quarter value.
    
    Handles numeric quarters (1-5) and overtime variations ('OT', 'Overtime') which map to quarter 5.
    """
    try:
        # Handle overtime variations - map to quarter 5
        if quarter_value in ['OT', 'Overtime']:
            quarter_num = 5
        else:
            # Handle numeric quarters
            try:
                quarter_num = int(quarter_value)
            except (ValueError, TypeError):
                print(f"[WARNING] Unable to convert quarter value to int: {quarter_value}")
                return None
        
        query = "SELECT nfl_quarter_id FROM nfl_gm_quarter WHERE quarter = %s"
        result = db.fetch_all(query, (quarter_num,))
        if result:
            return result[0][0]
        else:
            raise ValueError(f"No quarter_id found for quarter {quarter_num}")
    except Exception as e:
        print(f"[ERROR] Error looking up quarter_id for quarter {quarter_value}: {e}")
        raise


def get_team_abbreviation_from_partial_name(db: DatabaseConnector, partial_team_name: str) -> str:
    """Convert partial team name (like 'Steelers') to abbreviation (like 'PIT') by matching team_name column."""
    try:
        # Use LIKE to find team_name that contains the partial name
        query = "SELECT abrv FROM nfl_team WHERE team_name LIKE %s"
        result = db.fetch_all(query, (f"%{partial_team_name}%",))
        if result:
            return result[0][0]
        else:
            print(f"[WARNING] Could not find team abbreviation for partial team name: {partial_team_name}")
            return None
    except Exception as e:
        print(f"[ERROR] Error looking up team abbreviation for {partial_team_name}: {e}")
        return None


def process_drive_csv_file(db: DatabaseConnector, file_path: str, season_id: int, week: int) -> pd.DataFrame:
    """Process a single drive CSV file and return processed data."""
    print(f"Processing drive CSV file: {file_path}")
    
    try:
        df = pd.read_csv(file_path, header=0)
        
        if df.empty:
            print(f"[WARNING] Skipping empty file: {file_path}")
            return pd.DataFrame()
        
        # Get week number from DataFrame first, fallback to passed week number
        week_num = None
        if 'week' in df.columns:
            week_num = df['week'].iloc[0]

        if week_num is None:
            week_num = float(week)

        week_id = get_week_id(db, season_id, int(week_num))
        
        # Get unique team names from the 'team' column  
        unique_teams = df['team'].dropna().unique()
        if len(unique_teams) != 1:
            print(f"[WARNING] Expected 1 team in drive CSV file, found {len(unique_teams)}: {unique_teams}")
            return pd.DataFrame()
        
        # Extract team names from filename to get both teams for game_id lookup
        # Filename format: cleaned_Away_Team_Home_Team_wkX.0_2024_gm_{away|home}_drives_timestamp.csv
        filename = os.path.basename(file_path)
        try:
            # Parse filename to extract team names
            parts = filename.split('_')
            if len(parts) < 6:
                print(f"[WARNING] Unable to parse filename format: {filename}")
                return pd.DataFrame()
            
            # Find the position of the week indicator (wkX.0 or wkX)
            week_idx = -1
            for i, part in enumerate(parts):
                if part.startswith('wk'):
                    week_idx = i
                    break

            if week_idx == -1:
                print(f"[WARNING] Could not find week indicator in filename: {filename}")
                return pd.DataFrame()
            
            # Extract team names (everything between 'cleaned_' and '_wkX' or '_wkX.0')
            team_parts = parts[1:week_idx]
            
            # Split into away and home teams by finding where one team name ends and the other begins
            # This is tricky because team names can have multiple parts (e.g., "New_York_Giants")
            # We need to use the home_team flag from CSV to determine which team this file represents
            home_team_flag = df['home_team'].iloc[0] if 'home_team' in df.columns else None
            
            if home_team_flag is None:
                print(f"[WARNING] No home_team flag in CSV file: {file_path}")
                return pd.DataFrame()
            
            # Reconstruct full team names from filename
            # For now, we'll use a simpler approach - look for the team from CSV in the filename
            current_team = unique_teams[0]
            current_team_abrv = get_team_abbreviation_from_partial_name(db, current_team)
            
            if not current_team_abrv:
                print(f"[WARNING] Could not find team abbreviation for {current_team}")
                return pd.DataFrame()
            
            # For game_id lookup, we need both teams. We'll extract from filename differently.
            # The filename contains both team names, so we need to identify which is which
            
            # Alternative approach: Use the year/week and the known team to find the game
            # Find all games for this week and season where current team played
            query = """
                SELECT g.game_id, t1.abrv as home_team, t2.abrv as away_team
                FROM nfl_game g
                JOIN nfl_team t1 ON g.home_team_id = t1.team_id  
                JOIN nfl_team t2 ON g.away_team_id = t2.team_id
                WHERE g.season_id = %s AND g.week_id = %s 
                AND (t1.abrv = %s OR t2.abrv = %s)
            """
            
            games = db.fetch_all(query, (season_id, week_id, current_team_abrv, current_team_abrv))
            
            if len(games) == 1:
                game_id = games[0][0]
            else:
                print(f"[WARNING] Found {len(games)} games for team {current_team_abrv} in week {week}, expected 1")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"[ERROR] Error parsing filename or looking up game: {e}")
            return pd.DataFrame()
        
        # Process each drive row
        processed_rows = []
        for index, row in df.iterrows():
            try:
                # Get team name from row
                partial_team_name = row.get('team', '')
                
                if not partial_team_name or pd.isna(partial_team_name) or str(partial_team_name).strip() == '':
                    print(f"[WARNING] Skipping row {index} - no team information")
                    continue
                
                # Get team_id by matching partial team name in database team_name column
                try:
                    query = "SELECT team_id FROM nfl_team WHERE team_name LIKE %s"
                    result = db.fetch_all(query, (f"%{partial_team_name}%",))
                    if result:
                        team_id = result[0][0]
                    else:
                        print(f"[WARNING] Could not find team_id for partial team name: {partial_team_name}")
                        continue
                except Exception as e:
                    print(f"[ERROR] Error looking up team_id for {partial_team_name}: {e}")
                    continue
                
                # Create processed row
                processed_row = {
                    'team_id': team_id,
                    'week_id': week_id,
                    'game_id': game_id,
                    'season_id': season_id
                }
                
                # Map CSV columns to database columns
                processed_row['tm_gm_dr_num'] = row.get('#') if pd.notna(row.get('#')) else None
                processed_row['tm_gm_dr_ply'] = row.get('Plays') if pd.notna(row.get('Plays')) else None
                processed_row['tm_gm_dr_net_yds'] = row.get('Net Yds') if pd.notna(row.get('Net Yds')) else None
                
                # Get quarter_id from quarter value (handles numeric and overtime variations)
                quarter_value = row.get('Quarter')
                if pd.notna(quarter_value) and str(quarter_value).strip() != '':
                    quarter_id = get_quarter_id(db, quarter_value)
                    processed_row['tm_gm_dr_qtr_id'] = quarter_id
                else:
                    processed_row['tm_gm_dr_qtr_id'] = None
                
                # Parse special fields
                processed_row['tm_gm_dr_strt_fld_pos'] = parse_field_position(row.get('LOS'))
                processed_row['tm_gm_dr_time'] = parse_time_to_time_format(row.get('Length'))

                # Map drive result (returns tuple of primary, secondary)
                primary_result, secondary_result = map_drive_result(row.get('Result'))
                processed_row['tm_gm_dr_res'] = primary_result
                processed_row['tm_gm_dr_res_secondary'] = secondary_result
                
                processed_rows.append(processed_row)
                
            except Exception as e:
                print(f"[ERROR] Error processing row {index}: {e}")
                continue
        
        if processed_rows:
            df_result = pd.DataFrame(processed_rows)
            # Handle null values appropriately for database insertion
            df_result = handle_null_values(df_result)
            print(f"[SUCCESS] Processed {len(processed_rows)} drive records from: {os.path.basename(file_path)}")
            return df_result
        else:
            print(f"[WARNING] No valid drive records processed from: {file_path}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"[ERROR] Processing file {file_path}: {e}")
        return pd.DataFrame()


def get_drive_csv_files() -> list:
    """Get list of CSV files to process based on WEEK_START and WEEK_END."""
    base_dir = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\games"
    csv_files = []
    
    for week in range(WEEK_START, WEEK_END + 1):
        week_dir = os.path.join(base_dir, f"week_{week}.0")
        clean_dir = os.path.join(week_dir, "clean")
        if os.path.exists(clean_dir):
            pattern = os.path.join(clean_dir, "cleaned_*_gm_*_drives_*.csv")
            week_files = glob.glob(pattern)
            for file_path in week_files:
                csv_files.append((week, file_path))
        else:
            print(f"[WARNING] Clean directory not found: {clean_dir}")
    
    print(f"Found {len(csv_files)} drive CSV files")
    return sorted(csv_files)


def main():
    """Main function to process team game drive data."""
    print("=== Team Game Drive Import Script ===")
    
    db = DatabaseConnector()
    
    # Connect to database
    print("Connecting to database...")
    if not db.connect():
        print("[ERROR] Failed to connect to database")
        return
    
    # Create table if it doesn't exist
    print("Creating tm_gm_drive table if it doesn't exist...")
    if not create_tm_gm_drive_table(db):
        print("[ERROR] Failed to create tm_gm_drive table")
        return
    
    # Get season_id
    season_id = get_season_id(db, YEAR)
    print(f"Processing data for season {YEAR} (season_id: {season_id})")
    
    # Get CSV files
    csv_files = get_drive_csv_files()
    if not csv_files:
        print("[ERROR] No drive CSV files found")
        return
    
    # Process files and collect data
    all_processed_data = []
    for week, file_path in csv_files:
        try:
            processed_df = process_drive_csv_file(db, file_path, season_id, week)
            if not processed_df.empty:
                all_processed_data.append(processed_df)
        except Exception as e:
            print(f"[ERROR] Failed to process file {file_path}: {e}")
            continue
    
    if not all_processed_data:
        print("[ERROR] No data was successfully processed from any files")
        return
    
    # Combine all processed data
    final_df = pd.concat(all_processed_data, ignore_index=True)
    print(f"Total drive records to upsert: {len(final_df)}")
    
    # Batch upsert data
    print("Starting batch upsert operation...")
    success = batch_upsert_data(db, 'tm_gm_drive', final_df)
    
    if success:
        print(f"[SUCCESS] Successfully upserted {len(final_df)} team drive records")
    else:
        print("[ERROR] Failed to upsert team drive data")


if __name__ == "__main__":
    main()