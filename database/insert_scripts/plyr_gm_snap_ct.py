#!/usr/bin/env python3

import os
import sys
import pandas as pd
import glob
from pathlib import Path

# Add the parent directory to the Python path to import db_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_utils import DatabaseConnector, YEAR, WEEK_START, WEEK_END, batch_upsert_data, handle_null_values, get_season_id, get_week_id, get_player_id, get_game_id, convert_team_nickname_to_abbreviation


def create_plyr_gm_snap_ct_table(db: DatabaseConnector) -> bool:
    """Create the plyr_gm_snap_ct table if it doesn't exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS plyr_gm_snap_ct (
        plyr_gm_snap_count_id INT PRIMARY KEY AUTO_INCREMENT,
        plyr_id INT,
        week_id INT,
        game_id INT,
        season_id INT,
        team_id INT,
        plyr_gm_off_snap_ct INT,
        plyr_gm_off_snap_ct_pct FLOAT(5,4),
        plyr_gm_def_snap_ct INT,
        plyr_gm_def_snap_ct_pct FLOAT(5,4),
        plyr_gm_st_snap_ct INT,
        plyr_gm_st_snap_ct_pct FLOAT(5,4),
        FOREIGN KEY (plyr_id) REFERENCES plyr(plyr_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (game_id) REFERENCES nfl_game(game_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        FOREIGN KEY (team_id) REFERENCES nfl_team(team_id),
        UNIQUE KEY uk_player_game (plyr_id, game_id)
    );
    """
    
    try:
        success = db.execute_query(create_table_sql)
        if success:
            print("[OK] plyr_gm_snap_ct table created/verified successfully")
            return True
        else:
            print("[FAIL] Failed to create plyr_gm_snap_ct table")
            return False
    except Exception as e:
        print(f"[ERROR] Error creating plyr_gm_snap_ct table: {e}")
        return False


# NOTE: These functions are now imported from db_utils.py for consistency


def extract_teams_from_filename(filename: str) -> tuple:
    """Extract team names from snap count filename.
    
    Example: cleaned_Tampa_Bay_Buccaneers_Washington_Commanders_wk1.0_2024_gm_home_snap_counts_20250825_013423.csv
    Returns: ('Tampa Bay Buccaneers', 'Washington Commanders')
    """
    try:
        # Remove cleaned_ prefix and split by underscores
        if filename.startswith('cleaned_'):
            filename = filename[8:]  # Remove 'cleaned_' prefix
        
        parts = filename.split('_')
        
        # Find where the week indicator starts
        week_start_idx = None
        for i, part in enumerate(parts):
            if part.startswith('wk'):
                week_start_idx = i
                break
        
        if week_start_idx is None:
            return None
            
        # Get team parts (everything before wk indicator)
        team_parts = parts[:week_start_idx]
        
        # Known multi-word team names
        multi_word_teams = [
            ['Tampa', 'Bay', 'Buccaneers'],
            ['New', 'York', 'Giants'], 
            ['New', 'York', 'Jets'],
            ['New', 'England', 'Patriots'],
            ['Green', 'Bay', 'Packers'],
            ['Kansas', 'City', 'Chiefs'],
            ['Las', 'Vegas', 'Raiders'],
            ['Los', 'Angeles', 'Chargers'],
            ['Los', 'Angeles', 'Rams'],
            ['San', 'Francisco', '49ers'],
            ['New', 'Orleans', 'Saints']
        ]
        
        # Try to match known multi-word teams
        team_parts_str = ' '.join(team_parts)
        
        for multi_team in multi_word_teams:
            team_name = ' '.join(multi_team)
            if team_parts_str.startswith(team_name):
                team1_name = team_name
                # Remove team1 from the parts and get remaining as team2
                remaining = team_parts_str[len(team_name):].strip()
                
                # Check if remaining is also a known multi-word team
                for multi_team2 in multi_word_teams:
                    team2_name = ' '.join(multi_team2)
                    if remaining == team2_name:
                        return (team1_name, team2_name)
                
                # If not a known multi-word team, use remaining as single team
                if remaining:
                    return (team1_name, remaining)
        
        # Fallback: split in half
        mid_point = len(team_parts) // 2
        team1_parts = team_parts[:mid_point]
        team2_parts = team_parts[mid_point:]
        
        team1_name = ' '.join(team1_parts)
        team2_name = ' '.join(team2_parts)
        
        return (team1_name, team2_name)
        
    except Exception as e:
        print(f"[WARNING] Error parsing filename {filename}: {e}")
        return None


def get_team_abbreviation(db: DatabaseConnector, team_name: str) -> str:
    """Get team abbreviation from team name."""
    # Try exact match first
    query = "SELECT abrv FROM nfl_team WHERE team_name = %s"
    result = db.fetch_all(query, (team_name,))
    if result:
        return result[0][0]
    
    # Try partial match
    query = "SELECT abrv FROM nfl_team WHERE team_name LIKE %s"
    result = db.fetch_all(query, (f"%{team_name}%",))
    if result:
        return result[0][0]
    
    raise ValueError(f"No team abbreviation found for: {team_name}")


def get_team_id_by_partial_name(db: DatabaseConnector, partial_team_name: str) -> int:
    """Get team_id from nfl_team table using partial team name matching."""
    # Try exact match first with team_name
    query = "SELECT team_id FROM nfl_team WHERE team_name = %s"
    result = db.fetch_all(query, (partial_team_name,))
    if result:
        return result[0][0]
    
    # Try partial match with team_name (look for the partial name within the full team name)
    query = "SELECT team_id FROM nfl_team WHERE team_name LIKE %s"
    result = db.fetch_all(query, (f"%{partial_team_name}%",))
    if result:
        if len(result) == 1:
            return result[0][0]
        else:
            # Multiple matches, try to be more specific
            print(f"[WARNING] Multiple team matches found for '{partial_team_name}': {[r[0] for r in result]}")
            return result[0][0]
    
    # Try by abbreviation
    query = "SELECT team_id FROM nfl_team WHERE abrv = %s OR alt_abrv = %s"
    result = db.fetch_all(query, (partial_team_name, partial_team_name))
    if result:
        return result[0][0]
    
    # If still not found, raise error
    raise ValueError(f"No team found for partial name '{partial_team_name}'")


# NOTE: This function is replaced by the centralized get_game_id from db_utils.py
# Legacy implementation kept for reference
def get_game_id_legacy(db: DatabaseConnector, season_id: int, week_id: int, team1_name: str, team2_name: str) -> int:
    """Get game_id based on season_id, week_id and two team names from filename."""
    
    # Convert team names to standardized format for matching
    team1_standardized = team1_name.replace('_', ' ')
    team2_standardized = team2_name.replace('_', ' ')
    
    # Get team IDs by matching full team names
    query = """
    SELECT team_id, team_name FROM nfl_team 
    WHERE team_name = %s OR team_name = %s
    """
    result = db.fetch_all(query, (team1_standardized, team2_standardized))
    
    if len(result) != 2:
        raise ValueError(f"Could not find both teams: {team1_standardized} and {team2_standardized}")
    
    team1_id = result[0][0]
    team2_id = result[1][0]
    
    # Find game where both teams are present (home or away)
    query = """
    SELECT game_id 
    FROM nfl_game 
    WHERE season_id = %s 
    AND week_id = %s 
    AND ((home_team_id = %s AND away_team_id = %s) 
         OR (home_team_id = %s AND away_team_id = %s))
    """
    
    result = db.fetch_all(query, (season_id, week_id, team1_id, team2_id, team2_id, team1_id))
    if result:
        return result[0][0]
    else:
        raise ValueError(f"No game found for season {season_id}, week {week_id}, teams {team1_standardized} vs {team2_standardized}")


# NOTE: This function is replaced by the centralized get_player_id from db_utils.py  
# Legacy implementation kept for reference
def get_player_id_legacy(db: DatabaseConnector, player_name: str, team_partial_name: str, season_id: int) -> int:
    """Get player_id using logic borrowed from plyr_gm_def.py but with partial team matching."""
    
    # Generate name variations
    suffixes = ["II", "III", "IV", "Jr.", "Sr."]
    name_variations = [player_name] + [f"{player_name} {suffix}" for suffix in suffixes]
    placeholders = ', '.join(['%s'] * len(name_variations))
    
    if team_partial_name and team_partial_name.strip():
        # Try exact team matching first
        try:
            target_team_id = get_team_id_by_partial_name(db, team_partial_name)
            
            # Search with exact team match
            query = f"""
            SELECT p.plyr_id, p.plyr_name, 'plyr' AS source, p.team_id, t.abrv, p.plyr_pos, p.plyr_age
            FROM plyr p
            JOIN nfl_team t ON p.team_id = t.team_id
            WHERE p.plyr_name IN ({placeholders}) AND p.team_id = %s AND p.season_id = %s
            UNION
            SELECT mtp.plyr_id, mtp.plyr_name, 'multi_tm_plyr' AS source, 
                   COALESCE(mtp.former_tm_id, mtp.first_tm_id) AS team_id, 
                   COALESCE(t1.abrv, t2.abrv) AS abrv,
                   mtp.plyr_pos, mtp.plyr_age
            FROM multi_tm_plyr mtp
            LEFT JOIN nfl_team t1 ON mtp.former_tm_id = t1.team_id
            LEFT JOIN nfl_team t2 ON mtp.first_tm_id = t2.team_id
            WHERE mtp.plyr_name IN ({placeholders}) AND (mtp.former_tm_id = %s OR mtp.first_tm_id = %s) AND mtp.season_id = %s
            """
            params = name_variations + [target_team_id, season_id] + name_variations + [target_team_id, target_team_id, season_id]
            
            results = db.fetch_all(query, params)
            
            if len(results) == 1:
                return results[0][0]
            elif len(results) > 1:
                # Multiple exact matches - this shouldn't happen but use first
                print(f"[WARNING] Multiple exact team matches found for {player_name} ({team_partial_name}). Using first match.")
                return results[0][0]
            elif len(results) == 0:
                print(f"[WARNING] No exact team match for {player_name} ({team_partial_name}), trying broader search")
                
        except ValueError as e:
            print(f"[WARNING] Could not find team for '{team_partial_name}': {e}")
    
    # If exact team matching failed, try broader search and filter results
    query = f"""
    SELECT p.plyr_id, p.plyr_name, 'plyr' AS source, p.team_id, t.abrv, t.team_name, p.plyr_pos, p.plyr_age
    FROM plyr p
    JOIN nfl_team t ON p.team_id = t.team_id
    WHERE p.plyr_name IN ({placeholders}) AND p.season_id = %s
    UNION
    SELECT mtp.plyr_id, mtp.plyr_name, 'multi_tm_plyr' AS source, 
           COALESCE(mtp.former_tm_id, mtp.first_tm_id) AS team_id, 
           COALESCE(t1.abrv, t2.abrv) AS abrv,
           COALESCE(t1.team_name, t2.team_name) AS team_name,
           mtp.plyr_pos, mtp.plyr_age
    FROM multi_tm_plyr mtp
    LEFT JOIN nfl_team t1 ON mtp.former_tm_id = t1.team_id
    LEFT JOIN nfl_team t2 ON mtp.first_tm_id = t2.team_id
    WHERE mtp.plyr_name IN ({placeholders}) AND mtp.season_id = %s
    """
    params = name_variations + [season_id] + name_variations + [season_id]
    results = db.fetch_all(query, params)
    
    if len(results) == 0:
        raise ValueError(f"No player found for {player_name} ({team_partial_name}) in season {season_id}")
    elif len(results) == 1:
        return results[0][0]
    else:
        # Multiple matches - try to filter by team partial name
        if team_partial_name and team_partial_name.strip():
            team_filtered_results = []
            for result in results:
                team_name = result[5]  # team_name column
                abrv = result[4]       # abrv column
                
                # Check if partial team name matches any part of the full team name or abbreviation
                if (team_partial_name.lower() in team_name.lower() or 
                    team_partial_name.lower() == abrv.lower()):
                    team_filtered_results.append(result)
            
            if len(team_filtered_results) == 1:
                print(f"[OK] Found unique match for {player_name} with team filter ({team_partial_name})")
                return team_filtered_results[0][0]
            elif len(team_filtered_results) > 1:
                print(f"[WARNING] Multiple team-filtered matches for {player_name} ({team_partial_name}). Teams: {[r[5] for r in team_filtered_results]}. Using first match.")
                return team_filtered_results[0][0]
            else:
                print(f"[WARNING] No team-filtered matches for {player_name} ({team_partial_name}). Available teams: {[r[5] for r in results]}. Using first match.")
                return results[0][0]
        else:
            print(f"[WARNING] Multiple matches found for {player_name} (no team filter). Using first match.")
            return results[0][0]


# NOTE: This function is no longer used in the refactored version since context is embedded in CSV files
# Kept for potential future use or fallback scenarios
def parse_filename_for_teams_and_type(filename: str) -> tuple:
    """Extract team names and home/away type from filename."""
    # Example: Kansas_City_Chiefs_Baltimore_Ravens_wk1.0_2024_gm_home_snap_counts_20250817_174741.csv
    parts = filename.split('_')
    
    # Find the team names (they come before the week indicator)
    team_parts = []
    for i, part in enumerate(parts):
        if part.startswith('wk'):
            break
        team_parts.append(part)
    
    # For team name parsing, we need to be smarter about splitting
    # Look for common team name patterns and known team names
    team_name_full = '_'.join(team_parts)
    
    # Known team city/name combinations that should be kept together
    team_patterns = [
        'Kansas_City_Chiefs', 'New_York_Giants', 'New_York_Jets', 'New_England_Patriots',
        'Green_Bay_Packers', 'Tampa_Bay_Buccaneers', 'Las_Vegas_Raiders', 'Los_Angeles_Chargers',
        'Los_Angeles_Rams', 'San_Francisco_49ers', 'New_Orleans_Saints'
    ]
    
    # Try to match known patterns first
    team1_name = None
    team2_name = None
    
    for pattern in team_patterns:
        if team_name_full.startswith(pattern + '_'):
            team1_name = pattern
            remaining = team_name_full[len(pattern) + 1:]
            # Check if remaining part is also a known pattern
            for pattern2 in team_patterns:
                if remaining == pattern2:
                    team2_name = pattern2
                    break
            if not team2_name:
                # If not a known pattern, take the remaining as team2
                team2_name = remaining
            break
    
    # If no patterns matched, fall back to manual parsing based on common structures
    if not team1_name or not team2_name:
        # Try common 2-word + 1-word pattern (e.g., Kansas_City_Chiefs_Ravens)
        if len(team_parts) == 4:
            team1_name = '_'.join(team_parts[:3])
            team2_name = team_parts[3]
        # Try 2-word + 2-word pattern (e.g., Kansas_City_Baltimore_Ravens)
        elif len(team_parts) == 4:
            team1_name = '_'.join(team_parts[:2])
            team2_name = '_'.join(team_parts[2:])
        # Try 3-word + 1-word pattern or 1-word + 3-word pattern
        elif len(team_parts) == 4:
            # This case is already handled above
            team1_name = '_'.join(team_parts[:2])
            team2_name = '_'.join(team_parts[2:])
        # For other cases, split in half
        else:
            mid_point = len(team_parts) // 2
            team1_name = '_'.join(team_parts[:mid_point])
            team2_name = '_'.join(team_parts[mid_point:])
    
    # Convert underscores to spaces for database matching
    team1_name = team1_name.replace('_', ' ')
    team2_name = team2_name.replace('_', ' ')
    
    # Determine if this is home or away
    is_home = 'home' in filename
    is_away = 'away' in filename
    
    return team1_name, team2_name, is_home, is_away


def process_csv_file(db: DatabaseConnector, file_path: str, season_id: int, interactive: bool = False) -> pd.DataFrame:
    """Process a single CSV file and return processed DataFrame."""
    
    print(f"Processing file: {os.path.basename(file_path)}")
    
    # Read CSV with single-level headers
    df = pd.read_csv(file_path)
    
    # Verify required columns exist
    required_columns = ['Player', 'team', 'week', 'year']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Get context data from the CSV (embedded in the data)
    if df.empty:
        raise ValueError("CSV file is empty")
    
    # Use embedded data from first row (all rows should have same context)
    week_num = df.iloc[0]['week'] 
    year = df.iloc[0]['year']
    
    print(f"Embedded context - Year: {year}, Week: {week_num}")
    
    # Convert week to integer
    week_num = int(float(week_num))
    week_id = get_week_id(db, season_id, week_num)
    
    # Get unique teams from both home and away files for this game
    # Since each file contains only one team, we need to extract both teams from filename
    filename = os.path.basename(file_path)
    team_names = extract_teams_from_filename(filename)
    
    if not team_names:
        raise ValueError(f"Could not extract team names from filename: {filename}")
    
    team1_name, team2_name = team_names
    
    # Convert team names to abbreviations for game lookup
    team1_abrv = get_team_abbreviation(db, team1_name) 
    team2_abrv = get_team_abbreviation(db, team2_name)
    
    # Get game_id using database lookup
    game_id = get_game_id(db, season_id, week_id, team1_abrv, team2_abrv)
    
    print(f"Database lookup - Game ID: {game_id} for {team1_abrv} vs {team2_abrv}")
    
    # Process each row
    processed_rows = []
    for index, row in df.iterrows():
        try:
            # Get player name and team from row
            player_name = row.get('Player', '')
            team_partial_name = row.get('team', '')
            
            if not player_name or player_name.strip() == '':
                continue
            
            if not team_partial_name or team_partial_name.strip() == '':
                print(f"[WARNING] No team information for player {player_name}, skipping")
                continue
            
            # Convert team nickname to abbreviation for player lookup
            try:
                team_abbrev = convert_team_nickname_to_abbreviation(db, team_partial_name)
            except ValueError as e:
                print(f"[WARNING] Could not convert team nickname '{team_partial_name}' to abbreviation: {e}")
                # Fall back to using the original team name
                team_abbrev = team_partial_name
                
            # Get player_id using centralized function with team abbreviation
            plyr_id = get_player_id(db, player_name, team_abbrev, season_id, interactive=interactive)

            # Skip this player if user chose to skip in interactive mode
            if interactive and plyr_id == 0:
                print(f"[INFO] Skipping player {player_name} - user selection")
                continue
            
            # Get team_id using partial team name matching
            team_id = get_team_id_by_partial_name(db, team_partial_name)
            
            # Create processed row using database-derived data
            processed_row = {
                'plyr_id': plyr_id,
                'week_id': week_id,
                'game_id': game_id,  # Use database-derived game_id
                'season_id': season_id,
                'team_id': team_id
            }
            
            # Map snap count columns (using actual CSV column names)
            snap_mapping = {
                'plyr_gm_off_snap_ct': 'Off. Num',
                'plyr_gm_off_snap_ct_pct': 'Off. Pct', 
                'plyr_gm_def_snap_ct': 'Def. Num',
                'plyr_gm_def_snap_ct_pct': 'Def. Pct',
                'plyr_gm_st_snap_ct': 'ST Num',
                'plyr_gm_st_snap_ct_pct': 'ST Pct'
            }
            
            # Add snap count stats
            for db_col, csv_col in snap_mapping.items():
                if csv_col in df.columns:
                    value = row.get(csv_col)
                    if pd.notna(value) and value != '' and str(value).strip() != '':
                        try:
                            if 'pct' in db_col:
                                # Handle percentage values - maintain 3 decimal places
                                if isinstance(value, str) and '%' in value:
                                    processed_row[db_col] = round(float(value.replace('%', '')) / 100.0, 3)
                                else:
                                    processed_row[db_col] = round(float(value), 3)
                            else:
                                # Handle count values
                                processed_row[db_col] = int(float(value))
                        except (ValueError, TypeError):
                            processed_row[db_col] = None
                    else:
                        processed_row[db_col] = None
            
            processed_rows.append(processed_row)
            
        except Exception as e:
            print(f"[WARNING] Error processing row {index} for player {row.get('Player', 'Unknown')}: {e}")
            continue
    
    if processed_rows:
        processed_df = pd.DataFrame(processed_rows)
        processed_df = handle_null_values(processed_df)
        
        # Ensure proper data types
        int_columns = ['plyr_id', 'week_id', 'game_id', 'season_id', 'team_id', 'plyr_gm_off_snap_ct', 
                      'plyr_gm_def_snap_ct', 'plyr_gm_st_snap_ct']
        float_columns = ['plyr_gm_off_snap_ct_pct', 'plyr_gm_def_snap_ct_pct', 'plyr_gm_st_snap_ct_pct']
        
        for col in int_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        for col in float_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        return processed_df
    else:
        return pd.DataFrame()


def get_csv_files() -> list:
    """Get list of CSV files to process based on WEEK_START and WEEK_END."""
    base_dir = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\games"
    csv_files = []
    
    for week in range(WEEK_START, WEEK_END + 1):
        week_dir = os.path.join(base_dir, f"week_{week}.0", "clean")
        if os.path.exists(week_dir):
            pattern = os.path.join(week_dir, "cleaned_*_snap_counts_*.csv")
            week_files = glob.glob(pattern)
            
            if week_files:
                csv_files.extend(week_files)
                print(f"[OK] Found {len(week_files)} snap count files for week {week}")
            else:
                print(f"[WARNING] No snap count files found in {week_dir}")
        else:
            print(f"[WARNING] Week directory not found: {week_dir}")
    
    return sorted(csv_files)


def main():
    """Main function to process all CSV files."""
    
    print(f"Starting Player Game Snap Count Data Import for {YEAR}")
    print(f"Processing weeks {WEEK_START} to {WEEK_END}")
    print("[INFO] Interactive mode enabled - you will be prompted for player selection when multiple/no matches are found")
    
    # Initialize database connection
    db = DatabaseConnector()
    if not db.connect():
        print("[FAIL] Failed to connect to database")
        return
    
    try:
        # Create table if it doesn't exist
        if not create_plyr_gm_snap_ct_table(db):
            return
        
        # Get season_id
        season_id = get_season_id(db, YEAR)
        print(f"Season ID for {YEAR}: {season_id}")
        
        # Get CSV files to process
        csv_files = get_csv_files()
        if not csv_files:
            print("[WARNING] No CSV files found to process")
            return
        
        print(f"Found {len(csv_files)} snap count files to process")
        
        total_processed = 0
        total_inserted = 0
        files_processed = 0
        
        # Process each file
        for file_path in csv_files:
            try:
                print(f"\nProcessing file: {os.path.basename(file_path)}")
                
                # Process file with season_id parameter and interactive mode
                df = process_csv_file(db, file_path, season_id, interactive=True)
                
                if not df.empty:
                    success = batch_upsert_data(db, 'plyr_gm_snap_ct', df)
                    if success:
                        rows_in_file = len(df)
                        total_processed += rows_in_file
                        total_inserted += rows_in_file
                        files_processed += 1
                        print(f"[OK] Processed {rows_in_file} player snap count records")
                    else:
                        print(f"[FAIL] Failed to insert data from {os.path.basename(file_path)}")
                else:
                    print(f"[WARNING] No data to process in {os.path.basename(file_path)}")
                    
            except Exception as e:
                print(f"[ERROR] Error processing {os.path.basename(file_path)}: {e}")
                continue
        
        print(f"\nImport Summary:")
        print(f"   Total files found: {len(csv_files)}")
        print(f"   Total files processed successfully: {files_processed}")
        print(f"   Total player records processed: {total_processed}")
        print(f"   Total records inserted/updated: {total_inserted}")
        
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
    finally:
        db.disconnect()


if __name__ == "__main__":
    main()