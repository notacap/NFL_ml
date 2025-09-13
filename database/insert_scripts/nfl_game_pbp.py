#!/usr/bin/env python3

import os
import sys
import pandas as pd
import glob
import re
from pathlib import Path

# Add the parent directory to the Python path to import db_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_utils import (
    DatabaseConnector, YEAR, WEEK_START, WEEK_END, 
    batch_upsert_data, handle_null_values, 
    get_season_id, get_week_id, get_game_id, get_player_id, get_team_id,
    create_table_if_not_exists, convert_team_nickname_to_abbreviation
)


def create_nfl_game_pbp_table(db: DatabaseConnector) -> bool:
    """Create the nfl_game_pbp table if it doesn't exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS nfl_game_pbp (
        game_pbp_id INT PRIMARY KEY AUTO_INCREMENT,
        game_id INT,
        week_id INT,
        season_id INT,
        team_id INT,
        gm_pbp_qtr_id INT,
        gm_pbp_time VARCHAR(10),
        gm_pbp_dwn INT,
        gm_pbn_yds_to_go INT,
        gm_pbp_location_side VARCHAR(3),
        gm_pbp_location_yd_line INT,
        gm_pbp_home_tm_score INT,
        gm_pbp_away_tm_score INT,
        gm_pbp_detail VARCHAR(500),
        gm_pbp_ebp FLOAT,
        gm_pbp_epa FLOAT,
        FOREIGN KEY (gm_pbp_qtr_id) REFERENCES nfl_gm_quarter(nfl_quarter_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (game_id) REFERENCES nfl_game(game_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        FOREIGN KEY (team_id) REFERENCES nfl_team(team_id),
        UNIQUE KEY unique_play (game_id, gm_pbp_qtr_id, gm_pbp_time)
    );
    """
    
    return create_table_if_not_exists(db, 'nfl_game_pbp', create_table_sql)


def get_quarter_id(db: DatabaseConnector, quarter_value) -> int:
    """Get nfl_quarter_id from nfl_gm_quarter table based on quarter value."""
    if pd.isna(quarter_value) or quarter_value == '':
        return None
    
    # Convert quarter to integer (e.g., 1.0 -> 1)
    try:
        quarter_num = int(float(quarter_value))
    except (ValueError, TypeError):
        return None
    
    # Map quarter number to quarter_id
    # Assuming quarter_id matches quarter number (1=Q1, 2=Q2, 3=Q3, 4=Q4, 5=OT)
    if quarter_num in [1, 2, 3, 4]:
        return quarter_num
    elif quarter_num > 4:  # Overtime
        return 5
    else:
        return None


def parse_location(location_str: str) -> tuple:
    """Parse location string like 'PIT 35' into team_side and yard_line."""
    if pd.isna(location_str) or location_str == '' or str(location_str).strip() == '':
        return None, None
    
    # Parse format: "TEAM YD" (e.g., "PIT 35")
    match = re.match(r'([A-Z]{2,3})\s+(\d+)', str(location_str).strip())
    if match:
        team_side = match.group(1)
        yard_line = int(match.group(2))
        return team_side, yard_line
    
    return None, None


def extract_player_names_from_detail(detail: str) -> list:
    """Extract all potential player names from play detail string."""
    if pd.isna(detail) or detail == '':
        return []
    
    # Remove content in parentheses (tackle info) as per prompt requirements
    detail_clean = re.sub(r'\([^)]*\)', '', str(detail))
    detail_clean = detail_clean.strip()
    
    # Common patterns for player names in play descriptions
    player_names = []
    
    # Pattern 1: First name Last name at the start (most common)
    # Examples: "Bijan Robinson left tackle", "Kirk Cousins pass complete"
    first_name_pattern = r'^([A-Z][a-z]+(?:\.[A-Z]\.?)?\s+[A-Z][a-z]+(?:\s+[IVX]+|Jr\.?|Sr\.?)?)'
    match = re.search(first_name_pattern, detail_clean)
    if match:
        player_names.append(match.group(1).strip())
    
    # Pattern 2: Initial.LastName format
    # Examples: "C.Santos kicks", "D.Smith rush"
    initial_pattern = r'^([A-Z]\.[A-Z][a-z]+(?:\s+[IVX]+|Jr\.?|Sr\.?)?)'
    match = re.search(initial_pattern, detail_clean)
    if match:
        player_names.append(match.group(1).strip())
    
    # Pattern 3: Look for "pass complete/incomplete to PlayerName"
    pass_target_pattern = r'pass (?:complete|incomplete).*?(?:to|intended for)\s+([A-Z][a-z]+(?:\.[A-Z]\.?)?\s+[A-Z][a-z]+(?:\s+[IVX]+|Jr\.?|Sr\.?)?)'
    match = re.search(pass_target_pattern, detail_clean)
    if match:
        player_names.append(match.group(1).strip())
    
    # Remove duplicates while preserving order
    unique_names = []
    for name in player_names:
        if name not in unique_names:
            unique_names.append(name)
    
    return unique_names


def determine_team_id_from_play(db: DatabaseConnector, detail: str, snap_players: dict) -> int:
    """Determine which team ran the play based on player name matching with snap count files."""
    potential_players = extract_player_names_from_detail(detail)
    
    if not potential_players:
        return None
    
    # Try to match players against snap count data
    for player_name in potential_players:
        for team_abrv, players in snap_players.items():
            # Check for exact match first
            if player_name in players:
                return get_team_id(db, team_abrv)
            
            # Check for first name + last name variations (more reliable than partial matching)
            if ' ' in player_name:
                first_name, last_name = player_name.split()[0], player_name.split()[-1]
                for snap_player in players:
                    snap_parts = snap_player.split()
                    if len(snap_parts) >= 2:
                        snap_first, snap_last = snap_parts[0], snap_parts[-1]
                        # Match if both first and last names match
                        if (first_name.lower() == snap_first.lower() and 
                            last_name.lower() == snap_last.lower()):
                            return get_team_id(db, team_abrv)
            
            # Check for partial matches (last name only) - more restrictive
            player_last = player_name.split()[-1] if ' ' in player_name else player_name
            if len(player_last) > 3:  # Only match if last name is longer than 3 characters
                for snap_player in players:
                    snap_last = snap_player.split()[-1] if ' ' in snap_player else snap_player
                    if player_last.lower() == snap_last.lower():
                        return get_team_id(db, team_abrv)
    return None


def load_snap_count_players(db: DatabaseConnector, week: int, away_team_abrv: str, home_team_abrv: str) -> dict:
    """Load player names from snap count files for the given game."""
    base_dir = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\games"
    week_dir = os.path.join(base_dir, f"week_{week}.0", "clean")
    
    if not os.path.exists(week_dir):
        return {}
    
    # Find snap count files for this game by matching team names in filename
    # Pattern: cleaned_Team1_Team2_wk1.0_2024_gm_home_snap_counts_*.csv
    # Pattern: cleaned_Team1_Team2_wk1.0_2024_gm_away_snap_counts_*.csv
    
    home_pattern = os.path.join(week_dir, "cleaned_*_gm_home_snap_counts_*.csv")
    away_pattern = os.path.join(week_dir, "cleaned_*_gm_away_snap_counts_*.csv")
    
    home_files = glob.glob(home_pattern)
    away_files = glob.glob(away_pattern)
    
    snap_players = {}
    
    # Load home team snap counts
    for file in home_files:
        filename = os.path.basename(file)
        # Check if this snap count file matches our game by team names
        # Convert team abbreviations back to full names for matching
        try:
            home_team_name = db.fetch_all("SELECT team_name FROM nfl_team WHERE abrv = %s OR alt_abrv = %s", (home_team_abrv, home_team_abrv))[0][0]
            away_team_name = db.fetch_all("SELECT team_name FROM nfl_team WHERE abrv = %s OR alt_abrv = %s", (away_team_abrv, away_team_abrv))[0][0]
            
            # Convert team names to underscores for filename matching
            home_name_parts = home_team_name.replace(' ', '_')
            away_name_parts = away_team_name.replace(' ', '_')
            
            # Check if both team names are in the filename
            if home_name_parts in filename and away_name_parts in filename:
                df = pd.read_csv(file)
                if 'Player' in df.columns and 'team' in df.columns:
                    for _, row in df.iterrows():
                        player_name = row['Player']
                        team_nickname = row['team']
                        
                        if pd.notna(player_name) and pd.notna(team_nickname):
                            # Convert team nickname to abbreviation
                            try:
                                team_abrv = convert_team_nickname_to_abbreviation(db, team_nickname)
                                if team_abrv not in snap_players:
                                    snap_players[team_abrv] = set()
                                snap_players[team_abrv].add(player_name.strip())
                            except ValueError as e:
                                print(f"[WARNING] Could not convert team nickname '{team_nickname}' to abbreviation: {e}")
                                continue
        except Exception as e:
            print(f"[WARNING] Error processing home snap count file {os.path.basename(file)}: {e}")
    
    # Load away team snap counts
    for file in away_files:
        filename = os.path.basename(file)
        try:
            home_team_name = db.fetch_all("SELECT team_name FROM nfl_team WHERE abrv = %s OR alt_abrv = %s", (home_team_abrv, home_team_abrv))[0][0]
            away_team_name = db.fetch_all("SELECT team_name FROM nfl_team WHERE abrv = %s OR alt_abrv = %s", (away_team_abrv, away_team_abrv))[0][0]
            
            # Convert team names to underscores for filename matching
            home_name_parts = home_team_name.replace(' ', '_')
            away_name_parts = away_team_name.replace(' ', '_')
            
            # Check if both team names are in the filename
            if home_name_parts in filename and away_name_parts in filename:
                df = pd.read_csv(file)
                if 'Player' in df.columns and 'team' in df.columns:
                    for _, row in df.iterrows():
                        player_name = row['Player']
                        team_nickname = row['team']
                        
                        if pd.notna(player_name) and pd.notna(team_nickname):
                            # Convert team nickname to abbreviation
                            try:
                                team_abrv = convert_team_nickname_to_abbreviation(db, team_nickname)
                                if team_abrv not in snap_players:
                                    snap_players[team_abrv] = set()
                                snap_players[team_abrv].add(player_name.strip())
                            except ValueError as e:
                                print(f"[WARNING] Could not convert team nickname '{team_nickname}' to abbreviation: {e}")
                                continue
        except Exception as e:
            print(f"[WARNING] Error processing away snap count file {os.path.basename(file)}: {e}")
    
    print(f"[INFO] Loaded snap count players: {', '.join(f'{team}({len(players)})' for team, players in snap_players.items())}")
    
    return snap_players


def determine_home_away_teams(db: DatabaseConnector, game_id: int) -> tuple:
    """Determine which team is home and which is away for a given game."""
    query = """
    SELECT home_team_id, away_team_id
    FROM nfl_game
    WHERE game_id = %s
    """
    result = db.fetch_all(query, (game_id,))
    if result:
        return result[0][0], result[0][1]  # home_team_id, away_team_id
    return None, None


def process_pbp_file(db: DatabaseConnector, file_path: str, season_id: int) -> pd.DataFrame:
    """Process a single play-by-play CSV file and return processed DataFrame."""
    
    print(f"Processing file: {os.path.basename(file_path)}")
    
    # Read CSV
    df = pd.read_csv(file_path)
    
    # Get column headers to identify team abbreviations at indices 5 and 6
    columns = df.columns.tolist()
    if len(columns) < 7:
        raise ValueError(f"Expected at least 7 columns, found {len(columns)}")
    
    away_team_col = columns[5]  # Away team abbreviation
    home_team_col = columns[6]  # Home team abbreviation
    
    print(f"  Teams identified: {away_team_col} (away) vs {home_team_col} (home)")
    
    # Get week number from DataFrame
    if 'week' not in df.columns:
        raise ValueError("Week column not found in CSV")
    
    week_num = df['week'].iloc[0]
    if pd.isna(week_num):
        raise ValueError("Week number is null")
    
    week_num = int(float(week_num))
    week_id = get_week_id(db, season_id, week_num)
    
    # Get game_id using the team abbreviations
    game_id = get_game_id(db, season_id, week_id, away_team_col, home_team_col)
    
    # Determine which team_id is home and which is away
    home_team_id, away_team_id = determine_home_away_teams(db, game_id)
    
    # Extract team names from filename for snap count lookup
    filename = os.path.basename(file_path)
    # Pattern: cleaned_Team1_Team2_wk1.0_2024_gm_play_by_play_...
    match = re.match(r'cleaned_(.+?)_wk\d+', filename)
    if match:
        teams_str = match.group(1)
        # Split team names (handle multi-word team names)
        team_parts = teams_str.split('_')
        # This is complex due to multi-word team names, for now use the abbreviations
        game_teams = (away_team_col, home_team_col)
    else:
        game_teams = (away_team_col, home_team_col)
    
    # Load snap count players for this game
    snap_players = load_snap_count_players(db, week_num, away_team_col, home_team_col)
    
    processed_rows = []
    
    for index, row in df.iterrows():
        try:
            # Skip rows with empty Detail
            detail = row.get('Detail', '')
            if pd.isna(detail) or detail == '' or str(detail).strip() == '':
                continue
            
            # Skip non-play rows (like coin toss description)
            if 'coin toss' in str(detail).lower() or 'timeout' in str(detail).lower():
                continue
            
            # Skip rows where Down, ToGo, Location, EPB, or EPA are null
            down_val = row.get('Down')
            togo_val = row.get('ToGo') 
            location_val = row.get('Location')
            epb_val = row.get('EPB')
            epa_val = row.get('EPA')
            
            if (pd.isna(down_val) or pd.isna(togo_val) or pd.isna(location_val) or 
                pd.isna(epb_val) or pd.isna(epa_val) or
                location_val == '' or str(location_val).strip() == ''):
                continue
            
            # Parse location
            location_side, location_yd_line = parse_location(location_val)
            
            # Get quarter_id
            quarter_id = get_quarter_id(db, row.get('Quarter'))
            
            # Get time value
            time_val = row.get('Time', '')
            if pd.isna(time_val) or time_val == '':
                time_val = None
            else:
                time_val = str(time_val)[:10]  # Truncate to VARCHAR(10) limit
            
            # Determine team_id based on play detail and snap counts
            team_id = determine_team_id_from_play(db, detail, snap_players)
            
            # Get scores - use column names identified earlier
            away_score = row.get(away_team_col)
            home_score = row.get(home_team_col)
            
            # Determine which score goes where based on home/away teams
            if home_team_id and away_team_id:
                gm_pbp_home_tm_score = home_score if pd.notna(home_score) else None
                gm_pbp_away_tm_score = away_score if pd.notna(away_score) else None
            else:
                gm_pbp_home_tm_score = None
                gm_pbp_away_tm_score = None
            
            # Create processed row
            processed_row = {
                'game_id': game_id,
                'week_id': week_id,
                'season_id': season_id,
                'team_id': team_id,
                'gm_pbp_qtr_id': quarter_id,
                'gm_pbp_time': time_val,
                'gm_pbp_dwn': int(float(down_val)),
                'gm_pbn_yds_to_go': int(float(togo_val)),
                'gm_pbp_location_side': location_side,
                'gm_pbp_location_yd_line': location_yd_line,
                'gm_pbp_home_tm_score': int(float(gm_pbp_home_tm_score)) if pd.notna(gm_pbp_home_tm_score) else None,
                'gm_pbp_away_tm_score': int(float(gm_pbp_away_tm_score)) if pd.notna(gm_pbp_away_tm_score) else None,
                'gm_pbp_detail': str(detail)[:500] if pd.notna(detail) else None,  # Truncate to 500 chars
                'gm_pbp_ebp': float(epb_val),
                'gm_pbp_epa': float(epa_val)
            }
            
            processed_rows.append(processed_row)
            
        except Exception as e:
            print(f"[WARNING] Error processing row {index}: {e}")
            continue
    
    if processed_rows:
        processed_df = pd.DataFrame(processed_rows)
        processed_df = handle_null_values(processed_df)
        
        # Ensure proper data types for foreign key columns
        fk_columns = ['game_id', 'week_id', 'season_id', 'team_id', 'gm_pbp_qtr_id']
        for col in fk_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                # Convert to nullable integer type to handle NaN values
                processed_df[col] = processed_df[col].astype('Int64')
        
        # Handle integer columns
        int_columns = ['gm_pbp_dwn', 'gm_pbn_yds_to_go', 'gm_pbp_location_yd_line', 
                      'gm_pbp_home_tm_score', 'gm_pbp_away_tm_score']
        for col in int_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                processed_df[col] = processed_df[col].astype('Int64')
        
        # Handle float columns
        float_columns = ['gm_pbp_ebp', 'gm_pbp_epa']
        for col in float_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        return processed_df
    else:
        return pd.DataFrame()


def get_pbp_csv_files() -> list:
    """Get list of play-by-play CSV files to process based on WEEK_START and WEEK_END."""
    base_dir = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\games"
    csv_files = []
    
    for week in range(WEEK_START, WEEK_END + 1):
        week_dir = os.path.join(base_dir, f"week_{week}.0")
        clean_dir = os.path.join(week_dir, "clean")
        
        if os.path.exists(clean_dir):
            # Get play-by-play CSV files
            pattern = os.path.join(clean_dir, "cleaned_*_gm_play_by_play_*.csv")
            week_files = glob.glob(pattern)
            
            if week_files:
                csv_files.extend([(week, f) for f in week_files])
                print(f"[OK] Found {len(week_files)} play-by-play files for week {week}")
            else:
                print(f"[WARNING] No play-by-play files found in {clean_dir}")
        else:
            print(f"[WARNING] Clean directory not found: {clean_dir}")
    
    return sorted(csv_files)


def main():
    """Main function to process all play-by-play CSV files."""
    
    print(f"Starting NFL Game Play-by-Play Data Import for {YEAR}")
    print(f"Processing weeks {WEEK_START} to {WEEK_END}")
    
    # Initialize database connection
    db = DatabaseConnector()
    if not db.connect():
        print("[FAIL] Failed to connect to database")
        return
    
    try:
        # Create table if it doesn't exist
        if not create_nfl_game_pbp_table(db):
            return
        
        # Get season_id
        season_id = get_season_id(db, YEAR)
        print(f"Season ID for {YEAR}: {season_id}")
        
        # Get CSV files to process
        csv_file_pairs = get_pbp_csv_files()
        if not csv_file_pairs:
            print("[WARNING] No play-by-play CSV files found to process")
            return
        
        print(f"Found {len(csv_file_pairs)} play-by-play files to process")
        
        total_processed = 0
        total_inserted = 0
        
        # Process each file
        for week, file_path in csv_file_pairs:
            try:
                processed_df = process_pbp_file(db, file_path, season_id)
                
                if not processed_df.empty:
                    success = batch_upsert_data(db, 'nfl_game_pbp', processed_df)
                    if success:
                        rows_in_file = len(processed_df)
                        total_processed += rows_in_file
                        total_inserted += rows_in_file
                        print(f"[OK] Week {week}: Processed {rows_in_file} play-by-play records")
                    else:
                        print(f"[FAIL] Week {week}: Failed to insert data")
                else:
                    print(f"[WARNING] Week {week}: No data to process")
                    
            except Exception as e:
                print(f"[ERROR] Error processing week {week} file: {e}")
                print(f"   File: {os.path.basename(file_path)}")
                continue
        
        print(f"\nPlay-by-Play Import Summary:")
        print(f"   Total files processed: {len(csv_file_pairs)}")
        print(f"   Total play records processed: {total_processed}")
        print(f"   Total records inserted/updated: {total_inserted}")
        
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
    finally:
        db.disconnect()


if __name__ == "__main__":
    main()