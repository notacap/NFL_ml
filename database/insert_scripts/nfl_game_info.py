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
    get_season_id, get_week_id, get_game_id, get_team_id,
    create_table_if_not_exists
)


def create_nfl_game_info_table(db: DatabaseConnector) -> bool:
    """Create the nfl_game_info table if it doesn't exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS nfl_game_info (
        game_info_id INT PRIMARY KEY AUTO_INCREMENT,
        game_id INT,
        week_id INT,
        season_id INT,
        won_toss_tm_id INT,
        is_deferred TINYINT,
        stadium_roof VARCHAR(20),
        duration TIME,
        attendance INT,
        vegas_line_tm_id INT,
        vegas_line FLOAT,
        over_under_line FLOAT,
        over_under_result VARCHAR(10),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (game_id) REFERENCES nfl_game(game_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        FOREIGN KEY (won_toss_tm_id) REFERENCES nfl_team(team_id),
        FOREIGN KEY (vegas_line_tm_id) REFERENCES nfl_team(team_id),
        UNIQUE KEY (game_id) 
    );
    """
    
    return create_table_if_not_exists(db, 'nfl_game_info', create_table_sql)


def extract_teams_from_filename(filename: str) -> tuple:
    """Extract team names from filename, handling 2 and 3 word team names."""
    # Remove cleaned_ prefix and file extension
    basename = os.path.basename(filename)
    if basename.startswith('cleaned_'):
        basename = basename[8:]  # Remove 'cleaned_' prefix
    
    # Remove everything from _wk onwards
    if '_wk' in basename:
        team_part = basename.split('_wk')[0]
    else:
        raise ValueError(f"Could not find week marker in filename: {filename}")
    
    # Split by underscores
    parts = team_part.split('_')
    
    # Find the middle point - we need to identify where first team ends and second begins
    # Look for common city/state names to help identify team boundaries
    cities = ['New', 'Los', 'San', 'Green', 'Kansas', 'Tampa', 'Las']
    
    # Find potential split points
    split_candidates = []
    for i, part in enumerate(parts):
        if part in cities and i > 0:  # Don't split at the very beginning
            split_candidates.append(i)
    
    # If we found city markers, use them to split
    if split_candidates:
        split_point = split_candidates[0]
        team1_parts = parts[:split_point]
        team2_parts = parts[split_point:]
    else:
        # Fallback: assume teams are roughly equal length, favor 2-word teams
        mid_point = len(parts) // 2
        
        # Adjust for common 3-word teams
        if len(parts) == 5:  # Total 5 words = 2 + 3 or 3 + 2
            # Check if first 3 words could be a team
            potential_team1 = ' '.join(parts[:3])
            if any(x in potential_team1 for x in ['Kansas City', 'Green Bay', 'New England', 'New Orleans', 'New York', 'San Francisco', 'Los Angeles', 'Tampa Bay', 'Las Vegas']):
                split_point = 3
            else:
                split_point = 2
        elif len(parts) == 6:  # Total 6 words = 3 + 3
            split_point = 3
        else:
            split_point = mid_point
            
        team1_parts = parts[:split_point]
        team2_parts = parts[split_point:]
    
    team1_name = ' '.join(team1_parts)
    team2_name = ' '.join(team2_parts)
    
    return team1_name, team2_name


def parse_won_toss_value(won_toss_str: str) -> tuple:
    """Parse Won Toss value to extract team name and deferred status."""
    if pd.isna(won_toss_str) or not won_toss_str or str(won_toss_str).strip() == '':
        return None, None
    
    won_toss_str = str(won_toss_str).strip()
    
    # Check for deferred
    is_deferred = 1 if '(deferred)' in won_toss_str else 0
    
    # Remove (deferred) to get clean team name
    team_name = won_toss_str.replace('(deferred)', '').strip()
    
    return team_name, is_deferred


def parse_vegas_line_value(vegas_line_str: str) -> tuple:
    """Parse Vegas Line value to extract team name and line value."""
    if pd.isna(vegas_line_str) or not vegas_line_str or str(vegas_line_str).strip() == '':
        return None, None
    
    vegas_line_str = str(vegas_line_str).strip()
    
    # Find the last space followed by +/- and number
    match = re.search(r'^(.+)\s+([+-]?\d+\.?\d*)$', vegas_line_str)
    if match:
        team_name = match.group(1).strip()
        line_value = float(match.group(2))
        return team_name, line_value
    else:
        raise ValueError(f"Could not parse Vegas Line value: {vegas_line_str}")


def parse_over_under_value(over_under_str: str) -> tuple:
    """Parse Over/Under value to extract line and result."""
    if pd.isna(over_under_str) or not over_under_str or str(over_under_str).strip() == '':
        return None, None
    
    over_under_str = str(over_under_str).strip()
    
    # Extract number and result from format like "46.5 (over)" or "52 (under)"
    match = re.search(r'^(\d+\.?\d*)\s*\((\w+)\)$', over_under_str)
    if match:
        line_value = float(match.group(1))
        result = match.group(2)
        return line_value, result
    else:
        raise ValueError(f"Could not parse Over/Under value: {over_under_str}")


def get_team_id_partial_match(db: DatabaseConnector, team_name: str) -> int:
    """Get team_id using partial match for team nicknames."""
    if not team_name or str(team_name).strip() == '':
        raise ValueError("Team name is empty")
    
    team_name = str(team_name).strip()
    
    # Try exact match first
    try:
        return get_team_id(db, team_name)
    except ValueError:
        pass
    
    # Try partial match in team_name column
    query = "SELECT team_id, team_name FROM nfl_team WHERE team_name LIKE %s"
    result = db.fetch_all(query, (f"%{team_name}%",))
    
    if result:
        if len(result) == 1:
            return result[0][0]
        else:
            print(f"[WARNING] Multiple teams match '{team_name}': {[r[1] for r in result]}")
            return result[0][0]  # Use first match
    
    raise ValueError(f"No team found for name '{team_name}'")


def get_csv_files() -> list:
    """Get list of CSV files to process based on WEEK_START and WEEK_END."""
    base_dir = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\games"
    csv_files = []
    
    for week in range(WEEK_START, WEEK_END + 1):
        week_dir = os.path.join(base_dir, f"week_{week}.0")
        clean_dir = os.path.join(week_dir, "clean")
        
        if os.path.exists(clean_dir):
            # Pattern for game info files
            pattern = os.path.join(clean_dir, "cleaned_*_gm_info_*.csv")
            files = glob.glob(pattern)
            csv_files.extend(files)
        else:
            print(f"[WARNING] Clean directory not found: {clean_dir}")
    
    return sorted(csv_files)


def process_csv_file(db: DatabaseConnector, file_path: str, season_id: int) -> pd.DataFrame:
    """Process a single game info CSV file."""
    print(f"Processing: {os.path.basename(file_path)}")
    
    # Read CSV file
    df = pd.read_csv(file_path, header=0)
    
    if df.empty:
        print(f"[WARNING] Empty CSV file: {file_path}")
        return pd.DataFrame()
    
    # Get week from first row (all rows should have same week)
    if 'week' not in df.columns:
        raise ValueError(f"Week column not found in {file_path}")
    
    week_num = df['week'].iloc[0]
    if pd.isna(week_num):
        raise ValueError(f"Week number is null in {file_path}")
    
    week_num = int(float(week_num))
    week_id = get_week_id(db, season_id, week_num)
    
    # Extract team names from filename
    team1_name, team2_name = extract_teams_from_filename(file_path)
    print(f"  Extracted teams: {team1_name} vs {team2_name}")
    
    # Get team abbreviations for game lookup
    team1_id = get_team_id(db, team1_name)
    team2_id = get_team_id(db, team2_name)
    
    # Get team abbreviations for game_id lookup
    team1_query = "SELECT abrv FROM nfl_team WHERE team_id = %s"
    team2_query = "SELECT abrv FROM nfl_team WHERE team_id = %s"
    team1_abrv = db.fetch_all(team1_query, (team1_id,))[0][0]
    team2_abrv = db.fetch_all(team2_query, (team2_id,))[0][0]
    
    game_id = get_game_id(db, season_id, week_id, team1_abrv, team2_abrv)
    
    # Create data dictionary from the vertical CSV structure
    game_data = {
        'game_id': game_id,
        'week_id': week_id,
        'season_id': season_id
    }
    
    # Process each row to extract field-value pairs
    for _, row in df.iterrows():
        field_name = str(row.iloc[0]).strip() if not pd.isna(row.iloc[0]) else ''
        field_value = row.iloc[1] if not pd.isna(row.iloc[1]) else None
        
        if field_name == 'Won Toss' and field_value:
            team_name, is_deferred = parse_won_toss_value(field_value)
            if team_name:
                try:
                    won_toss_tm_id = get_team_id_partial_match(db, team_name)
                    game_data['won_toss_tm_id'] = won_toss_tm_id
                    game_data['is_deferred'] = is_deferred
                except ValueError as e:
                    print(f"[WARNING] Could not find team for Won Toss '{team_name}': {e}")
                    game_data['won_toss_tm_id'] = None
                    game_data['is_deferred'] = None
        
        elif field_name == 'Roof' and field_value:
            # Map roof types to shorter database values
            roof_mapping = {
                'retractable roof (open)': 'retractable_open',
                'retractable roof (closed)': 'retractable_closed', 
                'dome': 'dome',
                'outdoors': 'outdoors'
            }
            roof_value = str(field_value).strip()
            game_data['stadium_roof'] = roof_mapping.get(roof_value, roof_value[:20])  # Truncate to 20 chars
        
        elif field_name == 'Duration' and field_value:
            # Convert duration from "H:MM" to TIME format
            duration_str = str(field_value).strip()
            if ':' in duration_str:
                try:
                    hours, minutes = duration_str.split(':')
                    # Format as HH:MM:SS for MySQL TIME
                    game_data['duration'] = f"{int(hours):02d}:{int(minutes):02d}:00"
                except ValueError:
                    print(f"[WARNING] Could not parse duration '{duration_str}'")
                    game_data['duration'] = None
        
        elif field_name == 'Attendance' and field_value:
            try:
                game_data['attendance'] = int(float(str(field_value).replace(',', '')))
            except ValueError:
                print(f"[WARNING] Could not parse attendance '{field_value}'")
                game_data['attendance'] = None
        
        elif field_name == 'Vegas Line' and field_value:
            try:
                team_name, line_value = parse_vegas_line_value(field_value)
                if team_name and line_value is not None:
                    vegas_line_tm_id = get_team_id(db, team_name)
                    game_data['vegas_line_tm_id'] = vegas_line_tm_id
                    game_data['vegas_line'] = line_value
            except (ValueError, Exception) as e:
                print(f"[WARNING] Could not parse Vegas Line '{field_value}': {e}")
                game_data['vegas_line_tm_id'] = None
                game_data['vegas_line'] = None
        
        elif field_name == 'Over/Under' and field_value:
            try:
                line_value, result = parse_over_under_value(field_value)
                if line_value is not None and result:
                    game_data['over_under_line'] = line_value
                    game_data['over_under_result'] = result
            except (ValueError, Exception) as e:
                print(f"[WARNING] Could not parse Over/Under '{field_value}': {e}")
                game_data['over_under_line'] = None
                game_data['over_under_result'] = None
        
        # Skip Weather field as per requirements
    
    # Create DataFrame with single row
    processed_df = pd.DataFrame([game_data])
    processed_df = handle_null_values(processed_df)
    
    return processed_df


def main():
    """Main function to process all CSV files."""
    
    print(f"Starting NFL Game Info Data Import for {YEAR}")
    print(f"Processing weeks {WEEK_START} to {WEEK_END}")
    
    # Initialize database connection
    db = DatabaseConnector()
    if not db.connect():
        print("[FAIL] Failed to connect to database")
        return
    
    try:
        # Create table if it doesn't exist
        if not create_nfl_game_info_table(db):
            return
        
        # Get season_id
        season_id = get_season_id(db, YEAR)
        print(f"Season ID for {YEAR}: {season_id}")
        
        # Get CSV files to process
        csv_files = get_csv_files()
        if not csv_files:
            print("[WARNING] No CSV files found to process")
            return
        
        print(f"Found {len(csv_files)} game info files to process")
        
        total_processed = 0
        total_inserted = 0
        
        # Process each CSV file
        for csv_file in csv_files:
            try:
                processed_df = process_csv_file(db, csv_file, season_id)
                
                if not processed_df.empty:
                    success = batch_upsert_data(db, 'nfl_game_info', processed_df)
                    if success:
                        total_processed += 1
                        total_inserted += len(processed_df)
                        print(f"[OK] Successfully processed game info")
                    else:
                        print(f"[FAIL] Failed to insert data for {os.path.basename(csv_file)}")
                else:
                    print(f"[WARNING] No data to process for {os.path.basename(csv_file)}")
                    
            except Exception as e:
                print(f"[ERROR] Error processing {os.path.basename(csv_file)}: {e}")
                continue
        
        print(f"\nImport Summary:")
        print(f"   Total CSV files found: {len(csv_files)}")
        print(f"   Total games processed: {total_processed}")
        print(f"   Total records inserted/updated: {total_inserted}")
        
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
    finally:
        db.disconnect()


if __name__ == "__main__":
    main()