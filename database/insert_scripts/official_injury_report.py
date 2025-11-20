import pandas as pd
import os
import glob
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_utils import (
    DatabaseConnector,
    YEAR,
    WEEK,
    get_season_id,
    get_week_id,
    get_team_id,
    get_player_id,
    batch_upsert_data
)

def get_most_recent_csv(team_dir: str) -> str:
    csv_files = glob.glob(os.path.join(team_dir, "official_injury_report_*.csv"))
    if not csv_files:
        return None
    return max(csv_files, key=os.path.getctime)

def get_game_id_for_team(db: DatabaseConnector, season_id: int, week_id: int, team_id: int) -> int:
    query = """
        SELECT game_id FROM nfl_game 
        WHERE season_id = %s 
        AND week_id = %s
        AND (home_team_id = %s OR away_team_id = %s)
    """
    result = db.fetch_all(query, (season_id, week_id, team_id, team_id))
    if result:
        return result[0][0]
    return None

def detect_game_day(df: pd.DataFrame) -> str:
    columns = df.columns.tolist()
    if 'Mon' in columns and 'Tue' in columns:
        return 'Thursday'
    elif 'Sat' in columns:
        return 'Monday'
    else:
        return 'Sunday'

def map_practice_columns(row: pd.Series, game_day: str) -> dict:
    mapping = {
        'wed_practice_status': None,
        'thurs_practice_status': None,
        'fri_practice_status': None,
        'sat_practice_status': None
    }
    
    if game_day == 'Thursday':
        mapping['wed_practice_status'] = row.get('Mon', None)
        mapping['thurs_practice_status'] = row.get('Tue', None)
        mapping['fri_practice_status'] = row.get('Wed', None)
    elif game_day == 'Sunday':
        mapping['wed_practice_status'] = row.get('Wed', None)
        mapping['thurs_practice_status'] = row.get('Thu', None)
        mapping['fri_practice_status'] = row.get('Fri', None)
    elif game_day == 'Monday':
        mapping['thurs_practice_status'] = row.get('Thu', None)
        mapping['fri_practice_status'] = row.get('Fri', None)
        mapping['sat_practice_status'] = row.get('Sat', None)
    
    for key in mapping:
        if mapping[key] == '(-)' or mapping[key] == '' or pd.isna(mapping[key]):
            mapping[key] = 'No_Designation' if mapping[key] == '(-)' else None
    
    return mapping

def process_injury_report_file(db: DatabaseConnector, file_path: str, team_name: str, 
                                season_id: int, week_id: int) -> list:
    df = pd.read_csv(file_path)
    
    if df.empty:
        db.logger.warning(f"Empty CSV file: {file_path}")
        return []
    
    team_id = get_team_id(db, team_name)
    if not team_id:
        db.logger.error(f"Team not found: {team_name}")
        return []
    
    game_id = get_game_id_for_team(db, season_id, week_id, team_id)
    if not game_id:
        db.logger.error(f"Game not found for team {team_name}, week {week_id}")
        return []
    
    game_day = detect_game_day(df)
    db.logger.info(f"Processing {team_name} - Game day: {game_day}")
    
    records = []
    for _, row in df.iterrows():
        player_name = row['Player']
        position = row.get('Position', None)
        
        team_abbr_map = {
            'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
            'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
            'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
            'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
            'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
            'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
            'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
            'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
            'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
            'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
            'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
        }
        team_abbr = team_abbr_map.get(team_name, team_name[:3].upper())
        
        plyr_id = get_player_id(db, player_name, team_abbr, season_id, position=position)
        if not plyr_id:
            db.logger.warning(f"Player not found: {player_name} ({team_name})")
            continue
        
        practice_mapping = map_practice_columns(row, game_day)
        game_status = row.get('Game Status', None)
        if game_status == '(-)' or game_status == '':
            game_status = 'No_Designation'
        
        was_inactive = 1 if game_status == 'Inactive' else 0
        
        record = {
            'season_id': season_id,
            'plyr_id': plyr_id,
            'plyr_name': player_name,
            'team_id': team_id,
            'game_id': game_id,
            'week_id': week_id,
            'was_inactive': was_inactive,
            'wed_practice_status': practice_mapping['wed_practice_status'],
            'thurs_practice_status': practice_mapping['thurs_practice_status'],
            'fri_practice_status': practice_mapping['fri_practice_status'],
            'sat_practice_status': practice_mapping['sat_practice_status'],
            'game_status': game_status
        }
        records.append(record)
    
    return records

def main():
    db = DatabaseConnector()
    if not db.connect():
        print("Failed to connect to database")
        return False
    
    try:
        season_id = get_season_id(db, YEAR)
        week_id = get_week_id(db, season_id, str(WEEK))
        
        base_path = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\official_injury_report\\week_{WEEK}"
        
        if not os.path.exists(base_path):
            db.logger.error(f"Base path does not exist: {base_path}")
            return False
        
        team_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        
        all_records = []
        files_processed = 0
        
        for team_dir_name in team_dirs:
            team_dir_path = os.path.join(base_path, team_dir_name)
            most_recent_csv = get_most_recent_csv(team_dir_path)
            
            if not most_recent_csv:
                db.logger.warning(f"No CSV files found for {team_dir_name}")
                continue
            
            db.logger.info(f"Processing {team_dir_name}: {os.path.basename(most_recent_csv)}")
            
            team_name = team_dir_name.replace('_', ' ')
            records = process_injury_report_file(db, most_recent_csv, team_name, season_id, week_id)
            all_records.extend(records)
            files_processed += 1
        
        if all_records:
            df = pd.DataFrame(all_records)
            success = batch_upsert_data(db, 'injury_report', df)
            
            if success:
                db.logger.info(f"Successfully processed {files_processed} files with {len(all_records)} total records")
                return True
            else:
                db.logger.error("Failed to upsert injury report data")
                return False
        else:
            db.logger.warning("No records to process")
            return False
            
    finally:
        db.disconnect()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
