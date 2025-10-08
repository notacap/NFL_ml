#!/usr/bin/env python3

import os
import sys
import pandas as pd
import glob
from pathlib import Path

# Add the parent directory to the Python path to import db_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_utils import (
    DatabaseConnector, YEAR, WEEK_START, WEEK_END, 
    batch_upsert_data, handle_null_values, 
    get_season_id, get_week_id, get_game_id, get_player_id, get_team_id,
    create_table_if_not_exists
)


def create_plyr_gm_starters_table(db: DatabaseConnector) -> bool:
    """Create the plyr_gm_starters table if it doesn't exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS plyr_gm_starters (
        plyr_gm_starters_id INT PRIMARY KEY AUTO_INCREMENT,
        plyr_id INT,
        team_id INT,
        week_id INT,
        game_id INT,
        season_id INT,  
        plyr_gm_pos VARCHAR(5),
        FOREIGN KEY (plyr_id) REFERENCES plyr(plyr_id),
        FOREIGN KEY (team_id) REFERENCES nfl_team(team_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (game_id) REFERENCES nfl_game(game_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        UNIQUE KEY uk_player_game (plyr_id, game_id)
    );
    """
    
    return create_table_if_not_exists(db, 'plyr_gm_starters', create_table_sql)


def map_position(position: str) -> str:
    """Map source CSV position to database position using the provided mapping."""
    POSITION_MAPPING = {
        'QB': 'QB', 
        'RB': 'RB', 
        'WR': 'WR', 
        'TE': 'TE',
        'G': 'OL', 'C': 'OL', 'OG': 'OL', 'IOL': 'OL', 'OL': 'OL', 'LG': 'OL', 'RG': 'OL', 'LG/RG': 'OL',
        'T': 'OL', 'OT': 'OL', 'RT': 'OL', 'LT': 'OL', 'RT/LT': 'OL',
        'DE': 'DL', 'DT': 'DL', 'NT': 'DL', 'LDE': 'DL', 'RDE': 'DL', 'LDE/RDE': 'DL', 'LDT': 'DL', 'RDT': 'DL', 'LDT/RDT': 'DL',
        'LB': 'LB', 'ILB': 'LB', 'MLB': 'LB', 'RLB/MLB' : 'LB', 'OLB': 'LB', 'LOLB': 'LB', 'ROLB': 'LB', 'LILB': 'LB', 'RILB': 'LB', 'LILB/RILB': 'LB', 'RILB/LILB': 'LB', 'LLB': 'LB', 'RLB': 'LB',
        'CB': 'DB', 'DB': 'DB', 'LCB' : 'DB', 'RCB' : 'DB', 'LCB/RCB': 'DB', 'FS': 'DB', 'SS': 'DB','S': 'DB', 'SS/FS': 'DB',
        'K': 'K', 'PK': 'K',
        'P': 'P', 
        'LS': 'LS'
    }
    
    return POSITION_MAPPING.get(position, position)


def get_starter_csv_files() -> list:
    """Get list of starter CSV file pairs to process based on WEEK_START and WEEK_END."""
    base_dir = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\games"
    csv_file_pairs = []
    
    for week in range(WEEK_START, WEEK_END + 1):
        week_dir = os.path.join(base_dir, f"week_{week}.0")
        clean_dir = os.path.join(week_dir, "clean")
        
        if os.path.exists(clean_dir):
            # Find all home and away starter files
            home_pattern = os.path.join(clean_dir, "*_gm_home_starters_*.csv")
            away_pattern = os.path.join(clean_dir, "*_gm_away_starters_*.csv")
            
            home_files = glob.glob(home_pattern)
            away_files = glob.glob(away_pattern)
            
            # Group files by game (match by extracting game identifier)
            games = {}
            
            # Process home files
            for home_file in home_files:
                basename = os.path.basename(home_file)
                # Extract game identifier: everything before '_gm_home_starters_'
                parts = basename.split('_gm_home_starters_')
                if len(parts) == 2:
                    game_key = parts[0].replace('cleaned_', '')
                    if game_key not in games:
                        games[game_key] = {}
                    games[game_key]['home'] = home_file
            
            # Process away files  
            for away_file in away_files:
                basename = os.path.basename(away_file)
                # Extract game identifier: everything before '_gm_away_starters_'
                parts = basename.split('_gm_away_starters_')
                if len(parts) == 2:
                    game_key = parts[0].replace('cleaned_', '')
                    if game_key not in games:
                        games[game_key] = {}
                    games[game_key]['away'] = away_file
            
            # Only process games that have both home and away starter files
            for game_key, files in games.items():
                if 'home' in files and 'away' in files:
                    csv_file_pairs.append((week, files['home'], files['away']))
                else:
                    print(f"[WARNING] Week {week}: Missing file for game {game_key}")
        else:
            print(f"[WARNING] Clean directory not found: {clean_dir}")
    
    return sorted(csv_file_pairs)


def get_team_abbreviation(db: DatabaseConnector, team_name: str) -> str:
    """Get team abbreviation for a given team name/nickname."""
    # First try to get the team_id using our existing function
    try:
        team_id = get_team_id(db, team_name)
        # Now get the abbreviation for this team
        result = db.fetch_all("SELECT abrv FROM nfl_team WHERE team_id = %s", (team_id,))
        if result:
            return result[0][0]
    except Exception:
        pass
    
    # If that fails, return the original name
    return team_name


def process_starter_csv_files(db: DatabaseConnector, home_file: str, away_file: str, season_id: int, interactive: bool = False) -> pd.DataFrame:
    """Process both home and away starter CSV files and return consolidated DataFrame."""
    
    print(f"Processing home file: {os.path.basename(home_file)}")
    print(f"Processing away file: {os.path.basename(away_file)}")
    
    # Read both CSV files
    home_df = pd.read_csv(home_file, header=0)
    away_df = pd.read_csv(away_file, header=0)
    
    # Combine both DataFrames
    combined_df = pd.concat([home_df, away_df], ignore_index=True)
    
    if combined_df.empty:
        return pd.DataFrame()
    
    # Get week number from the CSV data
    week_num = combined_df['week'].iloc[0]
    if pd.isna(week_num):
        raise ValueError(f"Week number is null")
    
    # Convert to int (week_num might be 1.0, 2.0, etc.)
    week_num = int(float(week_num))
    
    week_id = get_week_id(db, season_id, week_num)
    
    # Get unique team names for game identification
    unique_teams = combined_df['team'].dropna().unique()
    if len(unique_teams) != 2:
        raise ValueError(f"Expected 2 teams in CSV files, found {len(unique_teams)}: {unique_teams}")
    
    team1_name, team2_name = unique_teams[0], unique_teams[1]
    game_id = get_game_id(db, season_id, week_id, team1_name, team2_name)
    
    # Process each player row
    processed_rows = []
    for index, row in combined_df.iterrows():
        try:
            # Get player info from CSV
            player_name = row.get('Player', '')
            team_name = row.get('team', '')
            position = row.get('Pos', '')
            
            if not player_name or player_name.strip() == '':
                continue
                
            # Skip rows where team column is null, empty, or just whitespace
            if pd.isna(team_name) or not team_name or str(team_name).strip() == '':
                print(f"[INFO] Skipping player {player_name} - no team information")
                continue
            
            # Map position using the provided mapping
            mapped_position = map_position(position)
            
            # Get team abbreviation for player lookup
            team_abrv = get_team_abbreviation(db, team_name)

            # Get player_id and team_id (pass mapped position for better disambiguation)
            plyr_id = get_player_id(db, player_name, team_abrv, season_id, position=mapped_position, interactive=interactive)

            # Skip this player if user chose to skip in interactive mode
            if interactive and plyr_id == 0:
                print(f"[INFO] Skipping player {player_name} - user selection")
                continue

            team_id = get_team_id(db, team_name)
            
            # Create processed row
            processed_row = {
                'plyr_id': plyr_id,
                'team_id': team_id,
                'week_id': week_id,
                'game_id': game_id,
                'season_id': season_id,
                'plyr_gm_pos': mapped_position
            }
            
            processed_rows.append(processed_row)
            
        except Exception as e:
            print(f"[WARNING] Error processing row {index} for player {row.get('Player', 'Unknown')}: {e}")
            continue
    
    if processed_rows:
        processed_df = pd.DataFrame(processed_rows)
        processed_df = handle_null_values(processed_df)
        
        # Ensure proper data types for foreign key columns
        fk_columns = ['plyr_id', 'team_id', 'week_id', 'game_id', 'season_id']
        for col in fk_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').astype('int64')
        
        return processed_df
    else:
        return pd.DataFrame()


def main():
    """Main function to process all CSV file pairs."""
    
    print(f"Starting Player Game Starters Data Import for {YEAR}")
    print(f"Processing weeks {WEEK_START} to {WEEK_END}")
    print("[INFO] Interactive mode enabled - you will be prompted for player selection when multiple/no matches are found")

    # Initialize database connection
    db = DatabaseConnector()
    if not db.connect():
        print("[FAIL] Failed to connect to database")
        return
    
    try:
        # Create table if it doesn't exist
        if not create_plyr_gm_starters_table(db):
            return
        
        # Get season_id
        season_id = get_season_id(db, YEAR)
        print(f"Season ID for {YEAR}: {season_id}")
        
        # Get starter CSV file pairs to process
        csv_file_pairs = get_starter_csv_files()
        if not csv_file_pairs:
            print("[WARNING] No starter CSV file pairs found to process")
            return
        
        print(f"Found {len(csv_file_pairs)} game pairs to process")
        
        total_processed = 0
        total_inserted = 0
        
        # Process each file pair
        for week, home_file, away_file in csv_file_pairs:
            try:
                processed_df = process_starter_csv_files(db, home_file, away_file, season_id, interactive=True)
                
                if not processed_df.empty:
                    success = batch_upsert_data(db, 'plyr_gm_starters', processed_df)
                    if success:
                        rows_in_file = len(processed_df)
                        total_processed += rows_in_file
                        total_inserted += rows_in_file
                        print(f"[OK] Week {week}: Processed {rows_in_file} starter records")
                    else:
                        print(f"[FAIL] Week {week}: Failed to insert data")
                else:
                    print(f"[WARNING] Week {week}: No data to process")
                    
            except Exception as e:
                print(f"[ERROR] Error processing week {week} files: {e}")
                print(f"   Home: {os.path.basename(home_file)}")
                print(f"   Away: {os.path.basename(away_file)}")
                continue
        
        print(f"\nStarter Import Summary:")
        print(f"   Total game pairs processed: {len(csv_file_pairs)}")
        print(f"   Total starter records processed: {total_processed}")
        print(f"   Total records inserted/updated: {total_inserted}")
        
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
    finally:
        db.disconnect()


if __name__ == "__main__":
    main()