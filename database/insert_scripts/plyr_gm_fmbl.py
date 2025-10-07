#!/usr/bin/env python3

import os
import sys
import pandas as pd
import glob
from pathlib import Path

# Add the parent directory to the Python path to import db_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_utils import (
    DatabaseConnector, YEAR, WEEK_START, WEEK_END, batch_upsert_data, 
    handle_null_values, get_season_id, get_week_id, get_team_id, 
    get_game_id, get_player_id, create_table_if_not_exists
)


def create_plyr_gm_fmbl_table(db: DatabaseConnector) -> bool:
    """Create the plyr_gm_fmbl table if it doesn't exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS plyr_gm_fmbl (
        plyr_gm_fmbl_id INT PRIMARY KEY AUTO_INCREMENT,
        plyr_id INT,
        week_id INT,
        game_id INT,
        season_id INT,
        team_id INT,
        plyr_gm_fmbl INT,
        plyr_gm_fl INT,
        FOREIGN KEY (plyr_id) REFERENCES plyr(plyr_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (game_id) REFERENCES nfl_game(game_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        FOREIGN KEY (team_id) REFERENCES nfl_team(team_id),
        UNIQUE KEY uk_player_game (plyr_id, game_id)
    );
    """
    
    return create_table_if_not_exists(db, 'plyr_gm_fmbl', create_table_sql)


def parse_csv_single_header(file_path: str) -> pd.DataFrame:
    """Parse CSV file with single header row and return DataFrame."""
    df = pd.read_csv(file_path, header=0)
    return df


def process_fumbles_data(file_path: str) -> pd.DataFrame:
    """Process fumbles CSV file and return DataFrame with standardized columns."""
    df = parse_csv_single_header(file_path)
    
    # Create mapping for fumbles stats (new single-header format)
    column_mapping = {
        'Fumbles Fmb': 'plyr_gm_fmbl',
        'Fumbles FL': 'plyr_gm_fl'
    }
    
    # Rename columns
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})
    
    return df


def process_game_files(db: DatabaseConnector, week: int, season_id: int, interactive: bool = False) -> pd.DataFrame:
    """Process fumbles files for a specific week and return DataFrame."""
    
    base_dir = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\games"
    week_dir = os.path.join(base_dir, f"week_{week}.0")
    clean_dir = os.path.join(week_dir, "clean")
    
    if not os.path.exists(clean_dir):
        print(f"[WARNING] Clean directory not found: {clean_dir}")
        return pd.DataFrame()
    
    # Find all fumbles files in this week
    fumbles_pattern = "*_gm_plyr_fumbles_*.csv"
    
    # Get all fumbles files from clean directory
    fumbles_files = glob.glob(os.path.join(clean_dir, f"cleaned_{fumbles_pattern}"))
    
    if not fumbles_files:
        print(f"[WARNING] No fumbles files found in {clean_dir}")
        return pd.DataFrame()
    
    all_processed_rows = []
    
    # Process each fumbles file
    for fumbles_file in fumbles_files:
        try:
            print(f"Processing fumbles file: {os.path.basename(fumbles_file)}")
            
            # Process fumbles data
            fumbles_df = process_fumbles_data(fumbles_file)
            
            if fumbles_df.empty:
                continue
            
            # Get week number from the DataFrame
            week_num = None
            if 'week' in fumbles_df.columns:
                week_num = fumbles_df['week'].iloc[0]
            
            if week_num is None:
                week_num = float(week)
            
            week_id = get_week_id(db, season_id, int(week_num))
            
            # Get unique teams from the fumbles file to determine game_id
            unique_teams = set()
            if 'Tm' in fumbles_df.columns:
                unique_teams.update(fumbles_df['Tm'].unique())
            
            if len(unique_teams) != 2:
                print(f"[WARNING] Expected 2 teams for fumbles file, found {len(unique_teams)}: {unique_teams}")
                continue
            
            team1_abrv, team2_abrv = list(unique_teams)
            game_id = get_game_id(db, season_id, week_id, team1_abrv, team2_abrv)
            
            # Process each player row
            for index, row in fumbles_df.iterrows():
                try:
                    player_name = row.get('Player', '')
                    if not player_name or player_name.strip() == '':
                        continue
                    
                    # Get team for this player
                    team_abrv = row.get('Tm', '')
                    if not team_abrv:
                        print(f"[WARNING] Could not determine team for player {player_name}")
                        continue
                    
                    # Get player_id and team_id
                    plyr_id = get_player_id(db, player_name, team_abrv, season_id, interactive=interactive)

                    # Skip this player if user chose to skip in interactive mode
                    if interactive and plyr_id == 0:
                        print(f"[INFO] Skipping player {player_name} - user selection")
                        continue

                    team_id = get_team_id(db, team_abrv)
                    
                    # Create processed row
                    processed_row = {
                        'plyr_id': plyr_id,
                        'week_id': week_id,
                        'game_id': game_id,
                        'season_id': season_id,
                        'team_id': team_id
                    }
                    
                    # Add fumbles stats
                    fumbles_columns = ['plyr_gm_fmbl', 'plyr_gm_fl']
                    
                    for col in fumbles_columns:
                        if col in fumbles_df.columns:
                            value = row.get(col)
                            if pd.notna(value) and value != '' and str(value).strip() != '':
                                try:
                                    processed_row[col] = int(float(value))
                                except (ValueError, TypeError):
                                    processed_row[col] = None
                            else:
                                processed_row[col] = None
                    
                    all_processed_rows.append(processed_row)
                    
                except Exception as e:
                    print(f"[WARNING] Error processing player {row.get('Player', 'Unknown')}: {e}")
                    continue
        
        except Exception as e:
            print(f"[ERROR] Error processing fumbles file {fumbles_file}: {e}")
            continue
    
    if all_processed_rows:
        final_df = pd.DataFrame(all_processed_rows)
        final_df = handle_null_values(final_df)
        
        # Ensure proper data types
        int_columns = ['plyr_id', 'week_id', 'game_id', 'season_id', 'team_id', 'plyr_gm_fmbl', 'plyr_gm_fl']
        
        for col in int_columns:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        
        return final_df
    else:
        return pd.DataFrame()


def main():
    """Main function to process all fumbles CSV files."""
    
    print(f"Starting Player Game Fumbles Data Import for {YEAR}")
    print(f"Processing weeks {WEEK_START} to {WEEK_END}")
    print("[INFO] Interactive mode enabled - you will be prompted for player selection when multiple/no matches are found")
    
    # Initialize database connection
    db = DatabaseConnector()
    if not db.connect():
        print("[FAIL] Failed to connect to database")
        return
    
    try:
        # Create table if it doesn't exist
        if not create_plyr_gm_fmbl_table(db):
            print("[FAIL] Failed to create plyr_gm_fmbl table")
            return
        
        # Get season_id
        season_id = get_season_id(db, YEAR)
        print(f"Season ID for {YEAR}: {season_id}")
        
        total_processed = 0
        total_inserted = 0
        
        # Process each week
        for week in range(WEEK_START, WEEK_END + 1):
            try:
                print(f"\nProcessing Week {week}...")
                processed_df = process_game_files(db, week, season_id, interactive=True)
                
                if not processed_df.empty:
                    success = batch_upsert_data(db, 'plyr_gm_fmbl', processed_df)
                    if success:
                        rows_in_week = len(processed_df)
                        total_processed += rows_in_week
                        total_inserted += rows_in_week
                        print(f"[OK] Week {week}: Processed {rows_in_week} fumbles records")
                    else:
                        print(f"[FAIL] Week {week}: Failed to insert fumbles data")
                else:
                    print(f"[WARNING] Week {week}: No fumbles data to process")
                    
            except Exception as e:
                print(f"[ERROR] Error processing week {week}: {e}")
                continue
        
        print(f"\nFumbles Import Summary:")
        print(f"   Total weeks processed: {WEEK_END - WEEK_START + 1}")
        print(f"   Total fumbles records processed: {total_processed}")
        print(f"   Total records inserted/updated: {total_inserted}")
        
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
    finally:
        db.disconnect()


if __name__ == "__main__":
    main()