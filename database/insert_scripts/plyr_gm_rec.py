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
    get_season_id, get_week_id, get_game_id, get_player_id,
    create_table_if_not_exists, get_team_id
)


def create_plyr_gm_rec_table(db: DatabaseConnector) -> bool:
    """Create the plyr_gm_rec table if it doesn't exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS plyr_gm_rec (
        adv_plyr_gm_rec_id INT PRIMARY KEY AUTO_INCREMENT,
        plyr_id INT,
        week_id INT,
        game_id INT,
        season_id INT,
        team_id INT,
        plyr_gm_rec_tgt INT,
        plyr_gm_rec INT,
        plyr_gm_rec_yds INT,
        plyr_gm_rec_td INT,
        plyr_gm_rec_lng INT,
        plyr_gm_rec_first_dwn INT,
        plyr_gm_rec_aybc INT,
        plyr_gm_rec_aybc_route FLOAT(7,4),
        plyr_gm_rec_yac INT,
        plyr_gm_rec_yac_route FLOAT(7,4),
        plyr_gm_rec_adot FLOAT(7,4),
        plyr_gm_rec_brkn_tkl INT,
        plyr_gm_rec_brkn_tkl_rec FLOAT(7,4),
        plyr_gm_rec_drp INT,
        plyr_gm_rec_drp_pct FLOAT(7,4),
        plyr_gm_rec_int INT,
        plyr_gm_rec_pass_rtg FLOAT(7,4),
        plyr_rec_catch_pct DECIMAL(6,4),
        plyr_rec_yds_tgt DECIMAL(7,4),
        FOREIGN KEY (plyr_id) REFERENCES plyr(plyr_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (game_id) REFERENCES nfl_game(game_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        FOREIGN KEY (team_id) REFERENCES nfl_team(team_id),
        UNIQUE KEY uk_player_game (plyr_id, game_id)
    );
    """
    
    return create_table_if_not_exists(db, 'plyr_gm_rec', create_table_sql)


def get_csv_files() -> list:
    """Get list of CSV files to process based on WEEK_START and WEEK_END."""
    base_dir = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\games"
    csv_files = []
    
    for week in range(WEEK_START, WEEK_END + 1):
        week_dir = os.path.join(base_dir, f"week_{week}.0")
        clean_dir = os.path.join(week_dir, "clean")
        if os.path.exists(clean_dir):
            # Get both basic and advanced receiving files for this week
            basic_pattern = os.path.join(clean_dir, "cleaned_*_gm_plyr_receiving_*.csv")
            advanced_pattern = os.path.join(clean_dir, "cleaned_*_gm_plyr_adv_receiving_*.csv")
            
            basic_files = glob.glob(basic_pattern)
            advanced_files = glob.glob(advanced_pattern)
            
            # Match files by game
            for basic_file in basic_files:
                basic_name = os.path.basename(basic_file)
                # Extract game identifier (everything before _gm_plyr_receiving)
                game_id_part = basic_name.replace('cleaned_', '').split('_gm_plyr_receiving')[0]
                
                # Find corresponding advanced file
                advanced_file = None
                for adv_file in advanced_files:
                    if game_id_part in os.path.basename(adv_file):
                        advanced_file = adv_file
                        break
                
                if advanced_file:
                    csv_files.append((week, basic_file, advanced_file))
                else:
                    print(f"[WARNING] No advanced receiving file found for {basic_name}")
        else:
            print(f"[WARNING] Clean directory not found: {clean_dir}")
    
    return sorted(csv_files)


def parse_csv_file(file_path: str) -> pd.DataFrame:
    """Parse CSV file with single header row and return DataFrame."""
    return pd.read_csv(file_path, header=0)


def process_csv_files(db: DatabaseConnector, basic_file: str, advanced_file: str, season_id: int, interactive: bool = False) -> pd.DataFrame:
    """Process both basic and advanced receiving CSV files and return combined DataFrame."""
    
    print(f"Processing files:")
    print(f"  Basic: {os.path.basename(basic_file)}")
    print(f"  Advanced: {os.path.basename(advanced_file)}")
    
    # Read both CSV files
    basic_df = parse_csv_file(basic_file)
    advanced_df = parse_csv_file(advanced_file)
    
    # Get week number and game context from advanced file (both should be the same)
    if 'week' in advanced_df.columns:
        week_num = advanced_df['week'].iloc[0]
    else:
        raise ValueError(f"Week column not found in file {advanced_file}")
    
    if pd.isna(week_num):
        raise ValueError(f"Week number is null in file {advanced_file}")
    
    # Convert to int (week_num might be 1.0, 2.0, etc.)
    week_num = int(float(week_num))
    week_id = get_week_id(db, season_id, week_num)
    
    # Get unique team abbreviations from advanced file
    unique_teams = advanced_df['Tm'].dropna().unique()
    if len(unique_teams) != 2:
        raise ValueError(f"Expected 2 teams in advanced CSV file, found {len(unique_teams)}: {unique_teams}")
    
    team1_abrv, team2_abrv = unique_teams[0], unique_teams[1]
    game_id = get_game_id(db, season_id, week_id, team1_abrv, team2_abrv)
    
    # Process each player in the advanced file (primary source)
    processed_rows = []
    for index, adv_row in advanced_df.iterrows():
        try:
            # Get player name and team from advanced row
            player_name = adv_row.get('Player', '')
            team_abrv = adv_row.get('Tm', '')
            
            if not player_name or player_name.strip() == '':
                continue
                
            # Skip rows where Tm column is null, empty, or just whitespace
            if pd.isna(team_abrv) or not team_abrv or str(team_abrv).strip() == '':
                print(f"[INFO] Skipping player {player_name} - no team information")
                continue
            
            # Get player_id and team_id
            plyr_id = get_player_id(db, player_name, team_abrv, season_id, interactive=interactive)

            # Skip this player if user chose to skip in interactive mode
            if interactive and plyr_id == 0:
                print(f"[INFO] Skipping player {player_name} - user selection")
                continue

            team_id = get_team_id(db, team_abrv)
            
            # Find corresponding basic file row for this player
            basic_row = None
            basic_matches = basic_df[basic_df['Player'] == player_name]
            if not basic_matches.empty:
                basic_row = basic_matches.iloc[0]
            
            # Create processed row with foreign keys
            processed_row = {
                'plyr_id': plyr_id,
                'week_id': week_id,
                'game_id': game_id,
                'season_id': season_id,
                'team_id': team_id
            }
            
            # Map basic receiving stats (prioritize advanced file, fall back to basic file)
            basic_stats = {
                'plyr_gm_rec_tgt': 'Tgt',           # Targets
                'plyr_gm_rec': 'Rec',               # Receptions
                'plyr_gm_rec_yds': 'Yds',           # Receiving yards
                'plyr_gm_rec_td': 'TD'              # Touchdowns
            }
            
            # Use advanced file for basic stats (since it has same data)
            for db_col, csv_col in basic_stats.items():
                value = adv_row.get(csv_col)
                if pd.notna(value) and value != '' and str(value).strip() != '':
                    try:
                        processed_row[db_col] = int(float(value))
                    except (ValueError, TypeError):
                        processed_row[db_col] = None
                else:
                    processed_row[db_col] = None
            
            # Get longest reception from basic file (only available there)
            if basic_row is not None:
                lng_value = basic_row.get('Receiving Lng')
                if pd.notna(lng_value) and lng_value != '' and str(lng_value).strip() != '':
                    try:
                        processed_row['plyr_gm_rec_lng'] = int(float(lng_value))
                    except (ValueError, TypeError):
                        processed_row['plyr_gm_rec_lng'] = None
                else:
                    processed_row['plyr_gm_rec_lng'] = None
            else:
                processed_row['plyr_gm_rec_lng'] = None
            
            # Map advanced receiving stats from advanced file
            advanced_stats = {
                'plyr_gm_rec_first_dwn': '1D',
                'plyr_gm_rec_aybc': 'YBC',
                'plyr_gm_rec_aybc_route': 'YBC/R',
                'plyr_gm_rec_yac': 'YAC',
                'plyr_gm_rec_yac_route': 'YAC/R',
                'plyr_gm_rec_adot': 'ADOT',
                'plyr_gm_rec_brkn_tkl': 'BrkTkl',
                'plyr_gm_rec_brkn_tkl_rec': 'Rec/Br',
                'plyr_gm_rec_drp': 'Drop',
                'plyr_gm_rec_drp_pct': 'Drop%',
                'plyr_gm_rec_int': 'Int',
                'plyr_gm_rec_pass_rtg': 'Rat'
            }
            
            # Process advanced stats
            for db_col, csv_col in advanced_stats.items():
                if csv_col in advanced_df.columns:
                    value = adv_row.get(csv_col)
                    
                    # Handle percentage columns specially (convert to decimal)
                    if db_col in ['plyr_gm_rec_drp_pct']:
                        if pd.notna(value) and value != '' and str(value).strip() != '':
                            try:
                                # Convert percentage to decimal (16.7 -> 0.167)
                                processed_row[db_col] = float(value) / 100.0
                            except (ValueError, TypeError):
                                processed_row[db_col] = None
                        else:
                            processed_row[db_col] = None
                    
                    # Handle float columns (ratios, ADOT, etc.)
                    elif db_col in ['plyr_gm_rec_aybc_route', 'plyr_gm_rec_yac_route', 'plyr_gm_rec_adot', 
                                   'plyr_gm_rec_brkn_tkl_rec', 'plyr_gm_rec_pass_rtg']:
                        if pd.notna(value) and value != '' and str(value).strip() != '':
                            try:
                                processed_row[db_col] = float(value)
                            except (ValueError, TypeError):
                                processed_row[db_col] = None
                        else:
                            processed_row[db_col] = None
                    
                    # Handle integer columns
                    elif db_col in ['plyr_gm_rec_first_dwn', 'plyr_gm_rec_aybc', 'plyr_gm_rec_yac', 
                                   'plyr_gm_rec_brkn_tkl', 'plyr_gm_rec_drp', 'plyr_gm_rec_int']:
                        if pd.notna(value) and value != '' and str(value).strip() != '':
                            try:
                                processed_row[db_col] = int(float(value))
                            except (ValueError, TypeError):
                                processed_row[db_col] = None
                        else:
                            processed_row[db_col] = None
            
            # Calculate new derived columns
            # plyr_rec_catch_pct = plyr_gm_rec / plyr_gm_rec_tgt
            targets = processed_row.get('plyr_gm_rec_tgt')
            receptions = processed_row.get('plyr_gm_rec')
            receiving_yards = processed_row.get('plyr_gm_rec_yds')
            
            if targets and targets > 0:
                if receptions is not None:
                    processed_row['plyr_rec_catch_pct'] = round(float(receptions) / float(targets), 4)
                else:
                    processed_row['plyr_rec_catch_pct'] = None
                
                if receiving_yards is not None:
                    processed_row['plyr_rec_yds_tgt'] = round(float(receiving_yards) / float(targets), 4)
                else:
                    processed_row['plyr_rec_yds_tgt'] = None
            else:
                processed_row['plyr_rec_catch_pct'] = None
                processed_row['plyr_rec_yds_tgt'] = None
            
            processed_rows.append(processed_row)
            
        except Exception as e:
            print(f"[WARNING] Error processing row {index} for player {adv_row.get('Player', 'Unknown')}: {e}")
            continue
    
    if processed_rows:
        processed_df = pd.DataFrame(processed_rows)
        processed_df = handle_null_values(processed_df)
        
        # Ensure proper data types for foreign key columns
        fk_columns = ['plyr_id', 'week_id', 'game_id', 'season_id', 'team_id']
        for col in fk_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').astype('int64')
        
        # Handle nullable integer columns
        nullable_int_columns = ['plyr_gm_rec_tgt', 'plyr_gm_rec', 'plyr_gm_rec_yds', 'plyr_gm_rec_td', 
                               'plyr_gm_rec_lng', 'plyr_gm_rec_first_dwn', 'plyr_gm_rec_aybc', 
                               'plyr_gm_rec_yac', 'plyr_gm_rec_brkn_tkl', 'plyr_gm_rec_drp', 'plyr_gm_rec_int']
        
        for col in nullable_int_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                processed_df[col] = processed_df[col].round().astype('Int64')
        
        # Handle float columns
        float_columns = ['plyr_gm_rec_aybc_route', 'plyr_gm_rec_yac_route', 'plyr_gm_rec_adot', 
                        'plyr_gm_rec_brkn_tkl_rec', 'plyr_gm_rec_drp_pct', 'plyr_gm_rec_pass_rtg',
                        'plyr_rec_catch_pct', 'plyr_rec_yds_tgt']
        
        for col in float_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        return processed_df
    else:
        return pd.DataFrame()


def main():
    """Main function to process all CSV files."""
    
    print(f"Starting Player Game Receiving Data Import for {YEAR}")
    print(f"Processing weeks {WEEK_START} to {WEEK_END}")
    print("[INFO] Interactive mode enabled - you will be prompted for player selection when multiple/no matches are found")
    
    # Initialize database connection
    db = DatabaseConnector()
    if not db.connect():
        print("[FAIL] Failed to connect to database")
        return
    
    try:
        # Create table if it doesn't exist
        if not create_plyr_gm_rec_table(db):
            return
        
        # Get season_id
        season_id = get_season_id(db, YEAR)
        print(f"Season ID for {YEAR}: {season_id}")
        
        # Get CSV files to process
        csv_files = get_csv_files()
        if not csv_files:
            print("[WARNING] No CSV files found to process")
            return
        
        print(f"Found {len(csv_files)} game file pairs to process")
        
        total_processed = 0
        total_inserted = 0
        
        # Process each file pair
        for week, basic_file, advanced_file in csv_files:
            try:
                processed_df = process_csv_files(db, basic_file, advanced_file, season_id, interactive=True)
                
                if not processed_df.empty:
                    success = batch_upsert_data(db, 'plyr_gm_rec', processed_df)
                    if success:
                        rows_in_file = len(processed_df)
                        total_processed += rows_in_file
                        total_inserted += rows_in_file
                        print(f"[OK] Week {week}: Processed {rows_in_file} player records")
                    else:
                        print(f"[FAIL] Week {week}: Failed to insert data")
                else:
                    print(f"[WARNING] Week {week}: No data to process")
                    
            except Exception as e:
                print(f"[ERROR] Error processing week {week}: {e}")
                continue
        
        print(f"\nImport Summary:")
        print(f"   Total game pairs processed: {len(csv_files)}")
        print(f"   Total player records processed: {total_processed}")
        print(f"   Total records inserted/updated: {total_inserted}")
        
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
    finally:
        db.disconnect()


if __name__ == "__main__":
    main()