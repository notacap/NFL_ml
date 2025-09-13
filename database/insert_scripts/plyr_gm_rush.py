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


def create_plyr_gm_rush_table(db: DatabaseConnector) -> bool:
    """Create the plyr_gm_rush table if it doesn't exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS plyr_gm_rush (
        adv_plyr_gm_rush_id INT PRIMARY KEY AUTO_INCREMENT,
        plyr_id INT,
        week_id INT,
        game_id INT,
        season_id INT,
        team_id INT,
        plyr_gm_rush_att INT,
        plyr_gm_rush_yds INT,
        plyr_gm_rush_yds_att DECIMAL(7,4),
        plyr_gm_rush_td INT,
        plyr_gm_rush_lng INT,
        plyr_gm_rush_first_dwn INT,
        plyr_gm_rush_ybc INT,
        plyr_gm_rush_ybc_att FLOAT(7,4),
        plyr_gm_rush_yac INT,
        plyr_gm_rush_yac_att FLOAT(7,4),
        plyr_gm_rush_brkn_tkl INT,
        plyr_gm_rush_brkn_tkl_att FLOAT(7,4),
        FOREIGN KEY (plyr_id) REFERENCES plyr(plyr_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (game_id) REFERENCES nfl_game(game_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        FOREIGN KEY (team_id) REFERENCES nfl_team(team_id),
        UNIQUE KEY uk_player_game (plyr_id, game_id)
    );
    """
    
    return create_table_if_not_exists(db, 'plyr_gm_rush', create_table_sql)


def get_csv_files() -> list:
    """Get list of CSV files to process based on WEEK_START and WEEK_END."""
    base_dir = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\games"
    csv_files = []
    
    for week in range(WEEK_START, WEEK_END + 1):
        week_dir = os.path.join(base_dir, f"week_{week}.0")
        clean_dir = os.path.join(week_dir, "clean")
        if os.path.exists(clean_dir):
            # Look for both basic and advanced rushing files
            basic_pattern = os.path.join(clean_dir, "cleaned_*_gm_plyr_rushing_*.csv")
            adv_pattern = os.path.join(clean_dir, "cleaned_*_gm_plyr_adv_rushing_*.csv")
            
            basic_files = glob.glob(basic_pattern)
            adv_files = glob.glob(adv_pattern)
            
            # Create pairs of basic and advanced files for the same games
            for basic_file in basic_files:
                # Extract game identifier from basic file
                basic_basename = os.path.basename(basic_file)
                game_prefix = basic_basename.replace("_gm_plyr_rushing_", "_GAME_").replace("cleaned_", "").split("_GAME_")[0]
                
                # Find corresponding advanced file
                adv_file = None
                for adv in adv_files:
                    adv_basename = os.path.basename(adv)
                    adv_game_prefix = adv_basename.replace("_gm_plyr_adv_rushing_", "_GAME_").replace("cleaned_", "").split("_GAME_")[0]
                    if game_prefix == adv_game_prefix:
                        adv_file = adv
                        break
                
                if adv_file:
                    csv_files.append((week, basic_file, adv_file))
                else:
                    print(f"[WARNING] No advanced rushing file found for basic file: {os.path.basename(basic_file)}")
        else:
            print(f"[WARNING] Clean directory not found: {clean_dir}")
    
    return sorted(csv_files)


def process_basic_rushing_csv(file_path: str) -> pd.DataFrame:
    """Process basic rushing CSV file and return DataFrame with standardized columns."""
    df = pd.read_csv(file_path, header=0)
    
    # Create mapping for basic rushing stats (from plyr_gm_off.py logic)
    column_mapping = {
        'Rushing Att': 'plyr_gm_rush_att',
        'Rushing Yds': 'plyr_gm_rush_yds', 
        'Rushing TD': 'plyr_gm_rush_td',
        'Rushing Lng': 'plyr_gm_rush_lng'
    }
    
    # Rename columns
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})
    
    return df


def process_advanced_rushing_csv(file_path: str) -> pd.DataFrame:
    """Process advanced rushing CSV file and return DataFrame with standardized columns."""
    df = pd.read_csv(file_path, header=0)
    
    # Create mapping for advanced rushing stats (from adv_plyr_gm_rush.py logic)  
    column_mapping = {
        '1D': 'plyr_gm_rush_first_dwn',
        'YBC': 'plyr_gm_rush_ybc',
        'YBC/Att': 'plyr_gm_rush_ybc_att',
        'YAC': 'plyr_gm_rush_yac',
        'YAC/Att': 'plyr_gm_rush_yac_att',
        'BrkTkl': 'plyr_gm_rush_brkn_tkl',
        'Att/Br': 'plyr_gm_rush_brkn_tkl_att'
    }
    
    # Rename columns
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})
    
    return df


def combine_rushing_data(basic_df: pd.DataFrame, adv_df: pd.DataFrame) -> pd.DataFrame:
    """Combine basic and advanced rushing data into a single DataFrame.
    
    Only includes players that appear in BOTH files to ensure complete data coverage.
    This filters out players with 0 rushing attempts who only appear in the basic file.
    """
    
    if basic_df.empty or adv_df.empty:
        print("[WARNING] One of the rushing DataFrames is empty")
        return pd.DataFrame()
    
    # Get unique players from each file for logging
    basic_players = set(basic_df['Player'].dropna().unique())
    adv_players = set(adv_df['Player'].dropna().unique())
    
    players_only_in_basic = basic_players - adv_players
    players_only_in_adv = adv_players - basic_players
    players_in_both = basic_players & adv_players
    
    print(f"[INFO] Players in both files: {len(players_in_both)}")
    print(f"[INFO] Players only in basic file (excluded): {len(players_only_in_basic)}")
    print(f"[INFO] Players only in advanced file (excluded): {len(players_only_in_adv)}")
    
    if players_only_in_basic:
        print(f"[INFO] Excluded from basic file: {sorted(list(players_only_in_basic))}")
    if players_only_in_adv:
        print(f"[INFO] Excluded from advanced file: {sorted(list(players_only_in_adv))}")
    
    # Use INNER merge to only include players that appear in BOTH files
    combined_df = basic_df.merge(adv_df, on=['Player', 'Tm'], how='inner', suffixes=('_basic', '_adv'))
    
    # Use week, year, game_id from basic file (they should be the same)
    for col in ['week', 'year', 'game_id']:
        if f'{col}_basic' in combined_df.columns:
            combined_df[col] = combined_df[f'{col}_basic']
        elif f'{col}_adv' in combined_df.columns:
            combined_df[col] = combined_df[f'{col}_adv']
        # Clean up duplicate columns
        for suffix in ['_basic', '_adv']:
            if f'{col}{suffix}' in combined_df.columns:
                combined_df = combined_df.drop(columns=[f'{col}{suffix}'])
    
    return combined_df


def process_csv_files(db: DatabaseConnector, basic_file: str, adv_file: str, season_id: int) -> pd.DataFrame:
    """Process a pair of basic and advanced rushing CSV files and return processed DataFrame."""
    
    print(f"Processing files:")
    print(f"  Basic: {os.path.basename(basic_file)}")
    print(f"  Advanced: {os.path.basename(adv_file)}")
    
    # Process both CSV files
    basic_df = process_basic_rushing_csv(basic_file)
    adv_df = process_advanced_rushing_csv(adv_file)
    
    # Combine the data
    combined_df = combine_rushing_data(basic_df, adv_df)
    
    if combined_df.empty:
        print("[WARNING] No data after combining basic and advanced rushing stats")
        return pd.DataFrame()
    
    # Get week number from the combined DataFrame
    if 'week' in combined_df.columns:
        week_num = combined_df['week'].iloc[0]
    else:
        raise ValueError(f"Week column not found in combined data. Available columns: {list(combined_df.columns)}")
    
    if pd.isna(week_num):
        raise ValueError(f"Week number is null in combined data")
    
    # Convert to int (week_num might be 1.0, 2.0, etc.)
    week_num = int(float(week_num))
    week_id = get_week_id(db, season_id, week_num)
    
    # Get unique team abbreviations from the Tm column
    unique_teams = combined_df['Tm'].dropna().unique()
    if len(unique_teams) != 2:
        raise ValueError(f"Expected 2 teams in CSV files, found {len(unique_teams)}: {unique_teams}")
    
    team1_abrv, team2_abrv = unique_teams[0], unique_teams[1]
    game_id = get_game_id(db, season_id, week_id, team1_abrv, team2_abrv)
    
    # Process each row
    processed_rows = []
    for index, row in combined_df.iterrows():
        try:
            # Get player name and team from row
            player_name = row.get('Player', '')
            team_abrv = row.get('Tm', '')
            
            if not player_name or player_name.strip() == '':
                continue
                
            # Skip rows where Tm column is null, empty, or just whitespace
            if pd.isna(team_abrv) or not team_abrv or str(team_abrv).strip() == '':
                print(f"[INFO] Skipping player {player_name} - no team information (Tm column is empty)")
                continue
                
            # Get player_id and team_id
            plyr_id = get_player_id(db, player_name, team_abrv, season_id)
            team_id = get_team_id(db, team_abrv)
            
            # Create processed row
            processed_row = {
                'plyr_id': plyr_id,
                'week_id': week_id,
                'game_id': game_id,
                'season_id': season_id,
                'team_id': team_id
            }
            
            # Add all rushing stats (both basic and advanced)
            stat_columns = [
                'plyr_gm_rush_att', 'plyr_gm_rush_yds', 'plyr_gm_rush_yds_att', 'plyr_gm_rush_td', 'plyr_gm_rush_lng',
                'plyr_gm_rush_first_dwn', 'plyr_gm_rush_ybc', 'plyr_gm_rush_ybc_att',
                'plyr_gm_rush_yac', 'plyr_gm_rush_yac_att', 'plyr_gm_rush_brkn_tkl', 'plyr_gm_rush_brkn_tkl_att'
            ]
            
            for col in stat_columns:
                if col == 'plyr_gm_rush_yds_att':
                    # Skip - this will be calculated separately
                    continue
                    
                if col in combined_df.columns:
                    value = row.get(col)
                    
                    # Handle empty values specially - convert empty strings to None
                    if pd.notna(value) and value != '' and str(value).strip() != '':
                        try:
                            # Handle float columns (ratios)
                            if col in ['plyr_gm_rush_ybc_att', 'plyr_gm_rush_yac_att', 'plyr_gm_rush_brkn_tkl_att']:
                                processed_row[col] = float(value)
                            # Handle integer columns  
                            else:
                                processed_row[col] = int(float(value))
                        except (ValueError, TypeError):
                            processed_row[col] = None
                    else:
                        processed_row[col] = None
            
            # Calculate plyr_gm_rush_yds_att = plyr_gm_rush_yds / plyr_gm_rush_att
            attempts = processed_row.get('plyr_gm_rush_att')
            yards = processed_row.get('plyr_gm_rush_yds')
            
            if attempts is not None and yards is not None and attempts > 0:
                try:
                    processed_row['plyr_gm_rush_yds_att'] = round(yards / attempts, 4)
                except (ZeroDivisionError, TypeError):
                    processed_row['plyr_gm_rush_yds_att'] = None
            else:
                processed_row['plyr_gm_rush_yds_att'] = None
            
            processed_rows.append(processed_row)
            
        except Exception as e:
            print(f"[WARNING] Error processing row {index} for player {row.get('Player', 'Unknown')}: {e}")
            continue
    
    if processed_rows:
        processed_df = pd.DataFrame(processed_rows)
        processed_df = handle_null_values(processed_df)
        
        # Ensure proper data types for foreign key columns (these must not be null)
        fk_columns = ['plyr_id', 'week_id', 'game_id', 'season_id', 'team_id']
        for col in fk_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').astype('int64')
        
        # Handle integer columns that can be null (using nullable integer type)
        nullable_int_columns = ['plyr_gm_rush_att', 'plyr_gm_rush_yds', 'plyr_gm_rush_td', 'plyr_gm_rush_lng',
                               'plyr_gm_rush_first_dwn', 'plyr_gm_rush_ybc', 'plyr_gm_rush_yac', 'plyr_gm_rush_brkn_tkl']
        
        for col in nullable_int_columns:
            if col in processed_df.columns:
                # Convert to numeric, coercing errors to NaN
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                # For integer columns, round the values and convert to nullable int
                processed_df[col] = processed_df[col].round().astype('Int64')
        
        # Handle float columns (ratios that need precise decimal representation)
        float_columns = ['plyr_gm_rush_ybc_att', 'plyr_gm_rush_yac_att', 'plyr_gm_rush_brkn_tkl_att', 'plyr_gm_rush_yds_att']
        
        for col in float_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        return processed_df
    else:
        return pd.DataFrame()


def main():
    """Main function to process all CSV files."""
    
    print(f"Starting Player Game Rushing Data Import for {YEAR}")
    print(f"Processing weeks {WEEK_START} to {WEEK_END}")
    
    # Initialize database connection
    db = DatabaseConnector()
    if not db.connect():
        print("[FAIL] Failed to connect to database")
        return
    
    try:
        # Create table if it doesn't exist
        if not create_plyr_gm_rush_table(db):
            return
        
        # Get season_id
        season_id = get_season_id(db, YEAR)
        print(f"Season ID for {YEAR}: {season_id}")
        
        # Get CSV files to process
        csv_files = get_csv_files()
        if not csv_files:
            print("[WARNING] No CSV file pairs found to process")
            return
        
        print(f"Found {len(csv_files)} CSV file pairs to process")
        
        total_processed = 0
        total_inserted = 0
        
        # Process each file pair
        for week, basic_file, adv_file in csv_files:
            try:
                processed_df = process_csv_files(db, basic_file, adv_file, season_id)
                
                if not processed_df.empty:
                    success = batch_upsert_data(db, 'plyr_gm_rush', processed_df)
                    if success:
                        rows_in_files = len(processed_df)
                        total_processed += rows_in_files
                        total_inserted += rows_in_files
                        print(f"[OK] Week {week}: Processed {rows_in_files} player records")
                    else:
                        print(f"[FAIL] Week {week}: Failed to insert data")
                else:
                    print(f"[WARNING] Week {week}: No data to process")
                    
            except Exception as e:
                print(f"[ERROR] Error processing week {week} files: {e}")
                continue
        
        print(f"\nImport Summary:")
        print(f"   Total file pairs processed: {len(csv_files)}")
        print(f"   Total player records processed: {total_processed}")
        print(f"   Total records inserted/updated: {total_inserted}")
        
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
    finally:
        db.disconnect()


if __name__ == "__main__":
    main()