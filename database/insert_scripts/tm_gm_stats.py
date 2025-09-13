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
    get_season_id, get_week_id, get_team_id, get_game_id,
    create_table_if_not_exists
)


def create_tm_gm_stats_table(db: DatabaseConnector) -> bool:
    """Create the tm_gm_stats table if it doesn't exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS tm_gm_stats (
        tm_gm_stats_id INT PRIMARY KEY AUTO_INCREMENT,
        team_id INT,
        week_id INT,
        game_id INT,
        season_id INT,
        tm_gm_first_dwn INT,
        tm_gm_rush_ply INT,
        tm_gm_rush_yds INT,
        tm_gm_rush_td INT,
        tm_gm_pass_cmp INT,
        tm_gm_pass_att INT,
        tm_gm_pass_yds INT,
        tm_gm_pass_td INT,
        tm_gm_pass_int INT,
        tm_gm_pass_sk INT,
        tm_gm_pass_sk_yds INT,
        tm_gm_net_pass_yds INT,
        tm_gm_tot_yds INT,
        tm_gm_fmb_tot INT,
        tm_gm_fmb_lost INT,
        tm_gm_to INT,
        tm_gm_pen INT,
        tm_gm_pen_yds_lost INT,
        tm_gm_third_dwn_conv INT,
        tm_gm_third_dwn_att INT,
        tm_gm_fourth_dwn_conv INT,
        tm_gm_fourth_dwn_att INT,
        tm_gm_top TIME,
        FOREIGN KEY (team_id) REFERENCES nfl_team(team_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (game_id) REFERENCES nfl_game(game_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        UNIQUE KEY uk_team_game (team_id, game_id)
    );
    """
    
    return create_table_if_not_exists(db, 'tm_gm_stats', create_table_sql)


def get_csv_files() -> list:
    """Get list of team game stats CSV files to process based on WEEK_START and WEEK_END."""
    base_dir = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\games"
    csv_files = []
    
    for week in range(WEEK_START, WEEK_END + 1):
        week_dir = os.path.join(base_dir, f"week_{week}.0")
        clean_dir = os.path.join(week_dir, "clean")
        if os.path.exists(clean_dir):
            # Find team stats files - pattern includes 'gm_tm_stats'
            pattern = os.path.join(clean_dir, "cleaned_*_gm_tm_stats_*.csv")
            week_files = glob.glob(pattern)
            
            for file_path in week_files:
                csv_files.append((week, file_path))
            
            if not week_files:
                print(f"[WARNING] No team stats files found in {clean_dir}")
        else:
            print(f"[WARNING] Clean directory not found: {clean_dir}")
    
    return sorted(csv_files)


def parse_dash_delimited_stat(value: str, expected_parts: int) -> list:
    """Parse dash-delimited statistics like 'Rush-Yds-TDs' into separate values."""
    if pd.isna(value) or value == '' or str(value).strip() == '':
        return [None] * expected_parts
    
    try:
        parts = str(value).strip().split('-')
        if len(parts) != expected_parts:
            print(f"[WARNING] Expected {expected_parts} parts in '{value}', got {len(parts)}")
            # Pad with None or truncate as needed
            while len(parts) < expected_parts:
                parts.append(None)
            parts = parts[:expected_parts]
        
        # Convert to integers, handle None values
        result = []
        for part in parts:
            if part is None or part == '':
                result.append(None)
            else:
                try:
                    result.append(int(part))
                except (ValueError, TypeError):
                    result.append(None)
        
        return result
    except Exception as e:
        print(f"[WARNING] Error parsing dash-delimited stat '{value}': {e}")
        return [None] * expected_parts


def convert_time_to_mysql_format(time_str: str) -> str:
    """Convert MM:SS format to HH:MM:SS for MySQL TIME type."""
    if pd.isna(time_str) or time_str == '' or str(time_str).strip() == '':
        return None
    
    try:
        time_str = str(time_str).strip()
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes, seconds = parts
                # Convert to HH:MM:SS format (assuming less than 1 hour)
                return f"00:{minutes.zfill(2)}:{seconds.zfill(2)}"
        return None
    except Exception as e:
        print(f"[WARNING] Error converting time '{time_str}': {e}")
        return None


def process_csv_file(db: DatabaseConnector, file_path: str, season_id: int) -> pd.DataFrame:
    """Process a single team game stats CSV file and return DataFrame."""
    
    print(f"Processing file: {os.path.basename(file_path)}")
    
    try:
        df = pd.read_csv(file_path, header=0)
        
        if df.empty:
            print(f"[WARNING] Empty CSV file: {file_path}")
            return pd.DataFrame()
        
        # Get metadata from the first row (all rows should have same values for these)
        week_num = df.iloc[0]['week'] if 'week' in df.columns else None
        year = df.iloc[0]['year'] if 'year' in df.columns else YEAR
        csv_game_id = df.iloc[0]['game_id'] if 'game_id' in df.columns else None
        
        if pd.isna(week_num):
            raise ValueError(f"Week number is null in file: {file_path}")
        
        week_num = int(float(week_num))
        week_id = get_week_id(db, season_id, week_num)
        
        # Get team abbreviations from column headers (columns 1 and 2)
        columns = df.columns.tolist()
        if len(columns) < 3:
            raise ValueError(f"Insufficient columns in file: {file_path}")
        
        team1_abrv = columns[1]  # Second column
        team2_abrv = columns[2]  # Third column
        
        # Get team IDs and game ID
        team1_id = get_team_id(db, team1_abrv)
        team2_id = get_team_id(db, team2_abrv)
        game_id = get_game_id(db, season_id, week_id, team1_abrv, team2_abrv)
        
        print(f"  Game: {team1_abrv} vs {team2_abrv} (Week {week_num})")
        print(f"  Game ID: {game_id}, Team IDs: {team1_id}, {team2_id}")
        
        # Create records for both teams
        processed_rows = []
        
        for team_idx, (team_abrv, team_id) in enumerate([(team1_abrv, team1_id), (team2_abrv, team2_id)], 1):
            team_record = {
                'team_id': team_id,
                'week_id': week_id,
                'game_id': game_id,
                'season_id': season_id
            }
            
            # Process each stat category (row in the CSV)
            for _, row in df.iterrows():
                stat_category = row.iloc[0]  # First column contains stat name
                team_value = row.iloc[team_idx]  # Team's value in corresponding column
                
                # Map stat categories to database columns
                if stat_category == 'First Downs':
                    if pd.notna(team_value):
                        try:
                            team_record['tm_gm_first_dwn'] = int(team_value)
                        except (ValueError, TypeError):
                            team_record['tm_gm_first_dwn'] = None
                
                elif stat_category == 'Rush-Yds-TDs':
                    rush_parts = parse_dash_delimited_stat(team_value, 3)
                    team_record['tm_gm_rush_ply'] = rush_parts[0]
                    team_record['tm_gm_rush_yds'] = rush_parts[1]
                    team_record['tm_gm_rush_td'] = rush_parts[2]
                
                elif stat_category == 'Cmp-Att-Yd-TD-INT':
                    pass_parts = parse_dash_delimited_stat(team_value, 5)
                    team_record['tm_gm_pass_cmp'] = pass_parts[0]
                    team_record['tm_gm_pass_att'] = pass_parts[1]
                    team_record['tm_gm_pass_yds'] = pass_parts[2]
                    team_record['tm_gm_pass_td'] = pass_parts[3]
                    team_record['tm_gm_pass_int'] = pass_parts[4]
                
                elif stat_category == 'Sacked-Yards':
                    sack_parts = parse_dash_delimited_stat(team_value, 2)
                    team_record['tm_gm_pass_sk'] = sack_parts[0]
                    team_record['tm_gm_pass_sk_yds'] = sack_parts[1]
                
                elif stat_category == 'Net Pass Yards':
                    if pd.notna(team_value):
                        try:
                            team_record['tm_gm_net_pass_yds'] = int(team_value)
                        except (ValueError, TypeError):
                            team_record['tm_gm_net_pass_yds'] = None
                
                elif stat_category == 'Total Yards':
                    if pd.notna(team_value):
                        try:
                            team_record['tm_gm_tot_yds'] = int(team_value)
                        except (ValueError, TypeError):
                            team_record['tm_gm_tot_yds'] = None
                
                elif stat_category == 'Fumbles-Lost':
                    fumble_parts = parse_dash_delimited_stat(team_value, 2)
                    team_record['tm_gm_fmb_tot'] = fumble_parts[0]
                    team_record['tm_gm_fmb_lost'] = fumble_parts[1]
                
                elif stat_category == 'Turnovers':
                    if pd.notna(team_value):
                        try:
                            team_record['tm_gm_to'] = int(team_value)
                        except (ValueError, TypeError):
                            team_record['tm_gm_to'] = None
                
                elif stat_category == 'Penalties-Yards':
                    penalty_parts = parse_dash_delimited_stat(team_value, 2)
                    team_record['tm_gm_pen'] = penalty_parts[0]
                    team_record['tm_gm_pen_yds_lost'] = penalty_parts[1]
                
                elif stat_category == 'Third Down Conv.':
                    third_down_parts = parse_dash_delimited_stat(team_value, 2)
                    team_record['tm_gm_third_dwn_conv'] = third_down_parts[0]
                    team_record['tm_gm_third_dwn_att'] = third_down_parts[1]
                
                elif stat_category == 'Fourth Down Conv.':
                    fourth_down_parts = parse_dash_delimited_stat(team_value, 2)
                    team_record['tm_gm_fourth_dwn_conv'] = fourth_down_parts[0]
                    team_record['tm_gm_fourth_dwn_att'] = fourth_down_parts[1]
                
                elif stat_category == 'Time of Possession':
                    team_record['tm_gm_top'] = convert_time_to_mysql_format(team_value)
            
            processed_rows.append(team_record)
        
        if processed_rows:
            processed_df = pd.DataFrame(processed_rows)
            processed_df = handle_null_values(processed_df)
            
            # Ensure proper data types for foreign key columns
            fk_columns = ['team_id', 'week_id', 'game_id', 'season_id']
            for col in fk_columns:
                if col in processed_df.columns:
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').astype('int64')
            
            # Handle integer columns that can be null
            nullable_int_columns = ['tm_gm_first_dwn', 'tm_gm_rush_ply', 'tm_gm_rush_yds', 'tm_gm_rush_td',
                                   'tm_gm_pass_cmp', 'tm_gm_pass_att', 'tm_gm_pass_yds', 'tm_gm_pass_td',
                                   'tm_gm_pass_int', 'tm_gm_pass_sk', 'tm_gm_pass_sk_yds', 'tm_gm_net_pass_yds',
                                   'tm_gm_tot_yds', 'tm_gm_fmb_tot', 'tm_gm_fmb_lost', 'tm_gm_to',
                                   'tm_gm_pen', 'tm_gm_pen_yds_lost', 'tm_gm_third_dwn_conv', 'tm_gm_third_dwn_att',
                                   'tm_gm_fourth_dwn_conv', 'tm_gm_fourth_dwn_att']
            
            for col in nullable_int_columns:
                if col in processed_df.columns:
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                    processed_df[col] = processed_df[col].round().astype('Int64')
            
            return processed_df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"[ERROR] Error processing file {file_path}: {e}")
        return pd.DataFrame()


def main():
    """Main function to process all team game stats CSV files."""
    
    print(f"Starting Team Game Statistics Import for {YEAR}")
    print(f"Processing weeks {WEEK_START} to {WEEK_END}")
    
    # Initialize database connection
    db = DatabaseConnector()
    if not db.connect():
        print("[FAIL] Failed to connect to database")
        return
    
    try:
        # Create table if it doesn't exist
        if not create_tm_gm_stats_table(db):
            return
        
        # Get season_id
        season_id = get_season_id(db, YEAR)
        print(f"Season ID for {YEAR}: {season_id}")
        
        # Get CSV files to process
        csv_files = get_csv_files()
        if not csv_files:
            print("[WARNING] No team game stats CSV files found to process")
            return
        
        print(f"Found {len(csv_files)} CSV files to process")
        
        total_processed = 0
        total_inserted = 0
        
        # Process each CSV file
        for week, file_path in csv_files:
            try:
                processed_df = process_csv_file(db, file_path, season_id)
                
                if not processed_df.empty:
                    success = batch_upsert_data(db, 'tm_gm_stats', processed_df)
                    if success:
                        rows_in_file = len(processed_df)
                        total_processed += rows_in_file
                        total_inserted += rows_in_file
                        print(f"[OK] Week {week}: Processed {rows_in_file} team records")
                    else:
                        print(f"[FAIL] Week {week}: Failed to insert data")
                else:
                    print(f"[WARNING] Week {week}: No data to process")
                    
            except Exception as e:
                print(f"[ERROR] Error processing week {week} file: {e}")
                print(f"   File: {os.path.basename(file_path)}")
                continue
        
        print(f"\nTeam Game Statistics Import Summary:")
        print(f"   Total CSV files processed: {len(csv_files)}")
        print(f"   Total team records processed: {total_processed}")
        print(f"   Total records inserted/updated: {total_inserted}")
        
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
    finally:
        db.disconnect()


if __name__ == "__main__":
    main()