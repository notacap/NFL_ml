#!/usr/bin/env python3

import os
import sys
import pandas as pd
import glob
from pathlib import Path

# Add the parent directory to the Python path to import db_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_utils import DatabaseConnector, YEAR, WEEK_START, WEEK_END, batch_upsert_data, handle_null_values, get_season_id, get_week_id, get_team_id, get_game_id


def create_tm_gm_exp_pts_table(db: DatabaseConnector) -> bool:
    """Create the tm_gm_exp_pts table if it doesn't exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS tm_gm_exp_pts (
        tm_gm_exp_pts_id INT PRIMARY KEY AUTO_INCREMENT,
        team_id INT,
        week_id INT,
        game_id INT,
        season_id INT,
        tm_gm_exp FLOAT(7,4),
        tm_gm_off_exp FLOAT(7,4),
        tm_gm_pass_exp FLOAT(7,4),
        tm_gm_rush_exp FLOAT(7,4),
        tm_gm_tovr_exp FLOAT(7,4),
        tm_gm_def_exp FLOAT(7,4),
        tm_gm_def_pass_exp FLOAT(7,4),
        tm_gm_def_rush_exp FLOAT(7,4),
        tm_gm_def_tovr_exp FLOAT(7,4),
        tm_gm_st_exp FLOAT(7,4),
        tm_gm_st_ko_exp FLOAT(7,4),
        tm_gm_st_kr_exp FLOAT(7,4),
        tm_gm_st_p_exp FLOAT(7,4),
        tm_gm_st_pr_exp FLOAT(7,4),
        tm_gm_st_fgxp_exp FLOAT(7,4),
        FOREIGN KEY (team_id) REFERENCES nfl_team(team_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (game_id) REFERENCES nfl_game(game_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        UNIQUE KEY uk_team_game (team_id, game_id)
    );
    """
    
    try:
        db.execute_query(create_table_sql)
        print("SUCCESS: tm_gm_exp_pts table created successfully")
        return True
    except Exception as e:
        print(f"ERROR: Creating tm_gm_exp_pts table: {e}")
        return False


def get_expected_points_csv_files() -> list:
    """Get list of CSV files to process based on WEEK_START and WEEK_END."""
    base_dir = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\games"
    csv_files = []
    
    for week in range(WEEK_START, WEEK_END + 1):
        week_dir = os.path.join(base_dir, f"week_{week}.0")
        clean_dir = os.path.join(week_dir, "clean")
        if os.path.exists(clean_dir):
            pattern = os.path.join(clean_dir, "cleaned_*_gm_expected_pts_*.csv")
            week_files = glob.glob(pattern)
            for file_path in week_files:
                csv_files.append((week, file_path))
        else:
            print(f"WARNING: Clean directory not found: {clean_dir}")
    
    print(f"Found {len(csv_files)} expected points CSV files")
    return sorted(csv_files)


def get_team_abbreviation_from_partial_name(db: DatabaseConnector, partial_team_name: str) -> str:
    """Convert partial team name (like 'Cardinals') to abbreviation (like 'ARI') by matching team_name column."""
    try:
        # Use LIKE to find team_name that contains the partial name
        query = "SELECT abrv FROM nfl_team WHERE team_name LIKE %s"
        result = db.fetch_all(query, (f"%{partial_team_name}%",))
        if result:
            return result[0][0]
        else:
            print(f"WARNING: Could not find team abbreviation for partial team name: {partial_team_name}")
            return None
    except Exception as e:
        print(f"ERROR: Error looking up team abbreviation for {partial_team_name}: {e}")
        return None


def process_expected_points_csv_file(db: DatabaseConnector, file_path: str, season_id: int, week: int) -> pd.DataFrame:
    """Process a single team expected points CSV file and return DataFrame for upsert."""
    try:
        df = pd.read_csv(file_path)
        
        if df.empty:
            print(f"WARNING: Skipping empty file: {file_path}")
            return pd.DataFrame()
        
        # Get week_id using the passed week number
        week_id = get_week_id(db, season_id, week)
        
        # Get unique team names from the Tm column
        unique_teams = df['Tm'].dropna().unique()
        if len(unique_teams) != 2:
            print(f"WARNING: Expected 2 teams in CSV file, found {len(unique_teams)}: {unique_teams}")
            return pd.DataFrame()
        
        # Convert partial team names to abbreviations for game_id lookup
        team1_name, team2_name = unique_teams[0], unique_teams[1]
        team1_abrv = get_team_abbreviation_from_partial_name(db, team1_name)
        team2_abrv = get_team_abbreviation_from_partial_name(db, team2_name)
        
        if not team1_abrv or not team2_abrv:
            print(f"WARNING: Could not find team abbreviations for {team1_name} or {team2_name}")
            return pd.DataFrame()
        
        # Get game_id using team abbreviations
        game_id = get_game_id(db, season_id, week_id, team1_abrv, team2_abrv)
        
        processed_data = []
        
        for _, row in df.iterrows():
            partial_team_name = row['Tm']  # e.g., "Cardinals", "Falcons"
            
            # Get team_id by matching partial team name in database team_name column
            try:
                query = "SELECT team_id FROM nfl_team WHERE team_name LIKE %s"
                result = db.fetch_all(query, (f"%{partial_team_name}%",))
                if result:
                    team_id = result[0][0]
                else:
                    print(f"WARNING: Could not find team_id for partial team name: {partial_team_name}")
                    continue
            except Exception as e:
                print(f"ERROR: Error looking up team_id for {partial_team_name}: {e}")
                continue
            
            # Map CSV columns to database columns
            record = {
                'team_id': team_id,
                'week_id': week_id,
                'game_id': game_id,
                'season_id': season_id,
                'tm_gm_exp': row['Total'],
                'tm_gm_off_exp': row['Offense Tot'],
                'tm_gm_pass_exp': row['Offense Pass'],
                'tm_gm_rush_exp': row['Offense Rush'],
                'tm_gm_tovr_exp': row['Offense TOvr'],
                'tm_gm_def_exp': row['Defense Tot'],
                'tm_gm_def_pass_exp': row['Defense Pass'],
                'tm_gm_def_rush_exp': row['Defense Rush'],
                'tm_gm_def_tovr_exp': row['Defense TOvr'],
                'tm_gm_st_exp': row['Special Teams Tot'],
                'tm_gm_st_ko_exp': row['Special Teams KO'],
                'tm_gm_st_kr_exp': row['Special Teams KR'],
                'tm_gm_st_p_exp': row['Special Teams P'],
                'tm_gm_st_pr_exp': row['Special Teams PR'],
                'tm_gm_st_fgxp_exp': row['Special Teams FG/XP']
            }
            
            processed_data.append(record)
        
        if processed_data:
            result_df = pd.DataFrame(processed_data)
            # Handle null values in the DataFrame
            result_df = handle_null_values(result_df)
            print(f"SUCCESS: Processed {len(processed_data)} team records from: {os.path.basename(file_path)}")
            return result_df
        else:
            print(f"WARNING: No valid team records processed from: {file_path}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"ERROR: Processing file {file_path}: {e}")
        return pd.DataFrame()


def main():
    """Main execution function."""
    print("Starting Team Game Expected Points data import...")
    
    # Initialize database connection
    db = DatabaseConnector()
    
    if not db.connect():
        print("ERROR: Failed to connect to database")
        return
    
    try:
        # Create table
        if not create_tm_gm_exp_pts_table(db):
            return
        
        # Get season_id
        season_id = get_season_id(db, YEAR)
        if not season_id:
            print(f"ERROR: Could not find season_id for year {YEAR}")
            return
        
        # Get CSV files
        csv_files = get_expected_points_csv_files()
        if not csv_files:
            print("ERROR: No expected points CSV files found")
            return
        
        # Process files and collect data
        all_data = []
        
        for week, file_path in csv_files:
            file_data = process_expected_points_csv_file(db, file_path, season_id, week)
            if not file_data.empty:
                all_data.append(file_data)
        
        if not all_data:
            print("ERROR: No data processed from CSV files")
            return
        
        # Combine all data
        final_df = pd.concat(all_data, ignore_index=True)
        
        print(f"Final DataFrame shape: {final_df.shape}")
        
        # Upsert data using batch operation
        success = batch_upsert_data(
            db=db,
            table_name='tm_gm_exp_pts',
            df=final_df
        )
        
        if success:
            print(f"SUCCESS: Upserted {len(final_df)} team game expected points records")
        else:
            print("ERROR: Failed to upsert team game expected points data")
    
    finally:
        db.disconnect()


if __name__ == "__main__":
    main()