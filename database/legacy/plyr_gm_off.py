#!/usr/bin/env python3

import os
import sys
import pandas as pd
import glob
from pathlib import Path

# Add the parent directory to the Python path to import db_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_utils import DatabaseConnector, YEAR, WEEK_START, WEEK_END, batch_upsert_data, handle_null_values, standardize_team_name


def create_plyr_gm_off_table(db: DatabaseConnector) -> bool:
    """Create the plyr_gm_off table if it doesn't exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS plyr_gm_off (
        plyr_gm_off_id INT PRIMARY KEY AUTO_INCREMENT,
        plyr_id INT,
        week_id INT,
        game_id INT,
        season_id INT,
        plyr_gm_pass_cmp INT,
        plyr_gm_pass_att INT,
        plyr_gm_pass_yds INT,
        plyr_gm_pass_td INT,
        plyr_gm_pass_int INT,
        plyr_gm_pass_sk INT,
        plyr_gm_pass_sk_yds INT,
        plyr_gm_pass_lng INT,
        plyr_gm_pass_rtg FLOAT(7,4),
        plyr_gm_rush_att INT,
        plyr_gm_rush_yds INT,
        plyr_gm_rush_td INT,
        plyr_gm_rush_lng INT,
        plyr_gm_rec_tgt INT,
        plyr_gm_rec INT,
        plyr_gm_rec_yds INT,
        plyr_gm_rec_td INT,
        plyr_gm_rec_lng INT,
        plyr_gm_fmbl INT,
        plyr_gm_fl INT,
        FOREIGN KEY (plyr_id) REFERENCES plyr(plyr_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (game_id) REFERENCES nfl_game(game_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        UNIQUE KEY uk_player_game (plyr_id, game_id)
    );
    """
    
    try:
        success = db.execute_query(create_table_sql)
        if success:
            print("[OK] plyr_gm_off table created/verified successfully")
            return True
        else:
            print("[FAIL] Failed to create plyr_gm_off table")
            return False
    except Exception as e:
        print(f"[ERROR] Error creating plyr_gm_off table: {e}")
        return False


def get_season_id(db: DatabaseConnector, year: int) -> int:
    """Get season_id from nfl_season table based on year."""
    query = "SELECT season_id FROM nfl_season WHERE year = %s"
    result = db.fetch_all(query, (year,))
    if result:
        return result[0][0]
    else:
        raise ValueError(f"No season found for year {year}")


def get_week_id(db: DatabaseConnector, season_id: int, week_num: str) -> int:
    """Get week_id from nfl_week table based on season_id and week_num."""
    query = "SELECT week_id FROM nfl_week WHERE season_id = %s AND week_num = %s"
    result = db.fetch_all(query, (season_id, str(week_num)))
    if result:
        return result[0][0]
    else:
        raise ValueError(f"No week found for season {season_id}, week {week_num}")


def get_team_id(db: DatabaseConnector, team_name: str) -> int:
    """Get team_id from nfl_team table based on team name."""
    # First try by team name
    query = "SELECT team_id FROM nfl_team WHERE team_name = %s"
    result = db.fetch_all(query, (team_name,))
    if result:
        return result[0][0]
    
    # If not found, try by abbreviation (in case abbreviation is passed)
    query = "SELECT team_id FROM nfl_team WHERE abrv = %s"
    result = db.fetch_all(query, (team_name,))
    if result:
        return result[0][0]
    
    # If still not found, try by alternative abbreviation
    query = "SELECT team_id FROM nfl_team WHERE alt_abrv = %s"
    result = db.fetch_all(query, (team_name,))
    if result:
        return result[0][0]
    
    # If still not found, raise error
    raise ValueError(f"No team found for name/abbreviation {team_name}")


def get_game_id(db: DatabaseConnector, season_id: int, week_id: int, team1_abrv: str, team2_abrv: str) -> int:
    """Get game_id based on season_id, week_id and two team abbreviations."""
    
    # First get team IDs
    team1_id = get_team_id(db, team1_abrv)
    team2_id = get_team_id(db, team2_abrv)
    
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
        raise ValueError(f"No game found for season {season_id}, week {week_id}, teams {team1_abrv} vs {team2_abrv}")


def get_player_id(db: DatabaseConnector, player_name: str, team_abrv: str, season_id: int) -> int:
    """Get player_id using logic borrowed from temp.py."""
    
    # Generate name variations
    suffixes = ["II", "III", "IV", "Jr.", "Sr."]
    name_variations = [player_name] + [f"{player_name} {suffix}" for suffix in suffixes]
    placeholders = ', '.join(['%s'] * len(name_variations))
    
    if team_abrv and team_abrv.strip():
        query = f"""
        SELECT p.plyr_id, p.plyr_name, 'plyr' AS source, p.team_id, t.abrv, p.plyr_pos, p.plyr_age
        FROM plyr p
        JOIN nfl_team t ON p.team_id = t.team_id
        WHERE p.plyr_name IN ({placeholders}) AND (t.abrv = %s OR t.alt_abrv = %s) AND p.season_id = %s
        UNION
        SELECT mtp.plyr_id, mtp.plyr_name, 'multi_tm_plyr' AS source, 
               COALESCE(mtp.former_tm_id, mtp.first_tm_id) AS team_id, 
               COALESCE(t1.abrv, t2.abrv) AS abrv,
               mtp.plyr_pos, mtp.plyr_age
        FROM multi_tm_plyr mtp
        LEFT JOIN nfl_team t1 ON mtp.former_tm_id = t1.team_id
        LEFT JOIN nfl_team t2 ON mtp.first_tm_id = t2.team_id
        WHERE mtp.plyr_name IN ({placeholders}) AND (t1.abrv = %s OR t1.alt_abrv = %s OR t2.abrv = %s OR t2.alt_abrv = %s) AND mtp.season_id = %s
        """
        params = name_variations + [team_abrv, team_abrv, season_id] + name_variations + [team_abrv, team_abrv, team_abrv, team_abrv, season_id]
    else:
        query = f"""
        SELECT p.plyr_id, p.plyr_name, 'plyr' AS source, p.team_id, t.abrv, p.plyr_pos, p.plyr_age
        FROM plyr p
        JOIN nfl_team t ON p.team_id = t.team_id
        WHERE p.plyr_name IN ({placeholders}) AND p.season_id = %s
        """
        params = name_variations + [season_id]

    results = db.fetch_all(query, params)
    
    if len(results) == 1:
        return results[0][0]
    elif len(results) > 1:
        print(f"[WARNING] Multiple matches found for {player_name} ({team_abrv}). Using first match.")
        return results[0][0]
    else:
        raise ValueError(f"No player found for {player_name} ({team_abrv}) in season {season_id}")


def parse_csv_single_header(file_path: str) -> pd.DataFrame:
    """Parse CSV file with single header row and return DataFrame."""
    
    # Read CSV with single header row (new structure)
    df = pd.read_csv(file_path, header=0)
    return df


def process_passing_data(file_path: str) -> pd.DataFrame:
    """Process passing CSV file and return DataFrame with standardized columns."""
    df = parse_csv_single_header(file_path)
    
    # Create mapping for passing stats (new single-header format)
    column_mapping = {
        'Passing Cmp': 'plyr_gm_pass_cmp',
        'Passing Att': 'plyr_gm_pass_att',
        'Passing Yds': 'plyr_gm_pass_yds',
        'Passing TD': 'plyr_gm_pass_td',
        'Passing Int': 'plyr_gm_pass_int',
        'Passing Sk': 'plyr_gm_pass_sk',
        'Passing Yds.1': 'plyr_gm_pass_sk_yds',
        'Passing Lng': 'plyr_gm_pass_lng',
        'Passing Rate': 'plyr_gm_pass_rtg'
    }
    
    # Rename columns
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})
    
    return df


def process_rushing_data(file_path: str) -> pd.DataFrame:
    """Process rushing CSV file and return DataFrame with standardized columns."""
    df = parse_csv_single_header(file_path)
    
    # Create mapping for rushing stats (new single-header format)
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


def process_receiving_data(file_path: str) -> pd.DataFrame:
    """Process receiving CSV file and return DataFrame with standardized columns."""
    df = parse_csv_single_header(file_path)
    
    # Create mapping for receiving stats (new single-header format)
    column_mapping = {
        'Receiving Tgt': 'plyr_gm_rec_tgt',
        'Receiving Rec': 'plyr_gm_rec',
        'Receiving Yds': 'plyr_gm_rec_yds',
        'Receiving TD': 'plyr_gm_rec_td',
        'Receiving Lng': 'plyr_gm_rec_lng'
    }
    
    # Rename columns
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})
    
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


def combine_offensive_stats(passing_df: pd.DataFrame, rushing_df: pd.DataFrame, 
                           receiving_df: pd.DataFrame, fumbles_df: pd.DataFrame) -> pd.DataFrame:
    """Combine all offensive stats into a single DataFrame."""
    
    # Start with all players from all DataFrames
    all_players = set()
    for df in [passing_df, rushing_df, receiving_df, fumbles_df]:
        if not df.empty and 'Player' in df.columns:
            all_players.update(df['Player'].tolist())
    
    if not all_players:
        return pd.DataFrame()
    
    # Create base DataFrame with all players
    combined_df = pd.DataFrame({'Player': list(all_players)})
    
    # Get week, year, and game_id from any non-empty DataFrame
    for df in [passing_df, rushing_df, receiving_df, fumbles_df]:
        if not df.empty:
            if 'week' in df.columns:
                combined_df['week'] = df['week'].iloc[0]
            if 'year' in df.columns:
                combined_df['year'] = df['year'].iloc[0]
            if 'game_id' in df.columns:
                combined_df['game_id'] = df['game_id'].iloc[0]
            break
    
    # Merge each stat type
    stat_columns = {
        'passing': ['plyr_gm_pass_cmp', 'plyr_gm_pass_att', 'plyr_gm_pass_yds', 'plyr_gm_pass_td', 
                   'plyr_gm_pass_int', 'plyr_gm_pass_sk', 'plyr_gm_pass_sk_yds', 'plyr_gm_pass_lng', 'plyr_gm_pass_rtg'],
        'rushing': ['plyr_gm_rush_att', 'plyr_gm_rush_yds', 'plyr_gm_rush_td', 'plyr_gm_rush_lng'],
        'receiving': ['plyr_gm_rec_tgt', 'plyr_gm_rec', 'plyr_gm_rec_yds', 'plyr_gm_rec_td', 'plyr_gm_rec_lng'],
        'fumbles': ['plyr_gm_fmbl', 'plyr_gm_fl']
    }
    
    for stat_type, df in [('passing', passing_df), ('rushing', rushing_df), 
                         ('receiving', receiving_df), ('fumbles', fumbles_df)]:
        if not df.empty:
            # Keep only Player column and relevant stat columns
            keep_cols = ['Player'] + [col for col in stat_columns[stat_type] if col in df.columns]
            merge_df = df[keep_cols]
            combined_df = combined_df.merge(merge_df, on='Player', how='left')
    
    return combined_df


def process_game_files(db: DatabaseConnector, week: int, season_id: int) -> pd.DataFrame:
    """Process all offensive stat files for a specific week and return combined DataFrame."""
    
    base_dir = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\games"
    week_dir = os.path.join(base_dir, f"week_{week}.0")
    clean_dir = os.path.join(week_dir, "clean")
    
    if not os.path.exists(clean_dir):
        print(f"[WARNING] Clean directory not found: {clean_dir}")
        return pd.DataFrame()
    
    # Find all games in this week
    game_patterns = {
        'passing': "*_gm_plyr_passing_*.csv",
        'rushing': "*_gm_plyr_rushing_*.csv", 
        'receiving': "*_gm_plyr_receiving_*.csv",
        'fumbles': "*_gm_plyr_fumbles_*.csv"
    }
    
    # Get all unique game prefixes from clean directory
    all_files = glob.glob(os.path.join(clean_dir, "cleaned_*.csv"))
    game_prefixes = set()
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        # Extract game prefix (everything before the stat type, removing cleaned_ prefix)
        if '_gm_plyr_' in filename and filename.startswith('cleaned_'):
            # Remove 'cleaned_' prefix and extract game prefix
            no_prefix = filename[8:]  # Remove 'cleaned_'
            prefix = no_prefix.split('_gm_plyr_')[0]
            game_prefixes.add(prefix)
    
    all_processed_rows = []
    
    # Process each game
    for game_prefix in game_prefixes:
        try:
            # Find files for this game in clean directory
            game_files = {}
            for stat_type, pattern in game_patterns.items():
                specific_pattern = os.path.join(clean_dir, f"cleaned_{game_prefix}_gm_plyr_{stat_type}_*.csv")
                files = glob.glob(specific_pattern)
                if files:
                    game_files[stat_type] = files[0]
            
            if not game_files:
                continue
                
            print(f"Processing game: {game_prefix}")
            
            # Process each stat type
            stat_dfs = {}
            for stat_type, file_path in game_files.items():
                try:
                    if stat_type == 'passing':
                        stat_dfs[stat_type] = process_passing_data(file_path)
                    elif stat_type == 'rushing':
                        stat_dfs[stat_type] = process_rushing_data(file_path)
                    elif stat_type == 'receiving':
                        stat_dfs[stat_type] = process_receiving_data(file_path)
                    elif stat_type == 'fumbles':
                        stat_dfs[stat_type] = process_fumbles_data(file_path)
                except Exception as e:
                    print(f"[WARNING] Error processing {stat_type} file {file_path}: {e}")
                    stat_dfs[stat_type] = pd.DataFrame()
            
            # Combine stats for this game
            combined_df = combine_offensive_stats(
                stat_dfs.get('passing', pd.DataFrame()),
                stat_dfs.get('rushing', pd.DataFrame()), 
                stat_dfs.get('receiving', pd.DataFrame()),
                stat_dfs.get('fumbles', pd.DataFrame())
            )
            
            if combined_df.empty:
                continue
            
            # Get week number from one of the DataFrames
            week_num = None
            for df in stat_dfs.values():
                if not df.empty and 'week' in df.columns:
                    week_num = df['week'].iloc[0]
                    break
            
            if week_num is None:
                week_num = float(week)
            
            week_id = get_week_id(db, season_id, int(week_num))
            
            # Get unique teams from the game files to determine game_id
            unique_teams = set()
            for df in stat_dfs.values():
                if not df.empty and 'Tm' in df.columns:
                    unique_teams.update(df['Tm'].unique())
            
            if len(unique_teams) != 2:
                print(f"[WARNING] Expected 2 teams for game {game_prefix}, found {len(unique_teams)}: {unique_teams}")
                continue
            
            team1_abrv, team2_abrv = list(unique_teams)
            game_id = get_game_id(db, season_id, week_id, team1_abrv, team2_abrv)
            
            # Process each player row
            for index, row in combined_df.iterrows():
                try:
                    player_name = row.get('Player', '')
                    if not player_name or player_name.strip() == '':
                        continue
                    
                    # Get team for this player from one of the source DataFrames
                    team_abrv = None
                    for df in stat_dfs.values():
                        if not df.empty and 'Player' in df.columns and 'Tm' in df.columns:
                            player_rows = df[df['Player'] == player_name]
                            if not player_rows.empty:
                                team_abrv = player_rows['Tm'].iloc[0]
                                break
                    
                    if not team_abrv:
                        print(f"[WARNING] Could not determine team for player {player_name}")
                        continue
                    
                    # Get player_id
                    plyr_id = get_player_id(db, player_name, team_abrv, season_id)
                    
                    # Create processed row
                    processed_row = {
                        'plyr_id': plyr_id,
                        'week_id': week_id,
                        'game_id': game_id,
                        'season_id': season_id
                    }
                    
                    # Add all offensive stats
                    stat_columns = [
                        'plyr_gm_pass_cmp', 'plyr_gm_pass_att', 'plyr_gm_pass_yds', 'plyr_gm_pass_td', 
                        'plyr_gm_pass_int', 'plyr_gm_pass_sk', 'plyr_gm_pass_sk_yds', 'plyr_gm_pass_lng', 'plyr_gm_pass_rtg',
                        'plyr_gm_rush_att', 'plyr_gm_rush_yds', 'plyr_gm_rush_td', 'plyr_gm_rush_lng',
                        'plyr_gm_rec_tgt', 'plyr_gm_rec', 'plyr_gm_rec_yds', 'plyr_gm_rec_td', 'plyr_gm_rec_lng',
                        'plyr_gm_fmbl', 'plyr_gm_fl'
                    ]
                    
                    for col in stat_columns:
                        if col in combined_df.columns:
                            value = row.get(col)
                            if pd.notna(value) and value != '' and str(value).strip() != '':
                                try:
                                    if col == 'plyr_gm_pass_rtg':  # Float column
                                        processed_row[col] = float(value)
                                    else:  # Integer columns
                                        processed_row[col] = int(float(value))
                                except (ValueError, TypeError):
                                    processed_row[col] = None
                            else:
                                processed_row[col] = None
                    
                    all_processed_rows.append(processed_row)
                    
                except Exception as e:
                    print(f"[WARNING] Error processing player {row.get('Player', 'Unknown')} in game {game_prefix}: {e}")
                    continue
        
        except Exception as e:
            print(f"[ERROR] Error processing game {game_prefix}: {e}")
            continue
    
    if all_processed_rows:
        final_df = pd.DataFrame(all_processed_rows)
        final_df = handle_null_values(final_df)
        
        # Ensure proper data types
        int_columns = ['plyr_id', 'week_id', 'game_id', 'season_id', 'plyr_gm_pass_cmp', 'plyr_gm_pass_att', 
                      'plyr_gm_pass_yds', 'plyr_gm_pass_td', 'plyr_gm_pass_int', 'plyr_gm_pass_sk', 
                      'plyr_gm_pass_sk_yds', 'plyr_gm_pass_lng', 'plyr_gm_rush_att', 'plyr_gm_rush_yds', 
                      'plyr_gm_rush_td', 'plyr_gm_rush_lng', 'plyr_gm_rec_tgt', 'plyr_gm_rec', 
                      'plyr_gm_rec_yds', 'plyr_gm_rec_td', 'plyr_gm_rec_lng', 'plyr_gm_fmbl', 'plyr_gm_fl']
        
        float_columns = ['plyr_gm_pass_rtg']
        
        for col in int_columns:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        
        for col in float_columns:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        
        return final_df
    else:
        return pd.DataFrame()


def main():
    """Main function to process all CSV files."""
    
    print(f"Starting Player Game Offensive Data Import for {YEAR}")
    print(f"Processing weeks {WEEK_START} to {WEEK_END}")
    
    # Initialize database connection
    db = DatabaseConnector()
    if not db.connect():
        print("[FAIL] Failed to connect to database")
        return
    
    try:
        # Create table if it doesn't exist
        if not create_plyr_gm_off_table(db):
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
                processed_df = process_game_files(db, week, season_id)
                
                if not processed_df.empty:
                    success = batch_upsert_data(db, 'plyr_gm_off', processed_df)
                    if success:
                        rows_in_week = len(processed_df)
                        total_processed += rows_in_week
                        total_inserted += rows_in_week
                        print(f"[OK] Week {week}: Processed {rows_in_week} player records")
                    else:
                        print(f"[FAIL] Week {week}: Failed to insert data")
                else:
                    print(f"[WARNING] Week {week}: No data to process")
                    
            except Exception as e:
                print(f"[ERROR] Error processing week {week}: {e}")
                continue
        
        print(f"\nImport Summary:")
        print(f"   Total weeks processed: {WEEK_END - WEEK_START + 1}")
        print(f"   Total player records processed: {total_processed}")
        print(f"   Total records inserted/updated: {total_inserted}")
        
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
    finally:
        db.disconnect()


if __name__ == "__main__":
    main()