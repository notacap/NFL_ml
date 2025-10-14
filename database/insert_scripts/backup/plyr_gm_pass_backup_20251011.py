#!/usr/bin/env python3

import os
import sys
import pandas as pd
import glob
from pathlib import Path

# Add the parent directory to the Python path to import db_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_utils import DatabaseConnector, YEAR, WEEK_START, WEEK_END, batch_upsert_data, handle_null_values, get_season_id, get_week_id, get_player_id, get_game_id, get_team_id


def create_plyr_gm_pass_table(db: DatabaseConnector) -> bool:
    """Create the plyr_gm_pass table if it doesn't exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS plyr_gm_pass (
        adv_plyr_gm_pass_id INT PRIMARY KEY AUTO_INCREMENT,
        plyr_id INT,
        week_id INT,
        game_id INT,
        season_id INT,
        team_id INT,
        plyr_gm_pass_cmp INT,
        plyr_gm_pass_att INT,
        plyr_gm_pass_yds INT,
        plyr_gm_pass_td INT,
        plyr_gm_pass_int INT,
        plyr_gm_pass_sk INT,
        plyr_gm_pass_sk_yds INT,
        plyr_gm_pass_lng INT,
        plyr_gm_pass_rtg FLOAT(7,4),
        plyr_gm_pass_first_dwn INT,
        plyr_gm_pass_first_dwn_pct FLOAT(5,4),
        plyr_gm_pass_iay INT,
        plyr_gm_pass_iay_att FLOAT(7,4),
        plyr_gm_pass_cay INT,
        plyr_gm_pass_cay_cmp FLOAT(7,4),
        plyr_gm_pass_cay_att FLOAT(7,4),
        plyr_gm_pass_yac INT,
        plyr_gm_pass_yac_cmp FLOAT(7,4),
        plyr_gm_pass_drp INT,
        plyr_gm_pass_drp_pct FLOAT(5,4),
        plyr_gm_pass_off_tgt INT,
        plyr_gm_pass_off_tgt_pct FLOAT(5,4),
        plyr_gm_pass_bltz INT,
        plyr_gm_pass_hrry INT,
        plyr_gm_pass_hit INT,
        plyr_gm_pass_prss INT,
        plyr_gm_pass_prss_pct FLOAT(5,4),
        plyr_gm_pass_scrmbl_tgt INT,
        plyr_gm_pass_yds_scrmbl FLOAT(7,4),
        plyr_gm_pass_cmp_pct DECIMAL(7,4),
        plyr_gm_pass_td_pct DECIMAL(7,4),
        plyr_gm_pass_int_pct DECIMAL(7,4),
        plyr_gm_pass_yds_att DECIMAL(7,4),
        plyr_gm_pass_adj_yds_att DECIMAL(7,4),
        plyr_gm_pass_yds_cmp DECIMAL(7,4),
        plyr_gm_pass_sk_pct DECIMAL(7,4),
        plyr_gm_pass_net_yds_att DECIMAL(7,4),
        plyr_gm_pass_adj_net_yds_att DECIMAL(7,4),
        FOREIGN KEY (plyr_id) REFERENCES plyr(plyr_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (game_id) REFERENCES nfl_game(game_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        FOREIGN KEY (team_id) REFERENCES nfl_team(team_id),
        UNIQUE KEY uk_player_game (plyr_id, game_id)
    );
    """
    
    try:
        success = db.execute_query(create_table_sql)
        if success:
            print("[OK] plyr_gm_pass table created/verified successfully")
            return True
        else:
            print("[FAIL] Failed to create plyr_gm_pass table")
            return False
    except Exception as e:
        print(f"[ERROR] Error creating plyr_gm_pass table: {e}")
        return False


def convert_percentage(pct_str) -> float:
    """Convert percentage string (e.g., '10.0%') to decimal (0.100) with 4 decimal places."""
    if pd.isna(pct_str) or pct_str == '' or str(pct_str).strip() == '':
        return None
    try:
        # Remove % symbol and convert to decimal with 4 decimal places
        return round(float(str(pct_str).rstrip('%')) / 100.0, 4)
    except (ValueError, TypeError):
        return None


def calculate_passing_metrics(processed_row: dict) -> dict:
    """Calculate additional passing efficiency metrics based on basic stats."""
    
    # Get basic stats for calculations
    cmp = processed_row.get('plyr_gm_pass_cmp', 0) or 0
    att = processed_row.get('plyr_gm_pass_att', 0) or 0
    yds = processed_row.get('plyr_gm_pass_yds', 0) or 0
    td = processed_row.get('plyr_gm_pass_td', 0) or 0
    int_count = processed_row.get('plyr_gm_pass_int', 0) or 0
    sk = processed_row.get('plyr_gm_pass_sk', 0) or 0
    sk_yds = processed_row.get('plyr_gm_pass_sk_yds', 0) or 0
    
    # Initialize calculated metrics
    calculated_metrics = {}
    
    try:
        # Completion percentage: (cmp) / (att)
        if att > 0:
            calculated_metrics['plyr_gm_pass_cmp_pct'] = round(cmp / att, 4)
        else:
            calculated_metrics['plyr_gm_pass_cmp_pct'] = None
        
        # TD percentage: (td) / (att)
        if att > 0:
            calculated_metrics['plyr_gm_pass_td_pct'] = round(td / att, 4)
        else:
            calculated_metrics['plyr_gm_pass_td_pct'] = None
        
        # Interception percentage: (int) / (att)
        if att > 0:
            calculated_metrics['plyr_gm_pass_int_pct'] = round(int_count / att, 4)
        else:
            calculated_metrics['plyr_gm_pass_int_pct'] = None
        
        # Yards per attempt: (yds) / (att)
        if att > 0:
            calculated_metrics['plyr_gm_pass_yds_att'] = round(yds / att, 4)
        else:
            calculated_metrics['plyr_gm_pass_yds_att'] = None
        
        # Adjusted yards per attempt: (yds + 20 * td - 45 * int) / (att)
        if att > 0:
            adj_yds = yds + (20 * td) - (45 * int_count)
            calculated_metrics['plyr_gm_pass_adj_yds_att'] = round(adj_yds / att, 4)
        else:
            calculated_metrics['plyr_gm_pass_adj_yds_att'] = None
        
        # Yards per completion: (yds) / (cmp)
        if cmp > 0:
            calculated_metrics['plyr_gm_pass_yds_cmp'] = round(yds / cmp, 4)
        else:
            calculated_metrics['plyr_gm_pass_yds_cmp'] = None
        
        # Sack percentage: (sk) / (att + sk)
        total_dropbacks = att + sk
        if total_dropbacks > 0:
            calculated_metrics['plyr_gm_pass_sk_pct'] = round(sk / total_dropbacks, 4)
        else:
            calculated_metrics['plyr_gm_pass_sk_pct'] = None
        
        # Net yards per attempt: (yds - sk_yds) / (att + sk)
        if total_dropbacks > 0:
            net_yds = yds - sk_yds
            calculated_metrics['plyr_gm_pass_net_yds_att'] = round(net_yds / total_dropbacks, 4)
        else:
            calculated_metrics['plyr_gm_pass_net_yds_att'] = None
        
        # Adjusted net yards per attempt: (yds - sk_yds + (20 * td) - (45 * int)) / (att + sk)
        if total_dropbacks > 0:
            adj_net_yds = yds - sk_yds + (20 * td) - (45 * int_count)
            calculated_metrics['plyr_gm_pass_adj_net_yds_att'] = round(adj_net_yds / total_dropbacks, 4)
        else:
            calculated_metrics['plyr_gm_pass_adj_net_yds_att'] = None
            
    except (ZeroDivisionError, TypeError, ValueError) as e:
        print(f"[WARNING] Error calculating passing metrics: {e}")
        # Set all metrics to None if calculation fails
        for metric in ['plyr_gm_pass_cmp_pct', 'plyr_gm_pass_td_pct', 'plyr_gm_pass_int_pct', 
                      'plyr_gm_pass_yds_att', 'plyr_gm_pass_adj_yds_att', 'plyr_gm_pass_yds_cmp',
                      'plyr_gm_pass_sk_pct', 'plyr_gm_pass_net_yds_att', 'plyr_gm_pass_adj_net_yds_att']:
            calculated_metrics[metric] = None
    
    return calculated_metrics


def load_and_merge_passing_files(basic_file: str, adv_file: str) -> pd.DataFrame:
    """Load and merge basic and advanced passing files for a single game."""
    
    # Load basic passing file
    try:
        basic_df = pd.read_csv(basic_file, header=0)
        print(f"  Loaded basic file: {len(basic_df)} rows")
    except Exception as e:
        print(f"  [WARNING] Failed to load basic file {basic_file}: {e}")
        basic_df = pd.DataFrame()
    
    # Load advanced passing file
    try:
        adv_df = pd.read_csv(adv_file, header=0)
        print(f"  Loaded advanced file: {len(adv_df)} rows")
    except Exception as e:
        print(f"  [WARNING] Failed to load advanced file {adv_file}: {e}")
        adv_df = pd.DataFrame()
    
    if basic_df.empty and adv_df.empty:
        return pd.DataFrame()
    
    # Use advanced file as the primary source since it has more data
    if not adv_df.empty:
        merged_df = adv_df.copy()
        
        # Add missing columns from basic file if available
        if not basic_df.empty:
            # Merge basic stats that are missing from advanced file
            basic_only_cols = ['Passing TD', 'Passing Int', 'Passing Lng', 'Passing Rate', 'Passing Yds.1']
            for col in basic_only_cols:
                if col in basic_df.columns:
                    # Create a merge on Player and Tm
                    if 'Player' in basic_df.columns and 'Player' in merged_df.columns:
                        basic_subset = basic_df[['Player', 'Tm', col]].drop_duplicates()
                        merged_df = merged_df.merge(basic_subset, on=['Player', 'Tm'], how='left', suffixes=('', '_basic'))
    else:
        # Fall back to basic file only
        merged_df = basic_df.copy()
    
    return merged_df


def process_game_files(db: DatabaseConnector, week: int, season_id: int, interactive: bool = False) -> pd.DataFrame:
    """Process passing files for a specific week and return combined DataFrame."""
    
    base_dir = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\games"
    week_dir = os.path.join(base_dir, f"week_{week}.0")
    clean_dir = os.path.join(week_dir, "clean")
    
    if not os.path.exists(clean_dir):
        print(f"[WARNING] Clean directory not found: {clean_dir}")
        return pd.DataFrame()
    
    # Find all passing files in this week
    basic_pattern = "cleaned_*_gm_plyr_passing_*.csv"
    adv_pattern = "cleaned_*_gm_plyr_adv_passing_*.csv"
    
    basic_files = glob.glob(os.path.join(clean_dir, basic_pattern))
    adv_files = glob.glob(os.path.join(clean_dir, adv_pattern))
    
    if not basic_files and not adv_files:
        print(f"[WARNING] No passing files found in {clean_dir}")
        return pd.DataFrame()
    
    # Create game prefix mapping
    game_data = {}
    
    # Map basic files by game prefix
    for file_path in basic_files:
        filename = os.path.basename(file_path)
        if '_gm_plyr_passing_' in filename:
            prefix = filename.split('_gm_plyr_passing_')[0].replace('cleaned_', '')
            game_data[prefix] = game_data.get(prefix, {})
            game_data[prefix]['basic'] = file_path
    
    # Map advanced files by game prefix
    for file_path in adv_files:
        filename = os.path.basename(file_path)
        if '_gm_plyr_adv_passing_' in filename:
            prefix = filename.split('_gm_plyr_adv_passing_')[0].replace('cleaned_', '')
            game_data[prefix] = game_data.get(prefix, {})
            game_data[prefix]['adv'] = file_path
    
    all_processed_rows = []
    
    # Process each game
    for game_prefix, files in game_data.items():
        try:
            print(f"Processing game: {game_prefix}")
            
            # Load and merge files for this game
            basic_file = files.get('basic', '')
            adv_file = files.get('adv', '')
            
            if not basic_file and not adv_file:
                continue
            
            merged_df = load_and_merge_passing_files(basic_file, adv_file)
            
            if merged_df.empty:
                continue
            
            # Get week number from DataFrame
            week_num = None
            if 'week' in merged_df.columns:
                week_num = merged_df['week'].iloc[0]
            
            if week_num is None:
                week_num = float(week)
            
            week_id = get_week_id(db, season_id, int(week_num))
            
            # Get unique teams to determine game_id
            unique_teams = set()
            if 'Tm' in merged_df.columns:
                unique_teams.update(merged_df['Tm'].unique())
            
            if len(unique_teams) != 2:
                print(f"[WARNING] Expected 2 teams for game {game_prefix}, found {len(unique_teams)}: {unique_teams}")
                continue
            
            team1_abrv, team2_abrv = list(unique_teams)
            game_id = get_game_id(db, season_id, week_id, team1_abrv, team2_abrv)
            
            # Process each player row
            for index, row in merged_df.iterrows():
                try:
                    player_name = row.get('Player', '')
                    team_abrv = row.get('Tm', '')
                    
                    if not player_name or player_name.strip() == '':
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
                    
                    # Map basic passing stats (from both files, prefer advanced file values)
                    basic_mapping = {
                        'plyr_gm_pass_cmp': ['Cmp', 'Passing Cmp'],
                        'plyr_gm_pass_att': ['Att', 'Passing Att'],
                        'plyr_gm_pass_yds': ['Yds', 'Passing Yds'],
                        'plyr_gm_pass_td': ['Passing TD'],
                        'plyr_gm_pass_int': ['Passing Int'],
                        'plyr_gm_pass_sk': ['Sk', 'Passing Sk'],
                        'plyr_gm_pass_sk_yds': ['Passing Yds.1'],
                        'plyr_gm_pass_lng': ['Passing Lng'],
                        'plyr_gm_pass_rtg': ['Passing Rate']
                    }
                    
                    # Add basic passing stats
                    for db_col, csv_cols in basic_mapping.items():
                        value = None
                        for csv_col in csv_cols:
                            if csv_col in merged_df.columns:
                                value = row.get(csv_col)
                                break
                        
                        if pd.notna(value) and value != '' and str(value).strip() != '':
                            try:
                                if db_col == 'plyr_gm_pass_rtg':
                                    processed_row[db_col] = float(value)
                                else:
                                    processed_row[db_col] = int(float(value))
                            except (ValueError, TypeError):
                                processed_row[db_col] = None
                        else:
                            processed_row[db_col] = None
                    
                    # Map advanced passing stats (only from advanced file)
                    advanced_mapping = {
                        'plyr_gm_pass_first_dwn': '1D',
                        'plyr_gm_pass_first_dwn_pct': '1D%',
                        'plyr_gm_pass_iay': 'IAY',
                        'plyr_gm_pass_iay_att': 'IAY/PA',
                        'plyr_gm_pass_cay': 'CAY',
                        'plyr_gm_pass_cay_cmp': 'CAY/Cmp',
                        'plyr_gm_pass_cay_att': 'CAY/PA',
                        'plyr_gm_pass_yac': 'YAC',
                        'plyr_gm_pass_yac_cmp': 'YAC/Cmp',
                        'plyr_gm_pass_drp': 'Drops',
                        'plyr_gm_pass_drp_pct': 'Drop%',
                        'plyr_gm_pass_off_tgt': 'BadTh',
                        'plyr_gm_pass_off_tgt_pct': 'Bad%',
                        'plyr_gm_pass_bltz': 'Bltz',
                        'plyr_gm_pass_hrry': 'Hrry',
                        'plyr_gm_pass_hit': 'Hits',
                        'plyr_gm_pass_prss': 'Prss',
                        'plyr_gm_pass_prss_pct': 'Prss%',
                        'plyr_gm_pass_scrmbl_tgt': 'Scrm',
                        'plyr_gm_pass_yds_scrmbl': 'Yds/Scr'
                    }
                    
                    # Add advanced passing stats
                    for db_col, csv_col in advanced_mapping.items():
                        if csv_col in merged_df.columns:
                            value = row.get(csv_col)
                            
                            # Handle percentage columns specially
                            if db_col.endswith('_pct'):
                                processed_row[db_col] = convert_percentage(value)
                            else:
                                if pd.notna(value) and value != '' and str(value).strip() != '':
                                    try:
                                        if db_col in ['plyr_gm_pass_iay_att', 'plyr_gm_pass_cay_cmp', 'plyr_gm_pass_cay_att', 
                                                     'plyr_gm_pass_yac_cmp', 'plyr_gm_pass_yds_scrmbl']:
                                            processed_row[db_col] = float(value)
                                        else:
                                            processed_row[db_col] = int(float(value))
                                    except (ValueError, TypeError):
                                        processed_row[db_col] = None
                                else:
                                    processed_row[db_col] = None
                    
                    # Calculate additional passing efficiency metrics
                    calculated_metrics = calculate_passing_metrics(processed_row)
                    processed_row.update(calculated_metrics)
                    
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
        int_columns = ['plyr_id', 'week_id', 'game_id', 'season_id', 'team_id', 'plyr_gm_pass_cmp', 'plyr_gm_pass_att',
                      'plyr_gm_pass_yds', 'plyr_gm_pass_td', 'plyr_gm_pass_int', 'plyr_gm_pass_sk',
                      'plyr_gm_pass_sk_yds', 'plyr_gm_pass_lng', 'plyr_gm_pass_first_dwn',
                      'plyr_gm_pass_iay', 'plyr_gm_pass_cay', 'plyr_gm_pass_yac', 'plyr_gm_pass_drp',
                      'plyr_gm_pass_off_tgt', 'plyr_gm_pass_bltz', 'plyr_gm_pass_hrry', 
                      'plyr_gm_pass_hit', 'plyr_gm_pass_prss', 'plyr_gm_pass_scrmbl_tgt']
        
        float_columns = ['plyr_gm_pass_rtg', 'plyr_gm_pass_first_dwn_pct', 'plyr_gm_pass_iay_att',
                        'plyr_gm_pass_cay_cmp', 'plyr_gm_pass_cay_att', 'plyr_gm_pass_yac_cmp',
                        'plyr_gm_pass_drp_pct', 'plyr_gm_pass_off_tgt_pct', 'plyr_gm_pass_prss_pct',
                        'plyr_gm_pass_yds_scrmbl', 'plyr_gm_pass_cmp_pct', 'plyr_gm_pass_td_pct',
                        'plyr_gm_pass_int_pct', 'plyr_gm_pass_yds_att', 'plyr_gm_pass_adj_yds_att',
                        'plyr_gm_pass_yds_cmp', 'plyr_gm_pass_sk_pct', 'plyr_gm_pass_net_yds_att',
                        'plyr_gm_pass_adj_net_yds_att']
        
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
    
    print(f"Starting Player Game Passing Data Import for {YEAR}")
    print(f"Processing weeks {WEEK_START} to {WEEK_END}")
    print("[INFO] Interactive mode enabled - you will be prompted for player selection when multiple/no matches are found")

    # Initialize database connection
    db = DatabaseConnector()
    if not db.connect():
        print("[FAIL] Failed to connect to database")
        return
    
    try:
        # Create table if it doesn't exist
        if not create_plyr_gm_pass_table(db):
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
                    success = batch_upsert_data(db, 'plyr_gm_pass', processed_df)
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