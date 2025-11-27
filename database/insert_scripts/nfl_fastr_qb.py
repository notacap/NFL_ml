#!/usr/bin/env python3
"""
nfl_fastr_qb.py - NFL FastR Quarterback Advanced Metrics Upsert Script

Processes fastr_qb.csv and ngs_qb.csv files to populate nfl_fastr_qb table.
Merges data from both sources on player_display_name, season, week, and team.

Source Directory: web_scrape/scraped_data/nflfastR/quarterback_data
"""

import os
import sys
import glob
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

# Add the parent directory to the Python path to import db_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_utils import (
    DatabaseConnector,
    batch_upsert_data,
    handle_null_values,
    get_season_id,
    get_week_id,
    get_player_id,
    create_table_if_not_exists
)

# Static source directory (unlike other scripts, we don't use YEAR/WEEK variables)
SOURCE_DIR = r"C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\nflfastR\quarterback_data"
PROCESSED_DIR = os.path.join(SOURCE_DIR, "processed")


def create_nfl_fastr_qb_table(db: DatabaseConnector) -> bool:
    """Create the nfl_fastr_qb table if it doesn't exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS nfl_fastr_qb (
        nfl_fastr_qb_id INT AUTO_INCREMENT PRIMARY KEY,
        plyr_id INT,
        season_id INT,
        week_id INT,
        plyr_gm_pass_avg_ttt DECIMAL(10,4),
        plyr_gm_pass_aggressiveness DECIMAL(7,4),
        plyr_gm_pass_avg_intended_ay DECIMAL(7,4),
        plyr_gm_pass_avg_ay_to_sticks DECIMAL(7,4),
        plyr_gm_pass_expected_cmp_pct DECIMAL(7,4),
        plyr_gm_pass_cmp_pct_above_expectation DECIMAL(7,4),
        plyr_gm_pass_epa DECIMAL(7,4),
        plyr_gm_pass_pacr DECIMAL(7,4),
        plyr_gm_pass_cpoe DECIMAL(7,4),
        CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uk_plyr_season (plyr_id, season_id, week_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id),
        FOREIGN KEY (plyr_id) REFERENCES plyr(plyr_id)
    );
    """
    return create_table_if_not_exists(db, 'nfl_fastr_qb', create_table_sql)


def get_most_recent_csv_files(directory: str) -> tuple:
    """
    Returns paths to the 2 most recently created CSV files.
    One should match pattern *fastr*.csv, other *ngs*.csv

    Returns:
        tuple: (fastr_file_path, ngs_file_path) or (None, None) if not found
    """
    all_csv = glob.glob(os.path.join(directory, "*.csv"))

    if len(all_csv) < 2:
        return None, None

    # Sort by creation time (most recent first)
    all_csv.sort(key=os.path.getctime, reverse=True)

    fastr_file = None
    ngs_file = None

    for f in all_csv:
        filename = os.path.basename(f).lower()
        if 'fastr' in filename and fastr_file is None:
            fastr_file = f
        elif 'ngs' in filename and ngs_file is None:
            ngs_file = f

        if fastr_file and ngs_file:
            break

    return fastr_file, ngs_file


def merge_source_files(fastr_df: pd.DataFrame, ngs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge fastr and ngs dataframes on player_display_name, season, week, and team.

    Note: fastr uses 'team' column, ngs uses 'team_abbr' column
    """
    # Standardize team column names for merge
    fastr_df = fastr_df.rename(columns={'team': 'team_abrv'})
    ngs_df = ngs_df.rename(columns={'team_abbr': 'team_abrv'})

    # Merge on common keys
    merged_df = pd.merge(
        fastr_df,
        ngs_df,
        on=['player_display_name', 'season', 'week', 'team_abrv'],
        how='inner',
        suffixes=('_fastr', '_ngs')
    )

    return merged_df


def process_merged_data(db: DatabaseConnector, merged_df: pd.DataFrame) -> dict:
    """
    Process merged dataframe and prepare records for database insertion.

    Returns:
        dict: Summary statistics of processing
    """
    summary = {
        'total_records': len(merged_df),
        'skipped_week_zero': 0,
        'skipped_week_over_18': 0,
        'processed': 0,
        'unidentified_players': []
    }

    processed_rows = []

    # Group by season for efficient season_id lookups
    for season in merged_df['season'].unique():
        try:
            season_id = get_season_id(db, int(season))
        except ValueError as e:
            print(f"[WARNING] Season {season} not found in database: {e}")
            continue

        season_df = merged_df[merged_df['season'] == season]

        for _, row in season_df.iterrows():
            # Skip rows where week = 0 (season aggregates)
            if row['week'] == 0:
                summary['skipped_week_zero'] += 1
                continue

            # Skip rows where week > 18 (playoff weeks not in regular season)
            if row['week'] > 18:
                summary['skipped_week_over_18'] += 1
                continue

            try:
                # Get week_id
                week_num = int(row['week'])
                week_id = get_week_id(db, season_id, week_num)

                # Get player_id
                player_name = row['player_display_name']
                team_abrv = row['team_abrv']

                try:
                    plyr_id = get_player_id(
                        db,
                        player_name,
                        team_abrv,
                        season_id,
                        position='QB',
                        interactive=True
                    )

                    # Check if user chose to skip (returns 0)
                    if plyr_id == 0:
                        summary['unidentified_players'].append({
                            'player_name': player_name,
                            'team': team_abrv,
                            'season': season,
                            'week': week_num
                        })
                        continue

                except ValueError as e:
                    summary['unidentified_players'].append({
                        'player_name': player_name,
                        'team': team_abrv,
                        'season': season,
                        'week': week_num
                    })
                    continue

                # Build record with column mappings
                record = {
                    'plyr_id': plyr_id,
                    'season_id': season_id,
                    'week_id': week_id,
                    # From fastr_qb.csv
                    'plyr_gm_pass_epa': safe_decimal(row.get('passing_epa')),
                    'plyr_gm_pass_pacr': safe_decimal(row.get('pacr')),
                    'plyr_gm_pass_cpoe': safe_decimal(row.get('passing_cpoe')),
                    # From ngs_qb.csv
                    'plyr_gm_pass_avg_ttt': safe_decimal(row.get('avg_time_to_throw')),
                    'plyr_gm_pass_aggressiveness': safe_decimal(row.get('aggressiveness')),
                    'plyr_gm_pass_avg_intended_ay': safe_decimal(row.get('avg_intended_air_yards')),
                    'plyr_gm_pass_avg_ay_to_sticks': safe_decimal(row.get('avg_air_yards_to_sticks')),
                    'plyr_gm_pass_expected_cmp_pct': safe_decimal(row.get('expected_completion_percentage')),
                    'plyr_gm_pass_cmp_pct_above_expectation': safe_decimal(row.get('completion_percentage_above_expectation'))
                }

                processed_rows.append(record)
                summary['processed'] += 1

            except ValueError as e:
                print(f"[WARNING] Error processing row for {row.get('player_display_name', 'Unknown')}: {e}")
                continue

    # Convert to DataFrame for batch upsert
    if processed_rows:
        result_df = pd.DataFrame(processed_rows)
        result_df = handle_null_values(result_df)

        # Ensure foreign key columns are integers
        for col in ['plyr_id', 'season_id', 'week_id']:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').astype('int64')

        summary['dataframe'] = result_df
    else:
        summary['dataframe'] = pd.DataFrame()

    return summary


def safe_decimal(value) -> float:
    """Convert value to float, handling NA/NaN/None values."""
    if pd.isna(value) or value is None or value == 'NA':
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def move_files_to_processed(fastr_file: str, ngs_file: str) -> bool:
    """Move processed files to the processed subdirectory."""
    try:
        # Create processed directory if it doesn't exist
        os.makedirs(PROCESSED_DIR, exist_ok=True)

        # Move files
        fastr_dest = os.path.join(PROCESSED_DIR, os.path.basename(fastr_file))
        ngs_dest = os.path.join(PROCESSED_DIR, os.path.basename(ngs_file))

        shutil.move(fastr_file, fastr_dest)
        shutil.move(ngs_file, ngs_dest)

        print(f"[OK] Moved {os.path.basename(fastr_file)} to processed/")
        print(f"[OK] Moved {os.path.basename(ngs_file)} to processed/")
        return True

    except Exception as e:
        print(f"[WARNING] Failed to move files to processed: {e}")
        return False


def print_summary(summary: dict, inserted: int, updated: int):
    """Print processing summary to terminal."""
    print("\n" + "=" * 50)
    print("=== Processing Summary ===")
    print("=" * 50)
    print(f"Total records in source files: {summary['total_records']}")
    print(f"Records skipped (week = 0): {summary['skipped_week_zero']}")
    print(f"Records skipped (week > 18): {summary['skipped_week_over_18']}")
    print(f"Records successfully processed: {summary['processed']}")
    print(f"Records inserted: {inserted}")
    print(f"Records updated: {updated}")

    if summary['unidentified_players']:
        print(f"\nRecords with unidentified plyr_id ({len(summary['unidentified_players'])}):")
        for player in summary['unidentified_players']:
            print(f"  - {player['player_name']} | Team: {player['team']} | Season: {player['season']} | Week: {player['week']}")
    else:
        print("\nAll players successfully identified!")
    print("=" * 50)


def main():
    """Main function to process FastR QB data."""
    print("=" * 60)
    print("NFL FastR Quarterback Advanced Metrics Import")
    print("=" * 60)
    print(f"Source directory: {SOURCE_DIR}")
    print("[INFO] Interactive mode enabled - you will be prompted for player selection when multiple/no matches are found")

    # Check source directory exists
    if not os.path.exists(SOURCE_DIR):
        print(f"[ERROR] Source directory does not exist: {SOURCE_DIR}")
        return

    # Get most recent CSV files
    fastr_file, ngs_file = get_most_recent_csv_files(SOURCE_DIR)

    if not fastr_file or not ngs_file:
        print("[ERROR] Could not find both fastr_*.csv and ngs_*.csv files in source directory")
        return

    print(f"\nProcessing files:")
    print(f"  FastR: {os.path.basename(fastr_file)}")
    print(f"  NGS:   {os.path.basename(ngs_file)}")

    # Load CSV files
    try:
        fastr_df = pd.read_csv(fastr_file)
        ngs_df = pd.read_csv(ngs_file)
        print(f"\n[OK] Loaded {len(fastr_df)} rows from FastR file")
        print(f"[OK] Loaded {len(ngs_df)} rows from NGS file")
    except Exception as e:
        print(f"[ERROR] Failed to load CSV files: {e}")
        return

    # Merge dataframes
    merged_df = merge_source_files(fastr_df, ngs_df)
    print(f"[OK] Merged dataframes: {len(merged_df)} records (inner join)")

    if merged_df.empty:
        print("[ERROR] No matching records found after merge. Check that files contain matching player/season/week/team data.")
        return

    # Connect to database
    db = DatabaseConnector()
    if not db.connect():
        print("[ERROR] Failed to connect to database")
        return

    try:
        # Create table if needed
        if not create_nfl_fastr_qb_table(db):
            return

        # Process data
        summary = process_merged_data(db, merged_df)

        if summary['dataframe'].empty:
            print("[WARNING] No valid records to insert")
            return

        # Insert/update data
        success = batch_upsert_data(db, 'nfl_fastr_qb', summary['dataframe'])

        if success:
            # Estimate inserted vs updated (based on batch_upsert_data behavior)
            total = len(summary['dataframe'])
            print_summary(summary, total, 0)  # Will be updated by batch_upsert_data output

            # Move files to processed directory
            move_files_to_processed(fastr_file, ngs_file)
        else:
            print("[ERROR] Failed to insert data into database")

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.disconnect()


if __name__ == "__main__":
    main()
