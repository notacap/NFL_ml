import sys
import os
from datetime import datetime

# Add the utilities directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_utils import connect_to_database

# Define table configurations based on the XML prompt
TABLE_CONFIG = {
    # Tables without week_id
    'nfl_game': {'contains_week_id': False, 'scope': 'season'},

    # Game-level tables (all 18 weeks for all seasons)
    'nfl_game_info': {'contains_week_id': True, 'scope': 'game'},
    'nfl_game_pbp': {'contains_week_id': True, 'scope': 'game'},
    'nfl_gm_weather': {'contains_week_id': True, 'scope': 'game'},
    'plyr_gm_def': {'contains_week_id': True, 'scope': 'game'},
    'plyr_gm_fmbl': {'contains_week_id': True, 'scope': 'game'},
    'plyr_gm_pass': {'contains_week_id': True, 'scope': 'game'},
    'plyr_gm_rec': {'contains_week_id': True, 'scope': 'game'},
    'plyr_gm_rush': {'contains_week_id': True, 'scope': 'game'},
    'plyr_gm_snap_ct': {'contains_week_id': True, 'scope': 'game'},
    'plyr_gm_starters': {'contains_week_id': True, 'scope': 'game'},
    'tm_gm_drive': {'contains_week_id': True, 'scope': 'game'},
    'tm_gm_exp_pts': {'contains_week_id': True, 'scope': 'game'},
    'tm_gm_stats': {'contains_week_id': True, 'scope': 'game'},

    # Season-level tables
    'nfl_standings': {'contains_week_id': True, 'scope': 'season'},
    'plyr': {'contains_week_id': False, 'scope': 'season'},
    'plyr_def': {'contains_week_id': True, 'scope': 'season'},
    'plyr_pass': {'contains_week_id': True, 'scope': 'season'},
    'plyr_rec': {'contains_week_id': True, 'scope': 'season'},
    'plyr_rush': {'contains_week_id': True, 'scope': 'season'},
    'plyr_rz_pass': {'contains_week_id': True, 'scope': 'season'},
    'plyr_rz_rec': {'contains_week_id': True, 'scope': 'season'},
    'plyr_rz_rush': {'contains_week_id': True, 'scope': 'season'},
    'plyr_scoring': {'contains_week_id': True, 'scope': 'season'},
    'tm_conv': {'contains_week_id': True, 'scope': 'season'},
    'tm_def': {'contains_week_id': True, 'scope': 'season'},
    'tm_def_conv': {'contains_week_id': True, 'scope': 'season'},
    'tm_def_dr_against_avg': {'contains_week_id': True, 'scope': 'season'},
    'tm_def_pass': {'contains_week_id': True, 'scope': 'season'},
    'tm_def_rush': {'contains_week_id': True, 'scope': 'season'},
    'tm_def_vs_qb': {'contains_week_id': True, 'scope': 'season'},
    'tm_def_vs_rb': {'contains_week_id': True, 'scope': 'season'},
    'tm_def_vs_te': {'contains_week_id': True, 'scope': 'season'},
    'tm_def_vs_wr': {'contains_week_id': True, 'scope': 'season'},
    'tm_pass': {'contains_week_id': True, 'scope': 'season'},
    'tm_rec': {'contains_week_id': True, 'scope': 'season'},
    'tm_rush': {'contains_week_id': True, 'scope': 'season'},
}

# Expected values
EXPECTED_SEASONS = [1, 2, 3, 4]  # season_id values for 2022, 2023, 2024, 2025
EXPECTED_WEEK_COUNTS_SEASON = {
    1: 1,   # 2022: 1 unique week_id
    2: 1,   # 2023: 1 unique week_id
    3: 1,   # 2024: 1 unique week_id
    4: 5,   # 2025: 5 unique week_id values
}
EXPECTED_WEEK_COUNTS_GAME = {
    1: 18,  # 2022: 18 unique week_id values
    2: 18,  # 2023: 18 unique week_id values
    3: 18,  # 2024: 18 unique week_id values
    4: 5,   # 2025: 5 unique week_id values (current season, only weeks 1-5 played so far)
}

def verify_table(cursor, table_name, config, log_lines):
    """Verify a single table for expected season_id and week_id values."""
    log_lines.append(f"\n{'='*80}")
    log_lines.append(f"Table: {table_name}")
    log_lines.append(f"{'='*80}")

    issues = []

    # Check if table exists
    cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
    if not cursor.fetchone():
        issues.append(f"Table {table_name} does not exist")
        log_lines.append(f"ERROR: Table does not exist")
        return issues

    # Verify season_id values
    cursor.execute(f"SELECT DISTINCT season_id FROM {table_name} ORDER BY season_id")
    actual_seasons = [row[0] for row in cursor.fetchall()]
    log_lines.append(f"Season IDs found: {actual_seasons}")

    missing_seasons = set(EXPECTED_SEASONS) - set(actual_seasons)
    if missing_seasons:
        issues.append(f"{table_name}: Missing season_id values: {sorted(missing_seasons)}")
        log_lines.append(f"WARNING: Missing season_id values: {sorted(missing_seasons)}")

    extra_seasons = set(actual_seasons) - set(EXPECTED_SEASONS)
    if extra_seasons:
        issues.append(f"{table_name}: Unexpected season_id values: {sorted(extra_seasons)}")
        log_lines.append(f"WARNING: Unexpected season_id values: {sorted(extra_seasons)}")

    # Verify week_id values if applicable
    if config['contains_week_id']:
        # First check if week_id column actually exists
        cursor.execute(f"SHOW COLUMNS FROM {table_name} LIKE 'week_id'")
        week_id_exists = cursor.fetchone() is not None

        if not week_id_exists:
            issues.append(f"{table_name}: Expected to have week_id column but it doesn't exist")
            log_lines.append(f"\nERROR: week_id column does not exist in this table")
        else:
            log_lines.append(f"\nWeek ID verification (scope: {config['scope']}):")

            expected_counts = EXPECTED_WEEK_COUNTS_GAME if config['scope'] == 'game' else EXPECTED_WEEK_COUNTS_SEASON

            for season_id in EXPECTED_SEASONS:
                if season_id not in actual_seasons:
                    continue

                cursor.execute(f"""
                    SELECT DISTINCT week_id
                    FROM {table_name}
                    WHERE season_id = {season_id}
                    ORDER BY week_id
                """)
                actual_weeks = [row[0] for row in cursor.fetchall()]
                actual_count = len(actual_weeks)
                expected_count = expected_counts[season_id]

                # Special case for plyr table in 2025
                if table_name == 'plyr' and season_id == 4:
                    if actual_count != 1:
                        issues.append(f"{table_name}: Season 2025 (season_id {season_id}) should have exactly 1 unique week_id, found {actual_count}")
                        log_lines.append(f"  Season {season_id}: FAIL - Expected 1 unique week_id, found {actual_count}: {actual_weeks}")
                    else:
                        log_lines.append(f"  Season {season_id}: OK - Found 1 unique week_id as expected: {actual_weeks}")
                else:
                    if actual_count != expected_count:
                        issues.append(f"{table_name}: Season {season_id} should have {expected_count} unique week_id values, found {actual_count}")
                        log_lines.append(f"  Season {season_id}: FAIL - Expected {expected_count} unique week_id values, found {actual_count}: {actual_weeks}")
                    else:
                        log_lines.append(f"  Season {season_id}: OK - Found {expected_count} unique week_id values as expected")
    else:
        log_lines.append("No week_id verification needed for this table")

    if not issues:
        log_lines.append(f"\n[PASS] Table {table_name} verification: PASSED")
    else:
        log_lines.append(f"\n[FAIL] Table {table_name} verification: FAILED")

    return issues

def main():
    """Main verification function."""
    print("Starting database verification...")

    # Connect to database
    connection = connect_to_database()
    cursor = connection.cursor()

    log_lines = []
    log_lines.append("="*80)
    log_lines.append("NFL DATABASE VERIFICATION REPORT")
    log_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append("="*80)

    all_issues = []

    # Verify each table
    for table_name, config in TABLE_CONFIG.items():
        issues = verify_table(cursor, table_name, config, log_lines)
        all_issues.extend(issues)

    # Summary
    log_lines.append(f"\n{'='*80}")
    log_lines.append("VERIFICATION SUMMARY")
    log_lines.append(f"{'='*80}")
    log_lines.append(f"Total tables checked: {len(TABLE_CONFIG)}")
    log_lines.append(f"Total issues found: {len(all_issues)}")

    if all_issues:
        log_lines.append("\nISSUES FOUND:")
        for i, issue in enumerate(all_issues, 1):
            log_lines.append(f"{i}. {issue}")
        log_lines.append("\nANSWER: NO - Not all expected data is present in the database")
    else:
        log_lines.append("\nANSWER: YES - All expected data is present in the database")

    log_lines.append(f"\n{'='*80}")
    log_lines.append("END OF REPORT")
    log_lines.append(f"{'='*80}")

    # Write log file
    log_dir = r'C:\Users\nocap\Desktop\code\NFL_ml\utilities\logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'verify_database_complete.log')

    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

    # Print to console
    print('\n'.join(log_lines))
    print(f"\nLog file written to: {log_file_path}")

    # Cleanup
    cursor.close()
    connection.close()

    # Return exit code based on results
    return 0 if not all_issues else 1

if __name__ == "__main__":
    sys.exit(main())
