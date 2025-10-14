#!/usr/bin/env python3

import sys
import os

# Add parent directory to path to import db_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_utils import DatabaseConnector

def validate_tm_pass_integrity():
    """Validate tm_pass table data integrity after first-time script execution"""
    
    db = DatabaseConnector()
    
    try:
        if not db.connect():
            print("Failed to connect to database")
            return False
        
        print("=== TM_PASS TABLE INTEGRITY VALIDATION ===\n")
        
        # 1. Check table structure and record count
        print("1. CHECKING TABLE STRUCTURE AND RECORD COUNT:")
        print("-" * 50)
        
        count_query = "SELECT COUNT(*) FROM tm_pass"
        result = db.fetch_all(count_query)
        total_records = result[0][0] if result else 0
        print(f"Total records in tm_pass table: {total_records}")
        
        # Expected: 32 teams
        if total_records == 32:
            print("[PASS] Expected 32 team records found")
        else:
            print(f"[FAIL] Expected 32 team records, found {total_records}")
        
        # 2. Verify all 32 NFL teams are present
        print("\n2. CHECKING ALL 32 NFL TEAMS ARE PRESENT:")
        print("-" * 50)
        
        teams_query = """
        SELECT t.team_name, t.abrv, tp.tm_pass_id, tp.team_id, tp.season_id, tp.week_id
        FROM tm_pass tp
        JOIN nfl_team t ON tp.team_id = t.team_id
        ORDER BY t.team_name
        """
        teams_result = db.fetch_all(teams_query)
        
        print(f"Teams found in tm_pass table: {len(teams_result)}")
        print(f"{'Team Name':<25} {'Abbrev':<6} {'Pass ID':<8} {'Team ID':<8} {'Season':<7} {'Week':<5}")
        print("-" * 70)
        
        for team in teams_result:
            print(f"{team[0]:<25} {team[1]:<6} {team[2]:<8} {team[3]:<8} {team[4]:<7} {team[5]:<5}")
        
        # 3. Check for duplicate records
        print(f"\n3. CHECKING FOR DUPLICATE RECORDS:")
        print("-" * 50)
        
        duplicate_query = """
        SELECT team_id, season_id, week_id, COUNT(*) as record_count
        FROM tm_pass 
        GROUP BY team_id, season_id, week_id 
        HAVING COUNT(*) > 1
        """
        duplicates = db.fetch_all(duplicate_query)
        
        if not duplicates:
            print("[PASS] No duplicate records found")
        else:
            print(f"[FAIL] Found {len(duplicates)} duplicate combinations:")
            for dup in duplicates:
                print(f"  Team ID {dup[0]}, Season {dup[1]}, Week {dup[2]}: {dup[3]} records")
        
        # 4. Validate foreign key relationships
        print(f"\n4. VALIDATING FOREIGN KEY RELATIONSHIPS:")
        print("-" * 50)
        
        # Check season_id = 3 (2024)
        season_check = """
        SELECT DISTINCT tp.season_id, ns.year, ns.season_type 
        FROM tm_pass tp
        JOIN nfl_season ns ON tp.season_id = ns.season_id
        """
        season_result = db.fetch_all(season_check)
        print(f"Season data: {season_result}")
        
        expected_season = (3, 2024, 'Regular Season')
        if season_result and season_result[0] == expected_season:
            print("[PASS] Season relationship validated (2024, ID: 3)")
        else:
            print(f"[FAIL] Expected {expected_season}, got {season_result}")
        
        # Check week_id = 54 (week 18)
        week_check = """
        SELECT DISTINCT tp.week_id, nw.week_num, nw.week_type
        FROM tm_pass tp
        JOIN nfl_week nw ON tp.week_id = nw.week_id
        """
        week_result = db.fetch_all(week_check)
        print(f"Week data: {week_result}")
        
        expected_week = (54, '18', 'Regular Season')
        if week_result and week_result[0] == expected_week:
            print("[PASS] Week relationship validated (Week 18, ID: 54)")
        else:
            print(f"[FAIL] Expected {expected_week}, got {week_result}")
        
        # Check team_id relationships
        invalid_teams_query = """
        SELECT tp.team_id, COUNT(*) as records
        FROM tm_pass tp
        LEFT JOIN nfl_team t ON tp.team_id = t.team_id
        WHERE t.team_id IS NULL
        GROUP BY tp.team_id
        """
        invalid_teams = db.fetch_all(invalid_teams_query)
        
        if not invalid_teams:
            print("[PASS] All team foreign key relationships valid")
        else:
            print(f"[FAIL] Found {len(invalid_teams)} invalid team relationships:")
            for inv_team in invalid_teams:
                print(f"  Invalid team_id {inv_team[0]}: {inv_team[1]} records")
        
        # 5. Sample data integrity check
        print(f"\n5. SAMPLE DATA INTEGRITY CHECK:")
        print("-" * 50)
        
        sample_query = """
        SELECT t.team_name, t.abrv, tp.tm_pass_cmp, tp.tm_pass_att, tp.tm_pass_yds, 
               tp.tm_pass_td, tp.tm_pass_int, tp.tm_pass_rtg
        FROM tm_pass tp
        JOIN nfl_team t ON tp.team_id = t.team_id
        ORDER BY tp.tm_pass_yds DESC
        LIMIT 5
        """
        sample_result = db.fetch_all(sample_query)
        
        print("Top 5 teams by passing yards:")
        print(f"{'Team':<20} {'Cmp':<5} {'Att':<5} {'Yds':<6} {'TD':<4} {'Int':<4} {'Rating':<7}")
        print("-" * 60)
        
        for sample in sample_result:
            team_name = sample[0][:18] + ".." if len(sample[0]) > 20 else sample[0]
            print(f"{team_name:<20} {sample[2] or 'N/A':<5} {sample[3] or 'N/A':<5} {sample[4] or 'N/A':<6} {sample[5] or 'N/A':<4} {sample[6] or 'N/A':<4} {sample[7] or 'N/A':<7}")
        
        # 6. Verify this was a first-time execution (no pre-existing data)
        print(f"\n6. FIRST-TIME EXECUTION VALIDATION:")
        print("-" * 50)
        
        # Check if we have exactly the records we expect for week 18 of 2024
        expected_records = 32  # 32 teams
        if total_records == expected_records:
            print(f"[PASS] Found exactly {expected_records} records as expected for first-time execution")
        else:
            print(f"[WARNING] Expected {expected_records} records for first-time execution, found {total_records}")
        
        # Check for any potential data inconsistencies that would indicate UPDATE operations
        print("\nChecking for data inconsistencies...")
        null_check_query = """
        SELECT 
            SUM(CASE WHEN tm_pass_cmp IS NULL THEN 1 ELSE 0 END) as null_completions,
            SUM(CASE WHEN tm_pass_att IS NULL THEN 1 ELSE 0 END) as null_attempts,  
            SUM(CASE WHEN tm_pass_yds IS NULL THEN 1 ELSE 0 END) as null_yards,
            SUM(CASE WHEN team_id IS NULL THEN 1 ELSE 0 END) as null_team_ids
        FROM tm_pass
        """
        null_result = db.fetch_all(null_check_query)
        
        if null_result:
            null_data = null_result[0]
            print(f"NULL value counts - Completions: {null_data[0]}, Attempts: {null_data[1]}, Yards: {null_data[2]}, Team IDs: {null_data[3]}")
            
            if all(count == 0 for count in null_data):
                print("[PASS] No unexpected NULL values in critical fields")
            else:
                print("[WARNING] Found NULL values in critical fields")
        
        print(f"\n=== VALIDATION COMPLETE ===")
        
        return True
        
    except Exception as e:
        print(f"Error during validation: {e}")
        return False
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    validate_tm_pass_integrity()