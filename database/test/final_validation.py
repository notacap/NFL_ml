#!/usr/bin/env python3

import sys
import os

# Add parent directory to path to import db_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_utils import DatabaseConnector

def final_foreign_key_validation():
    """Final validation of foreign key relationships with correct column names"""
    
    db = DatabaseConnector()
    
    try:
        if not db.connect():
            print("Failed to connect to database")
            return False
        
        print("=== FINAL FOREIGN KEY VALIDATION ===\n")
        
        # Check season_id = 3 (2024) with correct column names
        print("1. SEASON VALIDATION:")
        print("-" * 30)
        season_check = """
        SELECT DISTINCT tp.season_id, ns.year 
        FROM tm_pass tp
        JOIN nfl_season ns ON tp.season_id = ns.season_id
        """
        season_result = db.fetch_all(season_check)
        print(f"Season data: {season_result}")
        
        if season_result and season_result[0] == (3, 2024):
            print("[PASS] Season relationship validated (2024, ID: 3)")
        else:
            print(f"[FAIL] Expected (3, 2024), got {season_result}")
        
        # Check week_id = 54 (week 18) with correct column names
        print("\n2. WEEK VALIDATION:")  
        print("-" * 30)
        week_check = """
        SELECT DISTINCT tp.week_id, nw.week_num
        FROM tm_pass tp
        JOIN nfl_week nw ON tp.week_id = nw.week_id
        """
        week_result = db.fetch_all(week_check)
        print(f"Week data: {week_result}")
        
        if week_result and week_result[0] == (54, '18'):
            print("[PASS] Week relationship validated (Week 18, ID: 54)")
        else:
            print(f"[FAIL] Expected (54, '18'), got {week_result}")
        
        # Verify unique constraint is working (no duplicates possible)
        print("\n3. UNIQUE CONSTRAINT VALIDATION:")
        print("-" * 40)
        constraint_check = """
        SHOW CREATE TABLE tm_pass
        """
        constraint_result = db.fetch_all(constraint_check)
        
        if constraint_result:
            create_statement = constraint_result[0][1]  # The CREATE TABLE statement
            if "UNIQUE KEY `uk_tm_season` (`team_id`,`season_id`,`week_id`)" in create_statement:
                print("[PASS] Unique constraint uk_tm_season is properly defined")
                print("       This prevents duplicate team records for same season/week")
            else:
                print("[WARNING] Unique constraint may not be properly defined")
        
        # Final confirmation: Check for any orphaned foreign keys
        print("\n4. ORPHANED FOREIGN KEY CHECK:")
        print("-" * 40)
        
        # Check for invalid season_id references
        orphan_season_check = """
        SELECT COUNT(*) 
        FROM tm_pass tp
        LEFT JOIN nfl_season ns ON tp.season_id = ns.season_id
        WHERE ns.season_id IS NULL
        """
        orphan_seasons = db.fetch_all(orphan_season_check)
        
        if orphan_seasons and orphan_seasons[0][0] == 0:
            print("[PASS] No orphaned season_id references")
        else:
            print(f"[FAIL] Found {orphan_seasons[0][0]} orphaned season_id references")
        
        # Check for invalid week_id references  
        orphan_week_check = """
        SELECT COUNT(*)
        FROM tm_pass tp
        LEFT JOIN nfl_week nw ON tp.week_id = nw.week_id
        WHERE nw.week_id IS NULL
        """
        orphan_weeks = db.fetch_all(orphan_week_check)
        
        if orphan_weeks and orphan_weeks[0][0] == 0:
            print("[PASS] No orphaned week_id references")
        else:
            print(f"[FAIL] Found {orphan_weeks[0][0]} orphaned week_id references")
        
        return True
        
    except Exception as e:
        print(f"Error during validation: {e}")
        return False
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    final_foreign_key_validation()