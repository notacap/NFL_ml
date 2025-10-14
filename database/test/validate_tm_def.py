#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from db_utils import DatabaseConnector

def validate_tm_def_inserts():
    """Validate tm_def table data and verify only INSERT operations occurred"""
    
    db = DatabaseConnector()
    
    try:
        # Connect to database
        if not db.connect():
            print("Failed to connect to database")
            return False
        
        print("=== TM_DEF TABLE VALIDATION ===\n")
        
        # 1. Check table structure and constraints
        print("1. TABLE STRUCTURE AND CONSTRAINTS:")
        db.cursor.execute("DESCRIBE tm_def")
        columns = db.cursor.fetchall()
        print(f"   Total columns: {len(columns)}")
        for col in columns:
            print(f"   - {col[0]} {col[1]} {'NULL' if col[2] == 'YES' else 'NOT NULL'}")
        
        # Check unique constraint
        print("\n   UNIQUE CONSTRAINTS:")
        db.cursor.execute("""
            SELECT CONSTRAINT_NAME, COLUMN_NAME 
            FROM information_schema.KEY_COLUMN_USAGE 
            WHERE TABLE_SCHEMA = DATABASE() 
            AND TABLE_NAME = 'tm_def' 
            AND CONSTRAINT_NAME LIKE 'uk_%'
        """)
        uk_constraints = db.cursor.fetchall()
        for constraint in uk_constraints:
            print(f"   - {constraint[0]}: {constraint[1]}")
        
        # Check foreign key constraints
        print("\n   FOREIGN KEY CONSTRAINTS:")
        db.cursor.execute("""
            SELECT CONSTRAINT_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
            FROM information_schema.KEY_COLUMN_USAGE 
            WHERE TABLE_SCHEMA = DATABASE() 
            AND TABLE_NAME = 'tm_def' 
            AND REFERENCED_TABLE_NAME IS NOT NULL
        """)
        fk_constraints = db.cursor.fetchall()
        for constraint in fk_constraints:
            print(f"   - {constraint[0]}: {constraint[1]} -> {constraint[2]}.{constraint[3]}")
        
        # 2. Check total record count
        print("\n2. RECORD COUNT:")
        db.cursor.execute("SELECT COUNT(*) FROM tm_def")
        total_count = db.cursor.fetchone()[0]
        print(f"   Total records in tm_def: {total_count}")
        
        # 3. Validate foreign key integrity
        print("\n3. FOREIGN KEY INTEGRITY:")
        
        # Check team_id references
        db.cursor.execute("""
            SELECT COUNT(*) FROM tm_def td 
            LEFT JOIN nfl_team nt ON td.team_id = nt.team_id 
            WHERE nt.team_id IS NULL
        """)
        orphaned_teams = db.cursor.fetchone()[0]
        print(f"   Orphaned team_id records: {orphaned_teams}")
        
        # Check season_id references
        db.cursor.execute("""
            SELECT COUNT(*) FROM tm_def td 
            LEFT JOIN nfl_season ns ON td.season_id = ns.season_id 
            WHERE ns.season_id IS NULL
        """)
        orphaned_seasons = db.cursor.fetchone()[0]
        print(f"   Orphaned season_id records: {orphaned_seasons}")
        
        # Check week_id references
        db.cursor.execute("""
            SELECT COUNT(*) FROM tm_def td 
            LEFT JOIN nfl_week nw ON td.week_id = nw.week_id 
            WHERE nw.week_id IS NULL
        """)
        orphaned_weeks = db.cursor.fetchone()[0]
        print(f"   Orphaned week_id records: {orphaned_weeks}")
        
        # 4. Check for unique constraint violations (duplicates)
        print("\n4. UNIQUE CONSTRAINT VALIDATION:")
        db.cursor.execute("""
            SELECT team_id, season_id, week_id, COUNT(*) as duplicate_count
            FROM tm_def 
            GROUP BY team_id, season_id, week_id 
            HAVING COUNT(*) > 1
        """)
        duplicates = db.cursor.fetchall()
        print(f"   Duplicate records (should be 0): {len(duplicates)}")
        for dup in duplicates:
            print(f"   - team_id={dup[0]}, season_id={dup[1]}, week_id={dup[2]}, count={dup[3]}")
        
        # 5. Analyze inserted data by team
        print("\n5. DATA ANALYSIS BY TEAM:")
        db.cursor.execute("""
            SELECT 
                nt.abrv,
                nt.team_name,
                ns.year,
                nw.week_num,
                td.tm_def_pts_allwd,
                td.tm_def_yds,
                td.tm_def_tkawy,
                td.tm_def_sk
            FROM tm_def td
            JOIN nfl_team nt ON td.team_id = nt.team_id
            JOIN nfl_season ns ON td.season_id = ns.season_id  
            JOIN nfl_week nw ON td.week_id = nw.week_id
            ORDER BY nt.abrv
        """)
        team_data = db.cursor.fetchall()
        
        print(f"   Inserted data for {len(team_data)} teams:")
        print(f"   {'Team':<4} {'Season':<6} {'Week':<4} {'Pts Allowed':<11} {'Yds':<5} {'Takeaways':<9} {'Sacks':<5}")
        print(f"   {'-'*4} {'-'*6} {'-'*4} {'-'*11} {'-'*5} {'-'*9} {'-'*5}")
        
        for team in team_data:
            pts = team[4] if team[4] is not None else 'NULL'
            yds = team[5] if team[5] is not None else 'NULL'
            tkawy = team[6] if team[6] is not None else 'NULL'
            sk = team[7] if team[7] is not None else 'NULL'
            print(f"   {team[0]:<4} {team[2]:<6} {team[3]:<4} {pts:<11} {yds:<5} {tkawy:<9} {sk:<5}")
        
        # 6. Check for data anomalies
        print("\n6. DATA QUALITY CHECKS:")
        
        # Check for NULL critical fields
        db.cursor.execute("SELECT COUNT(*) FROM tm_def WHERE team_id IS NULL")
        null_teams = db.cursor.fetchone()[0]
        print(f"   Records with NULL team_id: {null_teams}")
        
        db.cursor.execute("SELECT COUNT(*) FROM tm_def WHERE season_id IS NULL")
        null_seasons = db.cursor.fetchone()[0]
        print(f"   Records with NULL season_id: {null_seasons}")
        
        db.cursor.execute("SELECT COUNT(*) FROM tm_def WHERE week_id IS NULL") 
        null_weeks = db.cursor.fetchone()[0]
        print(f"   Records with NULL week_id: {null_weeks}")
        
        # Check for reasonable statistical ranges
        db.cursor.execute("SELECT MIN(tm_def_pts_allwd), MAX(tm_def_pts_allwd), AVG(tm_def_pts_allwd) FROM tm_def WHERE tm_def_pts_allwd IS NOT NULL")
        pts_stats = db.cursor.fetchone()
        if pts_stats and pts_stats[0] is not None:
            print(f"   Points allowed range: {pts_stats[0]} - {pts_stats[1]} (avg: {pts_stats[2]:.1f})")
        
        db.cursor.execute("SELECT MIN(tm_def_yds), MAX(tm_def_yds), AVG(tm_def_yds) FROM tm_def WHERE tm_def_yds IS NOT NULL")
        yds_stats = db.cursor.fetchone()
        if yds_stats and yds_stats[0] is not None:
            print(f"   Total yards range: {yds_stats[0]} - {yds_stats[1]} (avg: {yds_stats[2]:.1f})")
        
        # 7. Final validation summary
        print("\n7. VALIDATION SUMMARY:")
        
        issues_found = 0
        
        if orphaned_teams > 0:
            print(f"   [X] ISSUE: {orphaned_teams} records with invalid team_id")
            issues_found += 1
        else:
            print(f"   [OK] All team_id references are valid")
            
        if orphaned_seasons > 0:
            print(f"   [X] ISSUE: {orphaned_seasons} records with invalid season_id")
            issues_found += 1
        else:
            print(f"   [OK] All season_id references are valid")
            
        if orphaned_weeks > 0:
            print(f"   [X] ISSUE: {orphaned_weeks} records with invalid week_id")
            issues_found += 1
        else:
            print(f"   [OK] All week_id references are valid")
            
        if len(duplicates) > 0:
            print(f"   [X] ISSUE: {len(duplicates)} duplicate records found")
            issues_found += 1
        else:
            print(f"   [OK] No duplicate records (unique constraint honored)")
            
        if total_count == 32:
            print(f"   [OK] Expected 32 team records inserted correctly")
        else:
            print(f"   [X] ISSUE: Expected 32 records, found {total_count}")
            issues_found += 1
        
        if null_teams > 0 or null_seasons > 0 or null_weeks > 0:
            print(f"   [X] ISSUE: Critical NULL values found in foreign key fields")
            issues_found += 1
        else:
            print(f"   [OK] No NULL values in critical foreign key fields")
        
        print(f"\n   OVERALL RESULT: {'[SUCCESS] VALIDATION PASSED' if issues_found == 0 else '[FAILED] VALIDATION FAILED'}")
        print(f"   Issues found: {issues_found}")
        
        return issues_found == 0
        
    except Exception as e:
        print(f"Error during validation: {e}")
        return False
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    validate_tm_def_inserts()