#!/usr/bin/env python3

import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from db_utils import DatabaseConnector

def validate_tm_def_vs_rb_execution():
    """Validate that tm_def_vs_rb.py first-time execution was correct"""
    
    db = DatabaseConnector()
    if not db.connect():
        print("Failed to connect to database")
        return False
    
    try:
        print("=== tm_def_vs_rb.py FIRST-TIME EXECUTION VALIDATION ===\n")
        
        # 1. Check NFL team count
        team_count = db.fetch_all('SELECT COUNT(*) FROM nfl_team')[0][0]
        print(f"Total NFL teams in database: {team_count}")
        
        # 2. Check database records
        db_records = db.fetch_all('SELECT COUNT(*) FROM tm_def_vs_rb WHERE season_id = 3 AND week_id = 54')[0][0]
        total_records = db.fetch_all('SELECT COUNT(*) FROM tm_def_vs_rb')[0][0]
        print(f"Records in tm_def_vs_rb for Season 3, Week 54: {db_records}")
        print(f"Total records in tm_def_vs_rb table: {total_records}")
        
        # 3. Analyze CSV source data  
        csv_path = r'C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\2024\tm_defense_vs_rb\week_18\clean\cleaned_tm_def_vs_rb_20250813_180739.csv'
        df = pd.read_csv(csv_path)
        
        # Apply same filtering logic as tm_def_vs_rb.py
        df_filtered = df[df['Tm'].notna()]
        df_filtered = df_filtered[df_filtered['Tm'].str.strip() != '']
        df_filtered = df_filtered[~df_filtered['Tm'].str.contains('Avg|League|Total', na=False)]
        
        print(f"\nCSV Analysis:")
        print(f"  Original CSV rows: {len(df)}")
        print(f"  After filtering: {len(df_filtered)}")
        print(f"  Teams from CSV: {len(df_filtered)}")
        
        # 4. Check for duplicate records (constraint violation)
        dupes = db.fetch_all("""
        SELECT team_id, season_id, week_id, COUNT(*)
        FROM tm_def_vs_rb 
        GROUP BY team_id, season_id, week_id 
        HAVING COUNT(*) > 1
        """)
        
        print(f"\n=== CONSTRAINT VALIDATION ===")
        print(f"Unique constraint: uk_tm_season (team_id, season_id, week_id)")
        if dupes:
            print(f"ERROR: Found {len(dupes)} duplicate records!")
            for d in dupes:
                print(f"  Duplicate: Team {d[0]}, Season {d[1]}, Week {d[2]}, Count {d[3]}")
        else:
            print("SUCCESS: No duplicate records - unique constraint working correctly")
        
        # 5. Validate execution type
        print(f"\n=== EXECUTION TYPE VALIDATION ===")
        print(f"Expected teams (NFL): 32")
        print(f"CSV teams (filtered): {len(df_filtered)}")
        print(f"Database records: {db_records}")
        print(f"Total table records: {total_records}")
        
        # Check if this was truly a first-time execution
        if total_records == db_records == 32:
            print("CONFIRMED: First-time execution with 32 INSERT operations")
            print("No pre-existing records found")
            print("No UPDATE operations occurred")
        else:
            print("WARNING: Record counts suggest potential issues")
        
        # 6. Data quality check - verify team names were resolved correctly
        team_mapping_check = db.fetch_all("""
        SELECT t.team_name, tr.team_id, tr.tm_def_rb_rush_att, tr.tm_def_rb_rush_yds
        FROM tm_def_vs_rb tr
        JOIN nfl_team t ON tr.team_id = t.team_id
        WHERE tr.season_id = 3 AND tr.week_id = 54
        ORDER BY tr.team_id
        LIMIT 5
        """)
        
        print(f"\n=== SAMPLE TEAM MAPPING VALIDATION ===")
        for record in team_mapping_check:
            print(f"Team: {record[0]} (ID: {record[1]}) - Rush Att: {record[2]}, Rush Yds: {record[3]}")
        
        # 7. Summary assessment
        print(f"\n=== FINAL ASSESSMENT ===")
        
        if (len(df_filtered) == 32 and 
            db_records == 32 and 
            total_records == 32 and 
            len(dupes) == 0):
            print("VALIDATION PASSED: tm_def_vs_rb.py executed correctly")
            print("  All 32 records were genuine INSERTs")
            print("  No unexpected UPDATE operations")
            print("  Unique constraints properly enforced") 
            print("  Data consistency verified")
            return True
        else:
            print("VALIDATION FAILED: Issues detected")
            if len(df_filtered) != 32:
                print(f"  - CSV filtering issue: Expected 32, got {len(df_filtered)}")
            if db_records != 32:
                print(f"  - Database record issue: Expected 32, got {db_records}")
            if total_records != db_records:
                print(f"  - Multiple season/week data present")
            if len(dupes) > 0:
                print(f"  - Duplicate records found: {len(dupes)}")
            return False
            
    finally:
        db.disconnect()

if __name__ == "__main__":
    validate_tm_def_vs_rb_execution()