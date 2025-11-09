#!/usr/bin/env python3
"""
Fix NULL plyr_draft_tm values in plyr table before migration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_utils import DatabaseConnector, generate_player_guid

def fix_null_draft_tm():
    """Fix NULL plyr_draft_tm values."""
    
    db = DatabaseConnector()
    
    print("=" * 80)
    print("FIXING NULL plyr_draft_tm VALUES")
    print("=" * 80)
    
    if not db.connect():
        print("[ERROR] Failed to connect to database")
        return False
    
    # Step 1: Find records with NULL draft_tm
    print("\n[1/3] Finding records with NULL plyr_draft_tm...")
    null_draft_query = """
        SELECT plyr_id, plyr_name, plyr_birthday, plyr_draft_tm, plyr_draft_yr
        FROM plyr 
        WHERE plyr_draft_tm IS NULL
        ORDER BY plyr_id
    """
    null_records = db.fetch_all(null_draft_query)
    
    print(f"Found {len(null_records)} records with NULL plyr_draft_tm:")
    for record in null_records:
        print(f"  ID {record[0]}: {record[1]} ({record[2]}) - Draft Yr: {record[4]}")
    
    # Step 2: Update NULL draft_tm values to 'UNDRAFTED FREE AGENT'
    print("\n[2/3] Updating NULL plyr_draft_tm to 'UNDRAFTED FREE AGENT'...")
    update_query = """
        UPDATE plyr 
        SET plyr_draft_tm = 'UNDRAFTED FREE AGENT'
        WHERE plyr_draft_tm IS NULL
    """
    
    success = db.execute_query(update_query)
    if success:
        print(f"[OK] Updated {len(null_records)} records")
    else:
        print("[ERROR] Failed to update NULL draft_tm values")
        db.disconnect()
        return False
    
    # Step 3: Also check multi_tm_plyr table
    print("\n[3/3] Checking multi_tm_plyr for NULL draft_tm...")
    null_mtp_query = """
        SELECT multi_tm_plyr_id, plyr_name, plyr_birthday, plyr_draft_tm
        FROM multi_tm_plyr 
        WHERE plyr_draft_tm IS NULL
        ORDER BY multi_tm_plyr_id
    """
    null_mtp_records = db.fetch_all(null_mtp_query)
    
    if null_mtp_records:
        print(f"Found {len(null_mtp_records)} records in multi_tm_plyr with NULL plyr_draft_tm:")
        for record in null_mtp_records:
            print(f"  ID {record[0]}: {record[1]} ({record[2]})")
        
        # Update multi_tm_plyr as well
        update_mtp_query = """
            UPDATE multi_tm_plyr 
            SET plyr_draft_tm = 'UNDRAFTED FREE AGENT'
            WHERE plyr_draft_tm IS NULL
        """
        
        success = db.execute_query(update_mtp_query)
        if success:
            print(f"[OK] Updated {len(null_mtp_records)} multi_tm_plyr records")
        else:
            print("[ERROR] Failed to update multi_tm_plyr NULL draft_tm values")
            db.disconnect()
            return False
    else:
        print("[OK] No NULL plyr_draft_tm values in multi_tm_plyr")
    
    db.disconnect()
    
    print("\n" + "=" * 80)
    print("NULL DRAFT_TM VALUES FIXED")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = fix_null_draft_tm()
    sys.exit(0 if success else 1)