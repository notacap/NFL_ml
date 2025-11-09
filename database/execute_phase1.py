#!/usr/bin/env python3
"""
Phase 1 Migration: Create plyr_master table and add player_guid columns
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_utils import DatabaseConnector

def execute_phase1():
    """Execute Phase 1 migration statements."""
    
    db = DatabaseConnector()
    
    print("=" * 80)
    print("PHASE 1: Creating plyr_master table and adding player_guid columns")
    print("=" * 80)
    
    if not db.connect():
        print("[ERROR] Failed to connect to database")
        return False
    
    # Step 1: Create plyr_master table
    print("\n[1/3] Creating plyr_master table...")
    create_master_sql = """
    CREATE TABLE IF NOT EXISTS plyr_master (
        player_guid VARCHAR(64) PRIMARY KEY,
        plyr_name VARCHAR(255) NOT NULL,
        plyr_birthday DATE,
        plyr_height INT,
        plyr_college VARCHAR(255),
        plyr_draft_tm VARCHAR(255),
        plyr_draft_rd INT,
        plyr_draft_pick INT,
        plyr_draft_yr INT,
        primary_pos VARCHAR(10),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_plyr_name (plyr_name),
        INDEX idx_plyr_birthday (plyr_birthday),
        INDEX idx_draft_yr (plyr_draft_yr),
        UNIQUE KEY uk_player_identity (plyr_name, plyr_birthday, plyr_draft_tm)
    )
    """
    
    if db.execute_query(create_master_sql):
        print("[OK] plyr_master table created successfully")
    else:
        print("[ERROR] Failed to create plyr_master table")
        db.disconnect()
        return False
    
    # Step 2: Add player_guid to plyr table
    print("\n[2/3] Adding player_guid column to plyr table...")
    try:
        # Check if column already exists
        check_col_query = """
        SELECT COUNT(*) FROM information_schema.columns 
        WHERE table_schema = DATABASE() AND table_name = 'plyr' AND column_name = 'player_guid'
        """
        result = db.fetch_all(check_col_query)
        
        if result[0][0] == 0:
            # Column doesn't exist, add it
            add_plyr_guid_sql = "ALTER TABLE plyr ADD COLUMN player_guid VARCHAR(64) AFTER plyr_id"
            if db.execute_query(add_plyr_guid_sql):
                print("[OK] player_guid column added to plyr table")
            else:
                print("[ERROR] Failed to add player_guid column to plyr table")
                db.disconnect()
                return False
        else:
            print("[OK] player_guid column already exists in plyr table")
    except Exception as e:
        print(f"[ERROR] Error adding player_guid to plyr table: {e}")
        db.disconnect()
        return False
    
    # Step 3: Add player_guid to multi_tm_plyr table
    print("\n[3/3] Adding player_guid column to multi_tm_plyr table...")
    try:
        # Check if column already exists
        check_mtp_col_query = """
        SELECT COUNT(*) FROM information_schema.columns 
        WHERE table_schema = DATABASE() AND table_name = 'multi_tm_plyr' AND column_name = 'player_guid'
        """
        result = db.fetch_all(check_mtp_col_query)
        
        if result[0][0] == 0:
            # Column doesn't exist, add it
            add_mtp_guid_sql = "ALTER TABLE multi_tm_plyr ADD COLUMN player_guid VARCHAR(64) AFTER multi_tm_plyr_id"
            if db.execute_query(add_mtp_guid_sql):
                print("[OK] player_guid column added to multi_tm_plyr table")
            else:
                print("[ERROR] Failed to add player_guid column to multi_tm_plyr table")
                db.disconnect()
                return False
        else:
            print("[OK] player_guid column already exists in multi_tm_plyr table")
    except Exception as e:
        print(f"[ERROR] Error adding player_guid to multi_tm_plyr table: {e}")
        db.disconnect()
        return False
    
    # Verify all tables exist
    print("\n[VERIFICATION] Checking table structures...")
    
    # Check plyr_master
    if db.table_exists('plyr_master'):
        print("[OK] plyr_master table exists")
    else:
        print("[ERROR] plyr_master table missing")
        db.disconnect()
        return False
    
    # Check plyr.player_guid column
    plyr_cols = db.fetch_all("DESCRIBE plyr")
    plyr_col_names = [col[0] for col in plyr_cols]
    if 'player_guid' in plyr_col_names:
        print("[OK] plyr.player_guid column exists")
    else:
        print("[ERROR] plyr.player_guid column missing")
        db.disconnect()
        return False
    
    # Check multi_tm_plyr.player_guid column
    mtp_cols = db.fetch_all("DESCRIBE multi_tm_plyr")
    mtp_col_names = [col[0] for col in mtp_cols]
    if 'player_guid' in mtp_col_names:
        print("[OK] multi_tm_plyr.player_guid column exists")
    else:
        print("[ERROR] multi_tm_plyr.player_guid column missing")
        db.disconnect()
        return False
    
    db.disconnect()
    
    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE: All tables and columns created successfully")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = execute_phase1()
    sys.exit(0 if success else 1)