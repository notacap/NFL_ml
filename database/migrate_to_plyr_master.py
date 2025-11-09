#!/usr/bin/env python3
"""
Phase 3 Migration: Migrate existing plyr data to plyr_master
One-time migration of existing plyr records to plyr_master table
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_utils import DatabaseConnector, generate_player_guid

def migrate_existing_data():
    """One-time migration of existing plyr records to plyr_master."""
    
    db = DatabaseConnector()
    
    print("=" * 80)
    print("MIGRATION: Existing plyr data -> plyr_master")
    print("=" * 80)
    
    if not db.connect():
        print("[ERROR] Failed to connect to database")
        return False
    
    # Step 1: Check for duplicates by natural key within same season
    print("\n[1/6] Checking for duplicate players within seasons...")
    dup_query = """
        SELECT plyr_name, plyr_birthday, plyr_draft_tm, season_id, COUNT(*) as cnt
        FROM plyr
        GROUP BY plyr_name, plyr_birthday, plyr_draft_tm, season_id
        HAVING cnt > 1
    """
    duplicates = db.fetch_all(dup_query)
    
    if duplicates:
        print(f"[WARNING] Found {len(duplicates)} duplicate player-season combinations")
        print("Manual review required before proceeding")
        for dup in duplicates:
            print(f"  - {dup[0]} ({dup[1]}) in season {dup[3]}: {dup[4]} records")
        return False
    else:
        print("[OK] No duplicates found")
    
    # Step 2: Extract unique players
    print("\n[2/6] Extracting unique players...")
    unique_query = """
        SELECT DISTINCT 
            plyr_name, plyr_birthday, plyr_height, plyr_college,
            plyr_draft_tm, plyr_draft_rd, plyr_draft_pick, plyr_draft_yr
        FROM plyr
        WHERE plyr_name IS NOT NULL 
          AND plyr_birthday IS NOT NULL 
          AND plyr_draft_tm IS NOT NULL
    """
    unique_players = db.fetch_all(unique_query)
    print(f"[OK] Found {len(unique_players)} unique players")
    
    # Step 3: Create plyr_master records
    print("\n[3/6] Creating plyr_master records...")
    insert_query = """
        INSERT INTO plyr_master 
        (player_guid, plyr_name, plyr_birthday, plyr_height, plyr_college,
         plyr_draft_tm, plyr_draft_rd, plyr_draft_pick, plyr_draft_yr)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
            plyr_height = COALESCE(VALUES(plyr_height), plyr_height),
            plyr_college = COALESCE(VALUES(plyr_college), plyr_college),
            plyr_draft_rd = COALESCE(VALUES(plyr_draft_rd), plyr_draft_rd),
            plyr_draft_pick = COALESCE(VALUES(plyr_draft_pick), plyr_draft_pick),
            plyr_draft_yr = COALESCE(VALUES(plyr_draft_yr), plyr_draft_yr)
    """
    
    created_count = 0
    for player in unique_players:
        player_guid = generate_player_guid(player[0], player[1], player[4])
        success = db.execute_query(insert_query, (player_guid,) + player)
        if success:
            created_count += 1
        else:
            print(f"[WARNING] Failed to create master record for {player[0]}")
    
    print(f"[OK] Created/updated {created_count}/{len(unique_players)} plyr_master records")
    
    # Step 4: Populate plyr.player_guid
    print("\n[4/6] Populating plyr.player_guid...")
    update_query = """
        UPDATE plyr p
        JOIN plyr_master pm ON 
            p.plyr_name = pm.plyr_name 
            AND p.plyr_birthday = pm.plyr_birthday 
            AND p.plyr_draft_tm = pm.plyr_draft_tm
        SET p.player_guid = pm.player_guid
        WHERE p.player_guid IS NULL
    """
    success = db.execute_query(update_query)
    if success:
        print("[OK] Updated plyr.player_guid")
    else:
        print("[ERROR] Failed to update plyr.player_guid")
        db.disconnect()
        return False
    
    # Step 5: Populate multi_tm_plyr.player_guid
    print("\n[5/6] Populating multi_tm_plyr.player_guid...")
    update_mtp_query = """
        UPDATE multi_tm_plyr mtp
        JOIN plyr_master pm ON 
            mtp.plyr_name = pm.plyr_name 
            AND mtp.plyr_birthday = pm.plyr_birthday 
            AND mtp.plyr_draft_tm = pm.plyr_draft_tm
        SET mtp.player_guid = pm.player_guid
        WHERE mtp.player_guid IS NULL
    """
    success = db.execute_query(update_mtp_query)
    if success:
        print("[OK] Updated multi_tm_plyr.player_guid")
    else:
        print("[ERROR] Failed to update multi_tm_plyr.player_guid")
        db.disconnect()
        return False
    
    # Step 6: Validation
    print("\n[6/6] Running validation checks...")
    
    # Check for NULL player_guids in plyr
    null_count_query = "SELECT COUNT(*) FROM plyr WHERE player_guid IS NULL"
    null_count = db.fetch_all(null_count_query)[0][0]
    if null_count > 0:
        print(f"[ERROR] {null_count} plyr records have NULL player_guid")
        
        # Show some examples
        null_examples_query = """
            SELECT plyr_id, plyr_name, plyr_birthday, plyr_draft_tm 
            FROM plyr 
            WHERE player_guid IS NULL 
            LIMIT 5
        """
        null_examples = db.fetch_all(null_examples_query)
        print("Examples of NULL player_guid records:")
        for example in null_examples:
            print(f"  ID {example[0]}: {example[1]} ({example[2]}) - {example[3]}")
        db.disconnect()
        return False
    print("[OK] All plyr records have player_guid")
    
    # Check for NULL player_guids in multi_tm_plyr
    null_mtp_count_query = "SELECT COUNT(*) FROM multi_tm_plyr WHERE player_guid IS NULL"
    null_mtp_count = db.fetch_all(null_mtp_count_query)[0][0]
    if null_mtp_count > 0:
        print(f"[ERROR] {null_mtp_count} multi_tm_plyr records have NULL player_guid")
        
        # Show some examples
        null_mtp_examples_query = """
            SELECT multi_tm_plyr_id, plyr_name, plyr_birthday, plyr_draft_tm 
            FROM multi_tm_plyr 
            WHERE player_guid IS NULL 
            LIMIT 5
        """
        null_mtp_examples = db.fetch_all(null_mtp_examples_query)
        print("Examples of NULL player_guid records in multi_tm_plyr:")
        for example in null_mtp_examples:
            print(f"  ID {example[0]}: {example[1]} ({example[2]}) - {example[3]}")
        db.disconnect()
        return False
    print("[OK] All multi_tm_plyr records have player_guid")
    
    # Count total master records
    master_count_query = "SELECT COUNT(*) FROM plyr_master"
    master_count = db.fetch_all(master_count_query)[0][0]
    print(f"[OK] Total unique players in plyr_master: {master_count}")
    
    # Count cross-season players
    cross_season_query = """
        SELECT COUNT(DISTINCT p.player_guid) as multi_season_players
        FROM (
            SELECT player_guid, COUNT(DISTINCT season_id) as season_count
            FROM plyr
            WHERE player_guid IS NOT NULL
            GROUP BY player_guid
            HAVING season_count >= 2
        ) p
    """
    cross_season_count = db.fetch_all(cross_season_query)[0][0]
    print(f"[OK] Players with multiple seasons: {cross_season_count}")
    
    db.disconnect()
    
    print("\n" + "=" * 80)
    print("MIGRATION COMPLETE")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = migrate_existing_data()
    sys.exit(0 if success else 1)