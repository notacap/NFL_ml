#!/usr/bin/env python3
"""
Final validation check for the migration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_utils import DatabaseConnector

def final_validation():
    """Run final validation check."""
    
    db = DatabaseConnector()
    
    print("=" * 80)
    print("FINAL MIGRATION VALIDATION")
    print("=" * 80)
    
    if not db.connect():
        print("[ERROR] Failed to connect to database")
        return False
    
    # Check for duplicate GUIDs properly
    print("\n[CHECK] Verifying no duplicate GUIDs...")
    duplicate_guid_query = """
        SELECT player_guid, COUNT(*) as cnt
        FROM plyr_master
        GROUP BY player_guid
        HAVING cnt > 1
    """
    
    duplicate_results = db.fetch_all(duplicate_guid_query)
    if duplicate_results:
        print(f"[FAIL] Found {len(duplicate_results)} duplicate GUIDs")
        for result in duplicate_results:
            print(f"  GUID {result[0]}: {result[1]} occurrences")
    else:
        print("[PASS] No duplicate GUIDs found")
    
    # Verify the natural key insight
    print("\n[CHECK] Why draft_tm was needed in natural key...")
    same_name_query = """
        SELECT plyr_name, plyr_birthday, 
               GROUP_CONCAT(DISTINCT plyr_draft_tm) as draft_teams,
               COUNT(DISTINCT plyr_draft_tm) as team_count
        FROM plyr_master
        GROUP BY plyr_name, plyr_birthday
        HAVING team_count > 1
        ORDER BY plyr_name
    """
    
    same_name_results = db.fetch_all(same_name_query)
    if same_name_results:
        print(f"[INFO] Found {len(same_name_results)} players with same name/birthday but different draft teams:")
        for result in same_name_results[:10]:  # Show first 10
            print(f"  {result[0]} ({result[1]}) -> Teams: {result[2]}")
        if len(same_name_results) > 10:
            print(f"  ... and {len(same_name_results) - 10} more")
    else:
        print("[INFO] No players found with same name/birthday but different draft teams")
    
    # Show migration statistics
    print("\n[STATS] Migration Statistics...")
    stats_queries = {
        "Total unique players": "SELECT COUNT(*) FROM plyr_master",
        "Total plyr records": "SELECT COUNT(*) FROM plyr", 
        "Total multi_tm_plyr records": "SELECT COUNT(*) FROM multi_tm_plyr",
        "Players with 1 season": """
            SELECT COUNT(*) FROM (
                SELECT player_guid FROM plyr GROUP BY player_guid HAVING COUNT(DISTINCT season_id) = 1
            ) t
        """,
        "Players with 2+ seasons": """
            SELECT COUNT(*) FROM (
                SELECT player_guid FROM plyr GROUP BY player_guid HAVING COUNT(DISTINCT season_id) >= 2
            ) t
        """,
        "Players with 4 seasons": """
            SELECT COUNT(*) FROM (
                SELECT player_guid FROM plyr GROUP BY player_guid HAVING COUNT(DISTINCT season_id) = 4
            ) t
        """
    }
    
    for stat_name, query in stats_queries.items():
        try:
            result = db.fetch_all(query)[0][0]
            print(f"  {stat_name}: {result:,}")
        except Exception as e:
            print(f"  {stat_name}: Error - {e}")
    
    # Test the updated get_player_id function
    print("\n[TEST] Updated get_player_id function with plyr_master join...")
    try:
        from db_utils import get_player_id, get_season_id
        
        # Get season ID for 2024
        season_2024 = get_season_id(db, 2024)
        
        # Test with a known player
        test_player_id = get_player_id(db, "Josh Allen", "BUF", season_2024)
        if test_player_id > 0:
            # Verify we can get player details
            player_details_query = """
                SELECT p.plyr_id, pm.plyr_name, pm.player_guid, p.plyr_pos, p.plyr_age
                FROM plyr p
                JOIN plyr_master pm ON p.player_guid = pm.player_guid
                WHERE p.plyr_id = %s
            """
            player_details = db.fetch_all(player_details_query, (test_player_id,))
            if player_details:
                details = player_details[0]
                print(f"[PASS] get_player_id working: Josh Allen -> ID {details[0]}, GUID {details[2][:8]}...")
            else:
                print("[WARN] Player details not found")
        else:
            print("[WARN] Josh Allen not found (may not be in 2024 data)")
            
    except Exception as e:
        print(f"[ERROR] get_player_id test failed: {e}")
    
    db.disconnect()
    
    print("\n" + "=" * 80)
    print("MIGRATION VALIDATION COMPLETE")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = final_validation()
    sys.exit(0 if success else 1)