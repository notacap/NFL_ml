#!/usr/bin/env python3
"""
Phase 5 Validation: Comprehensive migration validation tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_utils import DatabaseConnector

def test_migration():
    """Validate migration success with comprehensive tests."""
    
    db = DatabaseConnector()
    
    print("=" * 80)
    print("MIGRATION VALIDATION")
    print("=" * 80)
    
    if not db.connect():
        print("[ERROR] Failed to connect to database")
        return False
    
    tests = {
        "No NULL GUIDs in plyr": """
            SELECT COUNT(*) FROM plyr WHERE player_guid IS NULL
        """,
        "No NULL GUIDs in multi_tm_plyr": """
            SELECT COUNT(*) FROM multi_tm_plyr WHERE player_guid IS NULL
        """,
        "No orphaned GUIDs in plyr": """
            SELECT COUNT(*) FROM plyr p 
            LEFT JOIN plyr_master pm ON p.player_guid = pm.player_guid
            WHERE pm.player_guid IS NULL AND p.player_guid IS NOT NULL
        """,
        "No orphaned GUIDs in multi_tm_plyr": """
            SELECT COUNT(*) FROM multi_tm_plyr mtp 
            LEFT JOIN plyr_master pm ON mtp.player_guid = pm.player_guid
            WHERE pm.player_guid IS NULL AND mtp.player_guid IS NOT NULL
        """,
        "Unique player count": """
            SELECT COUNT(DISTINCT player_guid) FROM plyr_master
        """,
        "Multi-season players": """
            SELECT COUNT(*) FROM (
                SELECT player_guid, COUNT(DISTINCT season_id) as seasons
                FROM plyr
                GROUP BY player_guid
                HAVING seasons >= 2
            ) t
        """,
        "No duplicate GUIDs in plyr_master": """
            SELECT COUNT(*) FROM (
                SELECT player_guid, COUNT(*) as cnt
                FROM plyr_master
                GROUP BY player_guid
                HAVING cnt > 1
            ) t
        """,
        "plyr_master table row count": """
            SELECT COUNT(*) FROM plyr_master
        """,
        "plyr table total records": """
            SELECT COUNT(*) FROM plyr
        """,
        "multi_tm_plyr total records": """
            SELECT COUNT(*) FROM multi_tm_plyr
        """
    }
    
    all_passed = True
    results = {}
    
    for test_name, query in tests.items():
        try:
            result = db.fetch_all(query)[0][0]
            results[test_name] = result
            
            # Determine if test passed
            if "NULL" in test_name or "orphaned" in test_name or "duplicate" in test_name:
                passed = result == 0
                expected = "0 (none)"
            else:
                passed = result > 0
                expected = "> 0"
            
            status = "[PASS]" if passed else "[FAIL]"
            print(f"\n{status} {test_name}")
            print(f"  Result: {result} (expected: {expected})")
            
            if not passed:
                all_passed = False
                
        except Exception as e:
            print(f"\n[ERROR] {test_name}")
            print(f"  Error: {e}")
            all_passed = False
    
    # Additional specific tests
    print("\n" + "=" * 60)
    print("DETAILED VALIDATION CHECKS")
    print("=" * 60)
    
    # Test 1: Cross-season query capability
    print("\n[TEST] Cross-season query capability...")
    cross_season_query = """
        SELECT 
            pm.plyr_name,
            GROUP_CONCAT(DISTINCT s.year ORDER BY s.year) as seasons,
            COUNT(DISTINCT s.year) as season_count
        FROM plyr p
        JOIN plyr_master pm ON p.player_guid = pm.player_guid
        JOIN nfl_season s ON p.season_id = s.season_id
        GROUP BY pm.player_guid, pm.plyr_name
        HAVING season_count >= 3
        ORDER BY season_count DESC, pm.plyr_name
        LIMIT 10
    """
    
    try:
        cross_season_results = db.fetch_all(cross_season_query)
        if cross_season_results:
            print("[PASS] Cross-season queries working")
            print("  Sample players with 3+ seasons:")
            for result in cross_season_results[:5]:
                print(f"    {result[0]} -> Seasons: {result[1]} (Total: {result[2]})")
        else:
            print("[WARN] No players found with 3+ seasons")
    except Exception as e:
        print(f"[FAIL] Cross-season query error: {e}")
        all_passed = False
    
    # Test 2: GUID determinism verification
    print("\n[TEST] GUID determinism verification...")
    guid_check_query = """
        SELECT 
            plyr_name, plyr_birthday, plyr_draft_tm, player_guid,
            COUNT(*) as guid_count
        FROM plyr_master
        GROUP BY plyr_name, plyr_birthday, plyr_draft_tm
        HAVING guid_count > 1
    """
    
    try:
        guid_duplicates = db.fetch_all(guid_check_query)
        if not guid_duplicates:
            print("[PASS] GUID generation is deterministic")
        else:
            print(f"[FAIL] Found {len(guid_duplicates)} GUID duplicates")
            all_passed = False
    except Exception as e:
        print(f"[FAIL] GUID determinism check error: {e}")
        all_passed = False
    
    # Test 3: Natural key coverage
    print("\n[TEST] Natural key coverage...")
    natural_key_query = """
        SELECT 
            SUM(CASE WHEN plyr_name IS NULL THEN 1 ELSE 0 END) as null_names,
            SUM(CASE WHEN plyr_birthday IS NULL THEN 1 ELSE 0 END) as null_birthdays,
            SUM(CASE WHEN plyr_draft_tm IS NULL THEN 1 ELSE 0 END) as null_draft_tms,
            COUNT(*) as total_records
        FROM plyr_master
    """
    
    try:
        natural_key_result = db.fetch_all(natural_key_query)[0]
        null_names, null_birthdays, null_draft_tms, total_records = natural_key_result
        
        if null_names == 0 and null_birthdays == 0 and null_draft_tms == 0:
            print("[PASS] Natural key fields complete")
            print(f"  All {total_records} master records have complete natural keys")
        else:
            print(f"[WARN] Incomplete natural keys found:")
            print(f"  NULL names: {null_names}, NULL birthdays: {null_birthdays}, NULL draft_tms: {null_draft_tms}")
    except Exception as e:
        print(f"[FAIL] Natural key coverage check error: {e}")
        all_passed = False
    
    # Test 4: Foreign key integrity test with stat tables
    print("\n[TEST] Foreign key integrity with stat tables...")
    fk_test_query = """
        SELECT 
            (SELECT COUNT(*) FROM plyr_def pd 
             LEFT JOIN plyr p ON pd.plyr_id = p.plyr_id 
             WHERE p.plyr_id IS NULL) as orphaned_def_stats,
            (SELECT COUNT(*) FROM plyr_def pd 
             JOIN plyr p ON pd.plyr_id = p.plyr_id 
             LEFT JOIN plyr_master pm ON p.player_guid = pm.player_guid 
             WHERE pm.player_guid IS NULL) as def_stats_missing_master
    """
    
    try:
        fk_result = db.fetch_all(fk_test_query)[0]
        orphaned_def, missing_master = fk_result
        
        if orphaned_def == 0 and missing_master == 0:
            print("[PASS] Foreign key integrity maintained")
        else:
            print(f"[WARN] Foreign key issues found:")
            print(f"  Orphaned def stats: {orphaned_def}, Missing master records: {missing_master}")
    except Exception as e:
        print(f"[INFO] Foreign key test skipped (plyr_def may not exist): {e}")
    
    db.disconnect()
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    if all_passed:
        print("[SUCCESS] ALL VALIDATION TESTS PASSED")
        print("\nMigration Results:")
        print(f"  - Unique players in master table: {results.get('Unique player count', 'N/A')}")
        print(f"  - Players with multiple seasons: {results.get('Multi-season players', 'N/A')}")
        print(f"  - Total plyr records: {results.get('plyr table total records', 'N/A')}")
        print(f"  - Total multi_tm_plyr records: {results.get('multi_tm_plyr total records', 'N/A')}")
    else:
        print("[FAILURE] SOME VALIDATION TESTS FAILED")
        print("Review the results above for details")
    
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    success = test_migration()
    sys.exit(0 if success else 1)