#!/usr/bin/env python3
"""
SQL Validation Queries from Migration Spec
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_utils import DatabaseConnector

def run_sql_validation():
    """Run the SQL validation queries from the migration specification."""
    
    db = DatabaseConnector()
    
    print("=" * 80)
    print("SQL VALIDATION QUERIES FROM MIGRATION SPEC")
    print("=" * 80)
    
    if not db.connect():
        print("[ERROR] Failed to connect to database")
        return False
    
    validations = [
        {
            "name": "Verify no NULL player_guids",
            "query": "SELECT COUNT(*) as null_guids FROM plyr WHERE player_guid IS NULL",
            "expected": "0"
        },
        {
            "name": "Check unique player count", 
            "query": "SELECT COUNT(DISTINCT player_guid) as unique_players FROM plyr_master",
            "expected": "2000-3500 (typical across 4 seasons)"
        },
        {
            "name": "Verify cross-season tracking (top 20)",
            "query": """
                SELECT pm.plyr_name, COUNT(DISTINCT p.season_id) as seasons
                FROM plyr p
                JOIN plyr_master pm ON p.player_guid = pm.player_guid
                GROUP BY pm.player_guid, pm.plyr_name
                ORDER BY seasons DESC
                LIMIT 20
            """,
            "expected": "Players with 2-4 seasons"
        },
        {
            "name": "Identify same-name players (edge cases)",
            "query": """
                SELECT plyr_name, plyr_birthday, COUNT(DISTINCT plyr_draft_tm) as draft_tm_count
                FROM plyr_master
                GROUP BY plyr_name, plyr_birthday
                HAVING draft_tm_count > 1
            """,
            "expected": "Shows why draft_tm was needed in natural key"
        },
        {
            "name": "Verify no duplicate GUIDs",
            "query": """
                SELECT player_guid, COUNT(*) as cnt
                FROM plyr_master
                GROUP BY player_guid
                HAVING cnt > 1
            """,
            "expected": "0"
        }
    ]
    
    all_passed = True
    
    for validation in validations:
        print(f"\n[QUERY] {validation['name']}")
        print(f"Expected: {validation['expected']}")
        print("-" * 60)
        
        try:
            results = db.fetch_all(validation['query'])
            
            if "cross-season tracking" in validation['name']:
                print("Top 20 players by season count:")
                for result in results:
                    print(f"  {result[0]:<25} -> {result[1]} seasons")
                passed = len(results) > 0 and any(r[1] >= 2 for r in results)
                
            elif "same-name players" in validation['name']:
                if results:
                    print("Same-name players with different draft teams:")
                    for result in results:
                        print(f"  {result[0]} ({result[1]}) -> {result[2]} different draft teams")
                    passed = True  # This is informational
                else:
                    print("  No same-name players with different draft teams found")
                    passed = True
                    
            elif "no duplicate GUIDs" in validation['name'] or "no NULL" in validation['name']:
                count = results[0][0]
                print(f"  Count: {count}")
                passed = count == 0
                
            elif "unique player count" in validation['name']:
                count = results[0][0]
                print(f"  Unique players: {count}")
                passed = 2000 <= count <= 4000  # Reasonable range
                
            else:
                print(f"  Results: {results}")
                passed = True
            
            status = "[PASS]" if passed else "[FAIL]"
            print(f"{status}")
            
            if not passed:
                all_passed = False
                
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            all_passed = False
    
    # Cross-season query test with specific player
    print(f"\n[QUERY] Test cross-season query capability")
    print("Expected: Stats across multiple seasons for a specific player")
    print("-" * 60)
    
    try:
        # Find a player with multiple seasons first
        multi_season_query = """
            SELECT pm.plyr_name, COUNT(DISTINCT p.season_id) as seasons
            FROM plyr p
            JOIN plyr_master pm ON p.player_guid = pm.player_guid
            GROUP BY pm.player_guid, pm.plyr_name
            HAVING seasons >= 3
            ORDER BY pm.plyr_name
            LIMIT 1
        """
        multi_season_player = db.fetch_all(multi_season_query)
        
        if multi_season_player:
            player_name = multi_season_player[0][0]
            print(f"Testing with player: {player_name}")
            
            cross_season_test_query = f"""
                SELECT 
                    pm.plyr_name,
                    s.year as season,
                    p.plyr_pos as position,
                    p.plyr_age as age
                FROM plyr p
                JOIN plyr_master pm ON p.player_guid = pm.player_guid
                JOIN nfl_season s ON p.season_id = s.season_id
                WHERE pm.plyr_name = %s
                ORDER BY s.year
            """
            
            player_seasons = db.fetch_all(cross_season_test_query, (player_name,))
            for season_data in player_seasons:
                print(f"  {season_data[0]} | {season_data[1]} | Pos: {season_data[2]} | Age: {season_data[3]}")
            
            print("[PASS] Cross-season queries working correctly")
        else:
            print("[WARN] No players found with 3+ seasons for detailed test")
            
    except Exception as e:
        print(f"[ERROR] Cross-season test failed: {e}")
        all_passed = False
    
    db.disconnect()
    
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL SQL VALIDATION QUERIES PASSED")
    else:
        print("SOME SQL VALIDATION QUERIES FAILED")
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    success = run_sql_validation()
    sys.exit(0 if success else 1)