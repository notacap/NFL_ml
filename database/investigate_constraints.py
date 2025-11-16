#!/usr/bin/env python3
"""
Investigate constraint violations in player statistics tables.

This script examines existing constraint violations to understand the scope
of duplicate data issues before attempting the duplicate player removal.
"""

from db_utils import DatabaseConnector

def main():
    """Investigate constraint violations in player statistics tables."""
    
    print("="*80)
    print("CONSTRAINT VIOLATION INVESTIGATION")
    print("="*80)
    
    # Connect to database
    db = DatabaseConnector()
    if not db.connect():
        print("[ERROR] Failed to connect to database")
        return False
    
    try:
        # Tables to check for season-level duplicates
        season_tables = ['plyr_def', 'plyr_pass', 'plyr_rush', 'plyr_rec']
        
        for table in season_tables:
            print(f"\n{'='*60}")
            print(f"ANALYZING {table.upper()}")
            print('='*60)
            
            # Check for duplicates by (plyr_id, season_id)
            duplicate_query = f"""
                SELECT plyr_id, season_id, COUNT(*) as duplicate_count
                FROM {table}
                GROUP BY plyr_id, season_id
                HAVING COUNT(*) > 1
                ORDER BY duplicate_count DESC, plyr_id
                LIMIT 10
            """
            
            duplicates = db.fetch_all(duplicate_query)
            print(f"Found {len(duplicates)} duplicate combinations in {table}")
            
            if duplicates:
                print(f"\nTop duplicate combinations:")
                print(f"{'Player ID':<12} {'Season ID':<10} {'Count':<8}")
                print("-" * 30)
                
                for dup in duplicates:
                    plyr_id, season_id, count = dup
                    print(f"{plyr_id:<12} {season_id:<10} {count:<8}")
                    
                    # Get player details for first few duplicates
                    if duplicates.index(dup) < 3:
                        player_info_query = """
                            SELECT p.plyr_name, p.plyr_draft_tm, t.abrv as team
                            FROM plyr p
                            JOIN nfl_team t ON p.team_id = t.team_id
                            WHERE p.plyr_id = %s AND p.season_id = %s
                        """
                        player_info = db.fetch_all(player_info_query, (plyr_id, season_id))
                        if player_info:
                            print(f"    Player: {player_info[0][0]}, Draft: {player_info[0][1]}, Team: {player_info[0][2]}")
        
        # Check if these duplicates are related to our draft team duplicates
        print(f"\n{'='*60}")
        print("CHECKING RELATIONSHIP TO DRAFT TEAM DUPLICATES")
        print('='*60)
        
        # Get the plyr_ids from our duplicate mapping
        draft_duplicate_query = """
            SELECT 
                p1.plyr_id as duplicate_plyr_id,
                p2.plyr_id as correct_plyr_id,
                p1.plyr_name,
                p1.season_id
            FROM plyr p1
            INNER JOIN plyr p2 
                ON p1.season_id = p2.season_id
                AND p1.plyr_name = p2.plyr_name
                AND p1.plyr_birthday = p2.plyr_birthday 
                AND p1.plyr_draft_tm != p2.plyr_draft_tm
            WHERE p1.plyr_draft_tm = 'UNDRAFTED FREE AGENT'
        """
        
        draft_duplicates = db.fetch_all(draft_duplicate_query)
        
        # Check if any of the duplicate plyr_ids appear in the constraint violations
        for table in season_tables:
            print(f"\n[INFO] Checking {table} for draft duplicate impact...")
            
            for draft_dup in draft_duplicates:
                duplicate_plyr_id, correct_plyr_id, player_name, season_id = draft_dup
                
                # Check if either plyr_id appears in this table
                check_query = f"""
                    SELECT COUNT(*) FROM {table} 
                    WHERE plyr_id IN (%s, %s) AND season_id = %s
                """
                count_result = db.fetch_all(check_query, (duplicate_plyr_id, correct_plyr_id, season_id))
                record_count = count_result[0][0]
                
                if record_count > 0:
                    print(f"    {player_name} (season {season_id}): {record_count} records found")
                    
                    # Show details
                    detail_query = f"""
                        SELECT plyr_id, COUNT(*) as count
                        FROM {table} 
                        WHERE plyr_id IN (%s, %s) AND season_id = %s
                        GROUP BY plyr_id
                    """
                    details = db.fetch_all(detail_query, (duplicate_plyr_id, correct_plyr_id, season_id))
                    for detail in details:
                        print(f"      plyr_id {detail[0]}: {detail[1]} records")
        
        # Check for other types of duplicates not related to draft team
        print(f"\n{'='*60}")
        print("CHECKING FOR NON-DRAFT-RELATED DUPLICATES")
        print('='*60)
        
        # Look for players with same name/season but different plyr_ids (not draft related)
        other_duplicate_query = """
            SELECT 
                p1.plyr_id,
                p2.plyr_id,
                p1.plyr_name,
                p1.season_id,
                p1.plyr_draft_tm,
                p2.plyr_draft_tm,
                t1.abrv as team1,
                t2.abrv as team2
            FROM plyr p1
            INNER JOIN plyr p2 
                ON p1.season_id = p2.season_id
                AND p1.plyr_name = p2.plyr_name
                AND p1.plyr_id != p2.plyr_id
            JOIN nfl_team t1 ON p1.team_id = t1.team_id
            JOIN nfl_team t2 ON p2.team_id = t2.team_id
            WHERE NOT (
                (p1.plyr_draft_tm = 'UNDRAFTED FREE AGENT' AND p2.plyr_draft_tm != 'UNDRAFTED FREE AGENT')
                OR (p2.plyr_draft_tm = 'UNDRAFTED FREE AGENT' AND p1.plyr_draft_tm != 'UNDRAFTED FREE AGENT')
            )
            AND p1.plyr_birthday = p2.plyr_birthday
            ORDER BY p1.plyr_name, p1.season_id
            LIMIT 20
        """
        
        other_duplicates = db.fetch_all(other_duplicate_query)
        
        if other_duplicates:
            print(f"Found {len(other_duplicates)} non-draft-related player duplicates:")
            print(f"{'ID1':<8} {'ID2':<8} {'Name':<20} {'Season':<7} {'Draft1':<15} {'Draft2':<15} {'Team1':<5} {'Team2':<5}")
            print("-" * 90)
            
            for dup in other_duplicates:
                print(f"{dup[0]:<8} {dup[1]:<8} {dup[2]:<20} {dup[3]:<7} {dup[4]:<15} {dup[5]:<15} {dup[6]:<5} {dup[7]:<5}")
        else:
            print("No non-draft-related player duplicates found")
            
        return True
        
    except Exception as e:
        print(f"[ERROR] Investigation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    main()