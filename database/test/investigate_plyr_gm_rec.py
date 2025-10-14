#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_utils import DatabaseConnector

def investigate_plyr_gm_rec_table():
    """Investigate the plyr_gm_rec table insert operations."""
    
    print("="*80)
    print("PLYR_GM_REC TABLE INVESTIGATION")
    print("="*80)
    
    db = DatabaseConnector()
    if not db.connect():
        print("[FAIL] Could not connect to database")
        return
    
    try:
        # 1. Check table structure
        print("\n1. TABLE STRUCTURE:")
        print("-"*50)
        
        describe_query = "DESCRIBE plyr_gm_rec"
        columns = db.fetch_all(describe_query)
        
        print(f"{'Column':<25} {'Type':<20} {'Null':<5} {'Key':<5} {'Default':<10} {'Extra':<15}")
        print("-"*80)
        for col in columns:
            print(f"{col[0]:<25} {col[1]:<20} {col[2]:<5} {col[3]:<5} {str(col[4] or ''):<10} {col[5]:<15}")
        
        # 2. Check constraints and indexes
        print(f"\n2. CONSTRAINTS AND INDEXES:")
        print("-"*50)
        
        constraints_query = """
        SELECT 
            kcu.CONSTRAINT_NAME, 
            tc.CONSTRAINT_TYPE,
            kcu.COLUMN_NAME,
            kcu.REFERENCED_TABLE_NAME,
            kcu.REFERENCED_COLUMN_NAME
        FROM information_schema.KEY_COLUMN_USAGE kcu
        LEFT JOIN information_schema.TABLE_CONSTRAINTS tc 
            ON kcu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME 
            AND kcu.TABLE_SCHEMA = tc.TABLE_SCHEMA
        WHERE kcu.TABLE_SCHEMA = 'nfl_stats' 
            AND kcu.TABLE_NAME = 'plyr_gm_rec'
        ORDER BY kcu.ORDINAL_POSITION
        """
        
        constraints = db.fetch_all(constraints_query)
        if constraints:
            print(f"{'Constraint':<20} {'Type':<12} {'Column':<20} {'Ref Table':<15} {'Ref Column':<15}")
            print("-"*85)
            for constraint in constraints:
                print(f"{constraint[0]:<20} {constraint[1] or 'N/A':<12} {constraint[2]:<20} {constraint[3] or 'N/A':<15} {constraint[4] or 'N/A':<15}")
        else:
            print("No explicit constraints found in metadata")
        
        # 3. Check record count and sample data
        print(f"\n3. RECORD COUNT AND SAMPLE DATA:")
        print("-"*50)
        
        count_query = "SELECT COUNT(*) FROM plyr_gm_rec"
        count_result = db.fetch_all(count_query)
        total_records = count_result[0][0] if count_result else 0
        
        print(f"Total records in plyr_gm_rec: {total_records}")
        
        if total_records > 0:
            # Sample records with player details
            sample_query = """
            SELECT 
                pgr.adv_plyr_gm_rec_id,
                pgr.plyr_id,
                p.plyr_name,
                t.abrv as team,
                ns.year,
                nw.week_num,
                pgr.plyr_gm_rec_tgt,
                pgr.plyr_gm_rec,
                pgr.plyr_gm_rec_yds,
                pgr.plyr_gm_rec_td
            FROM plyr_gm_rec pgr
            JOIN plyr p ON pgr.plyr_id = p.plyr_id
            JOIN nfl_team t ON pgr.team_id = t.team_id
            JOIN nfl_season ns ON pgr.season_id = ns.season_id
            JOIN nfl_week nw ON pgr.week_id = nw.week_id
            ORDER BY pgr.adv_plyr_gm_rec_id
            LIMIT 10
            """
            
            sample_data = db.fetch_all(sample_query)
            if sample_data:
                print(f"\nSample records (first 10):")
                print(f"{'ID':<6} {'Plyr_ID':<8} {'Player Name':<20} {'Tm':<4} {'Year':<6} {'Wk':<3} {'Tgt':<4} {'Rec':<4} {'Yds':<5} {'TD':<3}")
                print("-"*75)
                for row in sample_data:
                    print(f"{row[0]:<6} {row[1]:<8} {row[2]:<20} {row[3]:<4} {row[4]:<6} {row[5]:<3} {row[6] or 0:<4} {row[7] or 0:<4} {row[8] or 0:<5} {row[9] or 0:<3}")
        
        # 4. Check for potential duplicate issues
        print(f"\n4. DUPLICATE ANALYSIS:")
        print("-"*50)
        
        # Check for duplicate player-game combinations
        duplicate_query = """
        SELECT 
            plyr_id, 
            game_id, 
            COUNT(*) as duplicate_count,
            GROUP_CONCAT(adv_plyr_gm_rec_id) as record_ids
        FROM plyr_gm_rec 
        GROUP BY plyr_id, game_id 
        HAVING COUNT(*) > 1
        """
        
        duplicates = db.fetch_all(duplicate_query)
        if duplicates:
            print(f"FOUND {len(duplicates)} DUPLICATE PLAYER-GAME COMBINATIONS:")
            print(f"{'Player ID':<10} {'Game ID':<8} {'Count':<6} {'Record IDs':<20}")
            print("-"*50)
            for dup in duplicates:
                print(f"{dup[0]:<10} {dup[1]:<8} {dup[2]:<6} {dup[3]:<20}")
        else:
            print("No duplicate player-game combinations found (OK)")
        
        # 5. Foreign Key Validation
        print(f"\n5. FOREIGN KEY VALIDATION:")
        print("-"*50)
        
        # Check for orphaned records
        fk_checks = [
            ("plyr_id", "plyr", "plyr_id"),
            ("week_id", "nfl_week", "week_id"), 
            ("game_id", "nfl_game", "game_id"),
            ("season_id", "nfl_season", "season_id"),
            ("team_id", "nfl_team", "team_id")
        ]
        
        for col, ref_table, ref_col in fk_checks:
            orphan_query = f"""
            SELECT COUNT(*) 
            FROM plyr_gm_rec pgr 
            LEFT JOIN {ref_table} ref ON pgr.{col} = ref.{ref_col}
            WHERE ref.{ref_col} IS NULL AND pgr.{col} IS NOT NULL
            """
            
            orphan_result = db.fetch_all(orphan_query)
            orphan_count = orphan_result[0][0] if orphan_result else 0
            
            if orphan_count > 0:
                print(f"[X] {col}: {orphan_count} orphaned records")
            else:
                print(f"[OK] {col}: All foreign key references valid")
        
        # 6. Player ID Resolution Analysis  
        print(f"\n6. PLAYER ID RESOLUTION ANALYSIS:")
        print("-"*50)
        
        # Check players by source table
        source_analysis_query = """
        SELECT 
            'plyr' as source_table,
            COUNT(DISTINCT pgr.plyr_id) as unique_players,
            COUNT(*) as total_records
        FROM plyr_gm_rec pgr
        JOIN plyr p ON pgr.plyr_id = p.plyr_id
        UNION ALL
        SELECT 
            'multi_tm_plyr' as source_table,
            COUNT(DISTINCT pgr.plyr_id) as unique_players,
            COUNT(*) as total_records
        FROM plyr_gm_rec pgr
        JOIN multi_tm_plyr mtp ON pgr.plyr_id = mtp.plyr_id
        """
        
        source_analysis = db.fetch_all(source_analysis_query)
        if source_analysis:
            print(f"{'Source Table':<15} {'Unique Players':<15} {'Total Records':<15}")
            print("-"*45)
            for source in source_analysis:
                print(f"{source[0]:<15} {source[1]:<15} {source[2]:<15}")
        
        # 7. Week 1 Specific Analysis
        print(f"\n7. WEEK 1 SPECIFIC ANALYSIS:")
        print("-"*50)
        
        week1_query = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT pgr.plyr_id) as unique_players,
            COUNT(DISTINCT pgr.game_id) as unique_games,
            COUNT(DISTINCT pgr.team_id) as unique_teams
        FROM plyr_gm_rec pgr
        JOIN nfl_week nw ON pgr.week_id = nw.week_id
        WHERE nw.week_num = '1'
        """
        
        week1_stats = db.fetch_all(week1_query)
        if week1_stats:
            stats = week1_stats[0]
            print(f"Week 1 Statistics:")
            print(f"  - Total records: {stats[0]}")  
            print(f"  - Unique players: {stats[1]}")
            print(f"  - Unique games: {stats[2]}")
            print(f"  - Unique teams: {stats[3]}")
            
            # Expected: 16 games, 32 teams, ~249 players based on user report
            expected_games = 16
            if stats[2] != expected_games:
                print(f"  [!] WARNING: Expected {expected_games} games, found {stats[2]}")
            else:
                print(f"  [OK] Game count matches expected ({expected_games})")
                
            if stats[0] == 249:
                print(f"  [OK] Record count matches user report (249)")
            else:
                print(f"  [!] WARNING: User reported 249 records, found {stats[0]}")
    
    except Exception as e:
        print(f"[ERROR] Investigation failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        db.disconnect()

if __name__ == "__main__":
    investigate_plyr_gm_rec_table()