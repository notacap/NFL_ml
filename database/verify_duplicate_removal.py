#!/usr/bin/env python3
"""
Verify that duplicate player removal was successful.

This script confirms that no draft team duplicates remain in the database.
"""

from db_utils import DatabaseConnector

def main():
    """Verify duplicate removal success."""
    
    print("="*80)
    print("DUPLICATE REMOVAL VERIFICATION")
    print("="*80)
    
    # Connect to database
    db = DatabaseConnector()
    if not db.connect():
        print("[ERROR] Failed to connect to database")
        return False
    
    try:
        # Check for remaining duplicates using the original query
        print("\n[INFO] Checking for remaining duplicate players...")
        
        duplicate_check_query = """
            SELECT 
                p1.plyr_id,
                p1.plyr_name,
                p1.plyr_birthday,
                p1.plyr_draft_tm,
                p2.plyr_draft_tm AS other_draft_tm,
                COUNT(*) as pair_count
            FROM plyr p1
            INNER JOIN plyr p2 
                ON p1.season_id = p2.season_id
                AND p1.plyr_name = p2.plyr_name
                AND p1.plyr_birthday = p2.plyr_birthday 
                AND p1.plyr_draft_tm != p2.plyr_draft_tm
            GROUP BY p1.plyr_id, p1.plyr_name, p1.plyr_birthday, p1.plyr_draft_tm, p2.plyr_draft_tm
            ORDER BY p1.plyr_name, p1.plyr_birthday
        """
        
        remaining_duplicates = db.fetch_all(duplicate_check_query)
        
        if remaining_duplicates:
            print(f"[ERROR] Found {len(remaining_duplicates)} remaining duplicate records:")
            print(f"{'Player ID':<10} {'Player Name':<25} {'Birthday':<12} {'Draft Team 1':<20} {'Draft Team 2':<20}")
            print("-" * 95)
            
            for dup in remaining_duplicates:
                print(f"{dup[0]:<10} {dup[1]:<25} {dup[2]:<12} {dup[3]:<20} {dup[4]:<20}")
            
            return False
        else:
            print("[SUCCESS] No duplicate players found - removal was successful!")
        
        # Check plyr_master for duplicates
        print("\n[INFO] Checking plyr_master for remaining duplicates...")
        
        master_duplicate_query = """
            SELECT 
                pm1.player_guid,
                pm1.plyr_name,
                pm1.plyr_birthday,
                pm1.plyr_draft_tm,
                pm2.plyr_draft_tm AS other_draft_tm,
                COUNT(*) as pair_count
            FROM plyr_master pm1
            INNER JOIN plyr_master pm2 
                ON pm1.plyr_name = pm2.plyr_name
                AND pm1.plyr_birthday = pm2.plyr_birthday 
                AND pm1.plyr_draft_tm != pm2.plyr_draft_tm
                AND pm1.player_guid != pm2.player_guid
            GROUP BY pm1.player_guid, pm1.plyr_name, pm1.plyr_birthday, pm1.plyr_draft_tm, pm2.plyr_draft_tm
            ORDER BY pm1.plyr_name, pm1.plyr_birthday
        """
        
        master_duplicates = db.fetch_all(master_duplicate_query)
        
        if master_duplicates:
            print(f"[ERROR] Found {len(master_duplicates)} remaining duplicate records in plyr_master:")
            print(f"{'Player GUID':<36} {'Player Name':<25} {'Birthday':<12} {'Draft Team 1':<20} {'Draft Team 2':<20}")
            print("-" * 115)
            
            for dup in master_duplicates:
                print(f"{dup[0]:<36} {dup[1]:<25} {dup[2]:<12} {dup[3]:<20} {dup[4]:<20}")
            
            return False
        else:
            print("[SUCCESS] No duplicate players found in plyr_master - removal was successful!")
        
        # Summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        # Count total records
        plyr_count = db.fetch_all("SELECT COUNT(*) FROM plyr")[0][0]
        master_count = db.fetch_all("SELECT COUNT(*) FROM plyr_master")[0][0]
        
        print(f"[INFO] Total plyr records: {plyr_count}")
        print(f"[INFO] Total plyr_master records: {master_count}")
        
        # Count unique players by name+birthday in each table
        unique_plyr_query = """
            SELECT COUNT(DISTINCT CONCAT(plyr_name, '|', plyr_birthday)) 
            FROM plyr
        """
        unique_plyr_count = db.fetch_all(unique_plyr_query)[0][0]
        
        unique_master_query = """
            SELECT COUNT(DISTINCT CONCAT(plyr_name, '|', plyr_birthday)) 
            FROM plyr_master
        """
        unique_master_count = db.fetch_all(unique_master_query)[0][0]
        
        print(f"[INFO] Unique players in plyr (by name+birthday): {unique_plyr_count}")
        print(f"[INFO] Unique players in plyr_master (by name+birthday): {unique_master_count}")
        
        # Check if numbers make sense
        if unique_master_count > unique_plyr_count:
            print(f"[WARNING] More unique players in plyr_master than plyr - this may be expected due to different seasons")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    success = main()
    if success:
        print("\n[SUCCESS] Duplicate removal verification passed!")
    else:
        print("\n[FAILED] Duplicate removal verification failed!")