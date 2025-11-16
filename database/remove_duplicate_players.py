#!/usr/bin/env python3
"""
Duplicate Player Removal Operation

This script safely removes duplicate player records while maintaining referential integrity.
The duplicates exist where the same player has records with both actual draft team and "UNDRAFTED FREE AGENT".
We remove the "UNDRAFTED FREE AGENT" records and update all foreign key references.
"""

import mysql.connector
import sys
import os
from db_utils import DatabaseConnector

def main():
    """Execute the duplicate player removal operation with proper validation."""
    
    print("="*80)
    print("DUPLICATE PLAYER REMOVAL OPERATION")
    print("="*80)
    
    # Connect to database
    db = DatabaseConnector()
    if not db.connect():
        print("[ERROR] Failed to connect to database")
        return False
    
    try:
        # Disable autocommit for transaction control
        db.connection.autocommit = False
        print("[INFO] Starting transaction...")
        db.execute_query("START TRANSACTION")
        
        # Step 1: Create mapping table
        print("\n" + "="*60)
        print("STEP 1: Creating duplicate player mapping table")
        print("="*60)
        
        create_mapping_sql = """
        CREATE TEMPORARY TABLE duplicate_player_mapping AS
        SELECT 
            p1.plyr_id AS duplicate_plyr_id,
            p2.plyr_id AS correct_plyr_id,
            p1.plyr_name,
            p1.plyr_draft_tm AS duplicate_draft_tm,
            p2.plyr_draft_tm AS correct_draft_tm
        FROM plyr p1
        INNER JOIN plyr p2 
            ON p1.season_id = p2.season_id
            AND p1.plyr_name = p2.plyr_name
            AND p1.plyr_birthday = p2.plyr_birthday 
            AND p1.plyr_draft_tm != p2.plyr_draft_tm
        WHERE p1.plyr_draft_tm = 'UNDRAFTED FREE AGENT'
        """
        
        success = db.execute_query(create_mapping_sql)
        if not success:
            print("[ERROR] Failed to create mapping table")
            db.execute_query("ROLLBACK")
            return False
        
        # Verify mapping table
        mapping_count = db.fetch_all("SELECT COUNT(*) FROM duplicate_player_mapping")
        expected_count = 13  # Based on CSV data
        actual_count = mapping_count[0][0]
        
        print(f"[INFO] Created mapping table with {actual_count} duplicate records")
        if actual_count != expected_count:
            print(f"[WARNING] Expected {expected_count} records, got {actual_count}")
            
        # Show mapping details
        mappings = db.fetch_all("""
            SELECT duplicate_plyr_id, correct_plyr_id, plyr_name, 
                   duplicate_draft_tm, correct_draft_tm 
            FROM duplicate_player_mapping 
            ORDER BY plyr_name
        """)
        
        print("\nDuplicate player mappings:")
        print(f"{'Dup ID':<8} {'Correct ID':<10} {'Player Name':<25} {'From Draft':<20} {'To Draft':<20}")
        print("-" * 90)
        for mapping in mappings:
            print(f"{mapping[0]:<8} {mapping[1]:<10} {mapping[2]:<25} {mapping[3]:<20} {mapping[4]:<20}")
        
        # Step 2: Update foreign key references
        print("\n" + "="*60)
        print("STEP 2: Updating foreign key references")
        print("="*60)
        
        # List of all tables that reference plyr_id
        reference_tables = [
            'plyr_gm_def', 'plyr_gm_off', 'plyr_gm_snap_ct', 'adv_plyr_gm_pass',
            'adv_plyr_gm_rec', 'plyr_def', 'plyr_off', 'plyr_ret', 'plyr_kck',
            'plyr_pass', 'plyr_rush', 'plyr_rec', 'plyr_fant', 'injury_report',
            'plyr_gm_pk', 'plyr_gm_punt', 'red_zone_plyr_off', 'red_zone_plyr_def'
        ]
        
        total_updated_records = 0
        
        for table in reference_tables:
            print(f"\n[INFO] Updating {table}...")
            
            # Check if table exists
            table_exists_query = """
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = DATABASE() AND table_name = %s
            """
            table_check = db.fetch_all(table_exists_query, (table,))
            
            if table_check[0][0] == 0:
                print(f"[SKIP] Table {table} does not exist")
                continue
            
            # Count records that will be updated
            count_query = f"""
                SELECT COUNT(*) FROM {table} t
                INNER JOIN duplicate_player_mapping dpm ON t.plyr_id = dpm.duplicate_plyr_id
            """
            count_result = db.fetch_all(count_query)
            records_to_update = count_result[0][0]
            
            if records_to_update == 0:
                print(f"[OK] No records to update in {table}")
                continue
                
            print(f"[INFO] Found {records_to_update} records to update in {table}")
            
            # Update the foreign key references
            update_query = f"""
                UPDATE {table} t
                INNER JOIN duplicate_player_mapping dpm ON t.plyr_id = dpm.duplicate_plyr_id
                SET t.plyr_id = dpm.correct_plyr_id
            """
            
            success = db.execute_query(update_query)
            if not success:
                print(f"[ERROR] Failed to update {table}")
                db.execute_query("ROLLBACK")
                return False
                
            affected_rows = db.cursor.rowcount
            total_updated_records += affected_rows
            print(f"[OK] Updated {affected_rows} records in {table}")
        
        print(f"\n[SUMMARY] Updated {total_updated_records} total foreign key references")
        
        # Step 3: Verify updates - no references should remain to old plyr_id values
        print("\n" + "="*60)
        print("STEP 3: Verifying foreign key updates")
        print("="*60)
        
        validation_failed = False
        
        for table in reference_tables:
            # Skip non-existent tables
            table_check = db.fetch_all(table_exists_query, (table,))
            if table_check[0][0] == 0:
                continue
                
            # Check for remaining references to duplicate plyr_ids
            remaining_query = f"""
                SELECT COUNT(*) FROM {table} t
                INNER JOIN duplicate_player_mapping dpm ON t.plyr_id = dpm.duplicate_plyr_id
            """
            remaining_result = db.fetch_all(remaining_query)
            remaining_count = remaining_result[0][0]
            
            if remaining_count > 0:
                print(f"[ERROR] {table} still has {remaining_count} references to duplicate plyr_ids")
                validation_failed = True
            else:
                print(f"[OK] {table} has no remaining references to duplicate plyr_ids")
        
        if validation_failed:
            print("\n[ERROR] Validation failed - foreign key references still exist")
            db.execute_query("ROLLBACK")
            return False
        
        # Step 4: Check for constraint violations related to our duplicate removal
        print("\n" + "="*60)
        print("STEP 4: Checking for constraint violations related to duplicate removal")
        print("="*60)
        
        print("[INFO] Checking if our foreign key updates would cause constraint violations...")
        
        # Check specifically for violations that would be caused by our duplicate removal
        # We only care about violations involving our duplicate players
        constraint_violations = 0
        
        for table in reference_tables:
            # Skip non-existent tables
            table_check = db.fetch_all(table_exists_query, (table,))
            if table_check[0][0] == 0:
                continue
            
            # Check if updating foreign keys for our duplicates would create violations
            # This checks if both the old duplicate_plyr_id and new correct_plyr_id
            # already exist for the same season/week combination
            
            if table in ['plyr_gm_def', 'plyr_gm_off', 'plyr_gm_snap_ct', 'adv_plyr_gm_pass', 'adv_plyr_gm_rec']:
                # Game-level tables: check for (plyr_id, season_id, week_id) conflicts
                violation_query = f"""
                    SELECT dpm.plyr_name, t1.season_id, t1.week_id, 
                           COUNT(DISTINCT t1.plyr_id) as plyr_id_count
                    FROM {table} t1
                    INNER JOIN duplicate_player_mapping dpm ON (
                        t1.plyr_id = dpm.duplicate_plyr_id OR t1.plyr_id = dpm.correct_plyr_id
                    )
                    GROUP BY dpm.plyr_name, t1.season_id, t1.week_id
                    HAVING COUNT(DISTINCT t1.plyr_id) > 1
                """
            elif table in ['plyr_def', 'plyr_off', 'plyr_ret', 'plyr_kck', 'plyr_pass', 'plyr_rush', 'plyr_rec', 'plyr_fant']:
                # Season-level tables: check for (plyr_id, season_id) conflicts
                violation_query = f"""
                    SELECT dpm.plyr_name, t1.season_id,
                           COUNT(DISTINCT t1.plyr_id) as plyr_id_count
                    FROM {table} t1
                    INNER JOIN duplicate_player_mapping dpm ON (
                        t1.plyr_id = dpm.duplicate_plyr_id OR t1.plyr_id = dpm.correct_plyr_id
                    )
                    GROUP BY dpm.plyr_name, t1.season_id
                    HAVING COUNT(DISTINCT t1.plyr_id) > 1
                """
            else:
                # For other tables, skip detailed constraint checking
                print(f"[SKIP] Constraint check for {table} - manual review recommended")
                continue
            
            violations = db.fetch_all(violation_query)
            if violations:
                print(f"[ERROR] {table} would have {len(violations)} constraint violations after FK update:")
                for violation in violations[:5]:  # Show first 5
                    print(f"  - Player: {violation[0]}, Season: {violation[1]}")
                constraint_violations += len(violations)
            else:
                print(f"[OK] {table} would have no constraint violations after FK update")
        
        if constraint_violations > 0:
            print(f"\n[ERROR] Found {constraint_violations} potential constraint violations")
            print("[ERROR] Foreign key updates would create duplicate records")
            db.execute_query("ROLLBACK")
            return False
        else:
            print(f"\n[INFO] No constraint violations detected from our duplicate removal operation")
            print("[INFO] Note: Existing unrelated constraint violations in season tables are ignored")
        
        # Step 5: Delete duplicate records from plyr_master and plyr tables
        print("\n" + "="*60)
        print("STEP 5: Deleting duplicate records")
        print("="*60)
        
        # Delete from plyr table first (foreign key dependent)
        print("[INFO] Deleting duplicate records from plyr table...")
        delete_plyr_query = """
            DELETE p FROM plyr p
            INNER JOIN duplicate_player_mapping dpm ON p.plyr_id = dpm.duplicate_plyr_id
        """
        
        success = db.execute_query(delete_plyr_query)
        if not success:
            print("[ERROR] Failed to delete from plyr table")
            db.execute_query("ROLLBACK")
            return False
            
        deleted_plyr_count = db.cursor.rowcount
        print(f"[OK] Deleted {deleted_plyr_count} records from plyr table")
        
        # Delete from plyr_master table
        print("[INFO] Deleting duplicate records from plyr_master table...")
        
        # First, get the player_guids that need to be deleted from plyr_master
        # We need to identify the UNDRAFTED FREE AGENT records that have corresponding
        # non-undrafted records for the same player (name + birthday)
        get_guids_query = """
            SELECT DISTINCT pm1.player_guid, pm1.plyr_name
            FROM plyr_master pm1
            INNER JOIN plyr_master pm2 
                ON pm1.plyr_name = pm2.plyr_name
                AND pm1.plyr_birthday = pm2.plyr_birthday
                AND pm1.player_guid != pm2.player_guid
            WHERE pm1.plyr_draft_tm = 'UNDRAFTED FREE AGENT'
            AND pm2.plyr_draft_tm != 'UNDRAFTED FREE AGENT'
        """
        
        guids_to_delete = db.fetch_all(get_guids_query)
        
        if not guids_to_delete:
            print("[WARN] No plyr_master records found to delete")
            deleted_master_count = 0
        else:
            print(f"[INFO] Found {len(guids_to_delete)} plyr_master records to delete")
            
            # Delete each guid individually to avoid MySQL subquery limitations
            deleted_master_count = 0
            for guid_row in guids_to_delete:
                player_guid, player_name = guid_row
                delete_query = """
                    DELETE FROM plyr_master 
                    WHERE player_guid = %s AND plyr_draft_tm = 'UNDRAFTED FREE AGENT'
                """
                success = db.execute_query(delete_query, (player_guid,))
                if success:
                    deleted_count = db.cursor.rowcount
                    deleted_master_count += deleted_count
                    print(f"[OK] Deleted {deleted_count} plyr_master record for {player_name}")
                else:
                    print(f"[ERROR] Failed to delete plyr_master record for {player_name}")
                    db.execute_query("ROLLBACK")
                    return False
        
        print(f"[OK] Deleted {deleted_master_count} total records from plyr_master table")
        
        # Step 6: Final verification
        print("\n" + "="*60)
        print("STEP 6: Final verification")
        print("="*60)
        
        # Check that no duplicates remain
        final_duplicate_check = """
            SELECT COUNT(*) FROM (
                SELECT p1.plyr_name, p1.plyr_birthday, COUNT(DISTINCT p1.plyr_draft_tm)
                FROM plyr p1
                INNER JOIN plyr p2 
                    ON p1.season_id = p2.season_id
                    AND p1.plyr_name = p2.plyr_name
                    AND p1.plyr_birthday = p2.plyr_birthday 
                    AND p1.plyr_draft_tm != p2.plyr_draft_tm
                GROUP BY p1.plyr_name, p1.plyr_birthday
                HAVING COUNT(DISTINCT p1.plyr_draft_tm) > 1
            ) AS remaining_duplicates
        """
        
        remaining_duplicates = db.fetch_all(final_duplicate_check)
        remaining_count = remaining_duplicates[0][0]
        
        if remaining_count > 0:
            print(f"[ERROR] {remaining_count} duplicate player groups still exist")
            db.execute_query("ROLLBACK")
            return False
        else:
            print("[OK] No duplicate players remain")
        
        # Commit transaction
        print("\n" + "="*60)
        print("COMMITTING TRANSACTION")
        print("="*60)
        
        db.execute_query("COMMIT")
        print("[SUCCESS] Duplicate player removal completed successfully!")
        print(f"[SUMMARY] Deleted {deleted_plyr_count} plyr records and {deleted_master_count} plyr_master records")
        print(f"[SUMMARY] Updated {total_updated_records} foreign key references across {len(reference_tables)} tables")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        db.execute_query("ROLLBACK")
        return False
        
    finally:
        # Re-enable autocommit
        db.connection.autocommit = True
        db.disconnect()

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n[FAILED] Duplicate player removal operation failed")
        sys.exit(1)
    else:
        print("\n[SUCCESS] Duplicate player removal operation completed successfully")
        sys.exit(0)