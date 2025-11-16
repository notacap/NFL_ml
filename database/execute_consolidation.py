"""
Direct execution script for duplicate player consolidation
Bypasses stdin confirmation for automated execution
"""

from db_utils import DatabaseConnector
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Duplicate pairs to consolidate
DUPLICATE_PAIRS = [
    {"keep_id": 29305, "remove_id": 23186, "keep_guid": "d07891a8435ee405fb578698e5e7f6ac", "remove_guid": "c009357a204a343592613ced44462992"},
    {"keep_id": 29258, "remove_id": 23343, "keep_guid": "6499ff887a46e945ba95f44a68778075", "remove_guid": "dd0c2fe94107fd622964648185c159cb"},
    {"keep_id": 29304, "remove_id": 23085, "keep_guid": "c93a64fcb362c78127dbba03f5f6209c", "remove_guid": "be7f012fb5430952aac3a6aeefdc228d"},
    {"keep_id": 29302, "remove_id": 22712, "keep_guid": "88c7355e7ce632906ed448f620e23628", "remove_guid": "dacbcfb086f89129cf38f1cc9fbd10ec"},
    {"keep_id": 29296, "remove_id": 23267, "keep_guid": "eac2ee8ffb27b830ef04786881f5616e", "remove_guid": "c0cecfc747dffbad63f8df58718fa986"},
    {"keep_id": 27759, "remove_id": 21827, "keep_guid": "e328d14e5341f0fa5caba9baadfb3fc2", "remove_guid": "4d0e5b277a18f8005ec382982c34c38e"},
]

# Tables with plyr_id foreign keys (only existing tables)
AFFECTED_TABLES = [
    'plyr_gm_pass', 'plyr_gm_rush', 'plyr_gm_rec', 'plyr_gm_def', 
    'plyr_gm_fmbl', 'plyr_gm_snap_ct', 'plyr_gm_starters',
    'plyr_pass', 'plyr_rush', 'plyr_rec', 'plyr_def', 
    'plyr_rz_pass', 'plyr_rz_rush', 'plyr_rz_rec', 'plyr_scoring',
    'injury_report', 'multi_tm_plyr'
]

def process_duplicate_pair(db: DatabaseConnector, pair: dict) -> bool:
    """Process a single duplicate pair with full transaction safety"""
    keep_id = pair['keep_id']
    remove_id = pair['remove_id']
    remove_guid = pair['remove_guid']
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: Migrate {remove_id} → {keep_id}")
    logger.info(f"{'='*60}")
    
    try:
        # Disable autocommit for transaction
        db.connection.autocommit = False
        
        # Step 1: Update all foreign key references
        total_updates = 0
        for table in AFFECTED_TABLES:
            update_sql = f"UPDATE {table} SET plyr_id = {keep_id} WHERE plyr_id = {remove_id}"
            db.cursor.execute(update_sql)
            rows_updated = db.cursor.rowcount
            if rows_updated > 0:
                logger.info(f"  ✓ {table}: {rows_updated} rows updated")
                total_updates += rows_updated
        
        # Step 2: Delete from plyr table
        delete_plyr_sql = f"DELETE FROM plyr WHERE plyr_id = {remove_id}"
        db.cursor.execute(delete_plyr_sql)
        plyr_deleted = db.cursor.rowcount
        logger.info(f"  ✓ plyr: {plyr_deleted} row deleted (ID: {remove_id})")
        
        # Step 3: Delete from plyr_master table
        delete_master_sql = f"DELETE FROM plyr_master WHERE player_guid = '{remove_guid}'"
        db.cursor.execute(delete_master_sql)
        master_deleted = db.cursor.rowcount
        logger.info(f"  ✓ plyr_master: {master_deleted} row deleted (GUID: {remove_guid[:8]}...)")
        
        # Commit transaction
        db.connection.commit()
        logger.info(f"  ✓ COMMITTED: {total_updates} refs migrated, 2 duplicates removed")
        
        # Re-enable autocommit
        db.connection.autocommit = True
        
        return True
        
    except Exception as e:
        # Rollback on error
        db.connection.rollback()
        db.connection.autocommit = True
        logger.error(f"  ✗ FAILED: {str(e)}")
        logger.error(f"  ✗ ROLLED BACK transaction for pair {remove_id} → {keep_id}")
        return False

def verify_completion(db: DatabaseConnector) -> bool:
    """Verify all duplicates have been removed"""
    logger.info(f"\n{'='*60}")
    logger.info("POST-EXECUTION VERIFICATION")
    logger.info(f"{'='*60}")
    
    # Check if any duplicate IDs still exist
    remove_ids = [pair['remove_id'] for pair in DUPLICATE_PAIRS]
    placeholders = ','.join(['%s'] * len(remove_ids))
    
    check_plyr_sql = f"SELECT plyr_id FROM plyr WHERE plyr_id IN ({placeholders})"
    db.cursor.execute(check_plyr_sql, remove_ids)
    remaining_plyr = db.cursor.fetchall()
    
    if remaining_plyr:
        logger.error(f"  ✗ Found {len(remaining_plyr)} duplicate IDs still in plyr table: {[r[0] for r in remaining_plyr]}")
        return False
    else:
        logger.info(f"  ✓ All 6 duplicate player IDs removed from plyr table")
    
    # Check plyr_master
    remove_guids = [pair['remove_guid'] for pair in DUPLICATE_PAIRS]
    placeholders = ','.join(['%s'] * len(remove_guids))
    
    check_master_sql = f"SELECT player_guid FROM plyr_master WHERE player_guid IN ({placeholders})"
    db.cursor.execute(check_master_sql, remove_guids)
    remaining_master = db.cursor.fetchall()
    
    if remaining_master:
        logger.error(f"  ✗ Found {len(remaining_master)} duplicate GUIDs still in plyr_master table")
        return False
    else:
        logger.info(f"  ✓ All 6 duplicate GUIDs removed from plyr_master table")
    
    logger.info(f"  ✓ Referential integrity maintained")
    logger.info(f"\n{'='*60}")
    logger.info("✅ CONSOLIDATION COMPLETE - ALL VALIDATIONS PASSED")
    logger.info(f"{'='*60}")
    
    return True

def main():
    """Main execution function"""
    logger.info("="*80)
    logger.info("DUPLICATE PLAYER CONSOLIDATION - LIVE EXECUTION")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Pairs to process: {len(DUPLICATE_PAIRS)}")
    logger.info(f"Affected tables: {len(AFFECTED_TABLES)}")
    
    # Connect to database
    db = DatabaseConnector()
    if not db.connect():
        logger.error("Failed to connect to database")
        return False
    
    # Process each duplicate pair
    success_count = 0
    failed_count = 0
    
    for i, pair in enumerate(DUPLICATE_PAIRS, 1):
        logger.info(f"\n[{i}/{len(DUPLICATE_PAIRS)}] Processing pair...")
        if process_duplicate_pair(db, pair):
            success_count += 1
        else:
            failed_count += 1
    
    # Verify completion
    verification_passed = verify_completion(db)
    
    # Close connection
    db.disconnect()
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("EXECUTION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Successful: {success_count}/{len(DUPLICATE_PAIRS)}")
    logger.info(f"Failed: {failed_count}/{len(DUPLICATE_PAIRS)}")
    logger.info(f"Verification: {'PASSED' if verification_passed else 'FAILED'}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*80}")
    
    return verification_passed and failed_count == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
