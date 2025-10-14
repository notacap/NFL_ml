"""
Precision Standardization Migration Script
Executes schema changes for percentage column standardization
Generated: 2025-10-11
"""

import json
from datetime import datetime
from db_utils import DatabaseConnector

def create_backup_table(db: DatabaseConnector, table_name: str, backup_suffix: str) -> bool:
    """Create a backup of the specified table"""
    backup_name = f"{table_name}_backup_{backup_suffix}"
    try:
        # Drop backup if exists
        db.cursor.execute(f"DROP TABLE IF EXISTS {backup_name}")
        # Create backup
        db.cursor.execute(f"CREATE TABLE {backup_name} AS SELECT * FROM {table_name}")
        db.connection.commit()

        # Verify backup
        db.cursor.execute(f"SELECT COUNT(*) FROM {backup_name}")
        count = db.cursor.fetchone()[0]
        print(f"[OK] Backup created: {backup_name} ({count} rows)")
        return True
    except Exception as e:
        print(f"[FAIL] Failed to create backup {backup_name}: {e}")
        return False

def get_pre_migration_stats(db: DatabaseConnector, table_name: str, columns: list) -> dict:
    """Get statistics before migration"""
    stats = {}

    # Get row count
    db.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    stats['row_count'] = db.cursor.fetchone()[0]

    # Get max/min values for each column
    stats['max_values'] = {}
    stats['min_values'] = {}

    for col in columns:
        db.cursor.execute(f"SELECT MAX({col}), MIN({col}) FROM {table_name}")
        max_val, min_val = db.cursor.fetchone()
        stats['max_values'][col] = float(max_val) if max_val is not None else None
        stats['min_values'][col] = float(min_val) if min_val is not None else None

    return stats

def validate_migration(db: DatabaseConnector, table_name: str, columns: list, pre_stats: dict) -> dict:
    """Validate migration results"""
    post_stats = get_pre_migration_stats(db, table_name, columns)

    validation = {
        'status': 'PASSED',
        'issues': []
    }

    # Check row count
    if pre_stats['row_count'] != post_stats['row_count']:
        validation['status'] = 'FAILED'
        validation['issues'].append(f"Row count mismatch: {pre_stats['row_count']} -> {post_stats['row_count']}")

    # Check max values (allowing for rounding differences)
    for col in columns:
        pre_max = pre_stats['max_values'][col]
        post_max = post_stats['max_values'][col]

        if pre_max is not None and post_max is not None:
            if abs(pre_max - post_max) > 0.0001:  # Allow small floating point differences
                validation['status'] = 'WARNING'
                validation['issues'].append(f"{col} max changed: {pre_max:.6f} -> {post_max:.6f}")

    return validation, post_stats

def execute_phase_1_critical(db: DatabaseConnector) -> dict:
    """CRITICAL Priority: plyr_pass table - Active data overflow"""
    print("\n" + "="*80)
    print("PHASE 1 - CRITICAL: plyr_pass Table Migration")
    print("="*80)

    table_name = "plyr_pass"
    columns = ['plyr_pass_cmp_pct', 'plyr_pass_int_pct', 'plyr_pass_sk_pct', 'plyr_pass_prss_pct']

    result = {
        'table': table_name,
        'priority': 'CRITICAL',
        'backup_table': f'{table_name}_backup_20251011',
        'columns_modified': len(columns),
        'status': 'PENDING'
    }

    try:
        # 1. Create backup
        print("\n1. Creating backup table...")
        if not create_backup_table(db, table_name, '20251011'):
            result['status'] = 'FAILED'
            result['issues'] = ['Backup creation failed']
            return result

        # 2. Get pre-migration stats
        print("\n2. Collecting pre-migration statistics...")
        pre_stats = get_pre_migration_stats(db, table_name, columns)
        result['pre_migration'] = pre_stats
        print(f"   Row count: {pre_stats['row_count']}")
        for col, val in pre_stats['max_values'].items():
            print(f"   {col} MAX: {val}")

        # 3. Execute ALTER TABLE statements
        print("\n3. Executing schema changes...")
        db.cursor.execute("START TRANSACTION")

        migrations = [
            ("plyr_pass_cmp_pct", "DECIMAL(6,4)"),
            ("plyr_pass_int_pct", "DECIMAL(6,4)"),
            ("plyr_pass_sk_pct", "DECIMAL(6,4)"),
            ("plyr_pass_prss_pct", "DECIMAL(6,4)")
        ]

        for col_name, new_type in migrations:
            print(f"   Modifying {col_name} to {new_type}...")
            db.cursor.execute(f"ALTER TABLE {table_name} MODIFY COLUMN {col_name} {new_type}")

        db.connection.commit()
        print("   [OK] All columns modified successfully")

        # 4. Validate migration
        print("\n4. Validating migration...")
        validation, post_stats = validate_migration(db, table_name, columns, pre_stats)
        result['post_migration'] = post_stats
        result['validation_status'] = validation['status']
        result['issues'] = validation['issues']

        if validation['status'] == 'PASSED':
            result['status'] = 'SUCCESS'
            print("   [OK] Validation PASSED")
        else:
            result['status'] = validation['status']
            print(f"   [WARNING] Validation {validation['status']}")
            for issue in validation['issues']:
                print(f"     - {issue}")

    except Exception as e:
        db.connection.rollback()
        result['status'] = 'FAILED'
        result['issues'] = [str(e)]
        print(f"\n[FAIL] Migration failed: {e}")

    return result

def execute_phase_2_high(db: DatabaseConnector) -> dict:
    """HIGH Priority: plyr_gm_pass table - Consistency standardization"""
    print("\n" + "="*80)
    print("PHASE 2 - HIGH: plyr_gm_pass Table Migration")
    print("="*80)

    table_name = "plyr_gm_pass"
    columns = [
        'plyr_gm_pass_first_dwn_pct', 'plyr_gm_pass_drp_pct',
        'plyr_gm_pass_off_tgt_pct', 'plyr_gm_pass_prss_pct',
        'plyr_gm_pass_cmp_pct', 'plyr_gm_pass_td_pct',
        'plyr_gm_pass_int_pct', 'plyr_gm_pass_sk_pct'
    ]

    result = {
        'table': table_name,
        'priority': 'HIGH',
        'backup_table': f'{table_name}_backup_20251011',
        'columns_modified': len(columns),
        'status': 'PENDING'
    }

    try:
        # 1. Create backup
        print("\n1. Creating backup table...")
        if not create_backup_table(db, table_name, '20251011'):
            result['status'] = 'FAILED'
            result['issues'] = ['Backup creation failed']
            return result

        # 2. Get pre-migration stats
        print("\n2. Collecting pre-migration statistics...")
        pre_stats = get_pre_migration_stats(db, table_name, columns)
        result['pre_migration'] = pre_stats
        print(f"   Row count: {pre_stats['row_count']}")

        # 3. Execute ALTER TABLE statements
        print("\n3. Executing schema changes...")
        db.cursor.execute("START TRANSACTION")

        migrations = [
            # FLOAT(5,4) -> DECIMAL(6,4)
            ("plyr_gm_pass_first_dwn_pct", "DECIMAL(6,4)"),
            ("plyr_gm_pass_drp_pct", "DECIMAL(6,4)"),
            ("plyr_gm_pass_off_tgt_pct", "DECIMAL(6,4)"),
            ("plyr_gm_pass_prss_pct", "DECIMAL(6,4)"),
            # DECIMAL(7,4) -> DECIMAL(6,4)
            ("plyr_gm_pass_cmp_pct", "DECIMAL(6,4)"),
            ("plyr_gm_pass_td_pct", "DECIMAL(6,4)"),
            ("plyr_gm_pass_int_pct", "DECIMAL(6,4)"),
            ("plyr_gm_pass_sk_pct", "DECIMAL(6,4)")
        ]

        for col_name, new_type in migrations:
            print(f"   Modifying {col_name} to {new_type}...")
            db.cursor.execute(f"ALTER TABLE {table_name} MODIFY COLUMN {col_name} {new_type}")

        db.connection.commit()
        print("   [OK] All columns modified successfully")

        # 4. Validate migration
        print("\n4. Validating migration...")
        validation, post_stats = validate_migration(db, table_name, columns, pre_stats)
        result['post_migration'] = post_stats
        result['validation_status'] = validation['status']
        result['issues'] = validation['issues']

        if validation['status'] == 'PASSED':
            result['status'] = 'SUCCESS'
            print("   [OK] Validation PASSED")
        else:
            result['status'] = validation['status']
            print(f"   [WARNING] Validation {validation['status']}")
            for issue in validation['issues']:
                print(f"     - {issue}")

    except Exception as e:
        db.connection.rollback()
        result['status'] = 'FAILED'
        result['issues'] = [str(e)]
        print(f"\n[FAIL] Migration failed: {e}")

    return result

def execute_phase_3_medium(db: DatabaseConnector) -> dict:
    """MEDIUM Priority: tm_def_pass table - Optimization"""
    print("\n" + "="*80)
    print("PHASE 3 - MEDIUM: tm_def_pass Table Migration")
    print("="*80)

    table_name = "tm_def_pass"
    columns = ['tm_def_pass_cmp_pct']

    result = {
        'table': table_name,
        'priority': 'MEDIUM',
        'backup_table': f'{table_name}_backup_20251011',
        'columns_modified': len(columns),
        'status': 'PENDING'
    }

    try:
        # 1. Create backup
        print("\n1. Creating backup table...")
        if not create_backup_table(db, table_name, '20251011'):
            result['status'] = 'FAILED'
            result['issues'] = ['Backup creation failed']
            return result

        # 2. Get pre-migration stats
        print("\n2. Collecting pre-migration statistics...")
        pre_stats = get_pre_migration_stats(db, table_name, columns)
        result['pre_migration'] = pre_stats
        print(f"   Row count: {pre_stats['row_count']}")
        for col, val in pre_stats['max_values'].items():
            print(f"   {col} MAX: {val}")

        # 3. Execute ALTER TABLE statements
        print("\n3. Executing schema changes...")
        db.cursor.execute("START TRANSACTION")

        print(f"   Modifying tm_def_pass_cmp_pct to DECIMAL(5,4)...")
        db.cursor.execute(f"ALTER TABLE {table_name} MODIFY COLUMN tm_def_pass_cmp_pct DECIMAL(5,4)")

        db.connection.commit()
        print("   [OK] Column modified successfully")

        # 4. Validate migration
        print("\n4. Validating migration...")
        validation, post_stats = validate_migration(db, table_name, columns, pre_stats)
        result['post_migration'] = post_stats
        result['validation_status'] = validation['status']
        result['issues'] = validation['issues']

        if validation['status'] == 'PASSED':
            result['status'] = 'SUCCESS'
            print("   [OK] Validation PASSED")
        else:
            result['status'] = validation['status']
            print(f"   [WARNING] Validation {validation['status']}")
            for issue in validation['issues']:
                print(f"     - {issue}")

    except Exception as e:
        db.connection.rollback()
        result['status'] = 'FAILED'
        result['issues'] = [str(e)]
        print(f"\n[FAIL] Migration failed: {e}")

    return result

def main():
    """Execute all migration phases"""
    print("="*80)
    print("PRECISION STANDARDIZATION MIGRATION")
    print("Execution Date: 2025-10-11")
    print("="*80)

    # Initialize database connection
    db = DatabaseConnector()
    if not db.connect():
        print("[FAIL] Could not connect to database")
        return

    # Execute phases
    results = {
        'status': 'PENDING',
        'execution_date': '2025-10-11',
        'phases_completed': [],
        'tables_modified': []
    }

    # Phase 1: CRITICAL
    phase1_result = execute_phase_1_critical(db)
    results['tables_modified'].append(phase1_result)
    if phase1_result['status'] == 'SUCCESS':
        results['phases_completed'].append('CRITICAL')

    # Phase 2: HIGH
    phase2_result = execute_phase_2_high(db)
    results['tables_modified'].append(phase2_result)
    if phase2_result['status'] == 'SUCCESS':
        results['phases_completed'].append('HIGH')

    # Phase 3: MEDIUM
    phase3_result = execute_phase_3_medium(db)
    results['tables_modified'].append(phase3_result)
    if phase3_result['status'] == 'SUCCESS':
        results['phases_completed'].append('MEDIUM')

    # Overall status
    if len(results['phases_completed']) == 3:
        results['status'] = 'SUCCESS'
    elif len(results['phases_completed']) > 0:
        results['status'] = 'PARTIAL'
    else:
        results['status'] = 'FAILURE'

    # Save report
    report_path = "C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\database\\migration_report_20251011.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("MIGRATION COMPLETE")
    print("="*80)
    print(f"Overall Status: {results['status']}")
    print(f"Phases Completed: {', '.join(results['phases_completed'])}")
    print(f"Report saved to: {report_path}")

    # Close database connection
    db.disconnect()

if __name__ == "__main__":
    main()
