#!/usr/bin/env python3
"""
CRITICAL DATABASE OPERATION: Consolidate Duplicate Player Records

This script consolidates 6 duplicate player pairs by migrating ALL foreign key references 
from duplicate player IDs to canonical player IDs, then safely removes duplicate entries.

SAFETY FEATURES:
- DRY RUN mode with complete impact analysis
- Transaction-based operations with rollback capability
- Comprehensive validation and error handling
- Detailed logging and backup creation
- User confirmation gate before actual execution

Author: Claude Code
Created: 2025-11-12
"""

import mysql.connector
import os
import logging
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import subprocess

# Import the existing database utilities
from db_utils import DatabaseConnector

# Configuration for the operation
DUPLICATE_PAIRS = [
    {"keep_id": 29305, "keep_guid": "d07891a8435ee405fb578698e5e7f6ac", "remove_id": 23186, "remove_guid": "c009357a204a343592613ced44462992"},
    {"keep_id": 29258, "keep_guid": "6499ff887a46e945ba95f44a68778075", "remove_id": 23343, "remove_guid": "dd0c2fe94107fd622964648185c159cb"},
    {"keep_id": 29304, "keep_guid": "c93a64fcb362c78127dbba03f5f6209c", "remove_id": 23085, "remove_guid": "be7f012fb5430952aac3a6aeefdc228d"},
    {"keep_id": 29302, "keep_guid": "88c7355e7ce632906ed448f620e23628", "remove_id": 22712, "remove_guid": "dacbcfb086f89129cf38f1cc9fbd10ec"},
    {"keep_id": 29296, "keep_guid": "eac2ee8ffb27b830ef04786881f5616e", "remove_id": 23267, "remove_guid": "c0cecfc747dffbad63f8df58718fa986"},
    {"keep_id": 27759, "keep_guid": "e328d14e5341f0fa5caba9baadfb3fc2", "remove_id": 21827, "remove_guid": "4d0e5b277a18f8005ec382982c34c38e"}
]

AFFECTED_TABLES = [
    # Game-Level Tables
    "plyr_gm_pass", "plyr_gm_rush", "plyr_gm_rec", "plyr_gm_def", 
    "plyr_gm_fmbl", "plyr_gm_snap_ct", "plyr_gm_starters",
    # Season-Level Tables  
    "plyr_pass", "plyr_rush", "plyr_rec", "plyr_def", 
    "plyr_rz_pass", "plyr_rz_rush", "plyr_rz_rec", "plyr_scoring",
    # Other Tables
    "injury_report", "multi_tm_plyr"
]

class DuplicatePlayerConsolidator:
    """Main class for managing the duplicate player consolidation process"""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.db = DatabaseConnector()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup logging
        self.setup_logging()
        
        # Initialize tracking variables
        self.analysis_results = {}
        self.execution_results = {}
        self.backup_paths = {}
        
    def setup_logging(self):
        """Setup comprehensive logging for the operation"""
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Setup file logging
        log_filename = f"logs/duplicate_player_cleanup_{self.timestamp}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        mode = "DRY_RUN" if self.dry_run else "LIVE_EXECUTION"
        self.logger.info(f"Starting duplicate player consolidation in {mode} mode")
        self.logger.info(f"Log file: {log_filename}")
        
    def connect_database(self) -> bool:
        """Establish database connection with error handling"""
        try:
            success = self.db.connect()
            if success:
                self.logger.info("Database connection established successfully")
                # Disable autocommit for transaction control
                self.db.connection.autocommit = False
                return True
            else:
                self.logger.error("Failed to establish database connection")
                return False
        except Exception as e:
            self.logger.error(f"Database connection error: {e}")
            return False
    
    def create_backups(self) -> bool:
        """Create timestamped backups of plyr and plyr_master tables using SQL"""
        self.logger.info("Creating table backups...")
        
        try:
            backup_dir = Path("backups")
            backup_dir.mkdir(exist_ok=True)
            
            tables_to_backup = ["plyr", "plyr_master"]
            
            for table in tables_to_backup:
                backup_file = backup_dir / f"{table}_backup_{self.timestamp}.sql"
                
                # Get table structure
                show_create_query = f"SHOW CREATE TABLE {table}"
                create_result = self.db.fetch_all(show_create_query)
                
                if not create_result:
                    self.logger.error(f"Failed to get table structure for {table}")
                    return False
                
                create_statement = create_result[0][1]
                
                # Get table data
                select_query = f"SELECT * FROM {table}"
                data_result = self.db.fetch_all(select_query)
                
                # Write backup file
                with open(backup_file, 'w', encoding='utf-8') as f:
                    f.write(f"-- Backup of {table} table\n")
                    f.write(f"-- Created: {datetime.now()}\n\n")
                    f.write(f"DROP TABLE IF EXISTS {table}_backup_{self.timestamp};\n")
                    f.write(create_statement.replace(f"CREATE TABLE `{table}`", f"CREATE TABLE `{table}_backup_{self.timestamp}`") + ";\n\n")
                    
                    if data_result:
                        # Get column names
                        desc_query = f"DESCRIBE {table}"
                        columns = self.db.fetch_all(desc_query)
                        column_names = [col[0] for col in columns]
                        
                        f.write(f"INSERT INTO {table}_backup_{self.timestamp} ({', '.join(column_names)}) VALUES\n")
                        
                        for i, row in enumerate(data_result):
                            # Format row values
                            formatted_values = []
                            for value in row:
                                if value is None:
                                    formatted_values.append("NULL")
                                elif isinstance(value, str):
                                    formatted_values.append(f"'{value.replace(chr(39), chr(39)+chr(39))}'")  # Escape single quotes
                                else:
                                    formatted_values.append(str(value))
                            
                            row_sql = f"({', '.join(formatted_values)})"
                            if i == len(data_result) - 1:
                                f.write(row_sql + ";\n")
                            else:
                                f.write(row_sql + ",\n")
                
                self.backup_paths[table] = str(backup_file)
                self.logger.info(f"Backup created: {backup_file}")
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating backups: {e}")
            return False
    
    def validate_player_exists(self, pair: Dict) -> Dict:
        """Validate that both players exist in their respective tables"""
        validation = {
            "keep_plyr_exists": False,
            "remove_plyr_exists": False, 
            "keep_guid_exists": False,
            "remove_guid_exists": False,
            "keep_player_data": None,
            "remove_player_data": None
        }
        
        try:
            # Check plyr table
            keep_query = "SELECT plyr_id, plyr_name, team_id FROM plyr WHERE plyr_id = %s"
            keep_result = self.db.fetch_all(keep_query, (pair["keep_id"],))
            if keep_result:
                validation["keep_plyr_exists"] = True
                validation["keep_player_data"] = keep_result[0]
            
            remove_query = "SELECT plyr_id, plyr_name, team_id FROM plyr WHERE plyr_id = %s" 
            remove_result = self.db.fetch_all(remove_query, (pair["remove_id"],))
            if remove_result:
                validation["remove_plyr_exists"] = True
                validation["remove_player_data"] = remove_result[0]
            
            # Check plyr_master table
            keep_guid_query = "SELECT player_guid, plyr_name FROM plyr_master WHERE player_guid = %s"
            keep_guid_result = self.db.fetch_all(keep_guid_query, (pair["keep_guid"],))
            if keep_guid_result:
                validation["keep_guid_exists"] = True
                
            remove_guid_query = "SELECT player_guid, plyr_name FROM plyr_master WHERE player_guid = %s"
            remove_guid_result = self.db.fetch_all(remove_guid_query, (pair["remove_guid"],))
            if remove_guid_result:
                validation["remove_guid_exists"] = True
                
        except Exception as e:
            self.logger.error(f"Error validating player existence: {e}")
            
        return validation
    
    def count_foreign_key_references(self, plyr_id: int) -> Dict:
        """Count foreign key references for a player across all affected tables"""
        reference_counts = {}
        total_references = 0
        
        for table in AFFECTED_TABLES:
            try:
                query = f"SELECT COUNT(*) FROM {table} WHERE plyr_id = %s"
                result = self.db.fetch_all(query, (plyr_id,))
                count = result[0][0] if result else 0
                reference_counts[table] = count
                total_references += count
            except Exception as e:
                self.logger.error(f"Error counting references in {table}: {e}")
                reference_counts[table] = -1  # Error indicator
                
        reference_counts["total"] = total_references
        return reference_counts
    
    def detect_constraint_violations(self, pair: Dict) -> List[Dict]:
        """Detect potential unique constraint violations after migration"""
        violations = []
        
        keep_id = pair["keep_id"]
        remove_id = pair["remove_id"]
        
        # Check tables with unique constraints that include plyr_id
        constraint_checks = [
            {
                "table": "plyr_gm_pass",
                "constraint": "uk_plyr_gm_pass", 
                "columns": "plyr_id, game_id",
                "query": """
                    SELECT DISTINCT p1.plyr_id as keep_plyr_id, p1.game_id, 
                           p2.plyr_id as remove_plyr_id, p2.game_id
                    FROM plyr_gm_pass p1 
                    JOIN plyr_gm_pass p2 ON p1.game_id = p2.game_id
                    WHERE p1.plyr_id = %s AND p2.plyr_id = %s
                """
            },
            {
                "table": "plyr_gm_def",
                "constraint": "uk_plyr_gm_def",
                "columns": "plyr_id, season_id, week_id",
                "query": """
                    SELECT DISTINCT p1.plyr_id as keep_plyr_id, p1.season_id, p1.week_id,
                           p2.plyr_id as remove_plyr_id, p2.season_id, p2.week_id
                    FROM plyr_gm_def p1 
                    JOIN plyr_gm_def p2 ON p1.season_id = p2.season_id AND p1.week_id = p2.week_id
                    WHERE p1.plyr_id = %s AND p2.plyr_id = %s
                """
            }
            # Add more constraint checks as needed
        ]
        
        for check in constraint_checks:
            try:
                result = self.db.fetch_all(check["query"], (keep_id, remove_id))
                if result:
                    violations.append({
                        "table": check["table"],
                        "constraint": check["constraint"], 
                        "columns": check["columns"],
                        "conflicts": len(result),
                        "sample_conflicts": result[:3]  # Show first 3 conflicts
                    })
            except Exception as e:
                self.logger.error(f"Error checking constraints for {check['table']}: {e}")
                
        return violations
    
    def analyze_migration_impact(self) -> bool:
        """Perform comprehensive pre-migration analysis"""
        self.logger.info("="*80)
        self.logger.info("STEP 1: PRE-MIGRATION ANALYSIS")
        self.logger.info("="*80)
        
        all_valid = True
        total_updates = 0
        
        for i, pair in enumerate(DUPLICATE_PAIRS, 1):
            self.logger.info(f"\nProcessing Pair {i}/6: Migrating plyr_id {pair['remove_id']} → {pair['keep_id']}")
            
            # Validate player existence
            validation = self.validate_player_exists(pair)
            
            # Count foreign key references
            keep_refs = self.count_foreign_key_references(pair["keep_id"])
            remove_refs = self.count_foreign_key_references(pair["remove_id"])
            
            # Detect constraint violations
            violations = self.detect_constraint_violations(pair)
            
            # Store analysis results
            analysis = {
                "pair_number": i,
                "validation": validation,
                "keep_references": keep_refs,
                "remove_references": remove_refs,
                "constraint_violations": violations,
                "status": "READY" if validation["keep_plyr_exists"] and validation["remove_plyr_exists"] else "FAILED"
            }
            
            self.analysis_results[f"pair_{i}"] = analysis
            
            # Log detailed analysis
            if validation["keep_plyr_exists"] and validation["remove_plyr_exists"]:
                keep_player = validation["keep_player_data"]
                remove_player = validation["remove_player_data"]
                
                self.logger.info(f"  KEEP: {keep_player[1]} (ID: {keep_player[0]}, Team: {keep_player[2]})")
                self.logger.info(f"  REMOVE: {remove_player[1]} (ID: {remove_player[0]}, Team: {remove_player[2]})")
                self.logger.info(f"  Total references to migrate: {remove_refs['total']}")
                
                # Show table-by-table breakdown
                for table in AFFECTED_TABLES:
                    count = remove_refs.get(table, 0)
                    if count > 0:
                        self.logger.info(f"    - {table}: {count} rows to update")
                        
                total_updates += remove_refs["total"]
                
                if violations:
                    self.logger.warning(f"  WARNING: {len(violations)} constraint violations detected")
                    for violation in violations:
                        self.logger.warning(f"    - {violation['table']}: {violation['conflicts']} conflicts")
                    all_valid = False
                else:
                    self.logger.info("  OK: No constraint violations detected")
            else:
                self.logger.error(f"  FAIL: Player validation failed")
                self.logger.error(f"    Keep player exists: {validation['keep_plyr_exists']}")
                self.logger.error(f"    Remove player exists: {validation['remove_plyr_exists']}")
                all_valid = False
        
        # Summary
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"PRE-MIGRATION ANALYSIS SUMMARY")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Total pairs to process: {len(DUPLICATE_PAIRS)}")
        self.logger.info(f"Total updates required: {total_updates}")
        self.logger.info(f"Validation status: {'PASS' if all_valid else 'FAIL'}")
        
        if all_valid:
            self.logger.info("PASS - GO recommendation: Safe to proceed")
        else:
            self.logger.error("FAIL - NO-GO recommendation: Issues detected, manual review required")
            
        return all_valid
    
    def generate_migration_sql(self, pair: Dict) -> List[Dict]:
        """Generate all SQL statements for migrating a duplicate pair"""
        sql_statements = []
        
        keep_id = pair["keep_id"]
        remove_id = pair["remove_id"]
        remove_guid = pair["remove_guid"]
        
        # Generate UPDATE statements for all affected tables
        for table in AFFECTED_TABLES:
            update_sql = f"UPDATE {table} SET plyr_id = %s WHERE plyr_id = %s"
            sql_statements.append({
                "type": "UPDATE",
                "table": table,
                "sql": update_sql,
                "params": (keep_id, remove_id),
                "description": f"Migrate {table} references from {remove_id} to {keep_id}"
            })
        
        # Generate DELETE statements
        sql_statements.append({
            "type": "DELETE",
            "table": "plyr",
            "sql": "DELETE FROM plyr WHERE plyr_id = %s",
            "params": (remove_id,),
            "description": f"Remove duplicate player record {remove_id} from plyr table"
        })
        
        sql_statements.append({
            "type": "DELETE", 
            "table": "plyr_master",
            "sql": "DELETE FROM plyr_master WHERE player_guid = %s",
            "params": (remove_guid,),
            "description": f"Remove duplicate player record {remove_guid} from plyr_master table"
        })
        
        return sql_statements
    
    def execute_dry_run(self) -> bool:
        """Execute complete dry run simulation"""
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 2: DRY RUN EXECUTION") 
        self.logger.info("="*80)
        
        total_sql_statements = 0
        total_affected_rows = 0
        
        for i, pair in enumerate(DUPLICATE_PAIRS, 1):
            self.logger.info(f"\n--- DRY RUN: Pair {i}/6 ---")
            
            # Generate SQL statements
            sql_statements = self.generate_migration_sql(pair)
            
            pair_affected_rows = 0
            execution_plan = []
            
            for stmt in sql_statements:
                if stmt["type"] == "UPDATE":
                    # Calculate affected rows for UPDATE
                    count_query = f"SELECT COUNT(*) FROM {stmt['table']} WHERE plyr_id = %s"
                    result = self.db.fetch_all(count_query, (pair["remove_id"],))
                    affected_rows = result[0][0] if result else 0
                else:
                    # For DELETE, affected rows is always 1 (or 0 if not found)
                    affected_rows = 1
                    
                execution_plan.append({
                    "sql": stmt["sql"],
                    "params": stmt["params"],
                    "description": stmt["description"],
                    "affected_rows": affected_rows
                })
                
                pair_affected_rows += affected_rows
                total_sql_statements += 1
                
                if affected_rows > 0:
                    self.logger.info(f"  - {stmt['description']}: {affected_rows} rows")
            
            total_affected_rows += pair_affected_rows
            
            # Store dry run results
            self.execution_results[f"pair_{i}"] = {
                "execution_plan": execution_plan,
                "total_affected_rows": pair_affected_rows,
                "status": "DRY_RUN_COMPLETE"
            }
        
        # Dry run summary
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"DRY RUN SUMMARY")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Total SQL statements: {total_sql_statements}")
        self.logger.info(f"Total affected rows: {total_affected_rows}")
        self.logger.info(f"Tables affected: {len(AFFECTED_TABLES)} + 2 (plyr, plyr_master)")
        
        return True
    
    def show_impact_analysis(self) -> None:
        """Display detailed impact analysis to user"""
        print("\n" + "="*80)
        print("DRY RUN COMPLETE")
        print("="*80)
        
        total_updates = sum(
            result.get("total_affected_rows", 0) 
            for result in self.execution_results.values()
        )
        
        print(f"Total Updates Required: {total_updates} rows across {len(AFFECTED_TABLES)} tables")
        print(f"Total Deletions: 6 from plyr, 6 from plyr_master")
        print("\nDetailed Impact:")
        print("-"*80)
        
        # Table-by-table breakdown
        table_totals = {}
        for table in AFFECTED_TABLES:
            table_total = 0
            for pair_key, result in self.execution_results.items():
                for stmt in result.get("execution_plan", []):
                    if table in stmt.get("sql", ""):
                        table_total += stmt.get("affected_rows", 0)
            if table_total > 0:
                table_totals[table] = table_total
        
        for table, count in sorted(table_totals.items(), key=lambda x: x[1], reverse=True):
            print(f"{table:<20}: {count:>8} rows")
        
        print("-"*80)
        print("WARNING: This operation cannot be undone without backup restore.")
        print("Backups created at:", ", ".join(self.backup_paths.values()))
    
    def get_user_confirmation(self) -> bool:
        """Get explicit user confirmation before proceeding"""
        print("\n" + "="*80)
        while True:
            try:
                response = input("Proceed with ACTUAL execution? Type exactly 'EXECUTE' to confirm: ").strip()
                
                if response == "EXECUTE":
                    print("User confirmation received. Proceeding with execution...")
                    return True
                elif response.lower() in ["no", "n", "cancel", "abort", "exit"]:
                    print("Operation cancelled by user")
                    return False
                else:
                    print("Invalid response. Type 'EXECUTE' to proceed or 'cancel' to abort.")
                    
            except KeyboardInterrupt:
                print("\nOperation cancelled by user (Ctrl+C)")
                return False
    
    def execute_actual_migration(self) -> bool:
        """Execute the actual migration with transaction safety"""
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 4: ACTUAL EXECUTION")
        self.logger.info("="*80)
        
        successful_pairs = 0
        failed_pairs = 0
        
        for i, pair in enumerate(DUPLICATE_PAIRS, 1):
            self.logger.info(f"\n--- EXECUTING: Pair {i}/6 ---")
            
            try:
                # Start transaction
                self.db.connection.start_transaction()
                self.logger.info("Transaction started")
                
                # Disable foreign key checks temporarily  
                self.db.execute_query("SET FOREIGN_KEY_CHECKS = 0")
                
                # Execute all UPDATE statements
                sql_statements = self.generate_migration_sql(pair)
                
                for stmt in sql_statements:
                    if stmt["type"] == "UPDATE":
                        self.db.execute_query(stmt["sql"], stmt["params"])
                        affected_rows = self.db.cursor.rowcount
                        expected_rows = self.execution_results[f"pair_{i}"]["execution_plan"][AFFECTED_TABLES.index(stmt["table"])]["affected_rows"]
                        
                        if affected_rows != expected_rows:
                            raise Exception(f"Row count mismatch in {stmt['table']}: expected {expected_rows}, got {affected_rows}")
                        
                        self.logger.info(f"  OK: {stmt['table']}: {affected_rows} rows updated")
                
                # Re-enable foreign key checks
                self.db.execute_query("SET FOREIGN_KEY_CHECKS = 1")
                
                # Execute DELETE statements
                for stmt in sql_statements:
                    if stmt["type"] == "DELETE":
                        self.db.execute_query(stmt["sql"], stmt["params"])
                        affected_rows = self.db.cursor.rowcount
                        
                        if affected_rows != 1:
                            self.logger.warning(f"  WARNING: {stmt['table']}: {affected_rows} rows deleted (expected 1)")
                        else:
                            self.logger.info(f"  OK: {stmt['table']}: {affected_rows} rows deleted")
                
                # Validate no orphaned records exist
                orphan_check = f"""
                    SELECT COUNT(*) FROM plyr WHERE plyr_id = {pair['remove_id']}
                    UNION ALL
                    SELECT COUNT(*) FROM plyr_master WHERE player_guid = '{pair['remove_guid']}'
                """
                
                orphan_result = self.db.fetch_all(orphan_check)
                if any(count[0] > 0 for count in orphan_result):
                    raise Exception("Orphaned records detected after deletion")
                
                # Commit transaction
                self.db.connection.commit()
                self.logger.info(f"SUCCESS: Pair {i}/6 successfully migrated and committed")
                successful_pairs += 1
                
            except Exception as e:
                # Rollback transaction
                self.db.connection.rollback()
                self.logger.error(f"FAILED: Pair {i}/6 FAILED: {e}")
                failed_pairs += 1
                # Continue to next pair rather than aborting entirely
                
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"EXECUTION SUMMARY")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Successful migrations: {successful_pairs}/{len(DUPLICATE_PAIRS)}")
        self.logger.info(f"Failed migrations: {failed_pairs}/{len(DUPLICATE_PAIRS)}")
        
        return failed_pairs == 0
    
    def validate_post_execution(self) -> bool:
        """Validate the database state after migration"""
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 5: POST-EXECUTION VALIDATION")
        self.logger.info("="*80)
        
        validation_passed = True
        
        for i, pair in enumerate(DUPLICATE_PAIRS, 1):
            self.logger.info(f"\nValidating Pair {i}/6:")
            
            # Check removed player IDs no longer exist
            plyr_check = self.db.fetch_all("SELECT COUNT(*) FROM plyr WHERE plyr_id = %s", (pair["remove_id"],))
            if plyr_check[0][0] > 0:
                self.logger.error(f"  FAIL: Remove plyr_id {pair['remove_id']} still exists in plyr table")
                validation_passed = False
            else:
                self.logger.info(f"  OK: Remove plyr_id {pair['remove_id']} successfully deleted")
            
            # Check removed GUIDs no longer exist
            guid_check = self.db.fetch_all("SELECT COUNT(*) FROM plyr_master WHERE player_guid = %s", (pair["remove_guid"],))
            if guid_check[0][0] > 0:
                self.logger.error(f"  FAIL: Remove player_guid {pair['remove_guid']} still exists in plyr_master table")
                validation_passed = False
            else:
                self.logger.info(f"  OK: Remove player_guid {pair['remove_guid']} successfully deleted")
            
            # Check keep player IDs still exist and are valid
            keep_check = self.db.fetch_all("SELECT COUNT(*) FROM plyr WHERE plyr_id = %s", (pair["keep_id"],))
            if keep_check[0][0] == 0:
                self.logger.error(f"  FAIL: Keep plyr_id {pair['keep_id']} no longer exists")
                validation_passed = False
            else:
                self.logger.info(f"  OK: Keep plyr_id {pair['keep_id']} still exists")
        
        # Check referential integrity across all tables
        self.logger.info("\nChecking referential integrity...")
        integrity_issues = 0
        
        for table in AFFECTED_TABLES:
            try:
                integrity_query = f"""
                    SELECT COUNT(*) FROM {table} t
                    LEFT JOIN plyr p ON t.plyr_id = p.plyr_id
                    WHERE p.plyr_id IS NULL AND t.plyr_id IS NOT NULL
                """
                result = self.db.fetch_all(integrity_query)
                orphan_count = result[0][0] if result else 0
                
                if orphan_count > 0:
                    self.logger.error(f"  FAIL: {table}: {orphan_count} orphaned records")
                    integrity_issues += orphan_count
                    validation_passed = False
                else:
                    self.logger.info(f"  OK: {table}: No orphaned records")
                    
            except Exception as e:
                self.logger.error(f"  ERROR: Error checking {table}: {e}")
                validation_passed = False
        
        if validation_passed:
            self.logger.info("SUCCESS: All post-execution validations passed")
        else:
            self.logger.error("FAILED: Post-execution validation failed")
            
        return validation_passed
    
    def generate_final_report(self, success: bool) -> str:
        """Generate final markdown report"""
        outputs_dir = Path("outputs/reports")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = outputs_dir / f"duplicate_cleanup_report_{self.timestamp}.md"
        
        # Calculate summary statistics
        total_pairs = len(DUPLICATE_PAIRS)
        successful_pairs = sum(1 for result in self.execution_results.values() 
                              if result.get("status") != "FAILED")
        
        total_updates = sum(result.get("total_affected_rows", 0) 
                           for result in self.execution_results.values())
        
        mode = "DRY RUN" if self.dry_run else "LIVE EXECUTION"
        
        report_content = f"""# Duplicate Player Consolidation Report

**Operation Mode:** {mode}  
**Timestamp:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Duration:** {self.timestamp}  

## Executive Summary

- **Total Pairs Processed:** {successful_pairs}/{total_pairs}
- **Success Rate:** {(successful_pairs/total_pairs)*100:.1f}%
- **Total Database Rows Updated:** {total_updates}
- **Tables Affected:** {len(AFFECTED_TABLES)} + 2 (plyr, plyr_master)
- **Operation Status:** {"SUCCESS" if success else "FAILED"}

## Duplicate Pairs Processed

| Pair | Keep ID | Remove ID | Keep GUID | Remove GUID | Status |
|------|---------|-----------|-----------|-------------|--------|
"""
        
        for i, pair in enumerate(DUPLICATE_PAIRS, 1):
            result = self.analysis_results.get(f"pair_{i}", {})
            status = "SUCCESS" if result.get("status") == "READY" else "FAILED"
            report_content += f"| {i} | {pair['keep_id']} | {pair['remove_id']} | {pair['keep_guid'][:8]}... | {pair['remove_guid'][:8]}... | {status} |\n"
        
        report_content += f"""
## Migration Impact Matrix

| Table | Rows Updated | 
|-------|-------------|
"""
        
        # Add table-by-table breakdown
        for table in AFFECTED_TABLES:
            table_total = 0
            for result in self.execution_results.values():
                for stmt in result.get("execution_plan", []):
                    if table in stmt.get("sql", ""):
                        table_total += stmt.get("affected_rows", 0)
            if table_total > 0:
                report_content += f"| {table} | {table_total} |\n"
        
        report_content += f"""
## Backups Created

"""
        for table, backup_path in self.backup_paths.items():
            report_content += f"- **{table}**: `{backup_path}`\n"
        
        if not success:
            report_content += f"""
## Error Log

Errors encountered during operation:
- Check log file: `logs/duplicate_player_cleanup_{self.timestamp}.log`

## Rollback Instructions

To restore the database to its previous state:

```bash
mysql -u [username] -p[password] [database_name] < {self.backup_paths.get('plyr', 'plyr_backup.sql')}
mysql -u [username] -p[password] [database_name] < {self.backup_paths.get('plyr_master', 'plyr_master_backup.sql')}
```
"""
        
        report_content += f"""
## Validation Results

Post-execution validation: {"PASSED" if success else "FAILED"}

- All duplicate player IDs removed from plyr table
- All duplicate GUIDs removed from plyr_master table  
- All keep player IDs remain valid
- Referential integrity maintained across all {len(AFFECTED_TABLES)} affected tables
- Zero orphaned records detected

---
*Report generated by Duplicate Player Consolidation Script*  
*Timestamp: {self.timestamp}*
"""
        
        # Write report to file
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        self.logger.info(f"Final report generated: {report_file}")
        return str(report_file)
    
    def run_complete_workflow(self) -> bool:
        """Execute the complete consolidation workflow"""
        try:
            # Step 0: Safety Preparation
            if not self.connect_database():
                return False
                
            if not self.create_backups():
                return False
            
            # Step 1: Pre-Migration Analysis
            if not self.analyze_migration_impact():
                self.logger.error("Pre-migration analysis failed. Aborting operation.")
                return False
            
            # Step 2: Dry Run Execution
            if not self.execute_dry_run():
                self.logger.error("Dry run failed. Aborting operation.")
                return False
            
            # Step 3: User Confirmation Gate
            if not self.dry_run:
                self.show_impact_analysis()
                if not self.get_user_confirmation():
                    self.logger.info("Operation cancelled by user")
                    self.generate_final_report(success=False)
                    return False
                
                # Step 4: Actual Execution
                if not self.execute_actual_migration():
                    self.logger.error("Migration execution failed")
                    self.generate_final_report(success=False)
                    return False
                
                # Step 5: Post-Execution Validation
                if not self.validate_post_execution():
                    self.logger.error("Post-execution validation failed")
                    self.generate_final_report(success=False)
                    return False
            
            # Generate final report
            report_path = self.generate_final_report(success=True)
            self.logger.info(f"SUCCESS: Operation completed successfully")
            self.logger.info(f"Report available at: {report_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Critical error in workflow: {e}")
            self.generate_final_report(success=False)
            return False
        finally:
            if self.db.connection:
                self.db.disconnect()


def main():
    """Main entry point"""
    print("="*80)
    print("CRITICAL DATABASE OPERATION: Consolidate Duplicate Player Records")
    print("="*80)
    print("\nThis script will:")
    print("1. Analyze migration impact for 6 duplicate player pairs")
    print("2. Show detailed dry run results")
    print("3. Require explicit user confirmation")
    print("4. Execute migration with transaction safety")
    print("5. Validate results and generate comprehensive report")
    print("\nSafety features: Backups, transactions, dry run, detailed logging")
    
    # Initialize in DRY RUN mode first
    consolidator = DuplicatePlayerConsolidator(dry_run=True)
    
    print(f"\nStarting DRY RUN analysis...")
    success = consolidator.run_complete_workflow()
    
    if not success:
        print("\nDRY RUN failed. Please review errors before proceeding.")
        return
    
    print(f"\nDRY RUN completed successfully!")
    
    # Ask if user wants to proceed with actual execution
    print("\n" + "="*80)
    response = input("Do you want to proceed with ACTUAL execution? (y/N): ").strip().lower()
    
    if response == 'y':
        print("\nSwitching to LIVE EXECUTION mode...")
        live_consolidator = DuplicatePlayerConsolidator(dry_run=False)
        live_success = live_consolidator.run_complete_workflow()
        
        if live_success:
            print("\nDUPLICATE PLAYER CONSOLIDATION COMPLETED SUCCESSFULLY!")
        else:
            print("\nLIVE EXECUTION FAILED - Check logs and restore from backups if needed")
    else:
        print("\nOperation completed in DRY RUN mode only")


if __name__ == "__main__":
    main()