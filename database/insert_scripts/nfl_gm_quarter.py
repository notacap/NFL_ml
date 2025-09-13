#!/usr/bin/env python3
"""
NFL Game Quarter Table Creation Script

Creates the nfl_gm_quarter table with quarter reference data.
This table stores the standard NFL game quarters (1-4).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_utils import DatabaseConnector, create_table_if_not_exists

def create_nfl_gm_quarter_table(db: DatabaseConnector) -> bool:
    """
    Create nfl_gm_quarter table if it doesn't exist.
    
    Returns:
        bool: True if successful, False otherwise
    """
    create_table_sql = """
    CREATE TABLE nfl_gm_quarter (
        nfl_quarter_id INT PRIMARY KEY AUTO_INCREMENT,
        quarter INT NOT NULL UNIQUE,
        INDEX idx_quarter (quarter)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """
    
    return create_table_if_not_exists(db, 'nfl_gm_quarter', create_table_sql)

def insert_quarters(db: DatabaseConnector) -> bool:
    """
    Insert standard NFL quarters (1-4) and overtime (5) into the table.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        quarters = [1, 2, 3, 4, 5]  # 5 = Overtime
        quarter_data = [{'quarter': q} for q in quarters]
        
        # Use upsert to handle re-runs
        insert_sql = """
        INSERT INTO nfl_gm_quarter (quarter) 
        VALUES (%(quarter)s)
        ON DUPLICATE KEY UPDATE quarter = VALUES(quarter)
        """
        
        cursor = db.connection.cursor()
        cursor.executemany(insert_sql, quarter_data)
        db.connection.commit()
        
        print(f"Successfully inserted {len(quarters)} quarters")
        return True
        
    except Exception as e:
        print(f"Error inserting quarters: {e}")
        db.connection.rollback()
        return False

def main():
    """Main execution function."""
    db = DatabaseConnector()
    
    try:
        # Establish database connection
        if not db.connect():
            print("Failed to connect to database")
            return
        
        print("Creating nfl_gm_quarter table...")
        if not create_nfl_gm_quarter_table(db):
            print("Failed to create table")
            return
        
        print("Inserting quarter data...")
        if not insert_quarters(db):
            print("Failed to insert quarter data")
            return
        
        print("nfl_gm_quarter table setup completed successfully!")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
    finally:
        db.disconnect()

if __name__ == "__main__":
    main()