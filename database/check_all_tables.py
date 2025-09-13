#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from db_utils import DatabaseConnector

def check_tables():
    """Check the structure of all related tables"""
    
    db = DatabaseConnector()
    
    try:
        if not db.connect():
            print("Failed to connect to database")
            return False
        
        tables = ['nfl_season', 'nfl_week', 'nfl_team', 'tm_def']
        
        for table_name in tables:
            print(f"\n=== {table_name.upper()} TABLE STRUCTURE ===")
            db.cursor.execute(f"DESCRIBE {table_name}")
            columns = db.cursor.fetchall()
            print(f"Total columns: {len(columns)}")
            for col in columns:
                print(f"- {col[0]} {col[1]} {'NULL' if col[2] == 'YES' else 'NOT NULL'}")
            
            # Show sample data
            print(f"\n=== SAMPLE {table_name.upper()} DATA ===")
            db.cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            sample_data = db.cursor.fetchall()
            for row in sample_data:
                print(f"- {row}")
            
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    check_tables()