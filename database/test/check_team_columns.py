#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from db_utils import DatabaseConnector

def check_team_table_structure():
    """Check the structure of nfl_team table"""
    
    db = DatabaseConnector()
    
    try:
        if not db.connect():
            print("Failed to connect to database")
            return False
        
        print("=== NFL_TEAM TABLE STRUCTURE ===")
        db.cursor.execute("DESCRIBE nfl_team")
        columns = db.cursor.fetchall()
        print(f"Total columns: {len(columns)}")
        for col in columns:
            print(f"- {col[0]} {col[1]} {'NULL' if col[2] == 'YES' else 'NOT NULL'}")
        
        print("\n=== SAMPLE NFL_TEAM DATA ===")
        db.cursor.execute("SELECT * FROM nfl_team LIMIT 5")
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
    check_team_table_structure()