#!/usr/bin/env python3
"""
Phase 2 Test: Verify GUID generation and master record functions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_utils import DatabaseConnector, generate_player_guid, get_or_create_player_guid

def test_phase2():
    """Test Phase 2 utility functions."""
    
    print("=" * 80)
    print("PHASE 2 TEST: Testing GUID generation and master record functions")
    print("=" * 80)
    
    # Test 1: GUID generation
    print("\n[TEST 1] GUID Generation...")
    
    # Test with sample data
    test_players = [
        ("Patrick Mahomes", "1995-09-17", "Kansas City Chiefs"),
        ("Tom Brady", "1977-08-03", "New England Patriots"), 
        ("Aaron Rodgers", "1983-12-02", "Green Bay Packers"),
        ("Josh Allen", "1996-05-21", "Buffalo Bills")
    ]
    
    for name, birthday, draft_tm in test_players:
        guid = generate_player_guid(name, birthday, draft_tm)
        print(f"  {name:<20} -> {guid}")
    
    # Test deterministic behavior
    guid1 = generate_player_guid("Patrick Mahomes", "1995-09-17", "Kansas City Chiefs")
    guid2 = generate_player_guid("Patrick Mahomes", "1995-09-17", "Kansas City Chiefs")
    
    if guid1 == guid2:
        print("[OK] GUID generation is deterministic")
    else:
        print("[ERROR] GUID generation is not deterministic!")
        return False
    
    # Test with None values
    guid_none = generate_player_guid(None, None, None)
    print(f"  NULL values        -> {guid_none}")
    
    # Test 2: Database connectivity
    print("\n[TEST 2] Database connection...")
    db = DatabaseConnector()
    if not db.connect():
        print("[ERROR] Failed to connect to database")
        return False
    print("[OK] Database connection established")
    
    # Test 3: Master record creation (if plyr_master table exists)
    print("\n[TEST 3] Master record management...")
    if db.table_exists('plyr_master'):
        print("[OK] plyr_master table exists")
        
        # Test creating a test record
        try:
            test_guid = get_or_create_player_guid(
                db,
                plyr_name="Test Player",
                plyr_birthday="1990-01-01",
                plyr_draft_tm="UNDRAFTED FREE AGENT",
                plyr_height=72,
                plyr_college="Test University",
                primary_pos="QB"
            )
            print(f"[OK] Created test master record: {test_guid}")
            
            # Test getting existing record
            test_guid2 = get_or_create_player_guid(
                db,
                plyr_name="Test Player",
                plyr_birthday="1990-01-01", 
                plyr_draft_tm="UNDRAFTED FREE AGENT"
            )
            
            if test_guid == test_guid2:
                print("[OK] Existing record retrieval works correctly")
            else:
                print("[ERROR] GUID mismatch on retrieval")
                db.disconnect()
                return False
            
        except Exception as e:
            print(f"[ERROR] Master record test failed: {e}")
            db.disconnect()
            return False
    else:
        print("[WARNING] plyr_master table not found - Phase 1 may not be complete")
    
    db.disconnect()
    
    print("\n" + "=" * 80)
    print("PHASE 2 TEST COMPLETE: All functions working correctly")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = test_phase2()
    sys.exit(0 if success else 1)