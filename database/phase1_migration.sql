-- Phase 1: Create plyr_master table and add player_guid columns
-- NFL Database Migration: Cross-Season Player Master Table

-- Step 1: Create plyr_master table
CREATE TABLE IF NOT EXISTS plyr_master (
    player_guid VARCHAR(64) PRIMARY KEY,
    plyr_name VARCHAR(255) NOT NULL,
    plyr_birthday DATE,
    plyr_height INT,
    plyr_college VARCHAR(255),
    plyr_draft_tm VARCHAR(255),
    plyr_draft_rd INT,
    plyr_draft_pick INT,
    plyr_draft_yr INT,
    primary_pos VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_plyr_name (plyr_name),
    INDEX idx_plyr_birthday (plyr_birthday),
    INDEX idx_draft_yr (plyr_draft_yr),
    UNIQUE KEY uk_player_identity (plyr_name, plyr_birthday, plyr_draft_tm)
);

-- Step 2: Add player_guid column to plyr table (allow NULL temporarily for migration)
ALTER TABLE plyr ADD COLUMN player_guid VARCHAR(64) AFTER plyr_id;

-- Step 3: Add player_guid column to multi_tm_plyr table (allow NULL temporarily for migration)
ALTER TABLE multi_tm_plyr ADD COLUMN player_guid VARCHAR(64) AFTER multi_tm_plyr_id;