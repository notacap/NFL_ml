-- Migration Script: Remove plyr_draft_tm from plyr table unique key
-- Date: 2025-01-22
-- Purpose: Allow draft team corrections for players with incorrect "UNDRAFTED FREE AGENT" data

USE nfl_db;

-- Step 1: Drop the existing unique constraint
ALTER TABLE plyr DROP INDEX uk_player_season_pos_exp;

-- Step 2: Add the new unique constraint without plyr_draft_tm
ALTER TABLE plyr ADD UNIQUE KEY uk_player_season_pos_exp (plyr_name, season_id, plyr_birthday);

-- Step 3: Verify the change
SHOW CREATE TABLE plyr;
