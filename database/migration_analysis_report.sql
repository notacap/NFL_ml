-- ============================================================================
-- UNIQUE CONSTRAINT MIGRATION ANALYSIS REPORT
-- ============================================================================
-- Analysis Date: 2025-11-09
-- Database: nfl_stats
-- 
-- MIGRATION PROPOSAL:
-- Change plyr table unique constraint from:
--   CURRENT:  UNIQUE KEY uk_player_season_pos_exp (plyr_name, season_id, plyr_pos, plyr_weight, plyr_yrs_played)
--   PROPOSED: UNIQUE KEY (plyr_name, plyr_birthdate, plyr_draft_tm)
--
-- RESULT: ❌ MIGRATION IS NOT SAFE
-- ============================================================================

-- 1. CHECK TABLE STRUCTURE AND COLUMN NAMES
DESCRIBE plyr;
-- Result: Column is named 'plyr_birthday' (not 'plyr_birthdate')

-- 2. CHECK FOR NULL VALUES IN PROPOSED CONSTRAINT COLUMNS
SELECT 
    'plyr_name' AS column_name,
    COUNT(*) AS null_count
FROM plyr 
WHERE plyr_name IS NULL OR plyr_name = ''
UNION ALL
SELECT 
    'plyr_birthday' AS column_name,
    COUNT(*) AS null_count
FROM plyr 
WHERE plyr_birthday IS NULL
UNION ALL
SELECT 
    'plyr_draft_tm' AS column_name,
    COUNT(*) AS null_count
FROM plyr 
WHERE plyr_draft_tm IS NULL OR plyr_draft_tm = '';
-- Result: 0 NULL values in all proposed constraint columns

-- 3. CHECK TOTAL RECORD COUNT
SELECT COUNT(*) AS total_players FROM plyr;
-- Result: 8,632 total player records

-- 4. IDENTIFY COLLISION GROUPS - MAIN ANALYSIS
-- This query finds groups of records that would violate the proposed unique constraint
SELECT 
    plyr_name,
    plyr_birthday, 
    plyr_draft_tm,
    COUNT(*) AS collision_count,
    GROUP_CONCAT(DISTINCT plyr_id ORDER BY plyr_id) AS player_ids,
    GROUP_CONCAT(DISTINCT season_id ORDER BY season_id) AS seasons,
    GROUP_CONCAT(DISTINCT plyr_pos ORDER BY plyr_pos) AS positions,
    GROUP_CONCAT(DISTINCT plyr_weight ORDER BY plyr_weight) AS weights,
    GROUP_CONCAT(DISTINCT plyr_yrs_played ORDER BY plyr_yrs_played) AS years_exp
FROM plyr 
GROUP BY plyr_name, plyr_birthday, plyr_draft_tm
HAVING COUNT(*) > 1
ORDER BY collision_count DESC, plyr_name;

-- CRITICAL RESULT: 2,421 collision groups found!

-- 5. EXAMPLE COLLISION ANALYSIS - A.J. Brown
SELECT 
    plyr_id,
    plyr_name,
    season_id,
    plyr_pos,
    plyr_weight,
    plyr_yrs_played,
    plyr_birthday,
    plyr_draft_tm,
    team_id
FROM plyr 
WHERE plyr_name = 'A.J. Brown'
ORDER BY season_id, plyr_id;

-- Result shows A.J. Brown has 4 records (one per season: 2022, 2023, 2024, 2025)
-- with identical (plyr_name, plyr_birthday, plyr_draft_tm) but different season_id

-- 6. PATTERN ANALYSIS - Understanding the data structure
SELECT 
    COUNT(DISTINCT plyr_name) AS unique_player_names,
    COUNT(*) AS total_records,
    ROUND(COUNT(*) / COUNT(DISTINCT plyr_name), 2) AS avg_records_per_player
FROM plyr;

-- Result: Shows players have multiple records (avg ~4 records per player)

-- 7. SEASON COVERAGE ANALYSIS
SELECT DISTINCT year FROM nfl_season ORDER BY year;
-- Result: Database covers 2022, 2023, 2024, 2025 seasons

-- 8. CHECK MULTI_TM_PLYR TABLE FOR SIMILAR ISSUES
SELECT COUNT(*) AS multi_tm_records FROM multi_tm_plyr;
-- Result: 295 records in multi_tm_plyr table

SELECT 
    plyr_name,
    plyr_birthday, 
    plyr_draft_tm,
    COUNT(*) AS collision_count,
    GROUP_CONCAT(DISTINCT plyr_id ORDER BY plyr_id) AS player_ids
FROM multi_tm_plyr 
GROUP BY plyr_name, plyr_birthday, plyr_draft_tm
HAVING COUNT(*) > 1
ORDER BY collision_count DESC;
-- Result: 14 collision groups in multi_tm_plyr table

-- ============================================================================
-- ANALYSIS CONCLUSIONS
-- ============================================================================

/*
FINDINGS:
1. The plyr table uses a season-based design where each player gets one record per season
2. 2,421 unique players would have constraint violations under the proposed change
3. The proposed constraint (plyr_name, plyr_birthday, plyr_draft_tm) lacks the season dimension
4. Column name is 'plyr_birthday' not 'plyr_birthdate' as proposed

ROOT CAUSE:
The database schema treats each player-season combination as a separate entity.
This allows tracking of changing player attributes (weight, years experience, team) over time.

MIGRATION SAFETY: ❌ NOT SAFE
- 2,421 constraint violations would occur
- All active players across multiple seasons would be affected
- Data loss or restructuring would be required

RECOMMENDATIONS:
1. KEEP CURRENT CONSTRAINT: (plyr_name, season_id, plyr_pos, plyr_weight, plyr_yrs_played)
2. ALTERNATIVE 1: Add season dimension: (plyr_name, plyr_birthday, plyr_draft_tm, season_id)  
3. ALTERNATIVE 2: Create separate player_identity table with proposed constraint
4. ALTERNATIVE 3: Restructure to have base player table + season-specific stats tables

IMPACT ASSESSMENT:
- Migration would require complete data restructuring
- Current application logic depends on season-based player records
- Foreign key relationships would need extensive updates
- Data integrity could be compromised during migration
*/