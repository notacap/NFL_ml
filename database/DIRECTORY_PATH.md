# DIRECTORY_PATH.md

This document explains how configuration variables in `db_utils.py` determine CSV source file directory paths for the NFL database import scripts.

## Configuration Variables

The following variables in `db_utils.py` control data processing:

```python
YEAR = 2024          # Current NFL season year
WEEK = 18            # Specific week for single-week operations
WEEK_START = 1       # Starting week for range operations
WEEK_END = 1         # Ending week for range operations
```

## Directory Path Structure

The base path for all scraped data follows this pattern:
```
C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\{YEAR}\
```

### Player Data (Season-Level)

**Scripts:** `plyr.py`, `multi_tm_plyr.py`

**Path Pattern:**
```
C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\{YEAR}\plyr\plyr_clean\{WEEK}\
```

**Examples:**
- YEAR=2024, WEEK=18: `C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\2024\plyr\plyr_clean\18\`
- YEAR=2023, WEEK=1: `C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\2023\plyr\plyr_clean\1\`

**Behavior:**
- Uses the `WEEK` variable to locate the specific week directory
- Finds the **most recently created** CSV file in that directory using `get_most_recent_csv_file()`
- Player data represents season-long statistics up to that week

### Game Data (Weekly-Level)

**Scripts:** `plyr_gm_def.py` (and other game-level scripts)

**Path Pattern:**
```
C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\{YEAR}\games\week_{week}.0\
```

**Examples:**
- YEAR=2024, processing weeks 1-3:
  - `C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\2024\games\week_1.0\`
  - `C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\2024\games\week_2.0\`
  - `C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\2024\games\week_3.0\`

**Behavior:**
- Uses `WEEK_START` and `WEEK_END` to process a range of weeks
- Processes **ALL** CSV files matching the pattern `*_gm_plyr_def_*.csv` in each week directory
- Each CSV file contains game-specific data for individual matchups

## Key Differences: Season vs Weekly Data

### Season-Level Data (plyr.py, multi_tm_plyr.py)
- **Purpose:** Player profiles and season statistics
- **Frequency:** Updated periodically (weekly snapshots)
- **Directory:** Uses single `WEEK` variable
- **File Selection:** Most recent file only
- **Data Scope:** Cumulative season stats through specified week

### Weekly Data (plyr_gm_def.py)
- **Purpose:** Individual game performance statistics
- **Frequency:** One dataset per game per week
- **Directory:** Uses `WEEK_START` to `WEEK_END` range
- **File Selection:** All matching files in each week directory
- **Data Scope:** Specific game performances

## File Naming Conventions

### Player Data Files
```
{timestamp}_players_data.csv
```
- Located in: `plyr\plyr_clean\{WEEK}\`
- Script selects most recent by creation time

### Game Defense Files
```
{Team1}_{Team2}_wk{week}.0_{year}_gm_plyr_def_{timestamp}.csv
```
- Located in: `games\week_{week}.0\`
- Example: `Atlanta_Falcons_Pittsburgh_Steelers_wk1.0_2024_gm_plyr_def_20250817_175033.csv`

## Configuration Examples

### Process Single Week of Games
```python
YEAR = 2024
WEEK_START = 1
WEEK_END = 1
```
Result: Processes game data from `week_1.0` directory only

### Process Multiple Weeks
```python
YEAR = 2024
WEEK_START = 1
WEEK_END = 4
```
Result: Processes game data from `week_1.0`, `week_2.0`, `week_3.0`, `week_4.0`

### Process Full Season Players
```python
YEAR = 2024
WEEK = 18  # Final week
```
Result: Uses player data from `plyr\plyr_clean\18\` directory

## Script Behavior Summary

| Script | Uses Variable | Path Type | File Selection |
|--------|---------------|-----------|----------------|
| `plyr.py` | `YEAR`, `WEEK` | Single directory | Most recent file |
| `multi_tm_plyr.py` | `YEAR`, `WEEK` | Single directory | Most recent file |
| `plyr_gm_def.py` | `YEAR`, `WEEK_START`, `WEEK_END` | Multiple directories | All matching files |

## Important Notes

1. **Decimal Week Numbers:** Week directories use `.0` suffix (e.g., `week_1.0`)
2. **File Discovery:** Game scripts search for missing directories and warn if not found
3. **Error Handling:** Scripts continue processing other weeks if individual weeks fail
4. **Team Identification:** Game files encode team matchups in filenames for proper game_id lookup