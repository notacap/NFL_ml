# CSV Output Rules and Directory Structure

## Overview
This document defines the standardized rules for CSV output file naming, directory structure, and data organization for the NFL web scraping project.

## Directory Structure Rules

### Base Structure
```
scraped_data/
├── {YEAR}/
│   ├── {data_type}/
│   │   └── {filename}_{timestamp}.csv
│   └── ...
```

### Year Configuration
- The year is controlled by the `YEAR` variable in `scraper_utils.py` (line 31)
- Currently set to `2022`
- All scrapers automatically use this year for both URL construction and directory creation

### Data Type Directory Naming
The `data_type` directory should match the Python script name:

| Script File | Directory Name | Example Output |
|-------------|----------------|----------------|
| `schedule.py` | `schedule/` | `scraped_data/2022/schedule/` |
| `plyr_passing.py` | `plyr_passing/` | `scraped_data/2022/plyr_passing/` |
| `plyr_rushing.py` | `plyr_rushing/` | `scraped_data/2022/plyr_rushing/` |
| `tm_adv_passing.py` | `tm_adv_passing/` | `scraped_data/2022/tm_adv_passing/` |
| `roster_details.py` | `roster_details/` | `scraped_data/2022/roster_details/` |

## CSV File Naming Rules

### Base Filename Convention
- **Root CSV filename must match the Python script name** (without `.py` extension)
- Examples:
  - `schedule.py` → `schedule.csv`
  - `plyr_passing.py` → `plyr_passing.csv`
  - `tm_adv_passing.py` → `tm_adv_passing.csv`

### Timestamp Format
All CSV files automatically receive timestamps via `save_data_to_csv()`:
- Format: `{base_name}_{YYYYMMDD_HHMMSS}.csv`
- Example: `schedule_20250811_194501.csv`

### Special Cases
Some scrapers create multiple CSV files or use subdirectories:
- **Team Advanced Passing**: Creates subdirectories like `tm_adv_passing/tm_airyards/`
- **Roster Details**: Creates separate CSV files per team

## Implementation Guidelines

### Using Shared Utilities
All scrapers should use the standardized functions:
```python
from scraper_utils import *

# Create directory matching script name
data_dir = create_directories(YEAR, "script_name_here")

# Save with matching base filename
save_data_to_csv(df, data_dir, "script_name_here.csv")
```

### Directory Creation
The `create_directories(year, data_type)` function:
- Creates the full path: `ROOT_DATA_DIR/{year}/{data_type}/`
- Handles nested subdirectories automatically
- Creates missing directories as needed

### CSV Output Features
The `save_data_to_csv()` function automatically:
- Adds timestamps to prevent filename conflicts
- Handles empty DataFrames gracefully
- Provides console feedback on save location
- Uses consistent CSV formatting (no index column)

## Data Organization Benefits

1. **Consistent Structure**: Easy to locate data by year and type
2. **No Conflicts**: Timestamps prevent file overwrites
3. **Scalable**: Easy to add new scrapers following the same pattern
4. **Centralized Control**: Change year in one place affects all scrapers
5. **Clear Mapping**: Script name → directory name → CSV name relationship

## Root Data Directory
Base location: `C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\`