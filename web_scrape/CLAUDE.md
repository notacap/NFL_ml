# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the web scraping component of an NFL machine learning project. The goal is to create 50+ web scraping scripts that collect comprehensive NFL data from pro-football-reference.com. Currently, this is an ongoing project with shared utilities established and initial scrapers implemented.

## Project Structure

- `scripts/` - Contains all web scraping modules (target: 50+ scrapers)
- `scraped_data/` - Data organized by year and data type (e.g., `2024/schedule/`, `2024/stats/`)

## Development Status

**Current Implementation:**
- `scraper_utils.py` - Core utilities and shared configuration 
- `schedule.py` - NFL schedule scraper (first implementation)

**Target:** 50+ specialized scrapers for different NFL data types (team stats, player stats, game details, etc.)

## Core Architecture

### Shared Utilities (`scraper_utils.py`)

All scrapers should use the common utilities:
- `setup_driver()` - Configured headless Chrome with error suppression
- `scrape_table_with_links()` - Generic table scraping with link extraction
- `create_directories()` - Standardized data directory creation
- `save_data_to_csv()` - Consistent CSV output formatting

**Logging System:**
- `start_scraper_session(scraper_name)` - Initialize session logging
- `end_scraper_session()` - Finalize session and show summary
- `log_failed_table()` / `log_successful_table()` - Manual logging (auto-called by utilities)
- `show_recent_failures(hours)` - View recent failed table scrapes

### Configuration

Global scraping year configured in `scraper_utils.py:30`:
```python
YEAR = 2024
```

Base data directory: `C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data`

### Data Storage Convention

- Directory structure: `scraped_data/{YEAR}/{data_type}/`
- File format: CSV with descriptive names
- Consistent column naming and data formatting across scrapers
- Log files: `scraped_data/logs/scraper_log.json` (session tracking and failure analysis)

## Development Guidelines

### Creating New Scrapers

1. Import from `scraper_utils`: `from scraper_utils import *`
2. Use `scrape_table_with_links()` for table-based data extraction
3. Follow the established pattern: scrape → process → save
4. Implement robust error handling and filtering
5. Create specific `process_*_data()` functions for data transformation

**REQUIRED: Implement Logging in All New Scrapers**

Every new scraper MUST include session logging to track CSV creation failures:

```python
def main():
    """Main execution function."""
    # REQUIRED: Start logging session
    start_scraper_session("your_scraper_name")
    
    try:
        # Your scraping logic here
        print(f"Scraping {data_type} data for year: {YEAR}")
        
        # Create directories
        data_dir = create_directories(YEAR, "data_type")
        
        # Scrape and save data
        df = your_scrape_function(YEAR)
        if df is not None:
            save_data_to_csv(df, data_dir, "filename.csv")
        else:
            print("Failed to scrape data")
    
    finally:
        # REQUIRED: End logging session (shows summary)
        end_scraper_session()

if __name__ == "__main__":
    main()
```

**Key Points:**
- Always wrap main logic in try/finally with `start_scraper_session()` and `end_scraper_session()`
- Use descriptive scraper names (e.g., "team_stats", "player_rushing", "game_logs")
- The logging system automatically tracks CSV creation failures
- Failed tables are logged with timestamps, error reasons, and URLs for easy re-scraping

### Running Scrapers

Execute individual scrapers from the scripts directory:
```bash
python {scraper_name}.py
```

Each scraper automatically:
- Creates required directory structure
- Scrapes data for the configured year
- Applies data processing and filtering
- Saves results to standardized CSV format
- Logs session data and tracks CSV creation failures

**Viewing Logs:**
```bash
python view_logs.py recent      # Show failures from last 24 hours
python view_logs.py recent 6    # Show failures from last 6 hours  
python view_logs.py all         # Show all scraper sessions
```

## Dependencies

Core packages for all scrapers:
- `pandas` - Data manipulation and CSV output
- `selenium` - Web browser automation  
- `webdriver-manager` - Chrome driver management

## Architecture Patterns

- **Factory pattern**: Reusable `scrape_table_with_links()` function
- **Separation of concerns**: Extract → Process → Store pipeline
- **Configuration management**: Centralized constants and settings
- **Error handling**: Comprehensive timeout and exception management
- **Windows optimization**: Chrome driver configured for Windows environment