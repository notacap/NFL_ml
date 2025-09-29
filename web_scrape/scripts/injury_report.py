"""
NFL Injury Report Web Scraper

Scrapes NFL injury data for all teams from pro-football-reference.com
Extracts injury report information and saves individual CSV files per team and table.
"""

from scraper_utils import *
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Team abbreviations for URL construction
TEAM_ABBREVIATIONS = [
    'buf', 'mia', 'nyj', 'nwe', 'rav', 'pit', 'cin', 'cle',
    'htx', 'clt', 'jax', 'oti', 'kan', 'sdg', 'den', 'rai',
    'phi', 'was', 'dal', 'nyg', 'det', 'min', 'gnb', 'chi',
    'tam', 'atl', 'car', 'nor', 'ram', 'sea', 'crd', 'sfo'
]

# Team abbreviation to team name mapping
TEAM_NAME_MAPPING = {
    'buf': 'Buffalo_Bills',
    'mia': 'Miami_Dolphins',
    'nyj': 'New_York_Jets',
    'nwe': 'New_England_Patriots',
    'rav': 'Baltimore_Ravens',
    'pit': 'Pittsburgh_Steelers',
    'cin': 'Cincinnati_Bengals',
    'cle': 'Cleveland_Browns',
    'htx': 'Houston_Texans',
    'clt': 'Indianapolis_Colts',
    'jax': 'Jacksonville_Jaguars',
    'oti': 'Tennessee_Titans',
    'kan': 'Kansas_City_Chiefs',
    'sdg': 'Los_Angeles_Chargers',
    'den': 'Denver_Broncos',
    'rai': 'Las_Vegas_Raiders',
    'phi': 'Philadelphia_Eagles',
    'was': 'Washington_Commanders',
    'dal': 'Dallas_Cowboys',
    'nyg': 'New_York_Giants',
    'det': 'Detroit_Lions',
    'min': 'Minnesota_Vikings',
    'gnb': 'Green_Bay_Packers',
    'chi': 'Chicago_Bears',
    'tam': 'Tampa_Bay_Buccaneers',
    'atl': 'Atlanta_Falcons',
    'car': 'Carolina_Panthers',
    'nor': 'New_Orleans_Saints',
    'ram': 'Los_Angeles_Rams',
    'sea': 'Seattle_Seahawks',
    'crd': 'Arizona_Cardinals',
    'sfo': 'San_Francisco_49ers'
}

def scrape_team_injuries_with_dnp(driver, table_id):
    """
    Scrapes the team_injuries table and extracts DNP status from td class attributes.

    Args:
        driver: WebDriver instance
        table_id (str): ID of the table element

    Returns:
        pd.DataFrame: DataFrame with injury status and playing_status_week_N columns
    """
    table = wait_for_table(driver, table_id, DEFAULT_TIMEOUT)

    # Get the header row to determine column count and names
    thead = table.find_element(By.TAG_NAME, "thead")
    header_row = thead.find_elements(By.TAG_NAME, "tr")[-1]  # Get last header row
    header_cells = header_row.find_elements(By.TAG_NAME, "th")

    # Extract column names
    columns = [cell.text.strip() for cell in header_cells]

    # Get tbody rows
    tbody = table.find_element(By.TAG_NAME, "tbody")
    rows = tbody.find_elements(By.TAG_NAME, "tr")

    # Filter out header rows (those with class containing 'thead')
    data_rows = [row for row in rows if 'thead' not in (row.get_attribute('class') or '')]

    # Build data structure
    data = []
    for row in data_rows:
        row_data = []
        dnp_statuses = []

        # Get all cells (th and td)
        cells = row.find_elements(By.TAG_NAME, "th") + row.find_elements(By.TAG_NAME, "td")

        for i, cell in enumerate(cells):
            cell_text = cell.text.strip()
            row_data.append(cell_text)

            # Check for DNP status in td elements (skip first column which is Player name)
            if i > 0 and cell.tag_name == "td":
                cell_class = cell.get_attribute("class") or ""
                # Check if 'dnp' is in the class
                has_dnp = 'dnp' in cell_class.split()
                dnp_statuses.append(1 if has_dnp else 0)

        # Combine row data with DNP statuses
        data.append(row_data + dnp_statuses)

    # Create column names for playing_status columns
    num_week_columns = len(columns) - 1  # Exclude Player column
    playing_status_columns = [f"playing_status_week_{i+1}" for i in range(num_week_columns)]

    # Create DataFrame with all columns
    all_columns = columns + playing_status_columns
    df = pd.DataFrame(data, columns=all_columns)

    return df

def scrape_team_injury_tables(team_abbr, year):
    """
    Scrapes injury data tables for a specific team and year.

    Args:
        team_abbr (str): Team abbreviation
        year (int): Year to scrape

    Returns:
        dict: Dictionary containing DataFrames for each table, or None if failed
    """
    url = f"{BASE_URL}/teams/{team_abbr}/{year}_injuries.htm"
    driver = setup_driver()

    try:
        driver.get(url)

        # Define the three table IDs to scrape
        table_ids = [
            "team_injuries",
            "team_injuries_totals",
            f"{team_abbr}_injury_report"
        ]

        tables_data = {}

        for table_id in table_ids:
            try:
                # Special handling for team_injuries table
                if table_id == "team_injuries":
                    df = scrape_team_injuries_with_dnp(driver, table_id)
                else:
                    # Wait for table to load
                    table = wait_for_table(driver, table_id, DEFAULT_TIMEOUT)
                    # Convert table to DataFrame
                    df = table_to_dataframe(table)

                if df is not None and not df.empty:
                    tables_data[table_id] = df
                    print(f"Successfully scraped table '{table_id}' for {team_abbr}")
                else:
                    print(f"Table '{table_id}' is empty or not found for {team_abbr}")
                    log_failed_table(f"{team_abbr}_{table_id}", "Table empty or not found", url)

            except TimeoutException:
                print(f"Timeout waiting for table '{table_id}' to load for {team_abbr}")
                log_failed_table(f"{team_abbr}_{table_id}", "Timeout waiting for table to load", url)
            except Exception as e:
                print(f"Error scraping table '{table_id}' for {team_abbr}: {e}")
                log_failed_table(f"{team_abbr}_{table_id}", f"Error scraping table: {e}", url)

        return tables_data if tables_data else None

    except Exception as e:
        print(f"Error accessing injury page for {team_abbr}: {e}")
        log_failed_table(f"{team_abbr}_injury_page", f"Error accessing page: {e}", url)
        return None
    finally:
        driver.quit()

def process_injury_data(tables_data, team_name, team_abbr):
    """
    Processes the injury DataFrames and adds Team column.

    Args:
        tables_data (dict): Dictionary of DataFrames
        team_name (str): Team name
        team_abbr (str): Team abbreviation

    Returns:
        dict: Dictionary of processed DataFrames
    """
    if not tables_data:
        return None

    processed_tables = {}

    for table_id, df in tables_data.items():
        if df is None or df.empty:
            continue

        # Create a copy to avoid modifying original
        processed_df = df.copy()

        # Filter out header rows (common patterns)
        if not processed_df.empty and len(processed_df.columns) > 0:
            # Remove rows where first column contains common header indicators
            first_col = processed_df.iloc[:, 0].astype(str)
            processed_df = processed_df[
                ~first_col.str.contains('Player|Week|Date|Pos|Age', case=False, na=False)
            ].copy()

        # Add Team column with team name
        processed_df['Team'] = team_name.replace('_', ' ')

        processed_tables[table_id] = processed_df

    return processed_tables

def create_team_directory(year, week_number, team_name):
    """
    Creates directory structure for team injury data.

    Args:
        year (int): Year
        week_number (int): Week number
        team_name (str): Team name

    Returns:
        str: Path to team directory
    """
    base_dir = os.path.join(ROOT_DATA_DIR, str(year), "injury_report", f"week_{week_number}")
    team_dir = os.path.join(base_dir, team_name)

    if not os.path.exists(team_dir):
        os.makedirs(team_dir, exist_ok=True)

    return team_dir

def save_injury_tables(tables_data, team_dir, team_abbr):
    """
    Saves injury tables to CSV files.

    Args:
        tables_data (dict): Dictionary of processed DataFrames
        team_dir (str): Directory to save files
        team_abbr (str): Team abbreviation

    Returns:
        int: Number of successfully saved files
    """
    if not tables_data:
        return 0

    successful_saves = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for table_id, df in tables_data.items():
        if df is not None and not df.empty:
            # Create filename based on table ID
            filename = f"{table_id}_{timestamp}.csv"

            try:
                csv_path = os.path.join(team_dir, filename)
                df.to_csv(csv_path, index=False)
                print(f"Data saved to: {csv_path}")
                log_successful_table(f"{team_abbr}_{table_id}", csv_path)
                successful_saves += 1
            except Exception as e:
                print(f"Error saving CSV file for {table_id}: {e}")
                log_failed_table(f"{team_abbr}_{table_id}", f"Error saving CSV: {e}")

    return successful_saves

def scrape_all_injury_reports(year, week_number):
    """
    Scrapes injury reports for all teams.

    Args:
        year (int): Year to scrape
        week_number (int): Week number for directory structure

    Returns:
        dict: Summary of scraping results
    """
    results = {
        'successful_teams': 0,
        'failed_teams': 0,
        'total_files_saved': 0
    }

    for i, team_abbr in enumerate(TEAM_ABBREVIATIONS, 1):
        team_name = TEAM_NAME_MAPPING.get(team_abbr, f"Unknown_Team_{team_abbr}")
        print(f"\nScraping injury data for {team_name} ({team_abbr}) ({i}/{len(TEAM_ABBREVIATIONS)})")

        # Scrape injury tables for the team
        tables_data = scrape_team_injury_tables(team_abbr, year)

        if tables_data:
            # Process the data
            processed_tables = process_injury_data(tables_data, team_name, team_abbr)

            if processed_tables:
                # Create team directory
                team_dir = create_team_directory(year, week_number, team_name)

                # Save tables to CSV files
                files_saved = save_injury_tables(processed_tables, team_dir, team_abbr)

                if files_saved > 0:
                    results['successful_teams'] += 1
                    results['total_files_saved'] += files_saved
                    print(f"Successfully saved {files_saved} injury tables for {team_name}")
                else:
                    results['failed_teams'] += 1
                    print(f"Failed to save any data for {team_name}")
            else:
                results['failed_teams'] += 1
                print(f"Failed to process injury data for {team_name}")
        else:
            results['failed_teams'] += 1
            print(f"Failed to scrape injury data for {team_name}")

        # Add small delay between requests to avoid rate limiting
        if i < len(TEAM_ABBREVIATIONS):
            time.sleep(1)

    return results

def main():
    """Main execution function."""
    start_scraper_session("injury_report")

    try:
        print(f"Scraping NFL injury reports for year: {YEAR}, week: {WEEK_NUMBER}")
        print(f"Data will be saved to: {ROOT_DATA_DIR}\\{YEAR}\\injury_report\\week_{WEEK_NUMBER}")

        # Scrape all team injury reports
        results = scrape_all_injury_reports(YEAR, WEEK_NUMBER)

        # Print summary
        print(f"\n=== INJURY REPORT SCRAPING SUMMARY ===")
        print(f"Successful teams: {results['successful_teams']}/{len(TEAM_ABBREVIATIONS)}")
        print(f"Failed teams: {results['failed_teams']}/{len(TEAM_ABBREVIATIONS)}")
        print(f"Total CSV files saved: {results['total_files_saved']}")
        print("=" * 40)

    finally:
        end_scraper_session()

if __name__ == "__main__":
    main()