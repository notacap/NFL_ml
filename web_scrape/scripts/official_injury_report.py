from scraper_utils import *
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

TEAM_URLS = {
    'Buffalo Bills': 'https://www.buffalobills.com/team/injury-report/',
    'Miami Dolphins': 'https://www.miamidolphins.com/team/injury-report/',
    'New England Patriots': 'https://www.patriots.com/team/injury-report/',
    'New York Jets': 'https://www.newyorkjets.com/team/injury-report/',
    'Baltimore Ravens': 'https://www.baltimoreravens.com/team/injury-report/',
    'Cincinnati Bengals': 'https://www.bengals.com/team/injury-report/',
    'Cleveland Browns': 'https://www.clevelandbrowns.com/team/injury-report/',
    'Pittsburgh Steelers': 'https://www.steelers.com/team/injury-report/',
    'Houston Texans': 'https://www.houstontexans.com/team/injury-report/',
    'Indianapolis Colts': 'https://www.colts.com/team/injury-report/',
    'Jacksonville Jaguars': 'https://www.jaguars.com/team/injury-report/',
    'Tennessee Titans': 'https://www.titansonline.com/team/injury-report/',
    'Denver Broncos': 'https://www.denverbroncos.com/team/injury-report/',
    'Kansas City Chiefs': 'https://www.chiefs.com/team/injury-report/',
    'Las Vegas Raiders': 'https://www.raiders.com/team/injury-report/',
    'Los Angeles Chargers': 'https://www.chargers.com/team/injury-report/',
    'Dallas Cowboys': 'https://www.dallascowboys.com/team/injury-report/',
    'New York Giants': 'https://www.giants.com/team/injury-report/',
    'Philadelphia Eagles': 'https://www.philadelphiaeagles.com/team/injury-report/',
    'Washington Commanders': 'https://www.commanders.com/team/injury-report/',
    'Chicago Bears': 'https://www.chicagobears.com/team/injury-report/',
    'Detroit Lions': 'https://www.detroitlions.com/team/injury-report/',
    'Green Bay Packers': 'https://www.packers.com/team/injury-report/',
    'Minnesota Vikings': 'https://www.vikings.com/team/injury-report/',
    'Atlanta Falcons': 'https://www.atlantafalcons.com/team/injury-report/',
    'Carolina Panthers': 'https://www.panthers.com/team/injury-report/',
    'New Orleans Saints': 'https://www.neworleanssaints.com/team/injury-report/',
    'Tampa Bay Buccaneers': 'https://www.buccaneers.com/team/injury-report/',
    'Arizona Cardinals': 'https://www.azcardinals.com/team/injury-report/',
    'Los Angeles Rams': 'https://www.therams.com/team/injury-report/',
    'San Francisco 49ers': 'https://www.49ers.com/team/injury-report/',
    'Seattle Seahawks': 'https://www.seahawks.com/team/injury-report/'
}

def scrape_official_injury_table(driver, url, team_name):
    try:
        driver.get(url)
        
        wait = WebDriverWait(driver, DEFAULT_TIMEOUT)
        table = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.d3-o-table.d3-o-table--row-striping")))
        
        table_html = table.get_attribute('outerHTML')
        df = pd.read_html(StringIO(table_html))[0]
        
        if df is not None and not df.empty:
            df['Team'] = team_name
            return df
        else:
            return None
            
    except TimeoutException:
        print(f"Timeout waiting for injury table to load for {team_name}")
        log_failed_table(f"{team_name}_official_injury", "Timeout waiting for table to load", url)
        return None
    except Exception as e:
        print(f"Error scraping injury data for {team_name}: {e}")
        log_failed_table(f"{team_name}_official_injury", f"Error scraping table: {e}", url)
        return None

def create_team_directory(year, week_number, team_name):
    base_dir = os.path.join(ROOT_DATA_DIR, str(year), "official_injury_report", f"week_{week_number + 1}")
    team_dir = os.path.join(base_dir, team_name.replace(' ', '_'))
    
    if not os.path.exists(team_dir):
        os.makedirs(team_dir, exist_ok=True)
    
    return team_dir

def scrape_all_official_injuries(year, week_number):
    results = {
        'successful_teams': 0,
        'failed_teams': 0,
        'total_files_saved': 0
    }
    
    driver = setup_driver()
    
    try:
        team_count = len(TEAM_URLS)
        for i, (team_name, url) in enumerate(TEAM_URLS.items(), 1):
            print(f"\nScraping official injury data for {team_name} ({i}/{team_count})")
            
            df = scrape_official_injury_table(driver, url, team_name)
            
            if df is not None and not df.empty:
                team_dir = create_team_directory(year, week_number, team_name)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"official_injury_report_{timestamp}.csv"
                csv_path = os.path.join(team_dir, filename)
                
                try:
                    df.to_csv(csv_path, index=False)
                    print(f"Data saved to: {csv_path}")
                    log_successful_table(f"{team_name}_official_injury", csv_path, url)
                    results['successful_teams'] += 1
                    results['total_files_saved'] += 1
                except Exception as e:
                    print(f"Error saving CSV for {team_name}: {e}")
                    log_failed_table(f"{team_name}_official_injury", f"Error saving CSV: {e}", url)
                    results['failed_teams'] += 1
            else:
                results['failed_teams'] += 1
                print(f"Failed to scrape injury data for {team_name}")
            
            if i < team_count:
                time.sleep(1)
    
    finally:
        driver.quit()
    
    return results

def main():
    start_scraper_session("official_injury_report")
    
    try:
        print(f"Scraping official NFL injury reports for year: {YEAR}, week: {WEEK_NUMBER}")
        print(f"Data will be saved to: {ROOT_DATA_DIR}\\{YEAR}\\official_injury_report\\week_{WEEK_NUMBER + 1}")
        
        results = scrape_all_official_injuries(YEAR, WEEK_NUMBER)
        
        print(f"\n=== OFFICIAL INJURY REPORT SCRAPING SUMMARY ===")
        print(f"Successful teams: {results['successful_teams']}/{len(TEAM_URLS)}")
        print(f"Failed teams: {results['failed_teams']}/{len(TEAM_URLS)}")
        print(f"Total CSV files saved: {results['total_files_saved']}")
        print("=" * 50)
        
    finally:
        end_scraper_session()

if __name__ == "__main__":
    main()
