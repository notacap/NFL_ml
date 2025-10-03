import os
import glob
import csv
import json
import requests
from datetime import datetime
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clean_utils import YEAR, WEEK

ACTIVE_PLAYERS_DIR = r"C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\all_active_players"
PLYR_RAW_DIR = rf"C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\{YEAR}\plyr\plyr_raw\{WEEK}"
CACHE_FILE = os.path.join(os.path.dirname(__file__), "player_match_cache.json")

TEAM_ABBR_TO_FULL = {
    'ARI': 'Arizona Cardinals',
    'ATL': 'Atlanta Falcons',
    'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills',
    'CAR': 'Carolina Panthers',
    'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals',
    'CLE': 'Cleveland Browns',
    'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos',
    'DET': 'Detroit Lions',
    'GNB': 'Green Bay Packers',
    'HOU': 'Houston Texans',
    'IND': 'Indianapolis Colts',
    'JAX': 'Jacksonville Jaguars',
    'KAN': 'Kansas City Chiefs',
    'LVR': 'Las Vegas Raiders',
    'LAC': 'Los Angeles Chargers',
    'LAR': 'Los Angeles Rams',
    'MIA': 'Miami Dolphins',
    'MIN': 'Minnesota Vikings',
    'NWE': 'New England Patriots',
    'NOR': 'New Orleans Saints',
    'NYG': 'New York Giants',
    'NYJ': 'New York Jets',
    'PHI': 'Philadelphia Eagles',
    'PIT': 'Pittsburgh Steelers',
    'SFO': 'San Francisco 49ers',
    'SEA': 'Seattle Seahawks',
    'TAM': 'Tampa Bay Buccaneers',
    'TEN': 'Tennessee Titans',
    'WAS': 'Washington Commanders'
}

def get_latest_file(directory, prefix=''):
    if prefix:
        files = glob.glob(os.path.join(directory, f"{prefix}*.csv"))
    else:
        files = glob.glob(os.path.join(directory, "*.csv"))
    if not files:
        return None
    return max(files, key=os.path.getctime)

def clean_player_name(name):
    """Clean player name by removing suffixes and periods"""
    # Remove suffixes
    suffixes = [' I', ' II', ' III', ' IV', ' V', ' Jr', ' Sr', ' Jr.', ' Sr.']
    name = name.upper()
    for suffix in suffixes:
        if name.endswith(suffix.upper()):
            name = name[:-len(suffix)]
    # Remove periods
    name = name.replace('.', '')
    return name.strip()

def convert_height_to_inches(height_str):
    """Convert height string (e.g., "6' 0"") to inches"""
    try:
        match = re.match(r"(\d+)'\s*(\d+)\"*", height_str)
        if match:
            feet, inches = map(int, match.groups())
            return (feet * 12) + inches
    except:
        pass
    return ''

def convert_weight(weight_str):
    """Convert weight string (e.g., "300 lbs") to integer"""
    try:
        return int(weight_str.split()[0])
    except:
        return ''

def convert_experience(exp_str):
    """Convert experience string to integer"""
    if not exp_str:
        return ''
    if exp_str.lower() == 'rookie':
        return 0
    try:
        return int(re.search(r'(\d+)', exp_str).group(1))
    except:
        return ''

def convert_birthdate(dob_str):
    if not dob_str:
        return ''
    try:
        match = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})', dob_str)
        if match:
            day, month, year = match.groups()
            return f"{month}/{day}/{year}"
    except:
        pass
    return ''

def parse_draft_info(draft_str):
    if not draft_str:
        return '', '', '', ''
    try:
        match = re.search(r'(\d{4}): Rd (\d+), Pk (\d+) \(([A-Z]+)\)', draft_str)
        if match:
            year = int(match.group(1))
            round_num = int(match.group(2))
            pick_num = int(match.group(3))
            team_abbr = match.group(4)
            return year, round_num, pick_num, team_abbr
    except:
        pass
    return '', '', '', ''

def get_player_data(player_id):
    """Fetch player data from ESPN API"""
    url = f"https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{player_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except:
        return None

def create_output_file(raw_file_path):
    base_name = os.path.basename(raw_file_path)
    name_without_ext = os.path.splitext(base_name)[0]
    output_dir = os.path.dirname(raw_file_path)
    output_file = os.path.join(output_dir, f"{name_without_ext}_updated.csv")

    import shutil
    shutil.copy2(raw_file_path, output_file)
    return output_file

def update_raw_csv(raw_file_path, updates):
    with open(raw_file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    for row in rows:
        key = (clean_player_name(row['plyr_name']), row['team_name'])
        if key in updates:
            for field, value in updates[key].items():
                if field in fieldnames:
                    if not row.get(field) or row[field].strip() == '':
                        row[field] = value

    with open(raw_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def load_match_cache():
    """Load cached player matches from file"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_match_cache(cache):
    """Save player match cache to file"""
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

def find_player_matches(cleaned_name, active_players):
    """Find all potential matches for a player name"""
    matches = {}
    if cleaned_name in active_players:
        for player_id in active_players[cleaned_name]:
            try:
                if int(player_id) >= 8439:
                    matches[player_id] = cleaned_name
            except (ValueError, TypeError):
                continue
    return matches

def process_player_matches(row, matches, active_players_data, match_cache):
    """Process matches and handle user input if needed"""
    if len(matches) == 0:
        return None, False

    if len(matches) == 1:
        player_id = list(matches.keys())[0]
        return player_id, False

    # Check cache first
    cleaned_name = clean_player_name(row['plyr_name'])
    cache_key = f"{cleaned_name}|{row['team_name']}"

    if cache_key in match_cache:
        cached_id = match_cache[cache_key]
        # Verify cached ID is still in the matches
        if cached_id in matches:
            print(f"Using cached match for {row['plyr_name']} -> ID: {cached_id}")
            return cached_id, False
        else:
            print(f"Warning: Cached match for {row['plyr_name']} no longer valid")

    # Multiple matches found - prompt user
    print(f"\nMultiple potential matches found for:")
    print(f"Unmatched player: {row['plyr_name']}")
    print(f"Team: {row['team_name']}")
    print(f"Position: {row['pos']}")
    print("\nPotential matches:")

    # Display all potential matches
    for idx, (player_id, name) in enumerate(matches.items(), 1):
        print(f"\n{idx}. {name}")
        print(f"   ID: {player_id}")

        # Get additional info from active_players_data if available
        player_info = next((p for p in active_players_data if p['player_id'] == player_id), None)
        if player_info:
            print(f"   Display Name: {player_info['display_name']}")
            print(f"   Full Name: {player_info['full_name']}")

    print("\n0. Skip this player")

    # Get user input
    while True:
        try:
            choice = input("\nEnter the number of the correct match (or 0 to skip): ")
            choice = int(choice)
            if 0 <= choice <= len(matches):
                break
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    if choice == 0:
        return None, False

    selected_id = list(matches.keys())[choice - 1]

    # Save to cache
    match_cache[cache_key] = selected_id

    return selected_id, True

def main():
    active_players_file = get_latest_file(ACTIVE_PLAYERS_DIR, "all_players")
    raw_file = get_latest_file(PLYR_RAW_DIR, "plyr_raw")

    if not all([active_players_file, raw_file]):
        print("Could not find required files")
        return

    output_file = create_output_file(raw_file)
    print(f"Created output file: {output_file}")

    # Load match cache
    match_cache = load_match_cache()
    cache_updated = False

    active_players = {}
    active_players_data = []
    with open(active_players_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(row['player_id']) >= 8439:
                    cleaned_name = clean_player_name(row['display_name'])
                    if cleaned_name not in active_players:
                        active_players[cleaned_name] = []
                    active_players[cleaned_name].append(row['player_id'])
                    active_players_data.append(row)
            except (ValueError, TypeError):
                continue

    updates = {}
    with open(output_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned_name = clean_player_name(row['plyr_name'])

            matches = find_player_matches(cleaned_name, active_players)

            player_id, was_cached = process_player_matches(row, matches, active_players_data, match_cache)
            if was_cached:
                cache_updated = True

            if player_id:
                print(f"Processing {row['plyr_name']}...")

                player_data = get_player_data(player_id)
                if player_data and 'athlete' in player_data:
                    athlete = player_data['athlete']

                    height = convert_height_to_inches(athlete.get('displayHeight', ''))
                    weight = convert_weight(athlete.get('displayWeight', ''))
                    experience = convert_experience(athlete.get('displayExperience', ''))
                    display_draft = athlete.get('displayDraft', '')
                    if display_draft:
                        draft_yr, draft_rnd, draft_sel, draft_team_abbr = parse_draft_info(display_draft)
                        draft_team = TEAM_ABBR_TO_FULL.get(draft_team_abbr, '') if draft_team_abbr else ''
                    else:
                        draft_yr, draft_rnd, draft_sel, draft_team_abbr = '', '', '', ''
                        draft_team = 'Undrafted Free Agent'
                    birthdate = convert_birthdate(athlete.get('displayDOB', ''))
                    college = athlete.get('college', {}).get('name', '') if athlete.get('college') else ''

                    key = (cleaned_name, row['team_name'])
                    updates[key] = {
                        'age': athlete.get('age', ''),
                        'weight': weight,
                        'height': height,
                        'yrs_played': experience,
                        'plyr_draft_tm': draft_team,
                        'plyr_draft_rd': draft_rnd,
                        'plyr_draft_pick': draft_sel,
                        'plyr_draft_yr': draft_yr,
                        'plyr_birthdate': birthdate,
                        'plyr_college': college
                    }
                else:
                    key = (cleaned_name, row['team_name'])
                    updates[key] = {
                        'plyr_draft_tm': 'Undrafted Free Agent'
                    }
            else:
                key = (cleaned_name, row['team_name'])
                updates[key] = {
                    'plyr_draft_tm': 'Undrafted Free Agent'
                }

    if updates:
        print(f"\nUpdating {len(updates)} players in output file...")
        update_raw_csv(output_file, updates)
        print("Update complete!")
    else:
        print("No matches found to update")

    # Save cache if it was updated
    if cache_updated:
        save_match_cache(match_cache)
        print(f"Match cache saved to {CACHE_FILE}")

if __name__ == "__main__":
    main()

