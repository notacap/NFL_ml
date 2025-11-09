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
CACHE_FILE = os.path.join(os.path.dirname(__file__), "cache", "player_match_cache.json")

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
    'LV': 'Las Vegas Raiders',
    'OAK': 'Oakland Raiders',
    'LAC': 'Los Angeles Chargers',
    'LAR': 'Los Angeles Rams',
    'MIA': 'Miami Dolphins',
    'MIN': 'Minnesota Vikings',
    'NE': 'New England Patriots',
    'NWE': 'New England Patriots',
    'NOR': 'New Orleans Saints',
    'NO': 'New Orleans Saints',
    'NYG': 'New York Giants',
    'NYJ': 'New York Jets',
    'PHI': 'Philadelphia Eagles',
    'PIT': 'Pittsburgh Steelers',
    'SFO': 'San Francisco 49ers',
    'SF': 'San Francisco 49ers',
    'SEA': 'Seattle Seahawks',
    'TAM': 'Tampa Bay Buccaneers',
    'TEN': 'Tennessee Titans',
    'WAS': 'Washington Commanders',
    'WSH': 'Washington Commanders',
}

POSITION_MAPPING = {
    'QB': 'QB',
    'RB': 'RB',
    'WR': 'WR',
    'TE': 'TE',
    'G': 'OL', 'C': 'OL', 'OG': 'OL', 'IOL': 'OL', 'OL': 'OL', 'LG': 'OL', 'RG': 'OL', 'LG/RG': 'OL',
    'T': 'OL', 'OT': 'OL', 'RT': 'OL', 'LT': 'OL', 'RT/LT': 'OL',
    'DE': 'DL', 'DT': 'DL', 'NT': 'DL', 'LDE': 'DL', 'RDE': 'DL', 'LDE/RDE': 'DL', 'LDT': 'DL', 'RDT': 'DL', 'LDT/RDT': 'DL',
    'LB': 'LB', 'ILB': 'LB', 'MLB': 'LB', 'RLB/MLB': 'LB', 'OLB': 'LB', 'LOLB': 'LB', 'ROLB': 'LB', 'LILB': 'LB', 'RILB': 'LB', 'LILB/RILB': 'LB', 'RILB/LILB': 'LB', 'LLB': 'LB', 'RLB': 'LB',
    'CB': 'DB', 'DB': 'DB', 'LCB': 'DB', 'RCB': 'DB', 'LCB/RCB': 'DB', 'FS': 'DB', 'SS': 'DB', 'S': 'DB', 'SS/FS': 'DB',
    'K': 'K', 'PK': 'K',
    'P': 'P',
    'LS': 'LS'
}

COLLEGE_ALIASES = {
    'MISSISSIPPI': 'OLE MISS',
    'OLE MISS': 'MISSISSIPPI',
    'MIAMI': 'MIAMI FL',
    'MIAMI FL': 'MIAMI',
    'MIAMI (FL)': 'MIAMI',
}

def normalize_college_name(college):
    """Normalize college name for fuzzy matching"""
    if not college:
        return ''

    # Convert to uppercase
    college = college.upper().strip()

    # Remove periods
    college = college.replace('.', '')

    # Replace "COL" with "COLLEGE"
    college = re.sub(r'\bCOL\b', 'COLLEGE', college)

    # Remove hyphens
    college = college.replace('-', ' ')

    # Remove parenthetical info like (FL)
    college = re.sub(r'\([^)]*\)', '', college)

    # Clean up extra whitespace
    college = ' '.join(college.split())

    return college

def colleges_match(csv_college, api_colleges):
    """Check if CSV college matches any API college field

    Args:
        csv_college: String from CSV (may contain comma-separated values)
        api_colleges: List of college strings from API to check against

    Returns:
        bool: True if any match found
    """
    if not csv_college:
        return False

    # Split by comma in case there are multiple colleges
    csv_colleges = [c.strip() for c in csv_college.split(',')]

    # Normalize all CSV colleges
    normalized_csv = [normalize_college_name(c) for c in csv_colleges if c]

    # Normalize all API colleges
    normalized_api = [normalize_college_name(c) for c in api_colleges if c]

    # Check for matches
    for csv_col in normalized_csv:
        if not csv_col:
            continue

        # Check direct match
        if csv_col in normalized_api:
            return True

        # Check aliases
        if csv_col in COLLEGE_ALIASES:
            alias = COLLEGE_ALIASES[csv_col]
            if alias in normalized_api:
                return True

        # Check if any API college contains the CSV college (or vice versa)
        for api_col in normalized_api:
            if not api_col:
                continue
            if csv_col in api_col or api_col in csv_col:
                return True

    return False

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
    """Load cached player matches from file for the current year"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                all_cache = json.load(f)
                year_key = str(YEAR)
                return all_cache.get(year_key, {}), all_cache
        except:
            return {}, {}
    return {}, {}

def save_match_cache(year_cache, all_cache):
    """Save player match cache to file"""
    try:
        year_key = str(YEAR)
        all_cache[year_key] = year_cache
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_cache, f, indent=2)
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

def compare_api_data_to_csv(row, player_id, debug=False):
    """Compare CSV data to API data for a given player_id

    Returns:
        tuple: (matches: bool, match_count: int) - whether it matches and how many attributes matched
    """
    player_data = get_player_data(player_id)
    if not player_data or 'athlete' not in player_data:
        if debug:
            print(f"   ID {player_id}: Failed to fetch API data")
        return False, 0

    athlete = player_data['athlete']

    # Get API values
    api_team_name = ''
    if 'playerSwitcher' in player_data and player_data['playerSwitcher']:
        if 'team' in player_data['playerSwitcher'] and player_data['playerSwitcher']['team']:
            api_team_name = player_data['playerSwitcher']['team'].get('displayName', '')

    api_dob = athlete.get('displayDOB', '')
    api_age = athlete.get('age', '')
    api_weight_str = athlete.get('displayWeight', '')  # Format: "322 lbs"
    api_height_str = athlete.get('displayHeight', '')  # Format: "5' 10""
    api_experience_str = athlete.get('displayExperience', '')  # Format: "7th Season"
    api_draft_str = athlete.get('displayDraft', '')
    api_position = athlete.get('position', {}).get('abbreviation', '') if athlete.get('position') else ''

    # Get college information from API
    api_college_fields = []
    if athlete.get('collegeTeam'):
        college_team = athlete['collegeTeam']
        api_college_fields.extend([
            college_team.get('location', ''),
            college_team.get('nickname', ''),
            college_team.get('displayName', ''),
            college_team.get('shortDisplayName', '')
        ])

    # Convert CSV birthdate to match API format
    csv_birthdate = convert_birthdate(row.get('plyr_birthdate', ''))

    # Get CSV values
    csv_team_name = row.get('team_name', '')
    csv_age = row.get('age', '')
    csv_weight = row.get('weight', '')
    csv_height = row.get('height', '')
    csv_yrs_played = row.get('yrs_played', '')
    csv_position = row.get('pos', '')
    csv_draft_yr = row.get('plyr_draft_yr', '')
    csv_draft_rnd = row.get('plyr_draft_rd', '')
    csv_draft_sel = row.get('plyr_draft_pick', '')
    csv_draft_tm = row.get('plyr_draft_tm', '')
    csv_college = row.get('plyr_college', '')

    # Compare birthdate (gating check - if both exist and don't match, fail)
    if csv_birthdate and api_dob:
        dob_match = api_dob == csv_birthdate
        if debug:
            print(f"   ID {player_id}: DOB: {api_dob} vs {csv_birthdate} = {dob_match}")
        if not dob_match:
            return False, 0
    elif debug:
        print(f"   ID {player_id}: DOB: Skipped (CSV: {csv_birthdate}, API: {api_dob})")

    # Count all matching attributes
    match_count = 0

    # Team name
    if csv_team_name and api_team_name:
        team_match = api_team_name == csv_team_name
        if team_match:
            match_count += 1
        if debug:
            print(f"   ID {player_id}: Team: {api_team_name} vs {csv_team_name} = {team_match}")
    elif debug:
        print(f"   ID {player_id}: Team: Skipped")

    # Age
    if csv_age and api_age:
        age_match = str(api_age) == str(csv_age)
        if age_match:
            match_count += 1
        if debug:
            print(f"   ID {player_id}: Age: {api_age} vs {csv_age} = {age_match}")
    elif debug:
        print(f"   ID {player_id}: Age: Skipped")

    # Weight
    if csv_weight and api_weight_str:
        api_weight = convert_weight(api_weight_str)
        try:
            csv_weight_int = int(csv_weight)
            weight_match = api_weight == csv_weight_int
            if weight_match:
                match_count += 1
            if debug:
                print(f"   ID {player_id}: Weight: {api_weight_str} ({api_weight}) vs {csv_weight} = {weight_match}")
        except (ValueError, TypeError):
            if debug:
                print(f"   ID {player_id}: Weight: Conversion error")
    elif debug:
        print(f"   ID {player_id}: Weight: Skipped")

    # Height
    if csv_height and api_height_str:
        api_height = convert_height_to_inches(api_height_str)
        try:
            csv_height_int = int(csv_height)
            height_match = api_height == csv_height_int
            if height_match:
                match_count += 1
            if debug:
                print(f"   ID {player_id}: Height: {api_height_str} ({api_height}) vs {csv_height} = {height_match}")
        except (ValueError, TypeError):
            if debug:
                print(f"   ID {player_id}: Height: Conversion error")
    elif debug:
        print(f"   ID {player_id}: Height: Skipped")

    # Years played / Experience
    if csv_yrs_played and api_experience_str:
        api_yrs_played = convert_experience(api_experience_str)
        try:
            csv_yrs_played_int = int(csv_yrs_played)
            if api_yrs_played != '':
                yrs_match = api_yrs_played == csv_yrs_played_int
                if yrs_match:
                    match_count += 1
                if debug:
                    print(f"   ID {player_id}: Yrs Played: {api_experience_str} ({api_yrs_played}) vs {csv_yrs_played} = {yrs_match}")
            elif debug:
                print(f"   ID {player_id}: Yrs Played: API conversion failed")
        except (ValueError, TypeError):
            if debug:
                print(f"   ID {player_id}: Yrs Played: CSV conversion error")
    elif debug:
        print(f"   ID {player_id}: Yrs Played: Skipped")

    # Position
    if csv_position and api_position:
        csv_pos_mapped = POSITION_MAPPING.get(csv_position, csv_position)
        api_pos_mapped = POSITION_MAPPING.get(api_position, api_position)
        pos_match = csv_pos_mapped == api_pos_mapped
        if pos_match:
            match_count += 1
        if debug:
            print(f"   ID {player_id}: Position: {api_position} ({api_pos_mapped}) vs {csv_position} ({csv_pos_mapped}) = {pos_match}")
    elif debug:
        print(f"   ID {player_id}: Position: Skipped")

    # College
    if csv_college and api_college_fields:
        college_match = colleges_match(csv_college, api_college_fields)
        if college_match:
            match_count += 1
        if debug:
            print(f"   ID {player_id}: College: {api_college_fields} vs {csv_college} = {college_match}")
    elif debug:
        print(f"   ID {player_id}: College: Skipped (CSV: {bool(csv_college)}, API: {bool(api_college_fields)})")

    # Draft information
    if api_draft_str:
        api_draft_yr, api_draft_rnd, api_draft_sel, api_draft_team_abbr = parse_draft_info(api_draft_str)
        api_draft_tm = TEAM_ABBR_TO_FULL.get(api_draft_team_abbr, '') if api_draft_team_abbr else ''

        # Draft year
        if csv_draft_yr and api_draft_yr:
            try:
                csv_draft_yr_int = int(csv_draft_yr)
                draft_yr_match = api_draft_yr == csv_draft_yr_int
                if draft_yr_match:
                    match_count += 1
                if debug:
                    print(f"   ID {player_id}: Draft Year: {api_draft_yr} vs {csv_draft_yr} = {draft_yr_match}")
            except (ValueError, TypeError):
                if debug:
                    print(f"   ID {player_id}: Draft Year: Conversion error")
        elif debug:
            print(f"   ID {player_id}: Draft Year: Skipped")

        # Draft round
        if csv_draft_rnd and api_draft_rnd:
            try:
                csv_draft_rnd_int = int(csv_draft_rnd)
                draft_rnd_match = api_draft_rnd == csv_draft_rnd_int
                if draft_rnd_match:
                    match_count += 1
                if debug:
                    print(f"   ID {player_id}: Draft Round: {api_draft_rnd} vs {csv_draft_rnd} = {draft_rnd_match}")
            except (ValueError, TypeError):
                if debug:
                    print(f"   ID {player_id}: Draft Round: Conversion error")
        elif debug:
            print(f"   ID {player_id}: Draft Round: Skipped")

        # Draft pick
        if csv_draft_sel and api_draft_sel:
            try:
                csv_draft_sel_int = int(csv_draft_sel)
                draft_sel_match = api_draft_sel == csv_draft_sel_int
                if draft_sel_match:
                    match_count += 1
                if debug:
                    print(f"   ID {player_id}: Draft Pick: {api_draft_sel} vs {csv_draft_sel} = {draft_sel_match}")
            except (ValueError, TypeError):
                if debug:
                    print(f"   ID {player_id}: Draft Pick: Conversion error")
        elif debug:
            print(f"   ID {player_id}: Draft Pick: Skipped")

        # Draft team
        if csv_draft_tm and api_draft_tm:
            draft_tm_match = api_draft_tm == csv_draft_tm
            if draft_tm_match:
                match_count += 1
            if debug:
                print(f"   ID {player_id}: Draft Team: {api_draft_tm} vs {csv_draft_tm} = {draft_tm_match}")
        elif debug:
            print(f"   ID {player_id}: Draft Team: Skipped")
    else:
        # No draft info in API - check if CSV also has no draft info (both undrafted)
        csv_has_no_draft = not csv_draft_yr and not csv_draft_rnd and not csv_draft_sel and not csv_draft_tm
        if csv_has_no_draft:
            match_count += 1
            if debug:
                print(f"   ID {player_id}: Both undrafted - Match!")
        elif debug:
            print(f"   ID {player_id}: Draft info: Not available in API, but CSV has draft data")

    final_match = match_count > 0
    if debug:
        print(f"   ID {player_id}: Total match count = {match_count}, Final match result = {final_match}")

    return final_match, match_count

def process_player_matches(row, matches, active_players_data, match_cache):
    """Process matches and handle user input if needed"""
    if len(matches) == 0:
        return None, False

    if len(matches) == 1:
        player_id = list(matches.keys())[0]
        return player_id, False

    # Check cache first
    cleaned_name = clean_player_name(row['plyr_name'])
    cache_key = f"{cleaned_name}|{row.get('team_name', '')}|{row.get('pos', '')}|{row.get('plyr_birthdate', '')}|{row.get('plyr_draft_yr', '')}|{row.get('plyr_college', '')}"

    if cache_key in match_cache:
        cached_id = match_cache[cache_key]
        # Verify cached ID is still in the matches
        if cached_id in matches:
            print(f"Using cached match for {row['plyr_name']} -> ID: {cached_id}")
            return cached_id, False
        else:
            print(f"Warning: Cached match for {row['plyr_name']} no longer valid")

    # Multiple matches found - try to auto-match by comparing API data
    print(f"\nAttempting auto-match for {row['plyr_name']} (Team: {row['team_name']})...")

    match_scores = {}
    for player_id in matches.keys():
        is_match, count = compare_api_data_to_csv(row, player_id)
        if is_match:
            match_scores[player_id] = count

    if len(match_scores) == 0:
        # No matches found
        print(f"Could not auto-match based on available attributes.")
        print(f"Debug comparison results:")
        for player_id in matches.keys():
            compare_api_data_to_csv(row, player_id, debug=True)
        print(f"Manual selection required.")
    elif len(match_scores) == 1:
        # Single match found
        selected_id = list(match_scores.keys())[0]
        print(f"Auto-matched {row['plyr_name']} -> ID: {selected_id} (score: {match_scores[selected_id]})")
        match_cache[cache_key] = selected_id
        return selected_id, True
    else:
        # Multiple matches - select the one with the highest score
        max_score = max(match_scores.values())
        best_matches = [pid for pid, score in match_scores.items() if score == max_score]

        if len(best_matches) == 1:
            # One clear winner
            selected_id = best_matches[0]
            print(f"Auto-matched {row['plyr_name']} -> ID: {selected_id} (score: {max_score}, beat {len(match_scores)-1} other candidate(s))")
            match_cache[cache_key] = selected_id
            return selected_id, True
        else:
            # Tie - need manual selection
            print(f"Multiple potential auto-matches with same score ({len(best_matches)} IDs with score {max_score}).")
            print(f"Tied IDs: {best_matches}")
            print(f"Debug comparison results:")
            for player_id in matches.keys():
                compare_api_data_to_csv(row, player_id, debug=True)
            print(f"Manual selection required.")

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
    print("hello")
    active_players_file = get_latest_file(ACTIVE_PLAYERS_DIR, "all_players")
    raw_file = get_latest_file(PLYR_RAW_DIR, "plyr_raw")

    if not all([active_players_file, raw_file]):
        print("Could not find required files")
        return

    output_file = create_output_file(raw_file)
    print(f"Created output file: {output_file}")

    # Load match cache for current year
    match_cache, all_cache = load_match_cache()
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
        save_match_cache(match_cache, all_cache)
        print(f"Match cache saved to {CACHE_FILE} for year {YEAR}")

if __name__ == "__main__":
    main()

