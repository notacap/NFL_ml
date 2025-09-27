import os
import glob
import csv
import re
from datetime import datetime
from collections import defaultdict
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clean_utils import YEAR, WEEK

# Define the base directory
BASE_DIR = rf"C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\{YEAR}"

# Define paths
GAMES_DIR = rf"C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\{YEAR}\games"
TEAM_ROSTERS_DIR = rf"C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\{YEAR}\roster_details\week_{WEEK}\cleaned"
OUTPUT_DIR = rf"C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\{YEAR}\plyr\plyr_raw\{WEEK}"

# Age cache file path
AGE_CACHE_FILE = rf"C:\Users\nocap\Desktop\code\NFL_ml\clean_data\players\player_ages_cache.json"

# Position standardization mapping
POSITION_MAPPING = {
    'QB': 'QB', 
    'RB': 'RB', 
    'WR': 'WR', 
    'TE': 'TE',
    'G': 'OL', 'C': 'OL', 'OG': 'OL', 'IOL': 'OL', 'OL': 'OL', 'LG': 'OL', 'RG': 'OL', 'LG/RG': 'OL',
    'T': 'OL', 'OT': 'OL', 'RT': 'OL', 'LT': 'OL', 'RT/LT': 'OL',
    'DE': 'DL', 'DT': 'DL', 'NT': 'DL', 'LDE': 'DL', 'RDE': 'DL', 'LDE/RDE': 'DL', 'LDT': 'DL', 'RDT': 'DL', 'LDT/RDT': 'DL',
    'LB': 'LB', 'ILB': 'LB', 'MLB': 'LB', 'RLB/MLB' : 'LB', 'OLB': 'LB', 'LOLB': 'LB', 'ROLB': 'LB', 'LILB': 'LB', 'RILB': 'LB', 'LILB/RILB': 'LB', 'RILB/LILB': 'LB', 'LLB': 'LB', 'RLB': 'LB',
    'CB': 'DB', 'DB': 'DB', 'LCB' : 'DB', 'RCB' : 'DB', 'LCB/RCB': 'DB', 'FS': 'DB', 'SS': 'DB','S': 'DB', 'SS/FS': 'DB',
    'K': 'K', 'PK': 'K',
    'P': 'P', 
    'LS': 'LS'
}

# Team name standardization mapping
TEAM_MAPPING = {
    "Cardinals": "Arizona Cardinals",
    "Falcons": "Atlanta Falcons",
    "Ravens": "Baltimore Ravens",
    "Bills": "Buffalo Bills",
    "Panthers": "Carolina Panthers",
    "Bears": "Chicago Bears",
    "Bengals": "Cincinnati Bengals",
    "Browns": "Cleveland Browns",
    "Cowboys": "Dallas Cowboys",
    "Broncos": "Denver Broncos",
    "Lions": "Detroit Lions",
    "Packers": "Green Bay Packers",
    "Texans": "Houston Texans",
    "Colts": "Indianapolis Colts",
    "Jaguars": "Jacksonville Jaguars",
    "Chiefs": "Kansas City Chiefs",
    "Dolphins": "Miami Dolphins",
    "Vikings": "Minnesota Vikings",
    "Patriots": "New England Patriots",
    "Saints": "New Orleans Saints",
    "Giants": "New York Giants",
    "Jets": "New York Jets",
    "Raiders": "Las Vegas Raiders",
    "Eagles": "Philadelphia Eagles",
    "Steelers": "Pittsburgh Steelers",
    "Chargers": "Los Angeles Chargers",
    "49ers": "San Francisco 49ers",
    "San Francisco 49Ers": "San Francisco 49ers",
    "Seahawks": "Seattle Seahawks",
    "Rams": "Los Angeles Rams",
    "Buccaneers": "Tampa Bay Buccaneers",
    "Titans": "Tennessee Titans",
    "Commanders": "Washington Commanders"
}

def load_age_cache():
    """Load the age cache from file if it exists"""
    if os.path.exists(AGE_CACHE_FILE):
        try:
            with open(AGE_CACHE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print("Warning: Could not load age cache file, starting with empty cache")
            return {}
    return {}

def save_age_cache(cache):
    """Save the age cache to file"""
    try:
        with open(AGE_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except IOError as e:
        print(f"Warning: Could not save age cache: {e}")

def get_player_age(player_name, team_name, cache):
    """Get player age from cache or prompt user"""
    # Create a unique key for the player
    cache_key = f"{player_name}|{team_name}"

    # Check if we already have this player's age in cache
    if cache_key in cache:
        return cache[cache_key]

    # Prompt user for age
    print(f"\n=== Missing Age Information ===")
    print(f"Player: {player_name}")
    print(f"Team: {team_name}")

    while True:
        age_input = input("Please enter the player's age (or 'skip' to leave blank): ").strip()

        if age_input.lower() == 'skip':
            # Store empty string in cache to avoid re-prompting
            cache[cache_key] = ''
            save_age_cache(cache)
            return ''

        try:
            age = int(age_input)
            if 18 <= age <= 50:  # Reasonable age range for NFL players
                cache[cache_key] = str(age)
                save_age_cache(cache)
                return str(age)
            else:
                print("Age must be between 18 and 50. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'skip'.")

def standardize_position(position):
    return POSITION_MAPPING.get(position, position)

def standardize_team_name(team_name):
    return TEAM_MAPPING.get(team_name, team_name)

def clean_to_int(value):
    try:
        # Convert to float first to handle both integer and float strings
        float_value = float(value)
        # Then convert to int to remove any decimal places
        return str(int(float_value))
    except ValueError:
        # If conversion fails, return an empty string
        return ''

def clean_player_name(player_name):
    """
    Cleans player names by:
    1. Handling character encoding issues
    2. Removing text in parentheses
    3. Handling first name variations
    4. Removing generational/patronymic suffixes
    5. Removing periods
    6. Converting to lowercase for matching
    7. Handling hyphenated names
    8. Stripping whitespace
    """
    # Handle character encoding issues
    player_name = player_name.replace('PiÃ±eiro', 'Pineiro')

    # Remove text within parentheses
    paren_index = player_name.find('(')
    if paren_index != -1:
        player_name = player_name[:paren_index]
    
    # Convert to lowercase and remove periods
    player_name = player_name.lower().replace('.', '')
    
    # First name variations mapping
    name_variations = {
        'patrick': ['pat'],
        'zachary': ['zach', 'zak'],
        'christopher': ['chris'],
        'michael': ['mike'],
        'jonathan': ['john', 'jon'],
        'stephen': ['steve'],
        'steven': ['steve'],
        'william': ['willy', 'bill'],
        'daniel': ['dan']
    }
    
    # Split into first name and rest of name
    parts = player_name.split(' ', 1)
    if len(parts) > 1:
        first_name, rest = parts
        variations = []
        
        # Check if first name has variations
        if first_name in name_variations:
            # Create variations with each alternative first name
            for variant in name_variations[first_name]:
                variations.append(f"{variant} {rest}")
        
        # Add original name to variations if we found alternatives
        if variations:
            variations.append(player_name)
            
            # Remove suffixes from each variation
            cleaned_variations = []
            for variant in variations:
                # Remove suffixes
                suffixes = [' ii', ' iii', ' iv', ' v', ' jr', ' sr', ' jr.', ' sr.']
                cleaned_variant = variant
                for suffix in suffixes:
                    if cleaned_variant.endswith(suffix):
                        cleaned_variant = cleaned_variant[:-len(suffix)]
                cleaned_variations.append(cleaned_variant.strip())
            
            # Handle hyphenated names for each variation
            final_variations = []
            for variant in cleaned_variations:
                if '-' in variant:
                    names = variant.split()
                    for i, name in enumerate(names):
                        if '-' in name:
                            parts = name.split('-')
                            final_variations.extend([
                                ' '.join(names[:i] + [parts[0]] + names[i+1:]),  # First part
                                ' '.join(names[:i] + [parts[1]] + names[i+1:]),  # Second part
                                ' '.join(names[:i] + [f"{parts[0]}-{parts[1]}"] + names[i+1:])  # Original hyphenated
                            ])
                else:
                    final_variations.append(variant)
            
            return list(set(final_variations))  # Remove any duplicates
        
    # If no first name variations, proceed with original logic
    # Remove suffixes
    suffixes = [' ii', ' iii', ' iv', ' v', ' jr', ' sr', ' jr.', ' sr.']
    for suffix in suffixes:
        if player_name.endswith(suffix):
            player_name = player_name[:-len(suffix)]
    
    # Handle hyphenated names
    if '-' in player_name:
        names = player_name.split()
        for i, name in enumerate(names):
            if '-' in name:
                parts = name.split('-')
                variations = [
                    ' '.join(names[:i] + [parts[0]] + names[i+1:]),  # First part
                    ' '.join(names[:i] + [parts[1]] + names[i+1:]),  # Second part
                    ' '.join(names[:i] + [f"{parts[0]}-{parts[1]}"] + names[i+1:])  # Original hyphenated
                ]
                return [var.strip() for var in variations]
    
    return [player_name.strip()]

def process_team_roster_files():
    # First, group files by team to detect duplicates
    team_files = defaultdict(list)

    for file in glob.glob(os.path.join(TEAM_ROSTERS_DIR, "*.csv")):
        # Read first data row to get team name
        with open(file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('Player') and row['Player'] != 'Team Total':
                    team_name = standardize_team_name(row['Team'])
                    team_files[team_name].append(file)
                    break

    # Report any duplicate team files found
    for team, files in team_files.items():
        if len(files) > 1:
            print(f"Found {len(files)} roster files for {team}:")
            for file in files:
                print(f"  - {os.path.basename(file)}")

    # Process files, merging duplicates for the same team
    players_data = {}

    for team_name, files in team_files.items():
        team_players = {}  # Temporary storage for this team's players

        for file in files:
            with open(file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Skip rows that don't have player data (like team totals)
                    if not row.get('Player') or row['Player'] == 'Team Total':
                        continue

                    original_name = row['Player']
                    cleaned_names = clean_player_name(row['Player'])
                    position = row['Pos']
                    standardized_pos = standardize_position(position)


                    # The cleaned files have separate draft columns (original format)
                    draft_team = row.get('Draft Team', '')
                    draft_round = row.get('Draft Round', '')
                    draft_pick = row.get('Draft Pick', '')
                    draft_year = row.get('Draft Year', '')

                    # Use original name and position as key for merging within team
                    merge_key = (original_name.lower(), standardized_pos)

                    # If player already exists in this team's data, check for conflicts
                    if merge_key in team_players:
                        existing = team_players[merge_key]
                        # Compare non-empty values and report conflicts
                        for field in ['age', 'weight', 'height', 'yrs_played', 'plyr_college',
                                    'plyr_birthdate', 'plyr_avg_value', 'plyr_draft_tm',
                                    'plyr_draft_rd', 'plyr_draft_pick', 'plyr_draft_yr',
                                    'gm_played', 'gm_started']:
                            new_val = row.get(field.replace('plyr_', '').replace('_', ' ').title().replace(' ', ''), '')
                            if field == 'plyr_college':
                                new_val = row.get('College/Univ', '')
                            elif field == 'plyr_birthdate':
                                new_val = row.get('BirthDate', '')
                            elif field == 'plyr_avg_value':
                                new_val = row.get('AV', '')
                            elif field == 'plyr_draft_tm':
                                new_val = draft_team
                            elif field == 'plyr_draft_rd':
                                new_val = draft_round
                            elif field == 'plyr_draft_pick':
                                new_val = draft_pick
                            elif field == 'plyr_draft_yr':
                                new_val = draft_year
                            elif field == 'gm_played':
                                new_val = row.get('G', '')
                            elif field == 'gm_started':
                                new_val = row.get('GS', '')
                            elif field == 'age':
                                new_val = row['Age']
                            elif field == 'weight':
                                new_val = row['Wt']
                            elif field == 'height':
                                new_val = row['Ht']
                            elif field == 'yrs_played':
                                new_val = row['Yrs']

                            # Use new value if existing is empty, otherwise keep existing
                            if not existing[field] and new_val:
                                existing[field] = new_val
                            elif existing[field] and new_val and existing[field] != new_val:
                                print(f"  Warning: Conflicting {field} for {original_name} ({team_name}): '{existing[field]}' vs '{new_val}'")
                    else:
                        # New player for this team
                        team_players[merge_key] = {
                            'team_name': team_name,
                            'plyr_name': original_name,
                            'pos': standardized_pos,
                            'age': row['Age'],
                            'weight': row['Wt'],
                            'height': row['Ht'],
                            'yrs_played': row['Yrs'],
                            'plyr_college': row.get('College/Univ', ''),
                            'plyr_birthdate': row.get('BirthDate', ''),
                            'plyr_avg_value': row.get('AV', ''),
                            'plyr_draft_tm': draft_team,
                            'plyr_draft_rd': draft_round,
                            'plyr_draft_pick': draft_pick,
                            'plyr_draft_yr': draft_year,
                            'gm_played': row.get('G', ''),
                            'gm_started': row.get('GS', ''),
                            'cleaned_names': cleaned_names
                        }

        # Add merged team players to final players_data with all name variations
        for (name_key, pos), player_info in team_players.items():
            # Create entries for all name variations
            for cleaned_name in player_info.get('cleaned_names', [player_info['plyr_name']]):
                player_key = (team_name, cleaned_name, player_info['pos'])
                # Remove the temporary cleaned_names field before storing
                final_player_info = {k: v for k, v in player_info.items() if k != 'cleaned_names'}
                players_data[player_key] = final_player_info


    # Print merge summary if duplicates were found
    duplicate_teams = [team for team, files in team_files.items() if len(files) > 1]
    if duplicate_teams:
        print(f"\nMerged {sum(len(f) for f in team_files.values())} total files into {len(team_files)} unique team rosters")
        print(f"Teams with duplicate files merged: {', '.join(duplicate_teams)}")

    return players_data

def process_snap_count_files():
    players_data = defaultdict(lambda: {'weeks': set(), 'team_name': '', 'plyr_name': '', 'pos': ''})
    
    # Find all week directories and process them in order
    week_dirs = []
    for week_num in range(1, 19):  # weeks 1-18
        week_dir = os.path.join(GAMES_DIR, f"week_{week_num}.0")
        if os.path.exists(week_dir):
            week_dirs.append(week_dir)
    
    print(f"Found {len(week_dirs)} week directories to process")
    
    # Process snap count files in each week directory
    for week_dir in week_dirs:
        week_name = os.path.basename(week_dir)
        print(f"Processing {week_name}...")
        
        clean_dir = os.path.join(week_dir, "clean")
        if not os.path.exists(clean_dir):
            print(f"Warning: Clean directory not found for {week_name}")
            continue
            
        for file in glob.glob(os.path.join(clean_dir, "*_snap_counts_*.csv")):
            with open(file, 'r', newline='') as f:
                # Read the single header row to get column positions
                header = f.readline().strip().split(',')
                
                # Find the positions of the columns we need
                team_col_idx = header.index('team') if 'team' in header else None
                week_col_idx = header.index('week') if 'week' in header else None
                player_col_idx = header.index('Player') if 'Player' in header else None
                pos_col_idx = header.index('Pos') if 'Pos' in header else None
                
                if None in [team_col_idx, week_col_idx, player_col_idx, pos_col_idx]:
                    print(f"Warning: Missing required columns in {file}")
                    continue
                
                # Read the data rows
                reader = csv.reader(f)
                for row in reader:
                    if len(row) <= max(team_col_idx, week_col_idx, player_col_idx, pos_col_idx):
                        continue  # Skip incomplete rows
                    
                    # Get data from the correct column positions
                    original_name = row[player_col_idx]
                    position = row[pos_col_idx]
                    team_name = row[team_col_idx]
                    week = int(float(row[week_col_idx]))

                    cleaned_names = clean_player_name(original_name)

                    
                    # Use the first cleaned name variation for the data storage
                    main_key = (standardize_team_name(team_name), cleaned_names[0], standardize_position(position))
                    
                    players_data[main_key]['weeks'].add(week)
                    players_data[main_key]['team_name'] = standardize_team_name(team_name)
                    players_data[main_key]['plyr_name'] = original_name
                    players_data[main_key]['pos'] = standardize_position(position)
                    players_data[main_key]['cleaned_names'] = cleaned_names  # Store all variations for matching
    
    result = []
    for player_key, player_info in players_data.items():
        result.append({
            'team_name': player_info['team_name'],
            'plyr_name': player_info['plyr_name'],
            'pos': player_info['pos'],
            'weeks': ','.join(map(str, sorted(player_info['weeks']))),
            'cleaned_names': player_info['cleaned_names']
        })
    
    return result

def merge_player_data(snap_count_data, roster_data, age_cache):
    merged_data = []
    unmatched_players = []

    print(f"\nTotal players in snap count data: {len(snap_count_data)}")

    # First pass - try matching with name, team, and position
    for player in snap_count_data:
        matched = False

        for cleaned_name in player['cleaned_names']:
            player_key = (player['team_name'], cleaned_name, player['pos'])

            if player_key in roster_data:
                # Found exact match with name, team, and position
                roster_info = roster_data[player_key]
                merged_player = create_merged_player(player, roster_info, age_cache)
                merged_data.append(merged_player)
                matched = True
                print(f"Exact match found: {player['plyr_name']} - {player['team_name']} - {player['pos']}")
                break

        if not matched:
            unmatched_players.append(player)
            print(f"No exact match found: {player['plyr_name']} - {player['team_name']} - {player['pos']}")

    print(f"\nPlayers matched in first pass: {len(merged_data)}")
    print(f"Players remaining for second pass: {len(unmatched_players)}")

    # Second pass - try matching with just name and team
    second_pass_matches = []
    still_unmatched = []
    
    for unmatched_player in unmatched_players:
        potential_matches = []
        
        # Find all potential matches based on name and team
        seen_matches = set()  # To prevent duplicate matches
        for cleaned_name in unmatched_player['cleaned_names']:
            for key, roster_info in roster_data.items():
                roster_team, roster_name, _ = key
                if roster_team == unmatched_player['team_name'] and roster_name == cleaned_name:
                    # Use player name as unique identifier to prevent duplicates
                    player_identifier = f"{roster_info['plyr_name']}_{roster_info['team_name']}_{roster_info['pos']}"
                    if player_identifier not in seen_matches:
                        potential_matches.append(roster_info)
                        seen_matches.add(player_identifier)

        if len(potential_matches) == 1:
            # Single match found
            merged_player = create_merged_player(unmatched_player, potential_matches[0], age_cache)
            second_pass_matches.append(merged_player)
            print(f"Single team/name match found: {unmatched_player['plyr_name']} - {unmatched_player['team_name']}")
        
        elif len(potential_matches) > 1:
            # Multiple UNIQUE matches found - prompt user for selection
            print(f"\nMultiple potential matches ({len(potential_matches)}) found for:")
            print(f"Snap count player: {unmatched_player['plyr_name']} "
                  f"(Team: {unmatched_player['team_name']}, Position: {unmatched_player['pos']})")
            print("\nPotential matches from team roster:")
            
            for idx, match in enumerate(potential_matches, 1):
                print(f"\n{idx}. {match['plyr_name']}")
                print(f"   Team: {match['team_name']}")
                print(f"   Position: {match['pos']}")
                print(f"   Age: {match['age']}")
                print(f"   Weight: {match['weight']}")
                print(f"   Height: {match['height']}")
                print(f"   College: {match['plyr_college']}")
                print(f"   Draft Team: {match['plyr_draft_tm']}")
                print(f"   Draft Round: {match['plyr_draft_rd']}")
            
            print("\n0. Skip this player")
            
            while True:
                try:
                    choice = input("\nEnter the number of the correct match (or 0 to skip): ")
                    choice = int(choice)
                    if 0 <= choice <= len(potential_matches):
                        break
                    print("Invalid choice. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            if choice > 0:
                merged_player = create_merged_player(unmatched_player, potential_matches[choice - 1], age_cache)
                second_pass_matches.append(merged_player)
            else:
                still_unmatched.append(unmatched_player)
        else:
            still_unmatched.append(unmatched_player)
            print(f"No team/name match found: {unmatched_player['plyr_name']} - {unmatched_player['team_name']}")

    merged_data.extend(second_pass_matches)
    
    print(f"\nFinal matching results:")
    print(f"Total players in snap count data: {len(snap_count_data)}")
    print(f"Players matched in first pass: {len(merged_data) - len(second_pass_matches)}")
    print(f"Players matched in second pass: {len(second_pass_matches)}")
    print(f"Unmatched players remaining: {len(still_unmatched)}")

    return merged_data, still_unmatched  # Return both matched and unmatched players

def create_merged_player(snap_count_player, roster_info, age_cache):
    """Helper function to create a merged player record"""
    # Get the age, prompting if necessary
    age = clean_to_int(roster_info['age'])
    if not age:  # Age is missing or empty
        age = get_player_age(roster_info['plyr_name'], snap_count_player['team_name'], age_cache)

    result = {
        'plyr_name': roster_info['plyr_name'],  # Always use roster name for matched players
        'team_name': snap_count_player['team_name'],
        'pos': roster_info['pos'],  # Always use roster position for matched players
        'weeks': snap_count_player['weeks'],
        'age': age,  # Use the age we got (either from roster or from prompt)
        'weight': clean_to_int(roster_info['weight']),
        'height': roster_info['height'],
        'yrs_played': clean_to_int(roster_info['yrs_played']),
        'plyr_college': roster_info['plyr_college'],
        'plyr_birthdate': roster_info['plyr_birthdate'],
        'plyr_avg_value': clean_to_int(roster_info['plyr_avg_value']),
        'plyr_draft_tm': roster_info['plyr_draft_tm'],
        'plyr_draft_rd': clean_to_int(roster_info['plyr_draft_rd']),
        'plyr_draft_pick': clean_to_int(roster_info['plyr_draft_pick']),
        'plyr_draft_yr': clean_to_int(roster_info['plyr_draft_yr']),
        'gm_played': clean_to_int(roster_info['gm_played']),
        'gm_started': clean_to_int(roster_info['gm_started'])
    }

    return result

def convert_weeks_to_set(weeks_str):
    """Convert comma-separated string of weeks to a set of integers"""
    return set(map(int, weeks_str.split(','))) if weeks_str else set()

def find_multi_team_matches(players_data):
    # Separate matched and unmatched players
    matched_players = [p for p in players_data if p.get('age')]
    unmatched_players = [p for p in players_data if not p.get('age')]
    
    print(f"\nLooking for multi-team matches:")
    print(f"Players with attributes: {len(matched_players)}")
    print(f"Players without attributes: {len(unmatched_players)}")
    
    final_players = matched_players.copy()  # Start with all matched players
    remaining_unmatched = []  # To track players that are still unmatched after multi-team matching
    
    for unmatched in unmatched_players:
        match_found = False
        potential_matches = []
        unmatched_weeks = convert_weeks_to_set(unmatched['weeks'])
        
        for matched in matched_players:
            matched_weeks = convert_weeks_to_set(matched['weeks'])
            
            # Check if this is a potential match
            if (unmatched['plyr_name'].lower() == matched['plyr_name'].lower() and
                unmatched['team_name'] != matched['team_name'] and
                not (unmatched_weeks & matched_weeks)):  # No overlapping weeks
                
                potential_matches.append(matched)
        
        if len(potential_matches) == 1:
            # Single match found - copy attributes
            updated_player = unmatched.copy()
            matched_source = potential_matches[0]
            
            # Copy all attributes except name, team, position, and weeks
            for field in ['age', 'weight', 'height', 'yrs_played', 'plyr_college', 
                         'plyr_birthdate', 'plyr_avg_value', 'plyr_draft_tm', 'plyr_draft_rd', 
                         'plyr_draft_pick', 'plyr_draft_yr', 'gm_played', 'gm_started']:
                updated_player[field] = matched_source[field]
            
            final_players.append(updated_player)
            match_found = True
            print(f"\nMatch found: {unmatched['plyr_name']}")
            print(f"Team 1: {matched_source['team_name']} - Weeks: {matched_source['weeks']}")
            print(f"Team 2: {unmatched['team_name']} - Weeks: {unmatched['weeks']}")
        
        elif len(potential_matches) > 1:
            # Multiple matches found - prompt user
            print(f"\nMultiple potential multi-team matches found for:")
            print(f"Player: {unmatched['plyr_name']}")
            print(f"Current team: {unmatched['team_name']}")
            print(f"Current position: {unmatched['pos']}")
            print(f"Current weeks: {unmatched['weeks']}")
            print("\nPotential matches:")
            
            for idx, match in enumerate(potential_matches, 1):
                print(f"\n{idx}. {match['plyr_name']}")
                print(f"   Team: {match['team_name']}")
                print(f"   Position: {match['pos']}")
                print(f"   Weeks: {match['weeks']}")
                print(f"   Age: {match['age']}")
                print(f"   Weight: {match['weight']}")
                print(f"   Height: {match['height']}")
                print(f"   Years played: {match['yrs_played']}")
            
            print("\n0. Skip this player")
            
            while True:
                try:
                    choice = input("\nEnter the number of the correct match (or 0 to skip): ")
                    choice = int(choice)
                    if 0 <= choice <= len(potential_matches):
                        break
                    print("Invalid choice. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            if choice > 0:
                # Copy attributes from selected match
                updated_player = unmatched.copy()
                matched_source = potential_matches[choice - 1]
                
                for field in ['age', 'weight', 'height', 'yrs_played', 'plyr_college', 
                             'plyr_birthdate', 'plyr_avg_value', 'plyr_draft_tm', 'plyr_draft_rd', 
                             'plyr_draft_pick', 'plyr_draft_yr', 'gm_played', 'gm_started']:
                    updated_player[field] = matched_source[field]
                
                final_players.append(updated_player)
                match_found = True
                
        if not match_found:
            remaining_unmatched.append(unmatched)
    
    return final_players, remaining_unmatched

def create_output_file(players_data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file_name = f"plyr_raw_{timestamp}.csv"
    output_file_path = os.path.join(OUTPUT_DIR, output_file_name)

    fieldnames = [
        'plyr_name', 'team_name', 'pos', 'weeks', 'age', 
        'weight', 'height', 'yrs_played', 'plyr_college', 
        'plyr_birthdate', 'plyr_avg_value', 'plyr_draft_tm', 'plyr_draft_rd', 
        'plyr_draft_pick', 'plyr_draft_yr', 'gm_played', 'gm_started'
    ]

    with open(output_file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for player in players_data:
            # Check if player has attributes by checking for any roster-derived fields
            has_attributes = any([
                player.get('weight'),
                player.get('height'),
                player.get('yrs_played'),
                player.get('plyr_college'),
                player.get('plyr_birthdate'),
                player.get('plyr_avg_value'),
                player.get('plyr_draft_tm'),
                player.get('plyr_draft_rd'),
                player.get('plyr_draft_pick'),
                player.get('plyr_draft_yr'),
                player.get('gm_played'),
                player.get('gm_started'),
                player.get('age')  # Include age but don't rely on it alone
            ])

            if has_attributes:  # Player has attributes
                # Create a new dict with only the fields we want
                row = {field: player.get(field, '') for field in fieldnames}
                writer.writerow(row)
            else:  # Player doesn't have attributes
                row = {
                    'plyr_name': player['plyr_name'],
                    'team_name': player['team_name'],
                    'pos': player['pos'],
                    'weeks': player['weeks'],
                    'age': '',
                    'weight': '',
                    'height': '',
                    'yrs_played': '',
                    'plyr_college': '',
                    'plyr_birthdate': '',
                    'plyr_avg_value': '',
                    'plyr_draft_tm': '',
                    'plyr_draft_rd': '',
                    'plyr_draft_pick': '',
                    'plyr_draft_yr': '',
                    'gm_played': '',
                    'gm_started': ''
                }
                writer.writerow(row)

    # Count players with and without attributes using the same logic as above
    matched_count = sum(1 for p in players_data if any([
        p.get('weight'), p.get('height'), p.get('yrs_played'),
        p.get('plyr_college'), p.get('plyr_birthdate'), p.get('plyr_avg_value'),
        p.get('plyr_draft_tm'), p.get('plyr_draft_rd'), p.get('plyr_draft_pick'),
        p.get('plyr_draft_yr'), p.get('gm_played'), p.get('gm_started'), p.get('age')
    ]))
    unmatched_count = len(players_data) - matched_count
    
    print(f"\nOutput file created: {output_file_path}")
    print(f"Total players: {len(players_data)}")
    print(f"Players with attributes: {matched_count}")
    print(f"Players without attributes: {unmatched_count}")


def main():
    # Load age cache
    age_cache = load_age_cache()
    print(f"Loaded age cache with {len(age_cache)} entries")

    snap_count_data = process_snap_count_files()
    roster_data = process_team_roster_files()

    merged_data, unmatched_players = merge_player_data(snap_count_data, roster_data, age_cache)
    
    # Combine matched and unmatched players for multi-team matching
    all_players = merged_data + unmatched_players
    
    # Find and process multi-team matches
    final_data, final_unmatched = find_multi_team_matches(all_players)
    
    # Add remaining unmatched players to final_data before creating output
    final_data.extend(final_unmatched)
    
    # Create final output file with ALL players
    create_output_file(final_data)

if __name__ == "__main__":
    main()
