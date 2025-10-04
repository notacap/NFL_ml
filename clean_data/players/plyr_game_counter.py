import os
import glob
import csv
import sys
from datetime import datetime
from collections import defaultdict

# Add parent directory to path to import clean_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clean_utils import YEAR, WEEK

# Define paths
PLYR_CLEAN_DIR = rf"C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\{YEAR}\plyr\plyr_clean\{WEEK}"
GAMES_DIR = rf"C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\{YEAR}\games"

# Position standardization mapping (copied from merge_weekly_snap_count_and_rosters.py)
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

def standardize_position(position):
    """Standardize position names"""
    return POSITION_MAPPING.get(position, position)

def standardize_team_name(team_name):
    """Standardize team names from short to full"""
    return TEAM_MAPPING.get(team_name, team_name)

def clean_player_name(player_name):
    """
    Cleans player names by:
    1. Detecting and handling IR status
    2. Handling character encoding issues
    3. Removing text in parentheses (except IR)
    4. Handling first name variations
    5. Removing generational/patronymic suffixes
    6. Removing periods
    7. Converting to lowercase for matching
    8. Handling hyphenated names
    9. Stripping whitespace

    Returns: (list of cleaned names, is_on_ir)
    """
    # Check for IR status before any other processing
    ir_variations = ['(IR)', '(PRA_)', '(NON)', '(IRD)', '(PUP)', '(SUS)', '(EXE)']
    is_on_ir = 1 if any(ir_tag in player_name for ir_tag in ir_variations) else 0

    # Remove IR variations from name if present
    for ir_tag in ir_variations:
        if ir_tag in player_name:
            player_name = player_name.replace(ir_tag, '').strip()

    # Handle character encoding issues
    player_name = player_name.replace('Piñeiro', 'Pineiro')

    # Remove text within parentheses (other than IR which was already handled)
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

            return (list(set(final_variations)), is_on_ir)  # Remove any duplicates

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
                return ([var.strip() for var in variations], is_on_ir)

    return ([player_name.strip()], is_on_ir)

def get_most_recent_file(directory):
    """Get the most recently created CSV file in the directory"""
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    # Get most recent file by creation time
    most_recent = max(csv_files, key=os.path.getctime)
    return most_recent

def count_weeks(weeks_str):
    """Count comma-separated weeks in the string"""
    if not weeks_str or weeks_str.strip() == '':
        return 0
    return len([w.strip() for w in weeks_str.split(',') if w.strip()])

def get_max_week(csv_data):
    """Find the maximum week number from all rows"""
    max_week = 0
    for row in csv_data:
        weeks_str = row.get('weeks', '')
        if weeks_str:
            weeks = [int(w.strip()) for w in weeks_str.split(',') if w.strip()]
            if weeks:
                max_week = max(max_week, max(weeks))
    return max_week

def load_starter_files(max_week):
    """Load all starter files from week_1.0 to max_week and count player appearances"""
    starters_data = defaultdict(int)  # (cleaned_name, pos, team) -> count

    print(f"\nLoading starter files from week 1 to {max_week}...")

    for week_num in range(1, max_week + 1):
        week_dir = os.path.join(GAMES_DIR, f"week_{week_num}.0", "clean")
        if not os.path.exists(week_dir):
            print(f"Warning: Week directory not found: {week_dir}")
            continue

        starter_files = glob.glob(os.path.join(week_dir, "*_starters_*.csv"))
        print(f"  Week {week_num}: Processing {len(starter_files)} starter files")

        for file in starter_files:
            with open(file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    player_name = row['Player']
                    pos = row['Pos']
                    team = row['team']

                    # Clean and standardize
                    cleaned_names, _ = clean_player_name(player_name)
                    standardized_pos = standardize_position(pos)
                    standardized_team = standardize_team_name(team)

                    # Store all name variations
                    for cleaned_name in cleaned_names:
                        key = (cleaned_name, standardized_pos, standardized_team)
                        starters_data[key] += 1

    return starters_data

def populate_gm_played(row):
    """Populate gm_played based on weeks column"""
    gm_played = row.get('gm_played', '').strip()
    weeks = row.get('weeks', '').strip()

    if not gm_played:  # NULL or empty
        row['gm_played'] = str(count_weeks(weeks))

    return row

def populate_gm_started(row, starters_data):
    """Populate gm_started based on starter files"""
    gm_started = row.get('gm_started', '').strip()

    if not gm_started:  # NULL or empty
        weeks = row.get('weeks', '').strip()
        gm_played = row.get('gm_played', '').strip()

        # Check if weeks is NULL or gm_played = 0
        if not weeks or (gm_played == '0'):
            row['gm_started'] = '0'
        else:
            # Look up in starters data
            player_name = row['plyr_name']
            pos = row['pos']

            # Need to check all team columns
            teams_to_check = []
            if row.get('current_team'):
                teams_to_check.append(row['current_team'])
            if row.get('former_team') and row.get('former_team') not in teams_to_check:
                teams_to_check.append(row['former_team'])
            if row.get('first_team') and row.get('first_team') not in teams_to_check:
                teams_to_check.append(row['first_team'])

            # Clean player name and get variations
            cleaned_names, _ = clean_player_name(player_name)

            # Count starts across all team/name variations
            total_starts = 0
            for team in teams_to_check:
                for cleaned_name in cleaned_names:
                    key = (cleaned_name, pos, team)
                    if key in starters_data:
                        total_starts += starters_data[key]

            row['gm_started'] = str(total_starts)

    return row

def main():
    print(f"Processing player game counter for Year: {YEAR}, Week: {WEEK}")

    # Get most recent file
    if not os.path.exists(PLYR_CLEAN_DIR):
        print(f"Error: Directory not found: {PLYR_CLEAN_DIR}")
        return

    input_file = get_most_recent_file(PLYR_CLEAN_DIR)
    print(f"\nInput file: {input_file}")

    # Create output filename
    base_name = os.path.basename(input_file)
    name_without_ext = os.path.splitext(base_name)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(PLYR_CLEAN_DIR, f"{name_without_ext}_populated_{timestamp}.csv")

    # Read CSV
    print("\nReading input CSV...")
    with open(input_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        csv_data = list(reader)
        fieldnames = reader.fieldnames

    print(f"Total rows read: {len(csv_data)}")

    # First pass: populate gm_played
    print("\nPopulating gm_played column...")
    gm_played_populated = 0
    for row in csv_data:
        original_value = row.get('gm_played', '').strip()
        populate_gm_played(row)
        if not original_value and row.get('gm_played'):
            gm_played_populated += 1

    print(f"Populated {gm_played_populated} NULL gm_played values")

    # Get max week to determine which starter files to load
    max_week = get_max_week(csv_data)
    print(f"\nMaximum week found in data: {max_week}")

    # Load starter files
    starters_data = load_starter_files(max_week)
    print(f"\nLoaded {len(starters_data)} unique player-team-position combinations from starter files")

    # Second pass: populate gm_started
    print("\nPopulating gm_started column...")
    gm_started_populated = 0
    for row in csv_data:
        original_value = row.get('gm_started', '').strip()
        populate_gm_started(row, starters_data)
        if not original_value and row.get('gm_started'):
            gm_started_populated += 1

    print(f"Populated {gm_started_populated} NULL gm_started values")

    # Write output
    print(f"\nWriting output file...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"\nSuccess!")
    print(f"Output file: {output_file}")
    print(f"Total rows processed: {len(csv_data)}")
    print(f"gm_played values populated: {gm_played_populated}")
    print(f"gm_started values populated: {gm_started_populated}")

if __name__ == "__main__":
    main()
