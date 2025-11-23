import csv
from collections import defaultdict
import os
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clean_utils import YEAR, WEEK

merged_players = []
dropped_players = []
birthdate_conflicts = []
alt_pos_populated = []

def birthdates_match(date1_str, date2_str):
    """Check if two birthdates match or differ by only 1 day"""
    if not date1_str or not date2_str:
        return False
    if date1_str == date2_str:
        return True
    
    try:
        date1 = datetime.strptime(date1_str, '%m/%d/%Y')
        date2 = datetime.strptime(date2_str, '%m/%d/%Y')
        diff = abs((date1 - date2).days)
        return diff == 1
    except:
        return False

def get_earlier_birthdate(date1_str, date2_str, player_name=None):
    """Return the earlier of two birthdates"""
    try:
        date1 = datetime.strptime(date1_str, '%m/%d/%Y')
        date2 = datetime.strptime(date2_str, '%m/%d/%Y')
        if date1_str != date2_str and player_name:
            birthdate_conflicts.append({
                'player': player_name,
                'birthdate1': date1_str,
                'birthdate2': date2_str,
                'selected': date1_str if date1 < date2 else date2_str
            })
        return date1_str if date1 < date2 else date2_str
    except:
        return date1_str

def players_match(player1, player2):
    """Check if two players match based on name and at least 4 matching fields"""
    if player1['plyr_name'] != player2['plyr_name']:
        return False

    match_count = 0
    fields_to_check = [
        'pos', 'age', 'weight', 'height', 'yrs_played',
        'plyr_college', 'plyr_draft_tm'
    ]

    for field in fields_to_check:
        val1 = player1.get(field, '').strip()
        val2 = player2.get(field, '').strip()
        if val1 and val2 and val1 == val2:
            match_count += 1
    
    if birthdates_match(player1.get('plyr_birthdate', '').strip(), player2.get('plyr_birthdate', '').strip()):
        match_count += 1

    return match_count >= 4

# Define input and output paths using YEAR and WEEK variables
INPUT_DIR = rf"C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\{YEAR}\plyr\plyr_raw\{WEEK}"
OUTPUT_DIR = rf"C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\{YEAR}\plyr\plyr_clean\{WEEK}"

def read_input_csv():
    # Get the most recent file in the input directory that starts with 'plyr_raw'
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv') and f.startswith('plyr_raw')]
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(INPUT_DIR, x)))
    input_file = os.path.join(INPUT_DIR, latest_file)

    players = defaultdict(list)
    null_weeks_players = []

    with open(input_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Discard players with insufficient data
            if not all([row['age'], row['yrs_played'], row['weight'] or row['height']]):
                continue

            # Separate players with NULL weeks
            if not row.get('weeks') or row['weeks'].strip() == '':
                null_weeks_players.append(row)
            else:
                found_match = False
                for existing_players in list(players.values()):
                    if existing_players and players_match(row, existing_players[0]):
                        existing_players.append(row)
                        found_match = True
                        merged_players.append({
                            'player': row['plyr_name'],
                            'team1': existing_players[0]['team_name'],
                            'team2': row['team_name']
                        })
                        break

                if not found_match:
                    key = (row['plyr_name'], id(row))
                    players[key].append(row)

    return players, null_weeks_players

def filter_null_weeks_players(null_weeks_players, players):
    """
    Match NULL weeks players with NOT NULL weeks players and other NULL weeks players.
    Returns only unmatched NULL weeks players.
    """
    unmatched_null_weeks = []
    excluded_null_indices = set()  # Track NULL players to completely exclude

    for i, null_player in enumerate(null_weeks_players):
        if i in excluded_null_indices:
            continue

        # Count duplicate NULL weeks matches (excluding self)
        null_matches_count = 0
        null_match_indices = []
        for j, other_null_player in enumerate(null_weeks_players):
            if i != j and j not in excluded_null_indices:
                if players_match(null_player, other_null_player):
                    null_matches_count += 1
                    null_match_indices.append(j)

        # Count NOT NULL weeks matches
        not_null_matches_count = 0
        matching_not_null_players = []
        for player_data_list in players.values():
            for player in player_data_list:
                if players_match(null_player, player):
                    not_null_matches_count += 1
                    matching_not_null_players.append(player)

        if null_matches_count > 0 and null_matches_count == not_null_matches_count:
            excluded_null_indices.add(i)
            excluded_null_indices.update(null_match_indices)
            dropped_players.append({
                'player': null_player['plyr_name'],
                'team': null_player['team_name'],
                'reason': 'NULL weeks matched with NOT NULL weeks'
            })
            continue

        if not_null_matches_count > 0:
            excluded_null_indices.add(i)
            dropped_players.append({
                'player': null_player['plyr_name'],
                'team': null_player['team_name'],
                'reason': 'NULL weeks matched with NOT NULL weeks'
            })
            continue

        # If no match found, keep this NULL weeks player
        if i not in excluded_null_indices:
            unmatched_null_weeks.append(null_player)

    return unmatched_null_weeks

def identify_multi_team_players(players):
    multi_team_players = {}
    for key, player_data in players.items():
        if len(set(p['team_name'] for p in player_data)) > 1:
            multi_team_players[key] = player_data
    return multi_team_players

def determine_team_order(player_data):
    def get_min_max_weeks(weeks_str):
        weeks = [int(w) for w in weeks_str.split(',')]
        return min(weeks), max(weeks)

    sorted_data = sorted(player_data, key=lambda x: get_min_max_weeks(x['weeks']))
    
    if len(sorted_data) == 2:
        first_min, first_max = get_min_max_weeks(sorted_data[0]['weeks'])
        second_min, second_max = get_min_max_weeks(sorted_data[1]['weeks'])
        
        if first_min < second_min < first_max:
            # Edge case: player returned to original team
            return sorted_data[0], sorted_data[1], None
        else:
            return sorted_data[0], sorted_data[1], None
    elif len(sorted_data) == 3:
        return sorted_data[0], sorted_data[1], sorted_data[2]
    else:
        return sorted_data[0], None, None

def process_players(players, multi_team_players, unmatched_null_weeks):
    processed_players = []
    for key, player_data in players.items():
        all_positions = set(p['pos'] for p in player_data)
        primary_pos = player_data[0]['pos']
        alt_pos = ','.join(sorted(all_positions - {primary_pos})) if len(all_positions) > 1 else ''
        
        all_birthdates = [p.get('plyr_birthdate', '').strip() for p in player_data if p.get('plyr_birthdate', '').strip()]
        if len(all_birthdates) > 1:
            earliest_birthdate = all_birthdates[0]
            for bd in all_birthdates[1:]:
                earliest_birthdate = get_earlier_birthdate(earliest_birthdate, bd, player_data[0]['plyr_name'])
        elif all_birthdates:
            earliest_birthdate = all_birthdates[0]
        else:
            earliest_birthdate = ''
        
        if alt_pos:
            alt_pos_populated.append({
                'player': player_data[0]['plyr_name'],
                'primary_pos': primary_pos,
                'alt_pos': alt_pos
            })
        
        if key in multi_team_players:
            first_team, second_team, third_team = determine_team_order(player_data)
            
            total_gm_played = sum(int(team['gm_played']) for team in player_data if team['gm_played'])
            total_gm_started = sum(int(team['gm_started']) for team in player_data if team['gm_started'])
            
            if third_team is None and second_team is not None:
                first_weeks = [int(w) for w in first_team['weeks'].split(',')]
                second_weeks = [int(w) for w in second_team['weeks'].split(',')]
                
                if max(first_weeks) > max(second_weeks):
                    current_team = first_team
                    former_team = second_team
                    first_team = None
                else:
                    current_team = second_team
                    former_team = first_team
                    first_team = None
            else:
                current_team = third_team if third_team else second_team
                former_team = second_team if third_team else first_team
            
            processed_player = current_team.copy()
            processed_player['current_team'] = processed_player.pop('team_name')
            processed_player['former_team'] = former_team['team_name'] if former_team else ''
            processed_player['first_team'] = first_team['team_name'] if first_team else ''
            
            processed_player['gm_played'] = str(total_gm_played)
            processed_player['gm_started'] = str(total_gm_started)
            processed_player['is_on_ir'] = processed_player.get('is_on_ir', '')

            processed_player['alt_pos'] = alt_pos
            processed_player['plyr_college'] = processed_player.get('plyr_college', '')
            processed_player['plyr_birthdate'] = earliest_birthdate
            processed_player['plyr_avg_value'] = processed_player.get('plyr_avg_value', '')
            processed_player['plyr_draft_tm'] = processed_player.get('plyr_draft_tm', '')
            processed_player['plyr_draft_rd'] = processed_player.get('plyr_draft_rd', '')
            processed_player['plyr_draft_pick'] = processed_player.get('plyr_draft_pick', '')
            processed_player['plyr_draft_yr'] = processed_player.get('plyr_draft_yr', '')
            
            # Calculate week information
            current_weeks = [int(w) for w in current_team['weeks'].split(',')]
            former_weeks = [int(w) for w in former_team['weeks'].split(',')] if former_team else []
            first_weeks = [int(w) for w in first_team['weeks'].split(',')] if first_team else []
            
            # Handle edge case
            if first_team is None and max(current_weeks) > max(former_weeks):
                # Player returned to original team
                processed_player['current_team_week'] = min([w for w in current_weeks if w > max(former_weeks)])
            else:
                processed_player['current_team_week'] = min(current_weeks)
            
            processed_player['former_team_last_week'] = max(former_weeks) if former_weeks else ''
            processed_player['former_team_first_week'] = min(former_weeks) if former_weeks else ''
            processed_player['first_team_last_week'] = max(first_weeks) if first_weeks else ''
            
            # Combine weeks from all teams
            all_weeks = sorted(set(current_weeks + former_weeks + first_weeks))
            processed_player['weeks'] = ','.join(map(str, all_weeks))
            
        else:
            processed_player = player_data[0].copy()
            processed_player['current_team'] = processed_player.pop('team_name')
            processed_player.setdefault('former_team', '')
            processed_player.setdefault('first_team', '')
            processed_player.setdefault('current_team_week', '')
            processed_player.setdefault('former_team_last_week', '')
            processed_player.setdefault('former_team_first_week', '')
            processed_player.setdefault('first_team_last_week', '')
            processed_player['is_on_ir'] = processed_player.get('is_on_ir', '')
            processed_player['alt_pos'] = alt_pos
            processed_player['plyr_college'] = processed_player.get('plyr_college', '')
            processed_player['plyr_birthdate'] = earliest_birthdate
            processed_player['plyr_avg_value'] = processed_player.get('plyr_avg_value', '')
            processed_player['plyr_draft_tm'] = processed_player.get('plyr_draft_tm', '')
            processed_player['plyr_draft_rd'] = processed_player.get('plyr_draft_rd', '')
            processed_player['plyr_draft_pick'] = processed_player.get('plyr_draft_pick', '')
            processed_player['plyr_draft_yr'] = processed_player.get('plyr_draft_yr', '')
            processed_player['gm_played'] = processed_player.get('gm_played', '')
            processed_player['gm_started'] = processed_player.get('gm_started', '')
        
        processed_players.append(processed_player)

    for null_player in unmatched_null_weeks:
        processed_player = null_player.copy()
        processed_player['current_team'] = processed_player.pop('team_name')
        processed_player['former_team'] = ''
        processed_player['first_team'] = ''
        processed_player['current_team_week'] = ''
        processed_player['former_team_last_week'] = ''
        processed_player['former_team_first_week'] = ''
        processed_player['first_team_last_week'] = ''
        processed_player['is_on_ir'] = processed_player.get('is_on_ir', '')
        processed_player['alt_pos'] = ''
        processed_player['plyr_college'] = processed_player.get('plyr_college', '')
        processed_player['plyr_birthdate'] = processed_player.get('plyr_birthdate', '')
        processed_player['plyr_avg_value'] = processed_player.get('plyr_avg_value', '')
        processed_player['plyr_draft_tm'] = processed_player.get('plyr_draft_tm', '')
        processed_player['plyr_draft_rd'] = processed_player.get('plyr_draft_rd', '')
        processed_player['plyr_draft_pick'] = processed_player.get('plyr_draft_pick', '')
        processed_player['plyr_draft_yr'] = processed_player.get('plyr_draft_yr', '')
        processed_player['gm_played'] = processed_player.get('gm_played', '')
        processed_player['gm_started'] = processed_player.get('gm_started', '')
        processed_player['weeks'] = processed_player.get('weeks', '')
        processed_players.append(processed_player)

    return processed_players

def write_output_csv(processed_players):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = os.path.join(OUTPUT_DIR, f"cleaned_players_{timestamp}.csv")
    
    fieldnames = ['plyr_name', 'current_team', 'pos', 'alt_pos', 'age', 'gm_played', 'gm_started', 'is_on_ir',
                  'weight', 'height', 'yrs_played', 'plyr_college', 'plyr_birthdate', 'plyr_avg_value',
                  'plyr_draft_tm', 'plyr_draft_rd', 'plyr_draft_pick', 'plyr_draft_yr', 'former_team',
                  'first_team', 'current_team_week', 'former_team_last_week', 'former_team_first_week',
                  'first_team_last_week', 'weeks']
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for player in processed_players:
            writer.writerow(player)
    
    print(f"Output file created: {output_file}")

def main():
    players, null_weeks_players = read_input_csv()
    unmatched_null_weeks = filter_null_weeks_players(null_weeks_players, players)

    multi_team_players = identify_multi_team_players(players)
    processed_players = process_players(players, multi_team_players, unmatched_null_weeks)
    write_output_csv(processed_players)
    
    print("\n=== PROCESSING SUMMARY ===")
    print(f"\nMerged Players: {len(merged_players)}")
    for item in merged_players:
        print(f"  - {item['player']}: {item['team1']} + {item['team2']}")
    
    print(f"\nDropped Players: {len(dropped_players)}")
    for item in dropped_players:
        print(f"  - {item['player']} ({item['team']}): {item['reason']}")
    
    print(f"\nBirthdate Conflicts Resolved: {len(birthdate_conflicts)}")
    for item in birthdate_conflicts:
        print(f"  - {item['player']}: {item['birthdate1']} vs {item['birthdate2']} → Selected {item['selected']}")
    
    print(f"\nPlayers with Alt Position: {len(alt_pos_populated)}")
    for item in alt_pos_populated:
        print(f"  - {item['player']}: {item['primary_pos']} (alt: {item['alt_pos']})")

if __name__ == "__main__":
    main()
