import csv
from collections import defaultdict
import os
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clean_utils import YEAR, WEEK

# Define position compatibility groups
def positions_are_compatible(pos1, pos2):
    """Check if two positions should be considered the same for player matching"""
    if pos1 == pos2:
        return True
    # LB and DL are compatible (players can move between these positions)
    compatible_groups = [
        {'LB', 'DL'},
        {'K', 'P'}
    ]
    for group in compatible_groups:
        if pos1 in group and pos2 in group:
            return True
    return False

def players_match(player1, player2):
    """Check if two players match based on name, position compatibility, and at least 2 matching fields"""
    # First check: plyr_name must match and positions must be compatible
    if (player1['plyr_name'] != player2['plyr_name'] or
        not positions_are_compatible(player1['pos'], player2['pos'])):
        return False

    # Second check: count matching fields
    match_count = 0
    fields_to_check = [
        'age', 'weight', 'height', 'yrs_played',
        'plyr_college', 'plyr_birthdate', 'plyr_draft_tm'
    ]

    for field in fields_to_check:
        val1 = player1.get(field, '').strip()
        val2 = player2.get(field, '').strip()
        # Only count as match if both values exist and are equal
        if val1 and val2 and val1 == val2:
            match_count += 1

    # If at least 2 additional fields match, consider it a match
    return match_count >= 2

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
                # Create a unique key for each player
                key = (row['plyr_name'], row['pos'], row['age'], row['yrs_played'],
                       row['weight'] or row['height'])  # Use weight or height, whichever is available

                # Check if we should merge with an existing player with compatible position
                found_compatible = False
                for existing_key in list(players.keys()):
                    if (existing_key[0] == key[0] and  # Same name
                        positions_are_compatible(existing_key[1], key[1]) and  # Compatible positions
                        existing_key[2] == key[2] and  # Same age
                        existing_key[3] == key[3] and  # Same years played
                        existing_key[4] == key[4]):    # Same weight/height
                        # Add to existing key group
                        players[existing_key].append(row)
                        found_compatible = True
                        break

                if not found_compatible:
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

        # Decision logic
        if null_matches_count > 0 and null_matches_count == not_null_matches_count:
            # Remove all NULL rows, don't modify team fields
            excluded_null_indices.add(i)
            excluded_null_indices.update(null_match_indices)
            continue

        # Check if this player matches any NOT NULL weeks player
        found_match = False
        if not_null_matches_count > 0 and null_matches_count <= not_null_matches_count:
            # Only execute match logic if NULL matches <= NOT NULL matches
            for player in matching_not_null_players:
                null_team = null_player.get('team_name', '').strip()
                not_null_team = player.get('team_name', '').strip()

                # Edge case: if teams are different, this is a multi-team player
                if null_team and not_null_team and null_team != not_null_team:
                    # Check if former_team is already populated (indicating a third team)
                    if player.get('former_team', ''):
                        # Move former_team → first_team
                        player['first_team'] = player['former_team']

                        # Move current_team_week → former_team_first_week
                        player['former_team_first_week'] = player.get('current_team_week', '')

                        # Move former_team_last_week → first_team_last_week
                        player['first_team_last_week'] = player.get('former_team_last_week', '')

                        # Now set the new former team information
                        player['former_team'] = not_null_team

                        # Parse weeks to get min and max
                        weeks = [int(w) for w in player['weeks'].split(',')]
                        max_week = max(weeks)
                        player['former_team_last_week'] = str(max_week)

                        # Set current team week to former_team_last_week + 1
                        player['current_team_week'] = str(max_week + 1)

                        # Update team_name to the null player's team
                        player['team_name'] = null_team
                    else:
                        # Set former team information (two-team scenario)
                        player['former_team'] = not_null_team

                        # Parse weeks to get min and max
                        weeks = [int(w) for w in player['weeks'].split(',')]
                        max_week = max(weeks)
                        player['former_team_last_week'] = str(max_week)
                        player['former_team_first_week'] = str(min(weeks))

                        # Set current team week to former_team_last_week + 1
                        player['current_team_week'] = str(max_week + 1)

                        # Update team_name to the null player's team
                        player['team_name'] = null_team

                found_match = True
                break

        # If no match found, keep this NULL weeks player
        if not found_match and i not in excluded_null_indices:
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
        if key in multi_team_players:
            first_team, second_team, third_team = determine_team_order(player_data)
            
            # Sum up gm_played and gm_started for all teams
            total_gm_played = sum(int(team['gm_played']) for team in player_data if team['gm_played'])
            total_gm_started = sum(int(team['gm_started']) for team in player_data if team['gm_started'])
            
            if third_team is None and second_team is not None:
                first_weeks = [int(w) for w in first_team['weeks'].split(',')]
                second_weeks = [int(w) for w in second_team['weeks'].split(',')]
                
                if max(first_weeks) > max(second_weeks):
                    # Edge case: player returned to original team
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
            
            # Use the summed up values for gm_played and gm_started
            processed_player['gm_played'] = str(total_gm_played)
            processed_player['gm_started'] = str(total_gm_started)
            processed_player['is_on_ir'] = processed_player.get('is_on_ir', '')

            # Add new columns from the updated CSV structure
            processed_player['plyr_college'] = processed_player.get('plyr_college', '')
            processed_player['plyr_birthdate'] = processed_player.get('plyr_birthdate', '')
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
            # Set default values for team transition fields if not already set
            processed_player.setdefault('former_team', '')
            processed_player.setdefault('first_team', '')
            processed_player.setdefault('current_team_week', '')
            processed_player.setdefault('former_team_last_week', '')
            processed_player.setdefault('former_team_first_week', '')
            processed_player.setdefault('first_team_last_week', '')
            # Ensure new columns are included for single-team players
            processed_player['is_on_ir'] = processed_player.get('is_on_ir', '')
            processed_player['plyr_college'] = processed_player.get('plyr_college', '')
            processed_player['plyr_birthdate'] = processed_player.get('plyr_birthdate', '')
            processed_player['plyr_avg_value'] = processed_player.get('plyr_avg_value', '')
            processed_player['plyr_draft_tm'] = processed_player.get('plyr_draft_tm', '')
            processed_player['plyr_draft_rd'] = processed_player.get('plyr_draft_rd', '')
            processed_player['plyr_draft_pick'] = processed_player.get('plyr_draft_pick', '')
            processed_player['plyr_draft_yr'] = processed_player.get('plyr_draft_yr', '')
            # Keep gm_played and gm_started as they are for single-team players
            processed_player['gm_played'] = processed_player.get('gm_played', '')
            processed_player['gm_started'] = processed_player.get('gm_started', '')
        
        processed_players.append(processed_player)

    # Add unmatched NULL weeks players as-is
    for null_player in unmatched_null_weeks:
        processed_player = null_player.copy()
        processed_player['current_team'] = processed_player.pop('team_name')
        processed_player['former_team'] = ''
        processed_player['first_team'] = ''
        processed_player['current_team_week'] = ''
        processed_player['former_team_last_week'] = ''
        processed_player['former_team_first_week'] = ''
        processed_player['first_team_last_week'] = ''
        # Ensure all expected columns are included
        processed_player['is_on_ir'] = processed_player.get('is_on_ir', '')
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
    
    fieldnames = ['plyr_name', 'current_team', 'pos', 'age', 'gm_played', 'gm_started', 'is_on_ir',
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

if __name__ == "__main__":
    main()
