import csv
from collections import defaultdict
import os
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clean_utils import YEAR, WEEK

# Define input and output paths using YEAR and WEEK variables
INPUT_DIR = rf"C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\{YEAR}\plyr\plyr_raw\{WEEK}"
OUTPUT_DIR = rf"C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\{YEAR}\plyr\plyr_clean\{WEEK}"

def read_input_csv():
    # Get the most recent file in the input directory that starts with 'plyr_raw'
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv') and f.startswith('plyr_raw')]
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(INPUT_DIR, x)))
    input_file = os.path.join(INPUT_DIR, latest_file)
    
    players = defaultdict(list)
    with open(input_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Create a unique key for each player
            key = (row['plyr_name'], row['pos'], row['age'], row['yrs_played'],
                   row['weight'] or row['height'])  # Use weight or height, whichever is available
            
            # Discard players with insufficient data
            if not all([row['age'], row['yrs_played'], row['weight'] or row['height']]):
                continue
            
            players[key].append(row)
    return players

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

def process_players(players, multi_team_players):
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
            processed_player['former_team'] = ''
            processed_player['first_team'] = ''
            processed_player['current_team_week'] = ''
            processed_player['former_team_last_week'] = ''
            processed_player['former_team_first_week'] = ''
            processed_player['first_team_last_week'] = ''
            # Ensure new columns are included for single-team players
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
    
    return processed_players

def write_output_csv(processed_players):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = os.path.join(OUTPUT_DIR, f"cleaned_players_{timestamp}.csv")
    
    fieldnames = ['plyr_name', 'current_team', 'pos', 'age', 'gm_played', 'gm_started', 
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
    players = read_input_csv()
    multi_team_players = identify_multi_team_players(players)
    processed_players = process_players(players, multi_team_players)
    write_output_csv(processed_players)

if __name__ == "__main__":
    main()
