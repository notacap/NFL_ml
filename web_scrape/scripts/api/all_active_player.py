import requests
import csv
import os
from datetime import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scraper_utils import YEAR, ROOT_DATA_DIR

# Define output directory using hardcoded path
OUTPUT_DIR = r"C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\all_active_players"

def fetch_active_players():
    # Make API request
    url = "https://sports.core.api.espn.com/v3/sports/football/nfl/athletes?limit=20000&active=true"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for bad status codes
        data = response.json()
        
        # Include active status in extraction, don't filter yet
        players = []
        for player in data['items']:
            players.append({
                'player_id': player.get('id', ''),
                'display_name': player.get('displayName', ''),
                'full_name': player.get('fullName', ''),
                'active': str(player.get('active', False))  # Convert boolean to string for CSV
            })
        
        return players
        
    except requests.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None

def create_output_file(players_data):
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = os.path.join(OUTPUT_DIR, f"all_players_{timestamp}.csv")
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['player_id', 'display_name', 'full_name', 'active']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(players_data)
    
    # Count active/inactive players
    active_count = sum(1 for player in players_data if player['active'] == 'True')
    inactive_count = sum(1 for player in players_data if player['active'] == 'False')
    
    print(f"\nOutput file created: {output_file}")
    print(f"Total players: {len(players_data)}")
    print(f"Active players: {active_count}")
    print(f"Inactive players: {inactive_count}")

def main():
    print("Fetching players from ESPN API...")
    players = fetch_active_players()
    
    if players:
        create_output_file(players)
    else:
        print("No data to process.")

if __name__ == "__main__":
    main()
