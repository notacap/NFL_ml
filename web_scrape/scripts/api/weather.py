import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scraper_utils import *

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Load environment variables from database/.env
load_dotenv(dotenv_path=r'C:\Users\nocap\Desktop\code\NFL_ml\database\.env')

# Database connection configuration from .env
db_config = {
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

def get_database_connection():
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
    return None

def get_season_id_for_year(year):
    """
    Gets the season_id for the specified year from the nfl_season table.
    
    Args:
        year (int): The year to look up
    
    Returns:
        int: The season_id for the given year, or None if not found
    """
    connection = get_database_connection()
    if connection is None:
        return None

    try:
        cursor = connection.cursor()
        query = "SELECT season_id FROM nfl_season WHERE year = %s"
        cursor.execute(query, (year,))
        result = cursor.fetchone()
        return result[0] if result else None
    except Error as e:
        print(f"Error fetching season_id: {e}")
        return None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def fetch_game_data():
    """
    Fetches game data for the current year from scraper_utils.YEAR
    Filters by START_WEEK and END_WEEK range
    """
    # Get season_id for the current year
    season_id = get_season_id_for_year(YEAR)
    if season_id is None:
        print(f"No season found for year {YEAR}")
        return []

    print(f"Using season_id {season_id} for year {YEAR}")
    print(f"Filtering games for weeks {START_WEEK} to {END_WEEK}")

    connection = get_database_connection()
    if connection is None:
        return []

    try:
        cursor = connection.cursor(dictionary=True)
        query = """
        SELECT
            ng.game_id,
            ng.week_id,
            ng.game_date,
            nw.week_num,
            nt_home.stadium_latitude,
            nt_home.stadium_longitude,
            nt_home.team_name as home_team,
            nt_away.team_name as away_team
        FROM nfl_game ng
        JOIN nfl_team nt_home ON ng.home_team_id = nt_home.team_id
        JOIN nfl_team nt_away ON ng.away_team_id = nt_away.team_id
        JOIN nfl_week nw ON ng.week_id = nw.week_id AND ng.season_id = nw.season_id
        WHERE ng.season_id = %s
        AND nw.week_num >= %s
        AND nw.week_num <= %s
        ORDER BY nw.week_num, ng.game_date
        """
        cursor.execute(query, (season_id, START_WEEK, END_WEEK))
        games = cursor.fetchall()
        return games
    except Error as e:
        print(f"Error fetching game data: {e}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def get_weather_data(game):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": game['stadium_latitude'],
        "longitude": game['stadium_longitude'],
        "start_date": game['game_date'].strftime('%Y-%m-%d'),
        "end_date": game['game_date'].strftime('%Y-%m-%d'),
        "hourly": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation", "rain", "snowfall", "snow_depth", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "auto"
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
        "apparent_temperature": hourly.Variables(2).ValuesAsNumpy(),
        "precipitation": hourly.Variables(3).ValuesAsNumpy(),
        "rain": hourly.Variables(4).ValuesAsNumpy(),
        "snowfall": hourly.Variables(5).ValuesAsNumpy(),
        "snow_depth": hourly.Variables(6).ValuesAsNumpy(),
        "wind_speed_10m": hourly.Variables(7).ValuesAsNumpy(),
        "wind_direction_10m": hourly.Variables(8).ValuesAsNumpy(),
        "wind_gusts_10m": hourly.Variables(9).ValuesAsNumpy()
    }

    df = pd.DataFrame(data=hourly_data)
    df['game_id'] = game['game_id']
    df['week_id'] = game['week_id']
    df['year'] = YEAR
    df['timezone_abbreviation'] = response.TimezoneAbbreviation()

    return df

def create_week_directory(week_num):
    """
    Creates directory structure for a specific week.

    Args:
        week_num (float or int): Week number

    Returns:
        str: Path to the week directory
    """
    weather_dir = os.path.join(ROOT_DATA_DIR, str(YEAR), "weather")
    week_dir = os.path.join(weather_dir, f"week_{week_num}")

    if not os.path.exists(week_dir):
        os.makedirs(week_dir, exist_ok=True)

    return week_dir

def generate_weather_filename(home_team, away_team, week_num):
    """
    Generates filename for weather data CSV.

    Args:
        home_team (str): Home team name
        away_team (str): Away team name
        week_num (float or int): Week number

    Returns:
        str: Formatted filename
    """
    from datetime import datetime

    # Clean team names (replace spaces with underscores)
    home_clean = home_team.replace(" ", "_")
    away_clean = away_team.replace(" ", "_")

    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{home_clean}_{away_clean}_wk{week_num}_{YEAR}_weather_{timestamp}.csv"
    return filename

def main():
    """Main execution function."""
    # Start logging session
    start_scraper_session("weather_api")

    try:
        print(f"Scraping NFL weather data for year: {YEAR}")
        print(f"Week range: {START_WEEK} - {END_WEEK}")

        games = fetch_game_data()

        if not games:
            print("No games found to process")
            return

        print(f"Found {len(games)} games to process")

        for game in games:
            week_num = game['week_num']
            home_team = game['home_team']
            away_team = game['away_team']

            print(f"\nProcessing Week {week_num}: {away_team} @ {home_team}")
            print(f"Game ID: {game['game_id']}")

            try:
                # Create week directory
                week_dir = create_week_directory(week_num)

                # Get weather data
                weather_data = get_weather_data(game)

                # Generate filename and save
                filename = generate_weather_filename(home_team, away_team, week_num)
                csv_path = os.path.join(week_dir, filename)

                weather_data.to_csv(csv_path, index=False)
                print(f"Saved: {filename}")
                log_successful_table(filename, csv_path, f"game_{game['game_id']}")

            except Exception as e:
                print(f"Error processing game {game['game_id']}: {e}")
                log_failed_table(f"{home_team}_{away_team}_wk{week_num}", f"Error processing weather data: {e}")
                continue

    finally:
        # End logging session
        end_scraper_session()

if __name__ == "__main__":
    main()
