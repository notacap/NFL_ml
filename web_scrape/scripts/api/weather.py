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
    """
    # Get season_id for the current year
    season_id = get_season_id_for_year(YEAR)
    if season_id is None:
        print(f"No season found for year {YEAR}")
        return []
    
    print(f"Using season_id {season_id} for year {YEAR}")
    
    connection = get_database_connection()
    if connection is None:
        return []

    try:
        cursor = connection.cursor(dictionary=True)
        query = """
        SELECT ng.game_id, ng.week_id, ng.game_date, nt.stadium_latitude, nt.stadium_longitude
        FROM nfl_game ng
        JOIN nfl_team nt ON ng.home_team_id = nt.team_id
        WHERE ng.season_id = %s
        """
        cursor.execute(query, (season_id,))
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
    df['timezone_abbreviation'] = response.TimezoneAbbreviation()
    
    return df

def main():
    """Main execution function."""
    # Start logging session
    start_scraper_session("weather_api")
    
    try:
        print(f"Scraping NFL weather data for year: {YEAR}")
        
        # Create directory structure
        weather_dir = create_directories(YEAR, "weather_api")
        print(f"Data will be saved to: {weather_dir}")
        
        games = fetch_game_data()
        
        if not games:
            print("No games found to process")
            return
            
        print(f"Found {len(games)} games to process")
        
        for game in games:
            print(f"Processing game ID: {game['game_id']}")
            try:
                weather_data = get_weather_data(game)
                
                filename = f"weather_game_{game['game_id']}_week_{game['week_id']}.csv"
                save_data_to_csv(weather_data, weather_dir, filename)
                
            except Exception as e:
                print(f"Error processing game {game['game_id']}: {e}")
                log_failed_table(f"game_{game['game_id']}", f"Error processing weather data: {e}")
                continue
    
    finally:
        # End logging session
        end_scraper_session()

if __name__ == "__main__":
    main()
