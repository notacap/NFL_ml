#!/usr/bin/env python3

import sys
import os
import pandas as pd
import glob
from datetime import datetime, time
import pytz
from pathlib import Path
from decimal import Decimal, InvalidOperation

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_utils import (
    DatabaseConnector,
    load_csv_data,
    handle_null_values,
    batch_upsert_data,
    get_season_id,
    get_week_id,
    create_table_if_not_exists,
    YEAR,
    WEEK_START,
    WEEK_END
)

def create_nfl_gm_weather_table(db: DatabaseConnector) -> bool:
    """Create nfl_gm_weather table if it doesn't exist"""
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS nfl_gm_weather (
        gm_weather_id INT AUTO_INCREMENT PRIMARY KEY,
        game_id INT,
        week_id INT,
        season_id INT,
        kickoff_temperature DECIMAL(7,4),
        kickoff_humidity DECIMAL(7,4),
        kickoff_feels_like_temperature DECIMAL(7,4),
        kickoff_precipitation DECIMAL(5,4),
        kickoff_rain DECIMAL(5,4),
        kickoff_snowfall DECIMAL(5,4),
        kickoff_snow_depth DECIMAL(5,4),
        kickoff_wind_speed DECIMAL(6,4),
        kickoff_wind_direction DECIMAL(7,4),
        kickoff_wind_gusts DECIMAL(6,4),
        first_half_temperature DECIMAL(7,4),
        first_half_humidity DECIMAL(7,4),
        first_half_feels_like_temperature DECIMAL(7,4),
        first_half_precipitation DECIMAL(5,4),
        first_half_rain DECIMAL(5,4),
        first_half_snowfall DECIMAL(5,4),
        first_half_snow_depth DECIMAL(5,4),
        first_half_wind_speed DECIMAL(6,4),
        first_half_wind_direction DECIMAL(7,4),
        first_half_wind_gusts DECIMAL(6,4),
        second_half_temperature DECIMAL(7,4),
        second_half_humidity DECIMAL(7,4),
        second_half_feels_like_temperature DECIMAL(7,4),
        second_half_precipitation DECIMAL(5,4),
        second_half_rain DECIMAL(5,4),
        second_half_snowfall DECIMAL(5,4),
        second_half_snow_depth DECIMAL(5,4),
        second_half_wind_speed DECIMAL(6,4),
        second_half_wind_direction DECIMAL(7,4),
        second_half_wind_gusts DECIMAL(6,4),
        end_of_game_temperature DECIMAL(7,4),
        end_of_game_humidity DECIMAL(7,4),
        end_of_game_feels_like_temperature DECIMAL(7,4),
        end_of_game_precipitation DECIMAL(5,4),
        end_of_game_rain DECIMAL(5,4),
        end_of_game_snowfall DECIMAL(5,4),
        end_of_game_snow_depth DECIMAL(5,4),
        end_of_game_wind_speed DECIMAL(6,4),
        end_of_game_wind_direction DECIMAL(7,4),
        end_of_game_wind_gusts DECIMAL(6,4),
        UNIQUE KEY (game_id, week_id),
        FOREIGN KEY (game_id) REFERENCES nfl_game(game_id),
        FOREIGN KEY (week_id) REFERENCES nfl_week(week_id),
        FOREIGN KEY (season_id) REFERENCES nfl_season(season_id)
    );
    """
    
    return create_table_if_not_exists(db, "nfl_gm_weather", create_table_sql)

def fetch_game_times(db: DatabaseConnector, season_id: int) -> dict:
    """Fetch game times for the specified season"""
    query = "SELECT game_id, game_time FROM nfl_game WHERE season_id = %s"
    results = db.fetch_all(query, (season_id,))
    return {result[0]: result[1] for result in results}

def convert_to_eastern_time(time_str, timezone_abbr):
    timezone_abbr = timezone_abbr.decode('utf-8') if isinstance(timezone_abbr, bytes) else timezone_abbr.strip("b'")
    
    timezone_map = {
        'EDT': 'US/Eastern',
        'CDT': 'US/Central',
        'MDT': 'US/Mountain',
        'PDT': 'US/Pacific'
    }
    
    source_tz = pytz.timezone(timezone_map.get(timezone_abbr, 'US/Eastern'))
    eastern_tz = pytz.timezone('US/Eastern')
    
    dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S%z')
    dt = source_tz.localize(dt.replace(tzinfo=None))
    
    eastern_time = dt.astimezone(eastern_tz)
    return eastern_time.strftime('%H:%M:%S')

def process_weather_data(file_path, game_times):
    """Process weather data from CSV file"""
    df = load_csv_data(file_path)
    if df.empty:
        return None
        
    game_id = df['game_id'].iloc[0]
    week_id = df['week_id'].iloc[0]
    game_time = game_times.get(game_id)
    
    if game_time is None:
        print(f"[WARNING] Game time not found for game_id {game_id}")
        return None

    # Convert game_time to datetime.time object
    game_time = datetime.strptime(str(game_time), '%H:%M:%S').time()

    # Convert weather data times to Eastern Time
    df['eastern_time'] = df.apply(lambda row: convert_to_eastern_time(row['date'], row['timezone_abbreviation']), axis=1)

    # Convert eastern_time to datetime.time objects with explicit format
    df['eastern_time'] = pd.to_datetime(df['eastern_time'], format='%H:%M:%S').dt.time

    # Calculate time difference in minutes
    df['time_diff'] = df['eastern_time'].apply(lambda x: 
        (datetime.combine(datetime.min, x) - 
         datetime.combine(datetime.min, game_time)).total_seconds() / 60)

    # Find the closest time to kickoff
    kickoff_index = df['time_diff'].abs().idxmin()

    weather_data = {
        'kickoff': df.iloc[kickoff_index],
        'first_half': df.iloc[kickoff_index + 1] if kickoff_index + 1 < len(df) else None,
        'second_half': df.iloc[kickoff_index + 2] if kickoff_index + 2 < len(df) else None,
        'end_of_game': df.iloc[kickoff_index + 3] if kickoff_index + 3 < len(df) else None
    }

    return weather_data, game_id, week_id

def safe_decimal(value, max_digits=7, decimal_places=4):
    """Convert value to Decimal with specified precision, handling overflow"""
    if pd.isna(value):
        return None
    try:
        dec_value = Decimal(str(value)).quantize(Decimal(f'0.{"0" * decimal_places}'))
        max_value = Decimal('9' * (max_digits - decimal_places) + '.' + '9' * decimal_places)
        if dec_value > max_value:
            print(f"[WARNING] Value {dec_value} exceeds maximum. Setting to {max_value}")
            return max_value
        return dec_value
    except InvalidOperation:
        print(f"[WARNING] Could not convert '{value}' to Decimal. Setting to None.")
        return None

def prepare_upsert_data(weather_data, game_id, week_id, season_id):
    data = {
        'game_id': game_id,
        'week_id': week_id,
        'season_id': season_id
    }

    for prefix, row in weather_data.items():
        if row is not None:
            data.update({
                f'{prefix}_temperature': safe_decimal(row['temperature_2m']),
                f'{prefix}_humidity': safe_decimal(row['relative_humidity_2m']),
                f'{prefix}_feels_like_temperature': safe_decimal(row['apparent_temperature']),
                f'{prefix}_precipitation': safe_decimal(row['precipitation']),
                f'{prefix}_rain': safe_decimal(row['rain']),
                f'{prefix}_snowfall': safe_decimal(row['snowfall']),
                f'{prefix}_snow_depth': safe_decimal(row['snow_depth']),
                f'{prefix}_wind_speed': safe_decimal(row['wind_speed_10m']),
                f'{prefix}_wind_direction': safe_decimal(row['wind_direction_10m']),
                f'{prefix}_wind_gusts': safe_decimal(row['wind_gusts_10m'])
            })

    return data

def get_weather_csv_files() -> list:
    """Get weather CSV files from the specified directory structure based on WEEK_START and WEEK_END."""
    base_dir = f"C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\web_scrape\\scraped_data\\{YEAR}\\weather"
    csv_files = []

    for week in range(WEEK_START, WEEK_END + 1):
        week_dir = os.path.join(base_dir, f"week_{week}")

        if os.path.exists(week_dir):
            # New filename pattern: {home_team}_{away_team}_wk{week}_{YEAR}_weather_{date}_{time}.csv
            # Example: Arizona_Cardinals_Kansas_City_Chiefs_wk1_2022_weather_20251010_172455.csv
            pattern = os.path.join(week_dir, "*_wk*_*_weather_*.csv")
            week_files = glob.glob(pattern)

            for file_path in week_files:
                csv_files.append((week, file_path))

            if not week_files:
                print(f"[WARNING] No weather files found in {week_dir}")
        else:
            print(f"[WARNING] Week directory not found: {week_dir}")

    if not csv_files:
        raise FileNotFoundError(f"No weather CSV files found in {base_dir} for weeks {WEEK_START}-{WEEK_END}")

    # Sort by week number and then by filename
    csv_files = sorted(csv_files, key=lambda x: (x[0], x[1]))

    print(f"Found {len(csv_files)} weather CSV files")
    print(f"Processing weeks {WEEK_START} to {WEEK_END}")

    return csv_files

def main():
    """Main execution function"""
    
    # Initialize database connection
    db = DatabaseConnector()
    
    try:
        # Connect to database
        if not db.connect():
            print("Failed to connect to database")
            return False
        
        print("Connected to database successfully")
        
        # Create table if it doesn't exist
        if not create_nfl_gm_weather_table(db):
            print("Failed to create nfl_gm_weather table")
            return False
        
        # Get season_id for the current year
        season_id = get_season_id(db, YEAR)
        print(f"Processing weather data for season {YEAR} (ID: {season_id})")
        
        # Fetch game times for the season
        game_times = fetch_game_times(db, season_id)
        print(f"Fetched {len(game_times)} game times from the database")

        # Get weather CSV files
        try:
            weather_files = get_weather_csv_files()
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            return False

        total_files = 0
        processed_records = []
        skipped_files = 0

        for week, file_path in weather_files:
            total_files += 1
            filename = os.path.basename(file_path)
            print(f"Processing file: {filename} (Week {week})")
            
            result = process_weather_data(file_path, game_times)
            
            if result:
                try:
                    # Extract weather_data, game_id, and week_id from CSV data
                    weather_data, game_id, week_id = result

                    upsert_data = prepare_upsert_data(weather_data, game_id, week_id, season_id)
                    processed_records.append(upsert_data)
                    
                    print(f"Successfully processed file: {filename} (game_id: {game_id}, week_id: {week_id})")
                    
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
                    skipped_files += 1
            else:
                skipped_files += 1
                print(f"Skipped file due to missing data: {filename}")

        if processed_records:
            # Convert to DataFrame and insert using batch upsert
            weather_df = pd.DataFrame(processed_records)
            weather_df = handle_null_values(weather_df)
            
            print(f"Inserting {len(weather_df)} weather records into nfl_gm_weather table...")
            success = batch_upsert_data(db, 'nfl_gm_weather', weather_df)
            
            if success:
                print(f"[SUCCESS] Weather data insertion completed successfully!")
                print(f"Total files: {total_files}")
                print(f"Processed files: {len(processed_records)}")
                print(f"Skipped files: {skipped_files}")
                return True
            else:
                print(f"[FAILED] Weather data insertion failed!")
                return False
        else:
            print("No data to insert")
            return False
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        return False
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    main()
