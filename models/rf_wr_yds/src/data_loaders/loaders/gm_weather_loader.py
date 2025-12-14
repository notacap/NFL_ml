"""
Game Weather Data Loader

Loads game-level weather data from nfl_gm_weather partitioned parquet files.
Provides weather conditions for game context in predictions.

"""

from typing import List
import pandas as pd
from ..base_loader import BaseDataLoader


class GameWeatherLoader(BaseDataLoader):
    """
    Loads gm_info/nfl_gm_weather table - game weather conditions.

    Features:
    - Temperature (kickoff, first half, second half, end of game)
    - Humidity levels
    - Feels-like temperature
    - Precipitation metrics (rain, snowfall, snow depth)
    - Wind conditions (speed, direction, gusts)

    Join Type:
        game - joins on [game_id] to base dataset

    Temporal Note:
        Weather data is game-level and should be joined to the player's
        NEXT game (prediction target), not current game.

    TODO: Data manipulation needed before joining to final dataset:
        - [ ] Determine join strategy (game_id mapping to player games)
        - [ ] Handle indoor stadium games (weather may be null/irrelevant)
        - [ ] Decide which time period features to use (kickoff vs averages)
        - [ ] Feature engineering (e.g., bad_weather_flag, wind_impact_score)
        - [ ] Null handling strategy
    """

    @property
    def table_path(self) -> str:
        return 'gm_info/nfl_gm_weather'

    @property
    def key_columns(self) -> List[str]:
        # TODO: Confirm key columns for joining
        # May need game_id mapping or season_id/week_id + team combination
        return ['game_id', 'season_id', 'week_id']

    @property
    def feature_columns(self) -> List[str]:
        # TODO: Select final feature set after data exploration
        # Currently listing all available weather metrics
        return [
            # Kickoff conditions
            'kickoff_temperature',
            'kickoff_humidity',
            'kickoff_feels_like_temperature',
            'kickoff_precipitation',
            'kickoff_rain',
            'kickoff_snowfall',
            'kickoff_snow_depth',
            'kickoff_wind_speed',
            'kickoff_wind_direction',
            'kickoff_wind_gusts',
            # First half conditions
            'first_half_temperature',
            'first_half_humidity',
            'first_half_feels_like_temperature',
            'first_half_precipitation',
            'first_half_rain',
            'first_half_snowfall',
            'first_half_snow_depth',
            'first_half_wind_speed',
            'first_half_wind_direction',
            'first_half_wind_gusts',
            # Second half conditions
            'second_half_temperature',
            'second_half_humidity',
            'second_half_feels_like_temperature',
            'second_half_precipitation',
            'second_half_rain',
            'second_half_snowfall',
            'second_half_snow_depth',
            'second_half_wind_speed',
            'second_half_wind_direction',
            'second_half_wind_gusts',
            # End of game conditions
            'end_of_game_temperature',
            'end_of_game_humidity',
            'end_of_game_feels_like_temperature',
            'end_of_game_precipitation',
            'end_of_game_rain',
            'end_of_game_snowfall',
            'end_of_game_snow_depth',
            'end_of_game_wind_speed',
            'end_of_game_wind_direction',
            'end_of_game_wind_gusts',
        ]

    @property
    def join_type(self) -> str:
        # TODO: May need custom join type for game-level data
        return self.JOIN_GAME

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply weather-specific transformations.

        TODO: Implement data manipulation logic:
            - Feature engineering (composite weather scores, binary flags)
            - Handle indoor stadiums
            - Aggregate time periods if needed
            - Null imputation strategy
        """
        # Placeholder - return unchanged for now
        return df

    def load(self) -> pd.DataFrame:
        """Load game weather data."""
        self.logger.info("Loading game weather data...")

        # Load raw data from partitioned parquet
        df = self._load_partitioned_table()

        # Apply transformations
        df = self.transform(df)

        # Select only key + feature columns
        df = self._select_output_columns(df)

        # Validate output
        self._validate_output(df)

        return df
