"""
NFL Data Loaders

Individual data source loaders for the dataset building pipeline.
"""

from .tm_def_plyr_agg_loader import TeamDefenseLoader
from .gm_weather_loader import GameWeatherLoader

__all__ = ['TeamDefenseLoader', 'GameWeatherLoader']
