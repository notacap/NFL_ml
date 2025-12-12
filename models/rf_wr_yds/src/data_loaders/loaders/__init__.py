"""
NFL Data Loaders

Individual data source loaders for the modular dataset building pipeline.
Each loader handles a specific data source and implements the BaseDataLoader interface.

Future loaders to be added:
- PlayerReceivingLoader: plyr_gm_rec, plyr_rec
- PlayerInfoLoader: plyr, plyr_master
- SnapCountLoader: plyr_gm_snap_ct
- OpponentDefenseLoader: tm_def_vs_wr
- WeatherLoader: nfl_gm_weather
- TeamPassingLoader: tm_pass
- GameContextLoader: nfl_game_info
- InjuryReportLoader: injury_report
"""

# Loaders will be exported here as they are implemented
# from .player_receiving_loader import PlayerReceivingLoader
# from .player_info_loader import PlayerInfoLoader
# etc.

from .nfl_fastr_wr_loader import NFLFastrWRLoader
from .tm_def_plyr_agg_loader import TeamDefenseLoader

__all__ = ['NFLFastrWRLoader', 'TeamDefenseLoader']
