"""
NFL FastR WR Loader

Loads advanced receiving metrics from NFL FastR data for wide receivers.
Provides target share, air yards, separation, and efficiency metrics.

"""

from typing import List
import pandas as pd
from ..base_loader import BaseDataLoader


class NFLFastrWRLoader(BaseDataLoader):
    """
    Loads plyr_gm/nfl_fastr_wr table - advanced WR receiving metrics from NFL FastR.

    Features:
    - Target share and air yards share metrics
    - Cushion and separation measurements
    - YAC and expected YAC comparisons
    - EPA and efficiency metrics (WOPR, RACR)

    Join Type:
        player_game - joins on [plyr_id, season_id, week_id]

    Temporal Note:
        No temporal shift required - these are same-game metrics.
    """

    @property
    def table_path(self) -> str:
        return 'plyr_gm/nfl_fastr_wr'

    @property
    def key_columns(self) -> List[str]:
        return ['plyr_id', 'season_id', 'week_id']

    @property
    def feature_columns(self) -> List[str]:
        return [
            'plyr_gm_rec_avg_cushion',
            'plyr_gm_rec_avg_separation',
            'plyr_gm_rec_avg_yac',
            'plyr_gm_rec_avg_expected_yac',
            'plyr_gm_rec_avg_yac_above_expectation',
            'plyr_gm_rec_pct_share_of_intended_ay',
            'plyr_gm_rec_tgt_share',
            'plyr_gm_rec_epa',
            'plyr_gm_rec_ay_share',
            'plyr_gm_rec_wopr',
            'plyr_gm_rec_racr'
        ]

    @property
    def join_type(self) -> str:
        return self.JOIN_PLAYER_GAME

    def load(self) -> pd.DataFrame:
        """Load NFL FastR WR advanced metrics."""
        self.logger.info("Loading NFL FastR WR advanced receiving metrics...")

        # Load raw data from partitioned parquet
        df = self._load_partitioned_table()

        # Apply any transformations (none needed currently)
        df = self.transform(df)

        # Select only key + feature columns
        df = self._select_output_columns(df)

        # Validate output
        self._validate_output(df)

        return df
