"""
Team Defense Player Aggregated Stats Loader

Loads team-level defensive statistics aggregated from player data.
Provides opponent defense metrics for matchup-based predictions.

"""

from typing import List
import pandas as pd
from ..base_loader import BaseDataLoader


class TeamDefenseLoader(BaseDataLoader):
    """
    Loads tm_szn/tm_def_plyr_agg table - team defensive statistics.

    Features:
    - Pass defense metrics (completions, yards, TDs allowed)
    - Pressure metrics (sacks, hurries, knockdowns, blitzes)
    - Coverage metrics (interceptions, pass deflections)
    - Tackling metrics (combined, solo, TFL, missed tackles)
    - Efficiency metrics (completion %, passer rating allowed)
    - Per-game normalized stats

    Join Type:
        opponent_team - joins on [team_id, week_id] to base dataset's
        [next_opponent_team_id, week_id]

    Temporal Note:
        These are cumulative season stats through each week.
        When predicting week N performance, we use opponent's defensive
        stats through week N-1 (the stats available at prediction time).
    """

    @property
    def table_path(self) -> str:
        return 'tm_szn/tm_def_plyr_agg'

    @property
    def key_columns(self) -> List[str]:
        return ['team_id', 'season_id', 'week_id']

    @property
    def feature_columns(self) -> List[str]:
        return [
            # Raw counting stats
            'tm_def_int',
            'tm_def_pass_def',
            'tm_def_comb_tkl',
            'tm_def_solo_tkl',
            'tm_def_qb_hit',
            'tm_def_tfl',
            'tm_def_tgt',
            'tm_def_cmp',
            'tm_def_pass_yds',
            'tm_def_ay',
            'tm_def_yac',
            'tm_def_bltz',
            'tm_def_hrry',
            'tm_def_qbkd',
            'tm_def_sk',
            'tm_def_prss',
            'tm_def_mtkl',
            # Efficiency metrics
            'tm_def_cmp_pct',
            'tm_def_pass_yds_cmp',
            'tm_def_pass_yds_tgt',
            'tm_def_adot',
            'tm_def_yac_cmp',
            'tm_def_mtkl_pct',
            'tm_def_pass_rtg',
            'tm_def_sk_pct',
            'tm_def_int_pct',
            # Per-game stats
            'tm_def_tkl_pg',
            'tm_def_sk_pg',
            'tm_def_prss_pg',
            'tm_def_to',
            'tm_def_to_pg'
        ]

    @property
    def join_type(self) -> str:
        return self.JOIN_OPPONENT_TEAM

    def load(self) -> pd.DataFrame:
        """Load team defense aggregated stats."""
        self.logger.info("Loading team defense player aggregated stats...")

        # Load raw data from partitioned parquet
        df = self._load_partitioned_table()

        # Apply any transformations (none needed currently)
        df = self.transform(df)

        # Select only key + feature columns
        df = self._select_output_columns(df)

        # Validate output
        self._validate_output(df)

        return df
