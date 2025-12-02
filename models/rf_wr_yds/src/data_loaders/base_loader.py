"""
Base Data Loader

Abstract base class providing shared functionality for all NFL data source loaders.
Handles partitioned parquet loading, validation, and temporal alignment.

"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import logging


class BaseDataLoader(ABC):
    """
    Abstract base class for NFL data source loaders.

    Provides:
    - Partitioned parquet loading (season/week structure)
    - Temporal alignment utilities
    - Validation framework
    - Logging infrastructure

    Subclasses must implement:
    - table_path: Relative path to the parquet table
    - key_columns: Columns used for joining
    - feature_columns: Columns this loader contributes
    - load(): Load and return processed data

    Optional overrides:
    - join_type: How this loader joins to the base dataset
    - transform(): Apply loader-specific transformations
    """

    # Join type constants
    JOIN_PLAYER_GAME = 'player_game'                # [plyr_id, season_id, week_id]
    JOIN_TEAM = 'team'                              # [team_id, season_id, week_id] with temporal shift
    JOIN_GAME = 'game'                              # [game_id]
    JOIN_OPPONENT_TEAM = 'opponent_team'            # [opp_team_id, season_id, week_id] with temporal shift
    JOIN_PLAYER = 'player'                          # [plyr_id] only (single-season player info)
    JOIN_PLAYER_CROSS_SEASON = 'player_cross_season'  # [plyr_guid] for cross-season player tracking

    def __init__(
        self,
        base_data_path: Path,
        seasons: List[int],
        weeks: Optional[List[int]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the data loader.

        Args:
            base_data_path: Root path to parquet data (e.g., parquet_files/clean)
            seasons: List of seasons to load
            weeks: Optional list of weeks to load (None for all)
            logger: Optional logger instance
        """
        self.base_data_path = Path(base_data_path)
        self.seasons = seasons
        self.weeks = weeks
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.validation_results: Dict = {}

    @property
    @abstractmethod
    def table_path(self) -> str:
        """
        Return the relative path to the parquet table.

        Examples:
            'plyr_gm/plyr_gm_rec'
            'tm_szn/tm_def_vs_wr'
            'gm_info/nfl_gm_weather'
        """
        pass

    @property
    @abstractmethod
    def key_columns(self) -> List[str]:
        """
        Return the columns used for joining to the base dataset.

        Examples:
            ['plyr_id', 'season_id', 'week_id']  # player_game join
            ['game_id']                           # game join
            ['team_id', 'season_id', 'week_id']  # team join
        """
        pass

    @property
    @abstractmethod
    def feature_columns(self) -> List[str]:
        """
        Return the columns this loader contributes as features.

        These are the columns (beyond key_columns) that will be
        added to the dataset when this loader is joined.
        """
        pass

    @property
    def join_type(self) -> str:
        """
        Return the join type for this loader.

        Options:
            'player_game': Direct join on [plyr_id, season_id, week_id]
            'team': Join on [team_id, season_id, week_id] with temporal shift
            'game': Join on [game_id]
            'opponent_team': Join on [opp_team_id, season_id, week_id] with shift
            'player': Join on [plyr_id] only (static info)

        Default is 'player_game' for backwards compatibility.
        """
        return self.JOIN_PLAYER_GAME

    @property
    def requires_temporal_shift(self) -> bool:
        """
        Return True if this loader's data needs temporal shifting.

        Team and opponent stats should use PRIOR week's data to prevent leakage.
        Override in subclass if temporal shift is needed.
        """
        return self.join_type in [self.JOIN_TEAM, self.JOIN_OPPONENT_TEAM]

    @property
    def temporal_shift_weeks(self) -> int:
        """
        Number of weeks to shift data back (for leakage prevention).

        Default is 1 week for team/opponent data.
        Override in subclass for different shift amounts.
        """
        return 1 if self.requires_temporal_shift else 0

    def get_full_table_path(self) -> Path:
        """Get the full path to the parquet table."""
        return self.base_data_path / self.table_path

    def _load_partitioned_table(self) -> pd.DataFrame:
        """
        Load partitioned parquet table with season/week filtering.

        Handles three partition patterns:
        1. season=YYYY/week=W/*.parquet (most common - player game stats)
        2. season=YYYY/*.parquet (season-only partitioning - plyr, nfl_week)
        3. table.parquet (non-partitioned - plyr_master, nfl_season)

        Returns:
            DataFrame with loaded data, including season_id/week_id columns
        """
        table_path = self.get_full_table_path()

        if not table_path.exists():
            raise FileNotFoundError(f"Table path does not exist: {table_path}")

        # Handle non-partitioned tables (single .parquet file)
        if table_path.suffix == '.parquet':
            self.logger.info(f"Loading non-partitioned table: {self.table_path}")
            return pd.read_parquet(table_path)

        # Load partitioned tables
        dfs = []

        for season in self.seasons:
            season_path = table_path / f"season={season}"

            if not season_path.exists():
                self.logger.warning(f"Season {season} not found for table {self.table_path}")
                continue

            # Check if this is a season-only partitioned table (no week subdirectories)
            week_dirs = [d for d in season_path.iterdir()
                        if d.is_dir() and d.name.startswith('week=')]

            if not week_dirs:
                # Season-only partitioning - load all parquet files in season directory
                season_df = pd.read_parquet(season_path)
                if 'season_id' not in season_df.columns:
                    season_df['season_id'] = season
                dfs.append(season_df)
                continue

            # Season/week partitioning - iterate through weeks
            for week_dir in week_dirs:
                week_num = int(week_dir.name.split('=')[1])

                # Apply week filter if specified
                if self.weeks is not None and week_num not in self.weeks:
                    continue

                try:
                    week_df = pd.read_parquet(week_dir)

                    # Add partition columns if not present
                    if 'season_id' not in week_df.columns:
                        week_df['season_id'] = season
                    if 'week_id' not in week_df.columns:
                        week_df['week_id'] = week_num

                    dfs.append(week_df)

                except Exception as e:
                    self.logger.warning(
                        f"Could not load {self.table_path} season={season} week={week_num}: {e}"
                    )

        if not dfs:
            raise ValueError(f"No data found for table {self.table_path}")

        result_df = pd.concat(dfs, ignore_index=True)
        self.logger.info(
            f"Loaded {self.table_path}: {len(result_df):,} rows across {len(dfs)} partitions"
        )

        return result_df

    def _validate_output(self, df: pd.DataFrame, stage_name: str = "output") -> Dict:
        """
        Run standard validation checks on loaded data.

        Args:
            df: DataFrame to validate
            stage_name: Name for logging/tracking

        Returns:
            Dictionary of validation results
        """
        self.logger.info(f"Validating {self.__class__.__name__} at stage: {stage_name}")

        if df.empty:
            raise ValueError(f"DataFrame is empty at stage: {stage_name}")

        # Check for required key columns
        missing_keys = set(self.key_columns) - set(df.columns)
        if missing_keys:
            raise ValueError(f"Missing key columns: {missing_keys}")

        # Check for feature columns (warn, don't error)
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            self.logger.warning(f"Missing feature columns: {missing_features}")

        # Check for duplicates in key columns
        duplicates = df.duplicated(subset=self.key_columns).sum()
        if duplicates > 0:
            self.logger.warning(
                f"Found {duplicates} duplicates in key columns {self.key_columns}"
            )

        # Store validation results
        results = {
            'rows': len(df),
            'columns': len(df.columns),
            'nulls_by_column': df.isnull().sum().to_dict(),
            'duplicates_in_keys': duplicates,
            'missing_features': list(missing_features)
        }

        self.validation_results[stage_name] = results
        self.logger.info(f"Validation complete: {len(df):,} rows, {len(df.columns)} columns")

        return results

    def _select_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select only key columns and feature columns for output.

        Args:
            df: Full DataFrame from load

        Returns:
            DataFrame with only relevant columns
        """
        output_cols = self.key_columns + self.feature_columns
        available_cols = [col for col in output_cols if col in df.columns]
        return df[available_cols]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply loader-specific transformations.

        Override in subclass to add custom transformations.
        Default implementation returns data unchanged.

        Args:
            df: Raw loaded data

        Returns:
            Transformed data
        """
        return df

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load and process the data source.

        Must be implemented by each loader subclass.

        Typical implementation:
            1. Call _load_partitioned_table()
            2. Call transform() for any custom processing
            3. Call _select_output_columns()
            4. Call _validate_output()
            5. Return processed DataFrame

        Returns:
            DataFrame with key_columns + feature_columns ready for joining
        """
        pass
