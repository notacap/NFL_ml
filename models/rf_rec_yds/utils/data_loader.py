"""
DataLoader: Unified interface for loading partitioned parquet files.

This module provides a DataLoader class that handles loading of partitioned
parquet files with different partition structures (game-level, season-level,
and static tables) and integrates with PathManager for path resolution.
"""

import logging
import re
from pathlib import Path
from typing import Optional, List, Union, Dict, Tuple
import pandas as pd
import pyarrow.parquet as pq
from utils.path_manager import PathManager


# Configure logging
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading of partitioned parquet files for NFL data.

    The DataLoader works with three types of table structures:
    1. Game-level tables: Partitioned by season/week (e.g., plyr_gm/*, tm_gm/*, gm_info/*)
    2. Season-level tables: Partitioned by season/week (e.g., plyr_szn/*, tm_szn/*)
    3. Reference tables: Partitioned by season only (e.g., players/*)
    4. Static tables: Single files with no partitions (e.g., nfl_team.parquet)

    Example usage:
        from utils.path_manager import PathManager
        from utils.data_loader import DataLoader

        paths = PathManager()
        loader = DataLoader(paths)

        # Load target variable for 2024 season
        target_df = loader.load_target(seasons=[2024], weeks=range(1, 18))

        # Load player info
        players_df = loader.load_table('players', seasons=[2024])

        # Check available data
        partitions = loader.get_available_partitions('player_game_receiving')
    """

    def __init__(self, path_manager: PathManager):
        """
        Initialize the DataLoader.

        Args:
            path_manager: PathManager instance for path resolution.

        Raises:
            ValueError: If path_manager is not a PathManager instance.
        """
        if not isinstance(path_manager, PathManager):
            raise ValueError(
                "path_manager must be a PathManager instance. "
                f"Got {type(path_manager).__name__} instead."
            )

        self.path_manager = path_manager
        self.data_config = path_manager.data_config

        # Validate configuration
        if 'tables' not in self.data_config:
            raise ValueError(
                "data_config.yaml must contain a 'tables' key with table definitions."
            )

        self.tables = self.data_config['tables']
        self.target_config = self.data_config.get('target', {})

        logger.info(
            f"DataLoader initialized with {len(self.tables)} table definitions"
        )

    def _get_table_path(self, table_key: str) -> str:
        """
        Get the table path from configuration.

        Args:
            table_key: Key name from data_config.yaml tables section.

        Returns:
            str: Table path (e.g., 'plyr_gm/plyr_gm_rec').

        Raises:
            KeyError: If table_key doesn't exist in configuration.
        """
        if table_key not in self.tables:
            available_keys = list(self.tables.keys())
            raise KeyError(
                f"Table key '{table_key}' not found in data_config.yaml.\n"
                f"Available tables: {available_keys}"
            )

        return self.tables[table_key]

    def _determine_partition_type(self, table_path: str) -> str:
        """
        Determine the partition type based on table path.

        Args:
            table_path: Table path from config (e.g., 'plyr_gm/plyr_gm_rec').

        Returns:
            str: One of 'game_level', 'season_level', 'reference', or 'static'.
        """
        # Static tables end with .parquet
        if table_path.endswith('.parquet'):
            return 'static'

        # Check directory prefix to determine type
        if table_path.startswith(('plyr_gm/', 'tm_gm/', 'gm_info/')):
            return 'game_level'
        elif table_path.startswith(('plyr_szn/', 'tm_szn/')):
            return 'season_level'
        elif table_path.startswith('players/'):
            return 'reference'
        elif table_path.startswith('static/'):
            # static/ prefix but no .parquet means season-only partitioning
            return 'reference'
        else:
            # Default to static for unknown patterns
            logger.warning(
                f"Unknown table path pattern: {table_path}. Treating as static."
            )
            return 'static'

    def _parse_partition_dirs(
        self,
        partition_dirs: List[Path]
    ) -> List[Tuple[int, Optional[int]]]:
        """
        Parse partition directory names to extract season/week values.

        Args:
            partition_dirs: List of partition directory paths.

        Returns:
            List of tuples (season, week). Week is None for season-only partitions.
        """
        partitions = []

        for dir_path in partition_dirs:
            # Extract season and week from directory names
            season_match = re.search(r'season=(\d+)', str(dir_path))
            week_match = re.search(r'week=(\d+)', str(dir_path))

            if season_match:
                season = int(season_match.group(1))
                week = int(week_match.group(1)) if week_match else None
                partitions.append((season, week))

        return sorted(set(partitions))

    def _get_partition_paths(
        self,
        table_path: str,
        partition_type: str,
        seasons: Optional[List[int]] = None,
        weeks: Optional[List[int]] = None
    ) -> List[Path]:
        """
        Get list of partition paths to read based on filters.

        Args:
            table_path: Table path from config.
            partition_type: Type of partition structure.
            seasons: List of seasons to include (None = all).
            weeks: List of weeks to include (None = all).

        Returns:
            List of Path objects pointing to parquet files or directories.

        Raises:
            FileNotFoundError: If table directory doesn't exist.
        """
        data_source = self.path_manager.get('data', 'source')
        base_path = data_source / table_path

        # Handle static tables
        if partition_type == 'static':
            if not base_path.exists():
                raise FileNotFoundError(
                    f"Static table file not found: {base_path}"
                )
            return [base_path]

        # For partitioned tables, verify base directory exists
        if not base_path.exists():
            raise FileNotFoundError(
                f"Table directory not found: {base_path}"
            )

        # Get available partitions
        available_partitions = self._scan_partitions(base_path, partition_type)

        if not available_partitions:
            logger.warning(f"No partitions found for table: {table_path}")
            return []

        # Filter by seasons
        if seasons is not None:
            available_partitions = [
                p for p in available_partitions
                if p[0] in seasons
            ]

        # Filter by weeks (only for game/season level)
        if weeks is not None and partition_type in ['game_level', 'season_level']:
            available_partitions = [
                p for p in available_partitions
                if p[1] is None or p[1] in weeks
            ]

        # Build paths
        paths = []
        for season, week in available_partitions:
            if partition_type == 'reference':
                # Season-only partitions
                partition_path = base_path / f"season={season}"
            else:
                # Season/week partitions
                partition_path = base_path / f"season={season}" / f"week={week}"

            if partition_path.exists():
                paths.append(partition_path)

        return paths

    def _scan_partitions(
        self,
        base_path: Path,
        partition_type: str
    ) -> List[Tuple[int, Optional[int]]]:
        """
        Scan directory to find available partitions.

        Args:
            base_path: Base directory path for the table.
            partition_type: Type of partition structure.

        Returns:
            List of tuples (season, week). Week is None for season-only partitions.
        """
        partitions = []

        if not base_path.exists():
            return partitions

        # Find season directories
        season_dirs = [
            d for d in base_path.iterdir()
            if d.is_dir() and d.name.startswith('season=')
        ]

        for season_dir in season_dirs:
            season_match = re.search(r'season=(\d+)', season_dir.name)
            if not season_match:
                continue

            season = int(season_match.group(1))

            if partition_type == 'reference':
                # Season-only partitions
                partitions.append((season, None))
            else:
                # Look for week subdirectories
                week_dirs = [
                    d for d in season_dir.iterdir()
                    if d.is_dir() and d.name.startswith('week=')
                ]

                for week_dir in week_dirs:
                    week_match = re.search(r'week=(\d+)', week_dir.name)
                    if week_match:
                        week = int(week_match.group(1))
                        partitions.append((season, week))

        return sorted(partitions)

    def _read_parquet_files(self, partition_paths: List[Path]) -> pd.DataFrame:
        """
        Read parquet files from partition paths and combine into single DataFrame.

        Args:
            partition_paths: List of paths to partition directories or files.

        Returns:
            pd.DataFrame: Combined data from all partitions.

        Raises:
            ValueError: If no data could be read.
        """
        dfs = []

        for path in partition_paths:
            try:
                if path.is_file():
                    # Single parquet file
                    df = pd.read_parquet(path)
                    logger.debug(f"Loaded {len(df)} rows from {path.name}")
                    dfs.append(df)
                elif path.is_dir():
                    # Directory with parquet files
                    parquet_files = list(path.glob('*.parquet'))

                    if not parquet_files:
                        logger.warning(f"No parquet files found in {path}")
                        continue

                    # Read all parquet files in directory
                    for pq_file in parquet_files:
                        df = pd.read_parquet(pq_file)
                        logger.debug(f"Loaded {len(df)} rows from {pq_file.name}")
                        dfs.append(df)

                    # Extract partition values from directory name
                    season_match = re.search(r'season=(\d+)', str(path))
                    week_match = re.search(r'week=(\d+)', str(path))

                    # Add partition columns if they don't exist
                    if dfs and season_match:
                        season = int(season_match.group(1))
                        for df in dfs[-len(parquet_files):]:
                            if 'season' not in df.columns:
                                df['season'] = season

                        if week_match:
                            week = int(week_match.group(1))
                            for df in dfs[-len(parquet_files):]:
                                if 'week' not in df.columns:
                                    df['week'] = week

            except Exception as e:
                logger.error(f"Error reading parquet from {path}: {e}")
                continue

        if not dfs:
            raise ValueError("No data could be read from the specified partitions")

        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)

        logger.info(f"Loaded total of {len(combined_df)} rows from {len(dfs)} partition(s)")

        return combined_df

    def load_table(
        self,
        table_key: str,
        seasons: Optional[Union[List[int], int]] = None,
        weeks: Optional[Union[List[int], int, range]] = None
    ) -> pd.DataFrame:
        """
        Load any table by key from data_config.yaml.

        This is the main entry point for loading data. It automatically detects
        the partition type and handles filtering appropriately.

        Args:
            table_key: Key name from data_config.yaml tables section
                      (e.g., 'player_game_receiving', 'players', 'teams').
            seasons: Season(s) to load. Can be:
                    - None: Load all available seasons (default)
                    - int: Single season (e.g., 2024)
                    - List[int]: Multiple seasons (e.g., [2023, 2024])
            weeks: Week(s) to load (only applies to game/season level tables). Can be:
                  - None: Load all available weeks (default)
                  - int: Single week (e.g., 5)
                  - List[int]: Multiple weeks (e.g., [1, 2, 3])
                  - range: Week range (e.g., range(1, 18))

        Returns:
            pd.DataFrame: Loaded data with partition columns (season, week) added.

        Raises:
            KeyError: If table_key doesn't exist in configuration.
            FileNotFoundError: If table directory/file doesn't exist.
            ValueError: If no data could be loaded.

        Examples:
            # Load all player receiving stats for 2024
            df = loader.load_table('player_game_receiving', seasons=2024)

            # Load multiple seasons and weeks
            df = loader.load_table(
                'player_game_receiving',
                seasons=[2023, 2024],
                weeks=range(1, 10)
            )

            # Load reference data (season-only partitions)
            df = loader.load_table('players', seasons=[2024])

            # Load static data (no partitions)
            df = loader.load_table('teams')
        """
        # Convert single values to lists
        if isinstance(seasons, int):
            seasons = [seasons]
        if isinstance(weeks, int):
            weeks = [weeks]
        elif isinstance(weeks, range):
            weeks = list(weeks)

        # Get table path and determine partition type
        table_path = self._get_table_path(table_key)
        partition_type = self._determine_partition_type(table_path)

        logger.info(
            f"Loading table '{table_key}' (type: {partition_type}, "
            f"seasons: {seasons}, weeks: {weeks})"
        )

        # Get partition paths
        partition_paths = self._get_partition_paths(
            table_path,
            partition_type,
            seasons,
            weeks
        )

        if not partition_paths:
            raise ValueError(
                f"No partitions found for table '{table_key}' with "
                f"seasons={seasons}, weeks={weeks}"
            )

        # Read and combine parquet files
        df = self._read_parquet_files(partition_paths)

        logger.info(
            f"Successfully loaded table '{table_key}': {len(df)} rows, "
            f"{len(df.columns)} columns"
        )

        return df

    def load_game_level(
        self,
        table_key: str,
        seasons: Union[List[int], int],
        weeks: Optional[Union[List[int], int, range]] = None
    ) -> pd.DataFrame:
        """
        Load game-level data for specific season/week combinations.

        Game-level tables are partitioned by season/week and contain data
        at the game granularity (e.g., player game stats, team game stats).

        Args:
            table_key: Key for a game-level table (e.g., 'player_game_receiving').
            seasons: Season(s) to load (required).
            weeks: Week(s) to load (None = all weeks).

        Returns:
            pd.DataFrame: Game-level data.

        Raises:
            ValueError: If table is not game-level type.

        Examples:
            # Load all weeks for 2024 season
            df = loader.load_game_level('player_game_receiving', seasons=2024)

            # Load specific weeks
            df = loader.load_game_level(
                'player_game_receiving',
                seasons=2024,
                weeks=[1, 2, 3]
            )
        """
        table_path = self._get_table_path(table_key)
        partition_type = self._determine_partition_type(table_path)

        if partition_type != 'game_level':
            raise ValueError(
                f"Table '{table_key}' is not a game-level table (type: {partition_type}). "
                f"Use load_table() for generic loading or the appropriate type-specific method."
            )

        return self.load_table(table_key, seasons=seasons, weeks=weeks)

    def load_season_level(
        self,
        table_key: str,
        seasons: Union[List[int], int],
        weeks: Optional[Union[List[int], int, range]] = None
    ) -> pd.DataFrame:
        """
        Load season cumulative data.

        Season-level tables contain cumulative statistics through each week
        of a season (e.g., player season totals, team season stats).

        Args:
            table_key: Key for a season-level table (e.g., 'player_season_receiving').
            seasons: Season(s) to load (required).
            weeks: Week(s) to load (None = all weeks).

        Returns:
            pd.DataFrame: Season cumulative data.

        Raises:
            ValueError: If table is not season-level type.

        Examples:
            # Load cumulative stats through all weeks
            df = loader.load_season_level('player_season_receiving', seasons=2024)

            # Load cumulative stats through specific week
            df = loader.load_season_level(
                'player_season_receiving',
                seasons=2024,
                weeks=10
            )
        """
        table_path = self._get_table_path(table_key)
        partition_type = self._determine_partition_type(table_path)

        if partition_type != 'season_level':
            raise ValueError(
                f"Table '{table_key}' is not a season-level table (type: {partition_type}). "
                f"Use load_table() for generic loading or the appropriate type-specific method."
            )

        return self.load_table(table_key, seasons=seasons, weeks=weeks)

    def load_static(self, table_key: str) -> pd.DataFrame:
        """
        Load non-partitioned static tables.

        Static tables have no partitions and contain reference data that
        doesn't change by season/week (e.g., teams, game quarters).

        Args:
            table_key: Key for a static table (e.g., 'teams', 'game_quarters').

        Returns:
            pd.DataFrame: Static table data.

        Raises:
            ValueError: If table is not static type.

        Examples:
            # Load team reference data
            teams_df = loader.load_static('teams')

            # Load game quarters reference
            quarters_df = loader.load_static('game_quarters')
        """
        table_path = self._get_table_path(table_key)
        partition_type = self._determine_partition_type(table_path)

        if partition_type != 'static':
            raise ValueError(
                f"Table '{table_key}' is not a static table (type: {partition_type}). "
                f"Use load_table() for generic loading or the appropriate type-specific method."
            )

        return self.load_table(table_key)

    def load_target(
        self,
        seasons: Union[List[int], int],
        weeks: Optional[Union[List[int], int, range]] = None
    ) -> pd.DataFrame:
        """
        Shortcut to load target variable.

        Loads the target variable defined in data_config.yaml (plyr_gm_rec_yds
        from plyr_gm_rec table by default).

        Args:
            seasons: Season(s) to load (required).
            weeks: Week(s) to load (None = all weeks).

        Returns:
            pd.DataFrame: Target variable data.

        Raises:
            ValueError: If target is not configured in data_config.yaml.

        Examples:
            # Load target for full 2024 season
            target_df = loader.load_target(seasons=2024)

            # Load target for specific weeks
            target_df = loader.load_target(
                seasons=2024,
                weeks=range(1, 18)
            )

            # Load multiple seasons
            target_df = loader.load_target(seasons=[2023, 2024])
        """
        if not self.target_config:
            raise ValueError(
                "No target variable configured in data_config.yaml. "
                "Please define 'target' section with 'table' key."
            )

        target_table = self.target_config.get('table')
        target_name = self.target_config.get('name')

        if not target_table:
            raise ValueError(
                "Target table not specified in data_config.yaml. "
                "Please define 'target.table' key."
            )

        # Find table key for target table
        table_key = None
        for key, path in self.tables.items():
            if target_table in path:
                table_key = key
                break

        if not table_key:
            raise ValueError(
                f"Target table '{target_table}' not found in tables configuration. "
                f"Available tables: {list(self.tables.keys())}"
            )

        logger.info(
            f"Loading target variable '{target_name}' from table '{table_key}'"
        )

        return self.load_table(table_key, seasons=seasons, weeks=weeks)

    def get_available_partitions(
        self,
        table_key: str
    ) -> List[Dict[str, Optional[int]]]:
        """
        List available season/week combinations for a table.

        Args:
            table_key: Key name from data_config.yaml tables section.

        Returns:
            List of dictionaries with 'season' and 'week' keys.
            For reference tables, 'week' will be None.
            For static tables, returns empty list.

        Raises:
            KeyError: If table_key doesn't exist in configuration.

        Examples:
            # Check available partitions
            partitions = loader.get_available_partitions('player_game_receiving')
            # Returns: [
            #     {'season': 2023, 'week': 1},
            #     {'season': 2023, 'week': 2},
            #     ...
            #     {'season': 2024, 'week': 1},
            #     ...
            # ]

            # Check reference table partitions
            partitions = loader.get_available_partitions('players')
            # Returns: [
            #     {'season': 2023, 'week': None},
            #     {'season': 2024, 'week': None}
            # ]

            # Check static table
            partitions = loader.get_available_partitions('teams')
            # Returns: []
        """
        table_path = self._get_table_path(table_key)
        partition_type = self._determine_partition_type(table_path)

        if partition_type == 'static':
            logger.info(f"Table '{table_key}' is static (no partitions)")
            return []

        data_source = self.path_manager.get('data', 'source')
        base_path = data_source / table_path

        if not base_path.exists():
            logger.warning(f"Table directory not found: {base_path}")
            return []

        # Scan for partitions
        partitions = self._scan_partitions(base_path, partition_type)

        # Convert to list of dictionaries
        partition_list = [
            {'season': season, 'week': week}
            for season, week in partitions
        ]

        logger.info(
            f"Found {len(partition_list)} partition(s) for table '{table_key}'"
        )

        return partition_list

    def validate_schema(
        self,
        df: pd.DataFrame,
        expected_columns: List[str],
        table_name: str = "table"
    ) -> bool:
        """
        Validate that DataFrame has expected columns.

        Args:
            df: DataFrame to validate.
            expected_columns: List of expected column names.
            table_name: Name of table for error messages.

        Returns:
            bool: True if valid, False otherwise.

        Raises:
            Warning: If columns are missing or extra columns are present.
        """
        actual_columns = set(df.columns)
        expected_columns_set = set(expected_columns)

        missing_columns = expected_columns_set - actual_columns
        extra_columns = actual_columns - expected_columns_set

        if missing_columns:
            logger.warning(
                f"Schema validation warning for {table_name}: "
                f"Missing columns: {sorted(missing_columns)}"
            )

        if extra_columns:
            logger.info(
                f"Schema validation info for {table_name}: "
                f"Extra columns found: {sorted(extra_columns)}"
            )

        is_valid = len(missing_columns) == 0

        if is_valid:
            logger.debug(f"Schema validation passed for {table_name}")

        return is_valid

    def __repr__(self) -> str:
        """String representation of DataLoader."""
        return (
            f"DataLoader(tables={len(self.tables)}, "
            f"target='{self.target_config.get('name', 'not configured')}')"
        )
