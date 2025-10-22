"""
PathManager: Centralized path management for the NFL ML project.

This module provides a unified interface for managing all project paths,
including handling of both absolute and relative paths, environment variable
substitution, and automatic directory creation.
"""

import os
import re
from pathlib import Path
from typing import Union, Optional, Tuple
import yaml


class PathManager:
    """
    Manages all project paths with support for relative/absolute paths,
    environment variable substitution, and partitioned parquet file paths.

    The PathManager automatically:
    - Detects the project root by locating the configs/ directory
    - Loads both paths.yaml and data_config.yaml
    - Resolves relative paths to absolute paths based on project root
    - Handles ${NFL_DATA_ROOT} environment variable substitution
    - Falls back to paths.yaml data.source if NFL_DATA_ROOT is not set
    - Returns pathlib.Path objects for all paths

    Example usage:
        paths = PathManager()
        parquet_source = paths.get('data', 'source')
        processed_dir = paths.ensure_exists('data', 'processed')
        rec_data = paths.get_parquet_path('plyr_gm/plyr_gm_rec', season=2024, week=5)
    """

    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        """
        Initialize the PathManager.

        Args:
            project_root: Optional explicit project root path. If not provided,
                         will auto-detect by finding the configs/ directory.

        Raises:
            FileNotFoundError: If configs directory or required config files not found.
        """
        # Detect or set project root
        if project_root is None:
            self.project_root = self._detect_project_root()
        else:
            self.project_root = Path(project_root).resolve()

        # Load configuration files
        self.configs_dir = self.project_root / 'configs'
        self._validate_configs_dir()

        self.paths_config = self._load_config('paths.yaml')
        self.data_config = self._load_config('data_config.yaml')

        # Resolve data source with environment variable substitution
        self._resolve_data_source()

    def _detect_project_root(self) -> Path:
        """
        Auto-detect the project root by finding the configs/ directory.

        Searches upward from the current file location until configs/ is found.

        Returns:
            Path: Absolute path to the project root.

        Raises:
            FileNotFoundError: If configs directory cannot be found.
        """
        # Start from the directory containing this file
        current_path = Path(__file__).resolve().parent

        # Search upward for configs directory
        max_levels = 5  # Prevent infinite loop
        for _ in range(max_levels):
            configs_path = current_path / 'configs'
            if configs_path.exists() and configs_path.is_dir():
                return current_path

            # Move up one level
            parent = current_path.parent
            if parent == current_path:  # Reached filesystem root
                break
            current_path = parent

        raise FileNotFoundError(
            "Could not detect project root. Unable to find 'configs' directory. "
            "Please ensure you're running from within the project or specify project_root explicitly."
        )

    def _validate_configs_dir(self) -> None:
        """
        Validate that the configs directory exists.

        Raises:
            FileNotFoundError: If configs directory doesn't exist.
        """
        if not self.configs_dir.exists():
            raise FileNotFoundError(
                f"Configs directory not found at: {self.configs_dir}"
            )

    def _load_config(self, config_filename: str) -> dict:
        """
        Load a YAML configuration file.

        Args:
            config_filename: Name of the config file (e.g., 'paths.yaml').

        Returns:
            dict: Parsed configuration data.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            yaml.YAMLError: If the config file is invalid YAML.
        """
        config_path = self.configs_dir / config_filename

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Expected location: {self.configs_dir / config_filename}"
            )

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config or {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Error parsing {config_filename}: {e}"
            )

    def _resolve_data_source(self) -> None:
        """
        Resolve the data source path with environment variable substitution.

        Handles ${NFL_DATA_ROOT} substitution in data_config.yaml and falls
        back to paths.yaml if the environment variable is not set.
        """
        data_source = self.data_config.get('data', {}).get('source', '')

        # Check for environment variable pattern
        env_var_pattern = r'\$\{([^}]+)\}'
        match = re.search(env_var_pattern, data_source)

        if match:
            env_var_name = match.group(1)
            env_var_value = os.getenv(env_var_name)

            if env_var_value:
                # Substitute the environment variable
                resolved_source = re.sub(env_var_pattern, env_var_value, data_source)
                self.data_config['data']['source'] = resolved_source
            else:
                # Fall back to paths.yaml
                fallback_source = self.paths_config.get('data', {}).get('source', '')
                if fallback_source:
                    self.data_config['data']['source'] = fallback_source
                else:
                    raise ValueError(
                        f"Environment variable ${{{env_var_name}}} is not set, "
                        f"and no fallback path found in paths.yaml data.source. "
                        f"Please either:\n"
                        f"  1. Set the {env_var_name} environment variable, or\n"
                        f"  2. Define data.source in configs/paths.yaml"
                    )

    def _resolve_path(self, path_str: str) -> Path:
        """
        Resolve a path string to an absolute Path object.

        Args:
            path_str: Path string that may be absolute or relative.

        Returns:
            Path: Absolute path object.
        """
        path = Path(path_str)

        # If already absolute, return as-is
        if path.is_absolute():
            return path

        # Otherwise, resolve relative to project root
        return (self.project_root / path).resolve()

    def _get_from_dict(self, config_dict: dict, keys: Tuple[str, ...]) -> str:
        """
        Retrieve a value from a nested dictionary using a sequence of keys.

        Args:
            config_dict: Dictionary to search.
            keys: Sequence of keys for nested lookup.

        Returns:
            str: The retrieved value.

        Raises:
            KeyError: If any key in the path doesn't exist.
        """
        current = config_dict

        for i, key in enumerate(keys):
            if not isinstance(current, dict):
                raise KeyError(
                    f"Invalid path: expected dict at key '{keys[i-1]}', "
                    f"but got {type(current).__name__}"
                )

            if key not in current:
                raise KeyError(
                    f"Path not found in configuration: {' -> '.join(keys)}\n"
                    f"Key '{key}' does not exist at this level.\n"
                    f"Available keys: {list(current.keys())}"
                )

            current = current[key]

        return current

    def get(self, *keys: str, config: str = 'auto') -> Path:
        """
        Retrieve a path from the configuration.

        Supports both dot notation and multiple arguments for nested keys.

        Args:
            *keys: Path keys (e.g., 'data', 'source' or 'data.source').
            config: Which config to use: 'paths', 'data', or 'auto' (default).
                   'auto' tries data_config first, then falls back to paths_config.

        Returns:
            Path: Resolved absolute path object.

        Raises:
            KeyError: If the path key doesn't exist in configuration.
            ValueError: If invalid config type specified.

        Examples:
            paths.get('data', 'source')  # Multiple args
            paths.get('data.source')     # Dot notation (NOT SUPPORTED - use multiple args)
        """
        if not keys:
            raise ValueError("At least one key must be provided")

        # Convert keys tuple to list for processing
        key_list = list(keys)

        # Determine which config(s) to search
        if config == 'auto':
            # Try data_config first, fall back to paths_config
            try:
                path_str = self._get_from_dict(self.data_config, tuple(key_list))
            except KeyError:
                path_str = self._get_from_dict(self.paths_config, tuple(key_list))
        elif config == 'data':
            path_str = self._get_from_dict(self.data_config, tuple(key_list))
        elif config == 'paths':
            path_str = self._get_from_dict(self.paths_config, tuple(key_list))
        else:
            raise ValueError(
                f"Invalid config type: {config}. Must be 'auto', 'data', or 'paths'."
            )

        # Resolve to absolute path
        return self._resolve_path(path_str)

    def ensure_exists(self, *keys: str, config: str = 'auto', parents: bool = True) -> Path:
        """
        Retrieve a path and ensure the directory exists.

        Args:
            *keys: Path keys (e.g., 'data', 'processed').
            config: Which config to use: 'paths', 'data', or 'auto' (default).
            parents: If True, create parent directories as needed (default: True).

        Returns:
            Path: Resolved absolute path object with directory created.

        Raises:
            KeyError: If the path key doesn't exist in configuration.

        Examples:
            paths.ensure_exists('data', 'processed')
            paths.ensure_exists('logs')
        """
        path = self.get(*keys, config=config)
        path.mkdir(parents=parents, exist_ok=True)
        return path

    def get_parquet_path(
        self,
        table_name: str,
        season: Optional[int] = None,
        week: Optional[int] = None,
        use_wildcard: bool = False
    ) -> Path:
        """
        Build a path to partitioned parquet files.

        Parquet files are partitioned by season/week in the format:
        {data_source}/{table_name}/season={YYYY}/week={W}/*.parquet

        Args:
            table_name: Name/path of the table (e.g., 'plyr_gm/plyr_gm_rec' or 'nfl_team.parquet').
            season: Optional season year (e.g., 2024).
            week: Optional week number (e.g., 5).
            use_wildcard: If True, append '*.parquet' to the path (default: False).

        Returns:
            Path: Path to the parquet file(s).

        Examples:
            # Get base table path
            paths.get_parquet_path('plyr_gm/plyr_gm_rec')
            # => C:/.../parquet_files/clean/plyr_gm/plyr_gm_rec

            # Get season partition
            paths.get_parquet_path('plyr_gm/plyr_gm_rec', season=2024)
            # => C:/.../parquet_files/clean/plyr_gm/plyr_gm_rec/season=2024

            # Get week partition with wildcard
            paths.get_parquet_path('plyr_gm/plyr_gm_rec', season=2024, week=5, use_wildcard=True)
            # => C:/.../parquet_files/clean/plyr_gm/plyr_gm_rec/season=2024/week=5/*.parquet

            # Static table (no partitions)
            paths.get_parquet_path('nfl_team.parquet')
            # => C:/.../parquet_files/clean/nfl_team.parquet
        """
        # Get the data source path
        data_source = self.get('data', 'source')

        # Start with base path
        path = data_source / table_name

        # Add season partition if specified
        if season is not None:
            path = path / f"season={season}"

        # Add week partition if specified (requires season)
        if week is not None:
            if season is None:
                raise ValueError(
                    "Cannot specify week without season. "
                    "Please provide both season and week parameters."
                )
            path = path / f"week={week}"

        # Add wildcard if requested
        if use_wildcard:
            path = path / "*.parquet"

        return path

    def list_partitions(
        self,
        table_name: str,
        season: Optional[int] = None
    ) -> list:
        """
        List available partitions for a table.

        Args:
            table_name: Name/path of the table.
            season: Optional season to filter partitions.

        Returns:
            list: List of partition paths that exist.

        Examples:
            # List all seasons for a table
            paths.list_partitions('plyr_gm/plyr_gm_rec')

            # List all weeks for a season
            paths.list_partitions('plyr_gm/plyr_gm_rec', season=2024)
        """
        base_path = self.get_parquet_path(table_name, season=season)

        if not base_path.exists():
            return []

        # Return sorted list of directories that match partition pattern
        partitions = [
            p for p in base_path.iterdir()
            if p.is_dir() and ('season=' in p.name or 'week=' in p.name)
        ]

        return sorted(partitions)

    def __repr__(self) -> str:
        """String representation of PathManager."""
        return f"PathManager(project_root='{self.project_root}')"
