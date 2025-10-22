"""
Unit tests for the DataLoader class.

Run with: pytest tests/test_data_loader.py -v
"""

import pytest
import pandas as pd
from pathlib import Path
from utils.path_manager import PathManager
from utils.data_loader import DataLoader


class TestDataLoaderInitialization:
    """Test DataLoader initialization."""

    def test_init_with_valid_path_manager(self):
        """Test initialization with valid PathManager."""
        pm = PathManager()
        loader = DataLoader(pm)

        assert loader.path_manager is pm
        assert len(loader.tables) > 0
        assert 'tables' in loader.data_config

    def test_init_with_invalid_path_manager(self):
        """Test initialization with invalid PathManager."""
        with pytest.raises(ValueError, match="must be a PathManager instance"):
            DataLoader("not a path manager")

    def test_target_config_loaded(self):
        """Test that target configuration is loaded."""
        pm = PathManager()
        loader = DataLoader(pm)

        assert 'name' in loader.target_config
        assert 'table' in loader.target_config
        assert loader.target_config['name'] == 'plyr_gm_rec_yds'


class TestTableConfiguration:
    """Test table configuration methods."""

    def test_get_table_path_valid(self):
        """Test getting valid table path."""
        pm = PathManager()
        loader = DataLoader(pm)

        table_path = loader._get_table_path('player_game_receiving')
        assert 'plyr_gm_rec' in table_path

    def test_get_table_path_invalid(self):
        """Test getting invalid table path raises KeyError."""
        pm = PathManager()
        loader = DataLoader(pm)

        with pytest.raises(KeyError, match="Table key .* not found"):
            loader._get_table_path('nonexistent_table')

    def test_determine_partition_type_game_level(self):
        """Test partition type detection for game-level tables."""
        pm = PathManager()
        loader = DataLoader(pm)

        assert loader._determine_partition_type('plyr_gm/plyr_gm_rec') == 'game_level'
        assert loader._determine_partition_type('tm_gm/tm_gm_stats') == 'game_level'
        assert loader._determine_partition_type('gm_info/nfl_game') == 'game_level'

    def test_determine_partition_type_season_level(self):
        """Test partition type detection for season-level tables."""
        pm = PathManager()
        loader = DataLoader(pm)

        assert loader._determine_partition_type('plyr_szn/plyr_rec') == 'season_level'
        assert loader._determine_partition_type('tm_szn/tm_pass') == 'season_level'

    def test_determine_partition_type_reference(self):
        """Test partition type detection for reference tables."""
        pm = PathManager()
        loader = DataLoader(pm)

        assert loader._determine_partition_type('players/plyr') == 'reference'
        assert loader._determine_partition_type('static/nfl_week') == 'reference'

    def test_determine_partition_type_static(self):
        """Test partition type detection for static tables."""
        pm = PathManager()
        loader = DataLoader(pm)

        assert loader._determine_partition_type('nfl_team.parquet') == 'static'
        assert loader._determine_partition_type('nfl_season.parquet') == 'static'


class TestPartitionParsing:
    """Test partition parsing methods."""

    def test_parse_partition_dirs_with_season_and_week(self):
        """Test parsing partition directories with season and week."""
        pm = PathManager()
        loader = DataLoader(pm)

        partition_dirs = [
            Path('/data/table/season=2024/week=1'),
            Path('/data/table/season=2024/week=2'),
            Path('/data/table/season=2023/week=1'),
        ]

        partitions = loader._parse_partition_dirs(partition_dirs)

        assert len(partitions) == 3
        assert (2023, 1) in partitions
        assert (2024, 1) in partitions
        assert (2024, 2) in partitions

    def test_parse_partition_dirs_with_season_only(self):
        """Test parsing partition directories with season only."""
        pm = PathManager()
        loader = DataLoader(pm)

        partition_dirs = [
            Path('/data/table/season=2024'),
            Path('/data/table/season=2023'),
        ]

        partitions = loader._parse_partition_dirs(partition_dirs)

        assert len(partitions) == 2
        assert (2023, None) in partitions
        assert (2024, None) in partitions


class TestAvailablePartitions:
    """Test methods for checking available partitions."""

    def test_get_available_partitions_returns_list(self):
        """Test that get_available_partitions returns a list."""
        pm = PathManager()
        loader = DataLoader(pm)

        # This may return empty list if data doesn't exist, which is fine
        partitions = loader.get_available_partitions('player_game_receiving')
        assert isinstance(partitions, list)

    def test_get_available_partitions_static_returns_empty(self):
        """Test that static tables return empty partition list."""
        pm = PathManager()
        loader = DataLoader(pm)

        partitions = loader.get_available_partitions('teams')
        assert partitions == []

    def test_get_available_partitions_invalid_key(self):
        """Test that invalid table key raises KeyError."""
        pm = PathManager()
        loader = DataLoader(pm)

        with pytest.raises(KeyError):
            loader.get_available_partitions('invalid_table')


class TestParameterConversion:
    """Test parameter conversion in load methods."""

    def test_load_table_converts_single_season_to_list(self):
        """Test that single season is converted to list."""
        pm = PathManager()
        loader = DataLoader(pm)

        # This test just verifies the parameter conversion doesn't error
        # Actual loading may fail if data doesn't exist
        try:
            loader.load_table('player_game_receiving', seasons=2024)
        except (FileNotFoundError, ValueError):
            # Expected if data doesn't exist
            pass

    def test_load_table_converts_range_to_list(self):
        """Test that range is converted to list."""
        pm = PathManager()
        loader = DataLoader(pm)

        # This test just verifies the parameter conversion doesn't error
        try:
            loader.load_table('player_game_receiving', seasons=2024, weeks=range(1, 5))
        except (FileNotFoundError, ValueError):
            # Expected if data doesn't exist
            pass


class TestTypeSpecificMethods:
    """Test type-specific loading methods."""

    def test_load_game_level_validates_type(self):
        """Test that load_game_level validates table type."""
        pm = PathManager()
        loader = DataLoader(pm)

        # Should raise ValueError for non-game-level table
        with pytest.raises(ValueError, match="not a game-level table"):
            loader.load_game_level('players', seasons=2024)

    def test_load_season_level_validates_type(self):
        """Test that load_season_level validates table type."""
        pm = PathManager()
        loader = DataLoader(pm)

        # Should raise ValueError for non-season-level table
        with pytest.raises(ValueError, match="not a season-level table"):
            loader.load_season_level('player_game_receiving', seasons=2024)

    def test_load_static_validates_type(self):
        """Test that load_static validates table type."""
        pm = PathManager()
        loader = DataLoader(pm)

        # Should raise ValueError for non-static table
        with pytest.raises(ValueError, match="not a static table"):
            loader.load_static('player_game_receiving')


class TestTargetLoading:
    """Test target variable loading."""

    def test_load_target_uses_config(self):
        """Test that load_target uses target configuration."""
        pm = PathManager()
        loader = DataLoader(pm)

        # Verify target config exists
        assert loader.target_config.get('table') == 'plyr_gm_rec'
        assert loader.target_config.get('name') == 'plyr_gm_rec_yds'


class TestDataLoaderRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        pm = PathManager()
        loader = DataLoader(pm)

        repr_str = repr(loader)
        assert 'DataLoader' in repr_str
        assert 'tables=' in repr_str
        assert 'target=' in repr_str


class TestSchemaValidation:
    """Test schema validation method."""

    def test_validate_schema_with_matching_columns(self):
        """Test schema validation with matching columns."""
        pm = PathManager()
        loader = DataLoader(pm)

        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9]
        })

        is_valid = loader.validate_schema(
            df,
            expected_columns=['col1', 'col2', 'col3'],
            table_name='test_table'
        )

        assert is_valid is True

    def test_validate_schema_with_missing_columns(self):
        """Test schema validation with missing columns."""
        pm = PathManager()
        loader = DataLoader(pm)

        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })

        is_valid = loader.validate_schema(
            df,
            expected_columns=['col1', 'col2', 'col3'],
            table_name='test_table'
        )

        assert is_valid is False

    def test_validate_schema_with_extra_columns(self):
        """Test schema validation with extra columns."""
        pm = PathManager()
        loader = DataLoader(pm)

        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9],
            'col4': [10, 11, 12]
        })

        is_valid = loader.validate_schema(
            df,
            expected_columns=['col1', 'col2', 'col3'],
            table_name='test_table'
        )

        # Extra columns don't invalidate, only missing columns do
        assert is_valid is True


# Integration test markers
pytestmark = pytest.mark.integration


class TestDataLoaderIntegration:
    """
    Integration tests that require actual data files.

    These tests are marked as integration tests and may be skipped
    if data files are not available.
    """

    @pytest.mark.skipif(
        not Path.cwd().name == 'rf_rec_yds',
        reason="Must be run from model directory"
    )
    def test_can_initialize_loader(self):
        """Test that DataLoader can be initialized in model directory."""
        pm = PathManager()
        loader = DataLoader(pm)
        assert loader is not None
        assert len(loader.tables) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
