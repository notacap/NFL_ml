"""
pytest configuration and fixtures for NFL receiving statistics tests.

This module provides shared fixtures and configuration for all test modules.
"""

import os
import sys
import pytest
import pandas as pd
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for importing receiving.py and common_utils
sys.path.insert(0, str(Path(__file__).parent.parent / 'build_cumulative_stats'))

# Import modules under test
import receiving
import common_utils

# Configure pytest markers
def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "critical: Critical tests that must pass for production")
    config.addinivalue_line("markers", "ground_truth: Ground truth validation tests")
    config.addinivalue_line("markers", "calculation: Calculation logic validation tests")
    config.addinivalue_line("markers", "integrity: Data integrity tests")
    config.addinivalue_line("markers", "edge_case: Edge case tests")
    config.addinivalue_line("markers", "performance: Performance and scale tests")
    config.addinivalue_line("markers", "level1: Quick validation (5 seconds)")
    config.addinivalue_line("markers", "level2: Sample validation (30 seconds)")
    config.addinivalue_line("markers", "level3: Full validation (5-10 minutes)")
    config.addinivalue_line("markers", "quick: Quick tests")
    config.addinivalue_line("markers", "sample: Sample data tests")
    config.addinivalue_line("markers", "full: Full dataset tests")
    config.addinivalue_line("markers", "slow: Slow running tests")


# ============================================================================
# Path Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def data_root():
    """Root directory for parquet data files."""
    return r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw"


@pytest.fixture(scope="session")
def output_root():
    """Root directory for output files."""
    return r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw"


@pytest.fixture(scope="session")
def test_results_dir():
    """Directory for test results and reports."""
    results_dir = Path(__file__).parent / "test_results"
    results_dir.mkdir(exist_ok=True)
    return str(results_dir)


# ============================================================================
# Data Loading Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def week_mapping_2024(data_root):
    """Load week mapping for 2024 season."""
    try:
        return common_utils.load_week_mapping(data_root, 2024)
    except Exception as e:
        pytest.skip(f"Could not load week mapping for 2024: {e}")


@pytest.fixture(scope="session")
def week_mapping_2023(data_root):
    """Load week mapping for 2023 season."""
    try:
        return common_utils.load_week_mapping(data_root, 2023)
    except Exception as e:
        pytest.skip(f"Could not load week mapping for 2023: {e}")


@pytest.fixture(scope="session")
def week_mapping_2022(data_root):
    """Load week mapping for 2022 season."""
    try:
        return common_utils.load_week_mapping(data_root, 2022)
    except Exception as e:
        pytest.skip(f"Could not load week mapping for 2022: {e}")


@pytest.fixture(scope="function")
def game_data_2024_week1(data_root, week_mapping_2024):
    """Load game-level receiving data for 2024 week 1."""
    try:
        return common_utils.load_game_level_data(
            source_root=data_root,
            table_path="plyr_gm/plyr_gm_rec",
            season=2024,
            weeks=[1],
            week_mapping=week_mapping_2024
        )
    except Exception as e:
        pytest.skip(f"Could not load game data for 2024 week 1: {e}")


@pytest.fixture(scope="session")
def cumulative_output_path(output_root):
    """Path to cumulative output data."""
    return os.path.join(output_root, "plyr_szn", "plyr_rec")


# ============================================================================
# Utility Functions
# ============================================================================

@pytest.fixture(scope="session")
def load_cumulative_output():
    """
    Factory fixture to load cumulative output for a specific season/week.

    Returns a function that loads cumulative data.
    """
    def _load(season: int, week: int, output_root: str = None) -> pd.DataFrame:
        """
        Load cumulative output data for a specific season and week.

        Args:
            season: Season year
            week: Week number
            output_root: Root directory (uses default if None)

        Returns:
            DataFrame with cumulative stats
        """
        if output_root is None:
            output_root = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw"

        path = os.path.join(output_root, "plyr_szn", "plyr_rec", f"season={season}", f"week={week}")

        if not os.path.exists(path):
            return pd.DataFrame()

        # Load all parquet files in the directory
        parquet_files = list(Path(path).glob("*.parquet"))

        if not parquet_files:
            return pd.DataFrame()

        dfs = []
        for file in parquet_files:
            try:
                df = pd.read_parquet(file, engine='pyarrow')
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not read {file}: {e}")
                continue

        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            # Add season/week back in for consistency
            result['season'] = season
            result['week'] = week
            return result
        else:
            return pd.DataFrame()

    return _load


@pytest.fixture(scope="session")
def load_game_level_output():
    """
    Factory fixture to load game-level data for comparison.

    Returns a function that loads game-level data.
    """
    def _load(season: int, week: int, data_root: str = None) -> pd.DataFrame:
        """
        Load game-level receiving data for a specific season and week.

        Args:
            season: Season year
            week: Week number
            data_root: Root directory (uses default if None)

        Returns:
            DataFrame with game-level stats
        """
        if data_root is None:
            data_root = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw"

        path = os.path.join(data_root, "plyr_gm", "plyr_gm_rec", f"season={season}", f"week={week}")

        if not os.path.exists(path):
            return pd.DataFrame()

        # Load all parquet files in the directory
        parquet_files = list(Path(path).glob("*.parquet"))

        if not parquet_files:
            return pd.DataFrame()

        dfs = []
        for file in parquet_files:
            try:
                df = pd.read_parquet(file, engine='pyarrow')
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not read {file}: {e}")
                continue

        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            # Add season/week back in for consistency
            result['season'] = season
            result['week'] = week
            return result
        else:
            return pd.DataFrame()

    return _load


# ============================================================================
# Comparison Utilities
# ============================================================================

@pytest.fixture(scope="session")
def assert_frame_equal_with_tolerance():
    """
    Factory fixture to compare DataFrames with tolerance.

    Returns a function that compares DataFrames.
    """
    def _compare(df1: pd.DataFrame, df2: pd.DataFrame,
                 float_tolerance: float = 0.01,
                 int_tolerance: int = 0,
                 check_dtype: bool = False,
                 sort_by: List[str] = None) -> None:
        """
        Compare two DataFrames with appropriate tolerances.

        Args:
            df1: First DataFrame
            df2: Second DataFrame
            float_tolerance: Tolerance for float comparisons
            int_tolerance: Tolerance for integer comparisons
            check_dtype: Whether to check data types
            sort_by: Columns to sort by before comparison
        """
        # Sort if specified
        if sort_by:
            df1 = df1.sort_values(sort_by).reset_index(drop=True)
            df2 = df2.sort_values(sort_by).reset_index(drop=True)

        # Check shape
        assert df1.shape == df2.shape, f"Shape mismatch: {df1.shape} vs {df2.shape}"

        # Check columns
        assert set(df1.columns) == set(df2.columns), \
            f"Column mismatch: {set(df1.columns) ^ set(df2.columns)}"

        # Compare each column
        for col in df1.columns:
            if pd.api.types.is_float_dtype(df1[col]) or pd.api.types.is_float_dtype(df2[col]):
                # Float comparison with tolerance
                pd.testing.assert_series_equal(
                    df1[col], df2[col],
                    check_dtype=check_dtype,
                    atol=float_tolerance,
                    rtol=0,
                    check_names=False
                )
            elif pd.api.types.is_integer_dtype(df1[col]) or pd.api.types.is_integer_dtype(df2[col]):
                # Integer comparison
                if int_tolerance == 0:
                    pd.testing.assert_series_equal(
                        df1[col], df2[col],
                        check_dtype=check_dtype,
                        check_names=False
                    )
                else:
                    # Allow tolerance for integers
                    diff = (df1[col] - df2[col]).abs()
                    assert diff.max() <= int_tolerance, \
                        f"Column {col} exceeds tolerance: max diff = {diff.max()}"
            else:
                # Other types (exact match)
                pd.testing.assert_series_equal(
                    df1[col], df2[col],
                    check_dtype=check_dtype,
                    check_names=False
                )

    return _compare


# ============================================================================
# Test Data Helpers
# ============================================================================

@pytest.fixture(scope="session")
def create_test_player_data():
    """
    Factory fixture to create synthetic test data for unit testing calculations.

    Returns a function that creates test player data.
    """
    def _create(weeks: List[int],
                targets: List[int],
                receptions: List[int] = None,
                yards: List[int] = None,
                air_yards: List[int] = None,
                touchdowns: List[int] = None,
                interceptions: List[int] = None,
                player_id: int = 1,
                season: int = 2024) -> pd.DataFrame:
        """
        Create synthetic player game data for testing.

        Args:
            weeks: List of week numbers
            targets: List of targets per week
            receptions: List of receptions per week (defaults to targets if None)
            yards: List of yards per week (defaults to receptions * 10)
            air_yards: List of air yards per week (defaults to targets * 8)
            touchdowns: List of TDs per week (defaults to 0)
            interceptions: List of INTs per week (defaults to 0)
            player_id: Player ID
            season: Season year

        Returns:
            DataFrame with game-level stats
        """
        n = len(weeks)

        if receptions is None:
            receptions = targets
        if yards is None:
            yards = [rec * 10 for rec in receptions]
        if air_yards is None:
            air_yards = [tgt * 8 for tgt in targets]
        if touchdowns is None:
            touchdowns = [0] * n
        if interceptions is None:
            interceptions = [0] * n

        return pd.DataFrame({
            'plyr_id': [player_id] * n,
            'season_id': [season] * n,
            'week_id': weeks,
            'week_num': weeks,
            'plyr_gm_rec_tgt': targets,
            'plyr_gm_rec': receptions,
            'plyr_gm_rec_yds': yards,
            'plyr_gm_rec_aybc': air_yards,
            'plyr_gm_rec_td': touchdowns,
            'plyr_gm_rec_int': interceptions,
            'plyr_gm_rec_first_dwn': [rec // 2 for rec in receptions],
            'plyr_gm_rec_yac': [yds - air for yds, air in zip(yards, air_yards)],
            'plyr_gm_rec_brkn_tkl': [rec // 3 for rec in receptions],
            'plyr_gm_rec_drp': [(tgt - rec) // 2 for tgt, rec in zip(targets, receptions)],
            'plyr_gm_rec_lng': [max(yds // rec, 0) if rec > 0 else 0 for yds, rec in zip(yards, receptions)],
            'plyr_gm_rec_fmbl': [0] * n,
        })

    return _create


# ============================================================================
# Logging and Reporting
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_logging(test_results_dir):
    """Set up logging for test execution."""
    import logging

    log_file = os.path.join(test_results_dir, "test_results_receiving.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Starting NFL Receiving Statistics Test Suite")
    logger.info("=" * 80)

    yield logger

    logger.info("=" * 80)
    logger.info("Test Suite Complete")
    logger.info("=" * 80)


# ============================================================================
# Parametrization Helpers
# ============================================================================

@pytest.fixture(scope="session")
def test_seasons():
    """List of seasons to test."""
    return [2022, 2023, 2024]


@pytest.fixture(scope="session")
def test_weeks():
    """List of weeks to test."""
    return list(range(1, 18))  # Weeks 1-17


@pytest.fixture(scope="session")
def sample_weeks():
    """Sample weeks for quick testing."""
    return [1, 5, 10, 17]
