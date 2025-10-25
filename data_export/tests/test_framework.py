"""
Reusable testing framework for cumulative statistics scripts.

This framework can be adapted for testing rushing.py, passing.py, defensive.py, etc.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Callable, Optional, Tuple
import pandas as pd
import numpy as np
from tabulate import tabulate

logger = logging.getLogger(__name__)


class CumulativeStatsTestFramework:
    """
    Reusable framework for testing cumulative statistics scripts.

    Can be adapted for rushing.py, passing.py, defensive.py, etc.
    """

    def __init__(
        self,
        script_name: str,
        input_tables: Dict[str, str],
        output_table: str,
        sum_columns: List[str],
        max_columns: List[str],
        calc_columns: Dict[str, str],
        data_root: str,
        output_root: str
    ):
        """
        Initialize the test framework.

        Args:
            script_name: Name of script under test (e.g., "receiving.py")
            input_tables: Dict of table names and relative paths
                         e.g., {"receiving": "plyr_gm/plyr_gm_rec", "fumbles": "plyr_gm/plyr_gm_fmbl"}
            output_table: Output table path (e.g., "plyr_szn/plyr_rec")
            sum_columns: Columns that should be summed (game-level names)
            max_columns: Columns that should be maxed (game-level names)
            calc_columns: Dict of calculated columns with their descriptions
            data_root: Root directory for input data
            output_root: Root directory for output data
        """
        self.script_name = script_name
        self.input_tables = input_tables
        self.output_table = output_table
        self.sum_columns = sum_columns
        self.max_columns = max_columns
        self.calc_columns = calc_columns
        self.data_root = data_root
        self.output_root = output_root

        # Test results storage
        self.test_results = {
            "passed": [],
            "failed": [],
            "warnings": [],
            "metrics": {}
        }

    def load_cumulative_output(self, season: int, week: int) -> pd.DataFrame:
        """
        Load cumulative output for a specific season and week.

        Args:
            season: Season year
            week: Week number

        Returns:
            DataFrame with cumulative stats
        """
        path = os.path.join(self.output_root, self.output_table, f"season={season}", f"week={week}")

        if not os.path.exists(path):
            logger.warning(f"Output path does not exist: {path}")
            return pd.DataFrame()

        parquet_files = list(Path(path).glob("*.parquet"))

        if not parquet_files:
            logger.warning(f"No parquet files found in {path}")
            return pd.DataFrame()

        dfs = []
        for file in parquet_files:
            try:
                df = pd.read_parquet(file, engine='pyarrow')
                dfs.append(df)
            except Exception as e:
                logger.error(f"Could not read {file}: {e}")
                continue

        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            # Add season/week for consistency
            result['season'] = season
            result['week'] = week
            return result
        else:
            return pd.DataFrame()

    def load_game_level_data(self, table_key: str, season: int, week: int) -> pd.DataFrame:
        """
        Load game-level data for a specific table, season, and week.

        Args:
            table_key: Key in input_tables dict
            season: Season year
            week: Week number

        Returns:
            DataFrame with game-level stats
        """
        if table_key not in self.input_tables:
            raise ValueError(f"Unknown table key: {table_key}")

        table_path = self.input_tables[table_key]
        path = os.path.join(self.data_root, table_path, f"season={season}", f"week={week}")

        if not os.path.exists(path):
            logger.warning(f"Game data path does not exist: {path}")
            return pd.DataFrame()

        parquet_files = list(Path(path).glob("*.parquet"))

        if not parquet_files:
            logger.warning(f"No parquet files found in {path}")
            return pd.DataFrame()

        dfs = []
        for file in parquet_files:
            try:
                df = pd.read_parquet(file, engine='pyarrow')
                dfs.append(df)
            except Exception as e:
                logger.error(f"Could not read {file}: {e}")
                continue

        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            result['season'] = season
            result['week'] = week
            return result
        else:
            return pd.DataFrame()

    def test_monotonic_progression(
        self,
        season: int,
        start_week: int = 1,
        end_week: int = 17
    ) -> Tuple[bool, List[str]]:
        """
        Test that cumulative stats are monotonically increasing week-over-week.

        Args:
            season: Season to test
            start_week: Starting week
            end_week: Ending week

        Returns:
            Tuple of (all_passed, list of violations)
        """
        logger.info(f"Testing monotonic progression for season {season}, weeks {start_week}-{end_week}")

        violations = []

        # Load all weeks
        week_data = {}
        for week in range(start_week, end_week + 1):
            df = self.load_cumulative_output(season, week)
            if not df.empty:
                week_data[week] = df

        if len(week_data) < 2:
            violations.append(f"Not enough weeks to test (found {len(week_data)})")
            return False, violations

        # Test monotonic progression for each player
        all_weeks = sorted(week_data.keys())

        for i in range(len(all_weeks) - 1):
            week1 = all_weeks[i]
            week2 = all_weeks[i + 1]

            df1 = week_data[week1]
            df2 = week_data[week2]

            # Merge on player_id
            merged = df1.merge(
                df2,
                on='plyr_id',
                how='inner',
                suffixes=(f'_w{week1}', f'_w{week2}')
            )

            # Check each sum column
            for col in self.sum_columns:
                # Get season-level column name (remove 'plyr_gm_' prefix)
                season_col = col.replace('plyr_gm_rec_', 'plyr_rec_')
                if season_col == 'plyr_rec':
                    season_col = 'plyr_rec'

                col1 = f'{season_col}_w{week1}'
                col2 = f'{season_col}_w{week2}'

                if col1 in merged.columns and col2 in merged.columns:
                    # Check if week2 >= week1 (allowing for equality if player didn't play)
                    decreased = merged[merged[col2] < merged[col1]]

                    if not decreased.empty:
                        violations.append(
                            f"Season {season}, Week {week1}→{week2}, Column {season_col}: "
                            f"{len(decreased)} players decreased (should be monotonic)"
                        )

        passed = len(violations) == 0

        if passed:
            logger.info(f"✓ Monotonic progression test PASSED for season {season}")
            self.test_results["passed"].append(f"Monotonic progression - Season {season}")
        else:
            logger.error(f"✗ Monotonic progression test FAILED for season {season}")
            self.test_results["failed"].append(f"Monotonic progression - Season {season}")
            for v in violations:
                self.test_results["warnings"].append(v)

        return passed, violations

    def test_week_1_equals_game_data(self, season: int, table_key: str) -> Tuple[bool, List[str]]:
        """
        Test that week 1 cumulative stats exactly match week 1 game data.

        Args:
            season: Season to test
            table_key: Key for input table to compare

        Returns:
            Tuple of (passed, list of discrepancies)
        """
        logger.info(f"Testing week 1 cumulative vs game data for season {season}")

        discrepancies = []

        # Load week 1 cumulative
        cum_df = self.load_cumulative_output(season, 1)
        if cum_df.empty:
            discrepancies.append(f"No cumulative data for season {season}, week 1")
            return False, discrepancies

        # Load week 1 game data
        game_df = self.load_game_level_data(table_key, season, 1)
        if game_df.empty:
            discrepancies.append(f"No game data for season {season}, week 1")
            return False, discrepancies

        # Compare sum columns (week 1 cumulative should equal week 1 game)
        for game_col in self.sum_columns:
            season_col = game_col.replace('plyr_gm_rec_', 'plyr_rec_')
            if season_col == 'plyr_rec':
                season_col = 'plyr_rec'

            if game_col not in game_df.columns:
                logger.warning(f"Column {game_col} not in game data")
                continue

            if season_col not in cum_df.columns:
                logger.warning(f"Column {season_col} not in cumulative data")
                continue

            # Merge and compare
            merged = game_df[['plyr_id', game_col]].merge(
                cum_df[['plyr_id', season_col]],
                on='plyr_id',
                how='inner'
            )

            # Check for differences
            diffs = merged[merged[game_col] != merged[season_col]]

            if not diffs.empty:
                discrepancies.append(
                    f"Column {game_col}: {len(diffs)} players have different values "
                    f"between week 1 game and week 1 cumulative"
                )

        passed = len(discrepancies) == 0

        if passed:
            logger.info(f"✓ Week 1 comparison test PASSED for season {season}")
            self.test_results["passed"].append(f"Week 1 equals game - Season {season}")
        else:
            logger.error(f"✗ Week 1 comparison test FAILED for season {season}")
            self.test_results["failed"].append(f"Week 1 equals game - Season {season}")
            for d in discrepancies:
                self.test_results["warnings"].append(d)

        return passed, discrepancies

    def test_no_missing_players(
        self,
        season: int,
        week: int,
        table_key: str
    ) -> Tuple[bool, List[int]]:
        """
        Test that all players from game data appear in cumulative output.

        Args:
            season: Season to test
            week: Week to test
            table_key: Key for input table

        Returns:
            Tuple of (all_present, list of missing player IDs)
        """
        logger.info(f"Testing for missing players in season {season}, week {week}")

        # Load game data for all weeks 1 through week
        all_game_players = set()
        for w in range(1, week + 1):
            game_df = self.load_game_level_data(table_key, season, w)
            if not game_df.empty and 'plyr_id' in game_df.columns:
                all_game_players.update(game_df['plyr_id'].unique())

        # Load cumulative output
        cum_df = self.load_cumulative_output(season, week)
        if cum_df.empty:
            return False, list(all_game_players)

        cum_players = set(cum_df['plyr_id'].unique())

        missing_players = all_game_players - cum_players

        passed = len(missing_players) == 0

        if passed:
            logger.info(f"✓ No missing players test PASSED for season {season}, week {week}")
            self.test_results["passed"].append(f"No missing players - S{season}W{week}")
        else:
            logger.error(f"✗ No missing players test FAILED for season {season}, week {week}: "
                        f"{len(missing_players)} missing")
            self.test_results["failed"].append(f"No missing players - S{season}W{week}")
            self.test_results["warnings"].append(
                f"Season {season}, Week {week}: {len(missing_players)} players missing from output"
            )

        return passed, list(missing_players)

    def test_no_duplicate_player_weeks(self, season: int, week: int) -> Tuple[bool, pd.DataFrame]:
        """
        Test that there are no duplicate player-week combinations.

        Args:
            season: Season to test
            week: Week to test

        Returns:
            Tuple of (no_duplicates, DataFrame of duplicates)
        """
        logger.info(f"Testing for duplicate player-weeks in season {season}, week {week}")

        cum_df = self.load_cumulative_output(season, week)

        if cum_df.empty:
            return True, pd.DataFrame()

        # Check for duplicates
        duplicates = cum_df[cum_df.duplicated(subset=['plyr_id'], keep=False)]

        passed = duplicates.empty

        if passed:
            logger.info(f"✓ No duplicates test PASSED for season {season}, week {week}")
            self.test_results["passed"].append(f"No duplicates - S{season}W{week}")
        else:
            logger.error(f"✗ No duplicates test FAILED for season {season}, week {week}: "
                        f"{len(duplicates)} duplicate rows")
            self.test_results["failed"].append(f"No duplicates - S{season}W{week}")
            self.test_results["warnings"].append(
                f"Season {season}, Week {week}: {len(duplicates)} duplicate player records"
            )

        return passed, duplicates

    def generate_summary_report(self) -> str:
        """
        Generate a summary report of all test results.

        Returns:
            String containing formatted report
        """
        report = []

        report.append("=" * 80)
        report.append("TEST EXECUTION SUMMARY")
        report.append("=" * 80)
        report.append("")

        # Overall statistics
        total_passed = len(self.test_results["passed"])
        total_failed = len(self.test_results["failed"])
        total_tests = total_passed + total_failed

        report.append(f"Total Tests Run: {total_tests}")
        report.append(f"Tests Passed: {total_passed}")
        report.append(f"Tests Failed: {total_failed}")

        if total_tests > 0:
            pass_rate = (total_passed / total_tests) * 100
            report.append(f"Pass Rate: {pass_rate:.1f}%")
        else:
            report.append("Pass Rate: N/A")

        report.append("")

        # Passed tests
        if self.test_results["passed"]:
            report.append("PASSED TESTS:")
            for test in self.test_results["passed"]:
                report.append(f"  ✓ {test}")
            report.append("")

        # Failed tests
        if self.test_results["failed"]:
            report.append("FAILED TESTS:")
            for test in self.test_results["failed"]:
                report.append(f"  ✗ {test}")
            report.append("")

        # Warnings
        if self.test_results["warnings"]:
            report.append("WARNINGS:")
            for warning in self.test_results["warnings"]:
                report.append(f"  ⚠ {warning}")
            report.append("")

        # Metrics
        if self.test_results["metrics"]:
            report.append("METRICS:")
            for metric, value in self.test_results["metrics"].items():
                report.append(f"  • {metric}: {value}")
            report.append("")

        report.append("=" * 80)

        return "\n".join(report)

    def save_report(self, output_file: str) -> None:
        """
        Save test report to file.

        Args:
            output_file: Path to output file
        """
        report = self.generate_summary_report()

        with open(output_file, 'w') as f:
            f.write(report)

        logger.info(f"Test report saved to {output_file}")


# ============================================================================
# Calculation Verification Helpers
# ============================================================================

def calculate_expected_adot(
    total_air_yards: float,
    total_targets: float
) -> float:
    """
    Calculate expected ADOT (Average Depth of Target).

    Args:
        total_air_yards: Total air yards across all weeks
        total_targets: Total targets across all weeks

    Returns:
        Expected ADOT value
    """
    if total_targets == 0:
        return np.nan
    return total_air_yards / total_targets


def calculate_expected_passer_rating(
    attempts: float,
    completions: float,
    yards: float,
    touchdowns: float,
    interceptions: float
) -> float:
    """
    Calculate expected NFL passer rating.

    Args:
        attempts: Total targets (attempts)
        completions: Total receptions (completions)
        yards: Total receiving yards
        touchdowns: Total receiving touchdowns
        interceptions: Total interceptions on targets

    Returns:
        Expected passer rating
    """
    if attempts == 0:
        return np.nan

    # Four components (each capped 0-2.375)
    a = min(max(((completions / attempts) - 0.3) * 5, 0), 2.375)
    b = min(max(((yards / attempts) - 3) * 0.25, 0), 2.375)
    c = min(max((touchdowns / attempts) * 20, 0), 2.375)
    d = min(max(2.375 - ((interceptions / attempts) * 25), 0), 2.375)

    rating = ((a + b + c + d) / 6) * 100

    return rating


def calculate_expected_catch_pct(
    total_receptions: float,
    total_targets: float
) -> float:
    """
    Calculate expected catch percentage.

    Args:
        total_receptions: Total receptions
        total_targets: Total targets

    Returns:
        Expected catch percentage (0-1)
    """
    if total_targets == 0:
        return np.nan
    return total_receptions / total_targets


def verify_calculation(
    actual: float,
    expected: float,
    tolerance: float = 0.01,
    metric_name: str = "metric"
) -> Tuple[bool, str]:
    """
    Verify that an actual calculation matches expected value within tolerance.

    Args:
        actual: Actual calculated value
        expected: Expected value
        tolerance: Acceptable difference
        metric_name: Name of metric being verified

    Returns:
        Tuple of (matches, message)
    """
    # Handle NaN cases
    if pd.isna(actual) and pd.isna(expected):
        return True, f"{metric_name}: Both NaN (OK)"

    if pd.isna(actual) or pd.isna(expected):
        return False, f"{metric_name}: One is NaN (actual={actual}, expected={expected})"

    # Check tolerance
    diff = abs(actual - expected)

    if diff <= tolerance:
        return True, f"{metric_name}: {actual:.4f} ≈ {expected:.4f} (diff={diff:.6f})"
    else:
        return False, f"{metric_name}: {actual:.4f} ≠ {expected:.4f} (diff={diff:.6f} > tolerance={tolerance})"
