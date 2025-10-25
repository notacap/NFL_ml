"""
Comprehensive tests for NFL receiving cumulative statistics.

This module tests the receiving.py script for:
- Ground truth validation
- Calculation logic correctness
- Data integrity
- Edge cases
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Import modules under test
sys.path.insert(0, str(Path(__file__).parent.parent / 'build_cumulative_stats'))
import receiving
import common_utils

# Import test framework
from test_framework import (
    CumulativeStatsTestFramework,
    calculate_expected_adot,
    calculate_expected_passer_rating,
    calculate_expected_catch_pct,
    verify_calculation
)


# ============================================================================
# Test Framework Setup
# ============================================================================

@pytest.fixture(scope="module")
def framework():
    """Create test framework instance for receiving stats."""
    return CumulativeStatsTestFramework(
        script_name="receiving.py",
        input_tables={
            "receiving": "plyr_gm/plyr_gm_rec",
            "fumbles": "plyr_gm/plyr_gm_fmbl"
        },
        output_table="plyr_szn/plyr_rec",
        sum_columns=receiving.GAME_SUM_COLUMNS,
        max_columns=receiving.GAME_MAX_COLUMNS,
        calc_columns=receiving.get_calculated_columns(),
        data_root=r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw",
        output_root=r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw"
    )


# ============================================================================
# CRITICAL TESTS - Ground Truth Validation
# ============================================================================

@pytest.mark.critical
@pytest.mark.ground_truth
@pytest.mark.parametrize("season", [2022, 2023, 2024])
def test_week_17_cumulative_exists(framework, season):
    """
    Test that week 17 cumulative data exists for each season.

    This is the baseline for ground truth validation.
    """
    df = framework.load_cumulative_output(season, 17)

    assert not df.empty, f"Week 17 cumulative data not found for season {season}"
    assert len(df) > 0, f"Week 17 cumulative data is empty for season {season}"

    # Check that key columns exist
    required_cols = ['plyr_id', 'season_id', 'plyr_rec_tgt', 'plyr_rec', 'plyr_rec_yds']
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"


# ============================================================================
# CRITICAL TESTS - Calculation Logic
# ============================================================================

@pytest.mark.critical
@pytest.mark.calculation
def test_adot_calculation_with_synthetic_data(create_test_player_data):
    """
    Test ADOT calculation with known synthetic data.

    ADOT should be sum(air_yards) / sum(targets), NOT average(weekly_adot).
    """
    # Create test data: 3 weeks with different volumes
    test_data = create_test_player_data(
        weeks=[1, 2, 3],
        targets=[5, 8, 6],
        receptions=[4, 6, 5],
        air_yards=[30, 50, 40]  # Total = 120
    )

    # Calculate cumulative stats through week 3
    cumulative = common_utils.calculate_cumulative_stats(
        game_df=test_data,
        group_cols=['plyr_id', 'season_id'],
        sum_cols=receiving.GAME_SUM_COLUMNS,
        max_cols=receiving.GAME_MAX_COLUMNS,
        calculated_cols=receiving.get_calculated_columns(),
        target_week=3,
        rename_func=receiving.rename_game_to_season_columns
    )

    assert len(cumulative) == 1, "Should have exactly 1 player"

    # Check ADOT calculation
    expected_adot = (30 + 50 + 40) / (5 + 8 + 6)  # 120 / 19 = 6.316
    actual_adot = cumulative['plyr_rec_adot'].iloc[0]

    assert abs(actual_adot - expected_adot) < 0.001, \
        f"ADOT mismatch: expected {expected_adot:.4f}, got {actual_adot:.4f}"


@pytest.mark.critical
@pytest.mark.calculation
def test_passer_rating_perfect_scenario():
    """
    Test passer rating calculation with perfect rating scenario.

    Perfect rating should be 158.3 (all completions, high YPA, TDs, no INTs).
    """
    # Create perfect scenario
    attempts = pd.Series([10])
    completions = pd.Series([10])
    yards = pd.Series([120])  # 12 YPA
    touchdowns = pd.Series([3])
    interceptions = pd.Series([0])

    rating = receiving.calculate_passer_rating_series(
        attempts, completions, yards, touchdowns, interceptions
    )

    assert abs(rating.iloc[0] - 158.3) < 0.1, \
        f"Perfect rating should be 158.3, got {rating.iloc[0]:.1f}"


@pytest.mark.critical
@pytest.mark.calculation
def test_passer_rating_known_example():
    """
    Test passer rating calculation with known example.

    50% completion, 10 YPA, 1 TD, 0 INT in 10 attempts = 118.75 rating
    """
    attempts = pd.Series([10])
    completions = pd.Series([5])
    yards = pd.Series([100])
    touchdowns = pd.Series([1])
    interceptions = pd.Series([0])

    rating = receiving.calculate_passer_rating_series(
        attempts, completions, yards, touchdowns, interceptions
    )

    # Calculate expected
    a = min(max(((5/10) - 0.3) * 5, 0), 2.375)  # 1.0
    b = min(max(((100/10) - 3) * 0.25, 0), 2.375)  # 1.75
    c = min(max((1/10) * 20, 0), 2.375)  # 2.0
    d = min(max(2.375 - ((0/10) * 25), 0), 2.375)  # 2.375
    expected = ((a + b + c + d) / 6) * 100  # 118.75

    assert abs(rating.iloc[0] - expected) < 0.1, \
        f"Rating mismatch: expected {expected:.1f}, got {rating.iloc[0]:.1f}"


@pytest.mark.critical
@pytest.mark.calculation
def test_passer_rating_zero_attempts():
    """
    Test that passer rating returns NaN for zero attempts (no division by zero error).
    """
    attempts = pd.Series([0])
    completions = pd.Series([0])
    yards = pd.Series([0])
    touchdowns = pd.Series([0])
    interceptions = pd.Series([0])

    rating = receiving.calculate_passer_rating_series(
        attempts, completions, yards, touchdowns, interceptions
    )

    assert pd.isna(rating.iloc[0]), "Rating should be NaN for zero attempts"


@pytest.mark.critical
@pytest.mark.calculation
def test_rate_stats_not_averaged(create_test_player_data):
    """
    Verify rate statistics are RECALCULATED from cumulative totals, not averaged.

    This is critical - averaging rates with different volumes gives wrong answers.
    """
    # Create scenario where averaging would give wrong answer
    test_data = create_test_player_data(
        weeks=[1, 2, 3],
        targets=[10, 5, 5],  # Different volumes
        receptions=[8, 3, 4]  # Different catch rates: 80%, 60%, 80%
    )

    # Game-level catch percentages would be: 80%, 60%, 80%
    # Average would be: 73.3%
    # Correct cumulative: 15/20 = 75%

    cumulative = common_utils.calculate_cumulative_stats(
        game_df=test_data,
        group_cols=['plyr_id', 'season_id'],
        sum_cols=receiving.GAME_SUM_COLUMNS,
        max_cols=receiving.GAME_MAX_COLUMNS,
        calculated_cols=receiving.get_calculated_columns(),
        target_week=3,
        rename_func=receiving.rename_game_to_season_columns
    )

    actual_catch_pct = cumulative['plyr_rec_catch_pct'].iloc[0]

    # Should be 15/20 = 0.75
    assert abs(actual_catch_pct - 0.75) < 0.001, \
        f"Catch % should be 0.75 (not averaged), got {actual_catch_pct:.4f}"

    # Should NOT be 0.733 (average of game rates)
    assert abs(actual_catch_pct - 0.7333) > 0.01, \
        f"Catch % should NOT be 0.733 (averaged), got {actual_catch_pct:.4f}"


@pytest.mark.critical
@pytest.mark.calculation
def test_safe_divide_zero_denominator():
    """
    Test that safe_divide returns NaN for zero denominator.
    """
    numerator = pd.Series([10, 20, 30])
    denominator = pd.Series([2, 0, 5])

    result = common_utils.safe_divide(numerator, denominator)

    assert result.iloc[0] == 5.0, "10/2 should be 5.0"
    assert pd.isna(result.iloc[1]), "10/0 should be NaN"
    assert result.iloc[2] == 6.0, "30/5 should be 6.0"


# ============================================================================
# CRITICAL TESTS - Data Integrity
# ============================================================================

@pytest.mark.critical
@pytest.mark.integrity
@pytest.mark.parametrize("season", [2022, 2023, 2024])
def test_monotonic_progression(framework, season):
    """
    Test that cumulative stats are monotonically increasing week-over-week.

    Volume stats should never decrease from one week to the next.
    """
    passed, violations = framework.test_monotonic_progression(season, start_week=1, end_week=17)

    if not passed:
        pytest.fail(f"Monotonic progression violated:\n" + "\n".join(violations[:5]))


@pytest.mark.critical
@pytest.mark.integrity
@pytest.mark.parametrize("season", [2022, 2023, 2024])
def test_week_1_equals_game_data(framework, season):
    """
    Test that week 1 cumulative stats exactly match week 1 game data.
    """
    passed, discrepancies = framework.test_week_1_equals_game_data(season, "receiving")

    if not passed:
        pytest.fail(f"Week 1 cumulative != Week 1 game:\n" + "\n".join(discrepancies[:5]))


@pytest.mark.critical
@pytest.mark.integrity
@pytest.mark.parametrize("season,week", [(2024, 1), (2024, 5), (2024, 10), (2024, 17)])
def test_no_missing_players(framework, season, week):
    """
    Test that all players from game data appear in cumulative output.
    """
    passed, missing_players = framework.test_no_missing_players(season, week, "receiving")

    if not passed:
        pytest.fail(f"Found {len(missing_players)} missing players. First 10: {missing_players[:10]}")


@pytest.mark.critical
@pytest.mark.integrity
@pytest.mark.parametrize("season,week", [(2024, 1), (2024, 5), (2024, 10), (2024, 17)])
def test_no_duplicate_player_weeks(framework, season, week):
    """
    Test that there are no duplicate player-week combinations.
    """
    passed, duplicates = framework.test_no_duplicate_player_weeks(season, week)

    if not passed:
        pytest.fail(f"Found {len(duplicates)} duplicate player records")


@pytest.mark.integrity
@pytest.mark.parametrize("season", [2024])
def test_column_completeness(framework, season):
    """
    Test that all expected columns are present in output.
    """
    df = framework.load_cumulative_output(season, 1)

    if df.empty:
        pytest.skip(f"No data for season {season}, week 1")

    expected_columns = [
        'plyr_id', 'season_id', 'week_id',
        'plyr_rec_tgt', 'plyr_rec', 'plyr_rec_yds', 'plyr_rec_td',
        'plyr_rec_catch_pct', 'plyr_rec_adot', 'plyr_rec_pass_rtg',
        'plyr_rec_yds_rec', 'plyr_rec_yds_tgt', 'plyr_rec_ypg',
        'plyr_rec_fmbl', 'plyr_rec_gm'
    ]

    missing = set(expected_columns) - set(df.columns)

    assert len(missing) == 0, f"Missing columns: {missing}"


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

@pytest.mark.edge_case
def test_player_with_zero_targets(create_test_player_data):
    """
    Test player with exactly 0 targets (should not error, rate stats = NaN).
    """
    test_data = create_test_player_data(
        weeks=[1],
        targets=[0],
        receptions=[0],
        yards=[0],
        air_yards=[0]
    )

    cumulative = common_utils.calculate_cumulative_stats(
        game_df=test_data,
        group_cols=['plyr_id', 'season_id'],
        sum_cols=receiving.GAME_SUM_COLUMNS,
        max_cols=receiving.GAME_MAX_COLUMNS,
        calculated_cols=receiving.get_calculated_columns(),
        target_week=1,
        rename_func=receiving.rename_game_to_season_columns
    )

    assert len(cumulative) == 1, "Should have 1 player"

    # Volume stats should be 0
    assert cumulative['plyr_rec_tgt'].iloc[0] == 0
    assert cumulative['plyr_rec'].iloc[0] == 0
    assert cumulative['plyr_rec_yds'].iloc[0] == 0

    # Rate stats should be NaN
    assert pd.isna(cumulative['plyr_rec_adot'].iloc[0])
    assert pd.isna(cumulative['plyr_rec_catch_pct'].iloc[0])
    assert pd.isna(cumulative['plyr_rec_pass_rtg'].iloc[0])


@pytest.mark.edge_case
def test_player_zero_receptions_multiple_targets(create_test_player_data):
    """
    Test player with targets but no completions (0% catch rate).
    """
    test_data = create_test_player_data(
        weeks=[1, 2, 3],
        targets=[3, 4, 5],
        receptions=[0, 0, 0],
        yards=[0, 0, 0]
    )

    cumulative = common_utils.calculate_cumulative_stats(
        game_df=test_data,
        group_cols=['plyr_id', 'season_id'],
        sum_cols=receiving.GAME_SUM_COLUMNS,
        max_cols=receiving.GAME_MAX_COLUMNS,
        calculated_cols=receiving.get_calculated_columns(),
        target_week=3,
        rename_func=receiving.rename_game_to_season_columns
    )

    # Catch pct should be 0.0 (not NaN)
    assert cumulative['plyr_rec_catch_pct'].iloc[0] == 0.0

    # Yards per reception should be NaN (0 receptions)
    assert pd.isna(cumulative['plyr_rec_yds_rec'].iloc[0])


@pytest.mark.edge_case
def test_player_starts_mid_season(create_test_player_data):
    """
    Test player who first appears in week 10 (not week 1).
    """
    test_data = create_test_player_data(
        weeks=[10, 11, 12],
        targets=[6, 8, 7],
        receptions=[4, 6, 5],
        yards=[50, 80, 65]
    )

    # Calculate cumulative through week 12
    cumulative = common_utils.calculate_cumulative_stats(
        game_df=test_data,
        group_cols=['plyr_id', 'season_id'],
        sum_cols=receiving.GAME_SUM_COLUMNS,
        max_cols=receiving.GAME_MAX_COLUMNS,
        calculated_cols=receiving.get_calculated_columns(),
        target_week=12,
        rename_func=receiving.rename_game_to_season_columns
    )

    # Should have cumulative stats from weeks 10-12
    assert cumulative['plyr_rec_tgt'].iloc[0] == 21  # 6+8+7
    assert cumulative['plyr_rec'].iloc[0] == 15  # 4+6+5
    assert cumulative['plyr_rec_yds'].iloc[0] == 195  # 50+80+65


@pytest.mark.edge_case
@pytest.mark.parametrize("season", [2024])
def test_longest_reception_less_than_total_yards(framework, season):
    """
    Test that longest reception <= total yards for all players.
    """
    df = framework.load_cumulative_output(season, 17)

    if df.empty:
        pytest.skip(f"No data for season {season}, week 17")

    invalid = df[df['plyr_rec_lng'] > df['plyr_rec_yds']]

    assert invalid.empty, \
        f"Found {len(invalid)} players where longest reception > total yards"


# ============================================================================
# PERFORMANCE TESTS (Optional - marked slow)
# ============================================================================

@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.parametrize("season", [2024])
def test_output_partition_structure(framework, season):
    """
    Test that output has correct partition structure.
    """
    output_root = framework.output_root
    table_path = framework.output_table

    base_path = Path(output_root) / table_path

    # Check that season partition exists
    season_path = base_path / f"season={season}"
    assert season_path.exists(), f"Season partition not found: {season_path}"

    # Check that week partitions exist
    found_weeks = []
    for week in range(1, 18):
        week_path = season_path / f"week={week}"
        if week_path.exists():
            # Check that parquet files exist
            parquet_files = list(week_path.glob("*.parquet"))
            if parquet_files:
                found_weeks.append(week)

    assert len(found_weeks) > 0, f"No week partitions with data found for season {season}"


# ============================================================================
# TEST SUITE REPORTING
# ============================================================================

def pytest_sessionfinish(session, exitstatus):
    """
    Generate summary report after all tests complete.
    """
    # This will be called automatically by pytest
    pass
