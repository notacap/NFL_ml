"""
Rolling Feature Verification Script

This script verifies that rolling feature calculations in the NFL WR
receiving statistics parquet file were computed correctly according to
the business rules specified in build_rolling_features.py.

Author: Data Analyst
Created: 2025-12-06
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Any
import json

warnings.filterwarnings('ignore')

# Constants from the feature builder
IMPUTATION_SENTINEL = -999
CROSS_SEASON_PLAYER_ID = 'plyr_guid'
ROLLING_WINDOWS = [3, 5]

# Features that have -999 imputed values
FEATURES_WITH_IMPUTATION = [
    'plyr_gm_rec_first_dwn',
    'plyr_gm_rec_tgt_share',
    'plyr_gm_rec_epa',
    'plyr_gm_rec_ay_share',
    'plyr_gm_rec_wopr',
    'plyr_gm_rec_racr'
]

# Source column mappings
SOURCE_COLUMN_MAP = {
    'yds': 'plyr_gm_rec_yds',
    'tgt': 'plyr_gm_rec_tgt',
    'rec': 'plyr_gm_rec',
    'yac': 'plyr_gm_rec_yac',
    'first_dwn': 'plyr_gm_rec_first_dwn',
    'aybc': 'plyr_gm_rec_aybc',
    'td': 'plyr_gm_rec_td',
    'tgt_share': 'plyr_gm_rec_tgt_share',
    'epa': 'plyr_gm_rec_epa',
    'ay_share': 'plyr_gm_rec_ay_share',
    'wopr': 'plyr_gm_rec_wopr',
    'racr': 'plyr_gm_rec_racr'
}


def load_parquet_file(file_path: str) -> pd.DataFrame:
    """Load the parquet file and return DataFrame."""
    print(f"Loading parquet file: {file_path}")
    df = pd.read_parquet(file_path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def generate_dataset_summary(df: pd.DataFrame) -> Dict:
    """Generate summary statistics for the dataset."""
    summary = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'unique_players': df['plyr_guid'].nunique() if 'plyr_guid' in df.columns else df['plyr_id'].nunique(),
        'seasons': sorted(df['season_id'].unique().tolist()) if 'season_id' in df.columns else [],
        'rolling_feature_columns': [c for c in df.columns if c.startswith('roll_')],
        'source_columns_present': [c for c in SOURCE_COLUMN_MAP.values() if c in df.columns]
    }
    return summary


def manual_rolling_calculation(
    df: pd.DataFrame,
    player_id: str,
    stat_col: str,
    window: int,
    has_imputation: bool = False
) -> pd.Series:
    """
    Manually calculate rolling average for a single player.

    Follows the exact logic from build_rolling_features.py:
    1. Filter to player's data
    2. Sort by season_id, week_id
    3. Replace -999 with NaN if has_imputation
    4. Shift by 1 (use only prior games)
    5. Rolling mean with min_periods=1
    """
    # Get player's data
    player_df = df[df[CROSS_SEASON_PLAYER_ID] == player_id].copy()
    player_df = player_df.sort_values(['season_id', 'week_id']).reset_index(drop=True)

    # Get the stat series
    stat_series = player_df[stat_col].copy()

    # Mask imputed values
    if has_imputation:
        stat_series = stat_series.replace(IMPUTATION_SENTINEL, np.nan)

    # Shift by 1 (critical: use only prior games)
    shifted = stat_series.shift(1)

    # Rolling mean with min_periods=1
    manual_rolling = shifted.rolling(window=window, min_periods=1).mean()

    return manual_rolling, player_df


def select_sample_players(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Select sample players for verification covering different scenarios.

    Returns dict with player info:
    - multi_season_veteran: Player with games in multiple seasons
    - single_season_player: Player with games in only one season
    - player_with_imputed: Player who has -999 values
    """
    sample_players = {}

    # Find multi-season veteran
    player_seasons = df.groupby(CROSS_SEASON_PLAYER_ID)['season_id'].nunique()
    multi_season_players = player_seasons[player_seasons >= 2].index.tolist()

    if multi_season_players:
        # Pick one with decent number of games
        player_games = df[df[CROSS_SEASON_PLAYER_ID].isin(multi_season_players)].groupby(CROSS_SEASON_PLAYER_ID).size()
        top_player = player_games.nlargest(5).index[0]
        player_data = df[df[CROSS_SEASON_PLAYER_ID] == top_player]
        sample_players['multi_season_veteran'] = {
            'plyr_guid': top_player,
            'seasons': sorted(player_data['season_id'].unique().tolist()),
            'total_games': len(player_data),
            'player_name': player_data['plyr_display_name'].iloc[0] if 'plyr_display_name' in player_data.columns else 'Unknown'
        }

    # Find single-season player
    single_season_players = player_seasons[player_seasons == 1].index.tolist()
    if single_season_players:
        player_games = df[df[CROSS_SEASON_PLAYER_ID].isin(single_season_players)].groupby(CROSS_SEASON_PLAYER_ID).size()
        # Get player with 5-10 games
        mid_range = player_games[(player_games >= 5) & (player_games <= 10)]
        if len(mid_range) > 0:
            selected = mid_range.index[0]
        else:
            selected = player_games.index[0]
        player_data = df[df[CROSS_SEASON_PLAYER_ID] == selected]
        sample_players['single_season_player'] = {
            'plyr_guid': selected,
            'seasons': sorted(player_data['season_id'].unique().tolist()),
            'total_games': len(player_data),
            'player_name': player_data['plyr_display_name'].iloc[0] if 'plyr_display_name' in player_data.columns else 'Unknown'
        }

    # Find player with -999 values
    for col in FEATURES_WITH_IMPUTATION:
        if col in df.columns:
            imputed_mask = df[col] == IMPUTATION_SENTINEL
            if imputed_mask.any():
                players_with_imputed = df[imputed_mask][CROSS_SEASON_PLAYER_ID].unique()
                # Pick one with multiple games
                player_games = df[df[CROSS_SEASON_PLAYER_ID].isin(players_with_imputed)].groupby(CROSS_SEASON_PLAYER_ID).size()
                selected = player_games.nlargest(1).index[0]
                player_data = df[df[CROSS_SEASON_PLAYER_ID] == selected]
                imputed_count = (player_data[col] == IMPUTATION_SENTINEL).sum()
                sample_players['player_with_imputed'] = {
                    'plyr_guid': selected,
                    'seasons': sorted(player_data['season_id'].unique().tolist()),
                    'total_games': len(player_data),
                    'imputed_column': col,
                    'imputed_count': int(imputed_count),
                    'player_name': player_data['plyr_display_name'].iloc[0] if 'plyr_display_name' in player_data.columns else 'Unknown'
                }
                break

    return sample_players


def verify_player_rolling_features(
    df: pd.DataFrame,
    player_id: str,
    stat_name: str,
    windows: List[int] = [3, 5]
) -> Dict:
    """
    Verify rolling feature calculations for a specific player and statistic.

    Returns detailed comparison of manual vs file values.
    """
    source_col = SOURCE_COLUMN_MAP.get(stat_name, f'plyr_gm_rec_{stat_name}')
    has_imputation = source_col in FEATURES_WITH_IMPUTATION

    results = {
        'player_id': player_id,
        'stat_name': stat_name,
        'source_col': source_col,
        'has_imputation': has_imputation,
        'verifications': []
    }

    # Get player data sorted correctly
    player_df = df[df[CROSS_SEASON_PLAYER_ID] == player_id].copy()
    player_df = player_df.sort_values(['season_id', 'week_id']).reset_index(drop=True)

    for window in windows:
        feature_name = f'roll_{window}g_{stat_name}'

        if source_col not in player_df.columns:
            results['verifications'].append({
                'window': window,
                'feature_name': feature_name,
                'status': 'SKIPPED',
                'reason': f'Source column {source_col} not found'
            })
            continue

        if feature_name not in player_df.columns:
            results['verifications'].append({
                'window': window,
                'feature_name': feature_name,
                'status': 'SKIPPED',
                'reason': f'Rolling feature {feature_name} not found'
            })
            continue

        # Manual calculation
        manual_values, _ = manual_rolling_calculation(
            df, player_id, source_col, window, has_imputation
        )

        # Get file values
        file_values = player_df[feature_name].values

        # Compare
        matches = []
        for i in range(len(player_df)):
            manual_val = manual_values.iloc[i] if i < len(manual_values) else np.nan
            file_val = file_values[i]

            # Use numpy isclose for comparison with nan handling
            if pd.isna(manual_val) and pd.isna(file_val):
                is_match = True
            elif pd.isna(manual_val) or pd.isna(file_val):
                is_match = False
            else:
                is_match = np.isclose(manual_val, file_val, rtol=1e-5, equal_nan=True)

            matches.append({
                'game_idx': i,
                'season_id': player_df['season_id'].iloc[i],
                'week_id': player_df['week_id'].iloc[i],
                'game_seq_num': player_df['game_seq_num'].iloc[i] if 'game_seq_num' in player_df.columns else None,
                'career_game_seq_num': player_df['career_game_seq_num'].iloc[i] if 'career_game_seq_num' in player_df.columns else None,
                'raw_value': player_df[source_col].iloc[i],
                'manual_calc': round(manual_val, 6) if not pd.isna(manual_val) else None,
                'file_value': round(file_val, 6) if not pd.isna(file_val) else None,
                'match': is_match
            })

        all_match = all(m['match'] for m in matches)
        mismatch_count = sum(1 for m in matches if not m['match'])

        results['verifications'].append({
            'window': window,
            'feature_name': feature_name,
            'status': 'PASS' if all_match else 'FAIL',
            'total_games': len(matches),
            'mismatches': mismatch_count,
            'details': matches
        })

    return results


def verify_first_career_game_nulls(df: pd.DataFrame) -> Dict:
    """
    Verify that all rolling features are NaN for first career games.

    Rule: career_game_seq_num == 1 should have NaN for all roll_* columns
    because there is no prior data.
    """
    results = {
        'test_name': 'First Career Game Null Check',
        'description': 'Rolling features should be NaN for career_game_seq_num == 1',
        'status': 'UNKNOWN'
    }

    if 'career_game_seq_num' not in df.columns:
        results['status'] = 'SKIPPED'
        results['reason'] = 'career_game_seq_num column not found'
        return results

    first_career_games = df[df['career_game_seq_num'] == 1]
    rolling_cols = [c for c in df.columns if c.startswith('roll_') and not c.endswith('_rate')]

    # Also check efficiency rolling features
    rolling_cols.extend([c for c in df.columns if c.startswith('roll_') and ('_per_' in c or '_rate' in c)])
    rolling_cols = list(set(rolling_cols))

    violations = []
    for col in rolling_cols:
        if col in first_career_games.columns:
            non_null = first_career_games[col].notna()
            if non_null.any():
                violation_count = non_null.sum()
                sample_players = first_career_games[non_null][CROSS_SEASON_PLAYER_ID].head(3).tolist()
                violations.append({
                    'column': col,
                    'non_null_count': int(violation_count),
                    'total_first_games': len(first_career_games),
                    'sample_player_ids': sample_players
                })

    results['first_career_games_count'] = len(first_career_games)
    results['rolling_columns_checked'] = len(rolling_cols)
    results['violations'] = violations
    results['status'] = 'PASS' if len(violations) == 0 else 'FAIL'

    return results


def verify_cross_season_carryover(df: pd.DataFrame) -> Dict:
    """
    Verify that returning players have rolling data in their first season games.

    Rule: is_season_carryover == True should have non-NaN rolling values
    (they have prior season data to use).
    """
    results = {
        'test_name': 'Cross-Season Carryover Check',
        'description': 'Returning players should have rolling values in early season games',
        'status': 'UNKNOWN'
    }

    if 'is_season_carryover' not in df.columns:
        results['status'] = 'SKIPPED'
        results['reason'] = 'is_season_carryover column not found'
        return results

    carryover_rows = df[df['is_season_carryover'] == True]
    rolling_cols = [c for c in df.columns if c.startswith('roll_') and '_yds' in c][:4]  # Check a few key ones

    col_stats = []
    for col in rolling_cols:
        if col in carryover_rows.columns:
            non_null = carryover_rows[col].notna()
            col_stats.append({
                'column': col,
                'total_carryover_rows': len(carryover_rows),
                'rows_with_data': int(non_null.sum()),
                'pct_with_data': round(non_null.mean() * 100, 2)
            })

    results['carryover_rows_count'] = len(carryover_rows)
    results['column_statistics'] = col_stats

    # Carryover check passes if most carryover rows have data
    avg_pct = np.mean([s['pct_with_data'] for s in col_stats]) if col_stats else 0
    results['avg_pct_with_data'] = round(avg_pct, 2)
    results['status'] = 'PASS' if avg_pct > 90 else 'FAIL'

    return results


def verify_imputed_value_exclusion(df: pd.DataFrame) -> Dict:
    """
    Verify that -999 values are excluded from rolling calculations.

    For features with imputation, manually recalculate excluding -999
    and compare to file values.
    """
    results = {
        'test_name': 'Imputed Value Exclusion Check',
        'description': '-999 sentinel values should be excluded from rolling means',
        'status': 'UNKNOWN',
        'checks': []
    }

    for source_col in FEATURES_WITH_IMPUTATION:
        if source_col not in df.columns:
            continue

        imputed_mask = df[source_col] == IMPUTATION_SENTINEL
        if not imputed_mask.any():
            continue

        stat_name = source_col.replace('plyr_gm_rec_', '')

        # Find a player with imputed values
        players_with_imputed = df[imputed_mask][CROSS_SEASON_PLAYER_ID].unique()
        if len(players_with_imputed) == 0:
            continue

        # Pick a player with multiple imputed values
        test_player = None
        for pid in players_with_imputed:
            player_df = df[df[CROSS_SEASON_PLAYER_ID] == pid]
            imputed_count = (player_df[source_col] == IMPUTATION_SENTINEL).sum()
            if imputed_count >= 2 and len(player_df) >= 5:
                test_player = pid
                break

        if test_player is None:
            test_player = players_with_imputed[0]

        # Get player data
        player_df = df[df[CROSS_SEASON_PLAYER_ID] == test_player].sort_values(['season_id', 'week_id'])

        for window in ROLLING_WINDOWS:
            feature_name = f'roll_{window}g_{stat_name}'
            if feature_name not in player_df.columns:
                continue

            # Manual calc with -999 excluded
            stat_series = player_df[source_col].replace(IMPUTATION_SENTINEL, np.nan)
            shifted = stat_series.shift(1)
            manual_rolling = shifted.rolling(window=window, min_periods=1).mean()

            # Compare to file
            file_values = player_df[feature_name].values

            match_count = 0
            mismatch_examples = []
            for i in range(len(player_df)):
                manual_val = manual_rolling.iloc[i]
                file_val = file_values[i]

                if pd.isna(manual_val) and pd.isna(file_val):
                    match_count += 1
                elif pd.isna(manual_val) or pd.isna(file_val):
                    mismatch_examples.append({
                        'idx': i,
                        'manual': None if pd.isna(manual_val) else float(manual_val),
                        'file': None if pd.isna(file_val) else float(file_val)
                    })
                elif np.isclose(manual_val, file_val, rtol=1e-5):
                    match_count += 1
                else:
                    mismatch_examples.append({
                        'idx': i,
                        'manual': float(manual_val),
                        'file': float(file_val)
                    })

            results['checks'].append({
                'source_column': source_col,
                'feature_name': feature_name,
                'test_player': test_player,
                'total_games': len(player_df),
                'matches': match_count,
                'status': 'PASS' if len(mismatch_examples) == 0 else 'FAIL',
                'mismatch_examples': mismatch_examples[:3]  # Show first 3
            })

    all_pass = all(c['status'] == 'PASS' for c in results['checks'])
    results['status'] = 'PASS' if all_pass else 'FAIL'

    return results


def verify_no_future_leakage(df: pd.DataFrame) -> Dict:
    """
    Verify that current game stats are NOT included in current row's rolling average.

    For game N, roll_Xg_stat should NOT include game N's stat value.
    This is verified by checking the manual shift(1) calculation.
    """
    results = {
        'test_name': 'No Future Leakage Check',
        'description': 'Current game stats must not be in current row rolling average',
        'status': 'UNKNOWN',
        'checks': []
    }

    # Select a sample player for verification
    sample_player = df.groupby(CROSS_SEASON_PLAYER_ID).size().nlargest(1).index[0]
    player_df = df[df[CROSS_SEASON_PLAYER_ID] == sample_player].sort_values(['season_id', 'week_id'])

    source_col = 'plyr_gm_rec_yds'
    stat_name = 'yds'

    for window in ROLLING_WINDOWS:
        feature_name = f'roll_{window}g_{stat_name}'

        if source_col not in player_df.columns or feature_name not in player_df.columns:
            continue

        # For each game, verify the rolling average is computed from PRIOR games only
        leakage_detected = False
        leakage_examples = []

        for i in range(1, len(player_df)):  # Start from game 2
            # Current game's actual stat
            current_stat = player_df[source_col].iloc[i]

            # Rolling value in file
            file_rolling = player_df[feature_name].iloc[i]

            if pd.isna(file_rolling):
                continue

            # Get prior games' stats
            start_idx = max(0, i - window)
            prior_stats = player_df[source_col].iloc[start_idx:i]

            # Manual mean of prior games (this is what it should be)
            expected_rolling = prior_stats.mean()

            # Check if current game stat is included (wrong)
            if not pd.isna(current_stat):
                # If file value equals mean including current game, that's leakage
                wrong_mean = player_df[source_col].iloc[start_idx:i+1].mean()

                if not np.isclose(file_rolling, expected_rolling, rtol=1e-5) and \
                   np.isclose(file_rolling, wrong_mean, rtol=1e-5):
                    leakage_detected = True
                    leakage_examples.append({
                        'game_idx': i,
                        'current_stat': float(current_stat),
                        'file_rolling': float(file_rolling),
                        'expected_rolling': float(expected_rolling),
                        'wrong_mean_with_current': float(wrong_mean)
                    })

        results['checks'].append({
            'feature_name': feature_name,
            'test_player': sample_player,
            'games_checked': len(player_df) - 1,
            'leakage_detected': leakage_detected,
            'status': 'FAIL' if leakage_detected else 'PASS',
            'examples': leakage_examples[:3]
        })

    all_pass = all(c['status'] == 'PASS' for c in results['checks'])
    results['status'] = 'PASS' if all_pass else 'FAIL'

    return results


def verify_efficiency_features(df: pd.DataFrame) -> Dict:
    """
    Verify derived efficiency features match their formulas.

    - roll_Xg_yds_per_tgt = roll_Xg_yds / roll_Xg_tgt
    - roll_Xg_yds_per_rec = roll_Xg_yds / roll_Xg_rec
    - roll_Xg_catch_rate = roll_Xg_rec / roll_Xg_tgt
    """
    results = {
        'test_name': 'Efficiency Feature Formula Check',
        'description': 'Derived efficiency features should match their formulas',
        'status': 'UNKNOWN',
        'checks': []
    }

    formulas = {
        'yds_per_tgt': ('yds', 'tgt'),
        'yds_per_rec': ('yds', 'rec'),
        'catch_rate': ('rec', 'tgt')
    }

    for window in ROLLING_WINDOWS:
        prefix = f'roll_{window}g'

        for feature_suffix, (num_suffix, denom_suffix) in formulas.items():
            feature_name = f'{prefix}_{feature_suffix}'
            numerator_col = f'{prefix}_{num_suffix}'
            denominator_col = f'{prefix}_{denom_suffix}'

            if feature_name not in df.columns:
                results['checks'].append({
                    'feature': feature_name,
                    'status': 'SKIPPED',
                    'reason': f'Column {feature_name} not found'
                })
                continue

            if numerator_col not in df.columns or denominator_col not in df.columns:
                results['checks'].append({
                    'feature': feature_name,
                    'status': 'SKIPPED',
                    'reason': f'Required columns not found'
                })
                continue

            # Calculate expected value
            expected = np.where(
                df[denominator_col] > 0,
                df[numerator_col] / df[denominator_col],
                np.nan
            )

            # Compare to file values
            file_values = df[feature_name].values

            # Count matches
            matches = 0
            mismatches = 0
            mismatch_examples = []

            for i in range(len(df)):
                exp = expected[i]
                file_val = file_values[i]

                if pd.isna(exp) and pd.isna(file_val):
                    matches += 1
                elif pd.isna(exp) or pd.isna(file_val):
                    mismatches += 1
                    if len(mismatch_examples) < 3:
                        mismatch_examples.append({
                            'idx': i,
                            'expected': None if pd.isna(exp) else float(exp),
                            'file': None if pd.isna(file_val) else float(file_val)
                        })
                elif np.isclose(exp, file_val, rtol=1e-5):
                    matches += 1
                else:
                    mismatches += 1
                    if len(mismatch_examples) < 3:
                        mismatch_examples.append({
                            'idx': i,
                            'expected': float(exp),
                            'file': float(file_val)
                        })

            results['checks'].append({
                'feature': feature_name,
                'formula': f'{numerator_col} / {denominator_col}',
                'total_rows': len(df),
                'matches': matches,
                'mismatches': mismatches,
                'status': 'PASS' if mismatches == 0 else 'FAIL',
                'mismatch_examples': mismatch_examples
            })

    all_pass = all(c['status'] == 'PASS' for c in results['checks'] if c['status'] != 'SKIPPED')
    results['status'] = 'PASS' if all_pass else 'FAIL'

    return results


def create_sample_player_verification_table(
    df: pd.DataFrame,
    sample_players: Dict
) -> List[Dict]:
    """
    Create detailed verification table for sample players.
    """
    verification_tables = []

    for player_type, player_info in sample_players.items():
        player_id = player_info['plyr_guid']

        # Verify for yds statistic (primary)
        verification = verify_player_rolling_features(df, player_id, 'yds')

        table_data = {
            'player_type': player_type,
            'player_info': player_info,
            'verification': verification
        }
        verification_tables.append(table_data)

        # Also verify tgt and rec for this player
        for stat in ['tgt', 'rec']:
            extra_verification = verify_player_rolling_features(df, player_id, stat)
            table_data[f'{stat}_verification'] = extra_verification

    return verification_tables


def run_full_verification(parquet_path: str) -> Dict:
    """
    Run the complete verification procedure and generate report.
    """
    report = {
        'file_path': parquet_path,
        'verification_status': 'UNKNOWN'
    }

    # Step 1: Load data
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    df = load_parquet_file(parquet_path)

    # Step 2: Generate summary
    print("\n" + "="*70)
    print("STEP 2: DATASET SUMMARY")
    print("="*70)
    summary = generate_dataset_summary(df)
    report['dataset_summary'] = summary
    print(f"Rows: {summary['row_count']:,}")
    print(f"Columns: {summary['column_count']}")
    print(f"Unique Players: {summary['unique_players']}")
    print(f"Seasons: {summary['seasons']}")
    print(f"Rolling Features: {len(summary['rolling_feature_columns'])}")

    # Step 3: Select sample players
    print("\n" + "="*70)
    print("STEP 3: SELECTING SAMPLE PLAYERS")
    print("="*70)
    sample_players = select_sample_players(df)
    report['sample_players'] = sample_players
    for ptype, pinfo in sample_players.items():
        print(f"\n{ptype}:")
        print(f"  Name: {pinfo.get('player_name', 'Unknown')}")
        print(f"  ID: {pinfo['plyr_guid']}")
        print(f"  Seasons: {pinfo['seasons']}")
        print(f"  Total Games: {pinfo['total_games']}")

    # Step 4: Manual verification for sample players
    print("\n" + "="*70)
    print("STEP 4: MANUAL ROLLING CALCULATION VERIFICATION")
    print("="*70)
    sample_verifications = create_sample_player_verification_table(df, sample_players)
    report['sample_verifications'] = sample_verifications

    for sv in sample_verifications:
        player_type = sv['player_type']
        player_name = sv['player_info'].get('player_name', 'Unknown')
        print(f"\n{player_type} ({player_name}):")

        for v in sv['verification']['verifications']:
            status = v['status']
            feature = v['feature_name']
            if status == 'PASS':
                print(f"  {feature}: PASS ({v['total_games']} games)")
            elif status == 'FAIL':
                print(f"  {feature}: FAIL ({v['mismatches']} mismatches out of {v['total_games']} games)")
            else:
                print(f"  {feature}: {status} - {v.get('reason', '')}")

    # Step 5: Edge case verification
    print("\n" + "="*70)
    print("STEP 5: EDGE CASE VERIFICATION")
    print("="*70)

    # 5.1 First career game nulls
    print("\n5.1 First Career Game Null Check")
    first_game_check = verify_first_career_game_nulls(df)
    report['first_career_game_check'] = first_game_check
    print(f"  Status: {first_game_check['status']}")
    print(f"  First career games: {first_game_check.get('first_career_games_count', 'N/A')}")
    print(f"  Columns checked: {first_game_check.get('rolling_columns_checked', 'N/A')}")
    if first_game_check['status'] == 'FAIL':
        print(f"  Violations: {len(first_game_check['violations'])}")
        for v in first_game_check['violations'][:3]:
            print(f"    - {v['column']}: {v['non_null_count']} non-null values")

    # 5.2 Cross-season carryover
    print("\n5.2 Cross-Season Carryover Check")
    carryover_check = verify_cross_season_carryover(df)
    report['cross_season_carryover_check'] = carryover_check
    print(f"  Status: {carryover_check['status']}")
    print(f"  Carryover rows: {carryover_check.get('carryover_rows_count', 'N/A')}")
    print(f"  Avg % with data: {carryover_check.get('avg_pct_with_data', 'N/A')}%")

    # 5.3 Imputed value exclusion
    print("\n5.3 Imputed Value Exclusion Check")
    imputed_check = verify_imputed_value_exclusion(df)
    report['imputed_value_check'] = imputed_check
    print(f"  Status: {imputed_check['status']}")
    for c in imputed_check['checks'][:3]:
        print(f"    {c['feature_name']}: {c['status']}")

    # 5.4 No future leakage
    print("\n5.4 No Future Leakage Check")
    leakage_check = verify_no_future_leakage(df)
    report['no_future_leakage_check'] = leakage_check
    print(f"  Status: {leakage_check['status']}")
    for c in leakage_check['checks']:
        print(f"    {c['feature_name']}: {c['status']}")

    # Step 6: Efficiency feature verification
    print("\n" + "="*70)
    print("STEP 6: EFFICIENCY FEATURE VERIFICATION")
    print("="*70)
    efficiency_check = verify_efficiency_features(df)
    report['efficiency_feature_check'] = efficiency_check
    print(f"  Status: {efficiency_check['status']}")
    for c in efficiency_check['checks']:
        if c['status'] != 'SKIPPED':
            print(f"    {c['feature']}: {c['status']} ({c['matches']}/{c['total_rows']} match)")

    # Determine overall status
    all_checks = [
        first_game_check['status'],
        carryover_check['status'],
        imputed_check['status'],
        leakage_check['status'],
        efficiency_check['status']
    ]

    # Also include sample player verifications
    for sv in sample_verifications:
        for v in sv['verification']['verifications']:
            if v['status'] not in ['SKIPPED']:
                all_checks.append(v['status'])

    passed = sum(1 for c in all_checks if c == 'PASS')
    failed = sum(1 for c in all_checks if c == 'FAIL')
    skipped = sum(1 for c in all_checks if c == 'SKIPPED')

    report['overall_summary'] = {
        'total_checks': len(all_checks),
        'passed': passed,
        'failed': failed,
        'skipped': skipped
    }

    if failed == 0:
        report['verification_status'] = 'PASS'
    else:
        report['verification_status'] = 'FAIL'

    # Print final summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print(f"\nOverall Status: {report['verification_status']}")
    print(f"Checks Passed: {passed}")
    print(f"Checks Failed: {failed}")
    print(f"Checks Skipped: {skipped}")

    if failed > 0:
        print("\nDISCREPANCIES FOUND:")
        if first_game_check['status'] == 'FAIL':
            print("  - First career game null check failed")
        if carryover_check['status'] == 'FAIL':
            print("  - Cross-season carryover check failed")
        if imputed_check['status'] == 'FAIL':
            print("  - Imputed value exclusion check failed")
        if leakage_check['status'] == 'FAIL':
            print("  - Future leakage check failed")
        if efficiency_check['status'] == 'FAIL':
            print("  - Efficiency feature formula check failed")

    return report


def format_report_as_json(report: Dict) -> str:
    """Format the verification report as JSON."""
    # Convert to serializable format
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj

    serializable_report = make_serializable(report)
    return json.dumps(serializable_report, indent=2)


if __name__ == "__main__":
    # File path
    parquet_path = r"C:\Users\nocap\Desktop\code\NFL_ml\models\rf_wr_yds\data\processed\nfl_wr_features_v1_20251206_163420.parquet"

    # Run verification
    report = run_full_verification(parquet_path)

    # Save report
    output_path = Path(parquet_path).parent / "verification_report.json"
    with open(output_path, 'w') as f:
        f.write(format_report_as_json(report))
    print(f"\nDetailed report saved to: {output_path}")
