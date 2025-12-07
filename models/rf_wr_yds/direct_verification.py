"""
Direct verification script that outputs results to a text file.
Run this script and read the output file for the verification report.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# File paths
parquet_path = r"C:\Users\nocap\Desktop\code\NFL_ml\models\rf_wr_yds\data\processed\nfl_wr_features_v1_20251206_163420.parquet"
output_path = r"C:\Users\nocap\Desktop\code\NFL_ml\models\rf_wr_yds\data\processed\verification_results.txt"

# Constants
IMPUTATION_SENTINEL = -999
CROSS_SEASON_PLAYER_ID = 'plyr_guid'

FEATURES_WITH_IMPUTATION = [
    'plyr_gm_rec_first_dwn',
    'plyr_gm_rec_tgt_share',
    'plyr_gm_rec_epa',
    'plyr_gm_rec_ay_share',
    'plyr_gm_rec_wopr',
    'plyr_gm_rec_racr'
]

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

def main():
    results = []
    results.append("="*80)
    results.append("NFL WR ROLLING FEATURE VERIFICATION REPORT")
    results.append("="*80)
    results.append(f"\nFile: {parquet_path}")
    results.append(f"Generated: {pd.Timestamp.now()}\n")

    # Load data
    results.append("Loading parquet file...")
    df = pd.read_parquet(parquet_path)
    results.append(f"Loaded successfully.\n")

    # ==========================================================================
    # SECTION 1: DATASET SUMMARY
    # ==========================================================================
    results.append("="*80)
    results.append("1. DATASET SUMMARY")
    results.append("="*80)
    results.append(f"Row count: {len(df):,}")
    results.append(f"Column count: {len(df.columns)}")
    results.append(f"Unique players: {df[CROSS_SEASON_PLAYER_ID].nunique()}")
    results.append(f"Seasons: {sorted(df['season_id'].unique().tolist())}")

    rolling_cols = [c for c in df.columns if c.startswith('roll_')]
    results.append(f"Rolling features: {len(rolling_cols)}")
    results.append(f"Rolling columns: {rolling_cols}")

    # ==========================================================================
    # SECTION 2: SAMPLE PLAYER SELECTION
    # ==========================================================================
    results.append("\n" + "="*80)
    results.append("2. SAMPLE PLAYER SELECTION")
    results.append("="*80)

    # Find multi-season veteran
    player_seasons = df.groupby(CROSS_SEASON_PLAYER_ID)['season_id'].nunique()
    multi_season = player_seasons[player_seasons >= 2].index.tolist()

    veteran_id = None
    veteran_name = None
    if multi_season:
        player_games = df[df[CROSS_SEASON_PLAYER_ID].isin(multi_season)].groupby(CROSS_SEASON_PLAYER_ID).size()
        veteran_id = player_games.nlargest(3).index[0]
        veteran_df = df[df[CROSS_SEASON_PLAYER_ID] == veteran_id]
        veteran_name = veteran_df['plyr_display_name'].iloc[0] if 'plyr_display_name' in veteran_df.columns else 'Unknown'
        veteran_seasons = sorted(veteran_df['season_id'].unique().tolist())
        results.append(f"\nMulti-season veteran selected: {veteran_name}")
        results.append(f"  Player ID: {veteran_id}")
        results.append(f"  Seasons: {veteran_seasons}")
        results.append(f"  Total games: {len(veteran_df)}")

    # Find single-season player
    single_season = player_seasons[player_seasons == 1].index.tolist()
    single_id = None
    single_name = None
    if single_season:
        player_games = df[df[CROSS_SEASON_PLAYER_ID].isin(single_season)].groupby(CROSS_SEASON_PLAYER_ID).size()
        mid = player_games[(player_games >= 5) & (player_games <= 10)]
        single_id = mid.index[0] if len(mid) > 0 else player_games.index[0]
        single_df = df[df[CROSS_SEASON_PLAYER_ID] == single_id]
        single_name = single_df['plyr_display_name'].iloc[0] if 'plyr_display_name' in single_df.columns else 'Unknown'
        results.append(f"\nSingle-season player selected: {single_name}")
        results.append(f"  Player ID: {single_id}")
        results.append(f"  Season: {single_df['season_id'].iloc[0]}")
        results.append(f"  Total games: {len(single_df)}")

    # ==========================================================================
    # SECTION 3: MANUAL VERIFICATION FOR VETERAN
    # ==========================================================================
    results.append("\n" + "="*80)
    results.append("3. MANUAL ROLLING CALCULATION VERIFICATION")
    results.append("="*80)

    if veteran_id:
        results.append(f"\nVerifying: {veteran_name}")

        player_df = df[df[CROSS_SEASON_PLAYER_ID] == veteran_id].copy()
        player_df = player_df.sort_values(['season_id', 'week_id']).reset_index(drop=True)

        for stat_name in ['yds', 'tgt', 'rec']:
            source_col = SOURCE_COLUMN_MAP[stat_name]
            has_imputation = source_col in FEATURES_WITH_IMPUTATION

            results.append(f"\n--- Statistic: {stat_name} (source: {source_col}) ---")

            for window in [3, 5]:
                feature_name = f'roll_{window}g_{stat_name}'

                if source_col not in player_df.columns or feature_name not in player_df.columns:
                    results.append(f"  {feature_name}: SKIPPED - column not found")
                    continue

                # Manual calculation
                stat_series = player_df[source_col].copy()
                if has_imputation:
                    stat_series = stat_series.replace(IMPUTATION_SENTINEL, np.nan)
                shifted = stat_series.shift(1)
                manual_rolling = shifted.rolling(window=window, min_periods=1).mean()

                file_values = player_df[feature_name].values

                # Compare
                matches = 0
                mismatches = []
                for i in range(len(player_df)):
                    manual_val = manual_rolling.iloc[i]
                    file_val = file_values[i]

                    if pd.isna(manual_val) and pd.isna(file_val):
                        matches += 1
                    elif pd.isna(manual_val) or pd.isna(file_val):
                        mismatches.append((i, manual_val, file_val))
                    elif np.isclose(manual_val, file_val, rtol=1e-5):
                        matches += 1
                    else:
                        mismatches.append((i, manual_val, file_val))

                if len(mismatches) == 0:
                    results.append(f"  {feature_name}: PASS ({matches}/{len(player_df)} match)")
                else:
                    results.append(f"  {feature_name}: FAIL ({len(mismatches)} mismatches)")
                    for idx, manual, file in mismatches[:3]:
                        results.append(f"    Game {idx}: manual={manual}, file={file}")

        # Show sample data
        results.append(f"\n  Sample data (first 10 games):")
        display_cols = ['season_id', 'week_id', 'game_seq_num', 'career_game_seq_num',
                        'plyr_gm_rec_yds', 'roll_3g_yds', 'roll_5g_yds']
        display_cols = [c for c in display_cols if c in player_df.columns]
        sample = player_df[display_cols].head(10)
        results.append(sample.to_string())

    # ==========================================================================
    # SECTION 4: EDGE CASE VERIFICATION
    # ==========================================================================
    results.append("\n" + "="*80)
    results.append("4. EDGE CASE VERIFICATION")
    results.append("="*80)

    # 4.1 First Career Game Null Check
    results.append("\n4.1 First Career Game Null Check")
    results.append("    Rule: All rolling features should be NaN for career_game_seq_num == 1")

    if 'career_game_seq_num' in df.columns:
        first_career = df[df['career_game_seq_num'] == 1]
        rolling_cols = [c for c in df.columns if c.startswith('roll_')]

        violations = []
        for col in rolling_cols:
            if col in first_career.columns:
                non_null = first_career[col].notna().sum()
                if non_null > 0:
                    violations.append((col, non_null))

        results.append(f"    First career games count: {len(first_career)}")
        results.append(f"    Rolling columns checked: {len(rolling_cols)}")

        if len(violations) == 0:
            results.append(f"    STATUS: PASS - All rolling values are NaN for first career games")
        else:
            results.append(f"    STATUS: FAIL - {len(violations)} columns have non-null values")
            for col, count in violations[:5]:
                results.append(f"      - {col}: {count} non-null values")
    else:
        results.append("    STATUS: SKIPPED - career_game_seq_num not found")

    # 4.2 Cross-Season Carryover Check
    results.append("\n4.2 Cross-Season Carryover Check")
    results.append("    Rule: Returning players should have rolling values in early season games")

    if 'is_season_carryover' in df.columns:
        carryover = df[df['is_season_carryover'] == True]
        results.append(f"    Carryover rows: {len(carryover)}")

        check_cols = ['roll_3g_yds', 'roll_5g_yds', 'roll_3g_tgt', 'roll_5g_tgt']
        for col in check_cols:
            if col in carryover.columns:
                pct = carryover[col].notna().mean() * 100
                results.append(f"    {col}: {pct:.1f}% have data")

        avg_pct = carryover[[c for c in check_cols if c in carryover.columns]].notna().mean().mean() * 100
        if avg_pct > 90:
            results.append(f"    STATUS: PASS ({avg_pct:.1f}% average have data)")
        else:
            results.append(f"    STATUS: FAIL (only {avg_pct:.1f}% have data)")
    else:
        results.append("    STATUS: SKIPPED - is_season_carryover not found")

    # 4.3 Imputed Value Exclusion
    results.append("\n4.3 Imputed Value (-999) Exclusion Check")
    results.append("    Rule: -999 sentinel values should be excluded from rolling means")

    imputed_check_passed = True
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

        test_player = players_with_imputed[0]
        player_df = df[df[CROSS_SEASON_PLAYER_ID] == test_player].sort_values(['season_id', 'week_id']).reset_index(drop=True)

        for window in [3, 5]:
            feature_name = f'roll_{window}g_{stat_name}'
            if feature_name not in player_df.columns:
                continue

            # Manual calc with -999 excluded
            stat_series = player_df[source_col].replace(IMPUTATION_SENTINEL, np.nan)
            shifted = stat_series.shift(1)
            manual_rolling = shifted.rolling(window=window, min_periods=1).mean()

            file_values = player_df[feature_name].values

            matches = 0
            for i in range(len(player_df)):
                m = manual_rolling.iloc[i]
                f = file_values[i]
                if (pd.isna(m) and pd.isna(f)) or (not pd.isna(m) and not pd.isna(f) and np.isclose(m, f, rtol=1e-5)):
                    matches += 1

            if matches == len(player_df):
                results.append(f"    {feature_name}: PASS")
            else:
                results.append(f"    {feature_name}: FAIL ({len(player_df) - matches} mismatches)")
                imputed_check_passed = False

        break  # Only check one column

    if imputed_check_passed:
        results.append("    STATUS: PASS - Imputed values correctly excluded")
    else:
        results.append("    STATUS: FAIL - Imputed values not correctly excluded")

    # 4.4 No Future Leakage Check
    results.append("\n4.4 No Future Leakage Check")
    results.append("    Rule: Current game stats must NOT be in current row's rolling average")

    test_player = df.groupby(CROSS_SEASON_PLAYER_ID).size().nlargest(1).index[0]
    player_df = df[df[CROSS_SEASON_PLAYER_ID] == test_player].sort_values(['season_id', 'week_id']).reset_index(drop=True)

    source_col = 'plyr_gm_rec_yds'
    leakage_detected = False

    for window in [3, 5]:
        feature_col = f'roll_{window}g_yds'

        if source_col not in player_df.columns or feature_col not in player_df.columns:
            continue

        for i in range(window, min(15, len(player_df))):
            current_stat = player_df[source_col].iloc[i]
            file_rolling = player_df[feature_col].iloc[i]

            if pd.isna(file_rolling):
                continue

            # Expected: mean of prior window games (NOT including current)
            prior_stats = player_df[source_col].iloc[max(0, i-window):i]
            expected = prior_stats.mean()

            # Wrong: mean including current game
            wrong_mean = player_df[source_col].iloc[max(0, i-window):i+1].mean()

            if np.isclose(file_rolling, wrong_mean, rtol=1e-5) and not np.isclose(file_rolling, expected, rtol=1e-5):
                leakage_detected = True
                results.append(f"    LEAKAGE at game {i}: file={file_rolling:.2f}, expected={expected:.2f}")

    if not leakage_detected:
        results.append("    STATUS: PASS - No future leakage detected")
    else:
        results.append("    STATUS: FAIL - Future leakage detected")

    # ==========================================================================
    # SECTION 5: EFFICIENCY FEATURE VERIFICATION
    # ==========================================================================
    results.append("\n" + "="*80)
    results.append("5. EFFICIENCY FEATURE VERIFICATION")
    results.append("="*80)

    formulas = [
        ('roll_3g_yds_per_tgt', 'roll_3g_yds', 'roll_3g_tgt'),
        ('roll_5g_yds_per_tgt', 'roll_5g_yds', 'roll_5g_tgt'),
        ('roll_3g_yds_per_rec', 'roll_3g_yds', 'roll_3g_rec'),
        ('roll_5g_yds_per_rec', 'roll_5g_yds', 'roll_5g_rec'),
        ('roll_3g_catch_rate', 'roll_3g_rec', 'roll_3g_tgt'),
        ('roll_5g_catch_rate', 'roll_5g_rec', 'roll_5g_tgt'),
    ]

    all_efficiency_pass = True
    for feature, num, denom in formulas:
        if feature not in df.columns or num not in df.columns or denom not in df.columns:
            results.append(f"  {feature}: SKIPPED")
            continue

        expected = np.where(df[denom] > 0, df[num] / df[denom], np.nan)
        file_vals = df[feature].values

        matches = 0
        for i in range(len(df)):
            e, f = expected[i], file_vals[i]
            if (pd.isna(e) and pd.isna(f)) or (not pd.isna(e) and not pd.isna(f) and np.isclose(e, f, rtol=1e-5)):
                matches += 1

        if matches == len(df):
            results.append(f"  {feature}: PASS ({num}/{denom})")
        else:
            results.append(f"  {feature}: FAIL ({len(df) - matches} mismatches)")
            all_efficiency_pass = False

    if all_efficiency_pass:
        results.append("\n  STATUS: PASS - All efficiency features match formulas")
    else:
        results.append("\n  STATUS: FAIL - Some efficiency features have mismatches")

    # ==========================================================================
    # SECTION 6: FINAL SUMMARY
    # ==========================================================================
    results.append("\n" + "="*80)
    results.append("6. VERIFICATION SUMMARY")
    results.append("="*80)

    # Count passes and fails
    pass_count = sum(1 for r in results if 'STATUS: PASS' in r)
    fail_count = sum(1 for r in results if 'STATUS: FAIL' in r)

    results.append(f"\nChecks Passed: {pass_count}")
    results.append(f"Checks Failed: {fail_count}")

    if fail_count == 0:
        results.append("\nOVERALL VERDICT: PASS")
        results.append("All rolling feature calculations verified successfully.")
    else:
        results.append("\nOVERALL VERDICT: FAIL")
        results.append("Some rolling feature calculations have discrepancies.")

    # Write results to file
    output_text = '\n'.join(results)
    with open(output_path, 'w') as f:
        f.write(output_text)

    print(f"Results written to: {output_path}")
    print(output_text)

if __name__ == "__main__":
    main()
