"""
Execute the verification and capture detailed output.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Constants
IMPUTATION_SENTINEL = -999
CROSS_SEASON_PLAYER_ID = 'plyr_guid'
ROLLING_WINDOWS = [3, 5]

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
    parquet_path = r"C:\Users\nocap\Desktop\code\NFL_ml\models\rf_wr_yds\data\processed\nfl_wr_features_v1_20251206_163420.parquet"

    print("="*80)
    print("NFL WR ROLLING FEATURE VERIFICATION REPORT")
    print("="*80)
    print(f"\nFile: {parquet_path}\n")

    # Load data
    print("Loading parquet file...")
    df = pd.read_parquet(parquet_path)

    # =========================================================================
    # SECTION 1: DATASET SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("1. DATASET SUMMARY")
    print("="*80)

    print(f"Row count: {len(df):,}")
    print(f"Column count: {len(df.columns)}")
    print(f"Unique players (plyr_guid): {df['plyr_guid'].nunique() if 'plyr_guid' in df.columns else 'N/A'}")
    print(f"Seasons covered: {sorted(df['season_id'].unique().tolist()) if 'season_id' in df.columns else 'N/A'}")

    rolling_cols = [c for c in df.columns if c.startswith('roll_')]
    print(f"Rolling feature columns: {len(rolling_cols)}")
    print(f"  Sample: {rolling_cols[:5]}")

    # =========================================================================
    # SECTION 2: SAMPLE PLAYER SELECTION
    # =========================================================================
    print("\n" + "="*80)
    print("2. SAMPLE PLAYER SELECTION")
    print("="*80)

    # Multi-season veteran
    player_seasons = df.groupby(CROSS_SEASON_PLAYER_ID)['season_id'].nunique()
    multi_season_players = player_seasons[player_seasons >= 2].index.tolist()

    if multi_season_players:
        player_games = df[df[CROSS_SEASON_PLAYER_ID].isin(multi_season_players)].groupby(CROSS_SEASON_PLAYER_ID).size()
        veteran_id = player_games.nlargest(5).index[0]
        veteran_df = df[df[CROSS_SEASON_PLAYER_ID] == veteran_id]
        veteran_name = veteran_df['plyr_display_name'].iloc[0] if 'plyr_display_name' in veteran_df.columns else 'Unknown'
        veteran_seasons = sorted(veteran_df['season_id'].unique().tolist())
        print(f"\nMulti-season veteran: {veteran_name}")
        print(f"  ID: {veteran_id}")
        print(f"  Seasons: {veteran_seasons}")
        print(f"  Total games: {len(veteran_df)}")

    # Single-season player
    single_season_players = player_seasons[player_seasons == 1].index.tolist()
    if single_season_players:
        player_games = df[df[CROSS_SEASON_PLAYER_ID].isin(single_season_players)].groupby(CROSS_SEASON_PLAYER_ID).size()
        mid_range = player_games[(player_games >= 5) & (player_games <= 10)]
        single_id = mid_range.index[0] if len(mid_range) > 0 else player_games.index[0]
        single_df = df[df[CROSS_SEASON_PLAYER_ID] == single_id]
        single_name = single_df['plyr_display_name'].iloc[0] if 'plyr_display_name' in single_df.columns else 'Unknown'
        print(f"\nSingle-season player: {single_name}")
        print(f"  ID: {single_id}")
        print(f"  Season: {single_df['season_id'].iloc[0]}")
        print(f"  Total games: {len(single_df)}")

    # Player with imputed values
    imputed_player_id = None
    for col in FEATURES_WITH_IMPUTATION:
        if col in df.columns:
            imputed_mask = df[col] == IMPUTATION_SENTINEL
            if imputed_mask.any():
                players_with_imputed = df[imputed_mask][CROSS_SEASON_PLAYER_ID].unique()
                player_games = df[df[CROSS_SEASON_PLAYER_ID].isin(players_with_imputed)].groupby(CROSS_SEASON_PLAYER_ID).size()
                imputed_player_id = player_games.nlargest(1).index[0]
                imputed_df = df[df[CROSS_SEASON_PLAYER_ID] == imputed_player_id]
                imputed_name = imputed_df['plyr_display_name'].iloc[0] if 'plyr_display_name' in imputed_df.columns else 'Unknown'
                imputed_count = (imputed_df[col] == IMPUTATION_SENTINEL).sum()
                print(f"\nPlayer with -999 values: {imputed_name}")
                print(f"  ID: {imputed_player_id}")
                print(f"  Imputed column: {col}")
                print(f"  Imputed count: {imputed_count}")
                break

    # =========================================================================
    # SECTION 3: MANUAL ROLLING CALCULATION VERIFICATION
    # =========================================================================
    print("\n" + "="*80)
    print("3. MANUAL ROLLING CALCULATION VERIFICATION")
    print("="*80)

    def verify_player_rolling(player_id, player_name, stat_name='yds'):
        """Manually verify rolling calculations for a player."""
        source_col = SOURCE_COLUMN_MAP.get(stat_name, f'plyr_gm_rec_{stat_name}')
        has_imputation = source_col in FEATURES_WITH_IMPUTATION

        player_df = df[df[CROSS_SEASON_PLAYER_ID] == player_id].copy()
        player_df = player_df.sort_values(['season_id', 'week_id']).reset_index(drop=True)

        print(f"\n--- {player_name} ({player_id[:20]}...) ---")
        print(f"Statistic: {stat_name} (source: {source_col})")
        print(f"Has imputation: {has_imputation}")
        print(f"Games: {len(player_df)}")

        for window in ROLLING_WINDOWS:
            feature_name = f'roll_{window}g_{stat_name}'

            if source_col not in player_df.columns or feature_name not in player_df.columns:
                print(f"  {feature_name}: SKIPPED (column not found)")
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
                print(f"  {feature_name}: PASS ({matches}/{len(player_df)} games match)")
            else:
                print(f"  {feature_name}: FAIL ({len(mismatches)} mismatches)")
                for idx, manual, file in mismatches[:3]:
                    print(f"    Game {idx}: manual={manual}, file={file}")

        # Show sample data
        print(f"\n  Sample data (first 8 games):")
        display_cols = ['season_id', 'week_id']
        if 'game_seq_num' in player_df.columns:
            display_cols.append('game_seq_num')
        if 'career_game_seq_num' in player_df.columns:
            display_cols.append('career_game_seq_num')
        display_cols.extend([source_col, f'roll_3g_{stat_name}', f'roll_5g_{stat_name}'])
        display_cols = [c for c in display_cols if c in player_df.columns]

        sample = player_df[display_cols].head(8)
        print(sample.to_string(index=False))

        return len(mismatches) == 0

    # Verify veteran
    if 'veteran_id' in dir():
        verify_player_rolling(veteran_id, veteran_name, 'yds')
        verify_player_rolling(veteran_id, veteran_name, 'tgt')

    # Verify single-season
    if 'single_id' in dir():
        verify_player_rolling(single_id, single_name, 'yds')

    # =========================================================================
    # SECTION 4: EDGE CASE VERIFICATION
    # =========================================================================
    print("\n" + "="*80)
    print("4. EDGE CASE VERIFICATION")
    print("="*80)

    # 4.1 First Career Game Null Check
    print("\n4.1 First Career Game Null Check")
    print("    (All rolling features should be NaN for career_game_seq_num == 1)")

    if 'career_game_seq_num' in df.columns:
        first_career = df[df['career_game_seq_num'] == 1]
        rolling_cols = [c for c in df.columns if c.startswith('roll_')]

        violations = []
        for col in rolling_cols:
            if col in first_career.columns:
                non_null = first_career[col].notna().sum()
                if non_null > 0:
                    violations.append((col, non_null))

        print(f"    First career games: {len(first_career)}")
        print(f"    Rolling columns checked: {len(rolling_cols)}")

        if len(violations) == 0:
            print(f"    Status: PASS (all rolling values are NaN)")
        else:
            print(f"    Status: FAIL ({len(violations)} columns have non-null values)")
            for col, count in violations[:5]:
                print(f"      - {col}: {count} non-null values")
    else:
        print("    Status: SKIPPED (career_game_seq_num not found)")

    # 4.2 Cross-Season Carryover Check
    print("\n4.2 Cross-Season Carryover Check")
    print("    (Returning players should have rolling values in early season games)")

    if 'is_season_carryover' in df.columns:
        carryover = df[df['is_season_carryover'] == True]

        print(f"    Carryover rows: {len(carryover)}")

        for col in ['roll_3g_yds', 'roll_5g_yds', 'roll_3g_tgt', 'roll_5g_tgt']:
            if col in carryover.columns:
                pct_with_data = carryover[col].notna().mean() * 100
                print(f"    {col}: {pct_with_data:.1f}% have data")

        avg_pct = carryover[[c for c in ['roll_3g_yds', 'roll_5g_yds'] if c in carryover.columns]].notna().mean().mean() * 100
        if avg_pct > 90:
            print(f"    Status: PASS ({avg_pct:.1f}% avg have data)")
        else:
            print(f"    Status: FAIL (only {avg_pct:.1f}% have data)")
    else:
        print("    Status: SKIPPED (is_season_carryover not found)")

    # 4.3 Imputed Value Exclusion Check
    print("\n4.3 Imputed Value (-999) Exclusion Check")
    print("    (-999 sentinel values should be excluded from rolling means)")

    if imputed_player_id:
        # Get player with imputed values
        test_col = 'plyr_gm_rec_first_dwn'
        stat_name = 'first_dwn'

        player_df = df[df[CROSS_SEASON_PLAYER_ID] == imputed_player_id].copy()
        player_df = player_df.sort_values(['season_id', 'week_id']).reset_index(drop=True)

        if test_col in player_df.columns:
            # Manual calc excluding -999
            stat_series = player_df[test_col].replace(IMPUTATION_SENTINEL, np.nan)
            shifted = stat_series.shift(1)
            manual_roll_3 = shifted.rolling(window=3, min_periods=1).mean()

            feature_col = f'roll_3g_{stat_name}'
            if feature_col in player_df.columns:
                file_values = player_df[feature_col].values

                matches = 0
                for i in range(len(player_df)):
                    m = manual_roll_3.iloc[i]
                    f = file_values[i]
                    if (pd.isna(m) and pd.isna(f)) or (not pd.isna(m) and not pd.isna(f) and np.isclose(m, f, rtol=1e-5)):
                        matches += 1

                if matches == len(player_df):
                    print(f"    Status: PASS (manual calc with -999 excluded matches file)")
                else:
                    print(f"    Status: FAIL ({len(player_df) - matches} mismatches)")

                # Show example where -999 was excluded
                imputed_rows = player_df[player_df[test_col] == IMPUTATION_SENTINEL].head(2)
                print(f"\n    Example rows with -999 in {test_col}:")
                show_cols = ['season_id', 'week_id', test_col, feature_col]
                print(imputed_rows[show_cols].to_string(index=False))
    else:
        print("    Status: SKIPPED (no player with -999 values found)")

    # 4.4 No Future Leakage Check
    print("\n4.4 No Future Leakage Check")
    print("    (Current game stats must NOT be in current row's rolling average)")

    # Pick a random player with enough games
    test_player = df.groupby(CROSS_SEASON_PLAYER_ID).size().nlargest(1).index[0]
    player_df = df[df[CROSS_SEASON_PLAYER_ID] == test_player].sort_values(['season_id', 'week_id']).reset_index(drop=True)

    source_col = 'plyr_gm_rec_yds'
    feature_col = 'roll_3g_yds'

    if source_col in player_df.columns and feature_col in player_df.columns:
        leakage_detected = False

        for i in range(3, min(10, len(player_df))):  # Check a few games
            current_stat = player_df[source_col].iloc[i]
            file_rolling = player_df[feature_col].iloc[i]

            if pd.isna(file_rolling):
                continue

            # Expected: mean of games i-3, i-2, i-1 (NOT including i)
            prior_stats = player_df[source_col].iloc[i-3:i]
            expected = prior_stats.mean()

            # Wrong: mean including current game
            wrong_mean = player_df[source_col].iloc[i-3:i+1].mean()

            if np.isclose(file_rolling, wrong_mean, rtol=1e-5) and not np.isclose(file_rolling, expected, rtol=1e-5):
                leakage_detected = True
                print(f"    LEAKAGE at game {i}: file={file_rolling:.2f}, expected={expected:.2f}, wrong={wrong_mean:.2f}")

        if not leakage_detected:
            print(f"    Status: PASS (no leakage detected)")
        else:
            print(f"    Status: FAIL (leakage detected)")

    # =========================================================================
    # SECTION 5: EFFICIENCY FEATURE VERIFICATION
    # =========================================================================
    print("\n" + "="*80)
    print("5. EFFICIENCY FEATURE VERIFICATION")
    print("="*80)

    formulas = [
        ('roll_3g_yds_per_tgt', 'roll_3g_yds', 'roll_3g_tgt'),
        ('roll_5g_yds_per_tgt', 'roll_5g_yds', 'roll_5g_tgt'),
        ('roll_3g_yds_per_rec', 'roll_3g_yds', 'roll_3g_rec'),
        ('roll_5g_yds_per_rec', 'roll_5g_yds', 'roll_5g_rec'),
        ('roll_3g_catch_rate', 'roll_3g_rec', 'roll_3g_tgt'),
        ('roll_5g_catch_rate', 'roll_5g_rec', 'roll_5g_tgt'),
    ]

    all_pass = True
    for feature, num, denom in formulas:
        if feature not in df.columns or num not in df.columns or denom not in df.columns:
            print(f"  {feature}: SKIPPED (columns not found)")
            continue

        # Calculate expected
        expected = np.where(df[denom] > 0, df[num] / df[denom], np.nan)
        file_vals = df[feature].values

        matches = 0
        for i in range(len(df)):
            e, f = expected[i], file_vals[i]
            if (pd.isna(e) and pd.isna(f)) or (not pd.isna(e) and not pd.isna(f) and np.isclose(e, f, rtol=1e-5)):
                matches += 1

        if matches == len(df):
            print(f"  {feature}: PASS ({num}/{denom})")
        else:
            print(f"  {feature}: FAIL ({len(df) - matches} mismatches)")
            all_pass = False

    if all_pass:
        print("\n  Overall Efficiency Status: PASS")
    else:
        print("\n  Overall Efficiency Status: FAIL")

    # =========================================================================
    # SECTION 6: FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("6. VERIFICATION SUMMARY")
    print("="*80)

    print("""
    Based on the verification checks above:

    1. Dataset Summary: Loaded and explored successfully
    2. Sample Players: Multi-season veteran, single-season, and imputed value players selected
    3. Manual Calculation: Compared manual rolling calculations to file values
    4. Edge Cases:
       - First career game nulls: Verified
       - Cross-season carryover: Verified
       - Imputed value exclusion: Verified
       - No future leakage: Verified
    5. Efficiency Features: Formula verification complete

    See detailed output above for PASS/FAIL status of each check.
    """)

if __name__ == "__main__":
    main()
