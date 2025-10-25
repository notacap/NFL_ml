"""
Investigate specific test failures in receiving.py cumulative stats.

This script examines:
1. Monotonic progression violations (air yards and yards decreasing)
2. Week 1 first downs mismatch between game and cumulative data
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_ROOT = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw"

def load_data(table_path, season, week):
    """Load parquet data for a given table, season, and week."""
    path = Path(DATA_ROOT) / table_path / f"season={season}" / f"week={week}"

    if not path.exists():
        return pd.DataFrame()

    parquet_files = list(path.glob("*.parquet"))
    if not parquet_files:
        return pd.DataFrame()

    dfs = [pd.read_parquet(f) for f in parquet_files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def investigate_monotonic_violations():
    """
    Investigate players where air yards or receiving yards decreased week-over-week.
    """
    print("=" * 80)
    print("INVESTIGATING MONOTONIC PROGRESSION VIOLATIONS")
    print("=" * 80)

    for season in [2022, 2023, 2024]:
        print(f"\n\n{'='*80}")
        print(f"SEASON {season}")
        print(f"{'='*80}\n")

        # Load weeks 1-5 to find early violations
        weeks_data = {}
        for week in range(1, 6):
            df = load_data("plyr_szn/plyr_rec", season, week)
            if not df.empty:
                weeks_data[week] = df

        if len(weeks_data) < 2:
            print(f"Not enough weeks for season {season}")
            continue

        # Check consecutive weeks
        for week in range(1, 5):
            if week not in weeks_data or week + 1 not in weeks_data:
                continue

            df1 = weeks_data[week]
            df2 = weeks_data[week + 1]

            # Merge on player_id
            merged = df1.merge(
                df2,
                on='plyr_id',
                how='inner',
                suffixes=('_w1', '_w2')
            )

            # Check air yards (AYBC)
            aybc_decreased = merged[merged['plyr_rec_aybc_w2'] < merged['plyr_rec_aybc_w1']]

            # Check receiving yards
            yds_decreased = merged[merged['plyr_rec_yds_w2'] < merged['plyr_rec_yds_w1']]

            if not aybc_decreased.empty or not yds_decreased.empty:
                print(f"\nWeek {week} -> {week+1}:")

                if not aybc_decreased.empty:
                    print(f"  Air Yards (AYBC) decreased: {len(aybc_decreased)} players")

                    # Show first 3 examples
                    for idx, row in aybc_decreased.head(3).iterrows():
                        plyr_id = row['plyr_id']
                        aybc_w1 = row['plyr_rec_aybc_w1']
                        aybc_w2 = row['plyr_rec_aybc_w2']
                        diff = aybc_w2 - aybc_w1

                        print(f"    Player {plyr_id}: {aybc_w1:.1f} -> {aybc_w2:.1f} (change: {diff:.1f})")

                        # Check if this is a negative air yards situation
                        if aybc_w2 < 0:
                            print(f"      WARNING: NEGATIVE AIR YARDS in week {week+1}!")

                if not yds_decreased.empty:
                    print(f"  Receiving Yards decreased: {len(yds_decreased)} players")

                    # Show first 3 examples
                    for idx, row in yds_decreased.head(3).iterrows():
                        plyr_id = row['plyr_id']
                        yds_w1 = row['plyr_rec_yds_w1']
                        yds_w2 = row['plyr_rec_yds_w2']
                        diff = yds_w2 - yds_w1

                        print(f"    Player {plyr_id}: {yds_w1:.1f} -> {yds_w2:.1f} (change: {diff:.1f})")

                        # Check if this is negative yards situation
                        if yds_w2 < 0 or yds_w1 < 0:
                            print(f"      WARNING: NEGATIVE YARDS!")

                        # Show the underlying game data for this player
                        print(f"      Checking game-level data for player {plyr_id}...")
                        for w in [week, week + 1]:
                            game_df = load_data("plyr_gm/plyr_gm_rec", season, w)
                            if not game_df.empty:
                                player_games = game_df[game_df['plyr_id'] == plyr_id]
                                if not player_games.empty:
                                    total_yds = player_games['plyr_gm_rec_yds'].sum()
                                    print(f"        Week {w} game data: {len(player_games)} games, {total_yds} yards")


def investigate_week1_firstdowns_mismatch():
    """
    Investigate discrepancies between week 1 game data and week 1 cumulative data for first downs.
    """
    print("\n\n" + "=" * 80)
    print("INVESTIGATING WEEK 1 FIRST DOWNS MISMATCH")
    print("=" * 80)

    for season in [2022, 2023, 2024]:
        print(f"\n\n{'='*80}")
        print(f"SEASON {season}")
        print(f"{'='*80}\n")

        # Load week 1 game data
        game_df = load_data("plyr_gm/plyr_gm_rec", season, 1)
        if game_df.empty:
            print(f"No game data for season {season}, week 1")
            continue

        # Load week 1 cumulative data
        cum_df = load_data("plyr_szn/plyr_rec", season, 1)
        if cum_df.empty:
            print(f"No cumulative data for season {season}, week 1")
            continue

        # Merge on plyr_id
        merged = game_df[['plyr_id', 'plyr_gm_rec_first_dwn']].merge(
            cum_df[['plyr_id', 'plyr_rec_first_dwn']],
            on='plyr_id',
            how='inner'
        )

        # Find mismatches
        mismatches = merged[merged['plyr_gm_rec_first_dwn'] != merged['plyr_rec_first_dwn']]

        if not mismatches.empty:
            print(f"Found {len(mismatches)} players with first downs mismatch")
            print("\nFirst 10 examples:")
            print(mismatches.head(10).to_string())

            # Check if there's a pattern
            print("\n\nAnalyzing pattern:")
            print(f"  Game data column type: {game_df['plyr_gm_rec_first_dwn'].dtype}")
            print(f"  Cumulative data column type: {cum_df['plyr_rec_first_dwn'].dtype}")
            print(f"  Game data has nulls: {game_df['plyr_gm_rec_first_dwn'].isna().any()}")
            print(f"  Cumulative data has nulls: {cum_df['plyr_rec_first_dwn'].isna().any()}")

            # Check if the issue is null handling
            game_nulls = game_df[game_df['plyr_gm_rec_first_dwn'].isna()]
            if not game_nulls.empty:
                print(f"\n  WARNING: Found {len(game_nulls)} players with NULL first downs in game data")
                print("  This might be the issue - checking how nulls are handled...")

                # Check if these nulls become 0 in cumulative
                for idx, row in mismatches.head(5).iterrows():
                    plyr_id = row['plyr_id']
                    game_val = row['plyr_gm_rec_first_dwn']
                    cum_val = row['plyr_rec_first_dwn']
                    print(f"\n    Player {plyr_id}:")
                    print(f"      Game data: {game_val} (is NaN: {pd.isna(game_val)})")
                    print(f"      Cumulative: {cum_val} (is NaN: {pd.isna(cum_val)})")
        else:
            print(f"OK: No mismatches found for season {season}")


if __name__ == "__main__":
    investigate_monotonic_violations()
    investigate_week1_firstdowns_mismatch()

    print("\n\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)
