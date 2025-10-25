"""
Deep dive into players 14586 and 14588 to understand the longest > total anomaly.
"""

import pandas as pd
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


def investigate_player_detailed(player_id, season, week):
    """
    Show all columns for a player in a specific week to understand the data.
    """
    print(f"\n{'='*80}")
    print(f"PLAYER {int(player_id)} - SEASON {season}, WEEK {week}")
    print(f"{'='*80}\n")

    game_df = load_data("plyr_gm/plyr_gm_rec", season, week)

    if game_df.empty:
        print(f"No game data found")
        return

    player_data = game_df[game_df['plyr_id'] == player_id]

    if player_data.empty:
        print(f"Player not found in week {week}")
        return

    # Show all relevant columns
    relevant_cols = [
        'plyr_id', 'plyr_gm_rec_tgt', 'plyr_gm_rec',
        'plyr_gm_rec_yds', 'plyr_gm_rec_lng',
        'plyr_gm_rec_yac', 'plyr_gm_rec_aybc'
    ]

    available_cols = [col for col in relevant_cols if col in player_data.columns]

    print("GAME-LEVEL DATA:")
    for col in available_cols:
        value = player_data[col].iloc[0]
        print(f"  {col}: {value}")

    # Calculate what the other reception(s) must have been
    total_yards = player_data['plyr_gm_rec_yds'].iloc[0]
    longest = player_data['plyr_gm_rec_lng'].iloc[0]
    receptions = player_data['plyr_gm_rec'].iloc[0]

    print(f"\nCALCULATION:")
    print(f"  Total yards: {total_yards}")
    print(f"  Longest reception: {longest}")
    print(f"  Number of receptions: {receptions}")

    if receptions == 2:
        other_reception = total_yards - longest
        print(f"  Other reception must be: {total_yards} - {longest} = {other_reception} yards")

        if other_reception < 0:
            print(f"\n  EXPLANATION: Player caught one pass for {longest} yards,")
            print(f"               and another for {other_reception} yards (negative!)")
            print(f"               Total = {total_yards} yards")
            print(f"               This is why longest ({longest}) > total ({total_yards})")
    elif receptions > 2:
        remaining_yards = total_yards - longest
        print(f"  Other {int(receptions - 1)} receptions combine for: {remaining_yards} yards")

        if remaining_yards < 0:
            print(f"\n  EXPLANATION: Player's longest reception was {longest} yards,")
            print(f"               but other receptions totaled {remaining_yards} yards (negative!)")
            print(f"               Total = {total_yards} yards")


if __name__ == "__main__":
    print("="*80)
    print("DETAILED INVESTIGATION OF ANOMALOUS PLAYERS")
    print("="*80)

    # Player 14586: Week 5, 2 receptions, 6 yards, longest = 11
    investigate_player_detailed(14586, 2024, 5)

    # Player 14588: Week 3, 2 receptions, 0 yards, longest = 2
    investigate_player_detailed(14588, 2024, 3)

    print("\n\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print("="*80)
