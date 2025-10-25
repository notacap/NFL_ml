"""
Investigate players where longest reception exceeds total receiving yards.

Players to investigate: 13873, 14479, 14586, 14588 (2024 season)
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


def investigate_player(player_id, season=2024):
    """
    Investigate a specific player's longest reception vs total yards discrepancy.
    """
    print(f"\n{'='*80}")
    print(f"PLAYER {int(player_id)} - SEASON {season}")
    print(f"{'='*80}\n")

    # Load week 17 cumulative data
    cum_df = load_data("plyr_szn/plyr_rec", season, 17)

    if cum_df.empty:
        print(f"No cumulative data found for season {season}, week 17")
        return

    player_cum = cum_df[cum_df['plyr_id'] == player_id]

    if player_cum.empty:
        print(f"Player {int(player_id)} not found in cumulative data")
        return

    # Get cumulative stats
    total_yards = player_cum['plyr_rec_yds'].iloc[0]
    longest = player_cum['plyr_rec_lng'].iloc[0]
    receptions = player_cum['plyr_rec'].iloc[0]
    targets = player_cum['plyr_rec_tgt'].iloc[0]

    print(f"CUMULATIVE STATS (through Week 17):")
    print(f"  Total Yards: {total_yards}")
    print(f"  Longest Reception: {longest}")
    print(f"  Total Receptions: {receptions}")
    print(f"  Total Targets: {targets}")
    print(f"  ISSUE: Longest ({longest}) > Total ({total_yards}) by {longest - total_yards} yards")

    # Now look at game-level data for all weeks
    print(f"\n\nGAME-BY-GAME BREAKDOWN:")
    print(f"{'-'*80}")

    all_games = []
    for week in range(1, 18):
        game_df = load_data("plyr_gm/plyr_gm_rec", season, week)

        if not game_df.empty:
            player_games = game_df[game_df['plyr_id'] == player_id]

            if not player_games.empty:
                for _, game in player_games.iterrows():
                    all_games.append({
                        'week': week,
                        'targets': game.get('plyr_gm_rec_tgt', 0),
                        'receptions': game.get('plyr_gm_rec', 0),
                        'yards': game.get('plyr_gm_rec_yds', 0),
                        'longest': game.get('plyr_gm_rec_lng', 0)
                    })

    if not all_games:
        print("No game-level data found for this player")
        return

    # Display game-by-game
    print(f"{'Week':<6} {'Tgt':<6} {'Rec':<6} {'Yards':<8} {'Long':<8} {'Running Total Yards':<20}")
    print(f"{'-'*80}")

    running_total = 0
    max_long = 0

    for game in all_games:
        running_total += game['yards']
        max_long = max(max_long, game['longest'])

        print(f"{game['week']:<6} {int(game['targets']):<6} {int(game['receptions']):<6} "
              f"{game['yards']:<8.0f} {game['longest']:<8.0f} {running_total:<20.0f}")

    print(f"{'-'*80}")
    print(f"TOTALS:      {sum(g['targets'] for g in all_games):<6.0f} "
          f"{sum(g['receptions'] for g in all_games):<6.0f} "
          f"{sum(g['yards'] for g in all_games):<8.0f} "
          f"{max(g['longest'] for g in all_games):<8.0f}")

    # Analysis
    print(f"\n\nANALYSIS:")
    total_game_yards = sum(g['yards'] for g in all_games)
    max_game_long = max(g['longest'] for g in all_games)

    print(f"  Sum of game yards: {total_game_yards}")
    print(f"  Cumulative total yards: {total_yards}")
    print(f"  Match: {'YES' if abs(total_game_yards - total_yards) < 0.01 else 'NO'}")
    print()
    print(f"  Max of game longest: {max_game_long}")
    print(f"  Cumulative longest: {longest}")
    print(f"  Match: {'YES' if abs(max_game_long - longest) < 0.01 else 'NO'}")

    # Check for negative yards
    negative_yard_games = [g for g in all_games if g['yards'] < 0]
    if negative_yard_games:
        print(f"\n  WARNING: Found {len(negative_yard_games)} game(s) with NEGATIVE yards:")
        for g in negative_yard_games:
            print(f"    Week {g['week']}: {g['yards']} yards, {g['receptions']} rec, longest={g['longest']}")

    # Explain the issue
    print(f"\n\nEXPLANATION:")
    if negative_yard_games and max_game_long > total_game_yards:
        print(f"  This player had negative yards in {len(negative_yard_games)} game(s).")
        print(f"  When a player has a positive longest reception (e.g., {max_game_long} yards)")
        print(f"  but also has games with negative receiving yards,")
        print(f"  the cumulative total yards ({total_game_yards}) can be LESS than")
        print(f"  the longest single reception ({max_game_long}).")
        print(f"\n  This is LEGITIMATE NFL DATA - the script is working correctly.")
    else:
        print(f"  Need to investigate further - no obvious negative yards causing this.")


if __name__ == "__main__":
    print("="*80)
    print("INVESTIGATING LONGEST RECEPTION > TOTAL YARDS ANOMALY")
    print("="*80)

    # Players identified in test report
    player_ids = [13873, 14479, 14586, 14588]

    for player_id in player_ids:
        investigate_player(player_id, season=2024)

    print("\n\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print("="*80)
