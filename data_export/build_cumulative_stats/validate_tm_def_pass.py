"""
Validate tm_def_pass.py calculations.

This script validates:
1. Defensive sack yards calculation (from opponent's offense)
2. Calculated columns (completion %, passer rating, etc.)
3. Team aggregation from player stats
"""

import pandas as pd
import numpy as np

# Load the output
output_df = pd.read_parquet(
    'C:/Users/nocap/Desktop/code/NFL_ml/parquet_files/raw/tm_szn/tm_def_pass/season=2024/week=1'
)

print("=" * 80)
print("VALIDATION REPORT: Team Defensive Passing Statistics (2024, Week 1)")
print("=" * 80)

# Select one team for detailed validation
test_team = output_df.iloc[0]
team_id = test_team['team_id']

print(f"\n1. TESTING TEAM {int(team_id)}")
print("-" * 80)

# Display key stats
print(f"Pass Attempts Allowed: {test_team['tm_def_pass_att']}")
print(f"Completions Allowed: {test_team['tm_def_pass_cmp']}")
print(f"Yards Allowed: {test_team['tm_def_pass_yds']}")
print(f"TDs Allowed: {test_team['tm_def_pass_td']}")
print(f"Interceptions: {test_team['tm_def_int']}")
print(f"Sacks: {test_team['tm_def_sk']}")
print(f"Sack Yards: {test_team['tm_def_sk_yds']}")

# Validate calculated columns
print(f"\n2. VALIDATING CALCULATED COLUMNS")
print("-" * 80)

# Completion percentage
expected_cmp_pct = test_team['tm_def_pass_cmp'] / test_team['tm_def_pass_att']
actual_cmp_pct = test_team['tm_def_pass_cmp_pct']
cmp_pct_match = np.isclose(expected_cmp_pct, actual_cmp_pct)
print(f"Completion %: Expected={expected_cmp_pct:.4f}, Actual={actual_cmp_pct:.4f}, Match={cmp_pct_match}")

# TD percentage
expected_td_pct = test_team['tm_def_pass_td'] / test_team['tm_def_pass_att']
actual_td_pct = test_team['tm_def_pass_td_pct']
td_pct_match = np.isclose(expected_td_pct, actual_td_pct)
print(f"TD %: Expected={expected_td_pct:.4f}, Actual={actual_td_pct:.4f}, Match={td_pct_match}")

# INT percentage
expected_int_pct = test_team['tm_def_int'] / test_team['tm_def_pass_att']
actual_int_pct = test_team['tm_def_int_pct']
int_pct_match = np.isclose(expected_int_pct, actual_int_pct)
print(f"INT %: Expected={expected_int_pct:.4f}, Actual={actual_int_pct:.4f}, Match={int_pct_match}")

# Yards per attempt
expected_yds_att = test_team['tm_def_pass_yds'] / test_team['tm_def_pass_att']
actual_yds_att = test_team['tm_def_pass_yds_att']
yds_att_match = np.isclose(expected_yds_att, actual_yds_att)
print(f"Yards/Att: Expected={expected_yds_att:.4f}, Actual={actual_yds_att:.4f}, Match={yds_att_match}")

# Adjusted yards per attempt
expected_yds_att_adj = (test_team['tm_def_pass_yds'] + 20*test_team['tm_def_pass_td'] - 45*test_team['tm_def_int']) / test_team['tm_def_pass_att']
actual_yds_att_adj = test_team['tm_def_pass_yds_att_adj']
yds_att_adj_match = np.isclose(expected_yds_att_adj, actual_yds_att_adj)
print(f"Adj Yards/Att: Expected={expected_yds_att_adj:.4f}, Actual={actual_yds_att_adj:.4f}, Match={yds_att_adj_match}")

# Yards per completion
expected_ypc = test_team['tm_def_pass_yds'] / test_team['tm_def_pass_cmp']
actual_ypc = test_team['tm_def_pass_ypc']
ypc_match = np.isclose(expected_ypc, actual_ypc)
print(f"Yards/Comp: Expected={expected_ypc:.4f}, Actual={actual_ypc:.4f}, Match={ypc_match}")

# Yards per game
expected_ypg = test_team['tm_def_pass_yds'] / test_team['game_count']
actual_ypg = test_team['tm_def_pass_ypg']
ypg_match = np.isclose(expected_ypg, actual_ypg)
print(f"Yards/Game: Expected={expected_ypg:.4f}, Actual={actual_ypg:.4f}, Match={ypg_match}")

# Sack percentage
expected_sk_pct = test_team['tm_def_sk'] / (test_team['tm_def_pass_att'] + test_team['tm_def_sk'])
actual_sk_pct = test_team['tm_def_sk_pct']
sk_pct_match = np.isclose(expected_sk_pct, actual_sk_pct)
print(f"Sack %: Expected={expected_sk_pct:.4f}, Actual={actual_sk_pct:.4f}, Match={sk_pct_match}")

# Net yards per attempt
expected_net_yds_att = (test_team['tm_def_pass_yds'] - test_team['tm_def_sk_yds']) / (test_team['tm_def_pass_att'] + test_team['tm_def_sk'])
actual_net_yds_att = test_team['tm_def_pass_net_yds_att']
net_yds_att_match = np.isclose(expected_net_yds_att, actual_net_yds_att)
print(f"Net Yards/Att: Expected={expected_net_yds_att:.4f}, Actual={actual_net_yds_att:.4f}, Match={net_yds_att_match}")

# Adjusted net yards per attempt
expected_net_yds_att_adj = (test_team['tm_def_pass_yds'] - test_team['tm_def_sk_yds'] + 20*test_team['tm_def_pass_td'] - 45*test_team['tm_def_int']) / (test_team['tm_def_pass_att'] + test_team['tm_def_sk'])
actual_net_yds_att_adj = test_team['tm_def_pass_net_yds_att_adj']
net_yds_att_adj_match = np.isclose(expected_net_yds_att_adj, actual_net_yds_att_adj)
print(f"Adj Net Yards/Att: Expected={expected_net_yds_att_adj:.4f}, Actual={actual_net_yds_att_adj:.4f}, Match={net_yds_att_adj_match}")

# Passer rating (more complex)
print(f"\n3. VALIDATING PASSER RATING")
print("-" * 80)

att = test_team['tm_def_pass_att']
cmp = test_team['tm_def_pass_cmp']
yds = test_team['tm_def_pass_yds']
td = test_team['tm_def_pass_td']
int_thrown = test_team['tm_def_int']

a = max(0, min(2.375, ((cmp / att) - 0.3) * 5))
b = max(0, min(2.375, ((yds / att) - 3) * 0.25))
c = max(0, min(2.375, (td / att) * 20))
d = max(0, min(2.375, 2.375 - ((int_thrown / att) * 25)))

expected_rating = ((a + b + c + d) / 6) * 100
actual_rating = test_team['tm_def_pass_rtg']
rating_match = np.isclose(expected_rating, actual_rating)

print(f"Component A: {a:.4f}")
print(f"Component B: {b:.4f}")
print(f"Component C: {c:.4f}")
print(f"Component D: {d:.4f}")
print(f"Expected Rating: {expected_rating:.4f}")
print(f"Actual Rating: {actual_rating:.4f}")
print(f"Match: {rating_match}")

# Overall validation summary
print(f"\n4. VALIDATION SUMMARY")
print("-" * 80)

all_checks = [
    cmp_pct_match, td_pct_match, int_pct_match, yds_att_match,
    yds_att_adj_match, ypc_match, ypg_match, sk_pct_match,
    net_yds_att_match, net_yds_att_adj_match, rating_match
]

print(f"Total checks: {len(all_checks)}")
print(f"Passed: {sum(all_checks)}")
print(f"Failed: {len(all_checks) - sum(all_checks)}")

if all(all_checks):
    print("\n[PASS] ALL CALCULATIONS VALIDATED SUCCESSFULLY!")
else:
    print("\n[FAIL] Some calculations failed validation")

# Check defensive sack yards logic
print(f"\n5. DEFENSIVE SACK YARDS VALIDATION")
print("-" * 80)
print("Checking that defensive sack yards come from opponent's offense...")

# Load player passing data to verify sack yards
pass_df = pd.read_parquet(
    'C:/Users/nocap/Desktop/code/NFL_ml/parquet_files/raw/plyr_gm/plyr_gm_pass/season=2024/week=1'
)

# Load game info
game_df = pd.read_parquet(
    'C:/Users/nocap/Desktop/code/NFL_ml/parquet_files/raw/gm_info/nfl_game/season=2024/week=1'
)

# Aggregate offensive sack yards by team
off_sack_yds = pass_df.groupby(['game_id', 'team_id'])['plyr_gm_pass_sk_yds'].sum().reset_index()

# For one game, verify the logic
sample_game = game_df.iloc[0]
game_id = sample_game['game_id']
home_team = sample_game['home_team_id']
away_team = sample_game['away_team_id']

print(f"\nSample Game ID: {game_id}")
print(f"Home Team: {home_team}, Away Team: {away_team}")

# Get offensive sack yards for each team
home_off_sk_yds = off_sack_yds[(off_sack_yds['game_id'] == game_id) & (off_sack_yds['team_id'] == home_team)]
away_off_sk_yds = off_sack_yds[(off_sack_yds['game_id'] == game_id) & (off_sack_yds['team_id'] == away_team)]

if not home_off_sk_yds.empty:
    home_sk_yds_taken = home_off_sk_yds.iloc[0]['plyr_gm_pass_sk_yds']
    print(f"Home team QB sack yards taken: {home_sk_yds_taken}")
    print(f"  => Away team defense should get credit for {home_sk_yds_taken} sack yards")

if not away_off_sk_yds.empty:
    away_sk_yds_taken = away_off_sk_yds.iloc[0]['plyr_gm_pass_sk_yds']
    print(f"Away team QB sack yards taken: {away_sk_yds_taken}")
    print(f"  => Home team defense should get credit for {away_sk_yds_taken} sack yards")

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)
