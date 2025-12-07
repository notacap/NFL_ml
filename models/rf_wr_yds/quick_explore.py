"""Quick data exploration script."""
import pandas as pd
import numpy as np

parquet_path = r"C:\Users\nocap\Desktop\code\NFL_ml\models\rf_wr_yds\data\processed\nfl_wr_features_v1_20251206_163420.parquet"
output_path = r"C:\Users\nocap\Desktop\code\NFL_ml\models\rf_wr_yds\data\processed\data_exploration.txt"

df = pd.read_parquet(parquet_path)

results = []
results.append("="*80)
results.append("DATA EXPLORATION")
results.append("="*80)

# Basic info
results.append(f"\nShape: {df.shape}")
results.append(f"\nColumns ({len(df.columns)}):")
for i, col in enumerate(df.columns):
    dtype = str(df[col].dtype)
    null_pct = df[col].isna().mean() * 100
    results.append(f"  {i+1}. {col} ({dtype}) - {null_pct:.1f}% null")

# Sample rows
results.append("\n" + "="*80)
results.append("SAMPLE DATA (first 5 rows, key columns)")
results.append("="*80)

key_cols = ['plyr_guid', 'plyr_display_name', 'season_id', 'week_id',
            'game_seq_num', 'career_game_seq_num', 'is_season_carryover',
            'plyr_gm_rec_yds', 'roll_3g_yds', 'roll_5g_yds']
key_cols = [c for c in key_cols if c in df.columns]
results.append(df[key_cols].head(20).to_string())

# Find a specific player to trace
results.append("\n" + "="*80)
results.append("SAMPLE PLAYER TRACE")
results.append("="*80)

# Get a multi-season player
player_seasons = df.groupby('plyr_guid')['season_id'].nunique()
multi_season = player_seasons[player_seasons >= 2].index[0]
player_df = df[df['plyr_guid'] == multi_season].sort_values(['season_id', 'week_id'])

player_name = player_df['plyr_display_name'].iloc[0] if 'plyr_display_name' in player_df.columns else 'Unknown'
results.append(f"\nPlayer: {player_name}")
results.append(f"ID: {multi_season}")

trace_cols = ['season_id', 'week_id', 'game_seq_num', 'career_game_seq_num',
              'is_season_carryover', 'plyr_gm_rec_yds', 'roll_3g_yds', 'roll_5g_yds']
trace_cols = [c for c in trace_cols if c in player_df.columns]
results.append(player_df[trace_cols].head(15).to_string())

# Check -999 values
results.append("\n" + "="*80)
results.append("-999 VALUE CHECK")
results.append("="*80)

for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        count_999 = (df[col] == -999).sum()
        if count_999 > 0:
            results.append(f"  {col}: {count_999} rows with -999 ({count_999/len(df)*100:.2f}%)")

# Write output
output_text = '\n'.join(results)
with open(output_path, 'w') as f:
    f.write(output_text)

print(f"Exploration written to: {output_path}")
print(output_text[:5000])  # Print first 5000 chars
