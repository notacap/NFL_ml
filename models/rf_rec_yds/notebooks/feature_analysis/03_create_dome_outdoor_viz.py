"""Quick script to create dome vs outdoor comparison visualization"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
OUTPUT_VIZ = Path(r'C:\Users\nocap\Desktop\code\NFL_ml\outputs\visualizations')

# Load the analysis results from weather study
BASE_PATH = Path(r'C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\clean')

def load_parquet_dataset(path, seasons=[2022, 2023, 2024]):
    dfs = []
    for season in seasons:
        season_path = path / f'season={season}'
        if season_path.exists():
            try:
                df = pd.read_parquet(season_path)
                dfs.append(df)
            except:
                pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

game_betting = load_parquet_dataset(BASE_PATH / 'gm_info' / 'nfl_game_info')
plyr_gm_rec = pd.read_parquet(BASE_PATH / 'plyr_gm' / 'plyr_gm_rec')

rec_production = plyr_gm_rec.groupby(['team_id', 'game_id', 'season_id', 'week_id']).agg({
    'plyr_gm_rec_yds': 'sum'
}).reset_index()
rec_production.columns = ['team_id', 'game_id', 'season_id', 'week_id', 'total_rec_yds']

analysis_df = rec_production.merge(
    game_betting[['game_id', 'stadium_roof']],
    on='game_id',
    how='left'
)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Dome vs Outdoor Receiver Production', fontsize=16, fontweight='bold')

# Plot 1: Box plot comparison
ax1 = axes[0]
dome_data = analysis_df[analysis_df['stadium_roof'] == 'dome']['total_rec_yds'].dropna()
outdoor_data = analysis_df[analysis_df['stadium_roof'] == 'outdoors']['total_rec_yds'].dropna()

bp = ax1.boxplot([dome_data, outdoor_data], labels=['Dome', 'Outdoors'],
                 patch_artist=True, medianprops=dict(color='red', linewidth=2))
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightgreen')

ax1.set_ylabel('Total Receiving Yards', fontsize=12)
ax1.set_title('Distribution Comparison', fontsize=12)
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Bar comparison with error bars
ax2 = axes[1]
means = [dome_data.mean(), outdoor_data.mean()]
stds = [dome_data.std(), outdoor_data.std()]
bars = ax2.bar(['Dome', 'Outdoors'], means, yerr=stds, capsize=5,
               color=['lightblue', 'lightgreen'], alpha=0.7, edgecolor='black')

ax2.set_ylabel('Average Receiving Yards', fontsize=12)
ax2.set_title(f'Mean Comparison\n(Dome Advantage: +{means[0] - means[1]:.1f} yards)', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

for i, (label, mean) in enumerate(zip(['Dome', 'Outdoors'], means)):
    ax2.text(i, mean + stds[i] + 5, f'{mean:.1f}', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_VIZ / 'dome_outdoor_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: dome_outdoor_comparison.png")
print(f"\nDome avg: {dome_data.mean():.1f} yards (n={len(dome_data)})")
print(f"Outdoor avg: {outdoor_data.mean():.1f} yards (n={len(outdoor_data)})")
print(f"Difference: +{dome_data.mean() - outdoor_data.mean():.1f} yards")
