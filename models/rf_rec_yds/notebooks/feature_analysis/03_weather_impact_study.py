"""
Weather & Environmental Impact Analysis for NFL Receiver Production
===================================================================

Objective: Quantify weather impacts on passing game and identify
thresholds where temperature, wind, and precipitation significantly
affect receiver production.

Analysis includes:
1. Temperature effects (cold, moderate, warm, hot)
2. Wind speed thresholds (10, 15, 20, 25 mph)
3. Precipitation impact (rain vs snow)
4. Stadium type (indoor vs outdoor)
5. Composite weather difficulty score
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
BASE_PATH = Path(r'C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\clean')
OUTPUT_VIZ = Path(r'C:\Users\nocap\Desktop\code\NFL_ml\outputs\visualizations')
OUTPUT_DATA = Path(r'C:\Users\nocap\Desktop\code\NFL_ml\outputs\predictions')

print("=" * 80)
print("WEATHER & ENVIRONMENTAL IMPACT ANALYSIS")
print("=" * 80)

# ============================================================================
# SECTION 1: Load Data
# ============================================================================
print("\n[1/5] Loading data...")

def load_parquet_dataset(path, seasons=[2022, 2023, 2024]):
    """Load partitioned parquet dataset"""
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

# Load datasets
weather = load_parquet_dataset(BASE_PATH / 'gm_info' / 'nfl_gm_weather')
game_info = load_parquet_dataset(BASE_PATH / 'gm_info' / 'nfl_game')
game_betting = load_parquet_dataset(BASE_PATH / 'gm_info' / 'nfl_game_info')
plyr_gm_rec = pd.read_parquet(BASE_PATH / 'plyr_gm' / 'plyr_gm_rec')
team_info = pd.read_parquet(BASE_PATH / 'nfl_team.parquet')

print(f"  - Weather data: {weather.shape}")
print(f"  - Game info: {game_info.shape}")
print(f"  - Game betting: {game_betting.shape}")
print(f"  - Player receiving: {plyr_gm_rec.shape}")

# ============================================================================
# SECTION 2: Prepare Data
# ============================================================================
print("\n[2/5] Preparing dataset...")

# Aggregate receiver production
rec_production = plyr_gm_rec.groupby(['team_id', 'game_id', 'season_id', 'week_id']).agg({
    'plyr_gm_rec_yds': 'sum',
    'plyr_gm_rec_tgt': 'sum',
    'plyr_gm_rec': 'sum'
}).reset_index()
rec_production.columns = ['team_id', 'game_id', 'season_id', 'week_id',
                          'total_rec_yds', 'total_targets', 'total_receptions']

# Merge weather with receiver production
analysis_df = rec_production.merge(weather, on=['game_id', 'season_id', 'week_id'], how='left')

# Merge stadium roof info from game_betting
if 'stadium_roof' in game_betting.columns:
    analysis_df = analysis_df.merge(
        game_betting[['game_id', 'stadium_roof']],
        on='game_id',
        how='left'
    )

# Add team info for indoor stadium flag
analysis_df = analysis_df.merge(
    team_info[['team_id', 'indoor_stadium', 'warm_climate']],
    on='team_id',
    how='left'
)

print(f"  - Analysis dataset: {analysis_df.shape}")
print(f"  - Weather columns available: {[c for c in analysis_df.columns if 'temp' in c.lower() or 'wind' in c.lower() or 'precip' in c.lower()]}")

# ============================================================================
# SECTION 3: Temperature Analysis
# ============================================================================
print("\n[3/5] Analyzing temperature effects...")

# Check for temperature column
temp_cols = [c for c in analysis_df.columns if 'temp' in c.lower()]
if temp_cols:
    temp_col = temp_cols[0]
    print(f"  Using temperature column: {temp_col}")

    # Create temperature bins
    analysis_df['temp_category'] = pd.cut(
        analysis_df[temp_col],
        bins=[-np.inf, 32, 50, 70, 85, np.inf],
        labels=['Very Cold (<32°F)', 'Cold (32-50°F)', 'Moderate (50-70°F)',
                'Warm (70-85°F)', 'Hot (>85°F)']
    )

    # Calculate stats by temperature category
    temp_stats = analysis_df.groupby('temp_category').agg({
        'total_rec_yds': ['mean', 'std', 'count']
    }).round(2)

    print("\nReceiver Production by Temperature:")
    print(temp_stats)

    # Statistical test: Cold vs Moderate
    if analysis_df['temp_category'].notna().any():
        cold_data = analysis_df[analysis_df[temp_col] < 40]['total_rec_yds'].dropna()
        moderate_data = analysis_df[(analysis_df[temp_col] >= 50) &
                                     (analysis_df[temp_col] <= 70)]['total_rec_yds'].dropna()

        if len(cold_data) > 0 and len(moderate_data) > 0:
            t_stat, p_value = ttest_ind(cold_data, moderate_data)
            print(f"\nCold (<40°F) vs Moderate (50-70°F):")
            print(f"  Cold avg: {cold_data.mean():.1f} yards")
            print(f"  Moderate avg: {moderate_data.mean():.1f} yards")
            print(f"  Difference: {cold_data.mean() - moderate_data.mean():.1f} yards")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")
else:
    print("  - No temperature data available")
    temp_col = None

# ============================================================================
# SECTION 4: Wind Analysis
# ============================================================================
print("\n[4/5] Analyzing wind effects...")

wind_cols = [c for c in analysis_df.columns if 'wind' in c.lower()]
if wind_cols:
    wind_col = wind_cols[0]
    print(f"  Using wind column: {wind_col}")

    # Create wind speed bins
    analysis_df['wind_category'] = pd.cut(
        analysis_df[wind_col],
        bins=[0, 5, 10, 15, 20, np.inf],
        labels=['Calm (0-5mph)', 'Light (5-10mph)', 'Moderate (10-15mph)',
                'Strong (15-20mph)', 'Very Strong (>20mph)']
    )

    # Calculate stats by wind category
    wind_stats = analysis_df.groupby('wind_category').agg({
        'total_rec_yds': ['mean', 'std', 'count']
    }).round(2)

    print("\nReceiver Production by Wind Speed:")
    print(wind_stats)

    # Test various wind thresholds
    wind_thresholds = [10, 15, 20, 25]
    wind_threshold_results = []

    for threshold in wind_thresholds:
        low_wind = analysis_df[analysis_df[wind_col] < threshold]['total_rec_yds'].dropna()
        high_wind = analysis_df[analysis_df[wind_col] >= threshold]['total_rec_yds'].dropna()

        if len(low_wind) > 10 and len(high_wind) > 10:
            t_stat, p_value = ttest_ind(low_wind, high_wind)
            effect = low_wind.mean() - high_wind.mean()

            wind_threshold_results.append({
                'threshold_mph': threshold,
                'low_wind_avg': low_wind.mean(),
                'high_wind_avg': high_wind.mean(),
                'effect_size_yards': effect,
                'p_value': p_value,
                'significant': 'Yes' if p_value < 0.05 else 'No',
                'n_low': len(low_wind),
                'n_high': len(high_wind)
            })

    wind_threshold_df = pd.DataFrame(wind_threshold_results)
    print("\nWind Threshold Analysis:")
    print(wind_threshold_df.to_string(index=False))
else:
    print("  - No wind data available")
    wind_col = None

# ============================================================================
# SECTION 5: Stadium Type & Precipitation
# ============================================================================
print("\n[5/5] Analyzing stadium and precipitation effects...")

# Indoor vs Outdoor
if 'stadium_roof' in analysis_df.columns:
    stadium_stats = analysis_df.groupby('stadium_roof').agg({
        'total_rec_yds': ['mean', 'std', 'count']
    }).round(2)

    print("\nReceiver Production by Stadium Type:")
    print(stadium_stats)

# Precipitation analysis
precip_cols = [c for c in analysis_df.columns if 'precip' in c.lower() or 'rain' in c.lower() or 'snow' in c.lower()]
if precip_cols:
    print(f"\nPrecipitation columns: {precip_cols}")

# ============================================================================
# SECTION 6: Create Visualizations
# ============================================================================
print("\nCreating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Weather Impact on Receiver Production', fontsize=16, fontweight='bold')

# Plot 1: Temperature Effect
if temp_col and 'temp_category' in analysis_df.columns:
    ax1 = axes[0, 0]
    temp_data = [analysis_df[analysis_df['temp_category'] == cat]['total_rec_yds'].dropna()
                 for cat in analysis_df['temp_category'].cat.categories
                 if (analysis_df['temp_category'] == cat).any()]
    labels = [cat for cat in analysis_df['temp_category'].cat.categories
              if (analysis_df['temp_category'] == cat).any()]

    bp1 = ax1.boxplot(temp_data, labels=labels, patch_artist=True,
                      medianprops=dict(color='red', linewidth=2))
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
    ax1.set_ylabel('Total Receiving Yards', fontsize=11)
    ax1.set_title('Temperature Impact', fontsize=12)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
else:
    axes[0, 0].text(0.5, 0.5, 'No Temperature Data', ha='center', va='center', fontsize=14)
    axes[0, 0].axis('off')

# Plot 2: Wind Speed Effect
if wind_col and 'wind_category' in analysis_df.columns:
    ax2 = axes[0, 1]
    wind_data = [analysis_df[analysis_df['wind_category'] == cat]['total_rec_yds'].dropna()
                 for cat in analysis_df['wind_category'].cat.categories
                 if (analysis_df['wind_category'] == cat).any()]
    wind_labels = [cat for cat in analysis_df['wind_category'].cat.categories
                   if (analysis_df['wind_category'] == cat).any()]

    bp2 = ax2.boxplot(wind_data, labels=wind_labels, patch_artist=True,
                      medianprops=dict(color='red', linewidth=2))
    for patch in bp2['boxes']:
        patch.set_facecolor('lightcoral')
    ax2.set_ylabel('Total Receiving Yards', fontsize=11)
    ax2.set_title('Wind Speed Impact', fontsize=12)
    ax2.set_xticklabels(wind_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
else:
    axes[0, 1].text(0.5, 0.5, 'No Wind Data', ha='center', va='center', fontsize=14)
    axes[0, 1].axis('off')

# Plot 3: Stadium Roof Type
if 'stadium_roof' in analysis_df.columns:
    ax3 = axes[1, 0]
    roof_types = analysis_df['stadium_roof'].dropna().unique()
    roof_means = [analysis_df[analysis_df['stadium_roof'] == rt]['total_rec_yds'].mean()
                  for rt in roof_types]
    roof_stds = [analysis_df[analysis_df['stadium_roof'] == rt]['total_rec_yds'].std()
                 for rt in roof_types]

    ax3.bar(range(len(roof_types)), roof_means, yerr=roof_stds, capsize=5,
            color='steelblue', alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(roof_types)))
    ax3.set_xticklabels(roof_types, rotation=45, ha='right')
    ax3.set_ylabel('Average Receiving Yards', fontsize=11)
    ax3.set_title('Stadium Roof Type Impact', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')

    for i, (mean, std) in enumerate(zip(roof_means, roof_stds)):
        ax3.text(i, mean + std + 5, f'{mean:.0f}', ha='center', fontsize=10, fontweight='bold')
else:
    axes[1, 0].text(0.5, 0.5, 'No Stadium Data', ha='center', va='center', fontsize=14)
    axes[1, 0].axis('off')

# Plot 4: Wind Threshold Analysis
if wind_col and len(wind_threshold_results) > 0:
    ax4 = axes[1, 1]
    thresholds = [r['threshold_mph'] for r in wind_threshold_results]
    effects = [r['effect_size_yards'] for r in wind_threshold_results]
    significant = [r['significant'] == 'Yes' for r in wind_threshold_results]

    colors = ['green' if sig else 'gray' for sig in significant]
    ax4.bar(range(len(thresholds)), effects, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_xticks(range(len(thresholds)))
    ax4.set_xticklabels([f'{t} mph' for t in thresholds])
    ax4.set_xlabel('Wind Speed Threshold', fontsize=11)
    ax4.set_ylabel('Effect Size (Low Wind - High Wind)', fontsize=11)
    ax4.set_title('Wind Threshold Impact\n(Green = Significant)', fontsize=12)
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4.grid(True, alpha=0.3, axis='y')

    for i, (thresh, effect) in enumerate(zip(thresholds, effects)):
        ax4.text(i, effect + 2, f'{effect:.1f}', ha='center', fontsize=10, fontweight='bold')
else:
    axes[1, 1].text(0.5, 0.5, 'No Wind Threshold Data', ha='center', va='center', fontsize=14)
    axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_VIZ / 'weather_impact_analysis.png', dpi=300, bbox_inches='tight')
print(f"  - Saved: weather_impact_analysis.png")
plt.close()

# ============================================================================
# Save Weather Thresholds
# ============================================================================
if wind_col and len(wind_threshold_results) > 0:
    wind_threshold_df.to_csv(OUTPUT_DATA / 'weather_thresholds.csv', index=False)
    print(f"  - Saved: weather_thresholds.csv")

print("\n" + "=" * 80)
print("KEY FINDINGS:")
print("=" * 80)

if temp_col:
    print("\n1. TEMPERATURE EFFECTS:")
    print(f"   - Temperature column analyzed: {temp_col}")
    if 'temp_stats' in locals():
        print(f"   - Categories analyzed: {len(temp_stats)} temperature ranges")

if wind_col and len(wind_threshold_results) > 0:
    print("\n2. WIND SPEED THRESHOLDS:")
    significant_thresholds = wind_threshold_df[wind_threshold_df['significant'] == 'Yes']
    if len(significant_thresholds) > 0:
        best_threshold = significant_thresholds.iloc[0]
        print(f"   - Optimal threshold: {best_threshold['threshold_mph']} mph")
        print(f"   - Effect size: {best_threshold['effect_size_yards']:.1f} yards")
        print(f"   - p-value: {best_threshold['p_value']:.4f}")
    else:
        print(f"   - No significant wind thresholds found")
    print(f"   - Thresholds tested: {wind_thresholds}")

if 'stadium_roof' in analysis_df.columns:
    print("\n3. STADIUM TYPE:")
    if 'stadium_stats' in locals():
        print(f"   - Stadium types analyzed: {len(stadium_stats)} types")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)
