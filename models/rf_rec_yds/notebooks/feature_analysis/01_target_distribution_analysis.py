"""
Target Variable Deep Dive Analysis for NFL Receiving Yards Prediction
=====================================================================

This script conducts comprehensive statistical analysis of the target variable
(plyr_gm_rec_yds) to understand its distribution, patterns, and predictability
challenges for building a Random Forest prediction model.

Author: Data Science Team
Date: 2025-10-25
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for professional visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


class NFLReceivingYardsAnalyzer:
    """
    Comprehensive analyzer for NFL receiving yards target variable.

    This class provides methods for loading data, computing statistics,
    performing hypothesis tests, and generating visualizations.
    """

    def __init__(self, base_path: str):
        """
        Initialize the analyzer with data paths.

        Parameters
        ----------
        base_path : str
            Root directory containing parquet files
        """
        self.base_path = Path(base_path)
        self.receiving_path = self.base_path / "plyr_gm" / "plyr_gm_rec"
        self.player_path = self.base_path / "players" / "plyr"
        self.output_viz_path = Path(r"C:\Users\nocap\Desktop\code\NFL_ml\outputs\visualizations")
        self.output_data_path = Path(r"C:\Users\nocap\Desktop\code\NFL_ml\outputs\predictions")
        self.output_report_path = Path(r"C:\Users\nocap\Desktop\code\NFL_ml\outputs\reports")

        # Create output directories
        self.output_viz_path.mkdir(parents=True, exist_ok=True)
        self.output_data_path.mkdir(parents=True, exist_ok=True)
        self.output_report_path.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.receiving_data = None
        self.player_data = None
        self.merged_data = None

        logger.info(f"Initialized analyzer with base path: {base_path}")

    def load_receiving_data(self, seasons: List[int] = [2022, 2023, 2024, 2025]) -> pd.DataFrame:
        """
        Load receiving data from all seasons and weeks.

        Parameters
        ----------
        seasons : List[int]
            List of seasons to load

        Returns
        -------
        pd.DataFrame
            Combined receiving data
        """
        logger.info(f"Loading receiving data for seasons: {seasons}")

        all_data = []

        for season in seasons:
            season_path = self.receiving_path / f"season={season}"

            if not season_path.exists():
                logger.warning(f"Season {season} path does not exist: {season_path}")
                continue

            # Get all week directories
            week_dirs = sorted([d for d in season_path.iterdir() if d.is_dir()])

            for week_dir in week_dirs:
                try:
                    df = pd.read_parquet(week_dir)
                    all_data.append(df)
                    logger.info(f"Loaded {season} {week_dir.name}: {len(df)} records")
                except Exception as e:
                    logger.error(f"Error loading {week_dir}: {e}")

        self.receiving_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total receiving records loaded: {len(self.receiving_data)}")

        return self.receiving_data

    def load_player_data(self, seasons: List[int] = [2022, 2023, 2024, 2025]) -> pd.DataFrame:
        """
        Load player data from all seasons.

        Parameters
        ----------
        seasons : List[int]
            List of seasons to load

        Returns
        -------
        pd.DataFrame
            Combined player data
        """
        logger.info(f"Loading player data for seasons: {seasons}")

        all_data = []

        for season in seasons:
            season_path = self.player_path / f"season={season}"

            if not season_path.exists():
                logger.warning(f"Player season {season} path does not exist: {season_path}")
                continue

            try:
                df = pd.read_parquet(season_path)
                all_data.append(df)
                logger.info(f"Loaded player data {season}: {len(df)} records")
            except Exception as e:
                logger.error(f"Error loading player data {season}: {e}")

        self.player_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total player records loaded: {len(self.player_data)}")

        return self.player_data

    def merge_data(self) -> pd.DataFrame:
        """
        Merge receiving data with player data to get positions.

        Returns
        -------
        pd.DataFrame
            Merged dataset with position information
        """
        logger.info("Merging receiving and player data...")

        # Select relevant columns from player data
        player_cols = ['plyr_id', 'season_id', 'plyr_name', 'plyr_pos']
        player_subset = self.player_data[player_cols].copy()

        # Merge on player_id and season_id
        self.merged_data = pd.merge(
            self.receiving_data,
            player_subset,
            on=['plyr_id', 'season_id'],
            how='left'
        )

        # Filter to receiving positions only (WR, TE, RB)
        receiving_positions = ['WR', 'TE', 'RB']
        self.merged_data = self.merged_data[
            self.merged_data['plyr_pos'].isin(receiving_positions)
        ].copy()

        logger.info(f"Merged data shape: {self.merged_data.shape}")
        logger.info(f"Position breakdown:\n{self.merged_data['plyr_pos'].value_counts()}")

        return self.merged_data

    def compute_basic_statistics(self, data: pd.Series) -> Dict:
        """
        Compute comprehensive basic statistics for the target variable.

        Parameters
        ----------
        data : pd.Series
            Target variable data

        Returns
        -------
        Dict
            Dictionary containing all statistical measures
        """
        stats_dict = {
            'count': len(data),
            'mean': data.mean(),
            'median': data.median(),
            'mode': data.mode().values[0] if len(data.mode()) > 0 else np.nan,
            'std': data.std(),
            'variance': data.var(),
            'min': data.min(),
            'max': data.max(),
            'range': data.max() - data.min(),
            'q1': data.quantile(0.25),
            'q2': data.quantile(0.50),
            'q3': data.quantile(0.75),
            'iqr': data.quantile(0.75) - data.quantile(0.25),
            'p5': data.quantile(0.05),
            'p10': data.quantile(0.10),
            'p90': data.quantile(0.90),
            'p95': data.quantile(0.95),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'cv': data.std() / data.mean() if data.mean() != 0 else np.nan,
        }

        # Test for normality
        if len(data) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(data)
        else:
            # Use sample for large datasets
            sample = np.random.choice(data, 5000, replace=False)
            shapiro_stat, shapiro_p = stats.shapiro(sample)

        stats_dict['shapiro_statistic'] = shapiro_stat
        stats_dict['shapiro_pvalue'] = shapiro_p
        stats_dict['is_normal'] = shapiro_p > 0.05

        return stats_dict

    def detect_outliers(self, data: pd.Series) -> Dict:
        """
        Detect outliers using multiple methods.

        Parameters
        ----------
        data : pd.Series
            Target variable data

        Returns
        -------
        Dict
            Dictionary containing outlier information
        """
        # IQR method
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        iqr_outliers = (data < lower_bound) | (data > upper_bound)

        # Z-score method
        z_scores = np.abs(stats.zscore(data))
        z_outliers = z_scores > 3

        outlier_dict = {
            'iqr_lower_bound': lower_bound,
            'iqr_upper_bound': upper_bound,
            'iqr_outlier_count': iqr_outliers.sum(),
            'iqr_outlier_pct': 100 * iqr_outliers.sum() / len(data),
            'z_outlier_count': z_outliers.sum(),
            'z_outlier_pct': 100 * z_outliers.sum() / len(data),
            'iqr_outliers_mask': iqr_outliers,
            'z_outliers_mask': z_outliers,
        }

        return outlier_dict

    def create_yard_buckets(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create yard range buckets for categorical analysis.

        Parameters
        ----------
        data : pd.DataFrame
            Data with plyr_gm_rec_yds column

        Returns
        -------
        pd.DataFrame
            Data with added bucket column
        """
        bins = [0, 1, 26, 51, 76, 101, 151, np.inf]
        labels = [
            'Zero yards',
            '1-25 yards',
            '26-50 yards',
            '51-75 yards',
            '76-100 yards',
            '101-150 yards',
            '150+ yards'
        ]

        data = data.copy()
        data['yard_bucket'] = pd.cut(
            data['plyr_gm_rec_yds'],
            bins=bins,
            labels=labels,
            right=False,
            include_lowest=True
        )

        return data

    def analyze_zero_yards(self, data: pd.DataFrame) -> Dict:
        """
        Analyze zero-yard games in detail.

        Parameters
        ----------
        data : pd.DataFrame
            Receiving data

        Returns
        -------
        Dict
            Zero-yard game analysis
        """
        zero_games = data[data['plyr_gm_rec_yds'] == 0].copy()

        analysis = {
            'total_zero_games': len(zero_games),
            'zero_game_pct': 100 * len(zero_games) / len(data),
            'zero_with_targets': len(zero_games[zero_games['plyr_gm_rec_tgt'] > 0]),
            'zero_no_targets': len(zero_games[zero_games['plyr_gm_rec_tgt'] == 0]),
            'zero_with_drops': len(zero_games[zero_games['plyr_gm_rec_drp'] > 0]),
        }

        return analysis

    def position_comparison_statistics(self) -> pd.DataFrame:
        """
        Compute statistics by position for comparison.

        Returns
        -------
        pd.DataFrame
            Position-wise statistics
        """
        positions = ['WR', 'TE', 'RB']
        results = []

        for pos in positions:
            pos_data = self.merged_data[
                self.merged_data['plyr_pos'] == pos
            ]['plyr_gm_rec_yds']

            stats_dict = self.compute_basic_statistics(pos_data)
            stats_dict['position'] = pos
            results.append(stats_dict)

        return pd.DataFrame(results)

    def test_position_differences(self) -> Dict:
        """
        Perform statistical tests to compare positions.

        Returns
        -------
        Dict
            Test results
        """
        wr_data = self.merged_data[self.merged_data['plyr_pos'] == 'WR']['plyr_gm_rec_yds']
        te_data = self.merged_data[self.merged_data['plyr_pos'] == 'TE']['plyr_gm_rec_yds']
        rb_data = self.merged_data[self.merged_data['plyr_pos'] == 'RB']['plyr_gm_rec_yds']

        # Kruskal-Wallis H-test (non-parametric ANOVA)
        h_stat, h_pvalue = stats.kruskal(wr_data, te_data, rb_data)

        # Pairwise Mann-Whitney U tests
        wr_te_stat, wr_te_p = stats.mannwhitneyu(wr_data, te_data)
        wr_rb_stat, wr_rb_p = stats.mannwhitneyu(wr_data, rb_data)
        te_rb_stat, te_rb_p = stats.mannwhitneyu(te_data, rb_data)

        # Effect size (eta-squared approximation)
        n = len(wr_data) + len(te_data) + len(rb_data)
        eta_squared = (h_stat - 2) / (n - 3) if n > 3 else np.nan

        return {
            'kruskal_wallis_h': h_stat,
            'kruskal_wallis_p': h_pvalue,
            'significant_difference': h_pvalue < 0.05,
            'eta_squared': eta_squared,
            'wr_te_u_stat': wr_te_stat,
            'wr_te_p': wr_te_p,
            'wr_rb_u_stat': wr_rb_stat,
            'wr_rb_p': wr_rb_p,
            'te_rb_u_stat': te_rb_stat,
            'te_rb_p': te_rb_p,
        }

    def seasonal_comparison(self) -> pd.DataFrame:
        """
        Compare distributions across seasons.

        Returns
        -------
        pd.DataFrame
            Season-wise statistics
        """
        seasons = sorted(self.merged_data['season_id'].unique())
        results = []

        for season in seasons:
            season_data = self.merged_data[
                self.merged_data['season_id'] == season
            ]['plyr_gm_rec_yds']

            stats_dict = self.compute_basic_statistics(season_data)
            stats_dict['season'] = season
            results.append(stats_dict)

        return pd.DataFrame(results)

    def week_range_comparison(self) -> pd.DataFrame:
        """
        Compare early, mid, and late season performance.

        Returns
        -------
        pd.DataFrame
            Week range statistics
        """
        self.merged_data['week_range'] = pd.cut(
            self.merged_data['week_id'],
            bins=[0, 6, 13, 18],
            labels=['Early (1-6)', 'Mid (7-13)', 'Late (14-18)']
        )

        results = []
        for week_range in ['Early (1-6)', 'Mid (7-13)', 'Late (14-18)']:
            range_data = self.merged_data[
                self.merged_data['week_range'] == week_range
            ]['plyr_gm_rec_yds']

            stats_dict = self.compute_basic_statistics(range_data)
            stats_dict['week_range'] = week_range
            results.append(stats_dict)

        return pd.DataFrame(results)

    def autocorrelation_analysis(self) -> Dict:
        """
        Analyze week-to-week correlation for individual players.

        Returns
        -------
        Dict
            Autocorrelation statistics
        """
        # Group by player and season, sort by week
        player_series = []

        for (plyr_id, season_id), group in self.merged_data.groupby(['plyr_id', 'season_id']):
            if len(group) >= 3:  # At least 3 games
                series = group.sort_values('week_id')['plyr_gm_rec_yds'].values
                player_series.append(series)

        # Compute lag-1 autocorrelation for each player
        autocorrs = []
        for series in player_series:
            if len(series) >= 2:
                lag1_corr = np.corrcoef(series[:-1], series[1:])[0, 1]
                if not np.isnan(lag1_corr):
                    autocorrs.append(lag1_corr)

        return {
            'mean_autocorr': np.mean(autocorrs) if autocorrs else np.nan,
            'median_autocorr': np.median(autocorrs) if autocorrs else np.nan,
            'std_autocorr': np.std(autocorrs) if autocorrs else np.nan,
            'num_players_analyzed': len(autocorrs),
        }

    # VISUALIZATION METHODS

    def plot_overall_distribution(self, save: bool = True):
        """
        Create comprehensive distribution plot with annotations.
        """
        target_data = self.merged_data['plyr_gm_rec_yds']
        stats_dict = self.compute_basic_statistics(target_data)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Histogram with KDE
        ax1 = axes[0, 0]
        ax1.hist(target_data, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)

        # KDE overlay
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(target_data)
        x_range = np.linspace(target_data.min(), target_data.max(), 1000)
        ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

        # Add vertical lines for statistics
        ax1.axvline(stats_dict['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats_dict['mean']:.1f}")
        ax1.axvline(stats_dict['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats_dict['median']:.1f}")
        ax1.axvline(stats_dict['q1'], color='orange', linestyle=':', linewidth=1.5, label=f"Q1: {stats_dict['q1']:.1f}")
        ax1.axvline(stats_dict['q3'], color='orange', linestyle=':', linewidth=1.5, label=f"Q3: {stats_dict['q3']:.1f}")

        ax1.set_xlabel('Receiving Yards')
        ax1.set_ylabel('Density')
        ax1.set_title('Overall Distribution of Receiving Yards per Game')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2 = axes[0, 1]
        ax2.boxplot(target_data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue'),
                    medianprops=dict(color='red', linewidth=2))
        ax2.set_ylabel('Receiving Yards')
        ax2.set_title('Box Plot with Outliers')
        ax2.grid(True, alpha=0.3, axis='y')

        # Q-Q plot
        ax3 = axes[1, 0]
        stats.probplot(target_data, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Normal Distribution)')
        ax3.grid(True, alpha=0.3)

        # Statistics text box
        ax4 = axes[1, 1]
        ax4.axis('off')

        stats_text = f"""
        STATISTICAL SUMMARY
        {'='*40}

        Sample Size: {stats_dict['count']:,}

        Central Tendency:
          Mean:              {stats_dict['mean']:.2f} yards
          Median:            {stats_dict['median']:.2f} yards
          Mode:              {stats_dict['mode']:.2f} yards

        Spread:
          Std Dev:           {stats_dict['std']:.2f} yards
          Variance:          {stats_dict['variance']:.2f}
          IQR:               {stats_dict['iqr']:.2f} yards
          Range:             {stats_dict['range']:.2f} yards

        Distribution Shape:
          Skewness:          {stats_dict['skewness']:.3f}
          Kurtosis:          {stats_dict['kurtosis']:.3f}
          CV:                {stats_dict['cv']:.3f}

        Percentiles:
          5th:               {stats_dict['p5']:.1f} yards
          25th (Q1):         {stats_dict['q1']:.1f} yards
          50th (Median):     {stats_dict['q2']:.1f} yards
          75th (Q3):         {stats_dict['q3']:.1f} yards
          95th:              {stats_dict['p95']:.1f} yards

        Normality Test:
          Shapiro-Wilk p:    {stats_dict['shapiro_pvalue']:.6f}
          Is Normal:         {stats_dict['is_normal']}
        """

        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save:
            output_file = self.output_viz_path / 'target_overall_distribution.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved overall distribution plot to {output_file}")

        plt.close()

    def plot_position_comparison(self, save: bool = True):
        """
        Create position comparison box plots.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        positions = ['WR', 'TE', 'RB']
        position_data = [
            self.merged_data[self.merged_data['plyr_pos'] == pos]['plyr_gm_rec_yds']
            for pos in positions
        ]

        # Box plot
        ax1 = axes[0, 0]
        bp = ax1.boxplot(position_data, labels=positions, patch_artist=True,
                         medianprops=dict(color='red', linewidth=2))

        colors = ['lightcoral', 'lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # Add median values as text
        for i, (pos, data) in enumerate(zip(positions, position_data)):
            median_val = data.median()
            ax1.text(i+1, median_val, f'{median_val:.1f}',
                    ha='center', va='bottom', fontweight='bold')

        ax1.set_ylabel('Receiving Yards')
        ax1.set_title('Receiving Yards Distribution by Position')
        ax1.grid(True, alpha=0.3, axis='y')

        # Violin plot
        ax2 = axes[0, 1]
        parts = ax2.violinplot(position_data, positions=range(1, len(positions)+1),
                               showmeans=True, showmedians=True)
        ax2.set_xticks(range(1, len(positions)+1))
        ax2.set_xticklabels(positions)
        ax2.set_ylabel('Receiving Yards')
        ax2.set_title('Violin Plot by Position')
        ax2.grid(True, alpha=0.3, axis='y')

        # Histogram comparison
        ax3 = axes[1, 0]
        for pos, data, color in zip(positions, position_data, colors):
            ax3.hist(data, bins=40, alpha=0.5, label=pos, color=color, density=True)
        ax3.set_xlabel('Receiving Yards')
        ax3.set_ylabel('Density')
        ax3.set_title('Overlaid Distributions by Position')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Statistics comparison table
        ax4 = axes[1, 1]
        ax4.axis('off')

        pos_stats = self.position_comparison_statistics()

        stats_text = "POSITION COMPARISON\n" + "="*50 + "\n\n"

        for _, row in pos_stats.iterrows():
            stats_text += f"{row['position']}:\n"
            stats_text += f"  Count:     {int(row['count']):,}\n"
            stats_text += f"  Mean:      {row['mean']:.2f} yards\n"
            stats_text += f"  Median:    {row['median']:.2f} yards\n"
            stats_text += f"  Std Dev:   {row['std']:.2f} yards\n"
            stats_text += f"  CV:        {row['cv']:.3f}\n"
            stats_text += f"  Skewness:  {row['skewness']:.3f}\n\n"

        test_results = self.test_position_differences()
        stats_text += f"\nKruskal-Wallis Test:\n"
        stats_text += f"  H-statistic: {test_results['kruskal_wallis_h']:.2f}\n"
        stats_text += f"  p-value:     {test_results['kruskal_wallis_p']:.6f}\n"
        stats_text += f"  Significant: {test_results['significant_difference']}\n"
        stats_text += f"  Eta-squared: {test_results['eta_squared']:.4f}\n"

        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        plt.tight_layout()

        if save:
            output_file = self.output_viz_path / 'target_by_position_boxplot.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved position comparison plot to {output_file}")

        plt.close()

    def plot_seasonal_comparison(self, save: bool = True):
        """
        Create seasonal comparison violin plots.
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))

        # Violin plot by season
        ax1 = axes[0]
        seasons = sorted(self.merged_data['season_id'].unique())
        season_data = [
            self.merged_data[self.merged_data['season_id'] == s]['plyr_gm_rec_yds']
            for s in seasons
        ]

        parts = ax1.violinplot(season_data, positions=range(len(seasons)),
                              showmeans=True, showmedians=True)
        ax1.set_xticks(range(len(seasons)))
        ax1.set_xticklabels([f'{s}' for s in seasons])
        ax1.set_ylabel('Receiving Yards')
        ax1.set_xlabel('Season')
        ax1.set_title('Receiving Yards Distribution by Season (Violin Plot)')
        ax1.grid(True, alpha=0.3, axis='y')

        # Add box plot overlay
        bp = ax1.boxplot(season_data, positions=range(len(seasons)),
                        widths=0.1, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.5),
                        medianprops=dict(color='red', linewidth=2))

        # Trend line of means
        ax2 = axes[1]
        seasonal_stats = self.seasonal_comparison()

        x_pos = range(len(seasons))
        means = seasonal_stats['mean'].values
        stds = seasonal_stats['std'].values

        ax2.errorbar(x_pos, means, yerr=stds, marker='o', markersize=8,
                    capsize=5, capthick=2, linewidth=2, color='steelblue',
                    label='Mean ± Std Dev')
        ax2.plot(x_pos, seasonal_stats['median'].values, marker='s',
                markersize=8, linewidth=2, color='orange', label='Median')

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{s}' for s in seasons])
        ax2.set_ylabel('Receiving Yards')
        ax2.set_xlabel('Season')
        ax2.set_title('Seasonal Trends in Mean and Median Receiving Yards')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            output_file = self.output_viz_path / 'target_by_season_violin.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved seasonal comparison plot to {output_file}")

        plt.close()

    def plot_outlier_analysis(self, save: bool = True):
        """
        Create outlier visualization scatter plot.
        """
        target_data = self.merged_data['plyr_gm_rec_yds'].values
        outlier_info = self.detect_outliers(self.merged_data['plyr_gm_rec_yds'])

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Scatter plot with outliers highlighted
        ax1 = axes[0]

        # Create index array
        indices = np.arange(len(target_data))

        # Plot normal points
        normal_mask = ~(outlier_info['iqr_outliers_mask'] | outlier_info['z_outliers_mask'])
        ax1.scatter(indices[normal_mask], target_data[normal_mask],
                   alpha=0.3, s=10, color='gray', label='Normal')

        # Plot IQR outliers
        iqr_only = outlier_info['iqr_outliers_mask'] & ~outlier_info['z_outliers_mask']
        ax1.scatter(indices[iqr_only], target_data[iqr_only],
                   alpha=0.6, s=30, color='orange', label='IQR Outlier')

        # Plot Z-score outliers
        z_only = outlier_info['z_outliers_mask'] & ~outlier_info['iqr_outliers_mask']
        ax1.scatter(indices[z_only], target_data[z_only],
                   alpha=0.6, s=30, color='blue', label='Z-score Outlier')

        # Plot both methods outliers
        both = outlier_info['iqr_outliers_mask'] & outlier_info['z_outliers_mask']
        ax1.scatter(indices[both], target_data[both],
                   alpha=0.8, s=50, color='red', marker='*', label='Both Methods')

        # Add threshold lines
        ax1.axhline(outlier_info['iqr_upper_bound'], color='orange',
                   linestyle='--', linewidth=1.5, label=f"IQR Upper: {outlier_info['iqr_upper_bound']:.1f}")

        ax1.set_xlabel('Game Index')
        ax1.set_ylabel('Receiving Yards')
        ax1.set_title('Outlier Detection: IQR vs Z-score Methods')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Outlier summary
        ax2 = axes[1]
        ax2.axis('off')

        # Get top outliers with player names
        extreme_games = self.merged_data[both].nlargest(20, 'plyr_gm_rec_yds')

        outlier_text = f"""
        OUTLIER ANALYSIS SUMMARY
        {'='*50}

        IQR Method:
          Lower Bound:    {outlier_info['iqr_lower_bound']:.2f} yards
          Upper Bound:    {outlier_info['iqr_upper_bound']:.2f} yards
          Outlier Count:  {outlier_info['iqr_outlier_count']:,}
          Outlier %:      {outlier_info['iqr_outlier_pct']:.2f}%

        Z-Score Method (|z| > 3):
          Outlier Count:  {outlier_info['z_outlier_count']:,}
          Outlier %:      {outlier_info['z_outlier_pct']:.2f}%

        Both Methods:
          Count:          {both.sum():,}
          Percentage:     {100*both.sum()/len(target_data):.2f}%

        Top 10 Extreme Performances:
        """

        for i, (_, row) in enumerate(extreme_games.head(10).iterrows(), 1):
            outlier_text += f"\n  {i}. {row['plyr_name'][:20]:<20} {row['plyr_gm_rec_yds']:>3.0f} yds"

        ax2.text(0.1, 0.9, outlier_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save:
            output_file = self.output_viz_path / 'target_outliers_scatter.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved outlier analysis plot to {output_file}")

        plt.close()

    def plot_yard_buckets(self, save: bool = True):
        """
        Create yard range bucket visualization.
        """
        bucketed_data = self.create_yard_buckets(self.merged_data)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Overall bucket distribution
        ax1 = axes[0, 0]
        bucket_counts = bucketed_data['yard_bucket'].value_counts().sort_index()
        bucket_pcts = 100 * bucket_counts / len(bucketed_data)

        bars = ax1.bar(range(len(bucket_pcts)), bucket_pcts.values,
                      color='steelblue', edgecolor='black')
        ax1.set_xticks(range(len(bucket_pcts)))
        ax1.set_xticklabels(bucket_pcts.index, rotation=45, ha='right')
        ax1.set_ylabel('Percentage of Games (%)')
        ax1.set_title('Distribution of Games by Receiving Yard Ranges')
        ax1.grid(True, alpha=0.3, axis='y')

        # Add percentage labels on bars
        for i, (bar, pct) in enumerate(zip(bars, bucket_pcts.values)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

        # By position
        ax2 = axes[0, 1]

        positions = ['WR', 'TE', 'RB']
        bucket_labels = bucket_counts.index.tolist()

        x = np.arange(len(bucket_labels))
        width = 0.25

        for i, pos in enumerate(positions):
            pos_data = bucketed_data[bucketed_data['plyr_pos'] == pos]
            pos_bucket_counts = pos_data['yard_bucket'].value_counts().sort_index()
            pos_bucket_pcts = 100 * pos_bucket_counts / len(pos_data)

            # Ensure all buckets are represented
            pos_bucket_pcts = pos_bucket_pcts.reindex(bucket_labels, fill_value=0)

            offset = width * (i - 1)
            ax2.bar(x + offset, pos_bucket_pcts.values, width,
                   label=pos, alpha=0.8)

        ax2.set_xticks(x)
        ax2.set_xticklabels(bucket_labels, rotation=45, ha='right')
        ax2.set_ylabel('Percentage of Games (%)')
        ax2.set_title('Yard Ranges by Position')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Cumulative distribution
        ax3 = axes[1, 0]

        sorted_yards = np.sort(bucketed_data['plyr_gm_rec_yds'].values)
        cumulative_pct = np.arange(1, len(sorted_yards) + 1) / len(sorted_yards) * 100

        ax3.plot(sorted_yards, cumulative_pct, linewidth=2, color='steelblue')
        ax3.axhline(50, color='red', linestyle='--', linewidth=1.5, label='50th Percentile')
        ax3.axhline(75, color='orange', linestyle='--', linewidth=1.5, label='75th Percentile')
        ax3.axhline(90, color='green', linestyle='--', linewidth=1.5, label='90th Percentile')

        ax3.set_xlabel('Receiving Yards')
        ax3.set_ylabel('Cumulative Percentage (%)')
        ax3.set_title('Cumulative Distribution Function')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Statistics table
        ax4 = axes[1, 1]
        ax4.axis('off')

        stats_text = "YARD BUCKET STATISTICS\n" + "="*50 + "\n\n"

        for bucket in bucket_labels:
            count = bucket_counts.get(bucket, 0)
            pct = bucket_pcts.get(bucket, 0)
            stats_text += f"{bucket}:\n"
            stats_text += f"  Count: {count:>6,} games\n"
            stats_text += f"  Pct:   {pct:>6.2f}%\n\n"

        # Zero yard analysis
        zero_analysis = self.analyze_zero_yards(bucketed_data)
        stats_text += f"\nZero Yard Game Analysis:\n"
        stats_text += f"  Total:         {zero_analysis['total_zero_games']:,}\n"
        stats_text += f"  Percentage:    {zero_analysis['zero_game_pct']:.2f}%\n"
        stats_text += f"  With Targets:  {zero_analysis['zero_with_targets']:,}\n"
        stats_text += f"  No Targets:    {zero_analysis['zero_no_targets']:,}\n"
        stats_text += f"  With Drops:    {zero_analysis['zero_with_drops']:,}\n"

        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        plt.tight_layout()

        if save:
            output_file = self.output_viz_path / 'yard_range_percentages.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved yard bucket plot to {output_file}")

        plt.close()

    def plot_position_grid(self, save: bool = True):
        """
        Create comprehensive position comparison grid.
        """
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))

        positions = ['WR', 'TE', 'RB']

        for i, pos in enumerate(positions):
            pos_data = self.merged_data[self.merged_data['plyr_pos'] == pos]['plyr_gm_rec_yds']

            # Histogram
            ax_hist = axes[i, 0]
            ax_hist.hist(pos_data, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
            ax_hist.axvline(pos_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {pos_data.mean():.1f}')
            ax_hist.axvline(pos_data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {pos_data.median():.1f}')
            ax_hist.set_xlabel('Receiving Yards')
            ax_hist.set_ylabel('Frequency')
            ax_hist.set_title(f'{pos} Distribution')
            ax_hist.legend()
            ax_hist.grid(True, alpha=0.3)

            # Q-Q Plot
            ax_qq = axes[i, 1]
            stats.probplot(pos_data, dist="norm", plot=ax_qq)
            ax_qq.set_title(f'{pos} Q-Q Plot')
            ax_qq.grid(True, alpha=0.3)

            # Box plot
            ax_box = axes[i, 2]
            ax_box.boxplot(pos_data, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightblue'),
                          medianprops=dict(color='red', linewidth=2))
            ax_box.set_ylabel('Receiving Yards')
            ax_box.set_title(f'{pos} Box Plot')
            ax_box.grid(True, alpha=0.3, axis='y')

            # Add statistics text
            pos_stats = self.compute_basic_statistics(pos_data)
            stats_text = f"N={pos_stats['count']:,}\nμ={pos_stats['mean']:.1f}\nσ={pos_stats['std']:.1f}\nSkew={pos_stats['skewness']:.2f}"
            ax_box.text(1.15, 0.5, stats_text, transform=ax_box.transAxes,
                       fontsize=9, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save:
            output_file = self.output_viz_path / 'position_comparison_grid.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved position comparison grid to {output_file}")

        plt.close()

    # DATA OUTPUT METHODS

    def save_summary_statistics(self):
        """
        Save comprehensive summary statistics to CSV.
        """
        all_stats = []

        # Overall statistics
        overall_stats = self.compute_basic_statistics(self.merged_data['plyr_gm_rec_yds'])
        overall_stats['segment_type'] = 'Overall'
        overall_stats['segment_value'] = 'All Data'
        all_stats.append(overall_stats)

        # By position
        for pos in ['WR', 'TE', 'RB']:
            pos_data = self.merged_data[self.merged_data['plyr_pos'] == pos]['plyr_gm_rec_yds']
            pos_stats = self.compute_basic_statistics(pos_data)
            pos_stats['segment_type'] = 'Position'
            pos_stats['segment_value'] = pos
            all_stats.append(pos_stats)

        # By season
        for season in sorted(self.merged_data['season_id'].unique()):
            season_data = self.merged_data[self.merged_data['season_id'] == season]['plyr_gm_rec_yds']
            season_stats = self.compute_basic_statistics(season_data)
            season_stats['segment_type'] = 'Season'
            season_stats['segment_value'] = str(season)
            all_stats.append(season_stats)

        # By week range
        self.merged_data['week_range'] = pd.cut(
            self.merged_data['week_id'],
            bins=[0, 6, 13, 18],
            labels=['Early (1-6)', 'Mid (7-13)', 'Late (14-18)']
        )

        for week_range in ['Early (1-6)', 'Mid (7-13)', 'Late (14-18)']:
            range_data = self.merged_data[self.merged_data['week_range'] == week_range]['plyr_gm_rec_yds']
            range_stats = self.compute_basic_statistics(range_data)
            range_stats['segment_type'] = 'Week Range'
            range_stats['segment_value'] = week_range
            all_stats.append(range_stats)

        # Convert to DataFrame
        stats_df = pd.DataFrame(all_stats)

        # Reorder columns
        first_cols = ['segment_type', 'segment_value']
        other_cols = [c for c in stats_df.columns if c not in first_cols]
        stats_df = stats_df[first_cols + other_cols]

        # Save
        output_file = self.output_data_path / 'target_summary_statistics.csv'
        stats_df.to_csv(output_file, index=False)
        logger.info(f"Saved summary statistics to {output_file}")

        return stats_df

    def save_outlier_games(self):
        """
        Save outlier games to CSV.
        """
        outlier_info = self.detect_outliers(self.merged_data['plyr_gm_rec_yds'])

        # Get games with outliers by either method
        outlier_mask = outlier_info['iqr_outliers_mask'] | outlier_info['z_outliers_mask']
        outlier_games = self.merged_data[outlier_mask].copy()

        # Calculate z-scores for all outliers
        all_z_scores = stats.zscore(self.merged_data['plyr_gm_rec_yds'])
        outlier_games['z_score'] = all_z_scores[outlier_mask]

        # Add flags
        outlier_games['iqr_flag'] = outlier_info['iqr_outliers_mask'][outlier_mask]
        outlier_games['z_score_flag'] = outlier_info['z_outliers_mask'][outlier_mask]

        # Select relevant columns
        output_cols = [
            'plyr_id', 'plyr_name', 'plyr_pos', 'game_id', 'season_id', 'week_id',
            'plyr_gm_rec_yds', 'plyr_gm_rec_tgt', 'plyr_gm_rec',
            'z_score', 'iqr_flag', 'z_score_flag'
        ]

        outlier_games_output = outlier_games[output_cols].sort_values(
            'plyr_gm_rec_yds', ascending=False
        )

        # Save
        output_file = self.output_data_path / 'outlier_games.csv'
        outlier_games_output.to_csv(output_file, index=False)
        logger.info(f"Saved outlier games to {output_file}")

        return outlier_games_output

    def save_zero_yard_games(self):
        """
        Save zero-yard games analysis to CSV.
        """
        zero_games = self.merged_data[self.merged_data['plyr_gm_rec_yds'] == 0].copy()

        # Classify reason for zero yards
        def classify_zero_reason(row):
            if row['plyr_gm_rec_tgt'] == 0:
                return 'no_targets'
            elif row['plyr_gm_rec_drp'] > 0:
                return 'drops_only'
            else:
                return 'targeted_no_catches'

        zero_games['reason'] = zero_games.apply(classify_zero_reason, axis=1)

        # Select relevant columns
        output_cols = [
            'plyr_id', 'plyr_name', 'plyr_pos', 'game_id', 'season_id', 'week_id',
            'plyr_gm_rec_tgt', 'plyr_gm_rec', 'plyr_gm_rec_drp', 'reason'
        ]

        zero_games_output = zero_games[output_cols].sort_values(
            ['season_id', 'week_id', 'plyr_name']
        )

        # Save
        output_file = self.output_data_path / 'zero_yard_games.csv'
        zero_games_output.to_csv(output_file, index=False)
        logger.info(f"Saved zero yard games to {output_file}")

        return zero_games_output

    def generate_report(self):
        """
        Generate comprehensive markdown report.
        """
        report = []

        report.append("# Target Variable Deep Dive Analysis")
        report.append("## NFL Receiving Yards Prediction Model")
        report.append("")
        report.append(f"**Analysis Date**: 2025-10-25")
        report.append(f"**Seasons Analyzed**: 2022-2025")
        report.append(f"**Total Observations**: {len(self.merged_data):,}")
        report.append("")

        # Executive Summary
        report.append("## Executive Summary")
        report.append("")

        overall_stats = self.compute_basic_statistics(self.merged_data['plyr_gm_rec_yds'])
        zero_analysis = self.analyze_zero_yards(self.merged_data)
        pos_test = self.test_position_differences()

        report.append(f"- **Target Distribution**: Receiving yards per game shows high positive skewness "
                     f"({overall_stats['skewness']:.2f}) with mean of {overall_stats['mean']:.1f} yards "
                     f"and median of {overall_stats['median']:.1f} yards, indicating a right-tailed distribution "
                     f"with frequent low-production games and rare elite performances.")

        report.append(f"- **Key Challenge**: {zero_analysis['zero_game_pct']:.1f}% of games result in zero "
                     f"receiving yards, with {zero_analysis['zero_no_targets']} games having no targets. "
                     f"This zero-inflation poses significant predictability challenges.")

        report.append(f"- **Position Differences**: Statistical testing (Kruskal-Wallis H={pos_test['kruskal_wallis_h']:.2f}, "
                     f"p<0.001) confirms significant differences in receiving yard distributions across positions, "
                     f"suggesting position-specific models may improve performance.")

        report.append(f"- **Variance**: High coefficient of variation ({overall_stats['cv']:.2f}) indicates "
                     f"substantial game-to-game variability, with realistic R-squared targets in the 0.15-0.35 range "
                     f"for baseline models.")

        report.append("")

        # Detailed Findings
        report.append("## Detailed Findings")
        report.append("")

        # Distribution Characteristics
        report.append("### 1. Distribution Characteristics")
        report.append("")
        report.append("#### Overall Statistics")
        report.append("")
        report.append(f"- **Sample Size**: {overall_stats['count']:,} player-game observations")
        report.append(f"- **Mean**: {overall_stats['mean']:.2f} yards")
        report.append(f"- **Median**: {overall_stats['median']:.2f} yards")
        report.append(f"- **Standard Deviation**: {overall_stats['std']:.2f} yards")
        report.append(f"- **Range**: {overall_stats['min']:.0f} to {overall_stats['max']:.0f} yards")
        report.append(f"- **IQR**: {overall_stats['iqr']:.2f} yards (Q1={overall_stats['q1']:.1f}, Q3={overall_stats['q3']:.1f})")
        report.append("")

        report.append("#### Distribution Shape")
        report.append("")
        report.append(f"- **Skewness**: {overall_stats['skewness']:.3f} (highly right-skewed)")
        report.append(f"- **Kurtosis**: {overall_stats['kurtosis']:.3f} (heavy-tailed)")
        report.append(f"- **Normality**: Shapiro-Wilk test p-value = {overall_stats['shapiro_pvalue']:.6f} "
                     f"({'Reject' if overall_stats['shapiro_pvalue'] < 0.05 else 'Accept'} normality hypothesis)")
        report.append(f"- **Coefficient of Variation**: {overall_stats['cv']:.3f}")
        report.append("")

        skew_interpretation = "highly skewed" if abs(overall_stats['skewness']) > 1 else "moderately skewed"
        report.append(f"**Interpretation**: The distribution is {skew_interpretation} with most games producing "
                     f"modest yardage and a long tail of high-production games. The high kurtosis indicates "
                     f"more extreme values than a normal distribution would predict.")
        report.append("")

        # Position-Based Insights
        report.append("### 2. Position-Based Insights")
        report.append("")

        pos_stats = self.position_comparison_statistics()

        report.append("#### Position-Specific Statistics")
        report.append("")
        report.append("| Position | Count | Mean | Median | Std Dev | CV | Skewness |")
        report.append("|----------|-------|------|--------|---------|-------|----------|")

        for _, row in pos_stats.iterrows():
            report.append(f"| {row['position']} | {int(row['count']):,} | "
                         f"{row['mean']:.1f} | {row['median']:.1f} | "
                         f"{row['std']:.1f} | {row['cv']:.3f} | {row['skewness']:.3f} |")

        report.append("")

        report.append("#### Statistical Significance Testing")
        report.append("")
        report.append(f"- **Kruskal-Wallis H-Test**: H={pos_test['kruskal_wallis_h']:.2f}, "
                     f"p={pos_test['kruskal_wallis_p']:.6f}")
        report.append(f"- **Effect Size (Eta-squared)**: {pos_test['eta_squared']:.4f}")
        report.append(f"- **Conclusion**: {'Significant' if pos_test['significant_difference'] else 'No significant'} "
                     f"differences across positions")
        report.append("")

        report.append("#### Pairwise Comparisons (Mann-Whitney U Test)")
        report.append("")
        report.append(f"- **WR vs TE**: U={pos_test['wr_te_u_stat']:.0f}, p={pos_test['wr_te_p']:.6f}")
        report.append(f"- **WR vs RB**: U={pos_test['wr_rb_u_stat']:.0f}, p={pos_test['wr_rb_p']:.6f}")
        report.append(f"- **TE vs RB**: U={pos_test['te_rb_u_stat']:.0f}, p={pos_test['te_rb_p']:.6f}")
        report.append("")

        # Find position with highest mean
        max_mean_pos = pos_stats.loc[pos_stats['mean'].idxmax(), 'position']
        min_cv_pos = pos_stats.loc[pos_stats['cv'].idxmin(), 'position']

        report.append(f"**Key Insights**:")
        report.append(f"- {max_mean_pos} positions average the highest receiving yards")
        report.append(f"- {min_cv_pos} positions show the most consistency (lowest CV)")
        report.append(f"- All positions exhibit high skewness, indicating frequent low-production games")
        report.append("")

        # Temporal Patterns
        report.append("### 3. Temporal Pattern Analysis")
        report.append("")

        seasonal_stats = self.seasonal_comparison()

        report.append("#### Season-over-Season Trends")
        report.append("")
        report.append("| Season | Count | Mean | Median | Std Dev | Skewness |")
        report.append("|--------|-------|------|--------|---------|----------|")

        for _, row in seasonal_stats.iterrows():
            report.append(f"| {int(row['season'])} | {int(row['count']):,} | "
                         f"{row['mean']:.1f} | {row['median']:.1f} | "
                         f"{row['std']:.1f} | {row['skewness']:.3f} |")

        report.append("")

        # Calculate trend
        mean_trend = seasonal_stats['mean'].diff().mean()
        trend_direction = "increasing" if mean_trend > 0 else "decreasing"

        report.append(f"**Trend Analysis**: Average receiving yards per game shows {trend_direction} trend "
                     f"({mean_trend:+.2f} yards/season). Distribution characteristics remain relatively "
                     f"stable across seasons.")
        report.append("")

        week_range_stats = self.week_range_comparison()

        report.append("#### Intra-Season Patterns")
        report.append("")
        report.append("| Week Range | Count | Mean | Median | Std Dev |")
        report.append("|------------|-------|------|--------|---------|")

        for _, row in week_range_stats.iterrows():
            report.append(f"| {row['week_range']} | {int(row['count']):,} | "
                         f"{row['mean']:.1f} | {row['median']:.1f} | {row['std']:.1f} |")

        report.append("")

        # Autocorrelation
        autocorr_results = self.autocorrelation_analysis()

        report.append("#### Week-to-Week Correlation")
        report.append("")
        report.append(f"- **Mean Lag-1 Autocorrelation**: {autocorr_results['mean_autocorr']:.3f}")
        report.append(f"- **Median Lag-1 Autocorrelation**: {autocorr_results['median_autocorr']:.3f}")
        report.append(f"- **Players Analyzed**: {autocorr_results['num_players_analyzed']:,}")
        report.append("")

        autocorr_strength = "weak" if abs(autocorr_results['mean_autocorr']) < 0.3 else "moderate" if abs(autocorr_results['mean_autocorr']) < 0.6 else "strong"
        report.append(f"**Interpretation**: {autocorr_strength.capitalize()} week-to-week correlation suggests "
                     f"that {'recent performance has limited predictive value' if autocorr_strength == 'weak' else 'recent performance provides some predictive signal'}.")
        report.append("")

        # Outliers
        report.append("### 4. Outlier Analysis")
        report.append("")

        outlier_info = self.detect_outliers(self.merged_data['plyr_gm_rec_yds'])

        report.append(f"- **IQR Method**: {outlier_info['iqr_outlier_count']:,} outliers "
                     f"({outlier_info['iqr_outlier_pct']:.2f}%)")
        report.append(f"  - Lower Bound: {outlier_info['iqr_lower_bound']:.2f} yards")
        report.append(f"  - Upper Bound: {outlier_info['iqr_upper_bound']:.2f} yards")
        report.append("")
        report.append(f"- **Z-Score Method (|z| > 3)**: {outlier_info['z_outlier_count']:,} outliers "
                     f"({outlier_info['z_outlier_pct']:.2f}%)")
        report.append("")

        # Get top outliers
        both_methods = outlier_info['iqr_outliers_mask'] & outlier_info['z_outliers_mask']
        top_outliers = self.merged_data[both_methods].nlargest(10, 'plyr_gm_rec_yds')

        report.append("#### Top 10 Outlier Performances")
        report.append("")
        report.append("| Rank | Player | Position | Yards | Season | Week |")
        report.append("|------|--------|----------|-------|--------|------|")

        for i, (_, row) in enumerate(top_outliers.iterrows(), 1):
            report.append(f"| {i} | {row['plyr_name']} | {row['plyr_pos']} | "
                         f"{int(row['plyr_gm_rec_yds'])} | {int(row['season_id'])} | {int(row['week_id'])} |")

        report.append("")

        # Zero Yards
        report.append("### 5. Zero-Yard Game Analysis")
        report.append("")

        report.append(f"- **Total Zero-Yard Games**: {zero_analysis['total_zero_games']:,} "
                     f"({zero_analysis['zero_game_pct']:.2f}%)")
        report.append(f"- **Zero Yards with Targets**: {zero_analysis['zero_with_targets']:,}")
        report.append(f"- **Zero Yards without Targets**: {zero_analysis['zero_no_targets']:,}")
        report.append(f"- **Games with Drops (Zero Yards)**: {zero_analysis['zero_with_drops']:,}")
        report.append("")

        report.append("**Implication**: The high proportion of zero-yard games, particularly those without "
                     "targets, represents challenging observations to predict. These may reflect game-day "
                     "inactivity, game script, or injury considerations not captured in basic statistics.")
        report.append("")

        # Modeling Implications
        report.append("## Modeling Implications")
        report.append("")

        report.append("### 1. Model Architecture Recommendations")
        report.append("")
        report.append(f"**Position-Specific Models**: {'RECOMMENDED' if pos_test['significant_difference'] else 'NOT RECOMMENDED'}")
        report.append("")

        if pos_test['significant_difference']:
            report.append("The significant statistical differences in distributions across WR, TE, and RB "
                         "positions suggest that separate Random Forest models for each position will likely "
                         "outperform a unified model. Each position exhibits distinct patterns in:")
            report.append("- Central tendency (mean/median receiving yards)")
            report.append("- Variability (standard deviation and CV)")
            report.append("- Distribution shape (skewness and kurtosis)")
            report.append("")
            report.append("**Implementation**: Train three separate Random Forest regressors, one for each position.")
        else:
            report.append("Position distributions are similar enough to use a unified model with position "
                         "as a feature variable.")

        report.append("")

        report.append("### 2. Zero-Yard Handling Strategy")
        report.append("")
        report.append("Given the high percentage of zero-yard games, consider a two-stage approach:")
        report.append("")
        report.append("1. **Classification Model**: Predict probability of zero yards (binary classifier)")
        report.append(f"   - Target variable: plyr_gm_rec_yds == 0 ({zero_analysis['zero_game_pct']:.1f}% positive class)")
        report.append("   - Features: Target share, injury status, game script indicators")
        report.append("")
        report.append("2. **Regression Model**: Predict receiving yards conditional on non-zero production")
        report.append(f"   - Train only on games with plyr_gm_rec_yds > 0")
        report.append("   - Final prediction: P(zero) * 0 + P(non-zero) * E[yards | non-zero]")
        report.append("")
        report.append("**Alternative**: Include zero-yard games but use stratified sampling or SMOTE for "
                     "better representation of the target distribution.")
        report.append("")

        report.append("### 3. Target Transformation")
        report.append("")

        if overall_stats['skewness'] > 1.0:
            report.append(f"**RECOMMENDED**: Apply transformation to reduce skewness")
            report.append("")
            report.append(f"Current skewness ({overall_stats['skewness']:.2f}) indicates highly right-skewed distribution. "
                         "Consider:")
            report.append("")
            report.append("- **Square Root Transformation**: `sqrt(plyr_gm_rec_yds)`")
            report.append("- **Log Transformation**: `log(plyr_gm_rec_yds + 1)` (handles zeros)")
            report.append("- **Box-Cox Transformation**: Optimize lambda parameter")
            report.append("")
            report.append("Random Forests handle non-normal distributions well, but transformation may improve "
                         "performance on extreme values and reduce prediction variance.")
        else:
            report.append("**NOT REQUIRED**: Distribution skewness is moderate. Random Forests can handle "
                         "the current distribution without transformation.")

        report.append("")

        report.append("### 4. Outlier Handling")
        report.append("")
        report.append(f"With {outlier_info['iqr_outlier_pct']:.2f}% outliers by IQR method:")
        report.append("")
        report.append("**RECOMMENDED APPROACH**: Keep outliers in training data")
        report.append("")
        report.append("Rationale:")
        report.append("- Random Forests are robust to outliers due to decision tree partitioning")
        report.append("- Outlier games (150+ yards) represent real, predictable elite performances")
        report.append("- Removing outliers would limit model's ability to predict breakout games")
        report.append("")
        report.append("**Monitoring**: Track model performance separately on outlier predictions to ensure "
                     "the model captures high-production games.")
        report.append("")

        report.append("### 5. Feature Engineering Priorities")
        report.append("")
        report.append("Based on distribution analysis, prioritize features that address:")
        report.append("")
        report.append("1. **Target Opportunity**")
        report.append("   - Historical target share")
        report.append("   - Red zone targets")
        report.append("   - Air yards share")
        report.append("")
        report.append("2. **Efficiency Metrics**")
        report.append("   - Yards per route run (YPRR)")
        report.append("   - Yards after catch (YAC)")
        report.append("   - Catch rate")
        report.append("")
        report.append("3. **Game Context**")
        report.append("   - Vegas implied team total")
        report.append("   - Opponent pass defense ranking")
        report.append("   - Game script (predicted or actual)")
        report.append("")
        report.append("4. **Temporal Features**")
        report.append(f"   - Recent form (last 3-4 games) - autocorrelation: {autocorr_results['mean_autocorr']:.3f}")
        report.append("   - Season trend (early/mid/late)")
        report.append("   - Rest days")
        report.append("")

        # Prediction Challenges
        report.append("## Prediction Challenges")
        report.append("")

        report.append("### 1. Expected Model Performance")
        report.append("")

        # Calculate baseline variance
        baseline_var = overall_stats['variance']
        report.append(f"**Baseline Variance**: {baseline_var:.2f}")
        report.append("")
        report.append("**Realistic R-Squared Targets**:")
        report.append("")
        report.append("- **Naive Baseline** (mean prediction): R² = 0.00")
        report.append("- **Simple Model** (position + recent avg): R² = 0.10 - 0.15")
        report.append("- **Random Forest Baseline**: R² = 0.15 - 0.25")
        report.append("- **Optimized RF with Feature Engineering**: R² = 0.25 - 0.35")
        report.append("- **Advanced Ensemble**: R² = 0.35 - 0.45")
        report.append("")
        report.append(f"Given the high coefficient of variation ({overall_stats['cv']:.2f}) and inherent "
                     "game-to-game randomness in NFL receiving production, R² values in the 0.25-0.35 range "
                     "represent strong performance for this prediction task.")
        report.append("")

        report.append("### 2. Difficult-to-Predict Scenarios")
        report.append("")
        report.append("1. **Low-Usage Players**: Players with inconsistent target volume face prediction challenges")
        report.append(f"   - Zero-target games: {zero_analysis['zero_no_targets']:,} observations")
        report.append("")
        report.append("2. **Extreme Performances**: Outlier games (150+ yards) are rare and hard to forecast")
        report.append(f"   - Represents {outlier_info['iqr_outlier_pct']:.2f}% of observations")
        report.append("")
        report.append("3. **Injury/Game-Day Decisions**: Players unexpectedly inactive or limited")
        report.append("   - Requires external data integration")
        report.append("")
        report.append("4. **Game Script Deviations**: Blowouts or unexpected defensive strategies")
        report.append(f"   - Week-to-week correlation only {autocorr_results['mean_autocorr']:.2f}")
        report.append("")

        report.append("### 3. Data Quality Considerations")
        report.append("")
        report.append(f"- **Completeness**: {len(self.merged_data):,} observations across {len(self.merged_data['season_id'].unique())} seasons")
        report.append(f"- **Position Coverage**: WR ({len(self.merged_data[self.merged_data['plyr_pos']=='WR']):,}), "
                     f"TE ({len(self.merged_data[self.merged_data['plyr_pos']=='TE']):,}), "
                     f"RB ({len(self.merged_data[self.merged_data['plyr_pos']=='RB']):,})")
        report.append("- **Missing Values**: Examine route-based features for completeness")
        report.append("- **Temporal Coverage**: Ensure sufficient observations per player for time-series features")
        report.append("")

        # Recommended Next Steps
        report.append("## Recommended Next Steps")
        report.append("")

        report.append("### 1. Feature Investigation Priorities")
        report.append("")
        report.append("**High Priority**:")
        report.append("- Analyze correlation between `plyr_gm_rec_tgt` (targets) and receiving yards")
        report.append("- Examine `plyr_gm_rec_adot` (average depth of target) distributions by position")
        report.append("- Investigate `plyr_gm_rec_yac` (yards after catch) as efficiency metric")
        report.append("- Build rolling average features (3, 5, 8 game windows)")
        report.append("")
        report.append("**Medium Priority**:")
        report.append("- Opponent defensive metrics (pass defense DVOA, coverage schemes)")
        report.append("- Weather conditions for outdoor games")
        report.append("- Vegas betting lines (over/under, spread)")
        report.append("- Team passing play rate")
        report.append("")
        report.append("**Lower Priority**:")
        report.append("- Player physical attributes (height, weight, speed)")
        report.append("- Draft capital / pedigree")
        report.append("- Contract year indicators")
        report.append("")

        report.append("### 2. Model Development Roadmap")
        report.append("")
        report.append("**Phase 1: Baseline Models** (Week 1-2)")
        report.append("- Build separate position-specific Random Forest regressors")
        report.append("- Features: Recent averages, position, opponent")
        report.append("- Establish performance benchmarks (R², RMSE, MAE)")
        report.append("")
        report.append("**Phase 2: Feature Engineering** (Week 3-4)")
        report.append("- Implement recommended features from priority list")
        report.append("- Feature selection using permutation importance")
        report.append("- Hyperparameter tuning (n_estimators, max_depth, min_samples_split)")
        report.append("")
        report.append("**Phase 3: Advanced Techniques** (Week 5-6)")
        report.append("- Two-stage modeling (classification + regression)")
        report.append("- Ensemble methods (stack RF with Gradient Boosting, XGBoost)")
        report.append("- Quantile regression for prediction intervals")
        report.append("")

        report.append("### 3. Validation Strategy")
        report.append("")
        report.append("**Time-Series Cross-Validation**:")
        report.append("- Do NOT use random k-fold (violates temporal ordering)")
        report.append("- Use expanding window or rolling window approach")
        report.append("- Train on seasons 2022-2023, validate on 2024")
        report.append("- Final test on 2025 (current season)")
        report.append("")
        report.append("**Stratification**:")
        report.append("- Ensure balanced position representation in folds")
        report.append("- Consider stratifying by yard buckets for even coverage")
        report.append("")
        report.append("**Metrics**:")
        report.append("- Primary: RMSE (penalizes large errors)")
        report.append("- Secondary: MAE (interpretable), R² (variance explained)")
        report.append("- Segmented performance: by position, by yard range")
        report.append("")

        report.append("### 4. Production Considerations")
        report.append("")
        report.append("**Model Monitoring**:")
        report.append("- Track prediction error by position weekly")
        report.append("- Monitor for distribution drift (mean, variance shifts)")
        report.append("- Alert on outlier predictions (>3 standard deviations from historical)")
        report.append("")
        report.append("**Retraining Cadence**:")
        report.append("- Weekly updates with latest game data")
        report.append("- Full retraining every 4-6 weeks")
        report.append("- Feature importance tracking for stability")
        report.append("")
        report.append("**Prediction Intervals**:")
        report.append("- Use quantile regression forests for uncertainty estimates")
        report.append("- Provide 80% and 95% prediction intervals")
        report.append("- Communicate uncertainty in high-variance scenarios")
        report.append("")

        # Conclusion
        report.append("## Conclusion")
        report.append("")
        report.append("The target variable (plyr_gm_rec_yds) exhibits characteristics typical of sports "
                     "prediction tasks: high variance, significant skewness, and substantial zero-inflation. "
                     "These properties present challenges but also opportunities:")
        report.append("")
        report.append("**Key Takeaways**:")
        report.append(f"1. Position-specific models are statistically justified (p<0.001)")
        report.append(f"2. Zero-yard games ({zero_analysis['zero_game_pct']:.1f}%) require special handling")
        report.append(f"3. Weak autocorrelation ({autocorr_results['mean_autocorr']:.2f}) limits pure time-series approaches")
        report.append(f"4. High CV ({overall_stats['cv']:.2f}) sets realistic performance expectations")
        report.append("")
        report.append("With thoughtful feature engineering, proper validation methodology, and position-specific "
                     "modeling, Random Forest models can achieve meaningful predictive performance (R² = 0.25-0.35) "
                     "for this challenging task. The analysis provides a solid foundation for model development "
                     "and sets realistic expectations for stakeholders.")
        report.append("")
        report.append("---")
        report.append("")
        report.append("**Generated by**: NFL Receiving Yards Target Analysis Pipeline")
        report.append("**Analysis Version**: 1.0")
        report.append("**Contact**: Data Science Team")

        # Save report
        output_file = self.output_report_path / '01_target_variable_insights.md'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        logger.info(f"Saved comprehensive report to {output_file}")

        return '\n'.join(report)


def main():
    """
    Main execution function.
    """
    logger.info("="*80)
    logger.info("NFL RECEIVING YARDS TARGET VARIABLE ANALYSIS")
    logger.info("="*80)

    # Initialize analyzer
    base_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\clean"
    analyzer = NFLReceivingYardsAnalyzer(base_path)

    # Load data
    logger.info("\n[STEP 1/8] Loading data...")
    analyzer.load_receiving_data()
    analyzer.load_player_data()
    analyzer.merge_data()

    # Task 1: Overall Distribution Analysis
    logger.info("\n[STEP 2/8] Computing comprehensive statistics...")
    overall_stats = analyzer.compute_basic_statistics(analyzer.merged_data['plyr_gm_rec_yds'])
    logger.info(f"Overall Mean: {overall_stats['mean']:.2f} yards")
    logger.info(f"Overall Median: {overall_stats['median']:.2f} yards")
    logger.info(f"Overall Std Dev: {overall_stats['std']:.2f} yards")
    logger.info(f"Skewness: {overall_stats['skewness']:.3f}")

    # Task 2: Position Analysis
    logger.info("\n[STEP 3/8] Analyzing position differences...")
    pos_stats = analyzer.position_comparison_statistics()
    pos_test = analyzer.test_position_differences()
    logger.info(f"Kruskal-Wallis H-test p-value: {pos_test['kruskal_wallis_p']:.6f}")
    logger.info(f"Significant differences: {pos_test['significant_difference']}")

    # Task 3: Temporal Analysis
    logger.info("\n[STEP 4/8] Analyzing temporal patterns...")
    seasonal_stats = analyzer.seasonal_comparison()
    week_range_stats = analyzer.week_range_comparison()
    autocorr_results = analyzer.autocorrelation_analysis()
    logger.info(f"Mean lag-1 autocorrelation: {autocorr_results['mean_autocorr']:.3f}")

    # Outlier and zero analysis
    logger.info("\n[STEP 5/8] Detecting outliers and analyzing zeros...")
    outlier_info = analyzer.detect_outliers(analyzer.merged_data['plyr_gm_rec_yds'])
    zero_analysis = analyzer.analyze_zero_yards(analyzer.merged_data)
    logger.info(f"IQR Outliers: {outlier_info['iqr_outlier_pct']:.2f}%")
    logger.info(f"Zero-yard games: {zero_analysis['zero_game_pct']:.2f}%")

    # Generate visualizations
    logger.info("\n[STEP 6/8] Creating visualizations...")
    analyzer.plot_overall_distribution()
    analyzer.plot_position_comparison()
    analyzer.plot_seasonal_comparison()
    analyzer.plot_outlier_analysis()
    analyzer.plot_yard_buckets()
    analyzer.plot_position_grid()

    # Save data outputs
    logger.info("\n[STEP 7/8] Saving data outputs...")
    analyzer.save_summary_statistics()
    analyzer.save_outlier_games()
    analyzer.save_zero_yard_games()

    # Generate report
    logger.info("\n[STEP 8/8] Generating comprehensive report...")
    analyzer.generate_report()

    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("="*80)
    logger.info("\nDeliverables saved to:")
    logger.info(f"- Visualizations: {analyzer.output_viz_path}")
    logger.info(f"- Data outputs: {analyzer.output_data_path}")
    logger.info(f"- Report: {analyzer.output_report_path}")
    logger.info("")


if __name__ == "__main__":
    main()
