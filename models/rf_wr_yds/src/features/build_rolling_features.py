"""
Rolling Feature Builder for NFL WR Receiving Yards Prediction

This module implements rolling average and efficiency features for predicting
next-week receiving yards using Random Forest regression.

Key Design Principles:
1. All rolling calculations are GAME-INDEXED (not week-indexed)
2. Rolling windows use ONLY PRIOR games (no future data leakage)
3. -999 imputed values are EXCLUDED from rolling calculations
4. Rolling windows RESET at season boundaries
5. Minimum games threshold documented for valid rolling averages

Author: Claude Code
Created: 2024-11-25
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Imputation sentinel value - NEVER include in calculations
IMPUTATION_SENTINEL = -999

# Rolling window sizes based on EDA recommendations
ROLLING_WINDOWS = [3, 5]

# Minimum games required before rolling average is considered valid
# This addresses the cold-start problem for players with insufficient history
MIN_GAMES_FOR_VALID_ROLLING = {
    3: 2,  # 3-game rolling requires at least 2 prior games
    5: 3,  # 5-game rolling requires at least 3 prior games
}

# Features to create rolling averages for (Priority 5 from EDA)
ROLLING_FEATURE_CONFIGS = {
    # Game-level stats (plyr_gm_rec_*)
    'plyr_gm_rec_yds': {
        'description': 'Receiving yards per game',
        'has_imputation': False,
        'priority': 5
    },
    'plyr_gm_rec_tgt': {
        'description': 'Targets per game',
        'has_imputation': False,
        'priority': 5
    },
    'plyr_gm_rec': {
        'description': 'Receptions per game',
        'has_imputation': False,
        'priority': 5
    },
    'plyr_gm_rec_yac': {
        'description': 'Yards after catch per game',
        'has_imputation': False,
        'priority': 5
    },
    'plyr_gm_rec_first_dwn': {
        'description': 'First downs per game',
        'has_imputation': True,  # Has -999 values (7.07%)
        'priority': 4
    },
    'plyr_gm_rec_aybc': {
        'description': 'Air yards before catch per game',
        'has_imputation': False,
        'priority': 4
    },
    'plyr_gm_rec_td': {
        'description': 'Touchdowns per game',
        'has_imputation': False,
        'priority': 4
    },
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def mask_imputed_values(series: pd.Series, sentinel: float = IMPUTATION_SENTINEL) -> pd.Series:
    """
    Replace imputed sentinel values with NaN for exclusion from calculations.

    Args:
        series: Pandas series containing potentially imputed values
        sentinel: The sentinel value used for imputation (default: -999)

    Returns:
        Series with sentinel values replaced by NaN
    """
    return series.replace(sentinel, np.nan)


def validate_no_future_leakage(df: pd.DataFrame, feature_col: str,
                               target_col: str = 'next_week_rec_yds') -> bool:
    """
    Validate that a feature column does not contain future information.

    The feature for week N should only use data from weeks < N.

    Args:
        df: DataFrame with features and target
        feature_col: Name of feature column to validate
        target_col: Name of target column

    Returns:
        True if no leakage detected, False otherwise
    """
    # For each player-season, verify feature at week N doesn't correlate
    # suspiciously with target (which is week N+1 yards)
    # A very high correlation (>0.8) might indicate leakage

    correlation = df[feature_col].corr(df[target_col])

    if abs(correlation) > 0.8:
        logger.warning(
            f"Potential data leakage detected: {feature_col} has correlation "
            f"{correlation:.3f} with target. Please investigate."
        )
        return False

    return True


def get_game_sequence_number(df: pd.DataFrame) -> pd.Series:
    """
    Calculate game sequence number for each player within each season.

    This is GAME-INDEXED, not week-indexed, handling cases where players
    miss games (injuries, bye weeks, etc.).

    Args:
        df: DataFrame sorted by plyr_id, season_id, week_id

    Returns:
        Series with game sequence numbers (1-indexed)
    """
    return df.groupby(['plyr_id', 'season_id']).cumcount() + 1


# =============================================================================
# ROLLING FEATURE FUNCTIONS
# =============================================================================

def calculate_rolling_average_single_stat(
    df: pd.DataFrame,
    stat_col: str,
    window: int,
    exclude_imputed: bool = True
) -> pd.Series:
    """
    Calculate rolling average for a single statistic, handling edge cases.

    Edge Case Handling:
    - Cross-season boundaries: Windows reset at season start (groupby season)
    - Injury gaps: Uses game-based windows (not calendar-based)
    - Imputed nulls: -999 values excluded from calculations
    - Cold start: NaN for insufficient history (handled by min_periods)

    CRITICAL: Uses shift(1) to ensure only PRIOR games are included.
    The rolling window looks at games [N-window, N-1], NOT including game N.

    Args:
        df: DataFrame sorted by plyr_id, season_id, week_id
        stat_col: Name of statistic column
        window: Rolling window size (number of games)
        exclude_imputed: Whether to exclude -999 values (default: True)

    Returns:
        Series with rolling averages
    """
    # Create a copy of the stat column
    stat_series = df[stat_col].copy()

    # Mask imputed values if required
    if exclude_imputed:
        stat_series = mask_imputed_values(stat_series)

    # Calculate rolling mean within each player-season
    # IMPORTANT: We shift by 1 BEFORE rolling to ensure we only use prior games
    # This prevents any data leakage from the current game
    rolling_avg = (
        df.assign(_stat_shifted=stat_series.groupby([df['plyr_id'], df['season_id']]).shift(1))
        .groupby(['plyr_id', 'season_id'])['_stat_shifted']
        .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    )

    return rolling_avg


def build_rolling_features(
    df: pd.DataFrame,
    stat_cols: List[str] = None,
    windows: List[int] = None,
    exclude_imputed: bool = True
) -> pd.DataFrame:
    """
    Build rolling average features for multiple statistics and window sizes.

    This function creates game-indexed rolling averages that:
    - Reset at season boundaries
    - Exclude -999 imputed values from calculations
    - Use only PRIOR games (no future leakage)
    - Handle players with varying game counts

    Args:
        df: DataFrame with player game stats
        stat_cols: List of statistic columns to create rolling features for
                   If None, uses default ROLLING_FEATURE_CONFIGS
        windows: List of window sizes (default: [3, 5])
        exclude_imputed: Whether to exclude -999 values (default: True)

    Returns:
        DataFrame with original columns plus new rolling features
    """
    if stat_cols is None:
        stat_cols = list(ROLLING_FEATURE_CONFIGS.keys())

    if windows is None:
        windows = ROLLING_WINDOWS

    # Validate required columns exist
    required_cols = ['plyr_id', 'season_id', 'week_id'] + stat_cols
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Create copy of DataFrame to avoid modifying original
    result_df = df.copy()

    # Sort by temporal order for correct rolling calculations
    result_df = result_df.sort_values(['plyr_id', 'season_id', 'week_id']).reset_index(drop=True)

    # Add game sequence number for reference
    result_df['game_seq_num'] = get_game_sequence_number(result_df)

    features_created = []

    for stat_col in stat_cols:
        if stat_col not in result_df.columns:
            logger.warning(f"Column {stat_col} not found in DataFrame, skipping")
            continue

        # Check if this column has imputation
        has_imputation = ROLLING_FEATURE_CONFIGS.get(stat_col, {}).get('has_imputation', False)

        for window in windows:
            # Generate feature name
            feature_name = f"roll_{window}g_{stat_col.replace('plyr_gm_rec_', '').replace('plyr_gm_', '')}"

            logger.info(f"Creating rolling feature: {feature_name}")

            # Calculate rolling average
            result_df[feature_name] = calculate_rolling_average_single_stat(
                result_df,
                stat_col,
                window,
                exclude_imputed=(exclude_imputed and has_imputation)
            )

            features_created.append(feature_name)

            # Log imputation handling
            if has_imputation:
                imputed_count = (df[stat_col] == IMPUTATION_SENTINEL).sum()
                logger.info(f"  {stat_col}: {imputed_count} imputed values excluded from rolling calc")

    # Add minimum games indicator for cold start handling
    for window in windows:
        min_games_col = f"has_min_games_{window}g"
        min_required = MIN_GAMES_FOR_VALID_ROLLING.get(window, window - 1)
        result_df[min_games_col] = result_df['game_seq_num'] > min_required
        features_created.append(min_games_col)

    logger.info(f"Created {len(features_created)} rolling features: {features_created}")

    return result_df


# =============================================================================
# EFFICIENCY RATIO FEATURES
# =============================================================================

def build_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build efficiency ratio features from season cumulative statistics.

    These features normalize cumulative stats by games played and provide
    efficiency metrics that are less volatile than single-game stats.

    Features created:
    - season_targets_per_game: plyr_rec_tgt / plyr_rec_gm
    - season_yards_per_game: plyr_rec_yds / plyr_rec_gm
    - yards_per_reception: plyr_rec_yds / plyr_rec
    - yards_per_target: plyr_rec_yds / plyr_rec_tgt

    Edge Case Handling:
    - Division by zero: Returns NaN (handled by model or imputation later)
    - -999 values: Checks numerator/denominator, returns NaN if imputed

    Args:
        df: DataFrame with season cumulative stats

    Returns:
        DataFrame with original columns plus efficiency features
    """
    result_df = df.copy()

    features_created = []

    # Season targets per game
    # Normalized target volume by games played
    if 'plyr_rec_tgt' in df.columns and 'plyr_rec_gm' in df.columns:
        result_df['season_targets_per_game'] = np.where(
            (result_df['plyr_rec_gm'] > 0) &
            (result_df['plyr_rec_tgt'] != IMPUTATION_SENTINEL) &
            (result_df['plyr_rec_gm'] != IMPUTATION_SENTINEL),
            result_df['plyr_rec_tgt'] / result_df['plyr_rec_gm'],
            np.nan
        )
        features_created.append('season_targets_per_game')
        logger.info("Created season_targets_per_game")

    # Season yards per game
    # Normalized cumulative yards by games played
    if 'plyr_rec_yds' in df.columns and 'plyr_rec_gm' in df.columns:
        result_df['season_yards_per_game'] = np.where(
            (result_df['plyr_rec_gm'] > 0) &
            (result_df['plyr_rec_yds'] != IMPUTATION_SENTINEL) &
            (result_df['plyr_rec_gm'] != IMPUTATION_SENTINEL),
            result_df['plyr_rec_yds'] / result_df['plyr_rec_gm'],
            np.nan
        )
        features_created.append('season_yards_per_game')
        logger.info("Created season_yards_per_game")

    # Yards per reception
    # Big-play ability indicator
    if 'plyr_rec_yds' in df.columns and 'plyr_rec' in df.columns:
        result_df['yards_per_reception'] = np.where(
            (result_df['plyr_rec'] > 0) &
            (result_df['plyr_rec_yds'] != IMPUTATION_SENTINEL) &
            (result_df['plyr_rec'] != IMPUTATION_SENTINEL),
            result_df['plyr_rec_yds'] / result_df['plyr_rec'],
            np.nan
        )
        features_created.append('yards_per_reception')
        logger.info("Created yards_per_reception")

    # Yards per target
    # Overall efficiency metric
    if 'plyr_rec_yds' in df.columns and 'plyr_rec_tgt' in df.columns:
        result_df['yards_per_target'] = np.where(
            (result_df['plyr_rec_tgt'] > 0) &
            (result_df['plyr_rec_yds'] != IMPUTATION_SENTINEL) &
            (result_df['plyr_rec_tgt'] != IMPUTATION_SENTINEL),
            result_df['plyr_rec_yds'] / result_df['plyr_rec_tgt'],
            np.nan
        )
        features_created.append('yards_per_target')
        logger.info("Created yards_per_target")

    logger.info(f"Created {len(features_created)} efficiency features: {features_created}")

    return result_df


# =============================================================================
# MAIN FEATURE BUILDER CLASS
# =============================================================================

class RollingFeatureBuilder:
    """
    Main class for building rolling and efficiency features.

    This class provides a complete pipeline for feature engineering with:
    - Proper edge case handling
    - Validation and logging
    - Configurable feature sets
    - Output to parquet format

    Usage:
        builder = RollingFeatureBuilder()
        result_df = builder.build_features(input_df)
        builder.save_dataset(result_df, output_path)
    """

    def __init__(
        self,
        rolling_windows: List[int] = None,
        rolling_stat_cols: List[str] = None,
        exclude_imputed: bool = True,
        validate_leakage: bool = True
    ):
        """
        Initialize the feature builder.

        Args:
            rolling_windows: Window sizes for rolling features (default: [3, 5])
            rolling_stat_cols: Statistics to create rolling features for
            exclude_imputed: Whether to exclude -999 from rolling calculations
            validate_leakage: Whether to validate for data leakage
        """
        self.rolling_windows = rolling_windows or ROLLING_WINDOWS
        self.rolling_stat_cols = rolling_stat_cols or list(ROLLING_FEATURE_CONFIGS.keys())
        self.exclude_imputed = exclude_imputed
        self.validate_leakage = validate_leakage

        self.features_created = []
        self.validation_results = {}

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all features on the input DataFrame.

        Args:
            df: Input DataFrame with raw features

        Returns:
            DataFrame with all engineered features added
        """
        logger.info("=" * 60)
        logger.info("Starting feature engineering pipeline")
        logger.info("=" * 60)

        initial_cols = list(df.columns)

        # Step 1: Build rolling features
        logger.info("\nStep 1: Building rolling features...")
        result_df = build_rolling_features(
            df,
            stat_cols=self.rolling_stat_cols,
            windows=self.rolling_windows,
            exclude_imputed=self.exclude_imputed
        )

        # Step 2: Build efficiency features
        logger.info("\nStep 2: Building efficiency features...")
        result_df = build_efficiency_features(result_df)

        # Track created features
        self.features_created = [col for col in result_df.columns if col not in initial_cols]

        # Step 3: Validate no data leakage
        if self.validate_leakage:
            logger.info("\nStep 3: Validating data integrity...")
            self._validate_features(result_df)

        # Step 4: Generate summary
        self._generate_summary(result_df)

        return result_df

    def _validate_features(self, df: pd.DataFrame):
        """Validate features for data leakage and quality."""
        logger.info("Validating features for data leakage...")

        for feature in self.features_created:
            if feature.startswith('has_min_games') or feature == 'game_seq_num':
                continue

            if 'next_week_rec_yds' in df.columns:
                is_valid = validate_no_future_leakage(df, feature)
                self.validation_results[feature] = {
                    'no_leakage': is_valid,
                    'null_count': df[feature].isna().sum(),
                    'null_pct': df[feature].isna().mean() * 100
                }

        # Validate rolling windows don't include current game
        for window in self.rolling_windows:
            self._validate_rolling_window_exclusion(df, window)

    def _validate_rolling_window_exclusion(self, df: pd.DataFrame, window: int):
        """
        Validate that rolling window for game N excludes game N's stats.

        For a 3-game rolling window at game N, we should be averaging
        games [N-3, N-2, N-1], NOT including game N.
        """
        # Sample check: for first game of season, rolling should be NaN
        # (no prior games to average)
        first_game_mask = df['game_seq_num'] == 1

        for col in df.columns:
            if col.startswith(f'roll_{window}g_'):
                first_game_values = df.loc[first_game_mask, col]
                non_null_first_games = first_game_values.notna().sum()

                if non_null_first_games > 0:
                    logger.warning(
                        f"Feature {col} has {non_null_first_games} non-null values "
                        f"for first games of season. This may indicate data leakage."
                    )
                else:
                    logger.info(f"Feature {col} correctly has NaN for first games (no prior data)")

    def _generate_summary(self, df: pd.DataFrame):
        """Generate summary of feature engineering results."""
        logger.info("\n" + "=" * 60)
        logger.info("FEATURE ENGINEERING SUMMARY")
        logger.info("=" * 60)

        logger.info(f"\nDataset shape: {df.shape}")
        logger.info(f"Features created: {len(self.features_created)}")

        logger.info("\nNew features:")
        for feature in sorted(self.features_created):
            if feature in df.columns:
                null_pct = df[feature].isna().mean() * 100
                logger.info(f"  - {feature}: {null_pct:.1f}% null")

        # Rolling feature statistics
        logger.info("\nRolling feature statistics:")
        for window in self.rolling_windows:
            rolling_cols = [c for c in df.columns if c.startswith(f'roll_{window}g_')]
            if rolling_cols:
                avg_null = df[rolling_cols].isna().mean().mean() * 100
                logger.info(f"  {window}-game rolling: {len(rolling_cols)} features, {avg_null:.1f}% avg null rate")

        # Efficiency feature statistics
        efficiency_cols = ['season_targets_per_game', 'season_yards_per_game',
                          'yards_per_reception', 'yards_per_target']
        existing_eff_cols = [c for c in efficiency_cols if c in df.columns]
        if existing_eff_cols:
            avg_null = df[existing_eff_cols].isna().mean().mean() * 100
            logger.info(f"  Efficiency features: {len(existing_eff_cols)} features, {avg_null:.1f}% avg null rate")

        logger.info("\n" + "=" * 60)

    def save_dataset(
        self,
        df: pd.DataFrame,
        output_path: str,
        filename: str = None
    ) -> str:
        """
        Save the feature-engineered dataset to parquet.

        Args:
            df: DataFrame with features
            output_path: Directory to save to
            filename: Optional filename (default: auto-generated)

        Returns:
            Full path to saved file
        """
        from datetime import datetime

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"nfl_wr_features_rolling_{timestamp}.parquet"

        output_file = output_dir / filename

        df.to_parquet(output_file, compression='snappy', index=False)
        logger.info(f"Dataset saved to: {output_file}")

        return str(output_file)

    def get_feature_documentation(self) -> Dict:
        """
        Generate documentation for all created features.

        Returns:
            Dictionary with feature documentation
        """
        docs = {
            'rolling_features': {},
            'efficiency_features': {},
            'metadata_features': {}
        }

        # Rolling feature documentation
        for stat_col, config in ROLLING_FEATURE_CONFIGS.items():
            for window in self.rolling_windows:
                feature_name = f"roll_{window}g_{stat_col.replace('plyr_gm_rec_', '').replace('plyr_gm_', '')}"
                docs['rolling_features'][feature_name] = {
                    'description': f"{window}-game rolling average of {config['description']}",
                    'source_column': stat_col,
                    'window_size': window,
                    'min_games_required': MIN_GAMES_FOR_VALID_ROLLING.get(window, window - 1),
                    'handles_imputation': config.get('has_imputation', False),
                    'priority': config.get('priority', 4)
                }

        # Efficiency feature documentation
        docs['efficiency_features'] = {
            'season_targets_per_game': {
                'description': 'Season-to-date targets normalized by games played',
                'formula': 'plyr_rec_tgt / plyr_rec_gm',
                'handles_division_by_zero': True,
                'handles_imputation': True
            },
            'season_yards_per_game': {
                'description': 'Season-to-date receiving yards normalized by games played',
                'formula': 'plyr_rec_yds / plyr_rec_gm',
                'handles_division_by_zero': True,
                'handles_imputation': True
            },
            'yards_per_reception': {
                'description': 'Average yards gained per reception (big-play indicator)',
                'formula': 'plyr_rec_yds / plyr_rec',
                'handles_division_by_zero': True,
                'handles_imputation': True
            },
            'yards_per_target': {
                'description': 'Average yards gained per target (overall efficiency)',
                'formula': 'plyr_rec_yds / plyr_rec_tgt',
                'handles_division_by_zero': True,
                'handles_imputation': True
            }
        }

        # Metadata features
        docs['metadata_features'] = {
            'game_seq_num': {
                'description': 'Game sequence number within player-season (1-indexed)',
                'use_case': 'Track player game count, identify cold start situations'
            },
            'has_min_games_3g': {
                'description': 'Boolean indicating player has enough history for valid 3-game rolling average',
                'threshold': f'>= {MIN_GAMES_FOR_VALID_ROLLING[3]} prior games'
            },
            'has_min_games_5g': {
                'description': 'Boolean indicating player has enough history for valid 5-game rolling average',
                'threshold': f'>= {MIN_GAMES_FOR_VALID_ROLLING[5]} prior games'
            }
        }

        return docs


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def build_all_basic_features(
    df: pd.DataFrame,
    output_path: str = None,
    save: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to build all basic features in one call.

    This is the main entry point for the feature engineering pipeline.

    Args:
        df: Input DataFrame with raw features
        output_path: Path to save output (optional if save=False)
        save: Whether to save the output dataset

    Returns:
        Tuple of (feature-engineered DataFrame, feature documentation)
    """
    builder = RollingFeatureBuilder()
    result_df = builder.build_features(df)

    if save and output_path:
        builder.save_dataset(result_df, output_path)

    documentation = builder.get_feature_documentation()

    return result_df, documentation


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Example usage and validation
    import sys
    from pathlib import Path

    # Get project root
    project_root = Path(__file__).parent.parent.parent

    # Load dataset
    data_path = project_root / "data" / "processed"

    # Find most recent dataset
    parquet_files = list(data_path.glob("nfl_wr_receiving_yards_dataset_*.parquet"))
    if not parquet_files:
        print("No dataset found. Please run build_dataset.py first.")
        sys.exit(1)

    latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading dataset: {latest_file}")

    df = pd.read_parquet(latest_file)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Build features
    builder = RollingFeatureBuilder()
    result_df = builder.build_features(df)

    # Save result
    output_file = builder.save_dataset(result_df, str(data_path))
    print(f"\nFeature-engineered dataset saved to: {output_file}")

    # Print feature documentation
    docs = builder.get_feature_documentation()
    print("\n" + "=" * 60)
    print("FEATURE DOCUMENTATION")
    print("=" * 60)

    print("\nRolling Features:")
    for name, info in docs['rolling_features'].items():
        print(f"  {name}:")
        print(f"    Description: {info['description']}")
        print(f"    Min games required: {info['min_games_required']}")

    print("\nEfficiency Features:")
    for name, info in docs['efficiency_features'].items():
        print(f"  {name}:")
        print(f"    Description: {info['description']}")
        print(f"    Formula: {info['formula']}")
