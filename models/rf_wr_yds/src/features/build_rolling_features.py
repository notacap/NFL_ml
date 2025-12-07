"""
Rolling Feature Builder for NFL WR Receiving Yards Prediction

This module implements rolling average and efficiency features for predicting
next-week receiving yards using Random Forest regression.

Key Design Principles:
1. All rolling calculations are GAME-INDEXED (not week-indexed)
2. Rolling windows use ONLY PRIOR games (no future data leakage)
3. -999 imputed values are EXCLUDED from rolling calculations
4. Cross-season carryover: Rolling windows carry forward from prior season
   (reduces cold-start nulls for returning players)
5. Expanding window fallback: Uses all available prior games when insufficient
   history exists for full rolling window
6. Staleness indicators track when using cross-season data

Note: NaN imputation is handled separately in the imputation module.

Author: Claude Code
Created: 2024-11-25
Updated: 2024-12-04 - Added cross-season carryover and expanding window strategies
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

# Minimum games required before rolling average is considered "fully valid"
# With cross-season carryover and expanding windows, we still create rolling
# features for fewer games, but these flags indicate confidence level
MIN_GAMES_FOR_VALID_ROLLING = {
    3: 2,  # 3-game rolling ideally needs 2 prior games for stability
    5: 3,  # 5-game rolling ideally needs 3 prior games for stability
}

# Cross-season carryover configuration
CROSS_SEASON_CARRYOVER_ENABLED = True  # Enable carrying forward prior season data
MAX_CARRYOVER_GAMES = 5  # Maximum games to carry from prior season

# Player identifier for cross-season tracking
# plyr_id is season-specific, plyr_guid is persistent across seasons
CROSS_SEASON_PLAYER_ID = 'plyr_guid'  # Use plyr_guid for cross-season matching

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
    # NFL FastR advanced metrics (plyr_gm_rec_*)
    # Note: The following columns were removed due to excessive null values (5500+):
    # - plyr_gm_rec_avg_cushion, plyr_gm_rec_avg_separation, plyr_gm_rec_avg_yac,
    # - plyr_gm_rec_avg_expected_yac, plyr_gm_rec_avg_yac_above_expectation,
    # - plyr_gm_rec_pct_share_of_intended_ay
    'plyr_gm_rec_tgt_share': {
        'description': 'Target share (percentage of team targets)',
        'has_imputation': True,
        'priority': 5
    },
    'plyr_gm_rec_epa': {
        'description': 'Expected points added from receptions',
        'has_imputation': True,
        'priority': 4
    },
    'plyr_gm_rec_ay_share': {
        'description': 'Air yards share (percentage of team air yards)',
        'has_imputation': True,
        'priority': 4
    },
    'plyr_gm_rec_wopr': {
        'description': 'Weighted Opportunity Rating (1.5*tgt_share + 0.7*ay_share)',
        'has_imputation': True,
        'priority': 5
    },
    'plyr_gm_rec_racr': {
        'description': 'Receiver Air Conversion Ratio (rec_yds / air_yds)',
        'has_imputation': True,
        'priority': 4
    },
}

# Rolling efficiency features computed from game-level rolling averages
# These are safe for cross-season calculations
ROLLING_EFFICIENCY_CONFIGS = {
    'yds_per_tgt': {
        'description': 'Yards per target (rolling efficiency)',
        'numerator': 'yds',      # Will be prefixed with roll_Xg_
        'denominator': 'tgt',    # Will be prefixed with roll_Xg_
    },
    'yds_per_rec': {
        'description': 'Yards per reception (rolling efficiency)',
        'numerator': 'yds',
        'denominator': 'rec',
    },
    'catch_rate': {
        'description': 'Catch rate (rolling efficiency)',
        'numerator': 'rec',
        'denominator': 'tgt',
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


def get_career_game_sequence_number(df: pd.DataFrame) -> pd.Series:
    """
    Calculate game sequence number for each player across ALL seasons.

    Uses plyr_guid (persistent ID) instead of plyr_id (season-specific ID)
    to properly track career games across season boundaries.

    Args:
        df: DataFrame sorted by plyr_guid, season_id, week_id

    Returns:
        Series with career game sequence numbers (1-indexed)
    """
    player_col = CROSS_SEASON_PLAYER_ID if CROSS_SEASON_PLAYER_ID in df.columns else 'plyr_id'
    return df.groupby(player_col).cumcount() + 1


def calculate_days_since_last_game(df: pd.DataFrame) -> pd.Series:
    """
    Calculate days elapsed since player's last game.

    Useful for identifying cross-season gaps and staleness of rolling data.
    Requires 'game_date' or similar date column, or approximates from week/season.

    Args:
        df: DataFrame sorted by plyr_id, season_id, week_id

    Returns:
        Series with days since last game (NaN for first game)
    """
    # Create approximate game date from season and week
    # Assuming season starts around week 1 of September
    # Each week is ~7 days apart
    if 'year' in df.columns and 'week_num' in df.columns:
        # Approximate: Season starts Sept 1, each week adds 7 days
        approx_date = pd.to_datetime(
            df['year'].astype(str) + '-09-01'
        ) + pd.to_timedelta(df['week_num'] * 7, unit='D')
    elif 'season_id' in df.columns and 'week_id' in df.columns:
        # Fallback: use sequential numbering
        # This won't give actual days but will flag cross-season gaps
        approx_date = df['season_id'] * 100 + df['week_id']
        approx_date = pd.to_datetime('2020-01-01') + pd.to_timedelta(approx_date, unit='D')
    else:
        logger.warning("Cannot calculate days_since_last_game: missing date columns")
        return pd.Series(np.nan, index=df.index)

    # Calculate days since previous game for each player (using persistent ID)
    player_col = CROSS_SEASON_PLAYER_ID if CROSS_SEASON_PLAYER_ID in df.columns else 'plyr_id'
    days_since = approx_date.groupby(df[player_col]).diff().dt.days

    return days_since


def identify_season_carryover(df: pd.DataFrame) -> pd.Series:
    """
    Identify rows where rolling features use cross-season data.

    A row is marked as using carryover if:
    - It's in the first few games of a season (game_seq_num <= window size)
    - The player has prior season data

    Args:
        df: DataFrame with game_seq_num and career_game_seq_num columns

    Returns:
        Boolean series (True = using cross-season carryover data)
    """
    if 'game_seq_num' not in df.columns or 'career_game_seq_num' not in df.columns:
        logger.warning("Cannot identify carryover: missing sequence columns")
        return pd.Series(False, index=df.index)

    # Using carryover if: early in season AND have prior career games
    max_window = max(ROLLING_WINDOWS)
    is_early_season = df['game_seq_num'] <= max_window
    has_prior_seasons = df['career_game_seq_num'] > df['game_seq_num']

    return is_early_season & has_prior_seasons


# =============================================================================
# ROLLING FEATURE FUNCTIONS
# =============================================================================

def calculate_rolling_average_single_stat(
    df: pd.DataFrame,
    stat_col: str,
    window: int,
    exclude_imputed: bool = True,
    cross_season_carryover: bool = True
) -> pd.Series:
    """
    Calculate rolling average for a single statistic, handling edge cases.

    Edge Case Handling:
    - Cross-season carryover: Windows carry forward from prior season (if enabled)
    - Expanding window fallback: Uses all available prior games when < window games exist
    - Injury gaps: Uses game-based windows (not calendar-based)
    - Imputed nulls: -999 values excluded from calculations
    - Cold start: Only truly new players (rookies) have NaN for first game

    CRITICAL: Uses shift(1) to ensure only PRIOR games are included.
    The rolling window looks at games [N-window, N-1], NOT including game N.

    Args:
        df: DataFrame sorted by plyr_id, season_id, week_id
        stat_col: Name of statistic column
        window: Rolling window size (number of games)
        exclude_imputed: Whether to exclude -999 values (default: True)
        cross_season_carryover: Whether to carry data across seasons (default: True)

    Returns:
        Series with rolling averages
    """
    # Create a copy of the stat column
    stat_series = df[stat_col].copy()

    # Mask imputed values if required
    if exclude_imputed:
        stat_series = mask_imputed_values(stat_series)

    if cross_season_carryover and CROSS_SEASON_CARRYOVER_ENABLED:
        # Cross-season carryover: Group by persistent player ID (plyr_guid)
        # This allows rolling windows to span season boundaries
        # IMPORTANT: We shift by 1 BEFORE rolling to ensure we only use prior games
        player_col = CROSS_SEASON_PLAYER_ID if CROSS_SEASON_PLAYER_ID in df.columns else 'plyr_id'
        rolling_avg = (
            df.assign(_stat_shifted=stat_series.groupby(df[player_col]).shift(1))
            .groupby(player_col)['_stat_shifted']
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )
    else:
        # Original behavior: Reset at season boundaries
        # Calculate rolling mean within each player-season
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
    exclude_imputed: bool = True,
    cross_season_carryover: bool = True
) -> pd.DataFrame:
    """
    Build rolling average features for multiple statistics and window sizes.

    This function creates game-indexed rolling averages that:
    - Carry forward from prior seasons (cross-season carryover)
    - Use expanding windows when insufficient history exists
    - Exclude -999 imputed values from calculations
    - Use only PRIOR games (no future leakage)
    - Track staleness indicators for cross-season data

    Args:
        df: DataFrame with player game stats
        stat_cols: List of statistic columns to create rolling features for
                   If None, uses default ROLLING_FEATURE_CONFIGS
        windows: List of window sizes (default: [3, 5])
        exclude_imputed: Whether to exclude -999 values (default: True)
        cross_season_carryover: Whether to carry data across seasons (default: True)

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

    # Determine player ID column for cross-season tracking
    player_col = CROSS_SEASON_PLAYER_ID if (cross_season_carryover and CROSS_SEASON_PLAYER_ID in df.columns) else 'plyr_id'

    # Sort by temporal order for correct rolling calculations
    # Use persistent player ID (plyr_guid) for cross-season carryover
    result_df = result_df.sort_values([player_col, 'season_id', 'week_id']).reset_index(drop=True)

    # Add game sequence numbers
    result_df['game_seq_num'] = get_game_sequence_number(result_df)
    result_df['career_game_seq_num'] = get_career_game_sequence_number(result_df)

    # Add staleness indicators
    result_df['days_since_last_game'] = calculate_days_since_last_game(result_df)
    result_df['is_season_carryover'] = identify_season_carryover(result_df)

    features_created = ['career_game_seq_num', 'days_since_last_game', 'is_season_carryover']

    # Log carryover statistics
    if cross_season_carryover:
        carryover_count = result_df['is_season_carryover'].sum()
        total_count = len(result_df)
        logger.info(f"Cross-season carryover enabled: {carryover_count}/{total_count} rows ({carryover_count/total_count*100:.1f}%) use prior season data")

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

            # Calculate rolling average with cross-season carryover
            result_df[feature_name] = calculate_rolling_average_single_stat(
                result_df,
                stat_col,
                window,
                exclude_imputed=(exclude_imputed and has_imputation),
                cross_season_carryover=cross_season_carryover
            )

            features_created.append(feature_name)

            # Log imputation handling
            if has_imputation:
                imputed_count = (df[stat_col] == IMPUTATION_SENTINEL).sum()
                logger.info(f"  {stat_col}: {imputed_count} imputed values excluded from rolling calc")

    # Add minimum games indicators for cold start handling
    # With carryover, these now check career games, not just season games
    for window in windows:
        min_games_col = f"has_min_games_{window}g"
        min_required = MIN_GAMES_FOR_VALID_ROLLING.get(window, window - 1)
        if cross_season_carryover:
            # Check career game count (includes prior seasons)
            result_df[min_games_col] = result_df['career_game_seq_num'] > min_required
        else:
            # Original behavior: check season game count only
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


def build_rolling_efficiency_features(
    df: pd.DataFrame,
    windows: List[int] = None
) -> pd.DataFrame:
    """
    Build efficiency features from game-level rolling averages.

    These features are safe for cross-season calculations because they're
    computed from per-game normalized statistics that share the same
    temporal window.

    MUST be called AFTER build_rolling_features() which creates the
    base rolling averages (roll_Xg_yds, roll_Xg_tgt, roll_Xg_rec, etc.)

    Args:
        df: DataFrame with rolling average features already computed
        windows: Rolling window sizes to compute efficiency for (default: [3, 5])

    Returns:
        DataFrame with rolling efficiency features added

    Features created (for each window size):
        - roll_Xg_yds_per_tgt: Rolling yards per target
        - roll_Xg_yds_per_rec: Rolling yards per reception
        - roll_Xg_catch_rate: Rolling catch rate (receptions / targets)
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    result_df = df.copy()
    features_created = []

    for window in windows:
        prefix = f'roll_{window}g'

        # Define source columns (match naming convention from build_rolling_features)
        yds_col = f'{prefix}_yds'
        tgt_col = f'{prefix}_tgt'
        rec_col = f'{prefix}_rec'

        # Yards per target (rolling efficiency)
        if yds_col in result_df.columns and tgt_col in result_df.columns:
            feature_name = f'{prefix}_yds_per_tgt'
            result_df[feature_name] = np.where(
                result_df[tgt_col] > 0,
                result_df[yds_col] / result_df[tgt_col],
                np.nan
            )
            features_created.append(feature_name)
            logger.info(f"Created {feature_name}")

        # Yards per reception (rolling efficiency)
        if yds_col in result_df.columns and rec_col in result_df.columns:
            feature_name = f'{prefix}_yds_per_rec'
            result_df[feature_name] = np.where(
                result_df[rec_col] > 0,
                result_df[yds_col] / result_df[rec_col],
                np.nan
            )
            features_created.append(feature_name)
            logger.info(f"Created {feature_name}")

        # Catch rate (rolling efficiency)
        if rec_col in result_df.columns and tgt_col in result_df.columns:
            feature_name = f'{prefix}_catch_rate'
            result_df[feature_name] = np.where(
                result_df[tgt_col] > 0,
                result_df[rec_col] / result_df[tgt_col],
                np.nan
            )
            features_created.append(feature_name)
            logger.info(f"Created {feature_name}")

    logger.info(f"Created {len(features_created)} rolling efficiency features: {features_created}")

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
        validate_leakage: bool = True,
        cross_season_carryover: bool = True
    ):
        """
        Initialize the feature builder.

        Args:
            rolling_windows: Window sizes for rolling features (default: [3, 5])
            rolling_stat_cols: Statistics to create rolling features for
            exclude_imputed: Whether to exclude -999 from rolling calculations
            validate_leakage: Whether to validate for data leakage
            cross_season_carryover: Whether to carry forward prior season data (default: True)
        """
        self.rolling_windows = rolling_windows or ROLLING_WINDOWS
        self.rolling_stat_cols = rolling_stat_cols or list(ROLLING_FEATURE_CONFIGS.keys())
        self.exclude_imputed = exclude_imputed
        self.validate_leakage = validate_leakage
        self.cross_season_carryover = cross_season_carryover

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
        logger.info(f"Cross-season carryover: {'ENABLED' if self.cross_season_carryover else 'DISABLED'}")
        result_df = build_rolling_features(
            df,
            stat_cols=self.rolling_stat_cols,
            windows=self.rolling_windows,
            exclude_imputed=self.exclude_imputed,
            cross_season_carryover=self.cross_season_carryover
        )

        # Step 2: Build rolling efficiency features (derived from rolling averages)
        logger.info("\nStep 2: Building rolling efficiency features...")
        result_df = build_rolling_efficiency_features(result_df, windows=self.rolling_windows)

        # Step 3: Build season cumulative efficiency features (with deprecation warning)
        logger.info("\nStep 3: Building season cumulative efficiency features...")
        logger.warning("NOTE: Season cumulative efficiency features (yards_per_target, etc.) "
                      "are NOT safe for cross-season analysis. Prefer roll_Xg_yds_per_tgt instead.")
        result_df = build_efficiency_features(result_df)

        # Track created features
        self.features_created = [col for col in result_df.columns if col not in initial_cols]

        # Step 4: Validate no data leakage
        if self.validate_leakage:
            logger.info("\nStep 4: Validating data integrity...")
            self._validate_features(result_df)

        # Step 5: Generate summary
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

        With cross-season carryover enabled:
        - First game of SEASON may have data (from prior season)
        - First game of CAREER should still be NaN (truly new player)
        """
        if CROSS_SEASON_CARRYOVER_ENABLED:
            # With carryover: check first CAREER game, not first season game
            first_career_game_mask = df['career_game_seq_num'] == 1
            check_mask = first_career_game_mask
            check_description = "first career games (rookies)"
        else:
            # Original: check first season game
            first_game_mask = df['game_seq_num'] == 1
            check_mask = first_game_mask
            check_description = "first games of season"

        for col in df.columns:
            if col.startswith(f'roll_{window}g_'):
                first_game_values = df.loc[check_mask, col]
                non_null_first_games = first_game_values.notna().sum()

                if non_null_first_games > 0:
                    logger.warning(
                        f"Feature {col} has {non_null_first_games} non-null values "
                        f"for {check_description}. This may indicate data leakage."
                    )
                else:
                    logger.info(f"Feature {col} correctly has NaN for {check_description} (no prior data)")

        # Additional validation: check carryover games have data
        if CROSS_SEASON_CARRYOVER_ENABLED and 'is_season_carryover' in df.columns:
            carryover_mask = df['is_season_carryover'] == True
            for col in df.columns:
                if col.startswith(f'roll_{window}g_'):
                    carryover_values = df.loc[carryover_mask, col]
                    carryover_with_data = carryover_values.notna().sum()
                    carryover_total = carryover_mask.sum()
                    if carryover_total > 0:
                        pct_with_data = carryover_with_data / carryover_total * 100
                        logger.info(f"Feature {col}: {pct_with_data:.1f}% of carryover games have data")

    def _generate_summary(self, df: pd.DataFrame):
        """Generate summary of feature engineering results."""
        logger.info("\n" + "=" * 60)
        logger.info("FEATURE ENGINEERING SUMMARY")
        logger.info("=" * 60)

        logger.info(f"\nDataset shape: {df.shape}")
        logger.info(f"Features created: {len(self.features_created)}")

        # Cross-season carryover statistics
        if self.cross_season_carryover and 'is_season_carryover' in df.columns:
            logger.info("\nCross-season carryover statistics:")
            carryover_count = df['is_season_carryover'].sum()
            total_count = len(df)
            logger.info(f"  Rows using prior season data: {carryover_count}/{total_count} ({carryover_count/total_count*100:.1f}%)")

            # Count true rookies (first career game)
            if 'career_game_seq_num' in df.columns:
                rookie_first_games = (df['career_game_seq_num'] == 1).sum()
                logger.info(f"  True rookie first games (only nulls remaining): {rookie_first_games}")

            # Staleness statistics
            if 'days_since_last_game' in df.columns:
                cross_season_games = df[df['is_season_carryover'] == True]
                if len(cross_season_games) > 0:
                    avg_days = cross_season_games['days_since_last_game'].mean()
                    max_days = cross_season_games['days_since_last_game'].max()
                    logger.info(f"  Avg days since last game (carryover rows): {avg_days:.0f} days")
                    logger.info(f"  Max days since last game (carryover rows): {max_days:.0f} days")

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

        # Rolling efficiency feature statistics
        rolling_eff_cols = [c for c in df.columns if '_yds_per_tgt' in c or '_yds_per_rec' in c or '_catch_rate' in c]
        rolling_eff_cols = [c for c in rolling_eff_cols if c.startswith('roll_')]
        if rolling_eff_cols:
            avg_null = df[rolling_eff_cols].isna().mean().mean() * 100
            logger.info(f"  Rolling efficiency features: {len(rolling_eff_cols)} features, {avg_null:.1f}% avg null rate")

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

        # Rolling efficiency feature documentation
        docs['rolling_efficiency_features'] = {}
        for window in self.rolling_windows:
            prefix = f'roll_{window}g'
            docs['rolling_efficiency_features'][f'{prefix}_yds_per_tgt'] = {
                'description': f'{window}-game rolling yards per target (cross-season safe)',
                'formula': f'{prefix}_yds / {prefix}_tgt',
                'handles_division_by_zero': True,
                'cross_season_safe': True,
            }
            docs['rolling_efficiency_features'][f'{prefix}_yds_per_rec'] = {
                'description': f'{window}-game rolling yards per reception (cross-season safe)',
                'formula': f'{prefix}_yds / {prefix}_rec',
                'handles_division_by_zero': True,
                'cross_season_safe': True,
            }
            docs['rolling_efficiency_features'][f'{prefix}_catch_rate'] = {
                'description': f'{window}-game rolling catch rate (cross-season safe)',
                'formula': f'{prefix}_rec / {prefix}_tgt',
                'handles_division_by_zero': True,
                'cross_season_safe': True,
            }

        # Metadata features
        docs['metadata_features'] = {
            'game_seq_num': {
                'description': 'Game sequence number within player-season (1-indexed)',
                'use_case': 'Track player game count within current season'
            },
            'career_game_seq_num': {
                'description': 'Game sequence number across all seasons (1-indexed)',
                'use_case': 'Track total career games, identify true rookies'
            },
            'days_since_last_game': {
                'description': 'Approximate days elapsed since player last game',
                'use_case': 'Measure staleness of rolling data, especially for cross-season carryover'
            },
            'is_season_carryover': {
                'description': 'Boolean indicating rolling features use prior season data',
                'use_case': 'Flag stale data for model to potentially discount'
            },
            'has_min_games_3g': {
                'description': 'Boolean indicating player has enough history for stable 3-game rolling average',
                'threshold': f'>= {MIN_GAMES_FOR_VALID_ROLLING[3]} prior games (career-wide with carryover)'
            },
            'has_min_games_5g': {
                'description': 'Boolean indicating player has enough history for stable 5-game rolling average',
                'threshold': f'>= {MIN_GAMES_FOR_VALID_ROLLING[5]} prior games (career-wide with carryover)'
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
