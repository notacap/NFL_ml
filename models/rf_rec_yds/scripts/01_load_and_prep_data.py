"""
Data Loading and Preprocessing Script for NFL Receiving Yards Prediction

This script creates a training dataset for predicting next week's receiving yards.
It loads minimal data and creates basic features for an MVP baseline model.

Target: plyr_gm_rec_yds (player game receiving yards)

Author: Generated with Claude Code
Date: 2025-10-21
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils.path_manager import PathManager
from utils.data_loader import DataLoader


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(project_root / 'logs' / '01_load_and_prep_data.log')
    ]
)
logger = logging.getLogger(__name__)


def load_configurations(paths):
    """Load configuration files."""
    logger.info("Loading configuration files...")

    # Data config is already loaded by PathManager
    data_config = paths.data_config

    # Load model config
    model_config_path = paths.configs_dir / 'model_config.yaml'
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    logger.info("Configuration files loaded successfully")
    return data_config, model_config


def load_raw_data(loader, data_config):
    """Load all required raw data tables."""
    logger.info("=" * 60)
    logger.info("LOADING RAW DATA")
    logger.info("=" * 60)

    # Get seasons from config
    seasons = data_config['temporal']['historical_seasons']
    logger.info(f"Loading target data for seasons {seasons}...")

    # Load target variable (plyr_gm_rec with plyr_gm_rec_yds)
    target_df = loader.load_target(seasons=seasons)
    logger.info(f"Loaded {len(target_df):,} player-game records")
    logger.info(f"Target columns: {list(target_df.columns)}")

    # Load player metadata
    logger.info("\nLoading player metadata...")
    players_df = loader.load_table('players', seasons=seasons)
    logger.info(f"Loaded {len(players_df):,} player records")
    logger.info(f"Player columns: {list(players_df.columns)}")

    # Load game info
    logger.info("\nLoading game information...")
    game_df = loader.load_table('game', seasons=seasons)
    logger.info(f"Loaded {len(game_df):,} game records")
    logger.info(f"Game columns: {list(game_df.columns)}")

    # Load player season receiving stats (cumulative)
    logger.info("\nLoading player season receiving stats...")
    plyr_season_rec_df = loader.load_table('player_season_receiving', seasons=seasons)
    logger.info(f"Loaded {len(plyr_season_rec_df):,} player-season records")
    logger.info(f"Season stats columns: {list(plyr_season_rec_df.columns)}")

    logger.info("\n" + "=" * 60)
    logger.info("RAW DATA LOADING COMPLETE")
    logger.info("=" * 60 + "\n")

    return {
        'target': target_df,
        'players': players_df,
        'game': game_df,
        'season_stats': plyr_season_rec_df
    }


def merge_data(data_dict):
    """Merge all data tables into a single dataframe."""
    logger.info("=" * 60)
    logger.info("MERGING DATA TABLES")
    logger.info("=" * 60)

    df = data_dict['target'].copy()
    initial_rows = len(df)
    logger.info(f"Starting with target data: {initial_rows:,} rows")

    # Merge player metadata
    logger.info("\nMerging player metadata...")
    # Join on 'season' from both tables (not 'season_id')
    # season_id is a sequential ID (1, 2, 3), but season is the actual year (2022, 2023, 2024)
    df = df.merge(
        data_dict['players'][['plyr_id', 'season', 'plyr_name', 'plyr_pos',
                              'plyr_age', 'plyr_gm_played', 'team_id']],
        on=['plyr_id', 'season'],
        how='left',
        suffixes=('', '_player')
    )
    logger.info(f"After player merge: {len(df):,} rows ({len(df) - initial_rows:+,} change)")

    # Merge game info to get home/away status
    logger.info("\nMerging game information...")
    game_info = data_dict['game'][['game_id', 'home_team_id', 'away_team_id']].copy()
    df = df.merge(game_info, on='game_id', how='left')

    # Create is_home feature
    df['is_home'] = (df['team_id'] == df['home_team_id']).astype(int)
    logger.info(f"Created is_home feature (1=home, 0=away)")
    logger.info(f"  Home games: {df['is_home'].sum():,}")
    logger.info(f"  Away games: {(df['is_home'] == 0).sum():,}")

    # Drop temporary columns
    df.drop(columns=['home_team_id', 'away_team_id'], errors='ignore', inplace=True)

    logger.info("\n" + "=" * 60)
    logger.info(f"MERGE COMPLETE: {len(df):,} rows")
    logger.info("=" * 60 + "\n")

    return df


def filter_positions(df, positions=['WR', 'TE', 'RB']):
    """Filter to relevant receiving positions."""
    logger.info("=" * 60)
    logger.info("FILTERING TO RELEVANT POSITIONS")
    logger.info("=" * 60)

    initial_rows = len(df)
    logger.info(f"Before filtering: {initial_rows:,} rows")

    # Count by position before filtering
    logger.info("\nPosition distribution before filtering:")
    pos_counts = df['plyr_pos'].value_counts()
    for pos, count in pos_counts.items():
        logger.info(f"  {pos}: {count:,}")

    # Filter to specified positions
    df_filtered = df[df['plyr_pos'].isin(positions)].copy()

    logger.info(f"\nAfter filtering to {positions}: {len(df_filtered):,} rows")
    logger.info(f"Removed: {initial_rows - len(df_filtered):,} rows")

    # Count by position after filtering
    logger.info("\nPosition distribution after filtering:")
    pos_counts_after = df_filtered['plyr_pos'].value_counts()
    for pos, count in pos_counts_after.items():
        logger.info(f"  {pos}: {count:,}")

    logger.info("\n" + "=" * 60)
    logger.info("POSITION FILTERING COMPLETE")
    logger.info("=" * 60 + "\n")

    return df_filtered


def create_rolling_features(df, rolling_windows=[3, 5]):
    """
    Create rolling average features with proper temporal ordering.

    CRITICAL: Only use data from weeks 1 through N-1 to predict week N.
    """
    logger.info("=" * 60)
    logger.info("CREATING ROLLING AVERAGE FEATURES")
    logger.info("=" * 60)

    # Sort by player, season, week to ensure proper temporal ordering
    logger.info("\nSorting data by player, season, week...")
    df = df.sort_values(['plyr_id', 'season', 'week']).reset_index(drop=True)

    # Columns to create rolling features for
    rolling_cols = ['plyr_gm_rec_yds', 'plyr_gm_rec', 'plyr_gm_rec_tgt']

    logger.info(f"\nCreating rolling features for: {rolling_cols}")
    logger.info(f"Rolling windows: {rolling_windows}")

    # Group by player and season
    grouped = df.groupby(['plyr_id', 'season'])

    for col in rolling_cols:
        # Check if column exists
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in dataframe, skipping...")
            continue

        for window in rolling_windows:
            feature_name = f'{col}_roll_{window}'
            logger.info(f"  Creating {feature_name}...")

            # Calculate rolling average using only previous games (shift by 1)
            # This ensures we only use data from weeks 1 to N-1 for week N
            df[feature_name] = grouped[col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )

    logger.info("\n" + "=" * 60)
    logger.info("ROLLING FEATURES CREATED")
    logger.info("=" * 60 + "\n")

    return df


def create_season_features(df, data_dict):
    """Create season-to-date aggregate features."""
    logger.info("=" * 60)
    logger.info("CREATING SEASON AGGREGATE FEATURES")
    logger.info("=" * 60)

    # Merge season cumulative stats
    logger.info("\nMerging season-to-date statistics...")
    season_stats = data_dict['season_stats'].copy()

    # Select relevant columns from season stats
    season_cols = ['plyr_id', 'season', 'week', 'plyr_rec_yds', 'plyr_rec_tgt', 'plyr_rec']
    available_season_cols = [col for col in season_cols if col in season_stats.columns]

    if len(available_season_cols) < len(season_cols):
        missing = set(season_cols) - set(available_season_cols)
        logger.warning(f"Missing season stat columns: {missing}")

    # Merge season stats
    df = df.merge(
        season_stats[available_season_cols],
        on=['plyr_id', 'season', 'week'],
        how='left',
        suffixes=('', '_season')
    )

    # Create season-to-date averages
    logger.info("\nCreating season-to-date averages...")

    # Group by player and season
    grouped = df.groupby(['plyr_id', 'season'])

    # Calculate games played so far (cumulative count)
    df['games_played_season'] = grouped.cumcount() + 1

    # Calculate season-to-date averages using shift to avoid data leakage
    if 'plyr_rec_yds' in df.columns:
        df['rec_yds_per_game_season'] = grouped['plyr_rec_yds'].transform(
            lambda x: x.shift(1) / grouped.cumcount()
        )

    if 'plyr_rec_tgt' in df.columns:
        df['rec_tgt_per_game_season'] = grouped['plyr_rec_tgt'].transform(
            lambda x: x.shift(1) / grouped.cumcount()
        )

    if 'plyr_rec' in df.columns:
        df['rec_per_game_season'] = grouped['plyr_rec'].transform(
            lambda x: x.shift(1) / grouped.cumcount()
        )

    logger.info(f"Created season-to-date features:")
    logger.info(f"  - games_played_season")
    logger.info(f"  - rec_yds_per_game_season")
    logger.info(f"  - rec_tgt_per_game_season")
    logger.info(f"  - rec_per_game_season")

    logger.info("\n" + "=" * 60)
    logger.info("SEASON FEATURES CREATED")
    logger.info("=" * 60 + "\n")

    return df


def drop_unwanted_columns(df):
    """Drop columns that should not be used in the model."""
    logger.info("=" * 60)
    logger.info("DROPPING UNWANTED COLUMNS")
    logger.info("=" * 60)

    columns_to_drop = [
        'plyr_rec_aybc_route',
        'plyr_rec_succ_rt',
        'plyr_rec_yac_route'
    ]

    # Only drop columns that actually exist in the dataframe
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]

    if existing_cols_to_drop:
        logger.info(f"\nDropping {len(existing_cols_to_drop)} columns:")
        for col in existing_cols_to_drop:
            logger.info(f"  - {col}")
        df = df.drop(columns=existing_cols_to_drop)
    else:
        logger.info("\nNo matching columns found to drop")

    logger.info("\n" + "=" * 60)
    logger.info("COLUMN DROPPING COMPLETE")
    logger.info("=" * 60 + "\n")

    return df


def filter_min_games(df, min_games=3):
    """Filter out players with insufficient historical data."""
    logger.info("=" * 60)
    logger.info(f"FILTERING PLAYERS WITH < {min_games} GAMES")
    logger.info("=" * 60)

    initial_rows = len(df)
    logger.info(f"Before filtering: {initial_rows:,} rows")

    # For rolling features, we need at least min_games of prior data
    # Filter out the first few weeks where we don't have enough history
    df_filtered = df[df['games_played_season'] > min_games].copy()

    logger.info(f"After filtering: {len(df_filtered):,} rows")
    logger.info(f"Removed: {initial_rows - len(df_filtered):,} rows")

    logger.info("\n" + "=" * 60)
    logger.info("MIN GAMES FILTERING COMPLETE")
    logger.info("=" * 60 + "\n")

    return df_filtered


def prepare_final_dataset(df):
    """Prepare final dataset with selected columns and handle missing values."""
    logger.info("=" * 60)
    logger.info("PREPARING FINAL DATASET")
    logger.info("=" * 60)

    # Define feature columns
    feature_cols = [
        # Player metadata
        'plyr_id', 'plyr_name', 'season', 'week', 'game_id', 'team_id', 'plyr_pos',
        'plyr_age', 'games_played_season',

        # Game context
        'is_home',

        # Rolling averages (3 games)
        'plyr_gm_rec_yds_roll_3', 'plyr_gm_rec_roll_3', 'plyr_gm_rec_tgt_roll_3',

        # Rolling averages (5 games)
        'plyr_gm_rec_yds_roll_5', 'plyr_gm_rec_roll_5', 'plyr_gm_rec_tgt_roll_5',

        # Season-to-date averages
        'rec_yds_per_game_season', 'rec_tgt_per_game_season', 'rec_per_game_season',

        # Target variable
        'plyr_gm_rec_yds'
    ]

    # Select only columns that exist
    available_cols = [col for col in feature_cols if col in df.columns]
    missing_cols = set(feature_cols) - set(available_cols)

    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")

    df_final = df[available_cols].copy()

    # Handle missing values
    logger.info("\nHandling missing values...")
    missing_before = df_final.isnull().sum()
    logger.info("Missing values per column:")
    for col, count in missing_before.items():
        if count > 0:
            logger.info(f"  {col}: {count:,} ({count/len(df_final)*100:.2f}%)")

    # Fill missing numeric features with 0 (represents no prior history)
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['plyr_id', 'season', 'week', 'game_id', 'team_id']]

    for col in numeric_cols:
        if df_final[col].isnull().sum() > 0:
            df_final[col].fillna(0, inplace=True)

    logger.info("\nAfter filling missing values:")
    missing_after = df_final.isnull().sum().sum()
    logger.info(f"Total missing values: {missing_after}")

    logger.info("\n" + "=" * 60)
    logger.info("FINAL DATASET PREPARED")
    logger.info("=" * 60 + "\n")

    return df_final


def print_summary_statistics(df):
    """Print summary statistics for the final dataset."""
    logger.info("=" * 60)
    logger.info("DATASET SUMMARY STATISTICS")
    logger.info("=" * 60)

    logger.info(f"\nTotal rows: {len(df):,}")
    logger.info(f"Total columns: {len(df.columns)}")

    # Date range
    logger.info(f"\nSeason range: {df['season'].min()} - {df['season'].max()}")
    logger.info(f"Week range: {df['week'].min()} - {df['week'].max()}")

    # Unique counts
    logger.info(f"\nUnique players: {df['plyr_id'].nunique():,}")
    logger.info(f"Unique teams: {df['team_id'].nunique()}")
    logger.info(f"Unique games: {df['game_id'].nunique():,}")

    # Position breakdown
    logger.info("\nPosition breakdown:")
    for pos, count in df['plyr_pos'].value_counts().items():
        logger.info(f"  {pos}: {count:,} ({count/len(df)*100:.1f}%)")

    # Feature columns
    feature_cols = [col for col in df.columns if col not in
                   ['plyr_id', 'plyr_name', 'season', 'week', 'game_id', 'team_id', 'plyr_pos', 'plyr_gm_rec_yds']]
    logger.info(f"\nFeature columns created ({len(feature_cols)}):")
    for col in feature_cols:
        logger.info(f"  - {col}")

    # Target variable statistics
    logger.info(f"\nTarget variable (plyr_gm_rec_yds) statistics:")
    logger.info(f"  Mean: {df['plyr_gm_rec_yds'].mean():.2f}")
    logger.info(f"  Median: {df['plyr_gm_rec_yds'].median():.2f}")
    logger.info(f"  Std: {df['plyr_gm_rec_yds'].std():.2f}")
    logger.info(f"  Min: {df['plyr_gm_rec_yds'].min():.0f}")
    logger.info(f"  Max: {df['plyr_gm_rec_yds'].max():.0f}")

    logger.info("\n" + "=" * 60)

    return


def save_processed_data(df, paths):
    """Save processed data to parquet file."""
    logger.info("=" * 60)
    logger.info("SAVING PROCESSED DATA")
    logger.info("=" * 60)

    # Ensure processed directory exists
    processed_dir = paths.ensure_exists('data', 'processed')
    output_path = processed_dir / 'training_data_v1.parquet'

    logger.info(f"\nSaving to: {output_path}")

    # Save to parquet
    df.to_parquet(output_path, index=False, compression='snappy')

    file_size = output_path.stat().st_size / (1024 * 1024)  # Convert to MB
    logger.info(f"File saved successfully ({file_size:.2f} MB)")

    logger.info("\n" + "=" * 60)
    logger.info("SAVE COMPLETE")
    logger.info("=" * 60 + "\n")

    return output_path


def main():
    """Main execution function."""
    logger.info("\n" + "=" * 60)
    logger.info("NFL RECEIVING YARDS PREDICTION - DATA PREPARATION")
    logger.info("MVP Baseline Model - Training Data Creation")
    logger.info("=" * 60 + "\n")

    try:
        # Initialize PathManager and DataLoader
        logger.info("Initializing PathManager and DataLoader...")
        paths = PathManager()
        loader = DataLoader(paths)
        logger.info("Initialization complete\n")

        # Load configurations
        data_config, model_config = load_configurations(paths)
        min_games = data_config['processing'].get('min_games_threshold', 3)

        # Load raw data
        data_dict = load_raw_data(loader, data_config)

        # Merge data tables
        df = merge_data(data_dict)
        logger.info(f"Data shape after merge: {df.shape}")

        # Filter to relevant positions
        df = filter_positions(df)
        logger.info(f"Data shape after position filter: {df.shape}")

        # Create rolling features
        df = create_rolling_features(df)
        logger.info(f"Data shape after rolling features: {df.shape}")

        # Create season aggregate features
        df = create_season_features(df, data_dict)
        logger.info(f"Data shape after season features: {df.shape}")

        # Drop unwanted columns
        df = drop_unwanted_columns(df)
        logger.info(f"Data shape after dropping columns: {df.shape}")

        # Filter minimum games
        df = filter_min_games(df, min_games=min_games)
        logger.info(f"Data shape after min games filter: {df.shape}")

        # Prepare final dataset
        df_final = prepare_final_dataset(df)
        logger.info(f"Final data shape: {df_final.shape}")

        # Print summary statistics
        print_summary_statistics(df_final)

        # Save processed data
        output_path = save_processed_data(df_final, paths)

        logger.info("\n" + "=" * 60)
        logger.info("DATA PREPARATION COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"\nFinal dataset: {len(df_final):,} rows, {len(df_final.columns)} columns")
        logger.info(f"Saved to: {output_path}")
        logger.info("\n" + "=" * 60 + "\n")

    except Exception as e:
        logger.error(f"\nERROR: Data preparation failed!")
        logger.error(f"Error message: {str(e)}")
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    main()
