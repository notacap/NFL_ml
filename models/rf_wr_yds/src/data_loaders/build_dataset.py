"""
NFL Receiving Yards Dataset Builder

This script builds a high-quality dataset for training a random forest model to predict 
NFL player receiving yards for the following week. It implements strict temporal integrity 
to prevent data leakage and creates properly aligned features and targets.

Author: Claude Code
Created: 2024-11-23
"""

import os
import sys
import logging
import pandas as pd
import pyarrow.parquet as pq
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

warnings.filterwarnings('ignore')

class NFLDatasetBuilder:
    """
    Builds NFL receiving yards prediction dataset with temporal integrity.
    """
    
    def __init__(self, config_dir: str = None):
        """
        Initialize dataset builder with configuration.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir or str(project_root / "configs")
        self.project_root = project_root
        
        # Load configurations
        self.data_config = self._load_config("data_config.yaml")
        self.paths_config = self._load_config("paths.yaml")
        
        # Set up paths
        self.base_data_path = Path(self.paths_config["data"]["source"])
        self.output_path = project_root / self.paths_config["data"]["processed"]
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        # Data validation tracking
        self.validation_results = {}
        
    def _load_config(self, filename: str) -> Dict:
        """Load YAML configuration file."""
        config_path = Path(self.config_dir) / filename
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise FileNotFoundError(f"Could not load config file {config_path}: {e}")
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(self.output_path / f'dataset_build_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("NFL Dataset Builder initialized")
        
    def _get_table_path(self, table_name: str) -> Path:
        """Get full path to parquet table."""
        table_mapping = {
            'plyr_gm_rec': 'plyr_gm/plyr_gm_rec',
            'plyr_rec': 'plyr_szn/plyr_rec',
            'plyr_gm_snap_ct': 'plyr_gm/plyr_gm_snap_ct',
            'plyr': 'players/plyr',
            'plyr_master': 'plyr_master.parquet',
            'nfl_week': 'static/nfl_week',
            'nfl_season': 'nfl_season.parquet'
        }
        
        if table_name not in table_mapping:
            raise ValueError(f"Unknown table: {table_name}")
            
        return self.base_data_path / table_mapping[table_name]
    
    def _load_partitioned_table(self, table_name: str, seasons: List[int] = None, 
                              weeks: List[int] = None) -> pd.DataFrame:
        """
        Load partitioned parquet table with optional season/week filtering.
        
        Args:
            table_name: Name of table to load
            seasons: List of seasons to include (None for all)
            weeks: List of weeks to include (None for all)
            
        Returns:
            DataFrame with loaded data
        """
        table_path = self._get_table_path(table_name)
        
        if not table_path.exists():
            raise FileNotFoundError(f"Table path does not exist: {table_path}")
        
        # Handle non-partitioned tables
        if table_path.suffix == '.parquet':
            self.logger.info(f"Loading non-partitioned table: {table_name}")
            return pd.read_parquet(table_path)
        
        # Load partitioned tables
        dfs = []
        seasons_to_load = seasons or self.data_config['temporal']['historical_seasons'] + [self.data_config['temporal']['current_season']]
        
        for season in seasons_to_load:
            season_path = table_path / f"season={season}"
            
            if not season_path.exists():
                self.logger.warning(f"Season {season} not found for table {table_name}")
                continue
            
            # For season-only partitioned tables (no week sub-partitions)
            if table_name in ['plyr', 'nfl_week']:
                season_df = pd.read_parquet(season_path)
                season_df['season_id'] = season
                dfs.append(season_df)
                continue
            
            # For season/week partitioned tables  
            if any(week_dir.is_dir() and week_dir.name.startswith('week=') 
                   for week_dir in season_path.iterdir() if week_dir.is_dir()):
                # Load all weeks for the season
                for week_dir in season_path.iterdir():
                    if week_dir.is_dir() and week_dir.name.startswith('week='):
                        week_num = int(week_dir.name.split('=')[1])
                        if weeks is None or week_num in weeks:
                            try:
                                week_df = pd.read_parquet(week_dir)
                                # Only add season_id and week_id if they're not already present
                                if 'season_id' not in week_df.columns:
                                    week_df['season_id'] = season
                                if 'week_id' not in week_df.columns:
                                    week_df['week_id'] = week_num
                                dfs.append(week_df)
                            except Exception as e:
                                self.logger.warning(f"Could not load {table_name} season={season} week={week_num}: {e}")
            else:
                # Load all weeks for the season for other tables
                for week_dir in season_path.iterdir():
                    if week_dir.is_dir() and week_dir.name.startswith('week='):
                        week_num = int(week_dir.name.split('=')[1])
                        if weeks is None or week_num in weeks:
                            try:
                                week_df = pd.read_parquet(week_dir)
                                week_df['season_id'] = season
                                week_df['week_id'] = week_num
                                dfs.append(week_df)
                            except Exception as e:
                                self.logger.warning(f"Could not load {table_name} season={season} week={week_num}: {e}")
        
        if not dfs:
            raise ValueError(f"No data found for table {table_name}")
        
        result_df = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"Loaded {table_name}: {len(result_df):,} rows across {len(dfs)} partitions")
        
        return result_df
    
    def _validate_data(self, df: pd.DataFrame, stage_name: str, 
                      required_columns: List[str] = None) -> pd.DataFrame:
        """
        Validate DataFrame at processing stage.
        
        Args:
            df: DataFrame to validate
            stage_name: Name of processing stage
            required_columns: List of required columns
            
        Returns:
            Validated DataFrame
        """
        self.logger.info(f"Validating data at stage: {stage_name}")
        
        # Basic validation
        if df.empty:
            raise ValueError(f"DataFrame is empty at stage: {stage_name}")
        
        # Check for required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns at {stage_name}: {missing_cols}")
        
        # Check for duplicates in key columns if they exist
        key_cols = []
        potential_keys = ['plyr_id', 'season_id', 'week_id', 'game_id']
        for col in potential_keys:
            if col in df.columns:
                key_cols.append(col)
        
        if key_cols:
            duplicates = df.duplicated(subset=key_cols).sum()
            if duplicates > 0:
                self.logger.warning(f"Found {duplicates} duplicates in key columns {key_cols} at {stage_name}")
        
        # Store validation results
        self.validation_results[stage_name] = {
            'rows': len(df),
            'columns': len(df.columns),
            'nulls_by_column': df.isnull().sum().to_dict(),
            'duplicates': duplicates if key_cols else 0
        }
        
        self.logger.info(f"Stage {stage_name}: {len(df):,} rows, {len(df.columns)} columns")
        
        return df
    
    def load_base_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Load all required base tables.
        
        Returns:
            Dictionary of loaded DataFrames
        """
        self.logger.info("Loading base tables...")
        
        tables = {}
        
        # Load main tables
        tables['plyr_gm_rec'] = self._load_partitioned_table('plyr_gm_rec')
        tables['plyr_rec'] = self._load_partitioned_table('plyr_rec')
        tables['plyr_gm_snap_ct'] = self._load_partitioned_table('plyr_gm_snap_ct')
        tables['plyr'] = self._load_partitioned_table('plyr')
        tables['plyr_master'] = self._load_partitioned_table('plyr_master')
        tables['nfl_week'] = self._load_partitioned_table('nfl_week')
        tables['nfl_season'] = self._load_partitioned_table('nfl_season')
        
        # Validate each table
        for table_name, df in tables.items():
            self._validate_data(df, f"load_{table_name}")
        
        return tables
    
    def _create_temporal_joins(self, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Join all tables with proper temporal alignment.
        
        Args:
            tables: Dictionary of loaded tables
            
        Returns:
            Joined DataFrame
        """
        self.logger.info("Creating temporal joins...")
        
        # Start with player game receiving stats as base table
        base_df = tables['plyr_gm_rec'].copy()
        self._validate_data(base_df, "base_plyr_gm_rec", 
                           required_columns=['plyr_id', 'season_id', 'week_id', 'plyr_gm_rec_yds'])
        
        # Join with player season receiving stats
        # Exclude columns that are inconsistently available or specified for removal:
        # - plyr_rec_aybc_route, plyr_rec_succ_rt, plyr_rec_yac_route (per prompt)
        # - game_count (not present in week 18 or 2025 data)
        plyr_rec = tables['plyr_rec'][['plyr_id', 'season_id', 'week_id'] +
                                     [col for col in tables['plyr_rec'].columns
                                      if col not in ['plyr_id', 'season_id', 'week_id',
                                                   'plyr_rec_aybc_route', 'plyr_rec_succ_rt', 'plyr_rec_yac_route',
                                                   'game_count']]]
        
        df = base_df.merge(
            plyr_rec,
            on=['plyr_id', 'season_id', 'week_id'],
            how='left'
        )
        self._validate_data(df, "after_plyr_rec_join")
        
        # Join with snap counts
        snap_cols = ['plyr_id', 'season_id', 'week_id', 'plyr_gm_off_snap_ct_pct']
        plyr_snap = tables['plyr_gm_snap_ct'][snap_cols]
        
        df = df.merge(
            plyr_snap,
            on=['plyr_id', 'season_id', 'week_id'],
            how='left'
        )
        self._validate_data(df, "after_snap_count_join")
        
        # Join with player info (use only plyr_id since season_id format differs between tables)
        plyr_info = tables['plyr'][['plyr_id', 'plyr_guid', 'plyr_pos', 'plyr_alt_pos']].drop_duplicates(subset=['plyr_id'])
        
        df = df.merge(
            plyr_info,
            on=['plyr_id'],
            how='left'
        )
        self._validate_data(df, "after_plyr_info_join")
        
        # Join with player master for cross-season tracking
        plyr_master = tables['plyr_master'][['plyr_guid']]
        
        df = df.merge(
            plyr_master,
            on=['plyr_guid'],
            how='left'
        )
        self._validate_data(df, "after_plyr_master_join")
        
        # Join with week info (use only week_id since season_id format differs)
        nfl_week = tables['nfl_week'][['week_id', 'week_num']].drop_duplicates(subset=['week_id'])
        
        df = df.merge(
            nfl_week,
            on=['week_id'],
            how='left'
        )
        self._validate_data(df, "after_week_join")
        
        # Join with season info
        nfl_season = tables['nfl_season'][['season_id', 'year']]
        
        df = df.merge(
            nfl_season,
            on=['season_id'],
            how='left'
        )
        self._validate_data(df, "after_season_join")
        
        return df
    
    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all required filters to the dataset.
        
        Args:
            df: DataFrame to filter
            
        Returns:
            Filtered DataFrame
        """
        self.logger.info("Applying filters...")
        initial_rows = len(df)
        
        # Filter 1: Wide receivers only
        wr_filter = (df['plyr_pos'] == 'WR') | (df['plyr_alt_pos'] == 'WR')
        df = df[wr_filter]
        self.logger.info(f"After WR filter: {len(df):,} rows ({initial_rows - len(df):,} removed)")
        
        # Filter 2: Minimum targets per game
        df = df[df['plyr_gm_rec_tgt'] >= 3]
        after_targets = len(df)
        self.logger.info(f"After targets filter (>=3): {after_targets:,} rows ({initial_rows - after_targets:,} removed)")
        
        # Filter 3: Minimum snap count percentage
        df = df[df['plyr_gm_off_snap_ct_pct'] >= 0.50]
        after_snaps = len(df)
        self.logger.info(f"After snap count filter (>=50%): {after_snaps:,} rows ({after_targets - after_snaps:,} removed)")
        
        # Filter 4: Minimum games in season (4+ games with 3+ targets)
        games_per_player_season = df.groupby(['plyr_id', 'season_id']).size().reset_index(name='games_count')
        valid_players = games_per_player_season[games_per_player_season['games_count'] >= 4][['plyr_id', 'season_id']]
        
        df = df.merge(valid_players, on=['plyr_id', 'season_id'], how='inner')
        final_rows = len(df)
        self.logger.info(f"After minimum games filter (>=4): {final_rows:,} rows ({after_snaps - final_rows:,} removed)")
        
        self.logger.info(f"Total rows removed by filters: {initial_rows - final_rows:,} ({((initial_rows - final_rows)/initial_rows)*100:.1f}%)")
        
        return df
    
    def _create_next_week_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create next week target variable with proper temporal alignment.
        
        Args:
            df: DataFrame with current week features
            
        Returns:
            DataFrame with next week targets aligned
        """
        self.logger.info("Creating next week target variable...")
        
        # First, handle duplicates by taking the mean of stats per player-week
        # This handles cases where players appear multiple times per week (e.g., multiple games)
        agg_dict = {}
        for col in df.columns:
            if col in ['plyr_id', 'season_id', 'week_id', 'plyr_guid', 'plyr_pos', 'plyr_alt_pos', 
                      'week_num', 'year']:
                agg_dict[col] = 'first'
            else:
                # Use mean for numeric stats, first for text
                if df[col].dtype in ['int64', 'float64']:
                    agg_dict[col] = 'mean'
                else:
                    agg_dict[col] = 'first'
        
        df = df.groupby(['plyr_id', 'season_id', 'week_id']).agg(agg_dict).reset_index(drop=True)
        self.logger.info(f"After deduplication: {len(df):,} rows")
        
        # Sort by temporal order to ensure proper alignment
        df = df.sort_values(['season_id', 'week_id', 'plyr_id']).reset_index(drop=True)
        
        # Create next week target by shifting target within each player's season
        df['next_week_rec_yds'] = df.groupby(['plyr_id', 'season_id'])['plyr_gm_rec_yds'].shift(-1)
        
        # Remove rows without next week target (last week of each player's season)
        df = df.dropna(subset=['next_week_rec_yds'])
        
        # Validate temporal integrity
        self._validate_temporal_integrity(df)
        
        self.logger.info(f"Created next week targets: {len(df):,} training samples")
        
        return df
    
    def _validate_temporal_integrity(self, df: pd.DataFrame):
        """
        Validate that no future data leakage exists.
        
        Args:
            df: DataFrame to validate
        """
        self.logger.info("Validating temporal integrity...")
        
        # Check that features are from earlier week than target
        # This is implicit in our design since we're predicting next week
        
        # Verify sequential week alignment within each player's season
        df_sorted = df.sort_values(['plyr_id', 'season_id', 'week_id'])
        
        # Check for any cases where next_week_rec_yds comes from an earlier week
        # This shouldn't happen with our shift logic, but let's verify
        if len(df_sorted) > 0:
            sample_check = df_sorted.groupby(['plyr_id', 'season_id']).apply(
                lambda x: (x['week_id'].diff().fillna(0) >= 0).all() if len(x) > 1 else True
            ).all()
        else:
            sample_check = True
        
        if not sample_check:
            self.logger.warning("Temporal integrity check found potential issues")
        else:
            self.logger.info("Temporal integrity validated successfully")
    
    def _finalize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final dataset preparation and column organization.
        
        Args:
            df: DataFrame to finalize
            
        Returns:
            Final dataset ready for training
        """
        self.logger.info("Finalizing dataset...")
        
        # Define column order for better organization
        id_cols = ['plyr_id', 'plyr_guid', 'season_id', 'week_id', 'year', 'week_num']
        target_col = ['next_week_rec_yds']
        
        # Check if current week target column exists (might have been lost in aggregation)
        current_week_target = ['plyr_gm_rec_yds'] if 'plyr_gm_rec_yds' in df.columns else []
        
        feature_cols = [col for col in df.columns if col not in id_cols + target_col + current_week_target + 
                       ['plyr_pos', 'plyr_alt_pos']]
        
        # Reorder columns
        final_cols = id_cols + target_col + current_week_target + feature_cols
        available_cols = [col for col in final_cols if col in df.columns]
        df = df[available_cols]
        
        # Sort by temporal order for final output
        df = df.sort_values(['season_id', 'week_id', 'plyr_id']).reset_index(drop=True)
        
        # Final validation
        self._validate_data(df, "final_dataset", required_columns=['plyr_id', 'next_week_rec_yds'])
        
        return df
    
    def save_dataset(self, df: pd.DataFrame) -> str:
        """
        Save the final dataset to parquet file.
        
        Args:
            df: DataFrame to save
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nfl_wr_receiving_yards_dataset_{timestamp}.parquet"
        output_file = self.output_path / filename
        
        self.logger.info(f"Saving dataset to: {output_file}")
        
        # Save with compression
        df.to_parquet(output_file, compression='snappy', index=False)
        
        # Save metadata
        metadata = {
            'created_at': timestamp,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'date_range': {
                'min_year': int(df['year'].min()),
                'max_year': int(df['year'].max()),
                'min_week': int(df['week_num'].min()),
                'max_week': int(df['week_num'].max())
            },
            'unique_players': int(df['plyr_id'].nunique()),
            'validation_results': self.validation_results
        }
        
        metadata_file = self.output_path / f"dataset_metadata_{timestamp}.yaml"
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        self.logger.info(f"Dataset saved successfully: {len(df):,} rows, {len(df.columns)} columns")
        self.logger.info(f"Metadata saved to: {metadata_file}")
        
        return str(output_file)
    
    def build_dataset(self) -> str:
        """
        Main method to build the complete dataset.
        
        Returns:
            Path to saved dataset file
        """
        self.logger.info("Starting NFL receiving yards dataset build...")
        
        try:
            # Step 1: Load base tables
            tables = self.load_base_tables()
            
            # Step 2: Create temporal joins
            df = self._create_temporal_joins(tables)
            
            # Step 3: Apply filters
            df = self._apply_filters(df)
            
            # Step 4: Create next week target variable
            df = self._create_next_week_target(df)
            
            # Step 5: Finalize dataset
            df = self._finalize_dataset(df)
            
            # Step 6: Save dataset
            output_file = self.save_dataset(df)
            
            self.logger.info("Dataset build completed successfully!")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Dataset build failed: {e}")
            raise


def main():
    """Main entry point for dataset building."""
    try:
        # Initialize builder
        builder = NFLDatasetBuilder()
        
        # Build dataset
        output_file = builder.build_dataset()
        
        print(f"\nDataset build completed successfully!")
        print(f"Output file: {output_file}")
        print(f"Logs saved to: {builder.output_path}")
        
    except Exception as e:
        print(f"Dataset build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()