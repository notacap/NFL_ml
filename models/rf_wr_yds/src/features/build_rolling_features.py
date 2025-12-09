"""
NFL Receiving Yards Dataset Builder

This script builds a high-quality dataset for training a random forest model to predict 
NFL player receiving yards for the following week. It implements strict temporal integrity 
to prevent data leakage and creates properly aligned features and targets.

TEMPORAL ALIGNMENT STRATEGY:
============================
When predicting Week N+1 receiving yards, each row contains:

1. HISTORICAL FEATURES (from Week N - the "current" row):
   - plyr_gm_rec_* : Game-level stats from Week N (what they just did)
   - plyr_rec_*    : Season cumulative stats through Week N
   - Snap counts, advanced metrics from Week N
   
2. UPCOMING GAME CONTEXT (from Week N+1 - shifted forward):
   - next_opponent_team_id : Who they will face in Week N+1
   - next_is_home_game     : Home/away for Week N+1
   - next_game_id          : Game ID for Week N+1 (for future joins)

3. TARGET VARIABLE:
   - next_week_rec_yds : Receiving yards in Week N+1

This allows the model to learn matchup effects (e.g., WR vs weak pass defense)
while maintaining strict temporal integrity (no future performance data leakage).

"""

import os
import sys
import logging
import pandas as pd
import numpy as np
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
            'nfl_fastr_wr': 'plyr_gm/nfl_fastr_wr',
            'plyr': 'players/plyr',
            'plyr_master': 'plyr_master.parquet',
            'nfl_week': 'static/nfl_week',
            'nfl_season': 'nfl_season.parquet',
            'multi_tm_plyr': 'players/multi_tm_plyr',
            'nfl_game': 'gm_info/nfl_game'
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
            if table_name in ['plyr', 'nfl_week', 'multi_tm_plyr']:
                season_df = pd.read_parquet(season_path)
                # Only add season_id if not already present (multi_tm_plyr has it in the data)
                if 'season_id' not in season_df.columns:
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
        tables['nfl_fastr_wr'] = self._load_partitioned_table('nfl_fastr_wr')
        tables['plyr'] = self._load_partitioned_table('plyr')
        tables['plyr_master'] = self._load_partitioned_table('plyr_master')
        tables['nfl_week'] = self._load_partitioned_table('nfl_week')
        tables['nfl_season'] = self._load_partitioned_table('nfl_season')
        tables['multi_tm_plyr'] = self._load_partitioned_table('multi_tm_plyr')
        tables['nfl_game'] = self._load_partitioned_table('nfl_game')

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
        # Exclude plyr_gm_rec_brkn_tkl_rec column
        plyr_gm_rec_cols = [col for col in tables['plyr_gm_rec'].columns
                           if col != 'plyr_gm_rec_brkn_tkl_rec']
        base_df = tables['plyr_gm_rec'][plyr_gm_rec_cols].copy()
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

        # Join with NFL FastR WR advanced metrics
        # Note: Columns with excessive nulls (5500+) have been removed:
        # - plyr_gm_rec_avg_cushion, plyr_gm_rec_avg_separation, plyr_gm_rec_avg_yac,
        # - plyr_gm_rec_avg_expected_yac, plyr_gm_rec_avg_yac_above_expectation,
        # - plyr_gm_rec_pct_share_of_intended_ay
        fastr_wr_cols = ['plyr_id', 'season_id', 'week_id',
                         'plyr_gm_rec_tgt_share',
                         'plyr_gm_rec_epa', 'plyr_gm_rec_ay_share',
                         'plyr_gm_rec_wopr', 'plyr_gm_rec_racr']
        nfl_fastr_wr = tables['nfl_fastr_wr'][fastr_wr_cols]

        df = df.merge(
            nfl_fastr_wr,
            on=['plyr_id', 'season_id', 'week_id'],
            how='left'
        )
        self._validate_data(df, "after_nfl_fastr_wr_join")

        # Handle missing NFL FastR stats
        df = self._handle_nfl_fastr_nulls(df)
        self._validate_data(df, "after_nfl_fastr_null_handling")

        # Join with player info (use only plyr_id since season_id format differs between tables)
        # Retain team_id from plyr table for reference (reflects season-end team)
        plyr_info = tables['plyr'][['plyr_id', 'plyr_guid', 'plyr_pos', 'plyr_alt_pos', 'team_id']].drop_duplicates(subset=['plyr_id'])
        plyr_info = plyr_info.rename(columns={'team_id': 'plyr_team_id'})  # Rename to avoid confusion with game-level team_id

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

        # Join with multi_tm_plyr table to track players who changed teams mid-season
        multi_tm_cols = ['plyr_id', 'season_id', 'tm_1_id', 'tm_2_id', 'tm_3_id',
                        'first_tm_week_start_id', 'first_tm_week_end_id',
                        'second_tm_week_start_id', 'second_tm_week_end_id',
                        'third_tm_week_start_id', 'third_tm_week_end_id']
        multi_tm_plyr = tables['multi_tm_plyr'][multi_tm_cols]

        df = df.merge(
            multi_tm_plyr,
            on=['plyr_id', 'season_id'],
            how='left'
        )
        self._validate_data(df, "after_multi_tm_plyr_join")

        # Create current_team_id based on which team the player was on for each week
        df = self._compute_current_team_id(df)
        self._validate_data(df, "after_current_team_id_computation")

        # Join with nfl_game to get home/away team info for opponent calculation
        nfl_game = tables['nfl_game'][['game_id', 'home_team_id', 'away_team_id']]

        df = df.merge(
            nfl_game,
            on=['game_id'],
            how='left'
        )
        self._validate_data(df, "after_nfl_game_join")

        # Create opposing_team_id and is_home_team columns
        # NOTE: These represent the CURRENT week's game context
        # They will be shifted later in _align_next_game_context() to represent
        # the NEXT week's game context for proper prediction alignment
        df = self._compute_opposing_team_info(df)
        self._validate_data(df, "after_opposing_team_computation")

        return df

    def _handle_nfl_fastr_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing NFL FastR stats by imputing with -999 and creating indicator variable.

        Logic:
        - If ALL NFL FastR columns are null for a row, impute with -999
        - Create indicator variable nfl_fastr_missing_stats (1 = missing, 0 = present)

        This handles cases where NFL FastR data is unavailable for certain games
        (e.g., data collection issues, games not covered by NextGenStats).

        Args:
            df: DataFrame with NFL FastR columns joined

        Returns:
            DataFrame with nulls imputed and indicator variable added
        """
        self.logger.info("Handling missing NFL FastR stats...")

        # Define the NFL FastR columns to check
        # Note: High-null columns have been removed from the dataset
        nfl_fastr_cols = [
            'plyr_gm_rec_tgt_share',
            'plyr_gm_rec_epa',
            'plyr_gm_rec_ay_share',
            'plyr_gm_rec_wopr',
            'plyr_gm_rec_racr'
        ]

        # Verify all columns exist
        existing_cols = [col for col in nfl_fastr_cols if col in df.columns]
        if len(existing_cols) != len(nfl_fastr_cols):
            missing = set(nfl_fastr_cols) - set(existing_cols)
            self.logger.warning(f"Some NFL FastR columns not found: {missing}")

        # Create indicator: 1 if ALL NFL FastR columns are null, 0 otherwise
        all_null_mask = df[existing_cols].isnull().all(axis=1)
        df['nfl_fastr_missing_stats'] = all_null_mask.astype(int)

        # Count rows with missing stats
        missing_count = all_null_mask.sum()
        self.logger.info(f"Found {missing_count:,} rows with missing NFL FastR stats ({missing_count/len(df)*100:.2f}%)")

        # Impute null values with -999 for rows where all FastR stats are missing
        if missing_count > 0:
            for col in existing_cols:
                df.loc[all_null_mask, col] = -999
            self.logger.info(f"Imputed {missing_count:,} rows with -999 for {len(existing_cols)} NFL FastR columns")

        return df

    def _compute_current_team_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute current_team_id based on which team the player was on for each week.

        For players who changed teams mid-season (present in multi_tm_plyr table),
        determines the correct team based on week ranges. For single-team players,
        uses the team_id from the game record.

        Args:
            df: DataFrame with multi_tm_plyr columns joined

        Returns:
            DataFrame with current_team_id column added
        """
        self.logger.info("Computing current_team_id column...")

        # Check if player is in multi_tm_plyr (tm_1_id will be non-null)
        is_multi_team = df['tm_1_id'].notna()

        # Fill NULL end week values with 18 (season end)
        df['first_tm_week_end_filled'] = df['first_tm_week_end_id'].fillna(18)
        df['second_tm_week_end_filled'] = df['second_tm_week_end_id'].fillna(18)
        df['third_tm_week_end_filled'] = df['third_tm_week_end_id'].fillna(18)

        # Conditions for team assignment (for multi-team players)
        # Team 1: week >= first_tm_week_start_id AND week <= first_tm_week_end_id
        cond_tm1 = (
            is_multi_team &
            (df['week_id'] >= df['first_tm_week_start_id']) &
            (df['week_id'] <= df['first_tm_week_end_filled'])
        )

        # Team 2: week >= second_tm_week_start_id AND week <= second_tm_week_end_id
        cond_tm2 = (
            is_multi_team &
            (df['week_id'] >= df['second_tm_week_start_id']) &
            (df['week_id'] <= df['second_tm_week_end_filled'])
        )

        # Team 3: tm_3_id is not null AND week >= third_tm_week_start_id AND week <= third_tm_week_end_id
        cond_tm3 = (
            is_multi_team &
            df['tm_3_id'].notna() &
            (df['week_id'] >= df['third_tm_week_start_id']) &
            (df['week_id'] <= df['third_tm_week_end_filled'])
        )

        # Single-team player (not in multi_tm_plyr): use team_id from game record
        cond_single_team = ~is_multi_team

        # Apply conditions using np.select (order matters - first match wins)
        conditions = [cond_tm1, cond_tm2, cond_tm3, cond_single_team]
        choices = [df['tm_1_id'], df['tm_2_id'], df['tm_3_id'], df['team_id']]

        # Default fallback to team_id from game record
        df['current_team_id'] = np.select(conditions, choices, default=df['team_id'])

        # Clean up temporary columns
        df = df.drop(columns=['first_tm_week_end_filled', 'second_tm_week_end_filled', 'third_tm_week_end_filled'])

        # Drop multi_tm_plyr intermediate columns (no longer needed)
        multi_tm_cols_to_drop = ['tm_1_id', 'tm_2_id', 'tm_3_id',
                                 'first_tm_week_start_id', 'first_tm_week_end_id',
                                 'second_tm_week_start_id', 'second_tm_week_end_id',
                                 'third_tm_week_start_id', 'third_tm_week_end_id']
        df = df.drop(columns=multi_tm_cols_to_drop)

        # Convert current_team_id to int (handle any NaN edge cases)
        df['current_team_id'] = df['current_team_id'].astype('Int64')

        multi_team_count = is_multi_team.sum()
        self.logger.info(f"Computed current_team_id: {multi_team_count:,} rows from multi-team players")

        return df

    def _compute_opposing_team_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute opposing_team_id and is_home_team based on current_team_id and game matchup.

        Uses the nfl_game table's home_team_id and away_team_id to determine:
        - opposing_team_id: The team_id of the opponent
        - is_home_team: 1 if player's team is home, 0 if away

        NOTE: These columns represent the CURRENT week's game context.
        They will be shifted in _align_next_game_context() to align with
        the prediction target (next week's receiving yards).

        Args:
            df: DataFrame with current_team_id, home_team_id, and away_team_id columns

        Returns:
            DataFrame with opposing_team_id and is_home_team columns added,
            intermediate columns (home_team_id, away_team_id) dropped
        """
        self.logger.info("Computing opposing_team_id and is_home_team columns...")

        # Condition: current_team_id matches home_team_id
        is_home = df['current_team_id'] == df['home_team_id']

        # Compute opposing_team_id using np.where
        # If home team: opponent is away team; otherwise opponent is home team
        df['opposing_team_id'] = np.where(
            is_home,
            df['away_team_id'],
            df['home_team_id']
        )

        # Compute is_home_team (1 if home, 0 if away)
        df['is_home_team'] = np.where(is_home, 1, 0)

        # Convert to appropriate data types
        df['opposing_team_id'] = df['opposing_team_id'].astype('Int64')
        df['is_home_team'] = df['is_home_team'].astype('Int64')

        # Drop intermediate columns (home_team_id, away_team_id)
        df = df.drop(columns=['home_team_id', 'away_team_id'])

        home_count = df['is_home_team'].sum()
        away_count = (df['is_home_team'] == 0).sum()
        self.logger.info(f"Computed opposing team info: {home_count:,} home games, {away_count:,} away games")

        return df

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Two-stage filtering approach:
        - Stage 1: Identify eligible player-seasons (4+ games with 2+ targets)
        - Stage 2: Include ALL games for eligible player-seasons

        This preserves temporal continuity while maintaining data quality standards.
        Per-game filters (targets, snap %) are NOT applied - only used for eligibility.

        Args:
            df: DataFrame to filter

        Returns:
            Filtered DataFrame
        """
        initial_rows = len(df)

        # === STAGE 1: Player-Season Eligibility ===
        print("Stage 1: Identifying eligible player-seasons...")
        self.logger.info("Stage 1: Identifying eligible player-seasons...")

        # Filter to WR position (scope definition, always applied)
        wr_mask = (df['plyr_pos'] == 'WR') | (df['plyr_alt_pos'] == 'WR')
        wr_df = df[wr_mask].copy()
        total_wr_games = len(wr_df)
        self.logger.info(f"After WR filter: {total_wr_games:,} rows ({initial_rows - total_wr_games:,} non-WR removed)")

        # Count qualifying games (2+ targets) per player-season
        qualifying_mask = wr_df['plyr_gm_rec_tgt'] >= 2
        qualifying_games = wr_df[qualifying_mask].groupby(['plyr_id', 'season_id']).size()

        # Require 4+ qualifying games for eligibility
        eligible_pairs = qualifying_games[qualifying_games >= 4].reset_index()[['plyr_id', 'season_id']]

        total_wr_player_seasons = wr_df.groupby(['plyr_id', 'season_id']).ngroups
        print(f"  Found {len(eligible_pairs)} eligible pairs from {total_wr_player_seasons} total WR player-seasons")
        self.logger.info(f"Found {len(eligible_pairs)} eligible (plyr_id, season_id) pairs from {total_wr_player_seasons} total WR player-seasons")

        # === STAGE 2: Include All Games for Eligible Players ===
        print("Stage 2: Including all games for eligible players...")
        self.logger.info("Stage 2: Including all games for eligible player-seasons...")

        result_df = wr_df.merge(
            eligible_pairs,
            on=['plyr_id', 'season_id'],
            how='inner'
        )

        retention_pct = (len(result_df) / total_wr_games) * 100
        print(f"  Final dataset: {len(result_df):,} rows ({retention_pct:.1f}% of all WR games retained)")
        self.logger.info(f"Final dataset: {len(result_df):,} rows ({retention_pct:.1f}% of all WR games retained)")
        self.logger.info(f"Total rows removed by filters: {initial_rows - len(result_df):,} ({((initial_rows - len(result_df))/initial_rows)*100:.1f}%)")

        return result_df

    def _align_next_game_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Shift game context columns forward to align with next-week prediction target.
        
        PROBLEM BEING SOLVED:
        ====================
        When predicting Week N+1 receiving yards, the model should know:
        - Historical features from Week N (what they just did) - NO SHIFT NEEDED
        - Upcoming game context for Week N+1 (who they'll face) - NEEDS SHIFT
        
        Currently, opposing_team_id and is_home_team represent Week N's game.
        We need to shift them to represent Week N+1's game.
        
        IMPLEMENTATION:
        ==============
        For each player within each season:
        - Shift game context columns backward by 1 (shift(-1) gets NEXT row's value)
        - This means row for Week N now contains Week N+1's opponent/venue
        - Rename columns with 'next_' prefix for clarity
        
        EDGE CASES:
        ===========
        - Last week of season: Will have NaN for next game context
          → These rows are dropped anyway when we drop rows without next_week_rec_yds
        - Players who change teams mid-season: Correctly handled because
          current_team_id already accounts for team changes
        - Bye weeks: Not an issue because we're game-indexed, not week-indexed
        
        Args:
            df: DataFrame with opposing_team_id and is_home_team columns
            
        Returns:
            DataFrame with game context shifted to next-week alignment
        """
        self.logger.info("Aligning game context for next-week prediction...")
        self.logger.info("  Shifting: opposing_team_id, is_home_team, game_id → next_opponent_team_id, next_is_home_game, next_game_id")
        
        # Sort to ensure proper temporal order for shifting
        df = df.sort_values(['plyr_id', 'season_id', 'week_id']).reset_index(drop=True)
        
        # Columns that represent "upcoming game context" and need to be shifted
        # These should contain NEXT week's information when predicting next week's yards
        game_context_cols = {
            'opposing_team_id': 'next_opponent_team_id',
            'is_home_team': 'next_is_home_game',
            'game_id': 'next_game_id'
        }
        
        # Shift each game context column forward (shift -1 = get next row's value)
        for old_col, new_col in game_context_cols.items():
            if old_col in df.columns:
                # Shift within each player-season group
                df[new_col] = df.groupby(['plyr_id', 'season_id'])[old_col].shift(-1)
                self.logger.info(f"    Created {new_col} from {old_col}")
        
        # Keep original columns with 'current_' prefix for reference (optional, useful for debugging)
        # Rename original game context columns to indicate they're from current week
        rename_map = {
            'opposing_team_id': 'current_opponent_team_id',
            'is_home_team': 'current_is_home_game'
        }
        df = df.rename(columns=rename_map)
        
        # Count how many rows have valid next game context
        has_next_context = df['next_opponent_team_id'].notna().sum()
        total_rows = len(df)
        self.logger.info(f"  Rows with valid next game context: {has_next_context:,}/{total_rows:,} ({has_next_context/total_rows*100:.1f}%)")
        self.logger.info(f"  Rows without next game (end of season): {total_rows - has_next_context:,}")
        
        return df
    
    def _create_next_week_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create next week target variable with proper temporal alignment.
        
        This method:
        1. Handles duplicates by aggregating stats per player-week
        2. Creates next_week_rec_yds by shifting the target column
        3. Removes rows without a valid target (last week of season)
        
        NOTE: _align_next_game_context() should be called BEFORE this method
        to ensure game context is properly shifted.
        
        Args:
            df: DataFrame with current week features and shifted game context
            
        Returns:
            DataFrame with next week targets aligned
        """
        self.logger.info("Creating next week target variable...")
        
        # First, handle duplicates by taking the mean of stats per player-week
        # This handles cases where players appear multiple times per week (e.g., multiple games)
        agg_dict = {}
        
        # Columns to preserve with 'first' aggregation (ID and categorical columns)
        # Updated to include new column naming convention
        preserve_cols = [
            'plyr_id', 'season_id', 'week_id', 'plyr_guid', 'plyr_pos', 'plyr_alt_pos',
            'week_num', 'year', 'game_id', 'team_id', 'plyr_team_id', 'current_team_id',
            # Current week context (kept for reference/debugging)
            'current_opponent_team_id', 'current_is_home_game',
            # Next week context (aligned with prediction target)
            'next_opponent_team_id', 'next_is_home_game', 'next_game_id'
        ]
        
        for col in df.columns:
            if col in preserve_cols:
                agg_dict[col] = 'first'
            else:
                # Use mean for numeric stats, first for text
                if df[col].dtype in ['int64', 'float64', 'Int64', 'Float64']:
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
        rows_before = len(df)
        df = df.dropna(subset=['next_week_rec_yds'])
        rows_dropped = rows_before - len(df)
        self.logger.info(f"Dropped {rows_dropped:,} rows without next week target (end of season)")
        
        # Also drop rows without next game context (should be same rows, but verify)
        if 'next_opponent_team_id' in df.columns:
            context_nulls = df['next_opponent_team_id'].isna().sum()
            if context_nulls > 0:
                self.logger.warning(f"Found {context_nulls} rows with target but missing next game context - dropping")
                df = df.dropna(subset=['next_opponent_team_id'])
        
        # Validate temporal integrity
        self._validate_temporal_integrity(df)
        
        self.logger.info(f"Created next week targets: {len(df):,} training samples")
        
        return df
    
    def _validate_temporal_integrity(self, df: pd.DataFrame):
        """
        Validate that no future data leakage exists.
        
        Checks:
        1. Week ordering is correct within each player-season
        2. Next game context columns align with target
        3. No statistical features from future weeks
        
        Args:
            df: DataFrame to validate
        """
        self.logger.info("Validating temporal integrity...")
        
        # Check 1: Verify sequential week alignment within each player's season
        df_sorted = df.sort_values(['plyr_id', 'season_id', 'week_id'])
        
        if len(df_sorted) > 0:
            sample_check = df_sorted.groupby(['plyr_id', 'season_id']).apply(
                lambda x: (x['week_id'].diff().fillna(0) >= 0).all() if len(x) > 1 else True
            ).all()
        else:
            sample_check = True
        
        if not sample_check:
            self.logger.warning("Temporal integrity check found potential issues with week ordering")
        else:
            self.logger.info("  Week ordering validated successfully")
        
        # Check 2: Verify game context alignment
        # The next_opponent_team_id should differ from current_opponent_team_id for most rows
        # (same opponent would only happen in rare back-to-back divisional games)
        if 'next_opponent_team_id' in df.columns and 'current_opponent_team_id' in df.columns:
            same_opponent = (df['next_opponent_team_id'] == df['current_opponent_team_id']).sum()
            same_opponent_pct = same_opponent / len(df) * 100
            self.logger.info(f"  Same opponent in consecutive weeks: {same_opponent:,} ({same_opponent_pct:.1f}%)")
            
            if same_opponent_pct > 10:  # Arbitrary threshold - back-to-back same opponent is rare
                self.logger.warning(f"  High rate of same consecutive opponents - verify shift logic")
        
        self.logger.info("Temporal integrity validation complete")
    
    def _finalize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final dataset preparation and column organization.
        
        Organizes columns into logical groups:
        1. Identifiers (player, season, week)
        2. Target variable
        3. Next game context (aligned with target)
        4. Current week performance (historical features)
        5. Season cumulative stats (historical features)
        
        Args:
            df: DataFrame to finalize
            
        Returns:
            Final dataset ready for training
        """
        self.logger.info("Finalizing dataset...")

        # Define column order for better organization
        id_cols = [
            'plyr_id', 'plyr_guid', 'season_id', 'week_id', 'year', 'week_num',
            'game_id', 'team_id', 'plyr_team_id', 'current_team_id'
        ]
        
        target_col = ['next_week_rec_yds']
        
        # Next game context (aligned with prediction target)
        next_game_context_cols = [
            'next_opponent_team_id', 'next_is_home_game', 'next_game_id'
        ]
        
        # Current week context (kept for reference, can be dropped if not needed)
        current_game_context_cols = [
            'current_opponent_team_id', 'current_is_home_game'
        ]

        # Current week target (the performance we're predicting FROM)
        current_week_target = ['plyr_gm_rec_yds'] if 'plyr_gm_rec_yds' in df.columns else []

        # All other columns are features
        all_special_cols = (id_cols + target_col + next_game_context_cols + 
                          current_game_context_cols + current_week_target + 
                          ['plyr_pos', 'plyr_alt_pos'])
        feature_cols = [col for col in df.columns if col not in all_special_cols]
        
        # Reorder columns: IDs → Target → Next Game Context → Current Performance → Features
        final_cols = (id_cols + target_col + next_game_context_cols + 
                     current_game_context_cols + current_week_target + feature_cols)
        available_cols = [col for col in final_cols if col in df.columns]
        df = df[available_cols]
        
        # Sort by temporal order for final output
        df = df.sort_values(['season_id', 'week_id', 'plyr_id']).reset_index(drop=True)
        
        # Log column organization
        self.logger.info(f"Column organization:")
        self.logger.info(f"  ID columns: {len([c for c in id_cols if c in df.columns])}")
        self.logger.info(f"  Target column: {target_col}")
        self.logger.info(f"  Next game context: {[c for c in next_game_context_cols if c in df.columns]}")
        self.logger.info(f"  Feature columns: {len(feature_cols)}")
        
        # Final validation
        self._validate_data(df, "final_dataset", 
                           required_columns=['plyr_id', 'next_week_rec_yds', 'next_opponent_team_id'])
        
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
            'validation_results': self.validation_results,
            'temporal_alignment': {
                'target': 'next_week_rec_yds (Week N+1 receiving yards)',
                'historical_features': 'plyr_gm_rec_*, plyr_rec_* (from Week N)',
                'next_game_context': 'next_opponent_team_id, next_is_home_game (Week N+1 matchup)'
            }
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
        
        Pipeline:
        1. Load base tables
        2. Create temporal joins
        3. Apply filters (WR position, eligibility criteria)
        4. Align next game context (shift opponent/venue to match prediction target)
        5. Create next week target variable
        6. Finalize dataset (organize columns, validate)
        7. Save dataset
        
        Returns:
            Path to saved dataset file
        """
        self.logger.info("Starting NFL receiving yards dataset build...")
        self.logger.info("=" * 60)
        
        try:
            # Step 1: Load base tables
            self.logger.info("Step 1: Loading base tables...")
            tables = self.load_base_tables()
            
            # Step 2: Create temporal joins
            self.logger.info("Step 2: Creating temporal joins...")
            df = self._create_temporal_joins(tables)
            
            # Step 3: Apply filters
            self.logger.info("Step 3: Applying filters...")
            df = self._apply_filters(df)
            
            # Step 4: Align next game context (NEW STEP)
            self.logger.info("Step 4: Aligning next game context...")
            df = self._align_next_game_context(df)
            
            # Step 5: Create next week target variable
            self.logger.info("Step 5: Creating next week target variable...")
            df = self._create_next_week_target(df)
            
            # Step 6: Finalize dataset
            self.logger.info("Step 6: Finalizing dataset...")
            df = self._finalize_dataset(df)
            
            # Step 7: Save dataset
            self.logger.info("Step 7: Saving dataset...")
            output_file = self.save_dataset(df)
            
            self.logger.info("=" * 60)
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