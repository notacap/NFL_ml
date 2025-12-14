"""
Enrich Dataset with Team Defense Stats

This script loads the base dataset (built via build_dataset.py) and enriches it
with opponent team defense statistics from tm_def_plyr_agg.

The team defense stats are joined on:
- next_opponent_team_id (from base) = team_id (from tm_def_plyr_agg)
- week_id (matching weeks)
- season_id (matching seasons)

This provides defensive context for the team the player will face in week N+1,
using the opponent's cumulative defensive stats through week N.

Author: Claude Code
Created: 2024-12-11
"""

import os
import sys
import logging
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional
import glob

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data_loaders.loaders import TeamDefenseLoader


class TeamDefenseEnricher:
    """
    Enriches the base WR receiving yards dataset with opponent team defense stats.
    """

    def __init__(self, config_dir: str = None):
        """
        Initialize the enricher.

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
        self.processed_path = project_root / self.paths_config["data"]["processed"]
        self.processed_path.mkdir(parents=True, exist_ok=True)

        # Set up subdirectory paths
        self.parquet_path = self.processed_path / "parquet"
        self.yaml_path = self.processed_path / "yaml"
        self.logs_path = self.processed_path / "logs"
        self.parquet_path.mkdir(parents=True, exist_ok=True)
        self.yaml_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self._setup_logging()

    def _load_config(self, filename: str) -> dict:
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
                logging.FileHandler(
                    self.logs_path / f'enrich_team_defense_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                ),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Team Defense Enricher initialized")

    def find_latest_base_dataset(self) -> Path:
        """
        Find the most recent base dataset file.

        Returns:
            Path to the latest base dataset parquet file
        """
        pattern = str(self.parquet_path / "nfl_wr_receiving_yards_dataset_*.parquet")
        files = glob.glob(pattern)

        if not files:
            raise FileNotFoundError(
                f"No base dataset found matching pattern: {pattern}"
            )

        # Sort by modification time (most recent first)
        latest_file = max(files, key=os.path.getmtime)
        self.logger.info(f"Found latest base dataset: {latest_file}")

        return Path(latest_file)

    def load_base_dataset(self, base_dataset_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load the base dataset.

        Args:
            base_dataset_path: Optional explicit path to base dataset.
                              If not provided, finds the latest one.

        Returns:
            DataFrame with base dataset
        """
        if base_dataset_path is None:
            base_dataset_path = self.find_latest_base_dataset()

        self.logger.info(f"Loading base dataset from: {base_dataset_path}")
        df = pd.read_parquet(base_dataset_path)
        self.logger.info(f"Loaded base dataset: {len(df):,} rows, {len(df.columns)} columns")

        return df

    def load_team_defense_stats(self) -> pd.DataFrame:
        """
        Load team defense stats using the TeamDefenseLoader.

        Returns:
            DataFrame with team defense statistics
        """
        seasons = self.data_config['temporal']['historical_seasons'] + \
                  [self.data_config['temporal']['current_season']]

        loader = TeamDefenseLoader(
            base_data_path=self.base_data_path,
            seasons=seasons,
            logger=self.logger
        )

        df = loader.load()
        self.logger.info(f"Loaded team defense stats: {len(df):,} rows")

        return df

    def enrich_dataset(self, base_df: pd.DataFrame, team_def_df: pd.DataFrame) -> pd.DataFrame:
        """
        Join team defense stats to the base dataset.

        The join connects:
        - base_df.next_opponent_team_id -> team_def_df.team_id
        - base_df.week_id -> team_def_df.week_id
        - base_df.season_id -> team_def_df.season_id

        This gives us the opponent's defensive stats through the current week,
        which is the information available when predicting next week's performance.

        Args:
            base_df: Base dataset from build_dataset.py
            team_def_df: Team defense statistics

        Returns:
            Enriched DataFrame with team defense columns added
        """
        self.logger.info("Joining team defense stats to base dataset...")

        initial_rows = len(base_df)
        initial_cols = len(base_df.columns)

        # Verify required columns exist in base dataset
        required_base_cols = ['next_opponent_team_id', 'week_id', 'season_id']
        missing_cols = set(required_base_cols) - set(base_df.columns)
        if missing_cols:
            raise ValueError(f"Base dataset missing required columns: {missing_cols}")

        # Verify required columns exist in team defense data
        required_def_cols = ['team_id', 'week_id', 'season_id']
        missing_def_cols = set(required_def_cols) - set(team_def_df.columns)
        if missing_def_cols:
            raise ValueError(f"Team defense data missing required columns: {missing_def_cols}")

        # Perform the join
        # Join on: next_opponent_team_id = team_id, week_id, season_id
        enriched_df = base_df.merge(
            team_def_df,
            left_on=['next_opponent_team_id', 'week_id', 'season_id'],
            right_on=['team_id', 'week_id', 'season_id'],
            how='left'
        )

        # Drop the team_id column from the joined data (redundant with next_opponent_team_id)
        if 'team_id_y' in enriched_df.columns:
            enriched_df = enriched_df.drop(columns=['team_id_y'])
            # Rename team_id_x back to team_id if it exists
            if 'team_id_x' in enriched_df.columns:
                enriched_df = enriched_df.rename(columns={'team_id_x': 'team_id'})
        elif 'team_id' in enriched_df.columns and 'team_id' in team_def_df.columns:
            # If team_id came from the team_def_df join, drop it
            # Check if this is the defense team_id by seeing if it matches next_opponent_team_id
            # For safety, we'll drop it if the merge added it
            pass  # team_id from base dataset should be preserved

        # Handle the case where team_id was added from the right table
        # We want to keep the original team_id from base_df but drop the one from team_def_df
        cols_to_check = enriched_df.columns.tolist()
        if 'team_id' in team_def_df.columns:
            # The merge may have created team_id_x (from base) and team_id_y (from defense)
            # or just added team_id if base didn't have it
            if 'team_id_x' in cols_to_check and 'team_id_y' in cols_to_check:
                enriched_df = enriched_df.drop(columns=['team_id_y'])
                enriched_df = enriched_df.rename(columns={'team_id_x': 'team_id'})
            elif 'team_id' in cols_to_check and 'team_id' not in base_df.columns:
                # team_id was added from defense table, drop it
                enriched_df = enriched_df.drop(columns=['team_id'])

        # Log join statistics
        final_rows = len(enriched_df)
        final_cols = len(enriched_df.columns)
        new_cols = final_cols - initial_cols

        # Check for null values in new columns (indicates failed joins)
        team_def_feature_cols = [col for col in team_def_df.columns
                                 if col not in ['team_id', 'week_id', 'season_id']]
        null_counts = enriched_df[team_def_feature_cols].isnull().sum()
        total_nulls = null_counts.sum()

        self.logger.info(f"Join complete: {final_rows:,} rows, {new_cols} new columns added")

        if total_nulls > 0:
            null_pct = (null_counts.max() / len(enriched_df)) * 100
            self.logger.warning(
                f"Found {total_nulls:,} null values in team defense columns "
                f"(max {null_pct:.1f}% in a single column)"
            )
            # Log rows with missing team defense data
            missing_mask = enriched_df[team_def_feature_cols[0]].isnull()
            missing_count = missing_mask.sum()
            self.logger.warning(f"{missing_count:,} rows missing team defense data")

        return enriched_df

    def save_enriched_dataset(self, df: pd.DataFrame) -> str:
        """
        Save the enriched dataset to parquet file.

        Args:
            df: Enriched DataFrame to save

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nfl_wr_receiving_yards_with_team_def_{timestamp}.parquet"
        output_file = self.parquet_path / filename

        self.logger.info(f"Saving enriched dataset to: {output_file}")

        # Save with compression
        df.to_parquet(output_file, compression='snappy', index=False)

        # Save metadata
        metadata = {
            'created_at': timestamp,
            'base_dataset': 'nfl_wr_receiving_yards_dataset',
            'enrichment': 'team_defense_stats',
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'team_defense_columns': [
                'tm_def_int', 'tm_def_pass_def', 'tm_def_comb_tkl', 'tm_def_solo_tkl',
                'tm_def_qb_hit', 'tm_def_tfl', 'tm_def_tgt', 'tm_def_cmp',
                'tm_def_pass_yds', 'tm_def_ay', 'tm_def_yac', 'tm_def_bltz',
                'tm_def_hrry', 'tm_def_qbkd', 'tm_def_sk', 'tm_def_prss',
                'tm_def_mtkl', 'tm_def_cmp_pct', 'tm_def_pass_yds_cmp',
                'tm_def_pass_yds_tgt', 'tm_def_adot', 'tm_def_yac_cmp',
                'tm_def_mtkl_pct', 'tm_def_pass_rtg', 'tm_def_sk_pct',
                'tm_def_int_pct', 'tm_def_tkl_pg', 'tm_def_sk_pg',
                'tm_def_prss_pg', 'tm_def_to', 'tm_def_to_pg'
            ]
        }

        metadata_file = self.yaml_path / f"enriched_metadata_{timestamp}.yaml"
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)

        self.logger.info(f"Enriched dataset saved: {len(df):,} rows, {len(df.columns)} columns")
        self.logger.info(f"Metadata saved to: {metadata_file}")

        return str(output_file)

    def run(self, base_dataset_path: Optional[Path] = None) -> str:
        """
        Run the full enrichment pipeline.

        Args:
            base_dataset_path: Optional explicit path to base dataset

        Returns:
            Path to the saved enriched dataset
        """
        self.logger.info("Starting team defense enrichment pipeline...")

        try:
            # Step 1: Load base dataset
            base_df = self.load_base_dataset(base_dataset_path)

            # Step 2: Load team defense stats
            team_def_df = self.load_team_defense_stats()

            # Step 3: Enrich dataset
            enriched_df = self.enrich_dataset(base_df, team_def_df)

            # Step 4: Save enriched dataset
            output_file = self.save_enriched_dataset(enriched_df)

            self.logger.info("Team defense enrichment completed successfully!")
            return output_file

        except Exception as e:
            self.logger.error(f"Enrichment failed: {e}")
            raise


def main():
    """Main entry point for team defense enrichment."""
    try:
        # Initialize enricher
        enricher = TeamDefenseEnricher()

        # Run enrichment
        output_file = enricher.run()

        print(f"\nEnrichment completed successfully!")
        print(f"Output file: {output_file}")
        print(f"Logs saved to: {enricher.processed_path}")

    except Exception as e:
        print(f"Enrichment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
