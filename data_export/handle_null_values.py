import pandas as pd
import pyarrow.parquet as pq
import os
import logging
from pathlib import Path
from typing import Dict, List, Set

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NullValueHandler:
    def __init__(self, raw_dir: str, output_dir: str = None):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir) if output_dir else None
        self.null_columns_summary = {}

    def read_partitioned_table(self, table_path: str) -> pd.DataFrame:
        """Read a partitioned parquet table into a DataFrame"""
        try:
            df = pd.read_parquet(table_path)
            logger.info(f"Successfully read table from {table_path}")
            logger.info(f"Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error reading table from {table_path}: {e}")
            raise

    def get_null_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of columns with at least one NULL value"""
        null_cols = df.columns[df.isnull().any()].tolist()
        return null_cols

    def handle_plyr_def_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values for plyr_def table"""
        logger.info("Starting plyr_def null value handling...")

        # Create a copy to avoid modifying original
        df = df.copy()

        # Initialize indicator columns
        df['plyr_def_no_targets'] = 0
        df['plyr_def_missing_stats'] = 0
        df['plyr_def_no_mtkl'] = 0
        df['plyr_def_no_completions'] = 0

        # Handle plyr_def_no_targets indicator
        no_target_cols = [
            'plyr_def_cmp_pct', 'plyr_def_pass_yds_cmp',
            'plyr_def_pass_yds_tgt', 'plyr_def_pass_rtg', 'plyr_def_adot'
        ]

        # Check if plyr_def_tgt = 0
        mask_no_targets = (df['plyr_def_tgt'] == 0)
        for col in no_target_cols:
            if col in df.columns:
                df.loc[mask_no_targets & df[col].isnull(), col] = -1
        df.loc[mask_no_targets, 'plyr_def_no_targets'] = 1

        logger.info(f"Applied plyr_def_no_targets indicator to {mask_no_targets.sum()} rows")

        # Handle plyr_def_missing_stats indicator (first set - based on plyr_def_tgt NULL)
        missing_stats_cols_1 = [
            'plyr_def_tgt', 'plyr_def_cmp', 'plyr_def_cmp_pct', 'plyr_def_pass_yds',
            'plyr_def_pass_yds_cmp', 'plyr_def_pass_yds_tgt', 'plyr_def_pass_td',
            'plyr_def_pass_rtg', 'plyr_def_adot', 'plyr_def_ay', 'plyr_def_yac',
            'plyr_def_bltz', 'plyr_def_hrry', 'plyr_def_qbkd', 'plyr_def_sk',
            'plyr_def_prss', 'plyr_def_comb_tkl', 'plyr_def_mtkl', 'plyr_def_mtkl_pct'
        ]

        mask_missing_tgt = df['plyr_def_tgt'].isnull()
        for col in missing_stats_cols_1:
            if col in df.columns:
                df.loc[mask_missing_tgt & df[col].isnull(), col] = -1
        df.loc[mask_missing_tgt, 'plyr_def_missing_stats'] = 1

        logger.info(f"Applied plyr_def_missing_stats indicator (tgt NULL) to {mask_missing_tgt.sum()} rows")

        # Handle plyr_def_missing_stats indicator (second set - based on plyr_def_int NULL)
        missing_stats_cols_2 = [
            'plyr_def_int', 'plyr_def_int_yds', 'plyr_def_int_td', 'plyr_def_int_lng',
            'plyr_pass_def', 'plyr_def_force_fmbl', 'plyr_def_fmbl_rec',
            'plyr_def_fmbl_rec_yds', 'plyr_def_fmbl_rec_td', 'plyr_def_solo_tkl',
            'plyr_def_tkl_assist', 'plyr_def_tfl', 'plyr_def_qb_hit', 'plyr_def_sfty'
        ]

        mask_missing_int = df['plyr_def_int'].isnull()
        for col in missing_stats_cols_2:
            if col in df.columns:
                df.loc[mask_missing_int & df[col].isnull(), col] = -1
        df.loc[mask_missing_int, 'plyr_def_missing_stats'] = 1

        logger.info(f"Applied plyr_def_missing_stats indicator (int NULL) to {mask_missing_int.sum()} rows")

        # Handle plyr_def_no_mtkl indicator
        no_mtkl_cols = ['plyr_def_mtkl_pct']

        mask_no_mtkl = (df['plyr_def_mtkl'] == 0) if 'plyr_def_mtkl' in df.columns else pd.Series([False] * len(df))
        for col in no_mtkl_cols:
            if col in df.columns:
                df.loc[mask_no_mtkl & df[col].isnull(), col] = -1
        df.loc[mask_no_mtkl, 'plyr_def_no_mtkl'] = 1

        logger.info(f"Applied plyr_def_no_mtkl indicator to {mask_no_mtkl.sum()} rows")

        # Handle plyr_def_no_completions indicator
        mask_no_completions = (df['plyr_def_cmp'] == 0) & df['plyr_def_pass_yds_cmp'].isnull()
        df.loc[mask_no_completions, 'plyr_def_pass_yds_cmp'] = -1
        df.loc[mask_no_completions, 'plyr_def_no_completions'] = 1

        logger.info(f"Applied plyr_def_no_completions indicator to {mask_no_completions.sum()} rows")

        return df

    def process_table(self, table_name: str, table_path: str) -> None:
        """Process a single table for null value handling"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing table: {table_name}")
        logger.info(f"Path: {table_path}")
        logger.info(f"{'='*60}")

        # Read the table
        df = self.read_partitioned_table(table_path)

        # Get columns with nulls before processing
        null_cols_before = self.get_null_columns(df)
        logger.info(f"Columns with NULL values BEFORE imputation: {len(null_cols_before)}")
        if null_cols_before:
            logger.info(f"  {null_cols_before}")

        # Apply table-specific null handling
        if table_name == 'plyr_def':
            df = self.handle_plyr_def_nulls(df)
        else:
            logger.warning(f"No null handling rules defined for table: {table_name}")
            return

        # Get columns with nulls after processing
        null_cols_after = self.get_null_columns(df)
        logger.info(f"\nColumns with NULL values AFTER imputation: {len(null_cols_after)}")
        if null_cols_after:
            logger.info(f"  {null_cols_after}")

        # Store summary
        self.null_columns_summary[table_name] = {
            'before': null_cols_before,
            'after': null_cols_after,
            'shape': df.shape
        }

        # Log detailed summary
        logger.info(f"\nSummary for {table_name}:")
        logger.info(f"  Original NULL columns: {len(null_cols_before)}")
        logger.info(f"  Remaining NULL columns: {len(null_cols_after)}")
        logger.info(f"  DataFrame shape: {df.shape}")

        if null_cols_after:
            logger.info(f"\nColumns still containing NULL values:")
            for col in null_cols_after:
                null_count = df[col].isnull().sum()
                logger.info(f"    {col}: {null_count} NULLs")

            # Export rows with remaining nulls to CSV
            rows_with_nulls = df[df[null_cols_after].isnull().any(axis=1)]
            if len(rows_with_nulls) > 0:
                csv_dir = Path(r"C:\Users\nocap\Desktop\code\NFL_ml\data_export\null_values")
                csv_dir.mkdir(parents=True, exist_ok=True)
                csv_filename = f"{table_name}_rows_with_nulls.csv"
                csv_path = csv_dir / csv_filename
                rows_with_nulls.to_csv(csv_path, index=False)
                logger.info(f"\nExported {len(rows_with_nulls)} rows with NULL values to {csv_filename}")
                logger.info(f"  Full path: {csv_path.absolute()}")

    def print_final_summary(self) -> None:
        """Print final summary of all processed tables"""
        logger.info(f"\n{'='*60}")
        logger.info("FINAL SUMMARY - All Tables")
        logger.info(f"{'='*60}")

        for table_name, summary in self.null_columns_summary.items():
            logger.info(f"\nTable: {table_name}")
            logger.info(f"  Shape: {summary['shape']}")
            logger.info(f"  NULL columns before: {len(summary['before'])}")
            logger.info(f"  NULL columns after: {len(summary['after'])}")
            if summary['after']:
                logger.info(f"  Remaining NULL columns: {summary['after']}")


def main():
    # Initialize handler
    raw_dir = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw"
    handler = NullValueHandler(raw_dir=raw_dir)

    # Process plyr_def table
    plyr_def_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw\plyr_szn\plyr_def"
    handler.process_table('plyr_def', plyr_def_path)

    # Print final summary
    handler.print_final_summary()


if __name__ == "__main__":
    main()