import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path to import null_utils
sys.path.append(str(Path(__file__).parent.parent))
from null_utils import BaseNullHandler, logger

class PlyrRzRecNullHandler(BaseNullHandler):
    def __init__(self, raw_dir: str, output_dir: str = None):
        super().__init__(raw_dir, output_dir)

    def handle_plyr_rz_rec_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values for plyr_rz_rec table"""
        logger.info("Starting plyr_rz_rec null value handling...")

        # Create a copy to avoid modifying original
        df = df.copy()

        # Initialize indicator columns
        df['plyr_rz_rec_missing_stats'] = 0
        df['plyr_rz_rec_tz_no_targets'] = 0

        # Handle plyr_rz_rec_missing_stats indicator for plyr_rec_rz_tgt_pct
        # When plyr_rec_rz_tgt != 0 and plyr_rec_rz_tgt_pct is NULL
        if all(col in df.columns for col in ['plyr_rec_rz_tgt', 'plyr_rec_rz_tgt_pct']):
            mask_has_rz_targets = df['plyr_rec_rz_tgt'] != 0
            mask_null = mask_has_rz_targets & df['plyr_rec_rz_tgt_pct'].isnull()
            df.loc[mask_null, 'plyr_rec_rz_tgt_pct'] = -1
            if mask_null.sum() > 0:
                df.loc[mask_null, 'plyr_rz_rec_missing_stats'] = 1
            logger.info(f"Applied plyr_rz_rec_missing_stats indicator for NULL plyr_rec_rz_tgt_pct to {mask_null.sum()} rows")

        # Handle plyr_rz_rec_missing_stats indicator for plyr_rec_tz_tgt_pct
        # When plyr_rec_tz_tgt != 0 and plyr_rec_tz_tgt_pct is NULL
        if all(col in df.columns for col in ['plyr_rec_tz_tgt', 'plyr_rec_tz_tgt_pct']):
            mask_has_tz_targets = df['plyr_rec_tz_tgt'] != 0
            mask_null = mask_has_tz_targets & df['plyr_rec_tz_tgt_pct'].isnull()
            df.loc[mask_null, 'plyr_rec_tz_tgt_pct'] = -1
            if mask_null.sum() > 0:
                df.loc[mask_null, 'plyr_rz_rec_missing_stats'] = 1
            logger.info(f"Applied plyr_rz_rec_missing_stats indicator for NULL plyr_rec_tz_tgt_pct to {mask_null.sum()} rows")

        # Handle plyr_rz_rec_tz_no_targets indicator for plyr_rec_tz_catch_pct and plyr_rec_tz_tgt_pct
        # When plyr_rec_tz_tgt = 0 and either column is NULL
        if all(col in df.columns for col in ['plyr_rec_tz_tgt', 'plyr_rec_tz_catch_pct', 'plyr_rec_tz_tgt_pct']):
            mask_no_tz_targets = df['plyr_rec_tz_tgt'] == 0

            # Handle plyr_rec_tz_catch_pct
            mask_null_catch = mask_no_tz_targets & df['plyr_rec_tz_catch_pct'].isnull()
            df.loc[mask_null_catch, 'plyr_rec_tz_catch_pct'] = -1
            if mask_null_catch.sum() > 0:
                df.loc[mask_null_catch, 'plyr_rz_rec_tz_no_targets'] = 1
            logger.info(f"Applied plyr_rz_rec_tz_no_targets indicator for NULL plyr_rec_tz_catch_pct to {mask_null_catch.sum()} rows")

            # Handle plyr_rec_tz_tgt_pct
            mask_null_tgt = mask_no_tz_targets & df['plyr_rec_tz_tgt_pct'].isnull()
            df.loc[mask_null_tgt, 'plyr_rec_tz_tgt_pct'] = -1
            if mask_null_tgt.sum() > 0:
                df.loc[mask_null_tgt, 'plyr_rz_rec_tz_no_targets'] = 1
            logger.info(f"Applied plyr_rz_rec_tz_no_targets indicator for NULL plyr_rec_tz_tgt_pct to {mask_null_tgt.sum()} rows")

        return df

    def process_plyr_rz_rec_table(self, table_path: str) -> None:
        """Process the plyr_rz_rec table for null value handling"""
        self.process_table('plyr_rz_rec', table_path, self.handle_plyr_rz_rec_nulls)

def main():
    # Initialize handler
    raw_dir = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw"
    handler = PlyrRzRecNullHandler(raw_dir=raw_dir)

    # Process plyr_rz_rec table
    plyr_rz_rec_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw\plyr_szn\plyr_rz_rec"
    handler.process_plyr_rz_rec_table(plyr_rz_rec_path)

    # Print final summary
    handler.print_final_summary()

if __name__ == "__main__":
    main()