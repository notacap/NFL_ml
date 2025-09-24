import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path to import null_utils
sys.path.append(str(Path(__file__).parent.parent))
from null_utils import BaseNullHandler, logger

class PlyrRzRushNullHandler(BaseNullHandler):
    def __init__(self, raw_dir: str, output_dir: str = None):
        super().__init__(raw_dir, output_dir)

    def handle_plyr_rz_rush_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values for plyr_rz_rush table"""
        logger.info("Starting plyr_rz_rush null value handling...")

        # Create a copy to avoid modifying original
        df = df.copy()

        # Initialize indicator columns
        df['plyr_rz_rush_missing_stats'] = 0
        df['plyr_rz_rush_fz_no_att'] = 0
        df['plyr_rz_rush_tz_no_att'] = 0
        df['plyr_rz_rush_no_att'] = 0

        # Handle plyr_rush_fz_usage NULL values
        if all(col in df.columns for col in ['plyr_rush_fz_att', 'plyr_rush_fz_usage']):
            # Case 1: plyr_rush_fz_att != 0 and plyr_rush_fz_usage is NULL
            mask_fz_has_att = df['plyr_rush_fz_att'] != 0
            mask_fz_null = mask_fz_has_att & df['plyr_rush_fz_usage'].isnull()
            df.loc[mask_fz_null, 'plyr_rush_fz_usage'] = -1
            if mask_fz_null.sum() > 0:
                df.loc[mask_fz_null, 'plyr_rz_rush_missing_stats'] = 1
            logger.info(f"Applied plyr_rz_rush_missing_stats indicator for NULL plyr_rush_fz_usage to {mask_fz_null.sum()} rows with fz attempts")

            # Case 2: plyr_rush_fz_att = 0 and plyr_rush_fz_usage is NULL
            mask_fz_no_att = df['plyr_rush_fz_att'] == 0
            mask_fz_no_att_null = mask_fz_no_att & df['plyr_rush_fz_usage'].isnull()
            df.loc[mask_fz_no_att_null, 'plyr_rush_fz_usage'] = -1
            if mask_fz_no_att_null.sum() > 0:
                df.loc[mask_fz_no_att_null, 'plyr_rz_rush_fz_no_att'] = 1
            logger.info(f"Applied plyr_rz_rush_fz_no_att indicator for NULL plyr_rush_fz_usage to {mask_fz_no_att_null.sum()} rows with no fz attempts")

        # Handle plyr_rush_tz_usage NULL values
        if all(col in df.columns for col in ['plyr_rush_tz_att', 'plyr_rush_tz_usage']):
            # Case 1: plyr_rush_tz_att != 0 and plyr_rush_tz_usage is NULL
            mask_tz_has_att = df['plyr_rush_tz_att'] != 0
            mask_tz_null = mask_tz_has_att & df['plyr_rush_tz_usage'].isnull()
            df.loc[mask_tz_null, 'plyr_rush_tz_usage'] = -1
            if mask_tz_null.sum() > 0:
                df.loc[mask_tz_null, 'plyr_rz_rush_missing_stats'] = 1
            logger.info(f"Applied plyr_rz_rush_missing_stats indicator for NULL plyr_rush_tz_usage to {mask_tz_null.sum()} rows with tz attempts")

            # Case 2: plyr_rush_tz_att = 0 and plyr_rush_tz_usage is NULL
            mask_tz_no_att = df['plyr_rush_tz_att'] == 0
            mask_tz_no_att_null = mask_tz_no_att & df['plyr_rush_tz_usage'].isnull()
            df.loc[mask_tz_no_att_null, 'plyr_rush_tz_usage'] = -1
            if mask_tz_no_att_null.sum() > 0:
                df.loc[mask_tz_no_att_null, 'plyr_rz_rush_tz_no_att'] = 1
            logger.info(f"Applied plyr_rz_rush_tz_no_att indicator for NULL plyr_rush_tz_usage to {mask_tz_no_att_null.sum()} rows with no tz attempts")

        # Handle plyr_rush_rz_usage NULL values
        if all(col in df.columns for col in ['plyr_rush_rz_att', 'plyr_rush_rz_usage']):
            # Case 1: plyr_rush_rz_att != 0 and plyr_rush_rz_usage is NULL
            mask_rz_has_att = df['plyr_rush_rz_att'] != 0
            mask_rz_null = mask_rz_has_att & df['plyr_rush_rz_usage'].isnull()
            df.loc[mask_rz_null, 'plyr_rush_rz_usage'] = -1
            if mask_rz_null.sum() > 0:
                df.loc[mask_rz_null, 'plyr_rz_rush_missing_stats'] = 1
            logger.info(f"Applied plyr_rz_rush_missing_stats indicator for NULL plyr_rush_rz_usage to {mask_rz_null.sum()} rows with rz attempts")

            # Case 2: plyr_rush_rz_att = 0 and plyr_rush_rz_usage is NULL
            mask_rz_no_att = df['plyr_rush_rz_att'] == 0
            mask_rz_no_att_null = mask_rz_no_att & df['plyr_rush_rz_usage'].isnull()
            df.loc[mask_rz_no_att_null, 'plyr_rush_rz_usage'] = -1
            if mask_rz_no_att_null.sum() > 0:
                df.loc[mask_rz_no_att_null, 'plyr_rz_rush_no_att'] = 1
            logger.info(f"Applied plyr_rz_rush_no_att indicator for NULL plyr_rush_rz_usage to {mask_rz_no_att_null.sum()} rows with no rz attempts")

        return df

    def process_plyr_rz_rush_table(self, table_path: str) -> None:
        """Process the plyr_rz_rush table for null value handling"""
        self.process_table('plyr_rz_rush', table_path, self.handle_plyr_rz_rush_nulls)

def main():
    # Initialize handler
    raw_dir = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw"
    handler = PlyrRzRushNullHandler(raw_dir=raw_dir)

    # Process plyr_rz_rush table
    plyr_rz_rush_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw\plyr_szn\plyr_rz_rush"
    handler.process_plyr_rz_rush_table(plyr_rz_rush_path)

    # Print final summary
    handler.print_final_summary()

if __name__ == "__main__":
    main()