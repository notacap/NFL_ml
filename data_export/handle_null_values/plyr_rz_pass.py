import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path to import null_utils
sys.path.append(str(Path(__file__).parent.parent))
from null_utils import BaseNullHandler, logger

class PlyrRzPassNullHandler(BaseNullHandler):
    def __init__(self, raw_dir: str, output_dir: str = None):
        super().__init__(raw_dir, output_dir)

    def handle_plyr_rz_pass_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values for plyr_rz_pass table"""
        logger.info("Starting plyr_rz_pass null value handling...")

        # Create a copy to avoid modifying original
        df = df.copy()

        # Initialize indicator columns
        df['plyr_rz_pass_no_tz_att'] = 0

        # Handle plyr_rz_pass_no_tz_att indicator for NULL values
        # When plyr_pass_tz_att = 0 and plyr_pass_tz_cmp_pct is NULL
        no_tz_att_cols = ['plyr_pass_tz_cmp_pct']

        if 'plyr_pass_tz_att' in df.columns:
            mask_no_tz_att = df['plyr_pass_tz_att'] == 0
            any_imputed = False

            for col in no_tz_att_cols:
                if col in df.columns:
                    mask_null = mask_no_tz_att & df[col].isnull()
                    df.loc[mask_null, col] = -1
                    if mask_null.sum() > 0:
                        df.loc[mask_null, 'plyr_rz_pass_no_tz_att'] = 1
                        any_imputed = True

            logger.info(f"Applied plyr_rz_pass_no_tz_att indicator for NULL values to {mask_no_tz_att.sum()} rows with no touchdown zone attempts")

        return df

    def process_plyr_rz_pass_table(self, table_path: str) -> None:
        """Process the plyr_rz_pass table for null value handling"""
        self.process_table('plyr_rz_pass', table_path, self.handle_plyr_rz_pass_nulls)

def main():
    # Initialize handler
    raw_dir = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw"
    handler = PlyrRzPassNullHandler(raw_dir=raw_dir)

    # Process plyr_rz_pass table
    plyr_rz_pass_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw\plyr_szn\plyr_rz_pass"
    handler.process_plyr_rz_pass_table(plyr_rz_pass_path)

    # Print final summary
    handler.print_final_summary()

if __name__ == "__main__":
    main()