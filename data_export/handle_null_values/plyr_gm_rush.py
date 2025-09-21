import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path to import null_utils
sys.path.append(str(Path(__file__).parent.parent))
from null_utils import BaseNullHandler, logger

class PlyrGmRushNullHandler(BaseNullHandler):
    def __init__(self, raw_dir: str, output_dir: str = None):
        super().__init__(raw_dir, output_dir)

    def handle_plyr_gm_rush_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values for plyr_gm_rush table"""
        logger.info("Starting plyr_gm_rush null value handling...")

        # Create a copy to avoid modifying original
        df = df.copy()

        # Initialize indicator columns
        df['plyr_gm_rush_no_rushes'] = 0
        df['plyr_gm_rush_no_brkn_tkl'] = 0
        df['plyr_gm_rush_no_first_dwn'] = 0

        # Handle plyr_gm_rush_no_rushes indicator for NULL values
        no_rushes_cols = [
            'plyr_gm_rush_yds_att', 'plyr_gm_rush_ybc_att',
            'plyr_gm_rush_yac_att', 'plyr_gm_rush_brkn_tkl_att'
        ]

        # Check if plyr_gm_rush_att = 0 and affected columns are NULL
        if 'plyr_gm_rush_att' in df.columns:
            mask_no_rushes = df['plyr_gm_rush_att'] == 0
            for col in no_rushes_cols:
                if col in df.columns:
                    mask_null = mask_no_rushes & df[col].isnull()
                    df.loc[mask_null, col] = -1
                    if mask_null.sum() > 0:
                        df.loc[mask_null, 'plyr_gm_rush_no_rushes'] = 1

            logger.info(f"Applied plyr_gm_rush_no_rushes indicator for NULL values to {mask_no_rushes.sum()} rows with no rush attempts")

        # Handle plyr_gm_rush_no_brkn_tkl indicator
        if 'plyr_gm_rush_brkn_tkl' in df.columns and 'plyr_gm_rush_brkn_tkl_att' in df.columns:
            mask_no_brkn_tkl = df['plyr_gm_rush_brkn_tkl'] == 0
            mask_null = mask_no_brkn_tkl & df['plyr_gm_rush_brkn_tkl_att'].isnull()
            df.loc[mask_null, 'plyr_gm_rush_brkn_tkl_att'] = -1
            df.loc[mask_null, 'plyr_gm_rush_no_brkn_tkl'] = 1

            logger.info(f"Applied plyr_gm_rush_no_brkn_tkl indicator to {mask_null.sum()} rows")

        # Handle plyr_gm_rush_no_first_dwn indicator
        if 'plyr_gm_rush_first_dwn' in df.columns:
            mask_null = df['plyr_gm_rush_first_dwn'].isnull()
            df.loc[mask_null, 'plyr_gm_rush_first_dwn'] = -1
            df.loc[mask_null, 'plyr_gm_rush_no_first_dwn'] = 1

            logger.info(f"Applied plyr_gm_rush_no_first_dwn indicator to {mask_null.sum()} rows")

        return df

    def process_plyr_gm_rush_table(self, table_path: str) -> None:
        """Process the plyr_gm_rush table for null value handling"""
        self.process_table('plyr_gm_rush', table_path, self.handle_plyr_gm_rush_nulls)

def main():
    # Initialize handler
    raw_dir = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw"
    handler = PlyrGmRushNullHandler(raw_dir=raw_dir)

    # Process plyr_gm_rush table
    plyr_gm_rush_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw\plyr_gm\plyr_gm_rush"
    handler.process_plyr_gm_rush_table(plyr_gm_rush_path)

    # Print final summary
    handler.print_final_summary()

if __name__ == "__main__":
    main()