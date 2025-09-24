import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path to import null_utils
sys.path.append(str(Path(__file__).parent.parent))
from null_utils import BaseNullHandler, logger

class PlyrRecNullHandler(BaseNullHandler):
    def __init__(self, raw_dir: str, output_dir: str = None):
        super().__init__(raw_dir, output_dir)

    def handle_plyr_rec_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values for plyr_rec table"""
        logger.info("Starting plyr_rec null value handling...")

        # Create a copy to avoid modifying original
        df = df.copy()

        # Initialize indicator columns
        df['plyr_rec_no_targets'] = 0
        df['plyr_rec_no_catches'] = 0
        df['plyr_rec_no_brkn_tkl'] = 0
        df['plyr_rec_no_positive_yds'] = 0

        # Handle plyr_rec_no_targets indicator for NULL values
        # When plyr_rec_tgt = 0 and various receiving columns are NULL
        no_targets_cols = [
            'plyr_rec_catch_pct', 'plyr_rec_yds_rec', 'plyr_rec_succ_rt', 'plyr_rec_lng',
            'plyr_rec_yds_tgt', 'plyr_rec_aybc', 'plyr_rec_aybc_rec', 'plyr_rec_yac',
            'plyr_rec_yac_rec', 'plyr_rec_adot', 'plyr_rec_brkn_tkl', 'plyr_rec_brkn_tkl_rec',
            'plyr_rec_drp', 'plyr_rec_drp_pct', 'plyr_rec_int', 'plyr_rec_pass_rtg'
        ]

        if 'plyr_rec_tgt' in df.columns:
            mask_no_targets = df['plyr_rec_tgt'] == 0
            any_imputed = False

            for col in no_targets_cols:
                if col in df.columns:
                    mask_null = mask_no_targets & df[col].isnull()
                    df.loc[mask_null, col] = -1
                    if mask_null.sum() > 0:
                        df.loc[mask_null, 'plyr_rec_no_targets'] = 1
                        any_imputed = True

            logger.info(f"Applied plyr_rec_no_targets indicator for NULL values to {mask_no_targets.sum()} rows with no targets")

        # Handle plyr_rec_no_catches indicator for NULL values
        # When plyr_rec = 0 and certain columns are NULL
        no_catches_cols = [
            'plyr_rec_yds_rec', 'plyr_rec_lng', 'plyr_rec_aybc_rec',
            'plyr_rec_yac_rec', 'plyr_rec_brkn_tkl_rec'
        ]

        if 'plyr_rec' in df.columns:
            mask_no_catches = df['plyr_rec'] == 0

            for col in no_catches_cols:
                if col in df.columns:
                    mask_null = mask_no_catches & df[col].isnull()
                    df.loc[mask_null, col] = -1
                    if mask_null.sum() > 0:
                        df.loc[mask_null, 'plyr_rec_no_catches'] = 1

            logger.info(f"Applied plyr_rec_no_catches indicator for NULL values to {mask_no_catches.sum()} rows with no catches")

        # Handle plyr_rec_no_brkn_tkl indicator for NULL values
        # When plyr_rec_brkn_tkl = 0 and plyr_rec_brkn_tkl_rec is NULL
        if all(col in df.columns for col in ['plyr_rec_brkn_tkl', 'plyr_rec_brkn_tkl_rec']):
            mask_no_brkn_tkl = df['plyr_rec_brkn_tkl'] == 0
            mask_null = mask_no_brkn_tkl & df['plyr_rec_brkn_tkl_rec'].isnull()
            df.loc[mask_null, 'plyr_rec_brkn_tkl_rec'] = -1
            if mask_null.sum() > 0:
                df.loc[mask_null, 'plyr_rec_no_brkn_tkl'] = 1

            logger.info(f"Applied plyr_rec_no_brkn_tkl indicator for NULL values to {mask_null.sum()} rows with no broken tackles")

        # Handle plyr_rec_no_positive_yds indicator for NULL plyr_rec_lng
        # When plyr_rec != 0 and plyr_rec_tgt != 0 and plyr_rec_lng is NULL
        if all(col in df.columns for col in ['plyr_rec', 'plyr_rec_tgt', 'plyr_rec_lng']):
            mask_has_rec_and_tgt = (df['plyr_rec'] != 0) & (df['plyr_rec_tgt'] != 0)
            mask_null = mask_has_rec_and_tgt & df['plyr_rec_lng'].isnull()
            df.loc[mask_null, 'plyr_rec_lng'] = -1
            if mask_null.sum() > 0:
                df.loc[mask_null, 'plyr_rec_no_positive_yds'] = 1

            logger.info(f"Applied plyr_rec_no_positive_yds indicator for NULL plyr_rec_lng to {mask_null.sum()} rows")

        return df

    def process_plyr_rec_table(self, table_path: str) -> None:
        """Process the plyr_rec table for null value handling"""
        self.process_table('plyr_rec', table_path, self.handle_plyr_rec_nulls)

def main():
    # Initialize handler
    raw_dir = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw"
    handler = PlyrRecNullHandler(raw_dir=raw_dir)

    # Process plyr_rec table
    plyr_rec_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw\plyr_szn\plyr_rec"
    handler.process_plyr_rec_table(plyr_rec_path)

    # Print final summary
    handler.print_final_summary()

if __name__ == "__main__":
    main()
