import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path to import null_utils
sys.path.append(str(Path(__file__).parent.parent))
from null_utils import BaseNullHandler, logger, parse_args, parse_season_filter, parse_week_filter

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
                    df.loc[mask_null, col] = -999
                    if mask_null.sum() > 0:
                        df.loc[mask_null, 'plyr_rz_pass_no_tz_att'] = 1
                        any_imputed = True

            logger.info(f"Applied plyr_rz_pass_no_tz_att indicator for NULL values to {mask_no_tz_att.sum()} rows with no touchdown zone attempts")

        return df

def main():
    # Parse command line arguments
    args = parse_args()

    # Parse season and week filters
    seasons = parse_season_filter(args.season) if args.season else None
    weeks = parse_week_filter(args.week) if args.week else None

    # Initialize handler with output directory for clean parquet files
    raw_dir = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw"
    clean_dir = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\clean"
    handler = PlyrRzPassNullHandler(raw_dir=raw_dir, output_dir=clean_dir)

    # Log filter information
    if seasons:
        logger.info(f"Processing seasons: {seasons}")
    else:
        logger.info("Processing all seasons")

    if weeks:
        logger.info(f"Processing weeks: {weeks}")
    else:
        logger.info("Processing all weeks")

    # Process plyr_rz_pass table with partitioning
    plyr_rz_pass_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw\plyr_szn\plyr_rz_pass"
    handler.process_partitioned_table(
        table_name='plyr_rz_pass',
        table_path=plyr_rz_pass_path,
        category='plyr_szn',
        handler_func=handler.handle_plyr_rz_pass_nulls,
        seasons=seasons,
        weeks=weeks
    )

    # Print final summary
    handler.print_final_summary()

if __name__ == "__main__":
    main()