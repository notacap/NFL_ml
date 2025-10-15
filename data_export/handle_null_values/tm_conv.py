import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path to import null_utils
sys.path.append(str(Path(__file__).parent.parent))
from null_utils import BaseNullHandler, logger, parse_args, parse_season_filter, parse_week_filter

class TmConvNullHandler(BaseNullHandler):
    def __init__(self, raw_dir: str, output_dir: str = None):
        super().__init__(raw_dir, output_dir)

    def handle_tm_conv_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values for tm_conv table"""
        logger.info("Starting tm_conv null value handling...")

        # Create a copy to avoid modifying original
        df = df.copy()

        # Initialize indicator column
        df['tm_conv_no_fourth_dwn_att'] = 0

        # Handle tm_conv_no_fourth_dwn_att indicator for NULL values
        # When tm_fourth_dwn_att = 0 and tm_fourth_dwn_conv_pct is NULL
        if all(col in df.columns for col in ['tm_fourth_dwn_att', 'tm_fourth_dwn_conv_pct']):
            mask_no_att = df['tm_fourth_dwn_att'] == 0
            mask_null = mask_no_att & df['tm_fourth_dwn_conv_pct'].isnull()
            df.loc[mask_null, 'tm_fourth_dwn_conv_pct'] = -999
            if mask_null.sum() > 0:
                df.loc[mask_null, 'tm_conv_no_fourth_dwn_att'] = 1

            logger.info(f"Applied tm_conv_no_fourth_dwn_att indicator for NULL values to {mask_null.sum()} rows with no fourth down attempts")

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
    handler = TmConvNullHandler(raw_dir=raw_dir, output_dir=clean_dir)

    # Log filter information
    if seasons:
        logger.info(f"Processing seasons: {seasons}")
    else:
        logger.info("Processing all seasons")

    if weeks:
        logger.info(f"Processing weeks: {weeks}")
    else:
        logger.info("Processing all weeks")

    # Process tm_conv table with partitioning
    tm_conv_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw\tm_szn\tm_conv"
    handler.process_partitioned_table(
        table_name='tm_conv',
        table_path=tm_conv_path,
        category='tm_szn',
        handler_func=handler.handle_tm_conv_nulls,
        seasons=seasons,
        weeks=weeks
    )

    # Print final summary
    handler.print_final_summary()

if __name__ == "__main__":
    main()
