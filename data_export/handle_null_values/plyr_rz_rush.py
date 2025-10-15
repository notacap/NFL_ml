import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path to import null_utils
sys.path.append(str(Path(__file__).parent.parent))
from null_utils import BaseNullHandler, logger, parse_args, parse_season_filter, parse_week_filter

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
            df.loc[mask_fz_null, 'plyr_rush_fz_usage'] = -999
            if mask_fz_null.sum() > 0:
                df.loc[mask_fz_null, 'plyr_rz_rush_missing_stats'] = 1
            logger.info(f"Applied plyr_rz_rush_missing_stats indicator for NULL plyr_rush_fz_usage to {mask_fz_null.sum()} rows with fz attempts")

            # Case 2: plyr_rush_fz_att = 0 and plyr_rush_fz_usage is NULL
            mask_fz_no_att = df['plyr_rush_fz_att'] == 0
            mask_fz_no_att_null = mask_fz_no_att & df['plyr_rush_fz_usage'].isnull()
            df.loc[mask_fz_no_att_null, 'plyr_rush_fz_usage'] = -999
            if mask_fz_no_att_null.sum() > 0:
                df.loc[mask_fz_no_att_null, 'plyr_rz_rush_fz_no_att'] = 1
            logger.info(f"Applied plyr_rz_rush_fz_no_att indicator for NULL plyr_rush_fz_usage to {mask_fz_no_att_null.sum()} rows with no fz attempts")

        # Handle plyr_rush_tz_usage NULL values
        if all(col in df.columns for col in ['plyr_rush_tz_att', 'plyr_rush_tz_usage']):
            # Case 1: plyr_rush_tz_att != 0 and plyr_rush_tz_usage is NULL
            mask_tz_has_att = df['plyr_rush_tz_att'] != 0
            mask_tz_null = mask_tz_has_att & df['plyr_rush_tz_usage'].isnull()
            df.loc[mask_tz_null, 'plyr_rush_tz_usage'] = -999
            if mask_tz_null.sum() > 0:
                df.loc[mask_tz_null, 'plyr_rz_rush_missing_stats'] = 1
            logger.info(f"Applied plyr_rz_rush_missing_stats indicator for NULL plyr_rush_tz_usage to {mask_tz_null.sum()} rows with tz attempts")

            # Case 2: plyr_rush_tz_att = 0 and plyr_rush_tz_usage is NULL
            mask_tz_no_att = df['plyr_rush_tz_att'] == 0
            mask_tz_no_att_null = mask_tz_no_att & df['plyr_rush_tz_usage'].isnull()
            df.loc[mask_tz_no_att_null, 'plyr_rush_tz_usage'] = -999
            if mask_tz_no_att_null.sum() > 0:
                df.loc[mask_tz_no_att_null, 'plyr_rz_rush_tz_no_att'] = 1
            logger.info(f"Applied plyr_rz_rush_tz_no_att indicator for NULL plyr_rush_tz_usage to {mask_tz_no_att_null.sum()} rows with no tz attempts")

        # Handle plyr_rush_rz_usage NULL values
        if all(col in df.columns for col in ['plyr_rush_rz_att', 'plyr_rush_rz_usage']):
            # Case 1: plyr_rush_rz_att != 0 and plyr_rush_rz_usage is NULL
            mask_rz_has_att = df['plyr_rush_rz_att'] != 0
            mask_rz_null = mask_rz_has_att & df['plyr_rush_rz_usage'].isnull()
            df.loc[mask_rz_null, 'plyr_rush_rz_usage'] = -999
            if mask_rz_null.sum() > 0:
                df.loc[mask_rz_null, 'plyr_rz_rush_missing_stats'] = 1
            logger.info(f"Applied plyr_rz_rush_missing_stats indicator for NULL plyr_rush_rz_usage to {mask_rz_null.sum()} rows with rz attempts")

            # Case 2: plyr_rush_rz_att = 0 and plyr_rush_rz_usage is NULL
            mask_rz_no_att = df['plyr_rush_rz_att'] == 0
            mask_rz_no_att_null = mask_rz_no_att & df['plyr_rush_rz_usage'].isnull()
            df.loc[mask_rz_no_att_null, 'plyr_rush_rz_usage'] = -999
            if mask_rz_no_att_null.sum() > 0:
                df.loc[mask_rz_no_att_null, 'plyr_rz_rush_no_att'] = 1
            logger.info(f"Applied plyr_rz_rush_no_att indicator for NULL plyr_rush_rz_usage to {mask_rz_no_att_null.sum()} rows with no rz attempts")

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
    handler = PlyrRzRushNullHandler(raw_dir=raw_dir, output_dir=clean_dir)

    # Log filter information
    if seasons:
        logger.info(f"Processing seasons: {seasons}")
    else:
        logger.info("Processing all seasons")

    if weeks:
        logger.info(f"Processing weeks: {weeks}")
    else:
        logger.info("Processing all weeks")

    # Process plyr_rz_rush table with partitioning
    plyr_rz_rush_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw\plyr_szn\plyr_rz_rush"
    handler.process_partitioned_table(
        table_name='plyr_rz_rush',
        table_path=plyr_rz_rush_path,
        category='plyr_szn',
        handler_func=handler.handle_plyr_rz_rush_nulls,
        seasons=seasons,
        weeks=weeks
    )

    # Print final summary
    handler.print_final_summary()

if __name__ == "__main__":
    main()