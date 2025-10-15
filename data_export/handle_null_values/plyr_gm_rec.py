import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path to import null_utils
sys.path.append(str(Path(__file__).parent.parent))
from null_utils import BaseNullHandler, logger, parse_args, parse_season_filter, parse_week_filter

class PlyrGmRecNullHandler(BaseNullHandler):
    def __init__(self, raw_dir: str, output_dir: str = None):
        super().__init__(raw_dir, output_dir)

    def handle_plyr_gm_rec_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values for plyr_gm_rec table"""
        logger.info("Starting plyr_gm_rec null value handling...")

        # Create a copy to avoid modifying original
        df = df.copy()

        # Initialize indicator columns
        df['plyr_gm_rec_no_catches'] = 0
        df['plyr_gm_rec_no_brkn_tkl'] = 0
        df['plyr_gm_rec_no_drops'] = 0
        df['plyr_gm_rec_no_first_dwn'] = 0

        # Handle plyr_gm_rec_no_catches indicator for NULL values
        no_catches_null_cols = [
            'plyr_gm_rec_first_dwn', 'plyr_gm_rec_aybc_route', 'plyr_gm_rec_yac_route'
        ]

        # Check if plyr_gm_rec = 0 and affected columns are NULL
        if 'plyr_gm_rec' in df.columns:
            mask_no_catches = df['plyr_gm_rec'] == 0
            for col in no_catches_null_cols:
                if col in df.columns:
                    mask_null = mask_no_catches & df[col].isnull()
                    df.loc[mask_null, col] = -999
                    if mask_null.sum() > 0:
                        df.loc[mask_null, 'plyr_gm_rec_no_catches'] = 1

            logger.info(f"Applied plyr_gm_rec_no_catches indicator for NULL values to {mask_no_catches.sum()} rows with no receptions")

        # Handle plyr_gm_rec_no_brkn_tkl indicator for NULL values
        if 'plyr_gm_rec_brkn_tkl' in df.columns and 'plyr_gm_rec_brkn_tkl_rec' in df.columns:
            mask_no_brkn_tkl = df['plyr_gm_rec_brkn_tkl'] == 0
            mask_null = mask_no_brkn_tkl & df['plyr_gm_rec_brkn_tkl_rec'].isnull()
            df.loc[mask_null, 'plyr_gm_rec_brkn_tkl_rec'] = -999
            df.loc[mask_null, 'plyr_gm_rec_no_brkn_tkl'] = 1

            logger.info(f"Applied plyr_gm_rec_no_brkn_tkl indicator to {mask_null.sum()} rows")

        # Handle plyr_gm_rec_no_first_dwn indicator for NULL values
        # When player has receptions but no first downs
        if 'plyr_gm_rec' in df.columns and 'plyr_gm_rec_first_dwn' in df.columns:
            mask_has_rec_no_fd = (df['plyr_gm_rec'] != 0) & df['plyr_gm_rec_first_dwn'].isnull()
            df.loc[mask_has_rec_no_fd, 'plyr_gm_rec_first_dwn'] = -999
            df.loc[mask_has_rec_no_fd, 'plyr_gm_rec_no_first_dwn'] = 1

            logger.info(f"Applied plyr_gm_rec_no_first_dwn indicator to {mask_has_rec_no_fd.sum()} rows with receptions but no first downs")


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
    handler = PlyrGmRecNullHandler(raw_dir=raw_dir, output_dir=clean_dir)

    # Log filter information
    if seasons:
        logger.info(f"Processing seasons: {seasons}")
    else:
        logger.info("Processing all seasons")

    if weeks:
        logger.info(f"Processing weeks: {weeks}")
    else:
        logger.info("Processing all weeks")

    # Process plyr_gm_rec table with partitioning
    plyr_gm_rec_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw\plyr_gm\plyr_gm_rec"
    handler.process_partitioned_table(
        table_name='plyr_gm_rec',
        table_path=plyr_gm_rec_path,
        category='plyr_gm',
        handler_func=handler.handle_plyr_gm_rec_nulls,
        seasons=seasons,
        weeks=weeks
    )

    # Print final summary
    handler.print_final_summary()

if __name__ == "__main__":
    main()