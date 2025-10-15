import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path to import null_utils
sys.path.append(str(Path(__file__).parent.parent))
from null_utils import BaseNullHandler, logger, parse_args, parse_season_filter, parse_week_filter

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
                    df.loc[mask_null, col] = -999
                    if mask_null.sum() > 0:
                        df.loc[mask_null, 'plyr_gm_rush_no_rushes'] = 1

            logger.info(f"Applied plyr_gm_rush_no_rushes indicator for NULL values to {mask_no_rushes.sum()} rows with no rush attempts")

        # Handle plyr_gm_rush_no_brkn_tkl indicator
        if 'plyr_gm_rush_brkn_tkl' in df.columns and 'plyr_gm_rush_brkn_tkl_att' in df.columns:
            mask_no_brkn_tkl = df['plyr_gm_rush_brkn_tkl'] == 0
            mask_null = mask_no_brkn_tkl & df['plyr_gm_rush_brkn_tkl_att'].isnull()
            df.loc[mask_null, 'plyr_gm_rush_brkn_tkl_att'] = -999
            df.loc[mask_null, 'plyr_gm_rush_no_brkn_tkl'] = 1

            logger.info(f"Applied plyr_gm_rush_no_brkn_tkl indicator to {mask_null.sum()} rows")

        # Handle plyr_gm_rush_no_first_dwn indicator
        if 'plyr_gm_rush_first_dwn' in df.columns:
            mask_null = df['plyr_gm_rush_first_dwn'].isnull()
            df.loc[mask_null, 'plyr_gm_rush_first_dwn'] = -999
            df.loc[mask_null, 'plyr_gm_rush_no_first_dwn'] = 1

            logger.info(f"Applied plyr_gm_rush_no_first_dwn indicator to {mask_null.sum()} rows")

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
    handler = PlyrGmRushNullHandler(raw_dir=raw_dir, output_dir=clean_dir)

    # Log filter information
    if seasons:
        logger.info(f"Processing seasons: {seasons}")
    else:
        logger.info("Processing all seasons")

    if weeks:
        logger.info(f"Processing weeks: {weeks}")
    else:
        logger.info("Processing all weeks")

    # Process plyr_gm_rush table with partitioning
    plyr_gm_rush_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw\plyr_gm\plyr_gm_rush"
    handler.process_partitioned_table(
        table_name='plyr_gm_rush',
        table_path=plyr_gm_rush_path,
        category='plyr_gm',
        handler_func=handler.handle_plyr_gm_rush_nulls,
        seasons=seasons,
        weeks=weeks
    )

    # Print final summary
    handler.print_final_summary()

if __name__ == "__main__":
    main()