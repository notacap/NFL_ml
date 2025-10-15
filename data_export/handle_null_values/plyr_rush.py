import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path to import null_utils
sys.path.append(str(Path(__file__).parent.parent))
from null_utils import BaseNullHandler, logger, parse_args, parse_season_filter, parse_week_filter

class PlyrRushNullHandler(BaseNullHandler):
    def __init__(self, raw_dir: str, output_dir: str = None):
        super().__init__(raw_dir, output_dir)

    def handle_plyr_rush_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values for plyr_rush table"""
        logger.info("Starting plyr_rush null value handling...")

        # Create a copy to avoid modifying original
        df = df.copy()

        # Initialize indicator columns
        df['plyr_rush_no_att'] = 0
        df['plyr_rush_no_brkn_tkl'] = 0
        df['plyr_rush_no_positive_yds'] = 0

        # Handle plyr_rush_no_att indicator for NULL values
        # When plyr_rush_att = 0 and various rushing columns are NULL
        no_att_cols = [
            'plyr_rush_succ_rt', 'plyr_rush_lng', 'plyr_rush_yds_att',
            'plyr_rush_ybc', 'plyr_rush_ybc_att', 'plyr_rush_yac',
            'plyr_rush_yac_att', 'plyr_rush_brkn_tkl', 'plyr_rush_brkn_tkl_att'
        ]

        if 'plyr_rush_att' in df.columns:
            mask_no_att = df['plyr_rush_att'] == 0
            any_imputed = False

            for col in no_att_cols:
                if col in df.columns:
                    mask_null = mask_no_att & df[col].isnull()
                    df.loc[mask_null, col] = -999
                    if mask_null.sum() > 0:
                        df.loc[mask_null, 'plyr_rush_no_att'] = 1
                        any_imputed = True

            logger.info(f"Applied plyr_rush_no_att indicator for NULL values to {mask_no_att.sum()} rows with no attempts")

        # Handle plyr_rush_no_brkn_tkl indicator for NULL values
        # When plyr_rush_brkn_tkl = 0 and plyr_rush_brkn_tkl_att is NULL
        if all(col in df.columns for col in ['plyr_rush_brkn_tkl', 'plyr_rush_brkn_tkl_att']):
            mask_no_brkn_tkl = df['plyr_rush_brkn_tkl'] == 0
            mask_null = mask_no_brkn_tkl & df['plyr_rush_brkn_tkl_att'].isnull()
            df.loc[mask_null, 'plyr_rush_brkn_tkl_att'] = -999
            if mask_null.sum() > 0:
                df.loc[mask_null, 'plyr_rush_no_brkn_tkl'] = 1

            logger.info(f"Applied plyr_rush_no_brkn_tkl indicator for NULL values to {mask_null.sum()} rows with no broken tackles")

        # Handle plyr_rush_no_positive_yds indicator for NULL plyr_rush_lng
        # When plyr_rush_att != 0 and plyr_rush_yds <= 0 and plyr_rush_lng is NULL
        if all(col in df.columns for col in ['plyr_rush_att', 'plyr_rush_yds', 'plyr_rush_lng']):
            mask_has_att_no_positive_yds = (df['plyr_rush_att'] != 0) & (df['plyr_rush_yds'] <= 0)
            mask_null = mask_has_att_no_positive_yds & df['plyr_rush_lng'].isnull()
            df.loc[mask_null, 'plyr_rush_lng'] = -999
            if mask_null.sum() > 0:
                df.loc[mask_null, 'plyr_rush_no_positive_yds'] = 1

            logger.info(f"Applied plyr_rush_no_positive_yds indicator for NULL plyr_rush_lng to {mask_null.sum()} rows")

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
    handler = PlyrRushNullHandler(raw_dir=raw_dir, output_dir=clean_dir)

    # Log filter information
    if seasons:
        logger.info(f"Processing seasons: {seasons}")
    else:
        logger.info("Processing all seasons")

    if weeks:
        logger.info(f"Processing weeks: {weeks}")
    else:
        logger.info("Processing all weeks")

    # Process plyr_rush table with partitioning
    plyr_rush_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw\plyr_szn\plyr_rush"
    handler.process_partitioned_table(
        table_name='plyr_rush',
        table_path=plyr_rush_path,
        category='plyr_szn',
        handler_func=handler.handle_plyr_rush_nulls,
        seasons=seasons,
        weeks=weeks
    )

    # Print final summary
    handler.print_final_summary()

if __name__ == "__main__":
    main()