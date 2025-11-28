import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path to import null_utils
sys.path.append(str(Path(__file__).parent.parent))
from null_utils import BaseNullHandler, logger, parse_args, parse_season_filter, parse_week_filter

class NflFastrWrNullHandler(BaseNullHandler):
    def __init__(self, raw_dir: str, output_dir: str = None):
        super().__init__(raw_dir, output_dir)

    def handle_nfl_fastr_wr_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values for nfl_fastr_wr table"""
        logger.info("Starting nfl_fastr_wr null value handling...")

        # Create a copy to avoid modifying original
        df = df.copy()

        # Initialize indicator column
        df['nfl_fastr_wr_no_yac'] = 0

        # Columns that require null handling
        yac_cols = [
            'plyr_gm_rec_avg_expected_yac',
            'plyr_gm_rec_avg_yac',
            'plyr_gm_rec_avg_yac_above_expectation'
        ]

        # Handle null values - if any of the affected columns are null, impute with -999 and set indicator
        for col in yac_cols:
            if col in df.columns:
                mask_null = df[col].isnull()
                df.loc[mask_null, col] = -999
                if mask_null.sum() > 0:
                    df.loc[mask_null, 'nfl_fastr_wr_no_yac'] = 1

        # Log the number of rows affected
        rows_affected = (df['nfl_fastr_wr_no_yac'] == 1).sum()
        logger.info(f"Applied nfl_fastr_wr_no_yac indicator to {rows_affected} rows with NULL YAC values")

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
    handler = NflFastrWrNullHandler(raw_dir=raw_dir, output_dir=clean_dir)

    # Log filter information
    if seasons:
        logger.info(f"Processing seasons: {seasons}")
    else:
        logger.info("Processing all seasons")

    if weeks:
        logger.info(f"Processing weeks: {weeks}")
    else:
        logger.info("Processing all weeks")

    # Process nfl_fastr_wr table with partitioning
    nfl_fastr_wr_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw\plyr_gm\nfl_fastr_wr"
    handler.process_partitioned_table(
        table_name='nfl_fastr_wr',
        table_path=nfl_fastr_wr_path,
        category='plyr_gm',
        handler_func=handler.handle_nfl_fastr_wr_nulls,
        seasons=seasons,
        weeks=weeks
    )

    # Print final summary
    handler.print_final_summary()

if __name__ == "__main__":
    main()
