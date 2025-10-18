import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path to import null_utils
sys.path.append(str(Path(__file__).parent.parent))
from null_utils import BaseNullHandler, logger, parse_args, parse_season_filter, parse_week_filter

class TmRecNullHandler(BaseNullHandler):
    def __init__(self, raw_dir: str, output_dir: str = None):
        super().__init__(raw_dir, output_dir)

    def handle_tm_rec_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values for tm_rec table"""
        logger.info("Starting tm_rec null value handling...")

        # Create a copy to avoid modifying original
        df = df.copy()

        # Handle tm_rec_brkn_tkl_rec: NULL -> -999 when tm_rec_brkn_tkl == 0
        if 'tm_rec_brkn_tkl_rec' in df.columns and 'tm_rec_brkn_tkl' in df.columns:
            # Create indicator column first (initialize to 0)
            df['tm_rec_no_brkn_tkl'] = 0

            # Identify rows where tm_rec_brkn_tkl == 0 AND tm_rec_brkn_tkl_rec is NULL
            mask = (df['tm_rec_brkn_tkl'] == 0) & (df['tm_rec_brkn_tkl_rec'].isnull())
            null_count = mask.sum()

            if null_count > 0:
                # Impute NULL values with -999
                df.loc[mask, 'tm_rec_brkn_tkl_rec'] = -999
                # Set indicator to 1 for imputed rows
                df.loc[mask, 'tm_rec_no_brkn_tkl'] = 1
                logger.info(f"Imputed {null_count} NULL values in tm_rec_brkn_tkl_rec with -999")
                logger.info(f"Created indicator column 'tm_rec_no_brkn_tkl' (marked {null_count} rows)")

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
    handler = TmRecNullHandler(raw_dir=raw_dir, output_dir=clean_dir)

    # Log filter information
    if seasons:
        logger.info(f"Processing seasons: {seasons}")
    else:
        logger.info("Processing all seasons")

    if weeks:
        logger.info(f"Processing weeks: {weeks}")
    else:
        logger.info("Processing all weeks")

    # Process tm_rec table with partitioning
    tm_rec_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw\tm_szn\tm_rec"
    handler.process_partitioned_table(
        table_name='tm_rec',
        table_path=tm_rec_path,
        category='tm_szn',
        handler_func=handler.handle_tm_rec_nulls,
        seasons=seasons,
        weeks=weeks
    )

    # Print final summary
    handler.print_final_summary()

if __name__ == "__main__":
    main()
