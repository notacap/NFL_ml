import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path to import null_utils
sys.path.append(str(Path(__file__).parent.parent))
from null_utils import BaseNullHandler, logger, parse_args, parse_season_filter, parse_week_filter

class TmDefNullHandler(BaseNullHandler):
    def __init__(self, raw_dir: str, output_dir: str = None):
        super().__init__(raw_dir, output_dir)

    def handle_tm_def_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values for tm_def table"""
        logger.info("Starting tm_def null value handling...")

        # Create a copy to avoid modifying original
        df = df.copy()

        # Handle tm_def_penalty_first_dwn: NULL -> 0 (no indicator)
        if 'tm_def_penalty_first_dwn' in df.columns:
            null_count = df['tm_def_penalty_first_dwn'].isnull().sum()
            if null_count > 0:
                df['tm_def_penalty_first_dwn'] = df['tm_def_penalty_first_dwn'].fillna(0)
                logger.info(f"Imputed {null_count} NULL values in tm_def_penalty_first_dwn with 0")

        # Handle tm_def_tkawy: NULL -> 0 (no indicator)
        if 'tm_def_tkawy' in df.columns:
            null_count = df['tm_def_tkawy'].isnull().sum()
            if null_count > 0:
                df['tm_def_tkawy'] = df['tm_def_tkawy'].fillna(0)
                logger.info(f"Imputed {null_count} NULL values in tm_def_tkawy with 0")

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
    handler = TmDefNullHandler(raw_dir=raw_dir, output_dir=clean_dir)

    # Log filter information
    if seasons:
        logger.info(f"Processing seasons: {seasons}")
    else:
        logger.info("Processing all seasons")

    if weeks:
        logger.info(f"Processing weeks: {weeks}")
    else:
        logger.info("Processing all weeks")

    # Process tm_def table with partitioning
    tm_def_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw\tm_szn\tm_def"
    handler.process_partitioned_table(
        table_name='tm_def',
        table_path=tm_def_path,
        category='tm_szn',
        handler_func=handler.handle_tm_def_nulls,
        seasons=seasons,
        weeks=weeks
    )

    # Print final summary
    handler.print_final_summary()

if __name__ == "__main__":
    main()
