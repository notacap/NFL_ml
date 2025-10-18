import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path to import null_utils
sys.path.append(str(Path(__file__).parent.parent))
from null_utils import BaseNullHandler, logger, parse_args, parse_season_filter, parse_week_filter

class TmRushNullHandler(BaseNullHandler):
    def __init__(self, raw_dir: str, output_dir: str = None):
        super().__init__(raw_dir, output_dir)

    def handle_tm_rush_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values for tm_rush table"""
        logger.info("Starting tm_rush null value handling...")

        # Create a copy to avoid modifying original
        df = df.copy()

        # Handle tm_rush_att_brkn_tkl: NULL -> -999 when tm_rush_brkn_tkl == 0
        # When no broken tackles occurred (tm_rush_brkn_tkl=0), attempts to break tackles are not applicable
        if 'tm_rush_att_brkn_tkl' in df.columns and 'tm_rush_brkn_tkl' in df.columns:
            # Create indicator column for when no broken tackles
            df['tm_rush_no_brkn_tkl'] = 0

            # Find rows where tm_rush_brkn_tkl == 0
            mask = df['tm_rush_brkn_tkl'] == 0

            # Set indicator to 1 for these rows
            df.loc[mask, 'tm_rush_no_brkn_tkl'] = 1

            # Count how many values will be imputed
            zero_count = mask.sum()
            if zero_count > 0:
                df.loc[mask, 'tm_rush_att_brkn_tkl'] = -999
                logger.info(f"Imputed {zero_count} values in tm_rush_att_brkn_tkl with -999 (where tm_rush_brkn_tkl == 0)")
                logger.info(f"Created indicator column tm_rush_no_brkn_tkl (set to 1 for {zero_count} rows)")

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
    handler = TmRushNullHandler(raw_dir=raw_dir, output_dir=clean_dir)

    # Log filter information
    if seasons:
        logger.info(f"Processing seasons: {seasons}")
    else:
        logger.info("Processing all seasons")

    if weeks:
        logger.info(f"Processing weeks: {weeks}")
    else:
        logger.info("Processing all weeks")

    # Process tm_rush table with partitioning
    tm_rush_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw\tm_szn\tm_rush"
    handler.process_partitioned_table(
        table_name='tm_rush',
        table_path=tm_rush_path,
        category='tm_szn',
        handler_func=handler.handle_tm_rush_nulls,
        seasons=seasons,
        weeks=weeks
    )

    # Print final summary
    handler.print_final_summary()

if __name__ == "__main__":
    main()
