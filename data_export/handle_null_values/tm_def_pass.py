import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path to import null_utils
sys.path.append(str(Path(__file__).parent.parent))
from null_utils import BaseNullHandler, logger, parse_args, parse_season_filter, parse_week_filter

class TmDefPassNullHandler(BaseNullHandler):
    def __init__(self, raw_dir: str, output_dir: str = None):
        super().__init__(raw_dir, output_dir)

    def handle_tm_def_pass_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values for tm_def_pass table"""
        logger.info("Starting tm_def_pass null value handling...")

        # Create a copy to avoid modifying original
        df = df.copy()

        # Initialize indicator columns
        df['tm_def_no_int'] = 0
        df['tm_def_no_pass_td'] = 0
        df['tm_def_no_sk'] = 0

        # Rule 1: Handle tm_def_int and tm_def_sk - NULL -> 0 (no indicator)
        if 'tm_def_int' in df.columns:
            mask_int_null = df['tm_def_int'].isnull()
            if mask_int_null.sum() > 0:
                df.loc[mask_int_null, 'tm_def_int'] = 0
                logger.info(f"Imputed tm_def_int NULL values with 0 for {mask_int_null.sum()} rows")

        if 'tm_def_sk' in df.columns:
            mask_sk_null = df['tm_def_sk'].isnull()
            if mask_sk_null.sum() > 0:
                df.loc[mask_sk_null, 'tm_def_sk'] = 0
                logger.info(f"Imputed tm_def_sk NULL values with 0 for {mask_sk_null.sum()} rows")

        # Rule 2: Handle tm_def_int_pct when tm_def_int is 0 or NULL
        if all(col in df.columns for col in ['tm_def_int', 'tm_def_int_pct']):
            # After rule 1, tm_def_int should have no NULLs, but check both 0 and NULL for safety
            mask_no_int = (df['tm_def_int'] == 0) | (df['tm_def_int'].isnull())
            mask_int_pct_null = mask_no_int & df['tm_def_int_pct'].isnull()

            if mask_int_pct_null.sum() > 0:
                df.loc[mask_int_pct_null, 'tm_def_int_pct'] = -999
                df.loc[mask_int_pct_null, 'tm_def_no_int'] = 1
                logger.info(f"Applied tm_def_no_int indicator for {mask_int_pct_null.sum()} rows with no interceptions")

        # Rule 3: Handle tm_def_pass_td_pct when tm_def_pass_td is 0
        if all(col in df.columns for col in ['tm_def_pass_td', 'tm_def_pass_td_pct']):
            mask_no_pass_td = df['tm_def_pass_td'] == 0
            mask_pass_td_pct_null = mask_no_pass_td & df['tm_def_pass_td_pct'].isnull()

            if mask_pass_td_pct_null.sum() > 0:
                df.loc[mask_pass_td_pct_null, 'tm_def_pass_td_pct'] = -999
                df.loc[mask_pass_td_pct_null, 'tm_def_no_pass_td'] = 1
                logger.info(f"Applied tm_def_no_pass_td indicator for {mask_pass_td_pct_null.sum()} rows with no passing touchdowns")

        # Rule 4: Handle tm_def_sk_pct and tm_def_sk_yds when tm_def_sk is 0 or NULL
        if 'tm_def_sk' in df.columns:
            # After rule 1, tm_def_sk should have no NULLs, but check both 0 and NULL for safety
            mask_no_sk = (df['tm_def_sk'] == 0) | (df['tm_def_sk'].isnull())

            # Handle tm_def_sk_pct
            if 'tm_def_sk_pct' in df.columns:
                mask_sk_pct_null = mask_no_sk & df['tm_def_sk_pct'].isnull()
                if mask_sk_pct_null.sum() > 0:
                    df.loc[mask_sk_pct_null, 'tm_def_sk_pct'] = -999
                    df.loc[mask_sk_pct_null, 'tm_def_no_sk'] = 1
                    logger.info(f"Applied tm_def_no_sk indicator for tm_def_sk_pct: {mask_sk_pct_null.sum()} rows with no sacks")

            # Handle tm_def_sk_yds
            if 'tm_def_sk_yds' in df.columns:
                mask_sk_yds_null = mask_no_sk & df['tm_def_sk_yds'].isnull()
                if mask_sk_yds_null.sum() > 0:
                    df.loc[mask_sk_yds_null, 'tm_def_sk_yds'] = -999
                    # Set indicator to 1 if not already set
                    df.loc[mask_sk_yds_null, 'tm_def_no_sk'] = 1
                    logger.info(f"Applied tm_def_no_sk indicator for tm_def_sk_yds: {mask_sk_yds_null.sum()} rows with no sacks")

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
    handler = TmDefPassNullHandler(raw_dir=raw_dir, output_dir=clean_dir)

    # Log filter information
    if seasons:
        logger.info(f"Processing seasons: {seasons}")
    else:
        logger.info("Processing all seasons")

    if weeks:
        logger.info(f"Processing weeks: {weeks}")
    else:
        logger.info("Processing all weeks")

    # Process tm_def_pass table with partitioning
    tm_def_pass_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw\tm_szn\tm_def_pass"
    handler.process_partitioned_table(
        table_name='tm_def_pass',
        table_path=tm_def_pass_path,
        category='tm_szn',
        handler_func=handler.handle_tm_def_pass_nulls,
        seasons=seasons,
        weeks=weeks
    )

    # Print final summary
    handler.print_final_summary()

if __name__ == "__main__":
    main()
