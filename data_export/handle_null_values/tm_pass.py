import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path to import null_utils
sys.path.append(str(Path(__file__).parent.parent))
from null_utils import BaseNullHandler, logger, parse_args, parse_season_filter, parse_week_filter

class TmPassNullHandler(BaseNullHandler):
    def __init__(self, raw_dir: str, output_dir: str = None):
        super().__init__(raw_dir, output_dir)

    def handle_tm_pass_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values for tm_pass table"""
        logger.info("Starting tm_pass null value handling...")

        # Create a copy to avoid modifying original
        df = df.copy()

        # Rule Group 1: Count statistics default to zero when missing
        # Columns: tm_fqc, tm_gwd, tm_pass_int, tm_pass_sk
        for col in ['tm_fqc', 'tm_gwd', 'tm_pass_int', 'tm_pass_sk']:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    df[col] = df[col].fillna(0)
                    logger.info(f"Imputed {null_count} NULL values in {col} with 0")

        # Rule Group 2: tm_pass_int_pct
        # Percentage undefined when no interceptions attempted
        if 'tm_pass_int_pct' in df.columns:
            # Create indicator column for when no interceptions
            df['tm_pass_no_int'] = 0

            # Find rows where tm_pass_int is NULL or 0
            mask = (df['tm_pass_int'].isnull()) | (df['tm_pass_int'] == 0)

            # Set indicator to 1 for these rows
            df.loc[mask, 'tm_pass_no_int'] = 1

            # Impute tm_pass_int_pct with -999 where tm_pass_int is NULL or 0
            null_or_zero_count = mask.sum()
            if null_or_zero_count > 0:
                df.loc[mask, 'tm_pass_int_pct'] = -999
                logger.info(f"Imputed {null_or_zero_count} NULL/zero values in tm_pass_int_pct with -999")
                logger.info(f"Created indicator column tm_pass_no_int (set to 1 for {null_or_zero_count} rows)")

        # Rule Group 3: tm_pass_sack_yds, tm_pass_sk_pct
        # Sack statistics undefined when no sacks occurred
        if 'tm_pass_sack_yds' in df.columns or 'tm_pass_sk_pct' in df.columns:
            # Create indicator column for when no sacks
            df['tm_pass_no_sk'] = 0

            # Find rows where tm_pass_sk is NULL or 0
            mask = (df['tm_pass_sk'].isnull()) | (df['tm_pass_sk'] == 0)

            # Set indicator to 1 for these rows
            df.loc[mask, 'tm_pass_no_sk'] = 1

            null_or_zero_count = mask.sum()
            if null_or_zero_count > 0:
                logger.info(f"Created indicator column tm_pass_no_sk (set to 1 for {null_or_zero_count} rows)")

                # Impute tm_pass_sack_yds with -999 where tm_pass_sk is NULL or 0
                if 'tm_pass_sack_yds' in df.columns:
                    df.loc[mask, 'tm_pass_sack_yds'] = -999
                    logger.info(f"Imputed {null_or_zero_count} NULL/zero values in tm_pass_sack_yds with -999")

                # Impute tm_pass_sk_pct with -999 where tm_pass_sk is NULL or 0
                if 'tm_pass_sk_pct' in df.columns:
                    df.loc[mask, 'tm_pass_sk_pct'] = -999
                    logger.info(f"Imputed {null_or_zero_count} NULL/zero values in tm_pass_sk_pct with -999")

        # Rule Group 4: tm_pass_yds_scrmbl
        # Scramble yards undefined when no scrambles occurred
        if 'tm_pass_yds_scrmbl' in df.columns and 'tm_pass_scrmbl' in df.columns:
            # Create indicator column for when no scrambles
            df['tm_pass_no_scrmbl'] = 0

            # Find rows where tm_pass_scrmbl == 0
            mask = df['tm_pass_scrmbl'] == 0

            # Set indicator to 1 for these rows
            df.loc[mask, 'tm_pass_no_scrmbl'] = 1

            zero_count = mask.sum()
            if zero_count > 0:
                df.loc[mask, 'tm_pass_yds_scrmbl'] = -999
                logger.info(f"Imputed {zero_count} values in tm_pass_yds_scrmbl with -999 (where tm_pass_scrmbl == 0)")
                logger.info(f"Created indicator column tm_pass_no_scrmbl (set to 1 for {zero_count} rows)")

        # Rule Group 5: tm_pass_td_pct
        # Touchdown percentage undefined when no touchdowns occurred
        if 'tm_pass_td_pct' in df.columns:
            # Create indicator column for when no touchdowns
            df['tm_pass_no_td'] = 0

            # Find rows where tm_pass_td is NULL or 0
            mask = (df['tm_pass_td'].isnull()) | (df['tm_pass_td'] == 0)

            # Set indicator to 1 for these rows
            df.loc[mask, 'tm_pass_no_td'] = 1

            # Impute tm_pass_td_pct with -999 where tm_pass_td is NULL or 0
            null_or_zero_count = mask.sum()
            if null_or_zero_count > 0:
                df.loc[mask, 'tm_pass_td_pct'] = -999
                logger.info(f"Imputed {null_or_zero_count} NULL/zero values in tm_pass_td_pct with -999")
                logger.info(f"Created indicator column tm_pass_no_td (set to 1 for {null_or_zero_count} rows)")

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
    handler = TmPassNullHandler(raw_dir=raw_dir, output_dir=clean_dir)

    # Log filter information
    if seasons:
        logger.info(f"Processing seasons: {seasons}")
    else:
        logger.info("Processing all seasons")

    if weeks:
        logger.info(f"Processing weeks: {weeks}")
    else:
        logger.info("Processing all weeks")

    # Process tm_pass table with partitioning
    tm_pass_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw\tm_szn\tm_pass"
    handler.process_partitioned_table(
        table_name='tm_pass',
        table_path=tm_pass_path,
        category='tm_szn',
        handler_func=handler.handle_tm_pass_nulls,
        seasons=seasons,
        weeks=weeks
    )

    # Print final summary
    handler.print_final_summary()

if __name__ == "__main__":
    main()
