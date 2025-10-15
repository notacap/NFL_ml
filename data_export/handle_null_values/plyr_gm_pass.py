import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path to import null_utils
sys.path.append(str(Path(__file__).parent.parent))
from null_utils import BaseNullHandler, logger, parse_args, parse_season_filter, parse_week_filter

class PlyrGmPassNullHandler(BaseNullHandler):
    def __init__(self, raw_dir: str, output_dir: str = None):
        super().__init__(raw_dir, output_dir)

    def handle_plyr_gm_pass_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values for plyr_gm_pass table"""
        logger.info("Starting plyr_gm_pass null value handling...")

        # Create a copy to avoid modifying original
        df = df.copy()

        # Initialize indicator columns
        df['plyr_gm_pass_no_cmp'] = 0
        df['plyr_gm_pass_no_scrmbl'] = 0
        df['plyr_gm_pass_no_first_dwn'] = 0
        df['plyr_gm_pass_no_td'] = 0
        df['plyr_gm_pass_no_int'] = 0
        df['plyr_gm_pass_no_prss'] = 0
        df['plyr_gm_pass_no_sk'] = 0

        # Handle plyr_gm_pass_no_cmp indicator for NULL values
        no_cmp_cols = [
            'plyr_gm_pass_cay_cmp', 'plyr_gm_pass_yac_cmp', 'plyr_gm_pass_yds_cmp',
            'plyr_gm_pass_first_dwn', 'plyr_gm_pass_first_dwn_pct'
        ]

        # Check if plyr_gm_pass_cmp = 0 and affected columns are NULL
        if 'plyr_gm_pass_cmp' in df.columns:
            mask_no_cmp = df['plyr_gm_pass_cmp'] == 0
            for col in no_cmp_cols:
                if col in df.columns:
                    mask_null = mask_no_cmp & df[col].isnull()
                    df.loc[mask_null, col] = -999
                    if mask_null.sum() > 0:
                        df.loc[mask_null, 'plyr_gm_pass_no_cmp'] = 1

            logger.info(f"Applied plyr_gm_pass_no_cmp indicator for NULL values to {mask_no_cmp.sum()} rows with no completions")

        # Handle plyr_gm_pass_no_scrmbl indicator
        if 'plyr_gm_pass_scrmbl_tgt' in df.columns and 'plyr_gm_pass_yds_scrmbl' in df.columns:
            mask_no_scrmbl = df['plyr_gm_pass_scrmbl_tgt'] == 0
            mask_null = mask_no_scrmbl & df['plyr_gm_pass_yds_scrmbl'].isnull()
            df.loc[mask_null, 'plyr_gm_pass_yds_scrmbl'] = -1
            df.loc[mask_null, 'plyr_gm_pass_no_scrmbl'] = 1

            logger.info(f"Applied plyr_gm_pass_no_scrmbl indicator to {mask_null.sum()} rows")

        # Additional handling for plyr_gm_pass_first_dwn NULL values when there are completions
        if 'plyr_gm_pass_cmp' in df.columns and 'plyr_gm_pass_first_dwn' in df.columns and 'plyr_gm_pass_first_dwn_pct' in df.columns:
            # Condition: plyr_gm_pass_cmp != 0 and plyr_gm_pass_first_dwn is NULL
            mask_has_cmp_no_fd = (df['plyr_gm_pass_cmp'] != 0) & df['plyr_gm_pass_first_dwn'].isnull()

            # Replace NULL values in plyr_gm_pass_first_dwn with 0 (no indicator for this)
            df.loc[mask_has_cmp_no_fd, 'plyr_gm_pass_first_dwn'] = 0

            # Impute NULL values in plyr_gm_pass_first_dwn_pct with -1 and set indicator
            mask_fd_pct_null = mask_has_cmp_no_fd & df['plyr_gm_pass_first_dwn_pct'].isnull()
            df.loc[mask_fd_pct_null, 'plyr_gm_pass_first_dwn_pct'] = -999
            df.loc[mask_fd_pct_null, 'plyr_gm_pass_no_first_dwn'] = 1

            logger.info(f"Handled NULL first downs for {mask_has_cmp_no_fd.sum()} rows with completions but no first downs")
            logger.info(f"Set plyr_gm_pass_no_first_dwn indicator for {mask_fd_pct_null.sum()} rows with NULL first_dwn_pct")

        # # Final catch-all: Impute any remaining NULL values with -1 and create plyr_gm_pass_missing_stats indicator
        # # Get all columns except the indicator columns we created
        # indicator_cols = [
        #     'plyr_gm_pass_no_cmp', 'plyr_gm_pass_no_scrmbl', 'plyr_gm_pass_no_first_dwn',
        #     'plyr_gm_pass_no_td', 'plyr_gm_pass_no_int', 'plyr_gm_pass_no_prss', 'plyr_gm_pass_no_sk'
        # ]

        # # Add missing_stats indicator if we need to catch remaining nulls
        # df['plyr_gm_pass_missing_stats'] = 0
        # indicator_cols.append('plyr_gm_pass_missing_stats')

        # data_cols = [col for col in df.columns if col not in indicator_cols]

        # # Find rows with any remaining NULL values in data columns
        # mask_any_nulls = df[data_cols].isnull().any(axis=1)

        # if mask_any_nulls.sum() > 0:
        #     # Impute all remaining NULL values with -1
        #     for col in data_cols:
        #         mask_col_null = df[col].isnull()
        #         if mask_col_null.sum() > 0:
        #             df.loc[mask_col_null, col] = -1

        #     # Set indicator for rows that had any remaining nulls
        #     df.loc[mask_any_nulls, 'plyr_gm_pass_missing_stats'] = 1
        #     logger.info(f"Final catch-all: Imputed remaining NULL values for {mask_any_nulls.sum()} rows using plyr_gm_pass_missing_stats indicator")

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
    handler = PlyrGmPassNullHandler(raw_dir=raw_dir, output_dir=clean_dir)

    # Log filter information
    if seasons:
        logger.info(f"Processing seasons: {seasons}")
    else:
        logger.info("Processing all seasons")

    if weeks:
        logger.info(f"Processing weeks: {weeks}")
    else:
        logger.info("Processing all weeks")

    # Process plyr_gm_pass table with partitioning
    plyr_gm_pass_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw\plyr_gm\plyr_gm_pass"
    handler.process_partitioned_table(
        table_name='plyr_gm_pass',
        table_path=plyr_gm_pass_path,
        category='plyr_gm',
        handler_func=handler.handle_plyr_gm_pass_nulls,
        seasons=seasons,
        weeks=weeks
    )

    # Print final summary
    handler.print_final_summary()

if __name__ == "__main__":
    main()