import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path to import null_utils
sys.path.append(str(Path(__file__).parent.parent))
from null_utils import BaseNullHandler, logger, parse_args, parse_season_filter, parse_week_filter

class PlyrPassNullHandler(BaseNullHandler):
    def __init__(self, raw_dir: str, output_dir: str = None):
        super().__init__(raw_dir, output_dir)

    def handle_plyr_pass_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values for plyr_pass table"""
        logger.info("Starting plyr_pass null value handling...")

        # Create a copy to avoid modifying original
        df = df.copy()

        # Initialize indicator columns
        df['plyr_pass_did_not_qualify'] = 0
        df['plyr_pass_no_att'] = 0
        df['plyr_pass_no_cmp'] = 0
        df['plyr_pass_no_scrmbl'] = 0
        df['plyr_pass_missing_stats'] = 0

        # Handle plyr_pass_did_not_qualify indicator for NULL values
        # When qb_win, qb_loss, qb_tie are all NULL
        qb_cols = ['qb_win', 'qb_loss', 'qb_tie']
        if all(col in df.columns for col in qb_cols):
            # Check if all QB record columns are NULL
            mask_all_null = df[qb_cols].isnull().all(axis=1)

            # Apply imputation for columns that are NULL
            for col in qb_cols:
                mask_null = mask_all_null & df[col].isnull()
                df.loc[mask_null, col] = -999
                if mask_null.sum() > 0:
                    df.loc[mask_null, 'plyr_pass_did_not_qualify'] = 1

            logger.info(f"Applied plyr_pass_did_not_qualify indicator for NULL values to {mask_all_null.sum()} rows")

        # Handle plyr_pass_did_not_qualify indicator for NULL plyr_pass_succ_rt
        # When plyr_pass_att != 0 and plyr_pass_succ_rt is NULL
        if all(col in df.columns for col in ['plyr_pass_att', 'plyr_pass_succ_rt']):
            mask_succ_rt = (df['plyr_pass_att'] != 0) & df['plyr_pass_succ_rt'].isnull()
            df.loc[mask_succ_rt, 'plyr_pass_succ_rt'] = -999
            if mask_succ_rt.sum() > 0:
                df.loc[mask_succ_rt, 'plyr_pass_did_not_qualify'] = 1

            logger.info(f"Applied plyr_pass_did_not_qualify indicator for NULL plyr_pass_succ_rt to {mask_succ_rt.sum()} rows")

        # Handle plyr_pass_no_att indicator for NULL values
        # When plyr_pass_att = 0 and various passing columns are NULL
        no_att_cols = [
            'plyr_pass_cmp_pct', 'plyr_pass_td_pct', 'plyr_pass_int_pct', 'plyr_pass_succ_rt',
            'plyr_pass_lng', 'plyr_pass_yds_att', 'plyr_pass_adj_yds_att', 'plyr_pass_yds_cmp',
            'plyr_pass_rtg', 'plyr_pass_adv_yds', 'plyr_pass_iay', 'plyr_pass_iay_att',
            'plyr_pass_cmp_ay', 'plyr_pass_cay_cmp', 'plyr_pass_cay_att', 'plyr_pass_yac',
            'plyr_pass_yac_cmp', 'plyr_rpo_play', 'plyr_rpo_yds', 'plyr_rpo_pass_att',
            'plyr_rpo_pass_yds', 'plyr_rpo_rush_att', 'plyr_rpo_rush_yds', 'plyr_pa_att',
            'plyr_pa_yds', 'plyr_pass_pkt_time', 'plyr_pass_bltz', 'plyr_pass_hrry',
            'plyr_pass_hit', 'plyr_pass_prss', 'plyr_pass_prss_pct', 'plyr_pass_scrmbl',
            'plyr_pass_yds_scrmbl', 'plyr_qbr'
        ]

        if 'plyr_pass_att' in df.columns:
            mask_no_att = df['plyr_pass_att'] == 0
            any_imputed = False

            for col in no_att_cols:
                if col in df.columns:
                    mask_null = mask_no_att & df[col].isnull()
                    df.loc[mask_null, col] = -999
                    if mask_null.sum() > 0:
                        df.loc[mask_null, 'plyr_pass_no_att'] = 1
                        any_imputed = True

            logger.info(f"Applied plyr_pass_no_att indicator for NULL values to {mask_no_att.sum()} rows with no attempts")

        # Handle plyr_pass_no_cmp indicator for NULL values
        # When plyr_pass_cmp = 0 or plyr_pass_yds < 0 and certain columns are NULL
        no_cmp_cols = ['plyr_pass_lng', 'plyr_pass_cay_cmp']

        if 'plyr_pass_cmp' in df.columns:
            mask_no_cmp = (df['plyr_pass_cmp'] == 0)
            if 'plyr_pass_yds' in df.columns:
                mask_no_cmp = mask_no_cmp | (df['plyr_pass_yds'] < 0)

            for col in no_cmp_cols:
                if col in df.columns:
                    mask_null = mask_no_cmp & df[col].isnull()
                    df.loc[mask_null, col] = -999
                    if mask_null.sum() > 0:
                        df.loc[mask_null, 'plyr_pass_no_cmp'] = 1

            logger.info(f"Applied plyr_pass_no_cmp indicator for NULL values to {mask_no_cmp.sum()} rows with no completions")

        # Handle plyr_pass_no_cmp indicator for NULL completion-dependent columns
        # When plyr_pass_cmp = 0 and plyr_pass_att != 0 and affected columns are NULL
        cmp_dependent_cols = ['plyr_pass_yds_cmp', 'plyr_pass_yac_cmp']

        if all(col in df.columns for col in ['plyr_pass_cmp', 'plyr_pass_att']):
            mask_no_cmp_att = (df['plyr_pass_cmp'] == 0) & (df['plyr_pass_att'] != 0)

            for col in cmp_dependent_cols:
                if col in df.columns:
                    mask_null = mask_no_cmp_att & df[col].isnull()
                    df.loc[mask_null, col] = -999
                    if mask_null.sum() > 0:
                        df.loc[mask_null, 'plyr_pass_no_cmp'] = 1

            logger.info(f"Applied plyr_pass_no_cmp indicator for completion-dependent NULL values to {mask_no_cmp_att.sum()} rows")

        # Handle plyr_pass_no_scrmbl indicator for NULL values
        # When plyr_pass_scrmbl = 0 and plyr_pass_yds_scrmbl is NULL
        if all(col in df.columns for col in ['plyr_pass_scrmbl', 'plyr_pass_yds_scrmbl']):
            mask_no_scrmbl = (df['plyr_pass_scrmbl'] == 0)
            mask_null = mask_no_scrmbl & df['plyr_pass_yds_scrmbl'].isnull()
            df.loc[mask_null, 'plyr_pass_yds_scrmbl'] = -999
            if mask_null.sum() > 0:
                df.loc[mask_null, 'plyr_pass_no_scrmbl'] = 1

            logger.info(f"Applied plyr_pass_no_scrmbl indicator for NULL values to {mask_null.sum()} rows with no scrambles")

        # Handle plyr_pass_missing_stats indicator for any remaining NULL values (catch-all)
        # Exclude the indicator columns we just created from the null check
        indicator_cols = ['plyr_pass_did_not_qualify', 'plyr_pass_no_att', 'plyr_pass_no_cmp',
                         'plyr_pass_no_scrmbl', 'plyr_pass_missing_stats']
        data_cols = [col for col in df.columns if col not in indicator_cols]

        # Find any remaining NULL values in data columns
        null_mask = df[data_cols].isnull()
        rows_with_nulls = null_mask.any(axis=1)

        # Impute remaining NULL values with -1 and set indicator
        for col in data_cols:
            if col in df.columns:
                mask_null = df[col].isnull()
                if mask_null.sum() > 0:
                    df.loc[mask_null, col] = -999
                    df.loc[mask_null, 'plyr_pass_missing_stats'] = 1

        logger.info(f"Applied plyr_pass_missing_stats indicator for remaining NULL values to {rows_with_nulls.sum()} rows")

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
    handler = PlyrPassNullHandler(raw_dir=raw_dir, output_dir=clean_dir)

    # Log filter information
    if seasons:
        logger.info(f"Processing seasons: {seasons}")
    else:
        logger.info("Processing all seasons")

    if weeks:
        logger.info(f"Processing weeks: {weeks}")
    else:
        logger.info("Processing all weeks")

    # Process plyr_pass table with partitioning
    plyr_pass_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw\plyr_szn\plyr_pass"
    handler.process_partitioned_table(
        table_name='plyr_pass',
        table_path=plyr_pass_path,
        category='plyr_szn',
        handler_func=handler.handle_plyr_pass_nulls,
        seasons=seasons,
        weeks=weeks
    )

    # Print final summary
    handler.print_final_summary()

if __name__ == "__main__":
    main()