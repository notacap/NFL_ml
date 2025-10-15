import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path to import null_utils
sys.path.append(str(Path(__file__).parent.parent))
from null_utils import BaseNullHandler, logger, parse_args, parse_season_filter, parse_week_filter

class PlyrDefNullHandler(BaseNullHandler):
    def __init__(self, raw_dir: str, output_dir: str = None):
        super().__init__(raw_dir, output_dir)

    def handle_plyr_def_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values for plyr_def table"""
        logger.info("Starting plyr_def null value handling...")

        # Create a copy to avoid modifying original
        df = df.copy()

        # Initialize indicator columns
        df['plyr_def_no_targets'] = 0
        df['plyr_def_missing_stats'] = 0
        df['plyr_def_no_mtkl'] = 0
        df['plyr_def_no_completions'] = 0

        # Handle plyr_def_no_targets indicator
        no_target_cols = [
            'plyr_def_cmp_pct', 'plyr_def_pass_yds_cmp',
            'plyr_def_pass_yds_tgt', 'plyr_def_pass_rtg', 'plyr_def_adot'
        ]

        # Check if plyr_def_tgt = 0
        mask_no_targets = (df['plyr_def_tgt'] == 0)
        for col in no_target_cols:
            if col in df.columns:
                df.loc[mask_no_targets & df[col].isnull(), col] = -999
        df.loc[mask_no_targets, 'plyr_def_no_targets'] = 1

        logger.info(f"Applied plyr_def_no_targets indicator to {mask_no_targets.sum()} rows")

        # Handle plyr_def_missing_stats indicator (first set - based on plyr_def_tgt NULL)
        missing_stats_cols_1 = [
            'plyr_def_tgt', 'plyr_def_cmp', 'plyr_def_cmp_pct', 'plyr_def_pass_yds',
            'plyr_def_pass_yds_cmp', 'plyr_def_pass_yds_tgt', 'plyr_def_pass_td',
            'plyr_def_pass_rtg', 'plyr_def_adot', 'plyr_def_ay', 'plyr_def_yac',
            'plyr_def_bltz', 'plyr_def_hrry', 'plyr_def_qbkd', 'plyr_def_sk',
            'plyr_def_prss', 'plyr_def_comb_tkl', 'plyr_def_mtkl', 'plyr_def_mtkl_pct'
        ]

        mask_missing_tgt = df['plyr_def_tgt'].isnull()
        for col in missing_stats_cols_1:
            if col in df.columns:
                df.loc[mask_missing_tgt & df[col].isnull(), col] = -999
        df.loc[mask_missing_tgt, 'plyr_def_missing_stats'] = 1

        logger.info(f"Applied plyr_def_missing_stats indicator (tgt NULL) to {mask_missing_tgt.sum()} rows")

        # Handle plyr_def_missing_stats indicator (second set - based on plyr_def_int NULL)
        missing_stats_cols_2 = [
            'plyr_def_int', 'plyr_def_int_yds', 'plyr_def_int_td', 'plyr_def_int_lng',
            'plyr_pass_def', 'plyr_def_force_fmbl', 'plyr_def_fmbl_rec',
            'plyr_def_fmbl_rec_yds', 'plyr_def_fmbl_rec_td', 'plyr_def_solo_tkl',
            'plyr_def_tkl_assist', 'plyr_def_tfl', 'plyr_def_qb_hit', 'plyr_def_sfty'
        ]

        mask_missing_int = df['plyr_def_int'].isnull()
        for col in missing_stats_cols_2:
            if col in df.columns:
                df.loc[mask_missing_int & df[col].isnull(), col] = -999
        df.loc[mask_missing_int, 'plyr_def_missing_stats'] = 1

        logger.info(f"Applied plyr_def_missing_stats indicator (int NULL) to {mask_missing_int.sum()} rows")

        # Handle plyr_def_no_mtkl indicator
        no_mtkl_cols = ['plyr_def_mtkl_pct']

        mask_no_mtkl = (df['plyr_def_mtkl'] == 0) if 'plyr_def_mtkl' in df.columns else pd.Series([False] * len(df))
        for col in no_mtkl_cols:
            if col in df.columns:
                df.loc[mask_no_mtkl & df[col].isnull(), col] = -999
        df.loc[mask_no_mtkl, 'plyr_def_no_mtkl'] = 1

        logger.info(f"Applied plyr_def_no_mtkl indicator to {mask_no_mtkl.sum()} rows")

        # Handle plyr_def_no_completions indicator
        mask_no_completions = (df['plyr_def_cmp'] == 0) & df['plyr_def_pass_yds_cmp'].isnull()
        df.loc[mask_no_completions, 'plyr_def_pass_yds_cmp'] = -999
        df.loc[mask_no_completions, 'plyr_def_no_completions'] = 1

        logger.info(f"Applied plyr_def_no_completions indicator to {mask_no_completions.sum()} rows")

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
    handler = PlyrDefNullHandler(raw_dir=raw_dir, output_dir=clean_dir)

    # Log filter information
    if seasons:
        logger.info(f"Processing seasons: {seasons}")
    else:
        logger.info("Processing all seasons")

    if weeks:
        logger.info(f"Processing weeks: {weeks}")
    else:
        logger.info("Processing all weeks")

    # Process plyr_def table with partitioning
    plyr_def_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw\plyr_szn\plyr_def"
    handler.process_partitioned_table(
        table_name='plyr_def',
        table_path=plyr_def_path,
        category='plyr_szn',
        handler_func=handler.handle_plyr_def_nulls,
        seasons=seasons,
        weeks=weeks
    )

    # Print final summary
    handler.print_final_summary()

if __name__ == "__main__":
    main()