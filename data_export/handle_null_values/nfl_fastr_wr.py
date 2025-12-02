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

        # Initialize indicator columns
        df['nfl_fastr_missing_nextgen'] = 0
        df['nfl_fastr_missing_fastr'] = 0
        df['nfl_fastr_wr_no_yac'] = 0

        # =====================================================================
        # 1) Handle missing NextGen stats (all columns must be null to impute)
        # =====================================================================
        nextgen_cols = [
            'plyr_gm_rec_avg_cushion',
            'plyr_gm_rec_avg_separation',
            'plyr_gm_rec_avg_yac',
            'plyr_gm_rec_avg_expected_yac',
            'plyr_gm_rec_avg_yac_above_expectation',
            'plyr_gm_rec_pct_share_of_intended_ay'
        ]

        # Check which nextgen columns exist in dataframe
        existing_nextgen_cols = [col for col in nextgen_cols if col in df.columns]

        if existing_nextgen_cols:
            # Create mask where ALL nextgen columns are null
            mask_all_nextgen_null = df[existing_nextgen_cols].isnull().all(axis=1)

            # Impute -999 for all nextgen columns where mask is True
            for col in existing_nextgen_cols:
                df.loc[mask_all_nextgen_null, col] = -999

            # Set indicator flag
            df.loc[mask_all_nextgen_null, 'nfl_fastr_missing_nextgen'] = 1

            rows_affected_nextgen = mask_all_nextgen_null.sum()
            logger.info(f"Applied nfl_fastr_missing_nextgen indicator to {rows_affected_nextgen} rows with ALL NextGen columns NULL")

        # =====================================================================
        # 2) Handle missing FastR stats (all columns must be null to impute)
        # =====================================================================
        fastr_cols = [
            'plyr_gm_rec_tgt_share',
            'plyr_gm_rec_epa',
            'plyr_gm_rec_ay_share',
            'plyr_gm_rec_wopr',
            'plyr_gm_rec_racr'
        ]

        # Check which fastr columns exist in dataframe
        existing_fastr_cols = [col for col in fastr_cols if col in df.columns]

        if existing_fastr_cols:
            # Create mask where ALL fastr columns are null
            mask_all_fastr_null = df[existing_fastr_cols].isnull().all(axis=1)

            # Impute -999 for all fastr columns where mask is True
            for col in existing_fastr_cols:
                df.loc[mask_all_fastr_null, col] = -999

            # Set indicator flag
            df.loc[mask_all_fastr_null, 'nfl_fastr_missing_fastr'] = 1

            rows_affected_fastr = mask_all_fastr_null.sum()
            logger.info(f"Applied nfl_fastr_missing_fastr indicator to {rows_affected_fastr} rows with ALL FastR columns NULL")

        # =====================================================================
        # 3) Handle remaining YAC nulls (any column null triggers imputation)
        # =====================================================================
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

        # =====================================================================
        # 4) Handle remaining EPA/RACR nulls (did not qualify)
        # =====================================================================
        df['nfl_fastr_did_not_qualify'] = 0

        dnq_cols = [
            'plyr_gm_rec_epa',
            'plyr_gm_rec_racr'
        ]

        # Check which dnq columns exist in dataframe
        existing_dnq_cols = [col for col in dnq_cols if col in df.columns]

        if existing_dnq_cols:
            # Create mask where any of the dnq columns are null
            mask_dnq_null = df[existing_dnq_cols].isnull().any(axis=1)

            # Impute -999 for all dnq columns where mask is True
            for col in existing_dnq_cols:
                df.loc[mask_dnq_null & df[col].isnull(), col] = -999

            # Set indicator flag
            df.loc[mask_dnq_null, 'nfl_fastr_did_not_qualify'] = 1

            rows_affected_dnq = mask_dnq_null.sum()
            logger.info(f"Applied nfl_fastr_did_not_qualify indicator to {rows_affected_dnq} rows with NULL EPA/RACR values")

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
