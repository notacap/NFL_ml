import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path to import null_utils
sys.path.append(str(Path(__file__).parent.parent))
from null_utils import BaseNullHandler, logger

class PlyrGmDefNullHandler(BaseNullHandler):
    def __init__(self, raw_dir: str, output_dir: str = None):
        super().__init__(raw_dir, output_dir)

    def handle_plyr_gm_def_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values for plyr_gm_def table"""
        logger.info("Starting plyr_gm_def null value handling...")

        # Create a copy to avoid modifying original
        df = df.copy()

        # Initialize indicator columns
        df['plyr_gm_def_no_targets'] = 0
        df['plyr_gm_def_missing_stats'] = 0
        df['plyr_gm_def_no_mtkl'] = 0
        df['plyr_gm_def_no_cmp'] = 0
        df['plyr_gm_def_no_pass_yds'] = 0

        # First indicator: plyr_gm_def_no_targets when plyr_gm_def_tgt = 0
        no_target_cols = [
            'plyr_gm_def_cmp_pct', 'plyr_gm_def_pass_yds', 'plyr_gm_def_pass_yds_cmp',
            'plyr_gm_def_pass_yds_tgt', 'plyr_gm_def_pass_td', 'plyr_gm_def_pass_rtg',
            'plyr_gm_def_adot', 'plyr_gm_def_ay', 'plyr_gm_def_yac'
        ]

        # Check if plyr_gm_def_tgt = 0
        mask_no_targets = (df['plyr_gm_def_tgt'] == 0) if 'plyr_gm_def_tgt' in df.columns else pd.Series([False] * len(df))
        for col in no_target_cols:
            if col in df.columns:
                df.loc[mask_no_targets & df[col].isnull(), col] = -1

        # Replace 0 values in plyr_gm_def_pass_yds_tgt when plyr_gm_def_tgt = 0
        if 'plyr_gm_def_pass_yds_tgt' in df.columns:
            mask_pass_yds_tgt_zero = mask_no_targets & (df['plyr_gm_def_pass_yds_tgt'] == 0)
            df.loc[mask_pass_yds_tgt_zero, 'plyr_gm_def_pass_yds_tgt'] = -1
            logger.info(f"Replaced 0 values in plyr_gm_def_pass_yds_tgt for {mask_pass_yds_tgt_zero.sum()} rows (no targets)")

        df.loc[mask_no_targets, 'plyr_gm_def_no_targets'] = 1

        logger.info(f"Applied plyr_gm_def_no_targets indicator (tgt=0) to {mask_no_targets.sum()} rows")

        # Second indicator: plyr_gm_def_missing_stats when plyr_gm_def_tgt is NULL
        missing_stats_cols = [
            'plyr_gm_def_tgt', 'plyr_gm_def_cmp', 'plyr_gm_def_cmp_pct', 'plyr_gm_def_pass_yds',
            'plyr_gm_def_pass_yds_cmp', 'plyr_gm_def_pass_yds_tgt', 'plyr_gm_def_pass_td',
            'plyr_gm_def_pass_rtg', 'plyr_gm_def_adot', 'plyr_gm_def_ay', 'plyr_gm_def_yac',
            'plyr_gm_def_bltz', 'plyr_gm_def_hrry', 'plyr_gm_def_qbkd', 'plyr_gm_def_prss',
            'plyr_gm_def_mtkl', 'plyr_gm_def_mtkl_pct'
        ]

        mask_missing_tgt = df['plyr_gm_def_tgt'].isnull() if 'plyr_gm_def_tgt' in df.columns else pd.Series([False] * len(df))
        for col in missing_stats_cols:
            if col in df.columns:
                df.loc[mask_missing_tgt & df[col].isnull(), col] = -1
        df.loc[mask_missing_tgt, 'plyr_gm_def_missing_stats'] = 1

        logger.info(f"Applied plyr_gm_def_missing_stats indicator (tgt NULL) to {mask_missing_tgt.sum()} rows")

        # Third exception case: Handle null values and replace 0 in plyr_gm_def_cmp_pct when plyr_gm_def_cmp = 0
        if 'plyr_gm_def_cmp' in df.columns:
            # Identify rows where plyr_gm_def_cmp = 0
            mask_no_completions = df['plyr_gm_def_cmp'] == 0



            # Handle null values for specified columns when plyr_gm_def_cmp = 0
            no_cmp_null_cols = [
                'plyr_gm_def_pass_yds', 'plyr_gm_def_pass_yds_cmp', 'plyr_gm_def_ay',
                'plyr_gm_def_yac', 'plyr_gm_def_cmp', 'plyr_gm_def_pass_td'
            ]

            for col in no_cmp_null_cols:
                if col in df.columns:
                    mask_null_impute = mask_no_completions & df[col].isnull()
                    df.loc[mask_null_impute, col] = -1

            # Set indicator for rows with no completions
            df.loc[mask_no_completions, 'plyr_gm_def_no_cmp'] = 1
            logger.info(f"Applied plyr_gm_def_no_cmp indicator and null imputation for {mask_no_completions.sum()} rows (no completions)")

        # Fourth exception case: Handle plyr_gm_def_mtkl_pct when plyr_gm_def_mtkl = 0
        if 'plyr_gm_def_mtkl' in df.columns and 'plyr_gm_def_mtkl_pct' in df.columns:
            # Impute null values in plyr_gm_def_mtkl_pct when plyr_gm_def_mtkl = 0 and plyr_gm_def_mtkl_pct is NULL
            mask_no_mtkl_null = (df['plyr_gm_def_mtkl'] == 0) & df['plyr_gm_def_mtkl_pct'].isnull()
            df.loc[mask_no_mtkl_null, 'plyr_gm_def_mtkl_pct'] = -1
            df.loc[mask_no_mtkl_null, 'plyr_gm_def_no_mtkl'] = 1
            logger.info(f"Imputed null values in plyr_gm_def_mtkl_pct for {mask_no_mtkl_null.sum()} rows (mtkl=0, mtkl_pct=NULL)")

        # Fifth exception case: Handle plyr_gm_def_no_pass_yds when targets/completions exist but passing yards are missing
        if 'plyr_gm_def_tgt' in df.columns and 'plyr_gm_def_cmp' in df.columns and 'plyr_gm_def_pass_yds' in df.columns:
            # Condition: tgt and cmp are not null and not 0, but pass_yds is null
            mask_has_activity = (df['plyr_gm_def_tgt'].notnull() & (df['plyr_gm_def_tgt'] != 0) &
                                df['plyr_gm_def_cmp'].notnull() & (df['plyr_gm_def_cmp'] != 0))
            mask_no_pass_yds = mask_has_activity & df['plyr_gm_def_pass_yds'].isnull()

            # Columns to impute when passing yards data is missing
            no_pass_yds_cols = [
                'plyr_gm_def_pass_yds', 'plyr_gm_def_pass_yds_cmp', 'plyr_gm_def_pass_yds_tgt',
                'plyr_gm_def_ay', 'plyr_gm_def_yac'
            ]

            for col in no_pass_yds_cols:
                if col in df.columns:
                    mask_impute = mask_no_pass_yds & df[col].isnull()
                    df.loc[mask_impute, col] = -1

            # Set indicator for rows with missing passing yards data
            df.loc[mask_no_pass_yds, 'plyr_gm_def_no_pass_yds'] = 1
            logger.info(f"Applied plyr_gm_def_no_pass_yds indicator and null imputation for {mask_no_pass_yds.sum()} rows (has activity but no passing yards)")

        # Sixth exception case: Handle plyr_gm_def_td and plyr_gm_def_adot when targets/completions exist but pass_td is missing
        if 'plyr_gm_def_tgt' in df.columns and 'plyr_gm_def_cmp' in df.columns and 'plyr_gm_def_pass_td' in df.columns:
            # Condition: tgt and cmp are not null and not 0, but pass_td is null
            mask_has_activity_td = (df['plyr_gm_def_tgt'].notnull() & (df['plyr_gm_def_tgt'] != 0) &
                                   df['plyr_gm_def_cmp'].notnull() & (df['plyr_gm_def_cmp'] != 0))
            mask_missing_td_stats = mask_has_activity_td & df['plyr_gm_def_pass_td'].isnull()

            # Columns to impute when pass_td data is missing
            missing_td_cols = ['plyr_gm_def_pass_td', 'plyr_gm_def_adot']

            for col in missing_td_cols:
                if col in df.columns:
                    mask_impute_td = mask_missing_td_stats & df[col].isnull()
                    df.loc[mask_impute_td, col] = -1

            # Set indicator for rows with missing TD stats (reusing existing plyr_gm_def_missing_stats)
            df.loc[mask_missing_td_stats, 'plyr_gm_def_missing_stats'] = 1
            logger.info(f"Applied plyr_gm_def_missing_stats indicator for TD/ADOT null imputation for {mask_missing_td_stats.sum()} rows (has activity but no pass_td)")

        # Final catch-all: Impute any remaining NULL values with -1 and set plyr_gm_def_missing_stats indicator
        # Get all columns except the indicator columns we created
        indicator_cols = ['plyr_gm_def_no_targets', 'plyr_gm_def_missing_stats', 'plyr_gm_def_no_mtkl',
                         'plyr_gm_def_no_cmp', 'plyr_gm_def_no_pass_yds']
        data_cols = [col for col in df.columns if col not in indicator_cols]

        # Find rows with any remaining NULL values in data columns
        mask_any_nulls = df[data_cols].isnull().any(axis=1)

        if mask_any_nulls.sum() > 0:
            # Impute all remaining NULL values with -1
            for col in data_cols:
                mask_col_null = df[col].isnull()
                if mask_col_null.sum() > 0:
                    df.loc[mask_col_null, col] = -1

            # Set indicator for rows that had any remaining nulls
            df.loc[mask_any_nulls, 'plyr_gm_def_missing_stats'] = 1
            logger.info(f"Final catch-all: Imputed remaining NULL values for {mask_any_nulls.sum()} rows using plyr_gm_def_missing_stats indicator")

        return df

    def process_plyr_gm_def_table(self, table_path: str) -> None:
        """Process the plyr_gm_def table for null value handling"""
        self.process_table('plyr_gm_def', table_path, self.handle_plyr_gm_def_nulls)

def main():
    # Initialize handler
    raw_dir = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw"
    handler = PlyrGmDefNullHandler(raw_dir=raw_dir)

    # Process plyr_gm_def table
    plyr_gm_def_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw\plyr_gm\plyr_gm_def"
    handler.process_plyr_gm_def_table(plyr_gm_def_path)

    # Print final summary
    handler.print_final_summary()

if __name__ == "__main__":
    main()