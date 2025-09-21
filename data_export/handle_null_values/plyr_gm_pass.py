import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path to import null_utils
sys.path.append(str(Path(__file__).parent.parent))
from null_utils import BaseNullHandler, logger

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
                    df.loc[mask_null, col] = -1
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

        # Handle zero value replacements

        # plyr_gm_pass_no_first_dwn: Replace 0 with -1 in plyr_gm_pass_first_dwn_pct
        if 'plyr_gm_pass_first_dwn' in df.columns and 'plyr_gm_pass_first_dwn_pct' in df.columns:
            mask_zero = (df['plyr_gm_pass_first_dwn'] == 0) & (df['plyr_gm_pass_first_dwn_pct'] == 0)
            df.loc[mask_zero, 'plyr_gm_pass_first_dwn_pct'] = -1
            df.loc[mask_zero, 'plyr_gm_pass_no_first_dwn'] = 1
            logger.info(f"Replaced 0 values in plyr_gm_pass_first_dwn_pct for {mask_zero.sum()} rows (no first downs)")

        # Additional handling for plyr_gm_pass_first_dwn NULL values when there are completions
        if 'plyr_gm_pass_cmp' in df.columns and 'plyr_gm_pass_first_dwn' in df.columns and 'plyr_gm_pass_first_dwn_pct' in df.columns:
            # Condition: plyr_gm_pass_cmp != 0 and plyr_gm_pass_first_dwn is NULL
            mask_has_cmp_no_fd = (df['plyr_gm_pass_cmp'] != 0) & df['plyr_gm_pass_first_dwn'].isnull()

            # Replace NULL values in plyr_gm_pass_first_dwn with 0 (no indicator for this)
            df.loc[mask_has_cmp_no_fd, 'plyr_gm_pass_first_dwn'] = 0

            # Impute NULL values in plyr_gm_pass_first_dwn_pct with -1 and set indicator
            mask_fd_pct_null = mask_has_cmp_no_fd & df['plyr_gm_pass_first_dwn_pct'].isnull()
            df.loc[mask_fd_pct_null, 'plyr_gm_pass_first_dwn_pct'] = -1
            df.loc[mask_fd_pct_null, 'plyr_gm_pass_no_first_dwn'] = 1

            logger.info(f"Handled NULL first downs for {mask_has_cmp_no_fd.sum()} rows with completions but no first downs")
            logger.info(f"Set plyr_gm_pass_no_first_dwn indicator for {mask_fd_pct_null.sum()} rows with NULL first_dwn_pct")

        # plyr_gm_pass_no_cmp: Replace 0 with -1 in plyr_gm_pass_cmp_pct
        if 'plyr_gm_pass_cmp' in df.columns and 'plyr_gm_pass_cmp_pct' in df.columns:
            mask_zero = (df['plyr_gm_pass_cmp'] == 0) & (df['plyr_gm_pass_cmp_pct'] == 0)
            df.loc[mask_zero, 'plyr_gm_pass_cmp_pct'] = -1
            df.loc[mask_zero, 'plyr_gm_pass_no_cmp'] = 1
            logger.info(f"Replaced 0 values in plyr_gm_pass_cmp_pct for {mask_zero.sum()} rows (no completions)")

        # plyr_gm_pass_no_td: Replace 0 with -1 in plyr_gm_pass_td_pct
        if 'plyr_gm_pass_td' in df.columns and 'plyr_gm_pass_td_pct' in df.columns:
            mask_zero = (df['plyr_gm_pass_td'] == 0) & (df['plyr_gm_pass_td_pct'] == 0)
            df.loc[mask_zero, 'plyr_gm_pass_td_pct'] = -1
            df.loc[mask_zero, 'plyr_gm_pass_no_td'] = 1
            logger.info(f"Replaced 0 values in plyr_gm_pass_td_pct for {mask_zero.sum()} rows (no touchdowns)")

        # plyr_gm_pass_no_int: Replace 0 with -1 in plyr_gm_pass_int_pct
        if 'plyr_gm_pass_int' in df.columns and 'plyr_gm_pass_int_pct' in df.columns:
            mask_zero = (df['plyr_gm_pass_int'] == 0) & (df['plyr_gm_pass_int_pct'] == 0)
            df.loc[mask_zero, 'plyr_gm_pass_int_pct'] = -1
            df.loc[mask_zero, 'plyr_gm_pass_no_int'] = 1
            logger.info(f"Replaced 0 values in plyr_gm_pass_int_pct for {mask_zero.sum()} rows (no interceptions)")

        # plyr_gm_pass_no_prss: Replace 0 with -1 in plyr_gm_pass_prss_pct
        # Note: The XML seems to have a typo here - it references plyr_gm_pass_int columns but the indicator is plyr_gm_pass_no_prss
        # I'll implement based on the indicator name which suggests it should be based on plyr_gm_pass_prss
        if 'plyr_gm_pass_prss' in df.columns and 'plyr_gm_pass_prss_pct' in df.columns:
            mask_zero = (df['plyr_gm_pass_prss'] == 0) & (df['plyr_gm_pass_prss_pct'] == 0)
            df.loc[mask_zero, 'plyr_gm_pass_prss_pct'] = -1
            df.loc[mask_zero, 'plyr_gm_pass_no_prss'] = 1
            logger.info(f"Replaced 0 values in plyr_gm_pass_prss_pct for {mask_zero.sum()} rows (no pressure)")

        # plyr_gm_pass_no_sk: Replace 0 with -1 in plyr_gm_pass_sk_pct
        if 'plyr_gm_pass_sk' in df.columns and 'plyr_gm_pass_sk_pct' in df.columns:
            mask_zero = (df['plyr_gm_pass_sk'] == 0) & (df['plyr_gm_pass_sk_pct'] == 0)
            df.loc[mask_zero, 'plyr_gm_pass_sk_pct'] = -1
            df.loc[mask_zero, 'plyr_gm_pass_no_sk'] = 1
            logger.info(f"Replaced 0 values in plyr_gm_pass_sk_pct for {mask_zero.sum()} rows (no sacks)")

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

    def process_plyr_gm_pass_table(self, table_path: str) -> None:
        """Process the plyr_gm_pass table for null value handling"""
        self.process_table('plyr_gm_pass', table_path, self.handle_plyr_gm_pass_nulls)

def main():
    # Initialize handler
    raw_dir = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw"
    handler = PlyrGmPassNullHandler(raw_dir=raw_dir)

    # Process plyr_gm_pass table
    plyr_gm_pass_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw\plyr_gm\plyr_gm_pass"
    handler.process_plyr_gm_pass_table(plyr_gm_pass_path)

    # Print final summary
    handler.print_final_summary()

if __name__ == "__main__":
    main()