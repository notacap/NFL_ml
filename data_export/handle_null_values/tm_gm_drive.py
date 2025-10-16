import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path to import null_utils
sys.path.append(str(Path(__file__).parent.parent))
from null_utils import BaseNullHandler, logger, parse_args, parse_season_filter, parse_week_filter

class TmGmDriveNullHandler(BaseNullHandler):
    """Handler for tm_gm_drive table null value imputation."""

    def __init__(self, raw_dir: str, output_dir: str = None):
        """Initialize the handler.

        Args:
            raw_dir: Path to raw Parquet directory
            output_dir: Path to clean output directory (optional)
        """
        super().__init__(raw_dir, output_dir)

    def handle_tm_gm_drive_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values for tm_gm_drive table.

        This method imputes NULL values in tm_gm_dr_strt_fld_pos with -999
        and creates an indicator variable tm_gm_drive_missing_stats to flag
        rows where this imputation occurred.

        Args:
            df: DataFrame with raw tm_gm_drive data

        Returns:
            DataFrame with null values handled
        """
        logger.info("Starting tm_gm_drive null value handling...")

        # Create a copy to avoid modifying original
        df = df.copy()

        # Initialize indicator column with default value 0
        df['tm_gm_drive_missing_stats'] = 0

        # Handle NULL values in tm_gm_dr_strt_fld_pos
        if 'tm_gm_dr_strt_fld_pos' in df.columns:
            # Create mask for NULL values in tm_gm_dr_strt_fld_pos
            mask_null = df['tm_gm_dr_strt_fld_pos'].isnull()

            # Apply imputation: replace NULL with -999
            df.loc[mask_null, 'tm_gm_dr_strt_fld_pos'] = -999

            # Set indicator flag for imputed rows
            df.loc[mask_null, 'tm_gm_drive_missing_stats'] = 1

            # Log number of rows affected
            num_imputed = mask_null.sum()
            logger.info(f"Applied tm_gm_drive_missing_stats indicator to {num_imputed} rows")

        return df

def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_args()

    # Parse season and week filters
    seasons = parse_season_filter(args.season) if args.season else None
    weeks = parse_week_filter(args.week) if args.week else None

    # Initialize handler with output directory for clean parquet files
    raw_dir = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw"
    clean_dir = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\clean"
    handler = TmGmDriveNullHandler(raw_dir=raw_dir, output_dir=clean_dir)

    # Log filter information
    if seasons:
        logger.info(f"Processing seasons: {seasons}")
    else:
        logger.info("Processing all seasons")

    if weeks:
        logger.info(f"Processing weeks: {weeks}")
    else:
        logger.info("Processing all weeks")

    # Process tm_gm_drive table with partitioning
    tm_gm_drive_path = r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw\tm_gm\tm_gm_drive"
    handler.process_partitioned_table(
        table_name='tm_gm_drive',
        table_path=tm_gm_drive_path,
        category='tm_gm',
        handler_func=handler.handle_tm_gm_drive_nulls,
        seasons=seasons,
        weeks=weeks
    )

    # Print final summary
    handler.print_final_summary()

if __name__ == "__main__":
    main()
