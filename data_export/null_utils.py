import pandas as pd
import pyarrow.parquet as pq
import os
import logging
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime

# Audit configuration - set to True to export CSV audit files
AUDIT = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseNullHandler:
    def __init__(self, raw_dir: str, output_dir: str = None):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir) if output_dir else None
        self.null_columns_summary = {}

    def read_partitioned_table(self, table_path: str) -> pd.DataFrame:
        """Read a partitioned parquet table into a DataFrame"""
        try:
            df = pd.read_parquet(table_path)
            logger.info(f"Successfully read table from {table_path}")
            logger.info(f"Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error reading table from {table_path}: {e}")
            raise

    def get_null_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of columns with at least one NULL value"""
        null_cols = df.columns[df.isnull().any()].tolist()
        return null_cols

    def export_null_rows_to_csv(self, df: pd.DataFrame, table_name: str, null_cols_after: List[str]) -> None:
        """Export rows with remaining nulls to CSV"""
        if null_cols_after:
            rows_with_nulls = df[df[null_cols_after].isnull().any(axis=1)]
            if len(rows_with_nulls) > 0:
                csv_dir = Path(r"C:\Users\nocap\Desktop\code\NFL_ml\data_export\null_values")
                csv_dir.mkdir(parents=True, exist_ok=True)
                csv_filename = f"{table_name}_rows_with_nulls.csv"
                csv_path = csv_dir / csv_filename
                rows_with_nulls.to_csv(csv_path, index=False)
                logger.info(f"\nExported {len(rows_with_nulls)} rows with NULL values to {csv_filename}")
                logger.info(f"  Full path: {csv_path.absolute()}")

    def export_audit_csv(self, df: pd.DataFrame, table_name: str) -> None:
        """Export processed dataframe to audit CSV if AUDIT is True"""
        if AUDIT:
            # Create audit directory structure
            audit_base_dir = Path(r"C:\Users\nocap\Desktop\code\NFL_ml\data_export\audit")
            table_dir = audit_base_dir / table_name
            table_dir.mkdir(parents=True, exist_ok=True)

            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"{table_name}_{timestamp}.csv"
            csv_path = table_dir / csv_filename

            # Export to CSV
            df.to_csv(csv_path, index=False)
            logger.info(f"\n[AUDIT] Exported processed data to audit CSV:")
            logger.info(f"  Filename: {csv_filename}")
            logger.info(f"  Full path: {csv_path.absolute()}")
            logger.info(f"  Shape: {df.shape}")

    def process_table(self, table_name: str, table_path: str, handler_func) -> None:
        """Process a single table for null value handling"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing table: {table_name}")
        logger.info(f"Path: {table_path}")
        logger.info(f"{'='*60}")

        # Read the table
        df = self.read_partitioned_table(table_path)

        # Get columns with nulls before processing
        null_cols_before = self.get_null_columns(df)
        logger.info(f"Columns with NULL values BEFORE imputation: {len(null_cols_before)}")
        if null_cols_before:
            logger.info(f"  {null_cols_before}")

        # Apply table-specific null handling
        df = handler_func(df)

        # Get columns with nulls after processing
        null_cols_after = self.get_null_columns(df)
        logger.info(f"\nColumns with NULL values AFTER imputation: {len(null_cols_after)}")
        if null_cols_after:
            logger.info(f"  {null_cols_after}")

        # Store summary
        self.null_columns_summary[table_name] = {
            'before': null_cols_before,
            'after': null_cols_after,
            'shape': df.shape
        }

        # Log detailed summary
        logger.info(f"\nSummary for {table_name}:")
        logger.info(f"  Original NULL columns: {len(null_cols_before)}")
        logger.info(f"  Remaining NULL columns: {len(null_cols_after)}")
        logger.info(f"  DataFrame shape: {df.shape}")

        if null_cols_after:
            logger.info(f"\nColumns still containing NULL values:")
            for col in null_cols_after:
                null_count = df[col].isnull().sum()
                logger.info(f"    {col}: {null_count} NULLs")

            # Export rows with remaining nulls to CSV
            self.export_null_rows_to_csv(df, table_name, null_cols_after)

        # Export audit CSV if enabled
        self.export_audit_csv(df, table_name)

    def print_final_summary(self) -> None:
        """Print final summary of all processed tables"""
        logger.info(f"\n{'='*60}")
        logger.info("FINAL SUMMARY - All Tables")
        logger.info(f"{'='*60}")

        for table_name, summary in self.null_columns_summary.items():
            logger.info(f"\nTable: {table_name}")
            logger.info(f"  Shape: {summary['shape']}")
            logger.info(f"  NULL columns before: {len(summary['before'])}")
            logger.info(f"  NULL columns after: {len(summary['after'])}")
            if summary['after']:
                logger.info(f"  Remaining NULL columns: {summary['after']}")