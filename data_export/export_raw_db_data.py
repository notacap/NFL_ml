#!/usr/bin/env python3
"""
Export NFL statistical and referential data from MySQL database to partitioned Parquet files.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')


class NFLDataExporter:
    """Export NFL data from MySQL to partitioned Parquet files."""

    def __init__(self, env_path: str = r"C:\Users\nocap\Desktop\code\NFL_ml\database\.env"):
        """Initialize the exporter with database credentials."""
        self.env_path = Path(env_path)
        self.output_root = Path(r"C:\Users\nocap\Desktop\code\NFL_ml\parquet_files\raw")
        self.engine = None
        self.logger = self._setup_logging()
        self.overwrite = False  # Default: skip existing partitions
        self.export_summary = {
            'successful': [],
            'failed': [],
            'total_rows': 0,
            'start_time': datetime.now()
        }

        # Table mappings organized by category
        self.table_mappings = self._get_table_mappings()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('NFLDataExporter')
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # File handler
        log_dir = Path(__file__).parent / 'export_logs'
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f'export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    def _get_table_mappings(self) -> Dict[str, List[Dict[str, Any]]]:
        """Define table mappings for export."""
        return {
            'game_level': [
                # Player game-level stats
                {'table': 'plyr_gm_pass', 'directory': 'plyr_gm', 'partition_by': ['season', 'week']},
                {'table': 'plyr_gm_rush', 'directory': 'plyr_gm', 'partition_by': ['season', 'week']},
                {'table': 'plyr_gm_rec', 'directory': 'plyr_gm', 'partition_by': ['season', 'week']},
                {'table': 'plyr_gm_def', 'directory': 'plyr_gm', 'partition_by': ['season', 'week']},
                {'table': 'plyr_gm_fmbl', 'directory': 'plyr_gm', 'partition_by': ['season', 'week']},
                {'table': 'plyr_gm_snap_ct', 'directory': 'plyr_gm', 'partition_by': ['season', 'week']},
                {'table': 'plyr_gm_starters', 'directory': 'plyr_gm', 'partition_by': ['season', 'week']},

                # Team game-level stats
                {'table': 'tm_gm_stats', 'directory': 'tm_gm', 'partition_by': ['season', 'week']},
                {'table': 'tm_gm_drive', 'directory': 'tm_gm', 'partition_by': ['season', 'week']},
                {'table': 'tm_gm_exp_pts', 'directory': 'tm_gm', 'partition_by': ['season', 'week']},

                # Game info
                {'table': 'nfl_game', 'directory': 'gm_info', 'partition_by': ['season', 'week']},
                {'table': 'nfl_game_info', 'directory': 'gm_info', 'partition_by': ['season', 'week']},
                {'table': 'nfl_gm_weather', 'directory': 'gm_info', 'partition_by': ['season', 'week']},
                {'table': 'nfl_game_pbp', 'directory': 'gm_info', 'partition_by': ['season', 'week'], 'chunk_size': 50000},
                {'table': 'injury_report', 'directory': 'gm_info', 'partition_by': ['season', 'week']},
            ],
            'season_cumulative': [
                # Player season cumulative stats
                {'table': 'plyr_pass', 'directory': 'plyr_szn', 'partition_by': ['season', 'week']},
                {'table': 'plyr_rush', 'directory': 'plyr_szn', 'partition_by': ['season', 'week']},
                {'table': 'plyr_rec', 'directory': 'plyr_szn', 'partition_by': ['season', 'week']},
                {'table': 'plyr_def', 'directory': 'plyr_szn', 'partition_by': ['season', 'week']},
                {'table': 'plyr_rz_pass', 'directory': 'plyr_szn', 'partition_by': ['season', 'week']},
                {'table': 'plyr_rz_rush', 'directory': 'plyr_szn', 'partition_by': ['season', 'week']},
                {'table': 'plyr_rz_rec', 'directory': 'plyr_szn', 'partition_by': ['season', 'week']},
                {'table': 'plyr_scoring', 'directory': 'plyr_szn', 'partition_by': ['season', 'week']},

                # Team season cumulative stats
                {'table': 'tm_pass', 'directory': 'tm_szn', 'partition_by': ['season', 'week']},
                {'table': 'tm_rush', 'directory': 'tm_szn', 'partition_by': ['season', 'week']},
                {'table': 'tm_rec', 'directory': 'tm_szn', 'partition_by': ['season', 'week']},
                {'table': 'tm_def', 'directory': 'tm_szn', 'partition_by': ['season', 'week']},
                {'table': 'tm_def_pass', 'directory': 'tm_szn', 'partition_by': ['season', 'week']},
                {'table': 'tm_def_rush', 'directory': 'tm_szn', 'partition_by': ['season', 'week']},
                {'table': 'tm_conv', 'directory': 'tm_szn', 'partition_by': ['season', 'week']},
                {'table': 'tm_def_conv', 'directory': 'tm_szn', 'partition_by': ['season', 'week']},
                {'table': 'tm_def_dr_against_avg', 'directory': 'tm_szn', 'partition_by': ['season', 'week']},
                {'table': 'tm_def_vs_qb', 'directory': 'tm_szn', 'partition_by': ['season', 'week']},
                {'table': 'tm_def_vs_rb', 'directory': 'tm_szn', 'partition_by': ['season', 'week']},
                {'table': 'tm_def_vs_te', 'directory': 'tm_szn', 'partition_by': ['season', 'week']},
                {'table': 'tm_def_vs_wr', 'directory': 'tm_szn', 'partition_by': ['season', 'week']},
                {'table': 'nfl_standings', 'directory': 'tm_szn', 'partition_by': ['season', 'week']},
            ],
            'reference': [
                {'table': 'plyr', 'directory': 'players', 'partition_by': ['season']},
                {'table': 'multi_tm_plyr', 'directory': 'players', 'partition_by': ['season']},
            ],
            'static': [
                {'table': 'nfl_team', 'file': 'nfl_team.parquet', 'partition_by': None},
                {'table': 'nfl_season', 'file': 'nfl_season.parquet', 'partition_by': None},
                {'table': 'nfl_week', 'directory': 'static', 'partition_by': ['season']},
                {'table': 'nfl_gm_quarter', 'file': 'nfl_gm_quarter.parquet', 'partition_by': None},
            ]
        }

    def connect_to_database(self) -> bool:
        """Establish connection to MySQL database."""
        try:
            # Load environment variables
            load_dotenv(self.env_path)

            # Get database credentials
            db_config = {
                'host': os.getenv('DB_HOST'),
                'port': os.getenv('DB_PORT', '3306'),
                'database': os.getenv('DB_NAME'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD')
            }

            # Validate credentials
            if not all(db_config.values()):
                self.logger.error("Missing database credentials in .env file")
                return False

            # Create connection string
            connection_string = (
                f"mysql+pymysql://{db_config['user']}:{db_config['password']}@"
                f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )

            # Create engine
            self.engine = create_engine(
                connection_string,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False
            )

            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()

            self.logger.info("Successfully connected to database")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to database: {str(e)}")
            return False

    def create_output_directories(self) -> None:
        """Create necessary output directories."""
        directories = ['plyr_gm', 'tm_gm', 'plyr_szn', 'tm_szn', 'gm_info', 'players']

        for directory in directories:
            dir_path = self.output_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Created output directories in {self.output_root}")

    def build_query_with_joins(self, table_name: str, season_filter: str = None,
                             week_filter: str = None, limit: int = None) -> str:
        """Build SQL query with necessary joins for season and week."""
        # Check if table has season_id and/or week_id columns
        column_check_query = f"""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = '{table_name}'
        AND COLUMN_NAME IN ('season_id', 'week_id')
        """

        with self.engine.connect() as conn:
            result = conn.execute(text(column_check_query))
            columns = [row[0] for row in result]

        has_season = 'season_id' in columns
        has_week = 'week_id' in columns

        # Base query
        select_clause = f"SELECT t.*"
        from_clause = f" FROM {table_name} t"
        where_clause = " WHERE 1=1"

        # Add season join if needed
        if has_season:
            select_clause += ", s.year as season"
            from_clause += " JOIN nfl_season s ON t.season_id = s.season_id"
            if season_filter:
                where_clause += f" AND {season_filter}"

        # Add week join if needed
        if has_week:
            select_clause += ", w.week_num as week"
            from_clause += " JOIN nfl_week w ON t.week_id = w.week_id"
            if week_filter:
                where_clause += f" AND {week_filter}"

        query = select_clause + from_clause + where_clause

        if limit:
            query += f" LIMIT {limit}"

        return query

    def export_table_chunked(self, table_info: Dict[str, Any], season_filter: str = None,
                           week_filter: str = None) -> Tuple[bool, int]:
        """Export a table using chunking for large datasets."""
        table_name = table_info['table']
        chunk_size = table_info.get('chunk_size', 100000)

        try:
            # Get total row count
            count_query = f"SELECT COUNT(*) FROM {table_name}"
            with self.engine.connect() as conn:
                total_rows = conn.execute(text(count_query)).scalar()

            if total_rows == 0:
                self.logger.warning(f"Table {table_name} is empty")
                return True, 0

            self.logger.info(f"Exporting {table_name} ({total_rows:,} rows) in chunks of {chunk_size:,}")

            rows_exported = 0
            offset = 0

            while offset < total_rows:
                # Build query with LIMIT and OFFSET
                query = self.build_query_with_joins(
                    table_name, season_filter, week_filter
                )
                query += f" LIMIT {chunk_size} OFFSET {offset}"

                # Read chunk
                df = pd.read_sql(query, self.engine)

                if not df.empty:
                    # Export chunk
                    self._write_partitioned_parquet(df, table_info)
                    rows_exported += len(df)

                    self.logger.info(f"  Exported {rows_exported:,}/{total_rows:,} rows ({rows_exported*100/total_rows:.1f}%)")

                offset += chunk_size

            return True, rows_exported

        except Exception as e:
            self.logger.error(f"Failed to export {table_name}: {str(e)}")
            return False, 0

    def export_table(self, table_info: Dict[str, Any], season_filter: str = None,
                    week_filter: str = None) -> Tuple[bool, int]:
        """Export a single table to Parquet format."""
        table_name = table_info['table']

        try:
            # Use chunking for large tables
            if 'chunk_size' in table_info:
                return self.export_table_chunked(table_info, season_filter, week_filter)

            # Build and execute query
            query = self.build_query_with_joins(table_name, season_filter, week_filter)

            self.logger.info(f"Exporting {table_name}...")
            df = pd.read_sql(query, self.engine)

            if df.empty:
                self.logger.warning(f"Table {table_name} returned no data")
                return True, 0

            # Write to Parquet
            self._write_partitioned_parquet(df, table_info)

            self.logger.info(f"  Exported {len(df):,} rows from {table_name}")
            return True, len(df)

        except Exception as e:
            self.logger.error(f"Failed to export {table_name}: {str(e)}")
            return False, 0

    def _write_partitioned_parquet(self, df: pd.DataFrame, table_info: Dict[str, Any]) -> None:
        """Write DataFrame to partitioned Parquet files."""
        # Handle static tables (no partitioning)
        if 'file' in table_info:
            output_path = self.output_root / table_info['file']
            df.to_parquet(output_path, compression='snappy', index=False)
            return

        # Handle partitioned tables
        directory = table_info.get('directory')
        partition_cols = table_info.get('partition_by', [])

        if not directory:
            raise ValueError(f"No directory specified for table {table_info['table']}")

        output_path = self.output_root / directory / table_info['table']

        # Ensure partition columns exist and are properly typed
        for col in partition_cols:
            if col in df.columns:
                # Convert to string for partitioning
                df[col] = df[col].astype(str)

        # Write partitioned Parquet
        if partition_cols and all(col in df.columns for col in partition_cols):
            # Create table with partitioning
            table = pa.Table.from_pandas(df)

            # Determine existing data behavior based on overwrite setting
            existing_behavior = 'delete_matching' if self.overwrite else 'overwrite_or_ignore'

            pq.write_to_dataset(
                table,
                root_path=str(output_path),
                partition_cols=partition_cols,
                compression='snappy',
                existing_data_behavior=existing_behavior
            )
        else:
            # Write without partitioning if columns don't exist
            output_file = output_path / f"{table_info['table']}.parquet"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_file, compression='snappy', index=False)

    def parse_season_filter(self, seasons: str) -> str:
        """Parse season filter string into SQL WHERE clause."""
        if not seasons:
            return None

        # Handle range (e.g., "2023-2024")
        if '-' in seasons and ',' not in seasons:
            parts = seasons.split('-')
            if len(parts) == 2:
                start_year = int(parts[0])
                end_year = int(parts[1])
                return f"s.year BETWEEN {start_year} AND {end_year}"

        # Handle list (e.g., "2023,2024")
        if ',' in seasons:
            years = [int(y.strip()) for y in seasons.split(',')]
            years_str = ','.join(map(str, years))
            return f"s.year IN ({years_str})"

        # Handle single year
        return f"s.year = {int(seasons)}"

    def parse_week_filter(self, weeks: str) -> str:
        """Parse week filter string into SQL WHERE clause."""
        if not weeks:
            return None

        # Handle range (e.g., "1-5")
        if '-' in weeks and ',' not in weeks:
            parts = weeks.split('-')
            if len(parts) == 2:
                start_week = int(parts[0])
                end_week = int(parts[1])
                # Convert to string comparison for week_num
                return f"CAST(w.week_num AS UNSIGNED) BETWEEN {start_week} AND {end_week}"

        # Handle list (e.g., "1,2,3")
        if ',' in weeks:
            week_nums = [w.strip() for w in weeks.split(',')]
            weeks_str = ','.join([f"'{w}'" for w in week_nums])
            return f"w.week_num IN ({weeks_str})"

        # Handle single week
        return f"w.week_num = '{weeks}'"

    def validate_filters(self, season_filter: str, week_filter: str) -> bool:
        """Validate season and week filters."""
        try:
            if season_filter:
                # Check if seasons exist
                query = f"SELECT COUNT(*) FROM nfl_season s WHERE {season_filter}"
                with self.engine.connect() as conn:
                    count = conn.execute(text(query)).scalar()
                    if count == 0:
                        self.logger.error("No matching seasons found for filter")
                        return False

            if week_filter:
                # Validate week numbers (1-18 for regular season + playoffs)
                # Note: This is a basic check, actual weeks may vary
                pass

            return True

        except Exception as e:
            self.logger.error(f"Filter validation failed: {str(e)}")
            return False

    def export_category(self, category: str, tables: List[str] = None,
                       season_filter: str = None, week_filter: str = None) -> None:
        """Export all tables in a category."""
        if category not in self.table_mappings:
            self.logger.error(f"Unknown category: {category}")
            return

        table_list = self.table_mappings[category]

        # Filter specific tables if requested
        if tables:
            table_list = [t for t in table_list if t['table'] in tables]

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Exporting {category} tables")
        self.logger.info(f"{'='*60}")

        for table_info in table_list:
            success, rows = self.export_table(table_info, season_filter, week_filter)

            if success:
                self.export_summary['successful'].append(table_info['table'])
                self.export_summary['total_rows'] += rows
            else:
                self.export_summary['failed'].append(table_info['table'])

    def print_summary(self) -> None:
        """Print export summary."""
        end_time = datetime.now()
        duration = end_time - self.export_summary['start_time']

        self.logger.info("\n" + "="*60)
        self.logger.info("EXPORT SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Start Time: {self.export_summary['start_time']}")
        self.logger.info(f"End Time: {end_time}")
        self.logger.info(f"Duration: {duration}")
        self.logger.info(f"Total Rows Exported: {self.export_summary['total_rows']:,}")
        self.logger.info(f"Successful Tables: {len(self.export_summary['successful'])}")
        self.logger.info(f"Failed Tables: {len(self.export_summary['failed'])}")

        if self.export_summary['failed']:
            self.logger.error("\nFailed Tables:")
            for table in self.export_summary['failed']:
                self.logger.error(f"  - {table}")

        self.logger.info(f"\nOutput Directory: {self.output_root}")

    def run(self, tables: List[str] = None, seasons: str = None,
            weeks: str = None, categories: List[str] = None,
            overwrite: bool = False) -> None:
        """Run the export process."""
        # Store overwrite setting
        self.overwrite = overwrite

        # Connect to database
        if not self.connect_to_database():
            return

        # Create output directories
        self.create_output_directories()

        # Parse filters
        season_filter = self.parse_season_filter(seasons) if seasons else None
        week_filter = self.parse_week_filter(weeks) if weeks else None

        # Validate filters
        if not self.validate_filters(season_filter, week_filter):
            return

        # Log filters
        if season_filter or week_filter:
            self.logger.info("Applying filters:")
            if season_filter:
                self.logger.info(f"  Seasons: {seasons}")
            if week_filter:
                self.logger.info(f"  Weeks: {weeks}")

        # Log overwrite mode
        if self.overwrite:
            self.logger.info("Overwrite mode: ENABLED - existing partitions will be replaced")
        else:
            self.logger.info("Overwrite mode: DISABLED - existing partitions will be skipped")

        # Export categories
        if not categories:
            categories = ['static', 'reference', 'game_level', 'season_cumulative']

        for category in categories:
            self.export_category(category, tables, season_filter, week_filter)

        # Print summary
        self.print_summary()

        # Close database connection
        if self.engine:
            self.engine.dispose()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Export NFL data from MySQL to partitioned Parquet files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all data
  python export_raw_db_data.py

  # Export specific seasons
  python export_raw_db_data.py --seasons 2024
  python export_raw_db_data.py --seasons 2023,2024
  python export_raw_db_data.py --seasons 2023-2024

  # Export specific weeks
  python export_raw_db_data.py --weeks 1-8
  python export_raw_db_data.py --weeks 1,2,3,8,9

  # Export specific tables
  python export_raw_db_data.py --tables plyr_pass,tm_pass

  # Combine filters
  python export_raw_db_data.py --seasons 2024 --weeks 1-8

  # Export specific categories
  python export_raw_db_data.py --categories game_level,season_cumulative

  # Overwrite existing partitions (default: skip existing)
  python export_raw_db_data.py --seasons 2024 --weeks 1-8 --overwrite
        """
    )

    parser.add_argument(
        '--seasons',
        type=str,
        help='Season filter (e.g., "2024" or "2023,2024" or "2023-2024")'
    )

    parser.add_argument(
        '--weeks',
        type=str,
        help='Week filter (e.g., "1" or "1,2,3" or "1-8")'
    )

    parser.add_argument(
        '--tables',
        type=str,
        help='Comma-separated list of specific tables to export'
    )

    parser.add_argument(
        '--categories',
        type=str,
        help='Comma-separated list of categories to export (static,reference,game_level,season_cumulative)'
    )

    parser.add_argument(
        '--env-path',
        type=str,
        default=r"C:\Users\nocap\Desktop\code\NFL_ml\database\.env",
        help='Path to .env file with database credentials'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing partition data (default: skip existing partitions)'
    )

    args = parser.parse_args()

    # Parse tables and categories
    tables = args.tables.split(',') if args.tables else None
    categories = args.categories.split(',') if args.categories else None

    # Create exporter and run
    exporter = NFLDataExporter(env_path=args.env_path)
    exporter.run(
        tables=tables,
        seasons=args.seasons,
        weeks=args.weeks,
        categories=categories,
        overwrite=args.overwrite
    )


if __name__ == '__main__':
    main()