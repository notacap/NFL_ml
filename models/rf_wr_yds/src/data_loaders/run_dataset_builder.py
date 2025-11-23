"""
Runner script for NFL Dataset Builder with enhanced monitoring and error handling.

This script provides a convenient interface to run the dataset builder with
progress monitoring, validation checks, and helpful output formatting.

Usage:
    python run_dataset_builder.py [--seasons YEAR1,YEAR2] [--test-mode]

Author: Claude Code
Created: 2024-11-23
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from build_dataset import NFLDatasetBuilder


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Build NFL receiving yards prediction dataset')
    
    parser.add_argument(
        '--seasons',
        type=str,
        help='Comma-separated list of seasons to include (e.g., 2022,2023,2024)',
        default=None
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run in test mode with limited data for validation'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Custom output directory (optional)',
        default=None
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def run_dataset_builder(args):
    """
    Run the dataset builder with monitoring and validation.
    
    Args:
        args: Parsed command line arguments
    """
    print("=" * 60)
    print("NFL RECEIVING YARDS DATASET BUILDER")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Initialize builder
        print("\n1. Initializing dataset builder...")
        builder = NFLDatasetBuilder()
        
        # Override seasons if specified
        if args.seasons:
            season_list = [int(s.strip()) for s in args.seasons.split(',')]
            builder.data_config['temporal']['historical_seasons'] = season_list[:-1] if len(season_list) > 1 else []
            builder.data_config['temporal']['current_season'] = season_list[-1]
            print(f"   Using custom seasons: {season_list}")
        
        # Test mode configuration
        if args.test_mode:
            print("   Running in TEST MODE - limited data")
            # Limit to recent seasons and fewer weeks for testing
            builder.data_config['temporal']['historical_seasons'] = [2024]
            builder.data_config['temporal']['current_season'] = 2024
        
        print(f"   Base data path: {builder.base_data_path}")
        print(f"   Output path: {builder.output_path}")
        
        # Build dataset
        print("\n2. Building dataset...")
        print("   This may take several minutes depending on data size...")
        
        output_file = builder.build_dataset()
        
        # Load and display summary
        print("\n3. Dataset Summary:")
        df = pd.read_parquet(output_file)
        
        print(f"   Total rows: {len(df):,}")
        print(f"   Total columns: {len(df.columns):,}")
        print(f"   Unique players: {df['plyr_id'].nunique():,}")
        print(f"   Date range: {df['year'].min()}-{df['year'].max()}")
        print(f"   Week range: {df['week_num'].min()}-{df['week_num'].max()}")
        
        # Target variable statistics
        target_stats = df['next_week_rec_yds'].describe()
        print(f"\n4. Target Variable Statistics (next_week_rec_yds):")
        print(f"   Mean: {target_stats['mean']:.1f} yards")
        print(f"   Median: {target_stats['50%']:.1f} yards")
        print(f"   Std Dev: {target_stats['std']:.1f} yards")
        print(f"   Min: {target_stats['min']:.0f} yards")
        print(f"   Max: {target_stats['max']:.0f} yards")
        
        # Check for missing values
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            print(f"\n5. Missing Values Found:")
            for col, count in null_counts[null_counts > 0].items():
                print(f"   {col}: {count:,} missing ({count/len(df)*100:.1f}%)")
        else:
            print(f"\n5. Data Quality: No missing values found [OK]")
        
        # Sample data preview
        print(f"\n6. Sample Data (first 3 rows):")
        sample_cols = ['plyr_id', 'year', 'week_num', 'next_week_rec_yds', 'plyr_gm_rec_yds', 'plyr_gm_rec_tgt']
        available_cols = [col for col in sample_cols if col in df.columns]
        print(df[available_cols].head(3).to_string(index=False))
        
        elapsed_time = time.time() - start_time
        print(f"\n" + "=" * 60)
        print(f"DATASET BUILD COMPLETED SUCCESSFULLY!")
        print(f"Time elapsed: {elapsed_time:.1f} seconds")
        print(f"Output file: {output_file}")
        print(f"File size: {Path(output_file).stat().st_size / 1024 / 1024:.1f} MB")
        print("=" * 60)
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n" + "!" * 60)
        print(f"DATASET BUILD FAILED!")
        print(f"Error: {e}")
        print(f"Time elapsed: {elapsed_time:.1f} seconds")
        print("!" * 60)
        
        if args.verbose:
            import traceback
            print("\nDetailed error trace:")
            traceback.print_exc()
        
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_arguments()
    run_dataset_builder(args)


if __name__ == "__main__":
    main()