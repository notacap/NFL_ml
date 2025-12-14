"""
NFL Dataset Pipeline Runner

Runs the complete dataset building pipeline:
1. build_dataset.py - Creates base WR receiving yards dataset
2. enrich_with_team_defense.py - Adds opponent team defense stats

Usage:
    python run_pipeline.py
    python run_pipeline.py --test-mode
    python run_pipeline.py --seasons 2023,2024
    python run_pipeline.py --skip-enrichment

Author: Claude Code
Created: 2024-12-13
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from build_dataset import NFLDatasetBuilder
from enrich_with_team_defense import TeamDefenseEnricher


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Build NFL receiving yards prediction dataset with enrichments'
    )

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
        '--skip-enrichment',
        action='store_true',
        help='Skip team defense enrichment step (build base dataset only)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def run_pipeline(args):
    """
    Run the complete dataset building pipeline.

    Args:
        args: Parsed command line arguments
    """
    print("=" * 60)
    print("NFL RECEIVING YARDS DATASET PIPELINE")
    print("=" * 60)

    start_time = time.time()

    try:
        # =====================================================================
        # STEP 1: Build Base Dataset
        # =====================================================================
        print("\n" + "-" * 60)
        print("STEP 1: Building base dataset...")
        print("-" * 60)

        builder = NFLDatasetBuilder()

        # Override seasons if specified
        if args.seasons:
            season_list = [int(s.strip()) for s in args.seasons.split(',')]
            builder.data_config['temporal']['historical_seasons'] = season_list[:-1] if len(season_list) > 1 else []
            builder.data_config['temporal']['current_season'] = season_list[-1]
            print(f"Using custom seasons: {season_list}")

        # Test mode configuration
        if args.test_mode:
            print("Running in TEST MODE - limited data")
            builder.data_config['temporal']['historical_seasons'] = [2024]
            builder.data_config['temporal']['current_season'] = 2024

        print(f"Base data path: {builder.base_data_path}")
        print(f"Output path: {builder.output_path}")

        base_output_file = builder.build_dataset()
        base_df = pd.read_parquet(base_output_file)

        print(f"\nBase dataset complete:")
        print(f"  Rows: {len(base_df):,}")
        print(f"  Columns: {len(base_df.columns)}")
        print(f"  Players: {base_df['plyr_id'].nunique():,}")
        print(f"  File: {base_output_file}")

        # =====================================================================
        # STEP 2: Enrich with Team Defense Stats
        # =====================================================================
        if args.skip_enrichment:
            print("\n" + "-" * 60)
            print("STEP 2: Skipped (--skip-enrichment flag)")
            print("-" * 60)
            final_output_file = base_output_file
        else:
            print("\n" + "-" * 60)
            print("STEP 2: Enriching with team defense stats...")
            print("-" * 60)

            enricher = TeamDefenseEnricher()
            enriched_output_file = enricher.run(base_dataset_path=Path(base_output_file))

            enriched_df = pd.read_parquet(enriched_output_file)

            print(f"\nEnriched dataset complete:")
            print(f"  Rows: {len(enriched_df):,}")
            print(f"  Columns: {len(enriched_df.columns)}")
            print(f"  New columns: {len(enriched_df.columns) - len(base_df.columns)}")
            print(f"  File: {enriched_output_file}")

            final_output_file = enriched_output_file

        # =====================================================================
        # Summary
        # =====================================================================
        elapsed_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Time elapsed: {elapsed_time:.1f} seconds")
        print(f"Final output: {final_output_file}")
        print(f"File size: {Path(final_output_file).stat().st_size / 1024 / 1024:.1f} MB")

        # Show target variable stats
        final_df = pd.read_parquet(final_output_file)
        target_stats = final_df['next_week_rec_yds'].describe()
        print(f"\nTarget variable (next_week_rec_yds):")
        print(f"  Mean: {target_stats['mean']:.1f} yards")
        print(f"  Median: {target_stats['50%']:.1f} yards")
        print(f"  Std Dev: {target_stats['std']:.1f} yards")

        print("=" * 60)

        return final_output_file

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n" + "!" * 60)
        print(f"PIPELINE FAILED!")
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
    run_pipeline(args)


if __name__ == "__main__":
    main()
