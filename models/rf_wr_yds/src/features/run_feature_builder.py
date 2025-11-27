"""
Feature Builder Runner Script

This script executes the rolling feature builder on the processed dataset
and validates the output.

Usage:
    python run_feature_builder.py

Author: Claude Code
Created: 2024-11-25
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import yaml

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from features.build_rolling_features import (
    RollingFeatureBuilder,
    build_all_basic_features,
    IMPUTATION_SENTINEL
)


def find_latest_dataset(data_path: Path) -> Path:
    """Find the most recent base dataset file."""
    parquet_files = list(data_path.glob("nfl_wr_receiving_yards_dataset_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No base dataset found in {data_path}")
    return max(parquet_files, key=lambda x: x.stat().st_mtime)


def validate_feature_output(df: pd.DataFrame, original_df: pd.DataFrame) -> dict:
    """
    Comprehensive validation of feature engineering output.

    Returns:
        Dictionary with validation results
    """
    results = {
        'passed': True,
        'checks': {}
    }

    # Check 1: All original columns preserved
    original_cols = set(original_df.columns)
    result_cols = set(df.columns)
    missing_cols = original_cols - result_cols
    results['checks']['original_columns_preserved'] = {
        'passed': len(missing_cols) == 0,
        'missing': list(missing_cols)
    }
    if missing_cols:
        results['passed'] = False
        print(f"FAIL: Missing original columns: {missing_cols}")

    # Check 2: Same number of rows (no data loss)
    results['checks']['row_count_preserved'] = {
        'passed': len(df) == len(original_df),
        'original': len(original_df),
        'result': len(df)
    }
    if len(df) != len(original_df):
        results['passed'] = False
        print(f"FAIL: Row count changed from {len(original_df)} to {len(df)}")

    # Check 3: Rolling features exist
    rolling_features = [c for c in df.columns if c.startswith('roll_')]
    results['checks']['rolling_features_created'] = {
        'passed': len(rolling_features) > 0,
        'count': len(rolling_features),
        'features': rolling_features
    }
    if not rolling_features:
        results['passed'] = False
        print("FAIL: No rolling features created")

    # Check 4: Efficiency features exist
    efficiency_features = ['season_targets_per_game', 'season_yards_per_game',
                          'yards_per_reception', 'yards_per_target']
    existing_eff = [f for f in efficiency_features if f in df.columns]
    results['checks']['efficiency_features_created'] = {
        'passed': len(existing_eff) == len(efficiency_features),
        'expected': efficiency_features,
        'found': existing_eff
    }

    # Check 5: No -999 values in new features (they should be NaN if imputed source)
    new_features = list(result_cols - original_cols)
    sentinel_in_new = {}
    for col in new_features:
        if df[col].dtype in ['float64', 'int64']:
            sentinel_count = (df[col] == IMPUTATION_SENTINEL).sum()
            if sentinel_count > 0:
                sentinel_in_new[col] = sentinel_count

    results['checks']['no_sentinel_in_new_features'] = {
        'passed': len(sentinel_in_new) == 0,
        'features_with_sentinel': sentinel_in_new
    }
    if sentinel_in_new:
        results['passed'] = False
        print(f"FAIL: New features contain -999 sentinel values: {sentinel_in_new}")

    # Check 6: First game of season has NaN rolling features (no leakage)
    first_game_mask = df['game_seq_num'] == 1
    first_game_rolling = df.loc[first_game_mask, rolling_features]
    non_null_first_games = first_game_rolling.notna().sum().sum()

    results['checks']['no_first_game_rolling_values'] = {
        'passed': non_null_first_games == 0,
        'non_null_count': int(non_null_first_games)
    }
    if non_null_first_games > 0:
        results['passed'] = False
        print(f"FAIL: {non_null_first_games} non-null rolling values for first games (potential leakage)")

    # Check 7: Rolling values are reasonable (not identical to source)
    sample_check_passed = True
    for col in rolling_features[:3]:  # Check first 3 rolling features
        # Get source column name
        source_col = None
        for src in ['plyr_gm_rec_yds', 'plyr_gm_rec_tgt', 'plyr_gm_rec']:
            if src.replace('plyr_gm_rec_', '').replace('plyr_gm_', '') in col:
                source_col = src
                break

        if source_col and source_col in df.columns:
            # Rolling should NOT be identical to source (unless window=1 which we don't use)
            identical = (df[col] == df[source_col]).sum()
            pct_identical = identical / len(df) * 100
            if pct_identical > 50:  # More than 50% identical is suspicious
                sample_check_passed = False
                print(f"WARNING: {col} is {pct_identical:.1f}% identical to {source_col}")

    results['checks']['rolling_not_identical_to_source'] = {
        'passed': sample_check_passed
    }

    # Check 8: Target correlation validation
    if 'next_week_rec_yds' in df.columns:
        high_corr_features = []
        for col in new_features:
            if df[col].dtype in ['float64', 'int64'] and not col.startswith('has_min'):
                corr = df[col].corr(df['next_week_rec_yds'])
                if abs(corr) > 0.8:
                    high_corr_features.append((col, corr))

        results['checks']['no_suspicious_correlations'] = {
            'passed': len(high_corr_features) == 0,
            'high_correlation_features': high_corr_features
        }
        if high_corr_features:
            results['passed'] = False
            print(f"FAIL: Features with suspiciously high correlation to target: {high_corr_features}")

    return results


def print_feature_summary(df: pd.DataFrame, new_features: list):
    """Print summary statistics for new features."""
    print("\n" + "=" * 70)
    print("NEW FEATURE SUMMARY")
    print("=" * 70)

    # Rolling features
    rolling_features = [f for f in new_features if f.startswith('roll_')]
    print(f"\nRolling Features ({len(rolling_features)} total):")

    for feat in sorted(rolling_features):
        if feat in df.columns:
            non_null = df[feat].notna().sum()
            mean_val = df[feat].mean()
            std_val = df[feat].std()
            print(f"  {feat:40s} | non-null: {non_null:5d} | mean: {mean_val:7.2f} | std: {std_val:7.2f}")

    # Efficiency features
    eff_features = ['season_targets_per_game', 'season_yards_per_game',
                   'yards_per_reception', 'yards_per_target']
    existing_eff = [f for f in eff_features if f in df.columns]
    print(f"\nEfficiency Features ({len(existing_eff)} total):")

    for feat in existing_eff:
        non_null = df[feat].notna().sum()
        mean_val = df[feat].mean()
        std_val = df[feat].std()
        print(f"  {feat:40s} | non-null: {non_null:5d} | mean: {mean_val:7.2f} | std: {std_val:7.2f}")

    # Metadata features
    meta_features = [f for f in new_features if f.startswith('has_min') or f == 'game_seq_num']
    print(f"\nMetadata Features ({len(meta_features)} total):")

    for feat in meta_features:
        if feat in df.columns:
            if feat.startswith('has_min'):
                true_pct = df[feat].mean() * 100
                print(f"  {feat:40s} | True: {true_pct:5.1f}%")
            else:
                mean_val = df[feat].mean()
                max_val = df[feat].max()
                print(f"  {feat:40s} | mean: {mean_val:5.2f} | max: {max_val:3.0f}")


def save_feature_metadata(output_path: Path, builder: RollingFeatureBuilder,
                         validation_results: dict, df: pd.DataFrame):
    """Save feature engineering metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    metadata = {
        'created_at': timestamp,
        'dataset_shape': {
            'rows': len(df),
            'columns': len(df.columns)
        },
        'features_created': builder.features_created,
        'rolling_windows': builder.rolling_windows,
        'validation_results': validation_results,
        'feature_documentation': builder.get_feature_documentation()
    }

    metadata_file = output_path / f"feature_metadata_{timestamp}.yaml"
    with open(metadata_file, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    print(f"\nMetadata saved to: {metadata_file}")


def main():
    """Main execution function."""
    print("=" * 70)
    print("NFL WR RECEIVING YARDS - FEATURE ENGINEERING PIPELINE")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Set up paths
    data_path = project_root / "data" / "processed"

    # Find and load dataset
    print("\n[1/4] Loading base dataset...")
    try:
        dataset_file = find_latest_dataset(data_path)
        print(f"  Found: {dataset_file.name}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    original_df = pd.read_parquet(dataset_file)
    print(f"  Loaded: {len(original_df):,} rows, {len(original_df.columns)} columns")

    # Build features
    print("\n[2/4] Building features...")
    builder = RollingFeatureBuilder(
        rolling_windows=[3, 5],
        exclude_imputed=True,
        validate_leakage=True
    )

    result_df = builder.build_features(original_df)

    new_features = [c for c in result_df.columns if c not in original_df.columns]
    print(f"  Created {len(new_features)} new features")

    # Validate output
    print("\n[3/4] Validating feature output...")
    validation_results = validate_feature_output(result_df, original_df)

    if validation_results['passed']:
        print("  All validation checks PASSED")
    else:
        print("  Some validation checks FAILED - see details above")

    # Print feature summary
    print_feature_summary(result_df, new_features)

    # Save output
    print("\n[4/4] Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"nfl_wr_features_v1_{timestamp}.parquet"
    output_file = data_path / output_filename

    result_df.to_parquet(output_file, compression='snappy', index=False)
    print(f"  Dataset saved: {output_file}")

    # Save metadata
    save_feature_metadata(data_path, builder, validation_results, result_df)

    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nOutput dataset: {output_file}")
    print(f"Total rows: {len(result_df):,}")
    print(f"Total columns: {len(result_df.columns)}")
    print(f"New features: {len(new_features)}")
    print(f"Validation: {'PASSED' if validation_results['passed'] else 'FAILED'}")

    # Return success/failure status
    return 0 if validation_results['passed'] else 1


if __name__ == "__main__":
    sys.exit(main())
