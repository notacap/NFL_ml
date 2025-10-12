#!/usr/bin/env python3
"""
Float Precision Analysis Script
Analyzes CSV data to determine optimal MySQL precision for float/decimal columns
"""

import pandas as pd
import numpy as np
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import re

# Base data directory
BASE_DIR = Path(r"C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\2024")

# Column mapping configuration
COLUMN_ANALYSES = [
    # plyr_gm_pass columns (game-level advanced passing)
    {
        "db_table": "plyr_gm_pass",
        "db_column": "plyr_gm_pass_first_dwn_pct",
        "csv_column": "1D%",
        "file_pattern": "games/week_*.0/clean/cleaned_*_gm_plyr_adv_passing_*.csv",
        "transform": "numeric_divide_by_100",  # Raw numeric, not %-formatted
        "current_type": "FLOAT(5,4)"
    },
    {
        "db_table": "plyr_gm_pass",
        "db_column": "plyr_gm_pass_drp_pct",
        "csv_column": "Drop%",
        "file_pattern": "games/week_*.0/clean/cleaned_*_gm_plyr_adv_passing_*.csv",
        "transform": "divide_by_100",
        "current_type": "FLOAT(5,4)"
    },
    {
        "db_table": "plyr_gm_pass",
        "db_column": "plyr_gm_pass_off_tgt_pct",
        "csv_column": "Bad%",
        "file_pattern": "games/week_*.0/clean/cleaned_*_gm_plyr_adv_passing_*.csv",
        "transform": "divide_by_100",
        "current_type": "FLOAT(5,4)"
    },
    {
        "db_table": "plyr_gm_pass",
        "db_column": "plyr_gm_pass_prss_pct",
        "csv_column": "Prss%",
        "file_pattern": "games/week_*.0/clean/cleaned_*_gm_plyr_adv_passing_*.csv",
        "transform": "divide_by_100",
        "current_type": "FLOAT(5,4)"
    },
    {
        "db_table": "plyr_gm_pass",
        "db_column": "plyr_gm_pass_cmp_pct",
        "csv_column": "CALCULATED",
        "file_pattern": "games/week_*.0/clean/cleaned_*_gm_plyr_adv_passing_*.csv",
        "transform": "calculated",
        "current_type": "DECIMAL(7,4)"
    },
    {
        "db_table": "plyr_gm_pass",
        "db_column": "plyr_gm_pass_td_pct",
        "csv_column": "CALCULATED",
        "file_pattern": "games/week_*.0/clean/cleaned_*_gm_plyr_adv_passing_*.csv",
        "transform": "calculated",
        "current_type": "DECIMAL(7,4)"
    },
    {
        "db_table": "plyr_gm_pass",
        "db_column": "plyr_gm_pass_int_pct",
        "csv_column": "CALCULATED",
        "file_pattern": "games/week_*.0/clean/cleaned_*_gm_plyr_adv_passing_*.csv",
        "transform": "calculated",
        "current_type": "DECIMAL(7,4)"
    },
    {
        "db_table": "plyr_gm_pass",
        "db_column": "plyr_gm_pass_sk_pct",
        "csv_column": "CALCULATED",
        "file_pattern": "games/week_*.0/clean/cleaned_*_gm_plyr_adv_passing_*.csv",
        "transform": "calculated",
        "current_type": "DECIMAL(7,4)"
    },

    # plyr_pass columns (season-level advanced passing)
    {
        "db_table": "plyr_pass",
        "db_column": "plyr_pass_cmp_pct",
        "csv_column": "Accuracy OnTgt%",  # Season-level has different column names
        "file_pattern": "plyr_adv_passing/week_*/clean/*.csv",
        "transform": "numeric_divide_by_100",  # Stored as numbers like 80.3
        "current_type": "DECIMAL(5,4)"
    },
    {
        "db_table": "plyr_pass",
        "db_column": "plyr_pass_td_pct",
        "csv_column": "CALCULATED_TD_PCT",  # TD / Att * 100 - need to calculate
        "file_pattern": "plyr_adv_passing/week_*/clean/*.csv",
        "transform": "calculated",
        "current_type": "DECIMAL(7,4)"
    },
    {
        "db_table": "plyr_pass",
        "db_column": "plyr_pass_int_pct",
        "csv_column": "CALCULATED_INT_PCT",  # Int / Att * 100 - need to calculate
        "file_pattern": "plyr_adv_passing/week_*/clean/*.csv",
        "transform": "calculated",
        "current_type": "DECIMAL(5,4)"
    },
    {
        "db_table": "plyr_pass",
        "db_column": "plyr_pass_sk_pct",
        "csv_column": "CALCULATED_SK_PCT",  # Sk / (Att + Sk) * 100 - need to calculate
        "file_pattern": "plyr_adv_passing/week_*/clean/*.csv",
        "transform": "calculated",
        "current_type": "DECIMAL(5,4)"
    },
    {
        "db_table": "plyr_pass",
        "db_column": "plyr_pass_prss_pct",
        "csv_column": "Pressure Prss%",
        "file_pattern": "plyr_adv_passing/week_*/clean/*.csv",
        "transform": "numeric_divide_by_100",  # Stored as numbers like 21.5
        "current_type": "DECIMAL(4,3)"
    },

    # tm_def_pass columns (team defense passing)
    {
        "db_table": "tm_def_pass",
        "db_column": "tm_def_pass_cmp_pct",
        "csv_column": "Cmp%",
        "file_pattern": "tm_def_pass/week_*/*.csv",
        "transform": "numeric_divide_by_100",  # Stored as numbers like 62.2
        "current_type": "DECIMAL(7,4)"
    },
    {
        "db_table": "tm_def_pass",
        "db_column": "tm_def_pass_td_pct",
        "csv_column": "TD%",
        "file_pattern": "tm_def_pass/week_*/*.csv",
        "transform": "numeric_divide_by_100",  # Stored as numbers like 4.1
        "current_type": "DECIMAL(5,4)"
    },
    {
        "db_table": "tm_def_pass",
        "db_column": "tm_def_int_pct",
        "csv_column": "Int%",
        "file_pattern": "tm_def_pass/week_*/*.csv",
        "transform": "numeric_divide_by_100",  # Stored as numbers like 2.4
        "current_type": "DECIMAL(5,4)"
    },
    {
        "db_table": "tm_def_pass",
        "db_column": "tm_def_sk_pct",
        "csv_column": "Sk%",
        "file_pattern": "tm_def_pass/week_*/*.csv",
        "transform": "numeric_divide_by_100",  # Stored as numbers like 7.0
        "current_type": "DECIMAL(5,4)"
    }
]


def extract_decimal_places(value) -> int:
    """Extract the number of decimal places from a numeric value"""
    if pd.isna(value) or value == '':
        return 0

    str_val = str(value).strip()

    # Remove percentage sign if present
    str_val = str_val.rstrip('%')

    if '.' in str_val:
        decimal_part = str_val.split('.')[1]
        # Remove trailing zeros for actual precision
        decimal_part = decimal_part.rstrip('0')
        return len(decimal_part)
    return 0


def parse_percentage(value) -> float:
    """Parse percentage string to float decimal (16.7% -> 0.167)"""
    if pd.isna(value) or value == '':
        return np.nan

    str_val = str(value).strip()
    if str_val.endswith('%'):
        numeric_val = float(str_val.rstrip('%'))
        return numeric_val / 100.0

    return float(str_val)


def analyze_column_data(files: List[Path], csv_column: str, transform: str) -> Dict[str, Any]:
    """Analyze data from a specific column across multiple CSV files"""

    all_values = []
    raw_values = []
    decimal_places = []

    for file_path in files:
        try:
            df = pd.read_csv(file_path)

            if csv_column not in df.columns:
                continue

            # Extract values
            column_data = df[csv_column]

            # Store raw values for analysis
            raw_values.extend(column_data.dropna().tolist())

            # Apply transformation
            if transform == "divide_by_100":
                # For percentage-formatted strings (e.g., "16.7%")
                transformed = column_data.apply(parse_percentage)
                all_values.extend(transformed.dropna().tolist())

                # Analyze decimal places in raw percentage values
                for val in column_data.dropna():
                    if str(val).strip().endswith('%'):
                        # For percentages like 16.7%, we need 3 decimal places after division (0.167)
                        raw_decimal = extract_decimal_places(str(val).rstrip('%'))
                        decimal_places.append(raw_decimal + 2)  # +2 because dividing by 100 adds 2 decimals
                    else:
                        decimal_places.append(extract_decimal_places(val))

            elif transform == "numeric_divide_by_100":
                # For raw numeric percentages (e.g., 16.7 -> 0.167)
                for val in column_data.dropna():
                    try:
                        numeric_val = float(val)
                        all_values.append(numeric_val / 100.0)
                        # Calculate decimal places needed after transformation
                        raw_decimal = extract_decimal_places(val)
                        decimal_places.append(raw_decimal + 2)  # +2 because dividing by 100 adds 2 decimals
                    except (ValueError, TypeError):
                        continue

            elif transform == "none":
                # Values are already in the correct format
                for val in column_data.dropna():
                    try:
                        all_values.append(float(val))
                        decimal_places.append(extract_decimal_places(val))
                    except (ValueError, TypeError):
                        continue

            elif transform == "calculated":
                # These are calculated columns - skip for now
                continue

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    if not all_values:
        return None

    # Convert to numpy array for statistical analysis
    values_array = np.array(all_values, dtype=float)

    # Calculate statistics
    stats = {
        "sample_size": len(values_array),
        "min": float(np.min(values_array)),
        "max": float(np.max(values_array)),
        "mean": float(np.mean(values_array)),
        "median": float(np.median(values_array)),
        "std_dev": float(np.std(values_array)),
        "percentiles": {
            "p1": float(np.percentile(values_array, 1)),
            "p5": float(np.percentile(values_array, 5)),
            "p25": float(np.percentile(values_array, 25)),
            "p75": float(np.percentile(values_array, 75)),
            "p95": float(np.percentile(values_array, 95)),
            "p99": float(np.percentile(values_array, 99))
        },
        "max_decimal_places": max(decimal_places) if decimal_places else 0,
        "min_decimal_places": min(decimal_places) if decimal_places else 0,
        "sample_values": [float(v) for v in np.random.choice(values_array, min(10, len(values_array)), replace=False)],
        "raw_sample_values": raw_values[:10] if raw_values else []
    }

    return stats


def determine_mysql_precision(stats: Dict[str, Any], transform: str) -> Tuple[str, str, str]:
    """Determine optimal MySQL data type and precision"""

    if not stats:
        return "UNKNOWN", "Unable to analyze", "Low"

    max_val = abs(stats["max"])
    min_val = abs(stats["min"])
    largest_magnitude = max(max_val, min_val)

    # Determine integer digits needed
    if largest_magnitude >= 1:
        integer_digits = len(str(int(largest_magnitude)))
    else:
        integer_digits = 1  # At least 1 digit before decimal

    # Determine decimal precision needed
    max_decimal = stats["max_decimal_places"]

    # For percentage columns, we want to preserve at least 4 decimal places (0.1% accuracy)
    if transform in ["divide_by_100", "numeric_divide_by_100"]:
        recommended_decimal = max(4, max_decimal)
    else:
        recommended_decimal = max(3, max_decimal)  # At least 3 decimal places

    # Total precision (M) = integer digits + decimal places
    total_precision = integer_digits + recommended_decimal

    # Decide between FLOAT and DECIMAL
    # DECIMAL is better for exact precision (financial, percentages)
    # FLOAT is acceptable for approximate scientific values

    if transform in ["divide_by_100", "numeric_divide_by_100"] or recommended_decimal <= 4:
        # Percentages should use DECIMAL for exactness
        recommended_type = f"DECIMAL({total_precision},{recommended_decimal})"
        confidence = "High"
    else:
        # Other values can use FLOAT
        recommended_type = f"FLOAT({total_precision},{recommended_decimal})"
        confidence = "Medium"

    # Justification
    justification = (
        f"Max value: {stats['max']:.6f}, requires {integer_digits} integer digit(s). "
        f"Max observed decimal precision: {max_decimal} places. "
        f"Recommended {recommended_decimal} decimal places to preserve data fidelity. "
    )

    if transform in ["divide_by_100", "numeric_divide_by_100"]:
        justification += "Using DECIMAL for percentage accuracy. "

    return recommended_type, justification, confidence


def main():
    """Main analysis function"""

    results = []

    print("=" * 80)
    print("FLOAT PRECISION ANALYSIS - NFL STATISTICS DATABASE")
    print("=" * 80)
    print()

    for config in COLUMN_ANALYSES:
        print(f"Analyzing: {config['db_table']}.{config['db_column']}")
        print(f"  CSV Column: {config['csv_column']}")
        print(f"  Current Type: {config['current_type']}")

        # Find matching files
        file_pattern = str(BASE_DIR / config["file_pattern"])
        files = [Path(f) for f in glob.glob(file_pattern)]

        print(f"  Found {len(files)} CSV files")

        if not files:
            print(f"  WARNING: No files found for pattern: {file_pattern}")
            print()
            continue

        # Analyze data
        if config["csv_column"] == "CALCULATED":
            print(f"  SKIPPING: Calculated column - requires separate analysis")
            print()
            continue

        stats = analyze_column_data(files, config["csv_column"], config["transform"])

        if not stats:
            print(f"  ERROR: Unable to extract data")
            print()
            continue

        # Determine optimal precision
        recommended_type, justification, confidence = determine_mysql_precision(stats, config["transform"])

        # Compile result
        result = {
            "db_table": config["db_table"],
            "db_column": config["db_column"],
            "csv_column": config["csv_column"],
            "current_type": config["current_type"],
            "transform": config["transform"],
            "statistics": stats,
            "recommended_type": recommended_type,
            "justification": justification,
            "confidence": confidence
        }

        results.append(result)

        # Print summary
        print(f"  Sample Size: {stats['sample_size']:,}")
        print(f"  Range: {stats['min']:.6f} to {stats['max']:.6f}")
        print(f"  Mean: {stats['mean']:.6f} | Median: {stats['median']:.6f}")
        print(f"  Decimal Places: {stats['min_decimal_places']} to {stats['max_decimal_places']}")
        print(f"  Recommended: {recommended_type}")
        print(f"  Confidence: {confidence}")
        print()

    # Save results to JSON
    output_file = Path(__file__).parent / "precision_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print(f"Analysis complete. Results saved to: {output_file}")
    print("=" * 80)

    # Print formatted report
    print("\n" + "=" * 80)
    print("DETAILED PRECISION RECOMMENDATIONS")
    print("=" * 80)

    for result in results:
        print(f"\nColumn: {result['db_table']}.{result['db_column']}")
        print(f"CSV Source: {result['csv_column']}")
        print(f"Transform: {result['transform']}")
        print(f"\nData Statistics:")
        print(f"  Sample Size: {result['statistics']['sample_size']:,}")
        print(f"  Range: {result['statistics']['min']:.6f} to {result['statistics']['max']:.6f}")
        print(f"  Mean: {result['statistics']['mean']:.6f}")
        print(f"  Median: {result['statistics']['median']:.6f}")
        print(f"  Std Dev: {result['statistics']['std_dev']:.6f}")
        print(f"  Percentiles:")
        print(f"    P1:  {result['statistics']['percentiles']['p1']:.6f}")
        print(f"    P5:  {result['statistics']['percentiles']['p5']:.6f}")
        print(f"    P25: {result['statistics']['percentiles']['p25']:.6f}")
        print(f"    P75: {result['statistics']['percentiles']['p75']:.6f}")
        print(f"    P95: {result['statistics']['percentiles']['p95']:.6f}")
        print(f"    P99: {result['statistics']['percentiles']['p99']:.6f}")
        print(f"  Decimal Places: {result['statistics']['min_decimal_places']} to {result['statistics']['max_decimal_places']}")
        print(f"\nPrecision Analysis:")
        print(f"  Current Type: {result['current_type']}")
        print(f"  Recommended Type: {result['recommended_type']}")
        print(f"  Justification: {result['justification']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"\nSample Values (transformed):")
        for val in result['statistics']['sample_values'][:5]:
            print(f"    {val:.6f}")
        print("-" * 80)


if __name__ == "__main__":
    main()
