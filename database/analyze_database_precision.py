"""
Database Precision Analysis Script
Analyzes actual database data to assess precision standardization impact
"""

import json
from db_utils import DatabaseConnector
from typing import Dict, List, Any
import decimal

def analyze_column_precision(db: DatabaseConnector, table_name: str, column_name: str,
                             current_type: str, recommended_type: str) -> Dict[str, Any]:
    """
    Analyze a single column's precision characteristics in the database.

    Returns detailed statistics about data range, precision usage, and migration risk.
    """

    analysis = {
        "db_table": table_name,
        "db_column": column_name,
        "current_type": current_type,
        "recommended_type": recommended_type,
        "existing_value_range": [None, None],
        "max_decimal_places_used": 0,
        "row_count": 0,
        "null_count": 0,
        "distinct_count": 0,
        "has_index": False,
        "has_constraints": False,
        "dependent_objects": [],
        "overflow_risk": "NONE",
        "migration_risk": "LOW",
        "notes": ""
    }

    # Get basic statistics
    try:
        stats_query = f"""
        SELECT
            COUNT(*) as total_rows,
            COUNT({column_name}) as non_null_count,
            COUNT(*) - COUNT({column_name}) as null_count,
            COUNT(DISTINCT {column_name}) as distinct_count,
            MIN({column_name}) as min_val,
            MAX({column_name}) as max_val,
            AVG({column_name}) as avg_val
        FROM {table_name}
        """

        result = db.fetch_all(stats_query)

        if result and result[0]:
            total_rows, non_null, null_count, distinct_count, min_val, max_val, avg_val = result[0]

            analysis["row_count"] = total_rows
            analysis["null_count"] = null_count
            analysis["distinct_count"] = distinct_count
            analysis["existing_value_range"] = [
                float(min_val) if min_val is not None else None,
                float(max_val) if max_val is not None else None
            ]

            # Check for DECIMAL(5,4) overflow risk (values >= 1.0 or <= -1.0)
            if recommended_type == "DECIMAL(5,4)":
                if max_val and float(max_val) >= 1.0:
                    analysis["overflow_risk"] = "HIGH"
                    analysis["notes"] += f"Max value {max_val} exceeds DECIMAL(5,4) range. "
                elif min_val and float(min_val) <= -1.0:
                    analysis["overflow_risk"] = "HIGH"
                    analysis["notes"] += f"Min value {min_val} below DECIMAL(5,4) range. "

            # Sample values to determine actual decimal precision used
            sample_query = f"""
            SELECT DISTINCT {column_name}
            FROM {table_name}
            WHERE {column_name} IS NOT NULL
            LIMIT 100
            """

            samples = db.fetch_all(sample_query)
            max_decimals = 0

            for row in samples:
                if row[0] is not None:
                    val_str = str(row[0])
                    if '.' in val_str:
                        decimals = len(val_str.split('.')[1].rstrip('0'))
                        max_decimals = max(max_decimals, decimals)

            analysis["max_decimal_places_used"] = max_decimals

    except Exception as e:
        analysis["notes"] += f"Error analyzing data: {str(e)}. "

    # Check for indexes on this column
    try:
        index_query = f"""
        SELECT DISTINCT INDEX_NAME, NON_UNIQUE
        FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = %s
        AND TABLE_NAME = %s
        AND COLUMN_NAME = %s
        """

        indexes = db.fetch_all(index_query, (db.config['database'], table_name, column_name))

        if indexes:
            analysis["has_index"] = True
            index_names = [idx[0] for idx in indexes]
            analysis["dependent_objects"].extend([f"Index: {name}" for name in index_names])
            analysis["notes"] += f"Column has {len(indexes)} index(es). "

    except Exception as e:
        analysis["notes"] += f"Error checking indexes: {str(e)}. "

    # Check for foreign key constraints
    try:
        fk_query = f"""
        SELECT CONSTRAINT_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
        FROM information_schema.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = %s
        AND TABLE_NAME = %s
        AND COLUMN_NAME = %s
        AND REFERENCED_TABLE_NAME IS NOT NULL
        """

        fks = db.fetch_all(fk_query, (db.config['database'], table_name, column_name))

        if fks:
            analysis["has_constraints"] = True
            for fk in fks:
                analysis["dependent_objects"].append(f"FK: {fk[0]} -> {fk[1]}.{fk[2]}")
            analysis["notes"] += f"Column has {len(fks)} foreign key(s). "

    except Exception as e:
        analysis["notes"] += f"Error checking constraints: {str(e)}. "

    # Check for views that reference this column
    try:
        view_query = f"""
        SELECT TABLE_NAME
        FROM information_schema.VIEWS
        WHERE TABLE_SCHEMA = %s
        AND VIEW_DEFINITION LIKE %s
        """

        views = db.fetch_all(view_query, (db.config['database'], f"%{column_name}%"))

        if views:
            view_names = [v[0] for v in views]
            analysis["dependent_objects"].extend([f"View: {name}" for name in view_names])
            analysis["notes"] += f"Referenced in {len(views)} view(s). "

    except Exception as e:
        analysis["notes"] += f"Error checking views: {str(e)}. "

    # Assess migration risk based on multiple factors
    risk_factors = []

    if analysis["row_count"] > 10000:
        risk_factors.append("Large table (>10K rows)")

    if analysis["has_index"]:
        risk_factors.append("Indexed column")

    if analysis["has_constraints"]:
        risk_factors.append("Has constraints")

    if len(analysis["dependent_objects"]) > 2:
        risk_factors.append("Multiple dependencies")

    if analysis["overflow_risk"] == "HIGH":
        risk_factors.append("OVERFLOW RISK")
        analysis["migration_risk"] = "HIGH"
    elif len(risk_factors) >= 3:
        analysis["migration_risk"] = "MEDIUM"
    elif len(risk_factors) >= 1:
        analysis["migration_risk"] = "LOW"
    else:
        analysis["migration_risk"] = "LOW"

    if risk_factors:
        analysis["notes"] += f"Risk factors: {', '.join(risk_factors)}. "

    # Add recommendation notes
    if current_type != recommended_type:
        analysis["notes"] += f"Type change from {current_type} to {recommended_type} recommended. "
    else:
        analysis["notes"] += "Already using recommended type. "

    return analysis


def main():
    """Main analysis function"""

    # Connect to database
    db = DatabaseConnector()
    if not db.connect():
        print("Failed to connect to database")
        return

    print("="*80)
    print("DATABASE PRECISION ANALYSIS")
    print("="*80)
    print(f"Connected to: {db.config['database']}")
    print()

    # Define columns to analyze based on agent prompt
    columns_to_analyze = [
        # plyr_gm_pass (8 columns)
        ("plyr_gm_pass", "plyr_gm_pass_first_dwn_pct", "FLOAT(5,4)", "DECIMAL(5,4)"),
        ("plyr_gm_pass", "plyr_gm_pass_drp_pct", "FLOAT(5,4)", "DECIMAL(5,4)"),
        ("plyr_gm_pass", "plyr_gm_pass_off_tgt_pct", "FLOAT(5,4)", "DECIMAL(5,4)"),
        ("plyr_gm_pass", "plyr_gm_pass_prss_pct", "FLOAT(5,4)", "DECIMAL(5,4)"),
        ("plyr_gm_pass", "plyr_gm_pass_cmp_pct", "DECIMAL(7,4)", "DECIMAL(5,4)"),
        ("plyr_gm_pass", "plyr_gm_pass_td_pct", "DECIMAL(7,4)", "DECIMAL(5,4)"),
        ("plyr_gm_pass", "plyr_gm_pass_int_pct", "DECIMAL(7,4)", "DECIMAL(5,4)"),
        ("plyr_gm_pass", "plyr_gm_pass_sk_pct", "DECIMAL(7,4)", "DECIMAL(5,4)"),

        # plyr_pass (5 columns)
        ("plyr_pass", "plyr_pass_cmp_pct", "DECIMAL(5,4)", "DECIMAL(5,4)"),
        ("plyr_pass", "plyr_pass_td_pct", "DECIMAL(7,4)", "DECIMAL(5,4)"),
        ("plyr_pass", "plyr_pass_int_pct", "DECIMAL(5,4)", "DECIMAL(5,4)"),
        ("plyr_pass", "plyr_pass_sk_pct", "DECIMAL(5,4)", "DECIMAL(5,4)"),
        ("plyr_pass", "plyr_pass_prss_pct", "DECIMAL(4,3)", "DECIMAL(5,4)"),

        # tm_def_pass (4 columns)
        ("tm_def_pass", "tm_def_pass_cmp_pct", "DECIMAL(7,4)", "DECIMAL(5,4)"),
        ("tm_def_pass", "tm_def_pass_td_pct", "DECIMAL(5,4)", "DECIMAL(5,4)"),
        ("tm_def_pass", "tm_def_int_pct", "DECIMAL(5,4)", "DECIMAL(5,4)"),
        ("tm_def_pass", "tm_def_sk_pct", "DECIMAL(5,4)", "DECIMAL(5,4)"),
    ]

    all_results = []

    # Analyze each column
    for table_name, column_name, current_type, recommended_type in columns_to_analyze:
        print(f"Analyzing: {table_name}.{column_name}")

        result = analyze_column_precision(
            db, table_name, column_name, current_type, recommended_type
        )

        all_results.append(result)

        # Print summary
        print(f"  Rows: {result['row_count']:,} | Nulls: {result['null_count']:,} | Distinct: {result['distinct_count']:,}")
        print(f"  Range: [{result['existing_value_range'][0]}, {result['existing_value_range'][1]}]")
        print(f"  Max Decimals Used: {result['max_decimal_places_used']}")
        print(f"  Overflow Risk: {result['overflow_risk']} | Migration Risk: {result['migration_risk']}")
        print(f"  Dependencies: {len(result['dependent_objects'])}")
        print()

    # Generate summary statistics
    summary = {
        "total_tables_analyzed": len(set(r["db_table"] for r in all_results)),
        "total_columns_analyzed": len(all_results),
        "safe_migrations": sum(1 for r in all_results if r["migration_risk"] == "LOW" and r["overflow_risk"] == "NONE"),
        "risky_migrations": sum(1 for r in all_results if r["migration_risk"] in ["MEDIUM", "HIGH"] or r["overflow_risk"] == "HIGH"),
        "total_affected_rows": sum(r["row_count"] for r in all_results),
        "overflow_columns": [r["db_column"] for r in all_results if r["overflow_risk"] == "HIGH"]
    }

    # Create final output structure
    output = {
        "analysis": all_results,
        "summary": summary
    }

    # Save to JSON file
    output_path = "C:\\Users\\nocap\\Desktop\\code\\NFL_ml\\database\\database_precision_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total Tables Analyzed: {summary['total_tables_analyzed']}")
    print(f"Total Columns Analyzed: {summary['total_columns_analyzed']}")
    print(f"Safe Migrations: {summary['safe_migrations']}")
    print(f"Risky Migrations: {summary['risky_migrations']}")
    print(f"Total Affected Rows: {summary['total_affected_rows']:,}")

    if summary['overflow_columns']:
        print(f"\nWARNING: Columns with overflow risk:")
        for col in summary['overflow_columns']:
            print(f"  - {col}")

    print(f"\nFull results saved to: {output_path}")
    print("="*80)

    # Disconnect from database
    db.disconnect()


if __name__ == "__main__":
    main()
