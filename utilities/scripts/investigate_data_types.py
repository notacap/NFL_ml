import sys
from datetime import datetime
from collections import defaultdict
import re

# Import common database utilities
sys.path.insert(0, '..')
from common_utils import connect_to_database, get_all_tables, get_table_schema

def parse_column_type(column_type):
    """Parse column type to extract base type, precision, and scale."""
    # Match patterns like DECIMAL(5,2), INT(11), VARCHAR(255), etc.
    match = re.match(r'(\w+)(?:\((\d+)(?:,(\d+))?\))?', column_type)
    if match:
        base_type = match.group(1).upper()
        precision = int(match.group(2)) if match.group(2) else None
        scale = int(match.group(3)) if match.group(3) else None
        return base_type, precision, scale
    return column_type.upper(), None, None

def get_column_statistics(cursor, table_name, column_name, data_type):
    """Get statistical information about a column's data."""
    stats = {}

    try:
        # Check if column is numeric
        numeric_types = ['INT', 'TINYINT', 'SMALLINT', 'MEDIUMINT', 'BIGINT',
                        'DECIMAL', 'FLOAT', 'DOUBLE', 'NUMERIC']
        base_type = parse_column_type(data_type)[0]

        if base_type in numeric_types:
            # Get min, max, avg for numeric columns
            query = f"""
                SELECT
                    MIN({column_name}) as min_val,
                    MAX({column_name}) as max_val,
                    AVG({column_name}) as avg_val,
                    COUNT(DISTINCT {column_name}) as distinct_count,
                    COUNT(*) as total_count
                FROM {table_name}
                WHERE {column_name} IS NOT NULL
            """
            cursor.execute(query)
            result = cursor.fetchone()

            if result and result[0] is not None:
                stats['min'] = float(result[0]) if result[0] is not None else None
                stats['max'] = float(result[1]) if result[1] is not None else None
                stats['avg'] = float(result[2]) if result[2] is not None else None
                stats['distinct_count'] = result[3]
                stats['total_count'] = result[4]

                # Check if values appear to be percentages
                if stats['min'] is not None and stats['max'] is not None:
                    if 0 <= stats['min'] <= 1 and 0 <= stats['max'] <= 1:
                        stats['likely_percentage_0_1'] = True
                    elif 0 <= stats['min'] <= 100 and 0 <= stats['max'] <= 100:
                        stats['likely_percentage_0_100'] = True
        else:
            # For non-numeric columns, get distinct count
            query = f"""
                SELECT
                    COUNT(DISTINCT {column_name}) as distinct_count,
                    COUNT(*) as total_count
                FROM {table_name}
                WHERE {column_name} IS NOT NULL
            """
            cursor.execute(query)
            result = cursor.fetchone()
            if result:
                stats['distinct_count'] = result[0]
                stats['total_count'] = result[1]

    except Exception as e:
        stats['error'] = str(e)

    return stats

def normalize_column_name(column_name):
    """Normalize column name for comparison (remove common prefixes/suffixes)."""
    # Convert to lowercase and remove common variations
    normalized = column_name.lower()
    # Remove trailing numbers/IDs
    normalized = re.sub(r'_id$', '', normalized)
    normalized = re.sub(r'_\d+$', '', normalized)
    return normalized

def identify_similar_columns(all_columns_data):
    """Group columns with similar names across different tables."""
    similar_columns = defaultdict(list)

    for table_name, columns in all_columns_data.items():
        for col in columns:
            # Use the normalized name as key
            normalized = normalize_column_name(col['name'])
            similar_columns[normalized].append({
                'table': table_name,
                'column': col['name'],
                'type': col['type'],
                'base_type': col['base_type'],
                'precision': col['precision'],
                'scale': col['scale'],
                'stats': col['stats']
            })

    return similar_columns

def find_inconsistencies(similar_columns):
    """Find inconsistencies in similar columns across tables."""
    inconsistencies = []

    for normalized_name, occurrences in similar_columns.items():
        if len(occurrences) < 2:
            continue

        # Group by base type
        type_groups = defaultdict(list)
        for occ in occurrences:
            type_groups[occ['base_type']].append(occ)

        # Check for type inconsistencies
        if len(type_groups) > 1:
            inconsistencies.append({
                'type': 'data_type_mismatch',
                'normalized_name': normalized_name,
                'occurrences': occurrences,
                'severity': 'HIGH'
            })

        # Check for precision/scale inconsistencies in decimal types
        decimal_types = ['DECIMAL', 'NUMERIC', 'FLOAT', 'DOUBLE']
        for base_type, group in type_groups.items():
            if base_type in decimal_types and len(group) > 1:
                precisions = set((occ['precision'], occ['scale']) for occ in group)
                if len(precisions) > 1:
                    inconsistencies.append({
                        'type': 'precision_scale_mismatch',
                        'normalized_name': normalized_name,
                        'occurrences': group,
                        'severity': 'MEDIUM'
                    })

        # Check for percentage range inconsistencies
        pct_0_1 = []
        pct_0_100 = []
        for occ in occurrences:
            stats = occ.get('stats', {})
            if stats.get('likely_percentage_0_1'):
                pct_0_1.append(occ)
            elif stats.get('likely_percentage_0_100'):
                pct_0_100.append(occ)

        if pct_0_1 and pct_0_100:
            inconsistencies.append({
                'type': 'percentage_range_mismatch',
                'normalized_name': normalized_name,
                'pct_0_1': pct_0_1,
                'pct_0_100': pct_0_100,
                'severity': 'HIGH'
            })

    return inconsistencies

def check_table_specific_issues(all_columns_data):
    """Check for issues within individual tables."""
    issues = []

    for table_name, columns in all_columns_data.items():
        # Look for potential percentage columns with different precisions
        pct_columns = []
        for col in columns:
            col_name_lower = col['name'].lower()
            if any(keyword in col_name_lower for keyword in ['pct', 'percent', 'rate', 'ratio']):
                pct_columns.append(col)

        # Check if percentage columns have different precisions
        if len(pct_columns) > 1:
            precisions = set((c['precision'], c['scale']) for c in pct_columns if c['precision'])
            if len(precisions) > 1:
                issues.append({
                    'type': 'table_percentage_precision_mismatch',
                    'table': table_name,
                    'columns': pct_columns,
                    'severity': 'MEDIUM'
                })

        # Check for mixed integer sizes for similar metrics
        int_columns = [c for c in columns if c['base_type'] in ['INT', 'TINYINT', 'SMALLINT', 'MEDIUMINT', 'BIGINT']]
        if len(int_columns) > 1:
            # Group by similar naming patterns
            count_cols = [c for c in int_columns if 'count' in c['name'].lower() or 'cnt' in c['name'].lower()]
            if len(count_cols) > 1:
                types = set(c['base_type'] for c in count_cols)
                if len(types) > 1:
                    issues.append({
                        'type': 'table_integer_type_mismatch',
                        'table': table_name,
                        'metric_type': 'count',
                        'columns': count_cols,
                        'severity': 'LOW'
                    })

    return issues

def analyze_data_types():
    """Main function to analyze data type consistency across the database."""
    connection = connect_to_database()
    cursor = connection.cursor()

    try:
        tables = get_all_tables(cursor)
        print(f"\nAnalyzing {len(tables)} tables for data type consistency...")
        print("=" * 80)

        # Collect all column data
        all_columns_data = {}

        for table_name in tables:
            print(f"Analyzing table: {table_name}")
            schema = get_table_schema(cursor, table_name)

            columns_info = []
            for col in schema:
                base_type, precision, scale = parse_column_type(col['type'])

                # Get statistics for the column
                stats = get_column_statistics(cursor, table_name, col['name'], col['type'])

                columns_info.append({
                    'name': col['name'],
                    'type': col['type'],
                    'base_type': base_type,
                    'precision': precision,
                    'scale': scale,
                    'null': col['null'],
                    'stats': stats
                })

            all_columns_data[table_name] = columns_info

        # Identify similar columns across tables
        print("\nIdentifying similar columns across tables...")
        similar_columns = identify_similar_columns(all_columns_data)

        # Find inconsistencies
        print("Finding data type inconsistencies...")
        inconsistencies = find_inconsistencies(similar_columns)

        # Check table-specific issues
        print("Checking table-specific issues...")
        table_issues = check_table_specific_issues(all_columns_data)

        # Generate report
        generate_report(all_columns_data, similar_columns, inconsistencies, table_issues)

        print("\nAnalysis complete! Results written to '../logs/data_type_analysis_report.log'")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()
        connection.close()

def generate_report(all_columns_data, similar_columns, inconsistencies, table_issues):
    """Generate a detailed report of data type analysis."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open('../logs/data_type_analysis_report.log', 'w') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("NFL DATABASE DATA TYPE CONSISTENCY ANALYSIS\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("=" * 80 + "\n\n")

        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total tables analyzed: {len(all_columns_data)}\n")
        f.write(f"Total columns analyzed: {sum(len(cols) for cols in all_columns_data.values())}\n")
        f.write(f"Cross-table inconsistencies found: {len(inconsistencies)}\n")
        f.write(f"Table-specific issues found: {len(table_issues)}\n\n")

        # HIGH severity issues summary
        high_severity = [i for i in inconsistencies if i['severity'] == 'HIGH']
        if high_severity:
            f.write(f"⚠️  HIGH SEVERITY ISSUES: {len(high_severity)}\n")
            f.write("   These should be addressed before training your ML model.\n\n")

        f.write("=" * 80 + "\n\n")

        # Section 1: Critical Inconsistencies
        if inconsistencies:
            f.write("SECTION 1: CROSS-TABLE INCONSISTENCIES\n")
            f.write("-" * 40 + "\n\n")

            # Sort by severity
            severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
            sorted_inconsistencies = sorted(inconsistencies, key=lambda x: severity_order[x['severity']])

            for issue in sorted_inconsistencies:
                f.write(f"[{issue['severity']}] {issue['type'].replace('_', ' ').title()}\n")
                f.write(f"Column pattern: {issue['normalized_name']}\n")

                if issue['type'] == 'data_type_mismatch':
                    f.write("\nDifferent data types found:\n")
                    for occ in issue['occurrences']:
                        f.write(f"  • {occ['table']}.{occ['column']}: {occ['type']}\n")
                        if occ['stats']:
                            if 'min' in occ['stats']:
                                f.write(f"    Range: {occ['stats'].get('min', 'N/A')} to {occ['stats'].get('max', 'N/A')}\n")
                    f.write("\nRecommendation: Standardize to a single data type across all tables.\n")

                elif issue['type'] == 'precision_scale_mismatch':
                    f.write("\nDifferent precision/scale found:\n")
                    for occ in issue['occurrences']:
                        precision_info = f"({occ['precision']},{occ['scale']})" if occ['scale'] else f"({occ['precision']})"
                        f.write(f"  • {occ['table']}.{occ['column']}: {occ['base_type']}{precision_info}\n")
                        if occ['stats'] and 'avg' in occ['stats']:
                            f.write(f"    Avg value: {occ['stats']['avg']:.6f}\n")
                    f.write("\nRecommendation: Use consistent decimal precision for similar metrics.\n")

                elif issue['type'] == 'percentage_range_mismatch':
                    f.write("\nPercentage range inconsistency detected:\n")
                    f.write("  Columns using 0-1 scale:\n")
                    for occ in issue['pct_0_1']:
                        f.write(f"    • {occ['table']}.{occ['column']}\n")
                    f.write("  Columns using 0-100 scale:\n")
                    for occ in issue['pct_0_100']:
                        f.write(f"    • {occ['table']}.{occ['column']}\n")
                    f.write("\nRecommendation: Normalize all percentages to same scale (0-1 recommended for ML).\n")

                f.write("\n" + "-" * 40 + "\n\n")

        # Section 2: Table-Specific Issues
        if table_issues:
            f.write("\nSECTION 2: TABLE-SPECIFIC ISSUES\n")
            f.write("-" * 40 + "\n\n")

            for issue in table_issues:
                f.write(f"[{issue['severity']}] {issue['type'].replace('_', ' ').title()}\n")
                f.write(f"Table: {issue['table']}\n")

                if 'columns' in issue:
                    f.write("Affected columns:\n")
                    for col in issue['columns']:
                        precision_info = f"({col['precision']},{col['scale']})" if col['scale'] else (f"({col['precision']})" if col['precision'] else "")
                        f.write(f"  • {col['name']}: {col['base_type']}{precision_info}\n")

                f.write("\n" + "-" * 40 + "\n\n")

        # Section 3: Column Groupings (for reference)
        f.write("\nSECTION 3: SIMILAR COLUMNS ACROSS TABLES\n")
        f.write("-" * 40 + "\n")
        f.write("This section groups columns with similar names for your reference.\n\n")

        # Only show groups with multiple occurrences
        multi_occurrence = {k: v for k, v in similar_columns.items() if len(v) > 1}
        for normalized_name in sorted(multi_occurrence.keys()):
            occurrences = multi_occurrence[normalized_name]
            f.write(f"\nColumn pattern: '{normalized_name}' ({len(occurrences)} occurrences)\n")
            for occ in occurrences:
                f.write(f"  • {occ['table']}.{occ['column']}: {occ['type']}\n")

        f.write("\n" + "=" * 80 + "\n\n")

        # Section 4: ML-Specific Recommendations
        f.write("SECTION 4: MACHINE LEARNING RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n\n")

        f.write("For optimal ML model performance, consider:\n\n")

        f.write("1. DECIMAL PRECISION:\n")
        f.write("   • Use consistent decimal places for similar metrics\n")
        f.write("   • Common standards: percentages (4-5 decimals), rates (3-4 decimals)\n\n")

        f.write("2. PERCENTAGE SCALING:\n")
        f.write("   • Standardize all percentages to 0-1 scale (easier for normalization)\n")
        f.write("   • Convert any 0-100 percentages during feature engineering\n\n")

        f.write("3. INTEGER TYPES:\n")
        f.write("   • Use appropriate sizes (TINYINT for 0-255, INT for larger ranges)\n")
        f.write("   • Consistency matters less than correct range coverage\n\n")

        f.write("4. NULL HANDLING:\n")
        f.write("   • Document which columns allow NULLs\n")
        f.write("   • Plan imputation strategy for missing values\n\n")

        f.write("5. FEATURE SCALING:\n")
        f.write("   • Different column types will need different scaling approaches\n")
        f.write("   • Note which columns need normalization vs standardization\n\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

if __name__ == "__main__":
    print("Starting NFL Database Data Type Consistency Analysis...")
    print("This script will identify data type inconsistencies that may affect ML model performance")
    print("-" * 80)

    analyze_data_types()
