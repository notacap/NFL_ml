import os
import sys
import pymysql
from dotenv import load_dotenv
from datetime import datetime
from collections import defaultdict

# Load environment variables from .env file
env_path = r'C:\Users\nocap\Desktop\code\NFL_ml\database\.env'
load_dotenv(env_path)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT')),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

def connect_to_database():
    """Establish connection to the MySQL database."""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        print(f"Successfully connected to database: {DB_CONFIG['database']}")
        return connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

def get_all_tables(cursor):
    """Get all table names from the database."""
    cursor.execute("SHOW TABLES")
    tables = [table[0] for table in cursor.fetchall()]
    return tables

def get_seasons(cursor):
    """Get all seasons from the nfl_season table."""
    cursor.execute("SELECT season_id, year FROM nfl_season ORDER BY year")
    seasons = cursor.fetchall()
    return seasons

def get_table_columns(cursor, table_name):
    """Get all column names for a given table."""
    cursor.execute(f"SHOW COLUMNS FROM {table_name}")
    columns = [column[0] for column in cursor.fetchall()]
    return columns

def check_nulls_for_season(cursor, table_name, columns, season_id=None):
    """Check which columns contain null values for a specific season or all data."""
    columns_with_nulls = []
    
    for column in columns:
        try:
            # Build query based on whether table has season_id
            if season_id and 'season_id' in columns:
                query = f"""
                    SELECT COUNT(*) 
                    FROM {table_name} 
                    WHERE {column} IS NULL 
                    AND season_id = %s
                """
                cursor.execute(query, (season_id,))
            else:
                # For tables without season_id, check all records
                query = f"""
                    SELECT COUNT(*) 
                    FROM {table_name} 
                    WHERE {column} IS NULL
                """
                cursor.execute(query)
            
            null_count = cursor.fetchone()[0]
            
            if null_count > 0:
                columns_with_nulls.append({
                    'column': column,
                    'null_count': null_count
                })
        except Exception as e:
            print(f"Error checking column {column} in table {table_name}: {e}")
    
    return columns_with_nulls

def analyze_database_nulls():
    """Main function to analyze null values across all tables and seasons."""
    connection = connect_to_database()
    cursor = connection.cursor()
    
    try:
        # Get all tables and seasons
        tables = get_all_tables(cursor)
        seasons = get_seasons(cursor)
        
        # Dictionary to store results organized by season
        results_by_season = defaultdict(lambda: defaultdict(list))
        
        # Tables without season_id (global tables)
        global_tables_results = defaultdict(list)
        
        print(f"\nAnalyzing {len(tables)} tables across {len(seasons)} seasons...")
        print("=" * 80)
        
        # Process each table
        for table_name in tables:
            print(f"Processing table: {table_name}")
            columns = get_table_columns(cursor, table_name)
            
            if 'season_id' in columns:
                # Table has season_id - check nulls per season
                for season_id, year in seasons:
                    nulls_found = check_nulls_for_season(cursor, table_name, columns, season_id)
                    if nulls_found:
                        results_by_season[year][table_name] = nulls_found
            else:
                # Table doesn't have season_id - check nulls globally
                nulls_found = check_nulls_for_season(cursor, table_name, columns)
                if nulls_found:
                    global_tables_results[table_name] = nulls_found
        
        # Generate log file
        generate_log_file(results_by_season, global_tables_results)
        
        print("\nAnalysis complete! Results written to 'null_columns_report.log'")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
    finally:
        cursor.close()
        connection.close()

def generate_log_file(results_by_season, global_tables_results):
    """Generate a structured log file with the null analysis results."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open('null_columns_report.log', 'w') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("NFL DATABASE NULL VALUES ANALYSIS REPORT\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write global tables (without season_id) first if any
        if global_tables_results:
            f.write("GLOBAL TABLES (No Season ID)\n")
            f.write("-" * 40 + "\n\n")
            
            for table_name, columns_with_nulls in sorted(global_tables_results.items()):
                f.write(f"Table: {table_name}\n")
                f.write("Columns with null values:\n")
                for col_info in columns_with_nulls:
                    f.write(f"  - {col_info['column']} (null count: {col_info['null_count']})\n")
                f.write("\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
        
        # Write season-specific results
        f.write("SEASON-SPECIFIC TABLES\n")
        f.write("-" * 40 + "\n\n")
        
        # Sort seasons chronologically
        for year in sorted(results_by_season.keys()):
            tables_with_nulls = results_by_season[year]
            
            if tables_with_nulls:
                f.write(f"{'=' * 20} SEASON {year} {'=' * 20}\n\n")
                
                # Sort tables alphabetically within each season
                for table_name in sorted(tables_with_nulls.keys()):
                    columns_with_nulls = tables_with_nulls[table_name]
                    
                    f.write(f"Table: {table_name}\n")
                    f.write("Columns with null values:\n")
                    
                    # Sort columns alphabetically
                    sorted_columns = sorted(columns_with_nulls, key=lambda x: x['column'])
                    for col_info in sorted_columns:
                        f.write(f"  - {col_info['column']} (null count: {col_info['null_count']})\n")
                    
                    f.write("\n")
                
                f.write("\n")
        
        # Write summary statistics
        f.write("=" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 40 + "\n\n")
        
        # Count total tables and columns with nulls
        total_tables_with_nulls = len(global_tables_results)
        total_columns_with_nulls = sum(len(cols) for cols in global_tables_results.values())
        
        for year, tables in results_by_season.items():
            total_tables_with_nulls += len(tables)
            for columns in tables.values():
                total_columns_with_nulls += len(columns)
        
        f.write(f"Total tables with null values: {total_tables_with_nulls}\n")
        f.write(f"Total columns with null values: {total_columns_with_nulls}\n")
        f.write(f"Seasons analyzed: {len(results_by_season)}\n")
        
        if global_tables_results:
            f.write(f"Global tables with nulls: {len(global_tables_results)}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

if __name__ == "__main__":
    print("Starting NFL Database Null Values Analysis...")
    print("This script will identify all columns containing null values")
    print("Results will be organized by season and saved to 'null_columns_report.log'")
    print("-" * 80)
    
    analyze_database_nulls()