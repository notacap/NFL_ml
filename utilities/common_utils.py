import os
import sys
import pymysql
from dotenv import load_dotenv

# Load environment variables from .env file
ENV_PATH = r'C:\Users\nocap\Desktop\code\NFL_ml\database\.env'
load_dotenv(ENV_PATH)

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

def get_table_columns(cursor, table_name):
    """Get all column names for a given table."""
    cursor.execute(f"SHOW COLUMNS FROM {table_name}")
    columns = [column[0] for column in cursor.fetchall()]
    return columns

def get_table_schema(cursor, table_name):
    """Get detailed schema information for a table."""
    cursor.execute(f"DESCRIBE {table_name}")
    columns = []
    for row in cursor.fetchall():
        columns.append({
            'name': row[0],
            'type': row[1],
            'null': row[2],
            'key': row[3],
            'default': row[4],
            'extra': row[5]
        })
    return columns
