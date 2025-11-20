import mysql.connector
import pandas as pd
import os
from dotenv import load_dotenv
import logging
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Configuration: Set the year and week for data processing
YEAR = 2022
WEEK = 1
WEEK_START = 1
WEEK_END = 1

# Load environment variables
load_dotenv()

class DatabaseConnector:
    """Utility class for handling MySQL database connections and operations"""
    
    def __init__(self):
        self.connection = None
        self.cursor = None
        self.config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'database': os.getenv('DB_NAME', 'nfl_stats'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'autocommit': True
        }
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(**self.config)
            self.cursor = self.connection.cursor()
            self.logger.info("Database connection established")
            return True
        except mysql.connector.Error as e:
            self.logger.error(f"Error connecting to database: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        self.logger.info("Database connection closed")
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> bool:
        """Execute a single query"""
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            return True
        except mysql.connector.Error as e:
            self.logger.error(f"Error executing query: {e}")
            return False
    
    def execute_many(self, query: str, data: List[tuple]) -> tuple:
        """Execute query with multiple parameter sets. Returns (success, rows_affected)"""
        try:
            self.cursor.executemany(query, data)
            rows_affected = self.cursor.rowcount
            self.logger.info(f"Processed {len(data)} rows, {rows_affected} rows affected")
            return True, rows_affected
        except mysql.connector.Error as e:
            self.logger.error(f"Error executing batch operation: {e}")
            return False, 0
    
    def fetch_all(self, query: str, params: Optional[tuple] = None) -> List[tuple]:
        """Fetch all results from a query"""
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            return self.cursor.fetchall()
        except mysql.connector.Error as e:
            self.logger.error(f"Error fetching data: {e}")
            return []
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists"""
        query = """
        SELECT COUNT(*)
        FROM information_schema.tables 
        WHERE table_schema = %s AND table_name = %s
        """
        result = self.fetch_all(query, (self.config['database'], table_name))
        return result[0][0] > 0 if result else False

def load_csv_data(file_path: str) -> pd.DataFrame:
    """Load CSV data into pandas DataFrame"""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading CSV file {file_path}: {e}")
        return pd.DataFrame()

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names for database compatibility"""
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('[^a-z0-9_]', '', regex=True)
    return df

def handle_null_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle null values in DataFrame"""
    # Replace NaN with None for database insertion
    df = df.where(pd.notnull(df), None)
    return df

def create_insert_query(table_name: str, columns: List[str]) -> str:
    """Generate INSERT query for given table and columns"""
    placeholders = ', '.join(['%s'] * len(columns))
    columns_str = ', '.join(columns)
    query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
    return query

def create_upsert_query(table_name: str, columns: List[str]) -> str:
    """Generate INSERT ... ON DUPLICATE KEY UPDATE query for given table and columns"""
    placeholders = ', '.join(['%s'] * len(columns))
    columns_str = ', '.join(columns)
    
    # Create UPDATE part - exclude auto increment primary key columns
    update_parts = []
    for col in columns:
        if not col.endswith('_id') or col in ['team_id', 'season_id', 'plyr_id', 'current_tm_id', 'former_tm_id', 'first_tm_id', 'current_tm_week_id', 'former_tm_lst_wk_id', 'first_tm_lst_wk_id', 'second_tm_week_id']:
            update_parts.append(f"{col} = VALUES({col})")
    
    update_str = ', '.join(update_parts)
    query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {update_str}"
    return query

def batch_insert_data(db: DatabaseConnector, table_name: str, df: pd.DataFrame, batch_size: int = 1000) -> bool:
    """Insert DataFrame data in batches"""
    try:
        columns = df.columns.tolist()
        insert_query = create_insert_query(table_name, columns)
        
        total_rows = len(df)
        inserted_rows = 0
        
        for i in range(0, total_rows, batch_size):
            batch_df = df.iloc[i:i+batch_size]
            # Convert DataFrame to native Python types to avoid numpy type issues
            batch_data = []
            for _, row in batch_df.iterrows():
                row_data = []
                for value in row:
                    if pd.isna(value):
                        row_data.append(None)
                    elif hasattr(value, 'item'):  # numpy scalar
                        row_data.append(value.item())
                    else:
                        row_data.append(value)
                batch_data.append(tuple(row_data))
            
            success, rows_affected = db.execute_many(insert_query, batch_data)
            if success:
                inserted_rows += len(batch_data)
                print(f"Inserted batch {i//batch_size + 1}: {len(batch_data)} rows")
            else:
                print(f"Failed to insert batch {i//batch_size + 1}")
                return False
        
        print(f"Successfully inserted {inserted_rows}/{total_rows} rows into {table_name}")
        return True
        
    except Exception as e:
        print(f"Error in batch insert: {e}")
        return False

def batch_upsert_data_with_logging(db: DatabaseConnector, table_name: str, df: pd.DataFrame, batch_size: int = 1000, debug_updates: bool = False) -> bool:
    """Enhanced INSERT/UPDATE with detailed logging for debugging duplicates.
    
    Args:
        db: Database connector
        table_name: Target table name  
        df: DataFrame to insert/update
        batch_size: Batch size for processing
        debug_updates: If True, provides detailed logging of what records are being updated and why
    
    Returns:
        bool: Success status
    """
    try:
        columns = df.columns.tolist()
        upsert_query = create_upsert_query(table_name, columns)
        
        total_rows = len(df)
        processed_rows = 0
        total_rows_affected = 0
        total_inserted = 0
        total_updated = 0
        
        # For debugging: track sample records that get updated
        updated_samples = []
        
        if debug_updates and table_name == 'plyr_def':
            print(f"\n[DEBUG] Pre-checking for existing records in {table_name}...")
            existing_records = []
            
            # Check each record to see if it already exists
            for _, row in df.iterrows():
                check_query = """
                SELECT plyr_def_id, plyr_id, season_id, week_id, p.plyr_name, t.abrv as team_abrv
                FROM plyr_def pd 
                JOIN plyr p ON pd.plyr_id = p.plyr_id
                JOIN nfl_team t ON p.team_id = t.team_id
                WHERE pd.plyr_id = %s AND pd.season_id = %s AND pd.week_id = %s
                """
                
                existing = db.fetch_all(check_query, (row['plyr_id'], row['season_id'], row['week_id']))
                if existing:
                    existing_records.append({
                        'row_index': len(existing_records),
                        'plyr_id': row['plyr_id'], 
                        'season_id': row['season_id'],
                        'week_id': row['week_id'],
                        'existing_record': existing[0]
                    })
            
            if existing_records:
                print(f"[DEBUG] Found {len(existing_records)} existing records that will be UPDATED:")
                print(f"[DEBUG] {'Index':<6} {'Player ID':<10} {'Season':<7} {'Week':<5} {'Player Name':<25} {'Team':<5}")
                print(f"[DEBUG] {'-'*70}")
                
                for i, record in enumerate(existing_records[:5]):  # Show first 5
                    existing_data = record['existing_record']
                    print(f"[DEBUG] {i+1:<6} {record['plyr_id']:<10} {record['season_id']:<7} {record['week_id']:<5} {existing_data[4]:<25} {existing_data[5]:<5}")
                
                if len(existing_records) > 5:
                    print(f"[DEBUG] ... and {len(existing_records) - 5} more records")
                print(f"[DEBUG] {'-'*70}")
                print(f"[DEBUG] This indicates either:")
                print(f"[DEBUG] 1. Script was run previously for this week/season")
                print(f"[DEBUG] 2. Duplicate plyr_id values in your DataFrame")
                print(f"[DEBUG] 3. Issue with player ID mapping logic")
        
        for i in range(0, total_rows, batch_size):
            batch_df = df.iloc[i:i+batch_size]
            # Convert DataFrame to native Python types to avoid numpy type issues
            batch_data = []
            for _, row in batch_df.iterrows():
                row_data = []
                for value in row:
                    if pd.isna(value):
                        row_data.append(None)
                    elif hasattr(value, 'item'):  # numpy scalar
                        row_data.append(value.item())
                    else:
                        row_data.append(value)
                batch_data.append(tuple(row_data))
            
            success, rows_affected = db.execute_many(upsert_query, batch_data)
            if success:
                processed_rows += len(batch_data)
                total_rows_affected += rows_affected
                
                # More accurate calculation for MySQL UPSERT behavior
                # rows_affected counts: 1 for new insert, 2 for update, 0 for no change
                if rows_affected == len(batch_data):
                    # All were inserts
                    batch_inserted = len(batch_data)
                    batch_updated = 0
                elif rows_affected == len(batch_data) * 2:
                    # All were updates
                    batch_inserted = 0
                    batch_updated = len(batch_data)
                else:
                    # Mix of inserts and updates - approximate based on rows_affected
                    # This is an approximation since MySQL doesn't provide exact counts
                    batch_inserted = max(0, len(batch_data) * 2 - rows_affected)
                    batch_updated = len(batch_data) - batch_inserted
                
                total_inserted += batch_inserted
                total_updated += batch_updated
                
                print(f"Processed batch {i//batch_size + 1}: {len(batch_data)} rows (est. {batch_inserted} new, {batch_updated} updated)")
            else:
                print(f"Failed to process batch {i//batch_size + 1}")
                return False
        
        print(f"Successfully processed {processed_rows}/{total_rows} rows for {table_name}")
        print(f"Estimated results: {total_inserted} rows inserted, {total_updated} rows updated")
        print(f"Total database rows affected: {total_rows_affected}")
        
        if debug_updates and total_updated > 0:
            print(f"\n[DEBUG SUMMARY] {total_updated} records were UPDATED instead of INSERTED")
            print(f"[DEBUG] This suggests the unique constraint uk_plyr_season (plyr_id, season_id, week_id) matched existing records")
            
        return True
        
    except Exception as e:
        print(f"Error in batch upsert: {e}")
        return False


def batch_upsert_data(db: DatabaseConnector, table_name: str, df: pd.DataFrame, batch_size: int = 1000) -> bool:
    """Insert/Update DataFrame data in batches using UPSERT"""
    try:
        columns = df.columns.tolist()
        upsert_query = create_upsert_query(table_name, columns)
        
        total_rows = len(df)
        processed_rows = 0
        total_rows_affected = 0
        total_inserted = 0
        total_updated = 0
        
        for i in range(0, total_rows, batch_size):
            batch_df = df.iloc[i:i+batch_size]
            # Convert DataFrame to native Python types to avoid numpy type issues
            batch_data = []
            for _, row in batch_df.iterrows():
                row_data = []
                for value in row:
                    if pd.isna(value):
                        row_data.append(None)
                    elif hasattr(value, 'item'):  # numpy scalar
                        row_data.append(value.item())
                    else:
                        row_data.append(value)
                batch_data.append(tuple(row_data))
            
            success, rows_affected = db.execute_many(upsert_query, batch_data)
            if success:
                processed_rows += len(batch_data)
                total_rows_affected += rows_affected
                
                # For UPSERT: rows_affected = 1 for INSERT, 2 for UPDATE, 0 for no change
                batch_inserted = sum(1 for j in range(len(batch_data)) if (i + j) < rows_affected and rows_affected % 2 == 1)
                batch_updated = rows_affected - batch_inserted
                
                # More accurate calculation for MySQL UPSERT behavior
                # rows_affected counts: 1 for new insert, 2 for update, 0 for no change
                if rows_affected == len(batch_data):
                    # All were inserts
                    batch_inserted = len(batch_data)
                    batch_updated = 0
                elif rows_affected == len(batch_data) * 2:
                    # All were updates
                    batch_inserted = 0
                    batch_updated = len(batch_data)
                else:
                    # Mix of inserts and updates - approximate based on rows_affected
                    # This is an approximation since MySQL doesn't provide exact counts
                    batch_inserted = max(0, len(batch_data) * 2 - rows_affected)
                    batch_updated = len(batch_data) - batch_inserted
                
                total_inserted += batch_inserted
                total_updated += batch_updated
                
                print(f"Processed batch {i//batch_size + 1}: {len(batch_data)} rows (est. {batch_inserted} new, {batch_updated} updated)")
            else:
                print(f"Failed to process batch {i//batch_size + 1}")
                return False
        
        print(f"Successfully processed {processed_rows}/{total_rows} rows for {table_name}")
        print(f"Estimated results: {total_inserted} rows inserted, {total_updated} rows updated")
        print(f"Total database rows affected: {total_rows_affected}")
        return True
        
    except Exception as e:
        print(f"Error in batch upsert: {e}")
        return False

def get_season_from_filename(filename: str) -> Optional[int]:
    """Extract season year from filename"""
    import re
    match = re.search(r'20\d{2}', filename)
    return int(match.group()) if match else None

def standardize_team_name(team_name: str) -> str:
    """Standardize team names for consistency"""
    team_mapping = {
        'Arizona Cardinals': 'ARI',
        'Atlanta Falcons': 'ATL',
        'Baltimore Ravens': 'BAL',
        'Buffalo Bills': 'BUF',
        'Carolina Panthers': 'CAR',
        'Chicago Bears': 'CHI',
        'Cincinnati Bengals': 'CIN',
        'Cleveland Browns': 'CLE',
        'Dallas Cowboys': 'DAL',
        'Denver Broncos': 'DEN',
        'Detroit Lions': 'DET',
        'Green Bay Packers': 'GNB',
        'Houston Texans': 'HOU',
        'Indianapolis Colts': 'IND',
        'Jacksonville Jaguars': 'JAX',
        'Kansas City Chiefs': 'KAN',
        'Las Vegas Raiders': 'LVR',
        'Los Angeles Chargers': 'LAC',
        'Los Angeles Rams': 'LAR',
        'Miami Dolphins': 'MIA',
        'Minnesota Vikings': 'MIN',
        'New England Patriots': 'NWE',
        'New Orleans Saints': 'NOR',
        'New York Giants': 'NYG',
        'New York Jets': 'NYJ',
        'Philadelphia Eagles': 'PHI',
        'Pittsburgh Steelers': 'PIT',
        'San Francisco 49ers': 'SFO',
        'Seattle Seahawks': 'SEA',
        'Tampa Bay Buccaneers': 'TAM',
        'Tennessee Titans': 'TEN',
        'Washington Commanders': 'WAS'
    }
    
    return team_mapping.get(team_name, team_name)


# ============================================
# FOREIGN KEY LOOKUP UTILITIES
# ============================================
# These functions eliminate duplication across all player game stat insert scripts

def get_season_id(db: DatabaseConnector, year: int) -> int:
    """Get season_id from nfl_season table based on year.
    
    Used by all player game stat insert scripts.
    Raises ValueError if season not found.
    """
    query = "SELECT season_id FROM nfl_season WHERE year = %s"
    result = db.fetch_all(query, (year,))
    if result:
        return result[0][0]
    else:
        raise ValueError(f"No season found for year {year}")


def get_week_id(db: DatabaseConnector, season_id: int, week_num: str) -> int:
    """Get week_id from nfl_week table based on season_id and week_num.
    
    Used by all player game stat insert scripts.
    Raises ValueError if week not found.
    """
    query = "SELECT week_id FROM nfl_week WHERE season_id = %s AND week_num = %s"
    result = db.fetch_all(query, (season_id, str(week_num)))
    if result:
        return result[0][0]
    else:
        raise ValueError(f"No week found for season {season_id}, week {week_num}")


def get_team_id(db: DatabaseConnector, team_name: str) -> int:
    """Get team_id from nfl_team table based on team name, abbreviation, or alt_abrv.
    
    Used by all player game stat insert scripts.
    Tries multiple matching strategies:
    1. Exact team_name match
    2. abrv match
    3. alt_abrv match
    4. Team nickname match (e.g., "Steelers" -> "Pittsburgh Steelers")
    
    Raises ValueError if team not found.
    """
    # First try by team name
    query = "SELECT team_id FROM nfl_team WHERE team_name = %s"
    result = db.fetch_all(query, (team_name,))
    if result:
        return result[0][0]
    
    # If not found, try by abbreviation (in case abbreviation is passed)
    query = "SELECT team_id FROM nfl_team WHERE abrv = %s"
    result = db.fetch_all(query, (team_name,))
    if result:
        return result[0][0]
    
    # If still not found, try by alternative abbreviation
    query = "SELECT team_id FROM nfl_team WHERE alt_abrv = %s"
    result = db.fetch_all(query, (team_name,))
    if result:
        return result[0][0]
    
    # Try by team nickname (partial match)
    query = "SELECT team_id FROM nfl_team WHERE team_name LIKE %s"
    result = db.fetch_all(query, (f"%{team_name}%",))
    if result:
        return result[0][0]
    
    # If still not found, raise error
    raise ValueError(f"No team found for name/abbreviation {team_name}")


def convert_team_nickname_to_abbreviation(db: DatabaseConnector, team_nickname: str) -> str:
    """Convert team nickname (like 'Steelers', 'Falcons') to abbreviation (like 'PIT', 'ATL').
    
    This function is specifically designed for snap count CSV files which contain 
    team nicknames/mascots instead of full team names or abbreviations.
    
    Args:
        db: Database connector instance
        team_nickname: Team nickname from CSV (e.g., 'Steelers', 'Falcons', 'Bears')
        
    Returns:
        str: Team abbreviation (e.g., 'PIT', 'ATL', 'CHI')
        
    Raises:
        ValueError: If team nickname not found
    """
    # Manual mapping for common team nicknames to abbreviations
    # This covers the team nicknames found in snap count CSV files
    nickname_to_abbrev = {
        # AFC North
        'Ravens': 'BAL',
        'Bengals': 'CIN', 
        'Browns': 'CLE',
        'Steelers': 'PIT',
        
        # AFC South
        'Texans': 'HOU',
        'Colts': 'IND',
        'Jaguars': 'JAX',
        'Titans': 'TEN',
        
        # AFC East
        'Bills': 'BUF',
        'Dolphins': 'MIA',
        'Patriots': 'NE',
        'Jets': 'NYJ',
        
        # AFC West
        'Broncos': 'DEN',
        'Chiefs': 'KC',
        'Raiders': 'LV',
        'Chargers': 'LAC',
        
        # NFC North
        'Bears': 'CHI',
        'Lions': 'DET',
        'Packers': 'GB',
        'Vikings': 'MIN',
        
        # NFC South
        'Falcons': 'ATL',
        'Panthers': 'CAR',
        'Saints': 'NO',
        'Buccaneers': 'TB',
        
        # NFC East
        'Cowboys': 'DAL',
        'Giants': 'NYG',
        'Eagles': 'PHI',
        'Commanders': 'WAS',
        
        # NFC West
        'Cardinals': 'ARI',
        'Rams': 'LAR',
        '49ers': 'SF',
        'Seahawks': 'SEA'
    }
    
    # Try direct nickname mapping first
    if team_nickname in nickname_to_abbrev:
        return nickname_to_abbrev[team_nickname]
    
    # If not found in mapping, try database lookup with partial matching
    query = "SELECT abrv FROM nfl_team WHERE team_name LIKE %s"
    result = db.fetch_all(query, (f"%{team_nickname}%",))
    if result:
        return result[0][0]
    
    # If still not found, raise error
    raise ValueError(f"No team abbreviation found for nickname '{team_nickname}'")


def get_game_id(db: DatabaseConnector, season_id: int, week_id: int, team1_abrv: str, team2_abrv: str) -> int:
    """Get game_id based on season_id, week_id and two team abbreviations.
    
    Used by all player game stat insert scripts that need to match games.
    Finds games where both teams played (home or away).
    
    Raises ValueError if game not found.
    """
    # First get team IDs
    team1_id = get_team_id(db, team1_abrv)
    team2_id = get_team_id(db, team2_abrv)
    
    # Find game where both teams are present (home or away)
    query = """
    SELECT game_id 
    FROM nfl_game 
    WHERE season_id = %s 
    AND week_id = %s 
    AND ((home_team_id = %s AND away_team_id = %s) 
         OR (home_team_id = %s AND away_team_id = %s))
    """
    
    result = db.fetch_all(query, (season_id, week_id, team1_id, team2_id, team2_id, team1_id))
    if result:
        return result[0][0]
    else:
        raise ValueError(f"No game found for season {season_id}, week {week_id}, teams {team1_abrv} vs {team2_abrv}")


def get_player_id(db: DatabaseConnector, player_name: str, team_abrv: str, season_id: int, age: int = None, position: str = None, interactive: bool = False) -> int:
    """Get player_id from plyr table with fallback to multi_tm_plyr.

    CRITICAL: Returns plyr.plyr_id (season-specific), NOT player_guid.
    All stat tables reference plyr.plyr_id for foreign key relationships.
    
    Used by all player game stat insert scripts.
    Handles name variations with suffixes and searches plyr table first,
    then falls back to multi_tm_plyr if needed.

    Args:
        db: Database connector instance
        player_name: Player name as it appears in CSV
        team_abrv: Team abbreviation to help with disambiguation
        season_id: Season ID for the lookup
        age: Player age for additional validation (optional)
        position: Player position for additional validation (optional)
        interactive: If True, prompts user to manually select from multiple matches (optional)

    Returns:
        int: player_id from plyr table, or 0 if user chooses to skip (interactive mode only)

    Raises:
        ValueError: If no player found or lookup fails (non-interactive mode)
    """
    # Apply position mapping if position provided
    mapped_position = apply_position_mapping(position) if position else None

    # Generate name variations to handle suffixes
    suffixes = ["II", "III", "IV", "Jr.", "Sr."]
    name_variations = [player_name] + [f"{player_name} {suffix}" for suffix in suffixes]
    placeholders = ', '.join(['%s'] * len(name_variations))

    # Build dynamic WHERE clauses for age and position
    age_clause = " AND p.plyr_age = %s" if age else ""
    pos_clause = " AND p.plyr_pos = %s" if mapped_position else ""

    # STEP 1: Search plyr table first
    if team_abrv and team_abrv.strip():
        query = f"""
        SELECT p.plyr_id, p.plyr_name, p.plyr_guid, p.team_id, t.abrv, p.plyr_pos, p.plyr_age
        FROM plyr p
        JOIN nfl_team t ON p.team_id = t.team_id
        WHERE p.plyr_name IN ({placeholders}) 
          AND (t.abrv = %s OR t.alt_abrv = %s) 
          AND p.season_id = %s{age_clause}{pos_clause}
        """
        params = name_variations + [team_abrv, team_abrv, season_id]
        if age:
            params.append(age)
        if mapped_position:
            params.append(mapped_position)
    else:
        query = f"""
        SELECT p.plyr_id, p.plyr_name, p.plyr_guid, p.team_id, t.abrv, p.plyr_pos, p.plyr_age
        FROM plyr p
        JOIN nfl_team t ON p.team_id = t.team_id
        WHERE p.plyr_name IN ({placeholders}) AND p.season_id = %s{age_clause}{pos_clause}
        """
        params = name_variations + [season_id]
        if age:
            params.append(age)
        if mapped_position:
            params.append(mapped_position)

    results = db.fetch_all(query, params)

    if len(results) == 1:
        return results[0][0]
    elif len(results) > 1:
        if interactive:
            selected_id = interactive_player_selection(player_name, team_abrv, age, position, results)
            if selected_id == 0:
                return 0
            return selected_id
        else:
            print(f"[WARNING] Multiple matches found for {player_name} ({team_abrv}). Using first match.")
            return results[0][0]
    
    # STEP 2: If no match in plyr table, try multi_tm_plyr
    if team_abrv and team_abrv.strip():
        # Get team_id for the provided abbreviation
        try:
            team_id = get_team_id(db, team_abrv)
        except ValueError:
            team_id = None
        
        if team_id:
            multi_tm_query = f"""
            SELECT p.plyr_id, mtp.plyr_name, mtp.plyr_guid, p.team_id, t.abrv, p.plyr_pos, p.plyr_age
            FROM multi_tm_plyr mtp
            JOIN plyr p ON mtp.plyr_id = p.plyr_id
            JOIN nfl_team t ON p.team_id = t.team_id
            WHERE mtp.plyr_name IN ({placeholders})
              AND mtp.season_id = %s
              AND (mtp.tm_1_id = %s OR mtp.tm_2_id = %s OR mtp.tm_3_id = %s){age_clause.replace('p.', 'p.')}{pos_clause.replace('p.', 'p.')}
            """
            multi_tm_params = name_variations + [season_id, team_id, team_id, team_id]
            if age:
                multi_tm_params.append(age)
            if mapped_position:
                multi_tm_params.append(mapped_position)
            
            multi_tm_results = db.fetch_all(multi_tm_query, multi_tm_params)
            
            if len(multi_tm_results) == 1:
                return multi_tm_results[0][0]
            elif len(multi_tm_results) > 1:
                if interactive:
                    selected_id = interactive_player_selection(player_name, team_abrv, age, position, multi_tm_results)
                    if selected_id == 0:
                        return 0
                    return selected_id
                else:
                    print(f"[WARNING] Multiple matches found in multi_tm_plyr for {player_name} ({team_abrv}). Using first match.")
                    return multi_tm_results[0][0]
    
    # STEP 3: Fall back to basic lookup without age/position if no exact match
    if age or position:
        print(f"[INFO] No exact match for {player_name} with age/position. Trying basic lookup...")
        return get_player_id(db, player_name, team_abrv, season_id, age=None, position=None, interactive=interactive)
    else:
        if interactive:
            print(f"[ERROR] No player found for {player_name} ({team_abrv}) in season {season_id}")
            return 0
        else:
            raise ValueError(f"No player found for {player_name} ({team_abrv}) in season {season_id}")


def interactive_player_selection(player_name: str, team_abrv: str, age: int, position: str, matches: list) -> int:
    """Interactive prompt to let user select the correct player from multiple matches.
    
    Args:
        player_name: Player name being searched for
        team_abrv: Team abbreviation from CSV
        age: Player age from CSV
        position: Player position from CSV  
        matches: List of database results [(plyr_id, plyr_name, source, team_id, abrv, plyr_pos, plyr_age), ...]
        
    Returns:
        int: Selected player_id, or 0 to skip this player
    """
    print(f"\n{'='*80}")
    print(f"MULTIPLE PLAYER MATCHES FOUND")
    print(f"{'='*80}")
    print(f"Player being processed: {player_name}")
    print(f"CSV Data - Team: {team_abrv}, Age: {age}, Position: {position}")
    print(f"\nFound {len(matches)} potential matches:")
    print(f"{'#':<3} {'Player ID':<10} {'Name':<25} {'Team':<6} {'Pos':<4} {'Age':<4} {'Source':<12}")
    print(f"{'-'*80}")
    
    for i, match in enumerate(matches):
        plyr_id, plyr_name, source, team_id, abrv, plyr_pos, plyr_age = match
        print(f"{i+1:<3} {plyr_id:<10} {plyr_name:<25} {abrv:<6} {plyr_pos or 'N/A':<4} {plyr_age or 'N/A':<4} {source:<12}")
    
    print(f"{'-'*80}")
    print(f"0   SKIP - Skip this player (will not be inserted)")
    print(f"{'='*80}")
    
    while True:
        try:
            choice = input(f"\nEnter your choice (0-{len(matches)}): ").strip()
            
            if choice == '0':
                print(f"[INFO] Skipping player {player_name}")
                return 0
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(matches):
                selected_match = matches[choice_num - 1]
                selected_id = selected_match[0]
                selected_name = selected_match[1] 
                selected_team = selected_match[4]
                print(f"[INFO] Selected: {selected_name} (ID: {selected_id}, Team: {selected_team})")
                return selected_id
            else:
                print(f"[ERROR] Invalid choice. Please enter a number between 0 and {len(matches)}")
                
        except ValueError:
            print(f"[ERROR] Invalid input. Please enter a number between 0 and {len(matches)}")
        except KeyboardInterrupt:
            print(f"\n[INFO] Process interrupted by user")
            return 0


def get_season_player_id(db: DatabaseConnector, player_name: str, team_abrv: str, season_id: int, age: int = None, position: str = None, interactive: bool = False) -> int:
    """Enhanced player_id lookup for season-level data with age and position validation.
    
    Used by season-level player stat insert scripts (plyr_def.py, etc.).
    Includes age and position matching for better player identification accuracy.
    
    Args:
        db: Database connector instance
        player_name: Player name as it appears in CSV
        team_abrv: Team abbreviation to help with disambiguation
        season_id: Season ID for the lookup
        age: Player age for additional validation (optional)
        position: Player position for additional validation (optional)
        interactive: If True, prompts user to manually select from multiple matches (optional)
        
    Returns:
        int: player_id from plyr or multi_tm_plyr table, or 0 if user chooses to skip (interactive mode only)
        
    Raises:
        ValueError: If no player found or lookup fails (non-interactive mode)
    """
    # Generate name variations to handle suffixes
    suffixes = ["II", "III", "IV", "Jr.", "Sr."]
    name_variations = [player_name] + [f"{player_name} {suffix}" for suffix in suffixes]
    placeholders = ', '.join(['%s'] * len(name_variations))
    
    # Build dynamic WHERE clauses for age and position
    age_clause = " AND p.plyr_age = %s" if age else ""
    pos_clause = " AND p.plyr_pos = %s" if position else ""
    age_clause_mtp = " AND mtp.plyr_age = %s" if age else ""
    pos_clause_mtp = " AND mtp.plyr_pos = %s" if position else ""
    
    if team_abrv and team_abrv.strip():
        # Search with team filtering for better accuracy
        query = f"""
        SELECT p.plyr_id, p.plyr_name, 'plyr' AS source, p.team_id, t.abrv, p.plyr_pos, p.plyr_age
        FROM plyr p
        JOIN nfl_team t ON p.team_id = t.team_id
        WHERE p.plyr_name IN ({placeholders}) AND (t.abrv = %s OR t.alt_abrv = %s) AND p.season_id = %s{age_clause}{pos_clause}
        UNION
        SELECT mtp.plyr_id, mtp.plyr_name, 'multi_tm_plyr' AS source, 
               COALESCE(mtp.former_tm_id, mtp.first_tm_id) AS team_id, 
               COALESCE(t1.abrv, t2.abrv) AS abrv,
               mtp.plyr_pos, mtp.plyr_age
        FROM multi_tm_plyr mtp
        LEFT JOIN nfl_team t1 ON mtp.former_tm_id = t1.team_id
        LEFT JOIN nfl_team t2 ON mtp.first_tm_id = t2.team_id
        WHERE mtp.plyr_name IN ({placeholders}) AND (t1.abrv = %s OR t1.alt_abrv = %s OR t2.abrv = %s OR t2.alt_abrv = %s) AND mtp.season_id = %s{age_clause_mtp}{pos_clause_mtp}
        """
        
        # Build parameter list dynamically
        params = name_variations + [team_abrv, team_abrv, season_id]
        if age:
            params.append(age)
        if position:
            params.append(position)
        params.extend(name_variations + [team_abrv, team_abrv, team_abrv, team_abrv, season_id])
        if age:
            params.append(age)
        if position:
            params.append(position)
    else:
        # Search without team filtering if team not available
        query = f"""
        SELECT p.plyr_id, p.plyr_name, 'plyr' AS source, p.team_id, t.abrv, p.plyr_pos, p.plyr_age
        FROM plyr p
        JOIN nfl_team t ON p.team_id = t.team_id
        WHERE p.plyr_name IN ({placeholders}) AND p.season_id = %s{age_clause}{pos_clause}
        """
        params = name_variations + [season_id]
        if age:
            params.append(age)
        if position:
            params.append(position)

    results = db.fetch_all(query, params)
    
    if len(results) == 1:
        return results[0][0]
    elif len(results) > 1:
        if interactive:
            # Use interactive selection for multiple matches
            selected_id = interactive_player_selection(player_name, team_abrv, age, position, results)
            if selected_id == 0:
                return 0  # User chose to skip this player
            return selected_id
        else:
            # Default behavior: warn and use first match
            print(f"[WARNING] Multiple matches found for {player_name} ({team_abrv}, age={age}, pos={position}). Using first match.")
            return results[0][0]
    else:
        # Fall back to basic lookup without age/position if no exact match
        if age or position:
            print(f"[INFO] No exact match for {player_name} with age/position. Trying basic lookup...")
            return get_player_id(db, player_name, team_abrv, season_id)
        else:
            if interactive:
                return 0  # Allow skipping in interactive mode
            else:
                raise ValueError(f"No player found for {player_name} ({team_abrv}) in season {season_id}")


def apply_position_mapping(position: str) -> str:
    """Apply positional mapping as specified in prompt
    
    Standardizes NFL position names across all player stat scripts.
    Maps specific position variations to general position categories.
    
    Args:
        position: Raw position string from CSV data
        
    Returns:
        str: Mapped position category
    """
    position_mapping = {
        'QB': 'QB', 
        'RB': 'RB', 
        'WR': 'WR', 
        'TE': 'TE', 'TE/QB': 'TE',
        'G': 'OL', 'C': 'OL', 'OG': 'OL', 'IOL': 'OL', 'OL': 'OL', 'LG': 'OL', 'RG': 'OL', 'LG/RG': 'OL', 'LG/RT': 'OL', 'C/RG': 'OL', 'LT/RT': 'OL', 'T-G': 'OL', 'RG/LG': 'OL', 'RG/LT': 'OL',
        'T': 'OL', 'OT': 'OL', 'RT': 'OL', 'LT': 'OL', 'RT/LT': 'OL',
        'FB/DL': 'FB',
        'DE': 'DL', 'DT': 'DL', 'NT': 'DL', 'LDE': 'DL', 'RDE': 'DL', 'LDE/RDE': 'DL', 'LDT': 'DL', 'RDT': 'DL', 'LDT/RDT': 'DL', 'DT/LB': 'DL', 'RDE/LDE': 'DL',
        'LB': 'LB', 'ILB': 'LB', 'MLB': 'LB', 'RLB/MLB' : 'LB', 'OLB': 'LB', 'LOLB': 'LB', 'ROLB': 'LB', 'LILB': 'LB', 'RILB': 'LB', 'LILB/RILB': 'LB', 'RILB/LILB': 'LB', 'LLB': 'LB', 'RLB': 'LB', 'MLB/RLB': 'LB', 'KB': 'LB',
        'CB': 'DB', 'DB': 'DB', 'LCB' : 'DB', 'RCB' : 'DB', 'LCB/RCB': 'DB', 'FS': 'DB', 'SS': 'DB','S': 'DB', 'SS/FS': 'DB', 'LBC/DB': 'DB', 'DB/LCB': 'DB', 'LCB/DB': 'DB',
        'K': 'K', 'PK': 'K',
        'P': 'P', 
        'LS': 'LS'
    }
    return position_mapping.get(position, position)


def create_table_if_not_exists(db: DatabaseConnector, table_name: str, create_sql: str) -> bool:
    """Generic table creation function used by all insert scripts.
    
    Standardizes table creation with consistent error handling and messaging.
    
    Args:
        db: Database connector instance
        table_name: Name of the table being created (for messaging)
        create_sql: Complete CREATE TABLE IF NOT EXISTS SQL statement
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        success = db.execute_query(create_sql)
        if success:
            print(f"[OK] {table_name} table created/verified successfully")
            return True
        else:
            print(f"[FAIL] Failed to create {table_name} table")
            return False
    except Exception as e:
        print(f"[ERROR] Error creating {table_name} table: {e}")
        return False


# ============================================
# PLAYER GUID AND MASTER RECORD UTILITIES
# ============================================
# Functions for managing player master records and cross-season identification

def check_birthday_tolerance(db: DatabaseConnector, plyr_name: str, plyr_birthday: str, 
                              plyr_draft_tm: str, season_id: int, table_name: str) -> Optional[Dict[str, Any]]:
    """Check for existing players with same name/draft_tm/season but birthday ±1 day.
    
    Args:
        db: Database connector instance
        plyr_name: Player full name
        plyr_birthday: Birthday in YYYY-MM-DD format
        plyr_draft_tm: Draft team name
        season_id: Season ID (only for plyr table check)
        table_name: 'plyr' or 'plyr_master'
    
    Returns:
        Dict with match details if found, None otherwise
    """
    if not plyr_birthday:
        return None
    
    try:
        birthday_date = datetime.strptime(plyr_birthday, "%Y-%m-%d")
        day_before = (birthday_date - timedelta(days=1)).strftime("%Y-%m-%d")
        day_after = (birthday_date + timedelta(days=1)).strftime("%Y-%m-%d")
        
        if table_name == 'plyr':
            query = """
                SELECT plyr_id, plyr_name, plyr_birthday, plyr_draft_tm, season_id
                FROM plyr
                WHERE plyr_name = %s 
                AND plyr_draft_tm = %s
                AND season_id = %s
                AND plyr_birthday IN (%s, %s)
            """
            params = (plyr_name, plyr_draft_tm, season_id, day_before, day_after)
        else:  # plyr_master
            query = """
                SELECT plyr_guid, plyr_name, plyr_birthday, plyr_draft_tm
                FROM plyr_master
                WHERE plyr_name = %s 
                AND plyr_draft_tm = %s
                AND plyr_birthday IN (%s, %s)
            """
            params = (plyr_name, plyr_draft_tm, day_before, day_after)
        
        result = db.fetch_all(query, params)
        
        if result:
            if table_name == 'plyr':
                return {
                    'table': table_name,
                    'plyr_id': result[0][0],
                    'plyr_name': result[0][1],
                    'db_birthday': result[0][2],
                    'plyr_draft_tm': result[0][3],
                    'season_id': result[0][4]
                }
            else:
                return {
                    'table': table_name,
                    'plyr_guid': result[0][0],
                    'plyr_name': result[0][1],
                    'db_birthday': result[0][2],
                    'plyr_draft_tm': result[0][3]
                }
        
        return None
        
    except Exception as e:
        logging.error(f"Error checking birthday tolerance: {e}")
        return None


def prompt_user_birthday_resolution(csv_data: Dict[str, Any], db_match: Dict[str, Any]) -> tuple:
    """Prompt user to resolve birthday discrepancy.
    
    Args:
        csv_data: Dict with CSV player data (name, birthday, draft_tm, season_id)
        db_match: Dict with database match details
    
    Returns:
        Tuple: (is_same_player: bool, correct_birthday: str or None)
    """
    print("\n" + "="*80)
    print("BIRTHDAY DISCREPANCY DETECTED - MANUAL INTERVENTION REQUIRED")
    print("="*80)
    print(f"\nPlayer Name: {csv_data['plyr_name']}")
    print(f"Draft Team: {csv_data['plyr_draft_tm']}")
    if 'season_id' in csv_data:
        print(f"Season ID: {csv_data['season_id']}")
    print(f"\nCSV Birthday:      {csv_data['plyr_birthday']}")
    print(f"Database Birthday: {db_match['db_birthday']}")
    print(f"Difference: ±1 day")
    print(f"\nDatabase Table: {db_match['table']}")
    
    # Ask if same player
    while True:
        response = input("\nIs this the SAME player? (y/n): ").strip().lower()
        if response in ['y', 'n']:
            break
        print("Invalid input. Please enter 'y' or 'n'")
    
    if response == 'n':
        print("\nTreating as UNIQUE player - will create separate database entry")
        return False, None
    
    # Same player - ask which birthday to keep
    print("\nWhich birthday is CORRECT?")
    print(f"1. CSV Birthday:      {csv_data['plyr_birthday']}")
    print(f"2. Database Birthday: {db_match['db_birthday']}")
    
    while True:
        choice = input("\nSelect option (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Invalid input. Please enter '1' or '2'")
    
    correct_birthday = csv_data['plyr_birthday'] if choice == '1' else db_match['db_birthday']
    print(f"\nUsing birthday: {correct_birthday}")
    print("="*80 + "\n")
    
    return True, correct_birthday


def generate_player_guid(plyr_name: str, plyr_birthday: str, plyr_draft_tm: str) -> str:
    """Generate deterministic player GUID from natural key.
    
    CRITICAL: Must use identical logic in Python and MySQL.
    This function provides cross-season player identification.
    
    Args:
        plyr_name: Player full name
        plyr_birthday: Birthday in YYYY-MM-DD format  
        plyr_draft_tm: Draft team name or 'UNDRAFTED FREE AGENT'
    
    Returns:
        str: MD5 hash (32 hex characters)
    """
    # Handle None values by converting to empty string
    name = plyr_name if plyr_name else ""
    birthday = plyr_birthday if plyr_birthday else ""
    draft_tm = plyr_draft_tm if plyr_draft_tm else ""
    
    key = f"{name}_{birthday}_{draft_tm}"
    return hashlib.md5(key.encode('utf-8')).hexdigest()


def get_or_create_player_guid(db: DatabaseConnector, plyr_name: str, 
                               plyr_birthday: str, plyr_draft_tm: str,
                               plyr_college: str = None,
                               plyr_draft_rd: int = None, plyr_draft_pick: int = None,
                               plyr_draft_yr: int = None, primary_pos: str = None) -> str:
    """Get existing plyr_guid or create new plyr_master record.
    
    This function ensures each unique player has exactly one master record
    across all seasons. Uses natural key (name, birthday, draft_tm) for identification.
    
    Handles birthday discrepancies with ±1 day tolerance, prompting user for resolution.
    
    Args:
        db: Database connector instance
        plyr_name: Player full name
        plyr_birthday: Birthday in YYYY-MM-DD format
        plyr_draft_tm: Draft team name or 'UNDRAFTED FREE AGENT'
        plyr_college: College name (optional)
        plyr_draft_rd: Draft round (optional)
        plyr_draft_pick: Draft pick number (optional)
        plyr_draft_yr: Draft year (optional)
        primary_pos: Primary position (optional)
    
    Returns:
        str: plyr_guid (MD5 hash)
        
    Raises:
        ValueError: If master record creation fails
    """
    # Check for birthday tolerance match BEFORE generating GUID
    tolerance_match = check_birthday_tolerance(
        db, plyr_name, plyr_birthday, plyr_draft_tm, 
        season_id=None, table_name='plyr_master'
    )
    
    if tolerance_match:
        csv_data = {
            'plyr_name': plyr_name,
            'plyr_birthday': plyr_birthday,
            'plyr_draft_tm': plyr_draft_tm
        }
        
        is_same_player, correct_birthday = prompt_user_birthday_resolution(csv_data, tolerance_match)
        
        if is_same_player:
            plyr_birthday = correct_birthday
            
            if correct_birthday != tolerance_match['db_birthday']:
                update_query = """
                    UPDATE plyr_master 
                    SET plyr_birthday = %s 
                    WHERE plyr_guid = %s
                """
                db.execute_query(update_query, (correct_birthday, tolerance_match['plyr_guid']))
                print(f"Updated plyr_master birthday to {correct_birthday}")
    
    # Generate GUID from natural key (potentially corrected birthday)
    player_guid = generate_player_guid(plyr_name, plyr_birthday, plyr_draft_tm)
    
    # Always execute UPSERT to update college/draft info if new data is available
    insert_query = """
        INSERT INTO plyr_master 
        (plyr_guid, plyr_name, plyr_birthday, plyr_college, 
         plyr_draft_tm, plyr_draft_rd, plyr_draft_pick, plyr_draft_yr, primary_pos)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            plyr_college = COALESCE(VALUES(plyr_college), plyr_college),
            plyr_draft_rd = COALESCE(VALUES(plyr_draft_rd), plyr_draft_rd),
            plyr_draft_pick = COALESCE(VALUES(plyr_draft_pick), plyr_draft_pick),
            plyr_draft_yr = COALESCE(VALUES(plyr_draft_yr), plyr_draft_yr),
            primary_pos = COALESCE(VALUES(primary_pos), primary_pos),
            updated_at = CURRENT_TIMESTAMP
    """
    
    success = db.execute_query(insert_query, (
        player_guid, plyr_name, plyr_birthday, plyr_college,
        plyr_draft_tm, plyr_draft_rd, plyr_draft_pick, plyr_draft_yr, primary_pos
    ))
    
    if not success:
        raise ValueError(f"Failed to create plyr_master record for {plyr_name}")
    
    return player_guid


if __name__ == "__main__":
    # Test database connection
    db = DatabaseConnector()
    if db.connect():
        print("Database connection test successful!")
        db.disconnect()
    else:
        print("Database connection test failed!")