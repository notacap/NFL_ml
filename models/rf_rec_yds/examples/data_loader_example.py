"""
Example usage of the DataLoader class.

This script demonstrates how to use DataLoader to load various types
of partitioned parquet files for the NFL ML project.
"""

import logging
from utils.path_manager import PathManager
from utils.data_loader import DataLoader

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def example_basic_usage():
    """Basic DataLoader usage examples."""
    print("\n" + "="*70)
    print("BASIC USAGE EXAMPLES")
    print("="*70)

    # Initialize PathManager and DataLoader
    paths = PathManager()
    loader = DataLoader(paths)

    print(f"\nDataLoader initialized: {loader}")

    # Example 1: Load target variable for 2024 season
    print("\n1. Loading target variable (player game receiving yards)...")
    try:
        target_df = loader.load_target(seasons=2024, weeks=range(1, 18))
        print(f"   Loaded {len(target_df)} rows, {len(target_df.columns)} columns")
        print(f"   Columns: {list(target_df.columns[:5])}...")
    except Exception as e:
        print(f"   Error: {e}")

    # Example 2: Load player info
    print("\n2. Loading player information...")
    try:
        players_df = loader.load_table('players', seasons=[2024])
        print(f"   Loaded {len(players_df)} rows, {len(players_df.columns)} columns")
        print(f"   Columns: {list(players_df.columns[:5])}...")
    except Exception as e:
        print(f"   Error: {e}")

    # Example 3: Load static table
    print("\n3. Loading static team data...")
    try:
        teams_df = loader.load_static('teams')
        print(f"   Loaded {len(teams_df)} rows, {len(teams_df.columns)} columns")
        print(f"   Teams: {teams_df['team_abbr'].tolist() if 'team_abbr' in teams_df.columns else 'N/A'}")
    except Exception as e:
        print(f"   Error: {e}")


def example_partition_inspection():
    """Example of inspecting available partitions."""
    print("\n" + "="*70)
    print("PARTITION INSPECTION EXAMPLES")
    print("="*70)

    paths = PathManager()
    loader = DataLoader(paths)

    # Check available partitions for different tables
    tables_to_check = [
        'player_game_receiving',
        'player_season_receiving',
        'players',
        'teams'
    ]

    for table_key in tables_to_check:
        print(f"\n{table_key}:")
        try:
            partitions = loader.get_available_partitions(table_key)
            if partitions:
                print(f"   Found {len(partitions)} partitions")
                # Show first and last few partitions
                if len(partitions) <= 5:
                    for p in partitions:
                        print(f"   - Season {p['season']}, Week {p['week']}")
                else:
                    print(f"   First: Season {partitions[0]['season']}, Week {partitions[0]['week']}")
                    print(f"   Last:  Season {partitions[-1]['season']}, Week {partitions[-1]['week']}")
            else:
                print("   No partitions (static table)")
        except Exception as e:
            print(f"   Error: {e}")


def example_advanced_filtering():
    """Advanced filtering examples."""
    print("\n" + "="*70)
    print("ADVANCED FILTERING EXAMPLES")
    print("="*70)

    paths = PathManager()
    loader = DataLoader(paths)

    # Example 1: Load specific weeks
    print("\n1. Loading weeks 1-5 only...")
    try:
        df = loader.load_game_level(
            'player_game_receiving',
            seasons=2024,
            weeks=range(1, 6)
        )
        print(f"   Loaded {len(df)} rows")
        if 'week' in df.columns:
            print(f"   Weeks present: {sorted(df['week'].unique())}")
    except Exception as e:
        print(f"   Error: {e}")

    # Example 2: Load multiple seasons
    print("\n2. Loading multiple seasons...")
    try:
        df = loader.load_table(
            'player_game_receiving',
            seasons=[2023, 2024],
            weeks=range(1, 10)
        )
        print(f"   Loaded {len(df)} rows")
        if 'season' in df.columns and 'week' in df.columns:
            print(f"   Seasons: {sorted(df['season'].unique())}")
            print(f"   Weeks: {sorted(df['week'].unique())}")
    except Exception as e:
        print(f"   Error: {e}")

    # Example 3: Load season cumulative data
    print("\n3. Loading season cumulative stats through week 10...")
    try:
        df = loader.load_season_level(
            'player_season_receiving',
            seasons=2024,
            weeks=10
        )
        print(f"   Loaded {len(df)} rows")
        print(f"   This contains cumulative stats through week 10")
    except Exception as e:
        print(f"   Error: {e}")


def example_type_specific_methods():
    """Examples using type-specific loading methods."""
    print("\n" + "="*70)
    print("TYPE-SPECIFIC LOADING METHODS")
    print("="*70)

    paths = PathManager()
    loader = DataLoader(paths)

    # Game-level loading
    print("\n1. Game-level data (load_game_level)...")
    try:
        df = loader.load_game_level(
            'player_game_receiving',
            seasons=2024,
            weeks=[1, 2, 3]
        )
        print(f"   Loaded {len(df)} rows of game-level data")
    except Exception as e:
        print(f"   Error: {e}")

    # Season-level loading
    print("\n2. Season-level data (load_season_level)...")
    try:
        df = loader.load_season_level(
            'player_season_receiving',
            seasons=2024
        )
        print(f"   Loaded {len(df)} rows of season cumulative data")
    except Exception as e:
        print(f"   Error: {e}")

    # Static loading
    print("\n3. Static data (load_static)...")
    try:
        df = loader.load_static('teams')
        print(f"   Loaded {len(df)} rows of static reference data")
    except Exception as e:
        print(f"   Error: {e}")

    # Target loading
    print("\n4. Target variable (load_target)...")
    try:
        df = loader.load_target(seasons=2024)
        print(f"   Loaded {len(df)} rows of target variable")
    except Exception as e:
        print(f"   Error: {e}")


def example_error_handling():
    """Examples of error handling."""
    print("\n" + "="*70)
    print("ERROR HANDLING EXAMPLES")
    print("="*70)

    paths = PathManager()
    loader = DataLoader(paths)

    # Example 1: Invalid table key
    print("\n1. Attempting to load non-existent table...")
    try:
        df = loader.load_table('invalid_table_key')
        print(f"   Loaded successfully (unexpected!)")
    except KeyError as e:
        print(f"   Caught KeyError (expected): {e}")

    # Example 2: Wrong method for table type
    print("\n2. Attempting to load game-level table as static...")
    try:
        df = loader.load_static('player_game_receiving')
        print(f"   Loaded successfully (unexpected!)")
    except ValueError as e:
        print(f"   Caught ValueError (expected): {e}")

    # Example 3: No data for filters
    print("\n3. Attempting to load non-existent season...")
    try:
        df = loader.load_table('player_game_receiving', seasons=[1900])
        print(f"   Loaded successfully (unexpected!)")
    except (ValueError, FileNotFoundError) as e:
        print(f"   Caught error (expected): {e}")


def example_practical_workflow():
    """Practical workflow example for model training."""
    print("\n" + "="*70)
    print("PRACTICAL WORKFLOW: MODEL TRAINING DATA PREPARATION")
    print("="*70)

    paths = PathManager()
    loader = DataLoader(paths)

    # Step 1: Load target variable
    print("\n1. Loading target variable (player receiving yards)...")
    try:
        target = loader.load_target(
            seasons=[2023, 2024],
            weeks=range(1, 18)
        )
        print(f"   Target shape: {target.shape}")
    except Exception as e:
        print(f"   Error: {e}")
        return

    # Step 2: Load game-level features
    print("\n2. Loading game-level features...")
    try:
        game_rec = loader.load_game_level(
            'player_game_receiving',
            seasons=[2023, 2024],
            weeks=range(1, 18)
        )
        print(f"   Game receiving stats: {game_rec.shape}")
    except Exception as e:
        print(f"   Error: {e}")

    # Step 3: Load season cumulative features
    print("\n3. Loading season cumulative features...")
    try:
        season_rec = loader.load_season_level(
            'player_season_receiving',
            seasons=[2023, 2024],
            weeks=range(1, 18)
        )
        print(f"   Season cumulative stats: {season_rec.shape}")
    except Exception as e:
        print(f"   Error: {e}")

    # Step 4: Load reference data
    print("\n4. Loading reference data...")
    try:
        players = loader.load_table('players', seasons=[2023, 2024])
        teams = loader.load_static('teams')
        print(f"   Players: {players.shape}")
        print(f"   Teams: {teams.shape}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n5. Next steps: Join tables and create features...")
    print("   - Merge game stats with player info")
    print("   - Create lagged features")
    print("   - Engineer interaction features")
    print("   - Prepare train/test split")


if __name__ == '__main__':
    # Run all examples
    example_basic_usage()
    example_partition_inspection()
    example_advanced_filtering()
    example_type_specific_methods()
    example_error_handling()
    example_practical_workflow()

    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70 + "\n")
