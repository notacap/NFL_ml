"""
Example usage of PathManager class.

This script demonstrates how to use the PathManager for various path operations.
"""

from path_manager import PathManager


def main():
    # Initialize PathManager (auto-detects project root)
    paths = PathManager()

    print("=" * 70)
    print("PathManager Usage Examples")
    print("=" * 70)

    # 1. Get paths from configuration
    print("\n1. Getting paths from config:")
    print(f"   Data source: {paths.get('data', 'source')}")
    print(f"   Processed data: {paths.get('data', 'processed')}")
    print(f"   Logs directory: {paths.get('logs')}")

    # 2. Create directories if they don't exist
    print("\n2. Ensuring directories exist:")
    feature_dir = paths.ensure_exists('data', 'feature_sets')
    print(f"   Feature sets: {feature_dir}")
    print(f"   Directory exists: {feature_dir.exists()}")

    # 3. Build parquet file paths
    print("\n3. Building parquet file paths:")

    # Base table path
    base_path = paths.get_parquet_path('plyr_gm/plyr_gm_rec')
    print(f"   Base table: {base_path}")

    # Season partition
    season_path = paths.get_parquet_path('plyr_gm/plyr_gm_rec', season=2024)
    print(f"   Season 2024: {season_path}")

    # Week partition
    week_path = paths.get_parquet_path('plyr_gm/plyr_gm_rec', season=2024, week=5)
    print(f"   Week 5: {week_path}")

    # With wildcard for reading multiple files
    wildcard_path = paths.get_parquet_path(
        'plyr_gm/plyr_gm_rec',
        season=2024,
        week=5,
        use_wildcard=True
    )
    print(f"   With wildcard: {wildcard_path}")

    # Static table (no partitions)
    static_path = paths.get_parquet_path('nfl_team.parquet')
    print(f"   Static table: {static_path}")

    # 4. List available partitions (if data exists)
    print("\n4. Listing partitions (if data exists):")
    try:
        partitions = paths.list_partitions('plyr_gm/plyr_gm_rec')
        if partitions:
            print(f"   Found {len(partitions)} season partitions")
            for p in partitions[:3]:  # Show first 3
                print(f"   - {p.name}")
        else:
            print("   No partitions found (data may not be downloaded yet)")
    except Exception as e:
        print(f"   Could not list partitions: {e}")

    # 5. Working with nested paths
    print("\n5. Nested configuration paths:")
    print(f"   Model checkpoints: {paths.get('models', 'checkpoints')}")
    print(f"   Viz feature importance: {paths.get('evaluation', 'viz_feature_importance')}")

    # 6. Using different configs
    print("\n6. Specifying config source:")
    data_source_auto = paths.get('data', 'source', config='auto')
    print(f"   Auto (tries data_config first): {data_source_auto}")

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
