import os
import pandas as pd
import glob
import csv
import shutil
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clean_utils import YEAR, WEEK_START, WEEK_END

def consolidate_header_values(header1, header2):
    """
    Consolidate two header values based on the specified rules.
    
    Args:
        header1: First header row value
        header2: Second header row value
        
    Returns:
        str: Consolidated header value
    """
    # Convert to strings and handle NaN/None values
    h1 = str(header1).strip() if header1 is not None else ""
    h2 = str(header2).strip() if header2 is not None else ""
    
    # Rule 1: If either contains 'Unnamed', use the other value
    if 'Unnamed' in h1 and 'Unnamed' not in h2:
        return h2
    elif 'Unnamed' in h2 and 'Unnamed' not in h1:
        return h1
    
    # Rule 2: If either is null/empty, use the non-null value
    if not h1 and h2:
        return h2
    elif not h2 and h1:
        return h1
    
    # Rule 3: If both have valid values, concatenate them
    if h1 and h2 and 'Unnamed' not in h1 and 'Unnamed' not in h2:
        return f"{h1} {h2}"
    
    # Fallback: return the first non-empty value or empty string
    return h1 if h1 else h2

def process_csv_file(file_path):
    """
    Process a CSV file to consolidate its header rows.
    
    Args:
        file_path (str): Path to the CSV file to process
        
    Returns:
        bool: True if successful, False if error occurred
    """
    try:
        # Read the file using CSV reader to properly handle quoted fields
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header1 = next(reader)  # First header row
            header2 = next(reader)  # Second header row
            
            # Store all data rows
            data_rows = list(reader)
        
        # Ensure both headers have the same length
        max_len = max(len(header1), len(header2))
        header1.extend([''] * (max_len - len(header1)))
        header2.extend([''] * (max_len - len(header2)))
        
        # Consolidate headers
        new_headers = []
        for h1, h2 in zip(header1, header2):
            consolidated_header = consolidate_header_values(h1, h2)
            new_headers.append(consolidated_header)
        
        # Create a new DataFrame with the consolidated headers and data
        df = pd.DataFrame(data_rows, columns=new_headers[:len(data_rows[0]) if data_rows else len(new_headers)])
        
        # Create clean directory path
        file_dir = os.path.dirname(file_path)
        clean_dir = os.path.join(file_dir, 'clean')
        os.makedirs(clean_dir, exist_ok=True)
        
        # Create new filename with 'cleaned_' prefix
        original_filename = os.path.basename(file_path)
        cleaned_filename = f"cleaned_{original_filename}"
        output_path = os.path.join(clean_dir, cleaned_filename)
        
        # Save the file to the clean directory with cleaned_ prefix
        df.to_csv(output_path, index=False)
        print(f"Processed: {file_path} -> {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def get_games_directory():
    """
    Get the games directory path based on the year.
    
    Returns:
        str: Full path to the games directory
    """
    return rf'C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\{YEAR}\games'

def copy_remaining_files(week_dir):
    """
    Copy all remaining files in the week directory to the clean subdirectory
    with 'cleaned_' prefix.

    Args:
        week_dir (str): Path to the week directory
    """
    clean_dir = os.path.join(week_dir, 'clean')
    os.makedirs(clean_dir, exist_ok=True)

    # Get all files in the week directory (excluding subdirectories)
    all_files = [f for f in os.listdir(week_dir) if os.path.isfile(os.path.join(week_dir, f))]

    for filename in all_files:
        source_path = os.path.join(week_dir, filename)

        # Check if file already exists in clean directory
        cleaned_filename = f"cleaned_{filename}"
        dest_path = os.path.join(clean_dir, cleaned_filename)

        # Only copy if the cleaned version doesn't already exist
        if not os.path.exists(dest_path):
            try:
                shutil.copy2(source_path, dest_path)
                print(f"Copied: {source_path} -> {dest_path}")
            except Exception as e:
                print(f"Error copying {source_path}: {str(e)}")

def main():
    """
    Main function to process CSV files in the specified week directories.
    """
    games_dir = get_games_directory()
    
    # Track failed files
    failed_files = []
    total_files_processed = 0
    
    # File patterns to process
    file_patterns = [
        'snap_counts',
        'expected_pts', 
        'plyr_def',
        'plyr_fumbles',
        'plyr_passing',
        'plyr_receiving',
        'plyr_rushing'
    ]
    
    # Process each week in the range
    for week_num in range(WEEK_START, WEEK_END + 1):
        week_dir = os.path.join(games_dir, f'week_{week_num}.0')
        
        if not os.path.exists(week_dir):
            print(f"Warning: Directory {week_dir} does not exist, skipping...")
            continue
            
        print(f"Processing week {week_num} directory: {week_dir}")
        
        # Process each file pattern for header consolidation
        for pattern in file_patterns:
            # Find all CSV files matching the pattern
            search_pattern = os.path.join(week_dir, f'*{pattern}*.csv')
            matching_files = glob.glob(search_pattern)
            
            # Filter out files containing 'adv_'
            filtered_files = [f for f in matching_files if 'adv_' not in os.path.basename(f)]
            
            if not filtered_files:
                print(f"No files found for pattern '{pattern}' in {week_dir}")
                continue
                
            for file_path in filtered_files:
                total_files_processed += 1
                if not process_csv_file(file_path):
                    failed_files.append(file_path)
        
        # Copy all remaining files to clean directory
        print(f"Copying remaining files in {week_dir} to clean directory...")
        copy_remaining_files(week_dir)
    
    # Display summary at the end
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total files processed: {total_files_processed}")
    print(f"Files failed: {len(failed_files)}")
    print(f"Files succeeded: {total_files_processed - len(failed_files)}")
    
    if failed_files:
        print("\nFiles that failed processing:")
        for failed_file in failed_files:
            print(f"  - {failed_file}")
    else:
        print("\nAll files processed successfully!")

if __name__ == "__main__":
    main()