import os
import pandas as pd
import glob
import csv
import shutil
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clean_utils import YEAR, WEEK

def consolidate_header_values(header1, header2):
    """
    Consolidate two header values based on the specified rules.
    
    Args:
        header1: First header row value
        header2: Second header row value
        
    Returns:
        str: Consolidated header value
    """
    h1 = str(header1).strip() if header1 is not None else ""
    h2 = str(header2).strip() if header2 is not None else ""
    
    if 'Unnamed' in h1 and 'Unnamed' not in h2:
        return h2
    elif 'Unnamed' in h2 and 'Unnamed' not in h1:
        return h1
    
    if not h1 and h2:
        return h2
    elif not h2 and h1:
        return h1
    
    if h1 and h2 and 'Unnamed' not in h1 and 'Unnamed' not in h2:
        return f"{h1} {h2}"
    
    return h1 if h1 else h2

def get_most_recent_file(directory_path):
    """
    Get the most recently created CSV file in a directory.
    
    Args:
        directory_path (str): Path to the directory to search
        
    Returns:
        str or None: Path to the most recent CSV file, or None if no files found
    """
    if not os.path.exists(directory_path):
        return None
    
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    if not csv_files:
        return None
    
    most_recent = max(csv_files, key=os.path.getctime)
    return most_recent

def process_csv_file(file_path):
    """
    Process a CSV file to consolidate its header rows.
    
    Args:
        file_path (str): Path to the CSV file to process
        
    Returns:
        bool: True if successful, False if error occurred
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header1 = next(reader)
            header2 = next(reader)
            
            data_rows = list(reader)
        
        max_len = max(len(header1), len(header2))
        header1.extend([''] * (max_len - len(header1)))
        header2.extend([''] * (max_len - len(header2)))
        
        new_headers = []
        for h1, h2 in zip(header1, header2):
            consolidated_header = consolidate_header_values(h1, h2)
            new_headers.append(consolidated_header)
        
        df = pd.DataFrame(data_rows, columns=new_headers[:len(data_rows[0]) if data_rows else len(new_headers)])
        
        file_dir = os.path.dirname(file_path)
        clean_dir = os.path.join(file_dir, 'clean')
        os.makedirs(clean_dir, exist_ok=True)
        
        original_filename = os.path.basename(file_path)
        cleaned_filename = f"cleaned_{original_filename}"
        output_path = os.path.join(clean_dir, cleaned_filename)
        
        df.to_csv(output_path, index=False)
        print(f"Processed: {file_path} -> {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def get_source_directories():
    """
    Get all source directory paths based on YEAR and WEEK variables.
    
    Returns:
        list: List of directory paths to process
    """
    base_path = rf'C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\{YEAR}'
    
    directories = [
        f'{base_path}\\tm_conv\\week_{WEEK}',
    ]
    
    return directories

def main():
    """
    Main function to process the most recent CSV file in each specified directory.
    """
    directories = get_source_directories()
    
    failed_files = []
    total_files_processed = 0
    
    print(f"Processing files for Year: {YEAR}, Week: {WEEK}")
    print("="*50)
    
    for directory in directories:
        print(f"\nChecking directory: {directory}")
        
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist, skipping...")
            continue
        
        most_recent_file = get_most_recent_file(directory)
        if not most_recent_file:
            print(f"No CSV files found in {directory}")
            continue
        
        print(f"Found most recent file: {os.path.basename(most_recent_file)}")
        total_files_processed += 1
        
        if not process_csv_file(most_recent_file):
            failed_files.append(most_recent_file)
    
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