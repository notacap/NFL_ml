import os

# Year variable to determine the data year
YEAR = 2025

# Week variable to determine the week number (1-18)
WEEK = 4

# Week range variables for processing multiple weeks
WEEK_START = 4
WEEK_END = 4

def get_source_directory():
    """
    Get the source CSV directory path based on the year and week.
    
    Returns:
        str: Full path to the source directory
    """
    return rf'C:\Users\nocap\Desktop\code\NFL_ml\web_scrape\scraped_data\{YEAR}\roster_details\week_{WEEK}'

def get_output_directory():
    """
    Get the output directory path for cleaned files.
    Creates a 'cleaned' folder in the source directory.
    
    Returns:
        str: Full path to the cleaned output directory
    """
    source_dir = get_source_directory()
    output_dir = os.path.join(source_dir, 'cleaned')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_cleaned_filename(original_filename):
    """
    Generate the cleaned filename by prepending 'clean_' to the original filename.
    
    Args:
        original_filename (str): The original CSV filename
        
    Returns:
        str: The new filename with 'clean_' prefix
    """
    return f'clean_{original_filename}'