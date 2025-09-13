import os
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clean_utils import get_source_directory, get_output_directory, get_cleaned_filename

# Get the input and output directory paths using utility functions
input_directory = get_source_directory()
output_directory = get_output_directory()

# Columns to remove
columns_to_remove = ['No.']

# Function to convert height from feet-inches format to total inches
def convert_height_to_inches(height_str):
    if pd.isna(height_str) or height_str == '':
        return None
    
    try:
        # Convert to string and strip whitespace
        height_str = str(height_str).strip()
        
        # Split by dash to get feet and inches
        if '-' in height_str:
            feet, inches = height_str.split('-')
            feet = int(feet.strip())
            inches = int(inches.strip())
            return feet * 12 + inches
        else:
            # If no dash, assume it's already in inches or invalid format
            return None
    except (ValueError, AttributeError):
        return None

# Function to process a single CSV file
def process_csv(file_path, team_name):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Remove specified columns
    df = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')
    
    # Remove rows that start with 'Team Total'
    df = df[~df.iloc[:, 0].astype(str).str.startswith('Team Total')]
    
    # Add 'Team' column and populate with team name
    df['Team'] = team_name
    
    # Convert 'Rook' to 0 in the 'Yrs' column
    if 'Yrs' in df.columns:
        df['Yrs'] = df['Yrs'].replace('Rook', '0')
        df['Yrs'] = pd.to_numeric(df['Yrs'], errors='coerce')
    
    # Convert height from feet-inches format to total inches
    if 'Ht' in df.columns:
        df['Ht'] = df['Ht'].apply(convert_height_to_inches)
    
    # Parse the "Drafted (tm/rnd/yr)" column
    if 'Drafted (tm/rnd/yr)' in df.columns:
        # Split the column by " / " and create new columns
        draft_split = df['Drafted (tm/rnd/yr)'].str.split(' / ', expand=True)
        
        # Create the new columns
        df['Draft Team'] = draft_split[0] if draft_split.shape[1] > 0 else None
        
        # Extract round number (remove ordinal suffixes like st, nd, rd, th)
        if draft_split.shape[1] > 1:
            df['Draft Round'] = draft_split[1].str.extract(r'(\d+)')[0]
            df['Draft Round'] = pd.to_numeric(df['Draft Round'], errors='coerce')
        else:
            df['Draft Round'] = None
            
        # Extract pick number (remove ordinal suffixes and "pick" text)
        if draft_split.shape[1] > 2:
            df['Draft Pick'] = draft_split[2].str.extract(r'(\d+)')[0]
            df['Draft Pick'] = pd.to_numeric(df['Draft Pick'], errors='coerce')
        else:
            df['Draft Pick'] = None
            
        # Extract year
        if draft_split.shape[1] > 3:
            df['Draft Year'] = pd.to_numeric(draft_split[3], errors='coerce')
        else:
            df['Draft Year'] = None
        
        # Remove the original column
        df = df.drop(columns=['Drafted (tm/rnd/yr)'])
    
    return df

# Process all CSV files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_directory, filename)
        
        # Extract team name from filename (format: City_Team_Name_timestamp.csv)
        filename_without_ext = filename.replace('.csv', '')
        filename_parts = filename_without_ext.split('_')
        
        # Find the timestamp part (starts with year like 2024)
        timestamp_index = None
        for i, part in enumerate(filename_parts):
            if part.startswith('202') and len(part) == 4:  # Year format
                timestamp_index = i
                break
        
        # Extract city and team name parts (everything before timestamp)
        if timestamp_index is not None:
            team_parts = filename_parts[:timestamp_index]
        else:
            # Fallback: assume last 2 parts are timestamp (year_datetime)
            team_parts = filename_parts[:-2]
        
        # Join city and team name with space
        team_name = ' '.join(team_parts).title()
        
        # Process the CSV
        processed_df = process_csv(file_path, team_name)
        
        # Save the processed CSV to the output directory
        output_filename = get_cleaned_filename(filename)
        output_path = os.path.join(output_directory, output_filename)
        processed_df.to_csv(output_path, index=False)
        
        print(f"Processed {filename} and saved as {output_filename} in the cleaned directory")
        print(f"Team name used: {team_name}")

print("All CSV files have been processed and saved in the cleaned directory.")

