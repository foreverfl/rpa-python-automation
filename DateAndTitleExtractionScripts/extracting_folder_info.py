import os
import pandas as pd
from datetime import datetime

def extract_date_title(filename):
    # example: '[230215] ハレ晴レユカイ'
    date_part, title_part = filename.split('] ', 1)
    date_str = date_part[1:]
    title = title_part.strip()
    return date_str, title

def process_files(folder_paths):
    all_data = []
    for folder_path in folder_paths:  # Iterate over each folder path
        files = os.listdir(folder_path)
        for file in files:
            if not file.startswith('['):  # Skip files not following the naming convention
                continue
            date_str, title = extract_date_title(file)
            all_data.append((date_str, title))
    
    df = pd.DataFrame(all_data, columns=['Date', 'Title']) # Convert the list to a DataFrame

    df.sort_values(by='Date', inplace=True) # Sort the DataFrame by date

    return df

def update_or_create_csv(csv_path, df_new):
    # Check if the CSV file exists
    try:
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new]).drop_duplicates().sort_values(by='Date')
    except FileNotFoundError:
        df_combined = df_new
    
    # Save the updated or new DataFrame to CSV
    df_combined.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
folder_path = ['C:\\Users\\forev\\OneDrive\\바탕 화면\\ユーチューブ\\あなたが知らなかった韓国']

df = process_files(folder_path)

csv_path = 'C:\\Users\\forev\\OneDrive\\바탕 화면\\videos_info.csv'

update_or_create_csv(csv_path, df)