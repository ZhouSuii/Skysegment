import pandas as pd
import glob
import os

def process_flight_data(source_dir, output_file):
    """
    Reads all CSV files from a source directory, combines them,
    and saves the result to a single file.

    Args:
        source_dir (str): The directory containing the source CSV files.
        output_file (str): The path to the output file (e.g., CSV or Parquet).
    """
    csv_files = glob.glob(os.path.join(source_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in directory: {source_dir}")
        return

    print(f"Found {len(csv_files)} CSV files to process.")

    all_dfs = []
    for f in csv_files:
        try:
            # Only read files that have a substantial size
            if os.path.getsize(f) > 1024: # Greater than 1KB
                df = pd.read_csv(f)
                all_dfs.append(df)
            else:
                print(f"Skipping small file: {f} (size: {os.path.getsize(f)} bytes)")
        except Exception as e:
            print(f"Could not read file {f} due to error: {e}")

    if not all_dfs:
        print("No data was loaded. Exiting.")
        return

    # Concatenate all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)

    print("Data combined successfully.")
    print("-----------------------------------")
    print("Initial data information:")
    combined_df.info(memory_usage='deep')
    print("-----------------------------------")
    
    # Sort data by flight and time to ensure trajectories are ordered
    if 'identification_id' in combined_df.columns and 'track_timestamp' in combined_df.columns:
        print("Sorting data by identification_id and track_timestamp...")
        combined_df.sort_values(by=['identification_id', 'track_timestamp'], inplace=True)
    
    # Save to a new file
    # Using Parquet is often more efficient for large datasets
    file_ext = os.path.splitext(output_file)[1]
    if file_ext.lower() == '.parquet':
        combined_df.to_parquet(output_file, index=False)
    else:
        combined_df.to_csv(output_file, index=False)
        
    print(f"Processed data saved to: {output_file}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Unique flights (identification_id): {combined_df['identification_id'].nunique()}")


if __name__ == '__main__':
    # Define the source directory and the output file
    SOURCE_DIRECTORY = '2024-11-11'
    OUTPUT_FILE = 'processed_data/all_flights_2024-11-11.parquet'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    process_flight_data(SOURCE_DIRECTORY, OUTPUT_FILE) 