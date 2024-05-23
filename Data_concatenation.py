import pandas as pd
import numpy as np
import os

def process_csv_files(base_dir):
    # Define the sample filenames you're expecting in each folder
    for freq in ["050", "100", "125", "150", "200", "250"]:
    # Recursively traverse all directories and subdirectories
        full_matrix = []
        for root, dirs, files in os.walk(base_dir):
            for name in files:
                if name.endswith(f'{freq}_kHz-allfeatures.csv'):
                    df0 = pd.read_csv(os.path.join(root, name))
                    concatenated_column = pd.concat([df0[col] for col in df0.columns], ignore_index=True)
                    # Add concatenated column to respective index (with respect to timestep) in full_matrix
                    full_matrix.append(concatenated_column)

        result_df = pd.DataFrame(full_matrix).T
        result_df.to_csv(base_dir, index=False)
        print(f"Result saved to {base_dir}")

# Prompt user for the base directory containing all subdirectories with CSV files
base_dir = str(input("Base directory containing all timesteps folders: "))
process_csv_files(base_dir)


#"C:\Users\pablo\Downloads\PZT-ONLY-FEATURES Main\PZT-ONLY-FEATURES-CSV-L1-23"