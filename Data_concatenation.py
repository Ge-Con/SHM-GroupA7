import pandas as pd
import numpy as np
import os
# C:\Users\pablo\Downloads\SHM_Concatenated_HLB_Features
# C:\Users\pablo\Downloads\PZT-FFT-HLB\PZT-FFT-HLB\PZT-FFT-HLB-L1-23
def process_csv_files(base_dir, panel, type):
    # Define the sample filenames you're expecting in each folder
    for freq in ["050", "100", "125", "150", "200", "250"]:
    # Recursively traverse all directories and subdirectories
        full_matrix = []
        for root, dirs, files in os.walk(base_dir + "\\" + panel):
            for name in files:
                if name.endswith(f'{freq}kHz_{type}.csv'):
                    df0 = pd.read_csv(os.path.join(root, name))
                    concatenated_column = pd.concat([df0[col] for col in df0.columns], ignore_index=True)
                    # Add concatenated column to respective index (with respect to timestep) in full_matrix
                    full_matrix.append(concatenated_column)
        if panel.endswith("03"):
            panel = "L103"
        if panel.endswith("04"):
            panel = "L104"
        if panel.endswith("05"):
            panel = "L105"
        if panel.endswith("09"):
            panel = "L109"
        if panel.endswith("23"):
            panel = "L123"
        result_df = pd.DataFrame(full_matrix).T
        output_file_path = os.path.join(base_dir, f"concatenated_{freq}_kHz_{panel}_{type}.csv")
        result_df.to_csv(output_file_path, index=False)
        print(type + ": " + panel + " " + freq + "kHz complete")

# Prompt user for the base directory containing all subdirectories with CSV files
#base_dir = str(input("Base directory containing all timesteps folders: "))
#process_csv_files(base_dir)


# C:\Users\pablo\Downloads\PZT-ONLY-FEATURES Main\PZT-ONLY-FEATURES-CSV-L1-23
# C:\Users\pablo\Downloads\PZT Output folder