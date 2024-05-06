import pandas as pd
import numpy as np
import os
# dir_folder: name of folder location where all timesteps csv files are present
dir_folder = str(input("CSV folder location of all timesteps: "))
# change to timesteps (name of timesteps files inside dir_folder)
samples = ["050_kHz-allfeatures", "060_kHz-allfeatures", "070_kHz-allfeatures"]

# sample file (to obatain amount of features - necessary to create full_matrix dimension)
dir0 = os.path.join(dir_folder, samples[0] + ".csv")
df0 = pd.read_csv(dir0)
# creates empty matrix with shape (no. of features x no. of timesteps)
full_matrix = np.zeros((np.shape(df0)[0]*np.shape(df0)[1], len(samples)))

for i, sample in enumerate(samples):
    dir_file = os.path.join(dir_folder, sample + ".csv")
    df = pd.read_csv(dir_file)
    # Concatenate all columns into a single column
    concatenated_column = pd.concat([df[col] for col in df.columns], ignore_index=True)
    # Add concatenated column to respective index (with respect to timestep) in full_matrix
    full_matrix[:, i] = concatenated_column
result_df = pd.DataFrame(full_matrix)
# Saves df as csv file with same name as folder
output_file_path = os.path.join(dir_folder, "result.csv")
result_df.to_csv(output_file_path, index=False)