import pandas as pd
import numpy as np
# dir_folder: name of folder location where all timesteps csv files are present
dir_folder = input("CSV folder location of all timesteps: ")
# change to timesteps (name of timesteps files inside dir_folder)
samples = ["PZT-CSV L1-03", "PZT-CSV L1-05", "PZT-CSV L1-09"] 

# sample file (to obatain amount of features - necessary to create full_matrix dimension)
dir0 = dir_folder + "\\" + samples[0]
df0 = pd.read_csv(dir0)
# creates empty matrix with shape (no. of features x no. of timesteps)
full_matrix = np.zeros((np.shape(df0)[0], len(samples))) 

for i in range(len(samples)):
    dir_file = dir_folder + "\\" + samples[i]
    df = pd.read_csv(dir_file)
    # Concatenate all columns into a single column
    concatenated_column = pd.concat([df[col] for col in df.columns], ignore_index=True)
    # Add concatenated column to respective index (with respect to timestep) in full_matrix 
    full_matrix[:, i] = concatenated_column
result_df = pd.DataFrame(full_matrix)
# Saves df as csv file with same name as folder
result_df.to_csv(dir_folder, index=False)
