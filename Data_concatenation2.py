import pandas as pd
import numpy as np
import os
# dir_folder: name of folder location where all timesteps csv files are present
loop = ["050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz"]
#dir_folder = str(input("CSV folder location of all timesteps: "))
# change to timesteps (name of timesteps files inside dir_folder)
#subdirs = [name for name in os.listdir(dir_folder) if os.path.isdir(os.path.join(dir_folder, name))]
#subdirs = sorted(subdirs, key=lambda x: int(x.split('_')[1]))
#print(subdirs)
for freq in loop:
    df1 = pd.read_csv(f"concatenated_{freq}_L1231.csv")
    df2 = pd.read_csv(f"concatenated_{freq}_L1232.csv")
    df3 = pd.read_csv(f"concatenated_{freq}_L1233.csv")
    df4 = pd.read_csv(f"concatenated_{freq}_L1234.csv")
    data = pd.concat([df1, df2, df3, df4], axis=1)
    print(data)
    data.to_csv(f"concatenated_{freq}_L123.csv", index=False)
    # samples = [subdir + "//" + f'{freq}-allfeatures' for subdir in subdirs]
    # print(samples)

    # # sample file (to obatain amount of features - necessary to create full_matrix dimension)
    # dir0 = os.path.join(dir_folder, samples[0] + ".csv")
    # df0 = pd.read_csv(dir0)
    # # creates empty matrix with shape (no. of features x no. of paths, no. of timesteps)
    # full_matrix = np.zeros((np.shape(df0)[0]*np.shape(df0)[1], len(samples)))
    #
    # for i, sample in enumerate(samples):
    #     dir_file = os.path.join(dir_folder, sample + ".csv")
    #     df = pd.read_csv(dir_file)
    #     # Concatenate all columns into a single column
    #     concatenated_column = pd.concat([df[col] for col in df.columns], ignore_index=True)
    #     # Add concatenated column to respective index (with respect to timestep) in full_matrix
    #     full_matrix[:, i] = concatenated_column
    # result_df = pd.DataFrame(full_matrix)
    # # Saves df as csv file with same name as folder
    # output_file_path = os.path.join(dir_folder, f"{dir_folder}{freq}.csv")
    # result_df.to_csv(output_file_path, index=False)