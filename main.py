import scipy.io
import pandas as pd
import numpy as np
import os
import csv

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 400)
np.set_printoptions(linewidth=400)


def list_files(dir):
    dir2 = dir
    csv_dir = dir2.replace('PZT', 'PZT-CSV')

    # Create the output directory if it doesn't exist
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith('.mat'):
                mat_array = scipy.io.loadmat(os.path.join(root, name))

                root2 = root
                root_new = root2.replace('PZT', 'PZT-CSV')
                if not os.path.exists(root_new):
                    os.makedirs(root_new)

                for key, value in mat_array.items():
                    if not key.startswith('_') and isinstance(value, (list, tuple, np.ndarray)):
                        csv_file_path = os.path.join(root_new, f"{name.replace('.mat', '')}_{key}.csv")
                        print(f"Creating {csv_file_path}")

                        # Ensure the array is 1D
                        flattened_data = np.ravel(value)

                        # Check if the length is divisible by 9
                        if len(flattened_data) % 9 == 0:
                            # Reshape the array into 9 columns
                            reshaped_data = np.reshape(flattened_data, (-1, 9))

                            # Create a DataFrame with 9 columns
                            data = pd.DataFrame(reshaped_data, columns=[f'Column_{i + 1}' for i in range(9)])

                            # Write the DataFrame to CSV
                            data.to_csv(csv_file_path, index=False)
                        elif (key == "Trigger_Info"):
                            data = pd.DataFrame(flattened_data)
                            data.to_csv(csv_file_path, index=False)
                        else:
                            print(f"Data format invalid for key {key}. Skipping.")
                


list_files(r"YOUR-DIRECTORY-HERE\PZT-L1-03") #This should be the directory leading up to the main PZT file. 
