import scipy.io
import pandas as pd
import numpy as np
import os
import csv
import pyexcel as p


pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 400)
np.set_printoptions(linewidth=400)


def matToCsv(dir):
    csv_dir = dir.replace('PZT','PZT-CSV')

    # Create the output directory if it doesn't exist
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    #Iterates over all the frequency files in the directory
    for root, dirs, files in os.walk(dir):
         for name in dirs:
             #print(name)
             if name.endswith("cycles"):
                #print("mat")
                root_new = root.replace('PZT', 'PZT-CSV')

                if not os.path.exists(root_new):
                    os.makedirs(root_new)
                arrayfile = combine_act(os.path.join(root,name))
                #for i in range(1,8):
                #time_col = arrayfile.iloc[:, 0]
                arrayfile = arrayfile.drop(columns = 'Sensor_0', axis=1)
                #print(arrayfile)
                csv_file_path = os.path.join(root_new, f"{name.replace('_5cycles', '')}.csv")
                #arrayfile = pd.concat([time_col, arrayfile], axis=1)
                #arrayfile.rename(columns={'Sensor_0': 'Time'}, inplace= True)
                # print(arrayfile)
                #print(arrayfile)
                arrayfile.to_csv(csv_file_path, index=False)


def combine_act(freq):
    freq_arr = []
    for root, dirs, files in os.walk(freq):
         for name in files:
             if name.endswith('rep_1.mat'):
                locat = os.path.join(root, name)
                actuatorNum = int(locat[-25])
                mat_array = scipy.io.loadmat(os.path.join(root, name))


                for key, value in mat_array.items():   #Iterating over the different keys(TriggerInfo and Data files)
                     if not key.startswith('_') and isinstance(value, (list, tuple, np.ndarray)):

                         # Ensure the array is 1D
                         flattened_data = np.ravel(value)

                         # Check if the length is divisible by 9
                         if len(flattened_data) % 9 == 0:
                             # Reshape the array into 9 columns
                             reshaped_data = np.reshape(flattened_data, (-1, 9))

                             # Create a DataFrame with 9 columns
                             data = pd.DataFrame(reshaped_data, columns=[f'Sensor_{i}' for i in range(9)])
                             print(data)
                             data = data.drop(data.columns[actuatorNum], axis=1)
                             print("NEW",data)
                             freq_arr.insert(actuatorNum-1,data)  #insterts data in the correct orden (in terms of which sensor acts as actuator) into a list


                         elif (key == "Trigger_Info"):
                             data = pd.DataFrame(flattened_data)
                             #data.to_csv(csv_file_path, index=False)
                         else:
                            pass
                            print(f"Data format invalid for key {key}. Skipping.")
                combined_df = pd.concat(freq_arr, axis=1) #combines all the entries in the list into a single dataframe
               #combined_df.columns = [f'Sensor{i}' for i in range(combined_df.shape[1])]

    return combined_df



