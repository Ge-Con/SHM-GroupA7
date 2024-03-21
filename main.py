import scipy.io
import pandas as pd
import numpy as np
import os
import csv
import pyexcel as p


pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 400)
np.set_printoptions(linewidth=400)


def list_files(dir):
    csv_dir = dir.replace('PZT','PZT-CSV')
    #     dir2 = dir
    #     csv_dir = dir2.replace('PZT', 'PZT-CSV')

    # Create the output directory if it doesn't exist
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    #
    for root, dirs, files in os.walk(dir):
         for name in dirs:
             if name.endswith('cycles'):
                print(name)
                root2 = root
                root_new = root2.replace('PZT', 'PZT-CSV')
                if not os.path.exists(root_new):
                    os.makedirs(root_new)
                arrayfile = combine_act(os.path.join(root,name))
               # print(arrayfile[0])
                #flatlist = [item for sublist1 in arrayfile for item in sublist1]
                #print(flatlist)
                makeCSV(arrayfile[0],arrayfile[1])

                csv_file_path = os.path.join(root_new, f"{name.replace('.mat', '')}_{"test2"}.csv")
                arrayfile[0].to_csv(csv_file_path, index=False)

    # for freq in state:
    #     combine_act(freq)



# for filename in files:
#     if filename.endswith('rep_1.mat'):
#
#


#
#     for root, dirs, files in os.walk(dir):
#         for name in files:
#             if name.endswith('rep_1.mat'):
#                 mat_array = scipy.io.loadmat(os.path.join(root, name))
#
#                 root2 = root
#                 root_new = root2.replace('PZT', 'PZT-CSV')
#                 if not os.path.exists(root_new):
#                     os.makedirs(root_new)
#
#                 for key, value in mat_array.items():
#                     if not key.startswith('_') and isinstance(value, (list, tuple, np.ndarray)):
#                         csv_file_path = os.path.join(root_new, f"{name.replace('.mat', '')}_{key}.csv")
#                         print(f"Creating {csv_file_path}")
#                         #chanbe
#
#                         # Ensure the array is 1D
#                         flattened_data = np.ravel(value)
#
#                         # Check if the length is divisible by 9
#                         if len(flattened_data) % 9 == 0:
#                             # Reshape the array into 9 columns
#                             reshaped_data = np.reshape(flattened_data, (-1, 9))
#
#                             # Create a DataFrame with 9 columns
#                             data = pd.DataFrame(reshaped_data, columns=[f'Column_{i + 1}' for i in range(9)])
#
#                             # Write the DataFrame to CSV
#                             data.to_csv(csv_file_path, index=False)
#                         elif (key == "Trigger_Info"):
#                             data = pd.DataFrame(flattened_data)
#                             data.to_csv(csv_file_path, index=False)
#                         else:
#                             pass
#                         print(f"Data format invalid for key {key}. Skipping.")


def combine_act(freq):
    freq_arr = []
    for root, dirs, files in os.walk(freq):
         for name in files:
             if name.endswith('rep_1.mat'):
                locat = os.path.join(root, name)
                actuatorNum = int(locat[-25])
                mat_array = scipy.io.loadmat(os.path.join(root, name))



                for key, value in mat_array.items():
                     if not key.startswith('_') and isinstance(value, (list, tuple, np.ndarray)):
                         #chanbe

                         # Ensure the array is 1D
                         flattened_data = np.ravel(value)

                         # Check if the length is divisible by 9
                         if len(flattened_data) % 9 == 0:
                             # Reshape the array into 9 columns
                             reshaped_data = np.reshape(flattened_data, (-1, 9))

                             # Create a DataFrame with 9 columns
                             data = pd.DataFrame(reshaped_data, columns=[f'Column_{i + 1}' for i in range(9)])
                             freq_arr.insert(actuatorNum-1,data)
                             # Write the DataFrame to CSV
                             # data.to_csv(csv_file_path, index=False)
                             #print("DONE")
                         elif (key == "Trigger_Info"):
                             data = pd.DataFrame(flattened_data)
                             #data.to_csv(csv_file_path, index=False)
                         else:
                            pass
                            print(f"Data format invalid for key {key}. Skipping.")
    return freq_arr


def makeCSV(list1,list2):
    p.isave_as(array=(a+b for a,b in zip(list1,list2)), dest_file_name='mergeddd.csv')
    p.free_resources()
    print("doine")



list_files(r"C:\Users\edlyn\Desktop\PZT-L1-03") #This should be the directory leading up to the main PZT file.
#print(combine_act(r"C:\Users\edlyn\Desktop\PZT-L1-03\L103_2019_12_06_14_02_38\State_1_2019_12_06_14_02_38\50kHz_5cycles"))
