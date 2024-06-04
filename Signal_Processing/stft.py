import numpy as np
import pandas as pd
from scipy import signal

def Short_Fourier(data):
    amp_arr = []

    # Loop to process the data and perform STFT
    for i in range(7):
        for j in range(8):
            k = 8 * i + j
            y = data.iloc[:, k]  # Extracting data for FFT
            fs = 1 / 5e-7  # Sampling frequency

            # Perform Short-Time Fourier Transform (STFT)
            f, t, Zxx = signal.stft(y, fs, nperseg=250)
            amp = np.abs(Zxx)  # Get the absolute values of the STFT
            amp_arr.append(amp)  # Append amplitude data to the list

    # Flatten the amp_arr into a 1D list
    flat = [item for sublist in amp_arr for item in sublist]

    # Dimensions of the original 3D list
    dim1 = 56  # Assuming this is the number of elements in the outermost dimension (amount of signals 8*7, we dont include the actuator(would be 8*8))
    dim2 = 126  # Assuming this is the number of elements in the middle dimension (each segment has 17 frequencies)
    dim3 = 17  # Assuming this is the number of elements in the innermost dimension (number of segments, on which FFT is applied on seperatly)

    #print("outer Dim", len(amp_arr))
    #print("Middle Dim", len(amp_arr[0]))
    #print("Inner Dim", len(amp_arr[0][0]))
    # Create an empty 3D list with the original dimensions
    unflattened_list = [[[0 for _ in range(dim3)] for _ in range(dim2)] for _ in range(dim1)]


    # #Unflattening happens here
    # # Populate the unflattened_list with values from flat
    # index = 0
    # for i in range(dim1):
    #     for j in range(dim2):
    #             unflattened_list[i][j] = flat[index]
    #             index += 1

    # Convert the unflattened_list to a pandas DataFrame
    unflattened_df = pd.DataFrame(np.array(unflattened_list).reshape((dim1, dim2 * dim3)))
    # print('flat')
    # print(flat)
    # print('unflatten')
    # print(unflattened_list[-1])
    # print('amplitude array')
    # print(amp_arr[-1])
    return pd.DataFrame(flat)



#----------------------------------Remake 23--------------------------------------------



# import numpy as np
# from scipy import signal
# import pandas as pd
# import matplotlib.pyplot as plt
# from torch.utils.data.datapipes.dataframe import dataframes
#
# """
# # Assuming df is your dataframe read from csv
# df = pd.read_csv('Actionneur1/measured_data_rep_1_Time_Response.csv')
#
# dataframes = [df[['Column_1', 'Column_2']], df[['Column_1', 'Column_3']], df[['Column_1', 'Column_4']],
#               df[['Column_1', 'Column_5']], df[['Column_1', 'Column_6']], df[['Column_1', 'Column_7']],
#               df[['Column_1', 'Column_8']], df[['Column_1', 'Column_9']]]
# """
#
#
# def Short_Fourier(data):
#     freq_arr = []
#     amp_arr = []
#     flat = []
#     for i in range(8):
#         for j in range(8):
#             k = 8 * i + j
#             # Extracting data to perform FFT on
#             # x = data.iloc[:, 0]
#             y = data.iloc[:, k]
#
#             # time_intervals = np.diff(x)
#             # mean_interval = np.mean(time_intervals)
#             # dont need bc all our time intervals are the same
#             fs = 1 / 5e-7
#
#             f, t, Zxx = signal.stft(y, fs, nperseg=250)
#
#             # amp = np.abs(Zxx).max()
#             amp = np.abs(Zxx)
#             # print(amp)
#             # print(len(amp))
#             freq_arr.insert(k, f)
#             # amp_arr.insert(k, amp)
#             amp_arr.insert(k, amp)
#             # print(amp_arr)
#             # pcm = plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
#             # plt.title('STFT Magnitude')
#             # plt.ylabel('Frequency [Hz]')
#             # plt.xlabel('Time [sec]')
#             # cbar = plt.colorbar(pcm)
#             # cbar.set_label('Amplitude')
#             # plt.show()
#     print(amp_arr)
#     print(len(amp_arr))
#
#
#     for i in range(len(amp_arr)):
#          for j in range(len(amp_arr[0])):
#              flat.insert(i, amp_arr[i][j])
#
#     # # Dimensions of the original 3D list
#     # dim1 = len(amp_arr)
#     # dim2 = len(amp_arr[0])
#     # dim3 = len(amp_arr[0][0])
#     #
#     # # Create an empty 3D list with the original dimensions
#     # unflattened_list = [[[0 for _ in range(dim3)] for _ in range(dim2)] for _ in range(dim1)]
#     #
#     # # Populate the unflattened_list with values from flat
#     # index = 0
#     # for i in range(dim1):
#     #     for j in range(dim2):
#     #         for k in range(dim3):
#     #             unflattened_list[i][j][k] = flat[index]
#     #             index += 1
#     #
#     # # Now unflattened_list contains the original 3D structure
#     #
#     # reenlarged=[]
#     # # for i in range(17)
#     # #     for i in range(len(flat)):
#     # #         reenlarged.insert()
#
#
#     print(pd.DataFrame(flat))
#     return freq_arr, pd.DataFrame(flat)
#     #return pd.DataFrame(freq_arr), pd.DataFrame(amp_arr)
#
#
# data = pd.read_csv(r"C:\Users\Martin\Downloads\PZT-CSV\PZT-CSV-L01-5\L109_2019_12_18_17_49_44\State_9_2019_12_19_02_02_36\50kHz.csv")
# Short_Fourier(data)

#--------------------------Orignal code--------------------------------------------------------

"""
x_values = []
for df in dataframes:
    # Extract the time (x) and the data column (y) for analysis
    x = df.iloc[:, 0]  # Assuming Column_1 is your time column
    y = df.iloc[:, 1]  # This is the data you want to perform FFT on

    # Calculate sampling frequency fs
    # Assuming time is in seconds and intervals are regular
    # Calculate the difference between each time step and find the mean interval
    time_intervals = np.diff(x)
    mean_interval = np.mean(time_intervals)
    fs = 1 / mean_interval  # Frequency is the inverse of the time interval

    # STFT calculation
    f, t, Zxx = signal.stft(y, fs, nperseg=1000)
    amp = np.abs(Zxx).max()  # You might want to dynamically set vmax for better visualization

    # Plotting

    pcm = plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    cbar = plt.colorbar(pcm)
    cbar.set_label('Amplitude')
    plt.show()


#print("THIS IS THE OUTPUT!!!!!!! ", x_values)
#print(len(x_values[1]))
#print(len(x_values))

# def make_features(sensor, samples):
#     #np 1D array of fft
#     features = np.empty(4)
#
#     S = np.abs(sensor**2)/samples
#
#     #Mean
#     features[0] = np.mean(S)
#
#     #variance
#     features[1] = np.var(S)

"""