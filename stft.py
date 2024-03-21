import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your dataframe read from csv
df = pd.read_csv('Actionneur1/measured_data_rep_1_Time_Response.csv')

dataframes = [df[['Column_1', 'Column_2']], df[['Column_1', 'Column_3']], df[['Column_1', 'Column_4']],
              df[['Column_1', 'Column_5']], df[['Column_1', 'Column_6']], df[['Column_1', 'Column_7']],
              df[['Column_1', 'Column_8']], df[['Column_1', 'Column_9']]]

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