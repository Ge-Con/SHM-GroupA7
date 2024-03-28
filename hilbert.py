import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
import pandas as pd

# Assuming df is your dataframe read from csv
df = pd.read_csv('Actionneur1/measured_data_rep_1_Time_Response.csv')

dataframes = [df[['Column_1', 'Column_2']], df[['Column_1', 'Column_3']], df[['Column_1', 'Column_4']],
              df[['Column_1', 'Column_5']], df[['Column_1', 'Column_6']], df[['Column_1', 'Column_7']],
              df[['Column_1', 'Column_8']], df[['Column_1', 'Column_9']]]

x_values = []
for df in dataframes:
    time = df.iloc[:, 0]
    data = df.iloc[:, 1]

    time_intervals = np.diff(time)
    mean_interval = np.mean(time_intervals)
    fs = 1 / mean_interval

    analytic_signal = hilbert(data)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) /
                               (2.0*np.pi) * fs)
    # if statement tu cut off noise
    mask = np.logical_and(time[:-1] >= 0.0006, np.abs(instantaneous_frequency) >= 100000)
    instantaneous_frequency[mask] = 0

    fig, (ax0, ax1) = plt.subplots(nrows=2)
    ax0.plot(time, data, label='signal')
    ax0.plot(time, amplitude_envelope, label='envelope')
    ax0.set_xlabel("time in seconds")
    ax0.legend()
    ax1.plot(time[1:], instantaneous_frequency)
    ax1.set_xlabel("time in seconds")
    fig.tight_layout()
    plt.show()