import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
import pandas as pd


# Assuming df is your dataframe read from csv
def Hilbert(data, time):
    x_values = []
    time = pd.DataFrame(time)
    inst_freq_arr = []
    amp_arr = []
    for i in range(8):
        x_values = []
        for j in range(8):
            k = 8 * i + j
            # Extracting data to perform FFT on
            x = data.iloc[:, k]
            time_intervals = np.diff(time)
            #mean_interval = np.mean(time_intervals)
            fs = 1 / 5e-7

            analytic_signal = hilbert(x)
            amplitude_envelope = np.abs(analytic_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * fs)

            # if statement tu cut off noise
            # print(time[:-1])
            # print(instantaneous_frequency)
            #mask = np.logical_and(time[:-1] >= 0.0006, np.abs(instantaneous_frequency) >= 100000)
            #instantaneous_frequency[mask] = 0


            inst_freq_arr.insert(k, amplitude_envelope)
    #print(pd.DataFrame(inst_freq_arr))
    return pd.DataFrame(inst_freq_arr).transpose()
    #
    # for i in range(8):
    #     data = df.iloc[:, i]
    #
    #     time_intervals = np.diff(time)
    #     mean_interval = np.mean(time_intervals)
    #     fs = 1 / 5e-7
    #
    #
    #     analytic_signal = hilbert(data)
    #     amplitude_envelope = np.abs(analytic_signal)
    #     instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    #     instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * fs)
    #
    #     # if statement tu cut off noise
    #     mask = np.logical_and(time[:-1] >= 0.0006, np.abs(instantaneous_frequency) >= 100000)
    #     instantaneous_frequency[mask] = 0
    #
        # save instantaneous freq
        # fig, (ax0, ax1) = plt.subplots(nrows=2)
        # ax0.plot(time, data, label='signal')
        # ax0.plot(time, amplitude_envelope, label='envelope')
        # ax0.set_xlabel("time in seconds")
        # ax0.legend()
        # ax1.plot(time[1:], instantaneous_frequency)
        # ax1.set_xlabel("time in seconds")
        # fig.tight_layout()
        # plt.show()
#GEORGE EDLYN HERE:)
#the instantaneous frequency(after mask) is what needs to be saved
#thus line 24


def giveTime():
    time = []
    for i in range(2000):
        time.append(i*(5e-7))
    return pd.DataFrame(time)

# def Hilbert(data, time):
#     freq_arr = []
#     amp_arr = []
#     for i in range(8):
#         for j in range(8):
#             k = 8 * i + j
#             x = data.iloc[:, k]
#             fs = 1 / 5e-7
#
#             analytic_signal = hilbert(x)  # bruv rename the file
#             amplitude_envelope = np.abs(analytic_signal)
#             instantaneous_phase = np.unwrap(np.angle(analytic_signal))
#             instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * fs)
#             # if statement tu cut off noise
#             mask = np.logical_and(time[:-1] >= 0.0006, np.abs(instantaneous_frequency) >= 100000)
#             instantaneous_frequency[mask] = 0
#
#             freq_arr.insert(k - 1, instantaneous_frequency)
#             amp_arr.insert(k - 1, amplitude_envelope)
#     return pd.DataFrame(freq_arr), pd.DataFrame(amp_arr)


"Shape of passed values is (1999, 1999), indices imply (1999, 1) "

"""
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
"""

