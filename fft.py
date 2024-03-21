import pandas as pd
import numpy as np
from numpy.fft import fft
from scipy import stats
from matplotlib import pyplot as plt

# Assuming df is your dataframe read from csv
df = pd.read_csv('Actionneur1/measured_data_rep_1_Time_Response.csv')

dataframes = [df[['Column_1', 'Column_2']], df[['Column_1', 'Column_3']], df[['Column_1', 'Column_4']],
              df[['Column_1', 'Column_5']], df[['Column_1', 'Column_6']], df[['Column_1', 'Column_7']],
              df[['Column_1', 'Column_8']], df[['Column_1', 'Column_9']]]

x_values = []
for df in dataframes:
    # Assuming the first column is time and the second column is the data to perform FFT on
    time = df.iloc[:, 0]
    data = df.iloc[:, 1]

    X = fft(data)
    print(X)
    N = len(X)
    n = np.arange(N)
    sr = 1 / (time.iloc[1] - time.iloc[0])  # Sample rate calculation
    T = N / sr
    freq = n / T
    print(freq)

    # Get the one-sided spectrum
    n_oneside = N // 2
    # get the one side frequency
    f_oneside = freq[:n_oneside]

    x_values.append(X[:n_oneside])

    plt.figure(figsize=(12, 6))
    plt.plot(f_oneside, np.abs(X[:n_oneside]), 'b')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.show()


#print("THIS IS THE OUTPUT!!!!!!! ", x_values)
print(len(x_values[1]))
print(len(x_values))

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
#
#     #skewness
#     features[2] = stats.skew(S)
#
#     #kurtosis
#     features[3] = stats.kurtosis(S)
#
#     return features
#
# print(make_features(x_values[0], samples=2000))