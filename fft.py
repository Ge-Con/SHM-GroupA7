import pandas as pd
import numpy as np
from numpy.fft import fft
from scipy import stats
from matplotlib import pyplot as plt

def fast_fourier(data):
    freq_arr = []
    amp_arr = []
    for i in range(8):
        x_values = []
        for j in range(8):
            k = 8 * i + j
            # Extracting data to perform FFT on
            x = data.iloc[:, k]

            X = fft(x)
            N = len(X)
            n = np.arange(N)
            sr = 1 / 5e-7 # Sample rate calculation
            T = N / sr                              # total time
            freq = n / T

            # Get the one-sided spectrum
            n_oneside = N // 2
            # get the one side frequency
            f_oneside = freq[:n_oneside]

            x_values.append(X[:n_oneside])  # why do we need this line?

            freq_arr.insert(k,f_oneside)
            amp_arr.insert(k,X[:n_oneside])
    #print(pd.DataFrame(amp_arr))
    return pd.DataFrame(freq_arr), pd.DataFrame(amp_arr)


'''
def fast_fourier(data):

    timecol = data.iloc[:,0]
    # freqcol = np.array([])
    # ampcol = np.array([])
    freq_arr = []
    amp_arr = []
    for i in range(8):
        x_values = []
        for j in range(8):
            k = 8 * i + j
            # Extracting data to perform FFT on
            x = data.iloc[:, k]

            X = fft(x)
            #print(X)
            N = len(X)
            n = np.arange(N)
            sr = 1 / 5e-7 # Sample rate calculation
            T = N / sr                              # total time
            freq = n / T
            #print(freq)

            # Get the one-sided spectrum
            n_oneside = N // 2
            # get the one side frequency
            f_oneside = freq[:n_oneside]

            x_values.append(X[:n_oneside])  #why do we need this line?

            # freq_arr = np.append(freq_arr, f_oneside, axis=1)
            # amp_array = np.append(amp_array, np.abs(X[:n_oneside]), axis=1)
            freq_arr.insert(k-1,f_oneside)
            amp_arr.insert(k-1,X[:n_oneside])
            # ampcol = np.append(ampcol, np.abs(X[:n_oneside]))

            # plt.figure(figsize=(12, 6))
            # plt.plot(f_oneside, np.abs(X[:n_oneside]), 'b')
            # plt.xlabel('Freq (Hz)')
            # plt.ylabel('FFT Amplitude |X(freq)|')
            # plt.show()
            #print("done ",k)
        #print(x_values[0])

    #print("THIS IS THE OUTPUT!!!!!!! ", x_values)
    # print(len(x_values[1]))
    # print(len(x_values))
    return pd.DataFrame(freq_arr), pd.DataFrame(amp_arr)
'''