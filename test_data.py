import numpy as np
import matplotlib.pyplot as plt

###This just generates fourier transforms of simple waves as test data

# sampling rate
sr = 100

def DFT(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape(N, 1)
    e = np.exp(-2j*np.pi*k*n/N)
    X = np.dot(e, x)
    return X

def inv_DFT(X):
    N = len(X)
    n = np.arange(N)
    k = n.reshape(N, 1)
    x = np.dot(X, np.exp(2j*np.pi*k*n/N))
    return x

def plot(x, y, label_x, label_y, col='r'):
    plt.figure(figsize = (8, 6))
    if col == 'b':
        plt.stem(x, y, 'b', markerfmt=" ", basefmt='b')
    else:
        plt.plot(x, y, col)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.show()

def gen(sr):
    
    # sampling interval
    ts = 1.0/sr
    t = np.arange(0,1,ts)
    freq = [1, 5, 7]
    amp = [3, 1, 0.5]
    x = np.zeros(len(t))

    for wave in range(len(freq)):
        x += amp[wave]*np.sin(2*np.pi*freq[wave]*t)

    #plot(t, x, 'Time', 'Amplitude')
    return x



#Inverse transform
#x = inv_DFT(X)
#t = np.arange(0,1,1/sr)
#plot(t, x, 'Time', 'Amplitude')

def get_data():
    #DFT
    X = DFT(gen(sr))

    N = len(X)
    n = np.arange(N)
    T = N/sr
    freq = n/T

    # Symmetrical about half of sample rate, so disregard upper half
    n_oneside = N//2
    f_oneside = freq[:n_oneside]
    X_oneside = abs(X[:n_oneside]/n_oneside) #Have to normalise amplitude
    plot(f_oneside, X_oneside, 'Freq (Hz)', 'DFT Amplitude |X(freq)|', 'b')
    return X_oneside
