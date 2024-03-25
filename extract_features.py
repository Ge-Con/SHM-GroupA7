import numpy as np
from scipy import stats
import test_data

                                 ## Frequency domain ##
def Frequency_domain_features(sensor, samples):
    #np 1D array of fft
    F_features = np.empty(4)
    S = np.abs(sensor**2)/samples

    #frequency value kth spectrum line (needs adjustment)
    f_k = sensor.values

    #Mean
    F_features[0] = np.mean(S)

    #Variance
    F_features[1] = np.var(S)

    #Skewness
    F_features[2] = stats.skew(S)

    #Kurtosis
    F_features[3] = stats.kurtosis(S)

    #P5 (Xfc)
    F_features[4] = np.sum(f_k * S) / np.sum(S)

    #P6
    F_features[5] = np.sqrt(np.mean( S * (f_k - (np.sum(f_k * S) / np.sum(S))) ** 2))

    #P7 (Xrmsf)
    F_features[6] = np.sqrt((np.sum(S * f_k ** 2)) / np.sum(S))

    #P8
    F_features[7] = np.sqrt( np.sum(S * f_k ** 4) / np.sum(S * f_k ** 2))

    #P9
    F_features[8] = np.sum(S * f_k ** 2) / (np.sqrt( np.sum(S) * np.sum(S * f_k ** 4)))

    #P10
    F_features[9] = F_features[5] / F_features[4]

    #P11
    F_features[10] = \frac{1}{F_features[5] ** 3} * np.mean(S * (f_k - F_features[4]) ** 3)

    #P12
    F_features[11] = \frac{1}{F_features[5] ** 4} * np.mean(S * (f_k - F_features[4]) ** 4)

    #P13
    F_features[12] = \frac{1}{F_features[5] ** 3} * np.mean(S * (f_k - F_features[4]) ** 3)

    return F_features

#print(Frequency_domain_features(test_data.get_data(), 2000))


                                   ## Time domain ##
def Time_domain_features(sensor):
    # np 1D array of time domain data
    T_features = np.empty(14)

    X = sensor.values
    # Mean
    T_features[0] = np.mean(X)

    # Standard deviation
    T_features[1] = np.std(X)

    #Median
    T_features[2] = np.median(X)

    #Root amplitude
    T_features[3] = ((np.mean(np.sqrt(X))) ** 2)

    #Root mean squared RMS
    T_features[4] = np.sqrt(np.mean(X ** 2))

    #Root standard squared RSS
    T_features[5] = np.sqrt(np.sum(X ** 2))

    #Peak (maximum)
    T_features[6] = np.max(X)

    #Skewness
    T_features[7] = stats.skew(X)

    #Kurtosis
    T_features[8] = stats.kurtosis(X)

    #Crest factor
    T_features[9] = np.max(X) / np.sqrt(np.mean(X ** 2))

    #Clearance factor
    T_features[10] = np.max(X) / ((np.mean(np.sqrt(X))) ** 2)

    #Shape factor
    T_features[11] = np.sqrt(np.mean(X ** 2)) / np.mean(X)

    #Impulse factor
    T_features[12] = np.max(X) / np.mean(X)

    #Max-Min difference
    T_features[13] = np.max(X) - np.min(X)

    #Central moment kth order (not good enough)
    #T_features[14] = \frac{1}{n} \sum_{i = 1}^n (x_i - \bar{x})^k

    #FM4 (close to kurtosis) (need central moment)
    #T_features[15] = (\frac{1}{n} \sum_{i = 1}^n (x_i - \bar{x})^k) / (np.std(X))

    return T_features
