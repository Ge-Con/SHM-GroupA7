import numpy as np
from scipy import stats
import test_data

#Frequency domain
def make_features(sensor, samples):
    #np 1D array of fft
    features = np.empty(4)

    S = np.abs(sensor**2)/samples

    #Mean
    features[0] = np.mean(S)

    #variance
    features[1] = np.var(S)

    #skewness
    features[2] = stats.skew(S)

    #kurtosis
    features[3] = stats.kurtosis(S)

    return features

#print(make_features(test_data.get_data(), 2000))


#Time domain
def Time_domain_features(sensor):
    # np 1D array of time domain data
    features = np.empty(2)

    X = sensor.values
    # Mean
    features[0] = np.mean(X)

    # Standard deviation
    features[1] = np.std(X)

    #Root mean squared RMS
    features[2] = np.sqrt(np.mean(X ** 2))

    #Root standard squared RSS
    features[3] = np.sqrt(np.sum(X ** 2))

    #Peak (maximum)
    features[4] = np.max(X)

    #Skewness
    features[5] = stats.skew(X)

    #Kurtosis
    features[6] = stats.kurtosis(X)

    #Crest factor
    features[7] = np.max(X)/np.sqrt(np.mean(X ** 2))





    return features
