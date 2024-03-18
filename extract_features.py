import numpy as np
from scipy import stats
import test_data

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