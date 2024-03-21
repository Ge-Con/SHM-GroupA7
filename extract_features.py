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


#Time domain
def Time_domain_features(sensor):
    # np 1D array of time domain data
    features = np.empty(2)

    # Mean
    features[0] = np.mean(sensor)

    # Standard deviation
    features[1] = np.std(sensor)

    #RMS
    Rms.append(np.sqrt(np.mean(X ** 2)))

    return features
