import numpy as np
from scipy import stats
import pandas as pd
import test_data


                                 ## Frequency domain ##
def Frequency_domain_features(sensor):
    """
       Extracts frequency domain features from sensor data.

       Parameters:
       - sensor (1D array): Frequency domain transform of data.

       Returns:
       - F_features (1D array): Array containing frequency domain features.

       Example:

        # Example usage of the function
       result = Frequency_domain_features(sensor_data)
    """

    #np 1D array of fft
    F_features = np.empty(14)

    #Power Spectral Density
    #S = np.abs(sensor**2)/samples
    S = sensor

    #frequency value kth spectrum line (needs adjustment)
    F = np.arange(1000, len(S)*1000+1, 1000)
    F_small = F/1000

    #Mean
    F_features[0] = np.mean(S)

    #Variance
    F_features[1] = np.var(S)

    #Skewness
    F_features[2] = stats.skew(S)

    #Kurtosis
    F_features[3] = stats.kurtosis(S)

    #P5 (Xfc)
    F_features[4] = np.sum(F * S) / np.sum(S)

    # P6
    F_features[5] = np.sqrt(np.mean( S * (F - F_features[4]) ** 2))

    #P7 (Xrmsf)
    F_features[6] = np.sqrt((np.sum(S * F_small ** 2)) / np.sum(S))*1000

    #P8
    F_features[7] = np.sqrt(np.sum(S * F_small ** 4) / np.sum(S * F_small ** 2))*1000

    #P9
    F_features[8] = np.sum(S * F_small ** 2) / (np.sqrt( np.sum(S) * np.sum(S * F_small ** 4)))

    #P10
    F_features[9] = F_features[5] / F_features[4]

    # #P11
    F_features[10] = np.mean(S * (F - F_features[4]) ** 3)/(F_features[5] ** 3)

    #P12
    F_features[11] = np.mean(S * (F - F_features[4]) ** 4)/(F_features[5] ** 4)

    #P13
    #Including forced absolute in sqrt which wasn't meant to be there
    F_features[12] = np.mean(np.sqrt(np.abs(F - F_features[4]))*S)/np.sqrt(F_features[5])

    #P14
    F_features[13] = np.sqrt(np.sum((F - F_features[4])**2*S)/np.sum(S))

    return F_features

#print(Frequency_domain_features(test_data.get_data(), 2000))


                                   ## Time domain ##
def Time_domain_features(sensor):
    """
        Extracts time domain features from sensor data.

        Parameters:
        - sensor (1D array): Array containing sensor data.

        Returns:
        - T_features (1D array): Array containing time domain features.

        Example:

        # Example usage of the function
        result = Time_domain_features(sensor_data)
    """

    # np 1D array of time domain data
    T_features = np.empty(19)

    X = sensor

    # Mean
    T_features[0] = np.mean(X)

    # Standard deviation
    T_features[1] = np.std(X)

    #Root amplitude
    T_features[2] = ((np.mean(np.sqrt(abs(X)))) ** 2)

    #Root mean squared RMS
    T_features[3] = np.sqrt(np.mean(X ** 2))

    #Root standard squared RSS
    T_features[4] = np.sqrt(np.sum(X ** 2))

    #Peak (maximum)
    T_features[5] = np.max(X)

    #Skewness
    T_features[6] = stats.skew(X)

    #Kurtosis
    T_features[7] = stats.kurtosis(X)

    #Crest factor
    T_features[8] = np.max(X) / np.sqrt(np.mean(X ** 2))

    #Clearance factor
    T_features[9] = np.max(X) / T_features[2]

    #Shape factor
    T_features[10] = np.sqrt(np.mean(X ** 2)) / np.mean(X)

    #Impulse factor
    T_features[11] = np.max(X) / np.mean(X)

    #Max-Min difference
    T_features[12] = np.max(X) - np.min(X)

    #Central moment kth order (not good enough)
    for k in range(3, 7):
        T_features[10+k] = np.mean((X - T_features[0])**k)

    #FM4 (close to kurtosis) (need central moment)
    T_features[17] = T_features[14]/T_features[1]**4

    #Median
    T_features[18] = np.median(X)

    return T_features

def feature_correlation(features):
    """
       Filters features based on correlation coefficient threshold.

       Parameters:
       - features (2D array): Feature data for each trial.

       Returns:
       - features (2D array): Reduced statistically significant feature array.
       - to_keep (list): Indices of features contained in returned array

       Example:

        # Example usage of the function
       result, indices = feature_correlation(feature_data)
    """
    #2D array

    correlation_matrix = np.corrcoef(features.T)
    correlation_threshold = 0.95
    correlation_bool = correlation_matrix > correlation_threshold

    to_delete = []

    for column in range(len(correlation_bool)):
        for row in range(column+1, len(correlation_bool)):
            if correlation_bool[column, row] == True and row not in to_delete:
                to_delete.append(row)

    to_delete.sort()
    #print("TODELETE: ", to_delete)

    #print(features)
    features = np.delete(features, to_delete, axis=1)
    #features = features.transpose()

    # indices = set(np.arange(len(features)))
    # to_keep = np.array(list(indices - set(to_delete)))

    return correlation_matrix, features, np.array(to_delete)

def time_to_feature(data):
    """
        Converts time domain sensor data to feature data.

        Parameters:
        - data (2D array): Time domain sensor data.

        Returns:
        - features (2D array): Feature data extracted from time domain data.

        Example:

        # Example usage of the function
        result = time_to_feature(sensor_data)
    """
    data = np.array(data).transpose()
    features = np.empty((len(data), 19))
    for i in range(len(data)):
        features[i] = Time_domain_features(data[i])

    # print(pd.DataFrame(y))
    # print(len(y))
    return pd.DataFrame(features).transpose()


def freq_to_feature(data):
    """
        Converts frequency domain sensor data to feature data.

        Parameters:
        - data (2D array): Frequency domain sensor data.

        Returns:
        - features (2D array): Feature data extracted from frequency domain data.

        Example:

        # Example usage of the function
        result = freq_to_feature(sensor_data)
    """

    data = np.array(data).transpose()

    features = np.empty((len(data), 14))

    for i in range(len(data)):
        features[i] = Frequency_domain_features(data[i])

    return pd.DataFrame(features).transpose()

#print(freq_to_feature([[2, 3, 4, 5], [1, 2, 3, 5], [4, 5, 7, 4], [1, 3, 5, 7]]))
#
# data = pd.read_csv(r"C:\Users\geort\Desktop\Universty\PZT-CSV-L1-03\L103_2019_12_06_14_02_38\State_1_2019_12_06_14_02_38\50_kHz_FFT_Freq.csv")
# data = np.array(data)
# print(freq_to_feature(data))
# print("last")

def STFT_domain_features(sensor):
    #np 1D array main features

    FT_features = np.empty(4)
    Y = sensor

    #print(sensor[0])

    FT_features[0] = np.mean(Y)
    FT_features[1] = np.std(Y)
    FT_features[2] = stats.skew(Y)
    FT_features[3] = stats.kurtosis(Y)

    return FT_features

# def STFT_to_feature(data3d):
#     out_list = []
#     features = 68*[0]
#     #print(data3d)
#     # Create the array using list comprehension
#     new_flat = np.array([[0 for _ in range(56)] for _ in range(2142)])
#
#     for path in range(56):
#         current_path = data3d[path]
#
#         for i in range(17):
#             for j in range(126):
#                 #print(j+126*i)
#                 new_flat[j+126*i][path] = current_path[j][i]
#     print(new_flat[1][1])
#     # new_flat2 = np.array([[0 for _ in range(56)] for _ in range(68)])
#     new_flat2= []
#
#     for path2 in range(56):
#         for i in range(17):
#             segment = new_flat[i*126:(i+1)*126, path2]
#             for feat in range(4):
#                 #print(segment[5])
#                 features[i*4+feat] = STFT_domain_features(segment)[feat]
#         new_flat2.append(features)
#     #print(new_flat2)
#         #out_list.append(features)
#
#     return pd.DataFrame(new_flat)






# def STFT_to_feature(data3d):
#    out_list = []
#    for path in range(len(data3d)):
#        print(data3d)
#        current_path = data3d[path] #current data is a 17x126 matrix
#
#        print(current_path)
#        features = np.empty((len(current_path), 4))
#        for i in range(len(current_path)):
#            features[i] = STFT_domain_features(current_path[i])
#         out_list.append(features)
#     print(out_list)
#     return pd.DataFrame(features).transpose()


def STFT_to_feature(data3d):
    out_list = []
    features = [0]*68

    new_flat = np.zeros((2142, 56))

    for path in range(56):
        current_path = data3d[path]
        for i in range(126):
            for j in range(17):
                new_flat[i + j * 126][path] = current_path[i][j]

    for path2 in range(56):
        for i in range(17):
            segment = new_flat[i * 126:(i + 1) * 126, path2]
            segment_features = STFT_domain_features(segment)
            features[i * 4:(i + 1) * 4] = segment_features
        out_list.append(features.copy())

    features_df = pd.DataFrame(out_list).transpose()
    return features_df

