import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

import extract_features
import PCA
from Signal_Processing import fft, emdfinal, stft, hilbert, Data_Preprocess
from prognosticcriteria_v2 import fitness
from DeepSAD import DeepSAD_train_run
import Graphs
import SP_save as SP
from Interpolating import scale_exact

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 400)
np.set_printoptions(linewidth=400)

np.set_printoptions(precision=4, suppress=True)


def saveFeatures(dir):
    """
        Feature Extraction

        Args:
            dir (str): The directory path containing .csv files to process.
        Returns:
            return_type: csv files in the directory of origin for different SPs
        Example:
            saveFeatures('/path/to/directory')
         Notes:
            This function processes files with names ending in 'kHz.csv', 'FFT.csv', 'HLB.csv',
            'EMD.csv' and 'SFT.csv' and saves the features in a new .csv file with '_FT'
            appended before the extension.
    """
    #Extract and save features for each frequency
    print("Extracting Features:...")
    for root, dirs, files in os.walk(dir):
        for name in files:

            if name.endswith('kHz.csv'):
                data = pd.read_csv(os.path.join(root, name))    #Don't put this line outside if statement
                features = extract_features.time_to_feature(data)
                new_filename = name.replace('kHz.csv', 'kHz_FT.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            elif name.endswith('FFT.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = extract_features.freq_to_feature(data)
                new_filename = name.replace('FFT.csv', 'FFT_FT.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            elif name.endswith('HLB.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = extract_features.time_to_feature(data)
                new_filename = name.replace('HLB.csv', 'HLB_FT.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            elif name.endswith('EMD.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = extract_features.time_to_feature(data)
                new_filename = name.replace('EMD.csv', 'EMD_FT.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            elif name.endswith('SFT.csv'):
                data = pd.read_csv(os.path.join(root, name))
                data3d = [[[0 for _ in range(17)] for _ in range(126)] for _ in range(56)]

                for k in range(56):
                    for i in range(126):
                        for j in range(17):
                            data3d[k][i][j] = data.iloc[126*k + i, j]

                features = extract_features.STFT_to_feature(data3d)
                new_filename = name.replace('SFT.csv', 'SFT_FT.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

def FFT_HLB_Reduced_Feat(dir):
    print("Extracting Features:...")
    for root, dirs, files in os.walk(dir):
        for name in files:

            if name.endswith('FFT.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = extract_features.FFT_Feat_reduced(data)
                new_filename = name.replace('FFT.csv', 'FFT_FT_Reduced.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            elif name.endswith('HLB.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = extract_features.HLB_Feat_reduced(data)
                new_filename = name.replace('HLB.csv', 'HLB_FT_Reduced.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)


def correlateALLFeatures(rootdir): #changed name to ALL
    # i dont think we're using this anymore
    # Correlate extracted features
    frequencies = ["050", "100", "125", "150", "200", "250"]
    samples = ["PZT-CSV-L1-03", "PZT-CSV-L1-04", "PZT-CSV-L1-05", "PZT-CSV-L1-09", "PZT-CSV-L1-23"]
    print("Combining Features:...")
    for sample in samples:
        dir = rootdir + "\\" + sample
        for root, dirs, files in os.walk(dir):  #For each folder location (state)
            all_feat = np.empty((6, 5), dtype=object)
            flag = False
            for name in files:
                for freq in frequencies:
                    #Read and add to correct position in all_feat array
                    if freq in name and name.endswith('_FT.csv'):
                        flag = True
                        data = np.array(pd.read_csv(os.path.join(root, name)))
                        if 'FFT' in name:
                            all_feat[frequencies.index(freq)][1] = data
                        elif 'HLB' in name:
                            all_feat[frequencies.index(freq)][2] = data
                        elif 'EMD' in name:
                            all_feat[frequencies.index(freq)][3] = data
                        elif 'SFT' in name:
                            all_feat[frequencies.index(freq)][4] = data
                        else: #Time domain
                            all_feat[frequencies.index(freq)][0] = data

            if flag:    #If at least one file was the correct type
                for freq in ["050", "100", "125", "150", "200", "250"]:
                    combinedfeatures = np.concatenate([all_feat[frequencies.index(freq), i] for i in range(all_feat[0].shape[0])], axis=0)
                    root2 = root
                    root_new = root2.replace("PZT", "PZT-ONLY-FEATURES")
                    if not os.path.exists(root_new):
                        os.makedirs(root_new)
                    csv_file_path1 = os.path.join(root_new, freq +"kHz_AF.csv") #AF = ALL FEATURES
                    csv_file_path = os.path.join(root, freq + "kHz_AF.csv")
                    pd.DataFrame(combinedfeatures).to_csv(csv_file_path1, index=False)
                    pd.DataFrame(combinedfeatures).to_csv(csv_file_path, index=False)

        #Average features at each state
        print("Averaging features...")
        meanfeatures = np.empty((6), dtype=object)
        for root, dirs, files in os.walk(dir):
            for name in files:
                if name.endswith("kHz_AF.csv"):
                    data = np.array(pd.read_csv(os.path.join(root, name)))
                    data = np.mean(data, axis=1)
                    if str(type(meanfeatures[frequencies.index(name[:3])])) == "<class 'NoneType'>":
                        meanfeatures[frequencies.index(name[:3])] = np.array([data])
                    else:
                        meanfeatures[frequencies.index(name[:3])] = np.concatenate((meanfeatures[frequencies.index(name[:3])], np.array([data])))

        #Correlate features
        alldelete = np.empty((6), dtype=object)
        for freq in frequencies:    #For each feature
            csv_file_path = os.path.join(dir, freq + "kHz_MF.csv") # MF = Mean Features
            pd.DataFrame(meanfeatures[frequencies.index(freq)]).to_csv(csv_file_path, index=False)  #Read mean features from file

            #Use feature_correlation to return the correlation matrix, reduced matrix and features deleted
            correlation_matrix, features, to_delete = extract_features.feature_correlation(meanfeatures[frequencies.index(freq)])
            #Save all to files
            alldelete[frequencies.index(freq)] = to_delete
            csv_file_path = os.path.join(dir, freq + "kHz_CF.csv") # CM = Correlated feature matrix
            pd.DataFrame(correlation_matrix).to_csv(csv_file_path, index=False)
            csv_file_path = os.path.join(dir, freq + "kHz_RF.csv") # RF = Remainder features
            pd.DataFrame(features).to_csv(csv_file_path, index=False)
        # Save lists of deleted features for inspection
        csv_file_path = os.path.join(dir, "DF.csv") # DF = Deleted features
        pd.DataFrame(alldelete).to_csv(csv_file_path, index=False)

def AverageFeatures(rootdir):
    # average features of all frequencies
    frequencies = ["050", "100", "125", "150", "200", "250"]
    samples = ["PZT-CSV-L1-03", "PZT-CSV-L1-04", "PZT-CSV-L1-05", "PZT-CSV-L1-09", "PZT-CSV-L1-23"]
    print("Combining Features:...")
    for sample in samples:
        dir = os.path.join(rootdir, sample)
        for root, dirs, files in os.walk(dir):  # For each folder location (state)
            all_feat = np.empty((6, 5), dtype=object)
            all_feat[:]
            flag = False

            for name in files:  # For each file
                for freq in frequencies:  # For each frequency
                    # Read and add to correct position in all_feat array
                    if freq in name and name.endswith('_FT.csv'):
                        flag = True
                        data = np.array(pd.read_csv(os.path.join(root, name)))
                        if 'FFT' in name:
                            all_feat[frequencies.index(freq)][1] = data
                        elif 'HLB' in name:
                            all_feat[frequencies.index(freq)][2] = data
                        elif 'EMD' in name:
                            all_feat[frequencies.index(freq)][3] = data
                        elif 'SFT' in name:
                            all_feat[frequencies.index(freq)][4] = data
                        else:  # Time domain
                            all_feat[frequencies.index(freq)][0] = data

            if flag:  # If at least one file was the correct type
                for freq in frequencies:
                    feature_list = [all_feat[frequencies.index(freq), i] for i in range(all_feat.shape[1]) if
                                    all_feat[frequencies.index(freq), i] is not None]
                    if feature_list:
                        combinedfeatures = np.concatenate(feature_list, axis=0)
                        root2 = root
                        root_new = root2.replace("PZT", "PZT-ONLY-FEATURES")
                        if not os.path.exists(root_new):
                            os.makedirs(root_new)
                        csv_file_path1 = os.path.join(root_new, freq + "kHz_AF.csv")  # AF = ALL FEATURES
                        csv_file_path = os.path.join(root, freq + "kHz_AF.csv")
                        pd.DataFrame(combinedfeatures).to_csv(csv_file_path1, index=False)
                        pd.DataFrame(combinedfeatures).to_csv(csv_file_path, index=False)
        # Average features at each state
        print("Averaging features...")
        meanfeatures = np.empty((6), dtype=object)
        meanfeatures[:] = None

        for root, dirs, files in os.walk(dir):
            for name in files:
                if name.endswith("kHz_AF.csv"):
                    data = np.array(pd.read_csv(os.path.join(root, name)))
                    data_mean = np.mean(data, axis=0)  # Corrected to mean over the correct axis
                    freq_index = frequencies.index(name[:3])
                    if meanfeatures[freq_index] is None:
                        meanfeatures[freq_index] = np.array([data_mean])
                    else:
                        meanfeatures[freq_index] = np.concatenate((meanfeatures[freq_index], np.array([data_mean])),
                                                                  axis=0)

        for freq in frequencies:  # For each feature
            freq_index = frequencies.index(freq)
            if meanfeatures[freq_index] is not None:
                csv_file_path = os.path.join(dir, freq + "kHz_MF.csv")  # MF = Mean Features
                pd.DataFrame(meanfeatures[freq_index]).to_csv(csv_file_path, index=False)  # Read mean features from files


def correlateSPFeatures(rootdir):
    # Correlate extracted features
    # also not using this
    frequencies = ["050", "100", "125", "150", "200", "250"]
    samples = ["PZT-CSV-L1-03", "PZT-CSV-L1-04", "PZT-CSV-L1-05", "PZT-CSV-L1-09", "PZT-CSV-L1-23"]
    print("Combining Features:...")
    for sample in samples:
        dir = rootdir + "\\" + sample
        for root, dirs, files in os.walk(dir):  #For each folder location (state)
            all_feat = np.empty((6, 5), dtype=object)
            flag = False
            for name in files:              #For each file
                for freq in frequencies:    #For each frequency
                    #Read and add to correct position in all_feat array
                    if freq in name and name.endswith('_FT.csv'):
                        flag = True
                        data = np.array(pd.read_csv(os.path.join(root, name)))
                        if 'FFT' in name:
                            all_feat[frequencies.index(freq)][1] = data
                        elif 'HLB' in name:
                            all_feat[frequencies.index(freq)][2] = data
                        elif 'EMD' in name:
                            all_feat[frequencies.index(freq)][3] = data
                        elif 'SFT' in name:
                            all_feat[frequencies.index(freq)][4] = data
                        else: #Time domain
                            all_feat[frequencies.index(freq)][0] = data

            if flag:    #If at least one file was the correct type
                for freq in ["050", "100", "125", "150", "200", "250"]:
                    combinedfeatures = np.concatenate([all_feat[frequencies.index(freq), i] for i in range(all_feat[0].shape[0])], axis=0)
                    root2 = root
                    root_new = root2.replace("PZT", "PZT-ONLY-FEATURES")
                    if not os.path.exists(root_new):
                        os.makedirs(root_new)
                    csv_file_path1 = os.path.join(root_new, freq +"kHz_AF.csv") #AF = ALL FEATURES
                    csv_file_path = os.path.join(root, freq + "kHz_AF.csv")
                    pd.DataFrame(combinedfeatures).to_csv(csv_file_path1, index=False)
                    pd.DataFrame(combinedfeatures).to_csv(csv_file_path, index=False)

        #Average features at each state
        print("Averaging features...")
        meanfeatures = np.empty((6), dtype=object)
        for root, dirs, files in os.walk(dir):
            for name in files:
                if name.endswith("kHz_AF.csv"):
                    data = np.array(pd.read_csv(os.path.join(root, name)))
                    data = np.mean(data, axis=1)
                    if str(type(meanfeatures[frequencies.index(name[:3])])) == "<class 'NoneType'>":
                        meanfeatures[frequencies.index(name[:3])] = np.array([data])
                    else:
                        meanfeatures[frequencies.index(name[:3])] = np.concatenate((meanfeatures[frequencies.index(name[:3])], np.array([data])))

        #Correlate features
        alldelete = np.empty((6), dtype=object)
        for freq in frequencies:    #For each feature
            csv_file_path = os.path.join(dir, freq + "kHz_MF.csv") # MF = Mean Features
            pd.DataFrame(meanfeatures[frequencies.index(freq)]).to_csv(csv_file_path, index=False)  #Read mean features from file

            #Use feature_correlation to return the correlation matrix, reduced matrix and features deleted
            correlation_matrix, features, to_delete = extract_features.feature_correlation(meanfeatures[frequencies.index(freq)])
            #Save all to files
            alldelete[frequencies.index(freq)] = to_delete
            csv_file_path = os.path.join(dir, freq + "kHz_CF.csv") # CM = Correlated feature matrix
            pd.DataFrame(correlation_matrix).to_csv(csv_file_path, index=False)
            csv_file_path = os.path.join(dir, freq + "kHz_RF.csv") # RF = Remainder features
            pd.DataFrame(features).to_csv(csv_file_path, index=False)
        # Save lists of deleted features for inspection
        csv_file_path = os.path.join(dir, "DF.csv") # DF = Deleted features
        pd.DataFrame(alldelete).to_csv(csv_file_path, index=False)


''' PCA '''
def savePCA(dir): #Calculates and saves 1 principle component PCA
    """
        PCA

        Args:
            dir (str): The directory path containing .csv files to process.
        Returns:
            None. Saves .csv
        Example:
            savePCA('/path/to/directory')
        Notes:
            This function processes each campaign directory separately and saves the
            files in a new directory PZT-ONLY-FEATURES for different SPs
    """

    pcs_upto = 10

    output = np.zeros((6, pcs_upto, 5, 30))
    for pc in range(1, pcs_upto+1):
        tempout = PCA.doPCA_multiple_Campaigns(dir, pc)
        for freq in range(6):
            output[freq][pc-1] = tempout[freq]
            #print(tempout)
            #print(output[freq])
    labels = []
    for pc in range(1, pcs_upto+1):
        if pc == 1:
            add = "st"
        elif pc == 2:
            add = "nd"
        elif pc == 3:
            add = "rd"
        else:
            add = "th"
        labels.append(str(pc) + add + " PC")
    save_evaluation(np.array(output), "PCA", dir, labels)

    """
    for k,folder in enumerate(folders):
        for i,freq in enumerate(["050", "100", "125", "150", "200", "250"]):
            root_new = folder.replace("PZT", "PZT-ONLY-FEATURES")
            if not os.path.exists(root_new):
                os.makedirs(root_new)
            #csv_file_path = os.path.join(root_new + f" Test Specimen{folder}, Frequency{freq}kHz_PCA_HI.csv")
            csv_file_path = os.path.join(root_new + f"{freq}kHz_PCA_HI.csv")
            pd.DataFrame(output[k][i]).to_csv(csv_file_path, index=False)
            plt.plot(output[k][i])
            plt.xlabel('Index')
            plt.ylabel('PCA Value')
            plt.title('PCA Values from CSV Files')
    """

    #save_evaluation(switch_dimensions(output),"PCA", dir, ["kHz-PCA"])
    #switched_output = switch_dimensions(output) #features_preprocessed

"""
def switch_dimensions(output):
    ""
        Args:
            output (arr): PCA output of all campaigns
        Returns:
            return_type: arr
        Example:
            switched_output = switch_dimensions(output)
        Notes:
            This function transposes the pca output for each campaign
    ""
    # Get the dimensions of the original output list
    num_folders = len(output)
    num_freqs = len(output[0])

    # Create a new list to store the switched dimensions
    switched_output = [[[[] for _ in range(num_folders)]] for _ in range(num_freqs)]

    # Switch the dimensions
    for folder_index in range(num_folders):
        for freq_index in range(num_freqs):
            switched_output[freq_index][0][folder_index] = output[folder_index][freq_index][-30:]

    return switched_output
"""

def save_evaluation(features, label, dir, files_used=[""]):  #Features is 6x freq, features, then HIs along the states within each
    """
        Args:
            features (arr): feature extraction output of all campaigns
            label (str) : name of the new folder
            dir (str) : directory of the original folder
            files_used (str)
        Returns:
            return_type: arr
        Example:
            switched_output = switch_dimensions(output)
        Notes:
            This function saves the pca output for each campaign
    """
    frequencies = ["050", "100", "125", "150", "200", "250"]
    # Initiliase arrays for feature extraction results, for fitness and the three criteria respectively
    criteria = np.empty((4, 6, len(features[0])))
    # Iterate through each frequency and calculate features
    featuresonly = files_used[0] == ""

    for freq in range(6):
        print("Saving: " + frequencies[freq] + "kHz")
        # print(components)
        for feat in range(len(features[0])):
            if feat % 50 == 0 and feat != 0:
                print(feat)
            # print(features[freq][feat])
            features[freq][feat] = np.array(features[freq][feat])
            ftn, mo, tr, pr, error = fitness(features[freq][feat])
            criteria[0][freq][feat] = float(ftn)
            criteria[1][freq][feat] = float(mo)
            criteria[2][freq][feat] = float(pr)
            criteria[3][freq][feat] = float(tr)
            #Save graphs
            Graphs.HI_graph(features[freq][feat], dir=dir, name=f"{label}-{frequencies[freq]}-{feat}")
        if featuresonly:
            files_used = np.array([str(i) for i in range(len(features[0]))])
        Graphs.criteria_chart(files_used, criteria[1][freq], criteria[2][freq], criteria[3][freq], dir=dir, name=f"{label}-{frequencies[freq]}")
    #Bar charts against frequency
    #for feat in range(len(features[0])):
    #    Graphs.criteria_chart(frequencies, criteria[1][:, feat], criteria[2][:, feat], criteria[3][:, feat], dir=dir, name=label + "-" + str(feat))

    if featuresonly:
        avs = np.empty((4, 2), dtype=object)
        for crit in range(4):
            avs[crit, 0] = np.expand_dims(np.mean(criteria[crit], axis= 0),axis=0)[0]
            avs[crit, 1] = np.std(criteria[crit], axis = 0)
        Graphs.criteria_chart(files_used, avs[1][0], avs[2][0], avs[3][0], dir=dir, name=f"{label}-Av")
        av_arr = np.vstack((avs[0, 0], avs[0, 1]))
        pd.DataFrame(av_arr).to_csv(os.path.join(dir, label + " Fit AF.csv"), index=False)

    # Save all to files
    pd.DataFrame(criteria[0]).to_csv(os.path.join(dir, label + " Fit.csv"), index=False)    #Feature against frequency
    pd.DataFrame(criteria[1]).to_csv(os.path.join(dir, label + " Mon.csv"), index=False)
    pd.DataFrame(criteria[2]).to_csv(os.path.join(dir, label + " Tre.csv"), index=False)
    pd.DataFrame(criteria[3]).to_csv(os.path.join(dir, label + " Pro.csv"), index=False)

def evaluate(dir):
    #Apply prognostic criteria to PCA and extracted features
    frequencies = ["050", "100", "125", "150", "200", "250"]
    features = np.empty((6, 139), dtype=object)  #6 frequencies, 139 features, 5 samples with a list of values at each location

    # Read all features to 'features', and all PCA to 'components' arrays
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith("MF.csv"): #MF
                data = np.array(pd.read_csv(os.path.join(root, name))).transpose()
                freq = frequencies.index(name[:3])
                for feat in range(139):
                    if str(type(features[freq][feat])) == "<class 'NoneType'>":
                        features[freq][feat] = np.array([scale_exact(data[feat])])
                    else:
                        features[freq][feat] = np.vstack([features[freq][feat], scale_exact(data[feat])])

    return features


# def giveTime():
#     time = []
#     for i in range(2000):
#         time.append(i*(5e-7))
#     return time

def saveDeepSAD(dir):
    frequencies = ["050", "100", "125", "150", "200", "250"]
    filename_HLB = "HLB_FT_Reduced"    #No need for .csv
    filename_FFT = "FFT_FT_Reduced"

    HIs_HLB = np.empty((6), dtype=object)
    HIs_FFT = np.empty((6), dtype=object)

    for freq in range(len(frequencies)):
        print(f"Processing frequency: {frequencies[freq]} kHz for HLB")
        HIs_HLB[freq] = DeepSAD_train_run(dir, frequencies[freq], filename_HLB)

        print(f"Processing frequency: {frequencies[freq]} kHz for FFT")
        HIs_FFT[freq] = DeepSAD_train_run(dir, frequencies[freq], filename_FFT)

    save_evaluation(HIs_HLB, "DeepSAD_HLB", dir, filename_HLB)
    save_evaluation(HIs_FFT, "DeepSAD_FFT", dir, filename_FFT)


def main_menu():
    print("Welcome! Please choose a signal processing method: ")
    print("1. FFT")
    print("2. EMD")
    print("3. Hilbert")
    print("4. STFT")
    print("5. All of the above")
    print("6. Extract all features (Requires 5)")
    print("7. AverageFeatures (Requires 6)")
    print("8. Apply PCA to all (Requires 7)")
    print("9. Evaluate feature HIs (Requires 8)")
    print("10. Execute DeepSAD (Requires 7)")
    print("11. Export FFT and Hilbert only (on a separate directory)")
    print("12. Extract reduced Features for FFT & Hilbert")
    print("0. Exit")

# Prompt the user to input the folder path
extract = bool(int(input("Extract Matlab (0=No, 1=Yes): ")))
if extract:
    folder_path = input("Enter the folder path of the Matlab files: ")
    Data_Preprocess.matToCsv(folder_path)
    print("Done")
    csv_dir = input("Enter the folder path of the CSV files: ")
    #quit()
    #csv_dir = folder_path.replace('PZT','PZT-CSV')
else:
    csv_dir = input("Enter the folder path of the CSV files: ")

# Main program loop
while True:
    main_menu()
    choice = input("Enter your choice: ")

    if choice == '1':
        SP.saveFFT(csv_dir)
    elif choice == '2':
        SP.saveEMD(csv_dir)
    elif choice == '3':
        SP.saveHilbert(csv_dir)
    elif choice == '4':
        SP.saveSTFT(csv_dir)
    elif choice == '5':
        SP.saveFFT(csv_dir)
        SP.saveEMD(csv_dir)
        SP.saveHilbert(csv_dir)
        SP.saveSTFT(csv_dir)
    elif choice == '6':
        saveFeatures(csv_dir)
    elif choice == '8':
        savePCA(csv_dir)
    elif choice == '7':
        AverageFeatures(csv_dir)
    elif choice == '9':
        features = evaluate(csv_dir)
        save_evaluation(features, "Features", csv_dir)
    elif choice == '10':
        saveDeepSAD(csv_dir)
    elif choice == '11':
        SP.saveFFTHLB(csv_dir)
    elif choice == '12':
        FFT_HLB_Reduced_Feat(csv_dir)
    elif choice == '0':
        print("Exiting...")
        quit()
    else:
        print("Invalid choice. Please enter a number between 0 and 10.")

