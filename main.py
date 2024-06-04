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
        dir = rootdir + "\\" + sample
        for root, dirs, files in os.walk(dir):  # For each folder location (state)
            all_feat = np.empty((6, 5), dtype=object)
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
                for freq in ["050", "100", "125", "150", "200", "250"]:
                    combinedfeatures = np.concatenate(
                        [all_feat[frequencies.index(freq), i] for i in range(all_feat[0].shape[0])], axis=0)
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
        for root, dirs, files in os.walk(dir):
            for name in files:
                if name.endswith("kHz_AF.csv"):
                    data = np.array(pd.read_csv(os.path.join(root, name)))
                    data = np.mean(data, axis=1)
                    if str(type(meanfeatures[frequencies.index(name[:3])])) == "<class 'NoneType'>":
                        meanfeatures[frequencies.index(name[:3])] = np.array([data])
                    else:
                        print(name[:3])
                        print(np.array([data]))
                        meanfeatures[frequencies.index(name[:3])] = np.concatenate((meanfeatures[frequencies.index(name[:3])], np.array([data])))
        for freq in frequencies:    #For each feature
            csv_file_path = os.path.join(dir, freq + "kHz_MF.csv") # MF = Mean Features
            pd.DataFrame(meanfeatures[frequencies.index(freq)]).to_csv(csv_file_path, index=False)  #Read mean features from file


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
            return_type: .csv
        Example:
            savePCA('/path/to/directory')
        Notes:
            This function processes each campaign directory separately and saves the
            files in a new directory PZT-ONLY-FEATURES for different SPs
    """
    output = []
    folders = []

    # Get the locations of train folders
    for i in range(5):
        folder_location = input(f"Enter the location of your 5 folders {i}: ")
        folders.append(folder_location)
    for i in range(5):
        output.append(PCA.doPCA_multiple_Campaigns(folders[i%5],folders[(i+1)%5],folders[(i+2)%5],folders[(i+3)%5],folders[(i+4)%5]))

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

    #save_evaluation(switch_dimensions(output),"PCA", dir, ["kHz-PCA"])
    switched_output = switch_dimensions(output) #features_preprocessed

    save_evaluation(switched_output, "PCA", dir, ["kHz_PCA"])

def switch_dimensions(output):
    """
        Args:
            output (arr): PCA output of all campaigns
        Returns:
            return_type: arr
        Example:
            switched_output = switch_dimensions(output)
        Notes:
            This function transposes the pca output for each campaign
    """
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

    for freq in range(6):
        print("Saving: " + frequencies[freq] + "kHz")
        # print(components)
        for feat in range(139):
            if feat % 50 == 0:
                print(feat)
            # print(features[freq][feat])
            features[freq][feat] = np.array(features[freq][feat])
            ftn, mo, tr, pr, error = fitness(features[freq][feat])
            criteria[0][freq][feat] = float(ftn)
            criteria[1][freq][feat] = float(mo)
            criteria[2][freq][feat] = float(tr)
            criteria[3][freq][feat] = float(pr)
            #Save graphs
            Graphs.HI_graph(features[freq][feat], dir=dir, name=label + "-" + frequencies[freq] + "-" + str(feat))
        if files_used[0] == "":     #Using features as HIs
            files_used = np.array([str(i) for i in range(len(features[0]))])
        Graphs.criteria_chart(files_used, criteria[1][freq], criteria[2][freq], criteria[3][freq], dir=dir, name=label + "-" + frequencies[freq])
    #Bar charts against frequency
    #for feat in range(len(features[0])):
    #    Graphs.criteria_chart(frequencies, criteria[1][:, feat], criteria[2][:, feat], criteria[3][:, feat], dir=dir, name=label + "-" + str(feat))

    if files_used[0] == "":
        avs = np.empty((4, 2), dtype=object)
        for crit in range(4):
            avs[crit, 0] = np.expand_dims(np.mean(criteria[crit], axis= 0),axis=0)[0]
            avs[crit, 1] = np.std(criteria[crit], axis = 0)
        Graphs.criteria_chart(files_used, avs[1][0], avs[2][0], avs[3][0], dir=dir, name=label + "- Av")
        av_arr = np.vstack((avs[0, 0], avs[0, 1]))
        pd.DataFrame(av_arr).to_csv(dir + "\\" + label + " Fit Av.csv", index=False)

    # Save all to files
    pd.DataFrame(criteria[0]).to_csv(dir + "\\" + label + " Fit.csv", index=False)    #Feature against frequency
    pd.DataFrame(criteria[1]).to_csv(dir + "\\" + label + " Mon.csv", index=False)
    pd.DataFrame(criteria[2]).to_csv(dir + "\\" + label + " Tre.csv", index=False)
    pd.DataFrame(criteria[3]).to_csv(dir + "\\" + label + " Pro.csv", index=False)

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
    filenames = ["kHz_AF"]    #No need for .csv
    HIs = np.empty((6, len(filenames)), dtype=object)
    for freq in range(len(frequencies)):
        for name in range(len(filenames)):
            HIs[freq][name] = DeepSAD_train_run(dir, frequencies[freq], filenames[name])
    save_evaluation(HIs, "DeepSAD", dir, filenames)


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
    elif choice == '0':
        print("Exiting...")
        quit()
    else:
        print("Invalid choice. Please enter a number between 0 and 9.")

