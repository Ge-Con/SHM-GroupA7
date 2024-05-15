import pandas as pd
import numpy as np
import os

import extract_features
import PCA
from Signal_Processing import fft, emdfinal, stft, hilbert, Data_Preprocess
from prognosticcriteria import fitness, Mo, Tr, Pr
from DeepSAD import DeepSAD_train_run
import Graphs
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 400)
np.set_printoptions(linewidth=400)

np.set_printoptions(precision=4, suppress=True)


def saveFFT(dir):
    print("Executing FFT on data:...")
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith('kHz.csv'):
                # print("good")
                data = pd.read_csv(os.path.join(root, name))
                #print(root, name)
                arrayfile1, arrayfile2 = fft.fast_fourier(data)
                csv_file_path1 = os.path.join(root, f"{name.replace('kHz.csv', '')}_{'kHz_FFT_Freq.csv'}")
                csv_file_path2 = os.path.join(root, f"{name.replace('kHz.csv', '')}_{'kHz_FFT_Amp.csv'}")
                # print(arrayfile)
                #arrayfile1.to_csv(csv_file_path1, index=False)
                arrayfile2.to_csv(csv_file_path2, index=False)

def saveSTFT(dir):
    print("Executing STFT on data:...")
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith('kHz.csv'):
                # print("good")
                data = pd.read_csv(os.path.join(root, name))
                arrayfile1 = stft.Short_Fourier(data)
                csv_file_path1 = os.path.join(root, f"{name.replace('kHz.csv', '')}_{'kHz_STFT_Amp.csv'}")

                arrayfile1.to_csv(csv_file_path1, index=False)

def saveEMD(dir):
    print("Executing EMD on data:...")
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith('kHz.csv'):
                data = pd.read_csv(os.path.join(root, name))
                arrayfile1 = emdfinal.runEMD(data, giveTime())
                csv_file_path1 = os.path.join(root, f"{name.replace('kHz.csv', '')}_{'kHz_EMD.csv'}")
                arrayfile1.to_csv(csv_file_path1, index=False)


def saveHilbert(dir):
    print("Executing Hilbert on data:...")
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith('kHz.csv'):
                # print("good")
                data = pd.read_csv(os.path.join(root, name))
                arrayfile1 = hilbert.Hilbert(data, giveTime())
                csv_file_path1 = os.path.join(root, f"{name.replace('kHz.csv', '')}_{'kHz_Hilbert.csv'}")
                # csv_file_path2 = os.path.join(root, f"{name.replace('kHz.csv', '')}_{'kHz_Hilb_Amp.csv'}")
                arrayfile1.to_csv(csv_file_path1, index=False)
                # arrayfile2.to_csv(csv_file_path2, index=False)

# def fixname(name):
#     if "50" in name and "150" not in name and "250" not in name:
#         name = "0" + name
#     return name

def saveFeatures(dir):
    #Extract and save features for each frequency
    print("Extracting Features:...")
    for root, dirs, files in os.walk(dir):
        for name in files:

            if name.endswith('kHz.csv'):
                data = pd.read_csv(os.path.join(root, name))    #Don't put this line outside if statement
                features = extract_features.time_to_feature(data)
                new_filename = name.replace('kHz.csv', 'kHz-Features.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            elif name.endswith('FFT_Amp.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = extract_features.freq_to_feature(data)
                new_filename = name.replace('FFT_Amp.csv', 'FFT_Amp-Features.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            elif name.endswith('Hilbert.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = extract_features.time_to_feature(data)
                new_filename = name.replace('Hilbert.csv', 'Hilbert-Features.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            elif name.endswith('EMD.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = extract_features.time_to_feature(data)
                new_filename = name.replace('EMD.csv', 'EMD-Features.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            elif name.endswith('STFT_Amp.csv'):
                data = pd.read_csv(os.path.join(root, name))
                data3d = [[[0 for _ in range(17)] for _ in range(126)] for _ in range(56)]

                for k in range(56):
                    for i in range(126):
                        for j in range(17):
                            data3d[k][i][j] = data.iloc[126*k + i, j]
                #
                # print(len(data3d))
                # print(len(data3d[0]))
                # print(len(data3d[0][0]))

                features = extract_features.STFT_to_feature(data3d)
                new_filename = name.replace('STFT_Amp.csv', 'STFT_Amp-Features.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)


def correlateFeatures(rootdir):
    #Correlate extracted features
    frequencies = ["050", "100", "125", "150", "200", "250"]
    samples = ["PZT-CSV-L1-03", "PZT-CSV-L1-04", "PZT-CSV-L1-05", "PZT-CSV-L1-09", "PZT-CSV-L1-23"]
    print("Combining Features:...")
    for sample in samples:
        dir = rootdir + "\\" + sample
        for root, dirs, files in os.walk(dir):  #For each folder location (state)
            allfeatures = np.empty((6, 5), dtype=object)
            flag = False
            for name in files:              #For each file
                for freq in frequencies:    #For each frequency
                    #Read and add to correct position in allfeatures array
                    if freq in name and name.endswith('-Features.csv'):
                        flag = True
                        data = np.array(pd.read_csv(os.path.join(root, name)))
                        if 'FFT_Amp' in name:
                            allfeatures[frequencies.index(freq)][1] = data
                        elif 'Hilbert' in name:
                            allfeatures[frequencies.index(freq)][2] = data
                        elif 'EMD' in name:
                            allfeatures[frequencies.index(freq)][3] = data
                        elif 'STFT_Amp' in name:
                            allfeatures[frequencies.index(freq)][4] = data
                        else: #Time domain
                            allfeatures[frequencies.index(freq)][0] = data

            if flag:    #If at least one file was the correct type
                for freq in ["050", "100", "125", "150", "200", "250"]:
                    combinedfeatures = np.concatenate([allfeatures[frequencies.index(freq), i] for i in range(allfeatures[0].shape[0])], axis=0)
                    root2 = root
                    root_new = root2.replace("PZT", "PZT-ONLY-FEATURES")
                    if not os.path.exists(root_new):
                        os.makedirs(root_new)
                    csv_file_path1 = os.path.join(root_new, freq +"_kHz-allfeatures.csv")
                    csv_file_path = os.path.join(root, freq + "_kHz-allfeatures.csv")
                    pd.DataFrame(combinedfeatures).to_csv(csv_file_path1, index=False)
                    pd.DataFrame(combinedfeatures).to_csv(csv_file_path, index=False)

        #Average features at each state
        print("Averaging features...")
        meanfeatures = np.empty((6), dtype=object)
        for root, dirs, files in os.walk(dir):
            for name in files:
                if name.endswith("kHz-allfeatures.csv"):
                    data = np.array(pd.read_csv(os.path.join(root, name)))
                    data = np.mean(data, axis=1)
                    if str(type(meanfeatures[frequencies.index(name[:3])])) == "<class 'NoneType'>":
                        meanfeatures[frequencies.index(name[:3])] = np.array([data])
                    else:
                        meanfeatures[frequencies.index(name[:3])] = np.concatenate((meanfeatures[frequencies.index(name[:3])], np.array([data])))

        #Correlate features
        alldelete = np.empty((6), dtype=object)
        for freq in frequencies:    #For each feature
            csv_file_path = os.path.join(dir, freq + "_kHz-meanfeatures.csv")
            pd.DataFrame(meanfeatures[frequencies.index(freq)]).to_csv(csv_file_path, index=False)  #Read mean features from file

            #Use feature_correlation to return the correlation matrix, reduced matrix and features deleted
            correlation_matrix, features, to_delete = extract_features.feature_correlation(meanfeatures[frequencies.index(freq)])
            #Save all to files
            alldelete[frequencies.index(freq)] = to_delete
            csv_file_path = os.path.join(dir, freq + "_kHz-cmatrix.csv")
            pd.DataFrame(correlation_matrix).to_csv(csv_file_path, index=False)
            csv_file_path = os.path.join(dir, freq + "_kHz-rfeatures.csv")
            pd.DataFrame(features).to_csv(csv_file_path, index=False)
        # Save lists of deleted features for inspection
        csv_file_path = os.path.join(dir, "deleted_features.csv")
        pd.DataFrame(alldelete).to_csv(csv_file_path, index=False)



def preprocess_data(X):
    # Step 1: Check for NaN values
    has_nan = np.isnan(X).any()

    if has_nan:
        # Step 2: Handle NaN values
        # For example, impute NaN values with the mean of each column
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        return X_imputed
    else:
        return X


def savePCA(dir): #Calculates and saves 1 principle component PCA
    # frequencies = ["050", "100", "125", "150", "200", "250"]
    # components = np.empty((6), dtype=object)
    # data = np.empty((6), dtype=object)
    output = []

    folders = []

    # Get the locations of train folders
    for i in range(5):
        folder_location = input(f"Enter the location of your 5 folders {i}: ")
        folders.append(folder_location)
    for i in range(5):
        output.append(PCA.doPCA_multiple_Campaigns(folders[i%5],folders[(i+1)%5],folders[(i+2)%5],folders[(i+3)%5],folders[(i+4)%5]))

    # print(len(output))
    # print(len(output[0]))
    # print(len(output[0][0]))
    for k in range(5):
        for i in range(6):
            root_new = folder_location.replace("PZT", "PZT-ONLY-FEATURES")
            if not os.path.exists(root_new):
                os.makedirs(root_new)
            csv_file_path = os.path.join(root_new + f" Test Specimen{k}, Frequency{i}-PCA-HI.csv")
            pd.DataFrame(output[k][i]).to_csv(csv_file_path, index=False)
            plt.plot(output[k][i])
            plt.xlabel('Index')
            plt.ylabel('PCA Value')
            plt.title('PCA Values from CSV Files')
            plt.show()
    # Example usage:
    #save_evaluation(switch_dimensions(output),"PCA", dir, ["_kHz-PCA"])
    switched_output = switch_dimensions(output)
    features = np.empty((6, 139, 1, 30), dtype=float)  # 6 frequencies, 139 features, 1 value, 30 time steps

    # Assign switched_output to features
    for freq_index, pca_list in enumerate(switched_output):
        for feature_index, pca_values in enumerate(pca_list):
            last_30_values = pca_values[-30:]
            features[freq_index][feature_index][0] = last_30_values

    # Preprocess the features to handle NaN values
    features_preprocessed = preprocess_data(features)

    save_evaluation(features, "PCA", dir, ["_kHz-PCA"])
    # return output

def switch_dimensions(output):
    # Get the dimensions of the original output list
    num_folders = len(output)
    num_freqs = len(output[0])

    # Create a new list to store the switched dimensions
    switched_output = [[[] for _ in range(num_folders)] for _ in range(num_freqs)]

    # Switch the dimensions
    for folder_index in range(num_folders):
        for freq_index in range(num_freqs):
            switched_output[freq_index][folder_index] = output[folder_index][freq_index]

    return switched_output




def save_evaluation(features, label, dir, files_used=[""]):  #Features is 6x freq, features, then HIs along the states within each
    frequencies = ["050", "100", "125", "150", "200", "250"]
    # Initiliase arrays for feature extraction results, for fitness and the three criteria respectively
    criteria = np.empty((4, 6, len(features[0])))
    # Iterate through each frequency and calculate features
    #print(features.shape)
    print(features)
    print("First Dimension: ", len(features))
    print("Second Dimension: ", len(features[0]))
    print("Third dimension ", len(features[0][0]))
    print("fourth dimension ", len(features[0][0][0]))

    for freq in range(6):
        # print(components)
        for feat in range(len(features[0])):
            criteria[0][freq][feat] = float(fitness(features[freq][feat])[0])
            criteria[1][freq][feat] = float(Mo(features[freq][feat]))
            criteria[2][freq][feat] = float(Tr(features[freq][feat]))
            criteria[3][freq][feat] = float(Pr(features[freq][feat]))
            #Save graphs
            Graphs.HI_graph(features[freq][feat], dir=dir, name=label + "-" + frequencies[freq] + "-" + str(feat))
        if files_used[0] == "":
            files_used = np.array([str(i) for i in range(len(criteria[1][freq]))])
        Graphs.criteria_chart(files_used, criteria[1][freq], criteria[2][freq], criteria[3][freq], dir=dir, name=label + "-" + frequencies[freq])
    for feat in range(len(features[0])):
        Graphs.criteria_chart(frequencies, criteria[1][:, feat], criteria[2][:, feat], criteria[3][:, feat], dir=dir, name=label + "-" + str(feat))

    # Save all to files
    pd.DataFrame(criteria[0]).to_csv(dir + "\\" + label + " Fit.csv", index=False)    #Feature against frequency
    pd.DataFrame(criteria[1]).to_csv(dir + "\\" + label + " Mo.csv", index=False)
    pd.DataFrame(criteria[2]).to_csv(dir + "\\" + label + " Tr.csv", index=False)
    pd.DataFrame(criteria[3]).to_csv(dir + " \\" + label + " Pr.csv", index=False)

def evaluate(dir):
    #Apply prognostic criteria to PCA and extracted features
    frequencies = ["050", "100", "125", "150", "200", "250"]
    features = np.empty((6, 139), dtype=object)  #6 frequencies, 71 features and a list of values at each location

    # Read all features to 'features', and all PCA to 'components' arrays
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith("meanfeatures.csv"):
                data = np.array(pd.read_csv(os.path.join(root, name))).transpose()
                freq = frequencies.index(name[:3])
                for feat in range(139):
                    if str(type(features[freq][feat])) == "<class 'NoneType'>":
                        features[freq][feat] = np.array([data[feat][-30::]])
                    else:
                        features[freq][feat] = np.vstack([features[freq][feat], data[feat][-30::]])

    save_evaluation(features, "Features", dir)




def giveTime():
    time = []
    for i in range(2000):
        time.append(i*(5e-7))
    return time

def saveDeepSAD(dir):
    frequencies = ["050", "100", "125", "150", "200", "250"]
    filenames = ["_kHz-allfeatures"]    #No need for .csv
    HIs = np.empty((6, len(filenames)), dtype=object)
    for freq in range(len(frequencies)):
        for name in range(len(filenames)):
            HIs[freq][name] = DeepSAD_train_run(dir, frequencies[freq], filenames[name])
    save_evaluation(HIs, "DeepSAD", dir, filenames)

# Data_Preprocess.matToCsv(r"C:\Users\geort\Desktop\Universty\PZT-L1-03")
#print("ok")
#saveFFT(r"C:\Users\geort\Desktop\Universty\PZT-CSV-L1-03")
#Data_Preprocess.matToCsv(r"C:\Users\geort\Desktop\Universty\PZT-L1-03")

#saveSTFT(r"C:\Users\geort\Desktop\Universty\PZT-CSV-L1-03")
def main_menu():
    print("Welcome! Please choose a signal processing method: ")
    print("1. FFT")
    print("2. EMD")
    print("3. Hilbert")
    print("4. STFT")
    print("5. All of the above")
    print("6. Extract all features (Requires 5)")
    print("7. Correlate features (Requires 6)")
    print("8. Apply PCA to all (Requires 7)")
    print("9. Evaluate all HIs (Requires 8)")
    print("10. DeepSAD")
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
        saveFFT(csv_dir)
    elif choice == '2':
        saveEMD(csv_dir)
    elif choice == '3':
        saveHilbert(csv_dir)
    elif choice == '4':
        saveSTFT(csv_dir)
    elif choice == '5':
        saveFFT(csv_dir)
        saveEMD(csv_dir)
        saveHilbert(csv_dir)
        saveSTFT(csv_dir)
    elif choice == '6':
        saveFeatures(csv_dir)
    elif choice == '8':
        savePCA(csv_dir)
    elif choice == '7':
        correlateFeatures(csv_dir)
    elif choice == '9':
        evaluate(csv_dir)
    elif choice == '10':
        saveDeepSAD(csv_dir)
    elif choice == '0':
        print("Exiting...")
        quit()
    else:
        print("Invalid choice. Please enter a number between 0 and 9.")

#C:\Users\geort\Desktop\Universty\PZT-L1-03
#C:\Users\geort\Desktop\Universty\PZT-CSV-L1-03

#C:\Users\Jamie\Documents\Uni\Year 2\Q3+4\Project\Files

#C:\Users\Martin\Downloads\PZT-CSV\PZT-CSV-L01-5
