import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA

import extract_features
from Signal_Processing import fft, emdfinal, stft, hilbert, Data_Preprocess
from prognosticcriteria import fitness, Mo, Tr, Pr

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 400)
np.set_printoptions(linewidth=400)


def saveFFT(dir):
    print("Executing FFT on data:...")
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith('kHz.csv'):
                # print("good")
                data = pd.read_csv(os.path.join(root, name))
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

def fixname(name):
    if "50" in name and "150" not in name and "250" not in name:
        name = "0" + name
    return name

def saveFeatures(dir):
    print("Extracting Features:...")
    for root, dirs, files in os.walk(dir):
        for name in files:

            if name.endswith('kHz.csv'):
                data = pd.read_csv(os.path.join(root, name))    #Don't put this line outside if statement
                features = extract_features.time_to_feature(data)
                new_filename = fixname(name).replace('kHz.csv', 'kHz-Features.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            elif name.endswith('FFT_Amp.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = extract_features.freq_to_feature(data)
                new_filename = fixname(name).replace('FFT_Amp.csv', 'FFT_Amp-Features.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            elif name.endswith('Hilbert.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = extract_features.time_to_feature(data)
                new_filename = fixname(name).replace('Hilbert.csv', 'Hilbert-Features.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            elif name.endswith('EMD.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = extract_features.time_to_feature(data)
                new_filename = fixname(name).replace('EMD.csv', 'EMD-Features.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            elif name.endswith('STFT_Amp.csv'):
                data = pd.read_csv(os.path.join(root, name))
                #Unflattening happens here
                # Populate the unflattened_list with values from flat
                # index = 0
                #data3d = [[[0 for _ in range(17)] for _ in range(126)] for _ in range(56)]
                #for i in range(56):
                    #for j in range(17):
                            #data3d[i][j] = data[index]
                            #index += 1
                features = extract_features.STFT_to_feature(data)
                new_filename = fixname(name).replace('STFT_Amp.csv', 'STFT_Amp-Features.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

def correlateFeatures(dir):
    frequencies = ["050", "100", "125", "150", "200", "250"]
    print("Combining Features:...")
    for root, dirs, files in os.walk(dir):
        allfeatures = np.empty((6, 5), dtype=object)
        flag = False
        for name in files:
            for freq in frequencies:
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

        if flag:
            for freq in ["050", "100", "125", "150", "200", "250"]:
                combinedfeatures = np.concatenate([allfeatures[frequencies.index(freq), i] for i in range(allfeatures[0].shape[0])], axis=0)
                csv_file_path = os.path.join(root, freq +"_kHz-allfeatures.csv")
                pd.DataFrame(combinedfeatures).to_csv(csv_file_path, index=False)

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

    alldelete = np.empty((6), dtype=object)
    for freq in frequencies:
        csv_file_path = os.path.join(dir, freq + "_kHz-meanfeatures.csv")
        pd.DataFrame(meanfeatures[frequencies.index(freq)]).to_csv(csv_file_path, index=False)

        correlation_matrix, features, to_delete = extract_features.feature_correlation(meanfeatures[frequencies.index(freq)])
        alldelete[frequencies.index(freq)] = to_delete
        csv_file_path = os.path.join(dir, freq + "_kHz-cmatrix.csv")
        pd.DataFrame(correlation_matrix).to_csv(csv_file_path, index=False)
        csv_file_path = os.path.join(dir, freq + "_kHz-rfeatures.csv")
        pd.DataFrame(features).to_csv(csv_file_path, index=False)
    csv_file_path = os.path.join(dir, "deleted_features.csv")
    pd.DataFrame(alldelete).to_csv(csv_file_path, index=False)

def savePCA(dir):
    frequencies = ["050", "100", "125", "150", "200", "250"]
    components = np.empty((6), dtype=object)
    print("VAF:")
    for freq in frequencies:
        data = pd.read_csv(os.path.join(dir, freq + "_kHz-meanfeatures.csv"))
        pca = PCA(n_components=1)
        components[frequencies.index(freq)] = pca.fit_transform(data).flatten()
        print(pca.explained_variance_ratio_)
    csv_file_path = os.path.join(dir, "1compPCA.csv")
    pd.DataFrame(np.array(components.tolist()).transpose()).to_csv(csv_file_path, index=False)

def evaluate():
    frequencies = ["050", "100", "125", "150", "200", "250"]
    dir = input("Enter the folder path of the CSV folders: ")
    components = np.empty((6), dtype=object)
    features = np.empty((6, 71), dtype=object)
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name == "1compPCA.csv":
                data = np.array(pd.read_csv(os.path.join(root, name))).transpose()
                for freq in range(6):
                    if str(type(components[freq])) == "<class 'NoneType'>":
                        components[freq] = np.array([data[freq][-30::]])
                    else:
                        #print(components[freq])
                        #print(data[freq])
                        components[freq] = np.vstack([components[freq], data[freq][-30::]])
                        #print(components[freq])
            elif name.endswith("meanfeatures.csv"):
                data = np.array(pd.read_csv(os.path.join(root, name))).transpose()
                freq = frequencies.index(name[:3])
                for feat in range(71):
                    if str(type(features[freq][feat])) == "<class 'NoneType'>":
                        features[freq][feat] = np.array([data[feat][-30::]])
                    else:
                        features[freq][feat] = np.vstack([features[freq][feat], data[feat][-30::]])

    results = np.empty((6, 71))
    criteria = np.empty((3, 6, 71))
    for freq in range(6):
        #print(components)
        print(frequencies[freq] + "kHz:" + str(fitness(components[freq])))
        for feat in range(71):
            results[freq][feat] = float(fitness(features[freq][feat])[0])
            criteria[0][freq][feat] = float(Mo(features[freq][feat]))
            criteria[1][freq][feat] = float(Tr(features[freq][feat]))
            criteria[2][freq][feat] = float(Pr(features[freq][feat]))
    pd.DataFrame(results).to_csv(dir + "\Fitness.csv", index=False)
    pd.DataFrame(criteria[0]).to_csv(dir + "\Mo.csv", index=False)
    pd.DataFrame(criteria[1]).to_csv(dir + "\Tr.csv", index=False)
    pd.DataFrame(criteria[2]).to_csv(dir + "\Pr.csv", index=False)

def giveTime():
    time = []
    for i in range(2000):
        time.append(i*(5e-7))
    return time

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
    print("0. Exit")

# Prompt the user to input the folder path
extract = bool(int(input("Extract Matlab (0=No, 1=Yes): ")))
if extract:
    folder_path = input("Enter the folder path of the Matlab files: ")
    Data_Preprocess.matToCsv(folder_path)
    print("Done")
    quit()
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
        evaluate()
    elif choice == '0':
        print("Exiting...")
        quit()
    else:
        print("Invalid choice. Please enter a number between 0 and 9.")

#C:\Users\geort\Desktop\Universty\PZT-L1-03
#C:\Users\geort\Desktop\Universty\PZT-CSV-L1-04\L104-AI_2019_12_11_12_59_25

#C:\Users\Jamie\Documents\Uni\Year 2\Q3+4\Project\Files

#C:\Users\Martin\Downloads\PZT-CSV\PZT-CSV-L01-5
