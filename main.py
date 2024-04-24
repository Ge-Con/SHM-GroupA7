import pandas as pd
import numpy as np
import os

import extract_features
from Signal_Processing import fft, emdfinal, stft, hilbert, Data_Preprocess

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
                arrayfile1.to_csv(csv_file_path1, index=False)
                arrayfile2.to_csv(csv_file_path2, index=False)

def saveSTFT(dir):
    print("Executing STFT on data:...")
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith('kHz.csv'):
                # print("good")
                data = pd.read_csv(os.path.join(root, name))
                arrayfile1, arrayfile2 = stft.Short_Fourier(data)
                csv_file_path1 = os.path.join(root, f"{name.replace('kHz.csv', '')}_{'kHz_STFT_Freq.csv'}")
                csv_file_path2 = os.path.join(root, f"{name.replace('kHz.csv', '')}_{'kHz_STFT_Amp.csv'}")
                arrayfile1.to_csv(csv_file_path1, index=False)
                arrayfile2.to_csv(csv_file_path2, index=False)

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

def save_Time_Features(dir):
    print("Extracting Time Domain Features:...")
    toDelete = np.zeros(20, dtype=int)
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith('EMD.csv') or name.endswith('Hilbert.csv'):
                data = pd.read_csv(os.path.join(root, name))
                data = np.array(data)
                arrayfile1, arrayfile2 = extract_features.time_to_feature(data)
                #print("IMPORTED FIRST: ",arrayfile2)
                for i in range(len(arrayfile2)):
                    toDelete[arrayfile2[i]] = toDelete[arrayfile2[i]] + 1
                #print("To Delete: ", toDelete)

    # Get the indices that would sort the array
    sorted_indices = np.argsort(toDelete)[::-1]
    # Get the indices of the 10 lowest integers in the initial array
    indices_of_10_highest = sorted_indices[:9]
    print(indices_of_10_highest)

    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith('EMD.csv') or name.endswith('Hilbert.csv'):
                data = pd.read_csv(os.path.join(root, name))
                data = np.array(data)
                features = extract_features.time_to_feature_Reduced(data, indices_of_10_highest)
                # Determine the new filename based on the original filename
                new_filename = name.replace('EMD.csv', 'EMD-Features.csv').replace('Hilbert.csv', 'Hilbert-Features.csv')
                # Construct the new file path
                csv_file_path = os.path.join(root, new_filename)

                features.to_csv(csv_file_path, index=False)
                '''call new function with toDelete array and return the reduced size feature array'''
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
    print("6. Extract features from time domain data")
    print("7. Exit")

def option1(loc):
    saveFFT(loc)

def option2(loc):
    saveEMD(loc)

def option3(loc):
    saveHilbert(loc)

def option4(loc):
    saveSTFT(loc)

def option5(loc):
    saveFFT(loc)
    saveEMD(loc)
    saveHilbert(loc)
    saveSTFT(loc)

def option6(loc):
    save_Time_Features(loc)

# Prompt the user to input the folder path
folder_path = input("Enter the folder path of the Matlab files: ")
Data_Preprocess.matToCsv(folder_path)
csv_dir = folder_path.replace('PZT','PZT-CSV')
# Main program loop
while True:
    main_menu()
    choice = input("Enter your choice: ")

    if choice == '1':
        option1(csv_dir)
    elif choice == '2':
        option2(csv_dir)
    elif choice == '3':
        option3(csv_dir)
    elif choice == '4':
        option4(csv_dir)
    elif choice == '5':
        option5(csv_dir)
    elif choice == '6':
        option6(csv_dir)
    elif choice == '7':
        print("Exiting...")
        break
    else:
        print("Invalid choice. Please enter a number between 1 and 7.")

#C:\Users\geort\Desktop\Universty\PZT-L1-03
#C:\Users\geort\Desktop\Universty\PZT-CSV-L1-04\L104-AI_2019_12_11_12_59_25