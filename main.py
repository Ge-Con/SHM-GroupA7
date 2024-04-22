import scipy.io
import pandas as pd
import numpy as np
import os
import Data_Preprocess
import emdfinal
import fft
import stft
import hilbert


pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 400)
np.set_printoptions(linewidth=400)


'''
Parameters:
    - param1 (type): Description of param1.
    - param2 (type): Description of param2.

    Returns:
    - return_type: Description of the return value(s).
'''
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
                arrayfile1 = emdfinal.runEMD(data,giveTime())
                csv_file_path1 = os.path.join(root, f"{name.replace('kHz.csv', '')}_{'kHz_EMD.csv'}")
                arrayfile1.to_csv(csv_file_path1, index=False)


def saveHilbert(dir):
    print("Executing Hilbert on data:...")
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith('kHz.csv'):
                # print("good")
                data = pd.read_csv(os.path.join(root, name))
                arrayfile1 = hilbert.Hilbert(data,giveTime())
                csv_file_path1 = os.path.join(root, f"{name.replace('kHz.csv', '')}_{'kHz_Hilbert.csv'}")
                # csv_file_path2 = os.path.join(root, f"{name.replace('kHz.csv', '')}_{'kHz_Hilb_Amp.csv'}")
                arrayfile1.to_csv(csv_file_path1, index=False)
                # arrayfile2.to_csv(csv_file_path2, index=False)

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
    print("5.All of the above")
    print("6. Exit")

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
        print("Exiting...")
        break
    else:
        print("Invalid choice. Please enter a number between 1 and 6.")