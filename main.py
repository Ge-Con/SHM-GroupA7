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
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith('kHz.csv'):
                data = pd.read_csv(os.path.join(root, name))
                arrayfile1 = emdfinal.runEMD(data,giveTime())
                csv_file_path1 = os.path.join(root, f"{name.replace('kHz.csv', '')}_{'kHz_EMD.csv'}")
                arrayfile1.to_csv(csv_file_path1, index=False)


def saveHilbert(dir):
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

saveEMD(r"C:\Users\geort\Desktop\Universty\PZT-CSV-L1-03")