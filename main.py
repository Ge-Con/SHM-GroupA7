import scipy.io
import pandas as pd
import numpy as np
import os
import Data_Preprocess
import fft


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
            if name.endswith('.csv'):
                print("good")
                data = pd.read_csv(os.path.join(root, name))
                arrayfile1, arrayfile2 = fft.fast_fourier(data)
                csv_file_path1 = os.path.join(root, f"{name.replace('kHz.csv', '')}_{'kHz_FFT_Freq.csv'}")
                csv_file_path2 = os.path.join(root, f"{name.replace('kHz.csv', '')}_{'kHz_FFT_Ampli.csv'}")
                # print(arrayfile)
                arrayfile1.to_csv(csv_file_path1, index=False)
                arrayfile2.to_csv(csv_file_path2, index=False)

saveFFT(r"C:\Users\edlyn\Desktop\PZT-CSV-L1-04")
#Data_Preprocess.matToCsv(r"C:\Users\edlyn\Desktop\PZT-L1-04")
'''
Frameworks

raw data to AI
raw data to signal processing
raw data to feature extraction

signal processing to feature extraction
signal processing to AI


'''

