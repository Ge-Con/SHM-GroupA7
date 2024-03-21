import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyemd import emd

# Assuming df is your dataframe read from csv
df = pd.read_csv('Actionneur1/measured_data_rep_1_Time_Response.csv')

dataframes = [df[['Column_1', 'Column_2']], df[['Column_1', 'Column_3']], df[['Column_1', 'Column_4']],
              df[['Column_1', 'Column_5']], df[['Column_1', 'Column_6']], df[['Column_1', 'Column_7']],
              df[['Column_1', 'Column_8']], df[['Column_1', 'Column_9']]]

x_values = []

for df in dataframes:
    # Assuming the first column is time and the second column is the data to perform EMD on
    time = df.iloc[:, 0]
    data = df.iloc[:, 1]

    # Perform Empirical Mode Decomposition (EMD)
    imfs = emd(data)

    # Plot the results
    plt.figure()
    for imf in imfs:
        plt.plot(time, imf)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('IMFs obtained from EMD')
    plt.show()
