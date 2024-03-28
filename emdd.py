import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import emd

# Assuming df is your dataframe read from csv
df = pd.read_csv('Actionneur1/measured_data_rep_1_Time_Response.csv')

dataframes = [df[['Column_1', 'Column_2']], df[['Column_1', 'Column_3']], df[['Column_1', 'Column_4']],
              df[['Column_1', 'Column_5']], df[['Column_1', 'Column_6']], df[['Column_1', 'Column_7']],
              df[['Column_1', 'Column_8']], df[['Column_1', 'Column_9']]]

x_values = []

for df in dataframes:
    # Assuming the first column is time and the second column is the data to perform FFT on
    time = df.iloc[:, 0].to_numpy()
    data = df.iloc[:, 1].to_numpy()

    proto_imf = data.copy()
    upper_env = emd.sift.interp_envelope(proto_imf, mode='upper')
    lower_env = emd.sift.interp_envelope(proto_imf, mode='lower')
    #average envelope
    avg_env = (upper_env + lower_env) / 2

    data = data - avg_env
    proto_imf = data.copy() - avg_env
    upper_env = emd.sift.interp_envelope(proto_imf, mode='upper')
    lower_env = emd.sift.interp_envelope(proto_imf, mode='lower')
    # average envelope
    avg_env = (upper_env + lower_env) / 2

    plt.plot(time, data, color='red')
    #plt.plot(time, upper_env, color='blue')
    #plt.plot(time, lower_env, color='green')
    plt.plot(time, avg_env, color='black')
    plt.legend(['Signal', 'Upper Env', 'Lower Env', 'Avg Env'])
    plt.show()