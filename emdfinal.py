import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import emd

def my_get_next_imf(x, zoom=None, sd_thresh=0.1, max_iters = 10):

    proto_imf = x.copy()  # Take a copy of the input so we don't overwrite anything
    continue_sift = True  # Define a flag indicating whether we should continue sifting
    niters = 0            # An iteration counter

    if zoom is None:
        zoom = (0, x.shape[0])

    # Main loop - we don't know how many iterations we'll need so we use a ``while`` loop
    while continue_sift and niters < max_iters:
        niters += 1  # Increment the counter

        # Compute upper and lower envelopes
        upper_env = emd.sift.interp_envelope(proto_imf, mode='upper')
        lower_env = emd.sift.interp_envelope(proto_imf, mode='lower')

        # Compute average envelope
        avg_env = (upper_env+lower_env) / 2

        # Add a summary subplot
        plt.subplot(10, 1, niters)
        plt.plot(proto_imf[zoom[0]:zoom[1]], 'k')
        plt.plot(upper_env[zoom[0]:zoom[1]])
        plt.plot(lower_env[zoom[0]:zoom[1]])
        plt.plot(avg_env[zoom[0]:zoom[1]])

        # Should we stop sifting?
        stop, val = emd.sift.stop_imf_sd(proto_imf-avg_env, proto_imf, sd=sd_thresh)

        # Remove envelope from proto IMF
        proto_imf = proto_imf - avg_env

        # and finally, stop if we're stopping
        if stop:
            continue_sift = False

    # Return extracted IMF
    return proto_imf
#This for loop will print 3,4 or 5 graphs. The output we need is the last graph that is made.
def runEMD(data, time):
    proto_imf = []
    for i in range(8):
        x_values = []
        for j in range(8):
            k = 8 * i + j
            # Extracting data to perform FFT on=
            # Assuming the first column is time and the second column is the data to perform FFT on
            time = time.to_numpy()
            data = data.iloc[:, k].to_numpy()

            my_get_next_imf(data, zoom=None, sd_thresh=0.1)
            # proto_imf.insert(k,my_get_next_imf(data, zoom=None, sd_thresh=0.1))
            plt.show()



    #plt.plot(time, data, color='red')
    #plt.plot(time, upper_env, color='blue')
    #plt.plot(time, lower_env, color='green')
    #plt.plot(time, avg_env, color='black')
    #plt.legend(['Signal', 'Upper Env', 'Lower Env', 'Avg Env'])
    #plt.show()

'''
    def my_get_next_imf(x, zoom=None, sd_thresh=0.1, max_iters=10):

        proto_imf = x.copy()  # Take a copy of the input so we don't overwrite anything
        continue_sift = True  # Define a flag indicating whether we should continue sifting
        niters = 0  # An iteration counter

        if zoom is None:
            zoom = (0, x.shape[0])

        # Main loop - we don't know how many iterations we'll need so we use a ``while`` loop
        while continue_sift and niters < max_iters:
            niters += 1  # Increment the counter

            # Compute upper and lower envelopes
            upper_env = emd.sift.interp_envelope(proto_imf, mode='upper')
            lower_env = emd.sift.interp_envelope(proto_imf, mode='lower')

            # Compute average envelope
            avg_env = (upper_env + lower_env) / 2

            # Add a summary subplot
            plt.subplot(10, 1, niters)
            plt.plot(proto_imf[zoom[0]:zoom[1]], 'k')
            plt.plot(upper_env[zoom[0]:zoom[1]])
            plt.plot(lower_env[zoom[0]:zoom[1]])
            plt.plot(avg_env[zoom[0]:zoom[1]])

            # Should we stop sifting?
            stop, val = emd.sift.stop_imf_sd(proto_imf - avg_env, proto_imf, sd=sd_thresh)

            # Remove envelope from proto IMF
            proto_imf = proto_imf - avg_env

            # and finally, stop if we're stopping
            if stop:
                continue_sift = False

        # Return extracted IMF
        return proto_imf


    # This for loop will print 3,4 or 5 graphs. The output we need is the last graph that is made.
    for df in dataframes:
        # Assuming the first column is time and the second column is the data to perform FFT on
        time = df.iloc[:, 0].to_numpy()
        data = df.iloc[:, 1].to_numpy()

        my_get_next_imf(data, zoom=None, sd_thresh=0.1)
        plt.show()

'''