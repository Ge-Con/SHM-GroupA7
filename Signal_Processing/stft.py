import numpy as np
import pandas as pd
from scipy import signal

def Short_Fourier(data):
    """
            Extracts the next Intrinsic Mode Function (IMF) from a time-domain signal (to be used in empirical mode decomposition(EMD)).
            Simple explanation: EMD splits a signal into a multitude of signals(sifting). Important to be noted is that this.
            also includes non-sinus functions(like polynomials). These functions are called intrinsic mode functions(IMF).

            Parameters:
            - x (1D array): The input time-domain signal.
            - sd_thresh (float, optional): The threshold for standard deviation to decide when to stop the sifting process.
            - max_iters (int, optional): The maximum number of iterations for the sifting process.

            Returns:
            - proto_imf (1D array): The extracted Intrinsic Mode Function (IMF).

            Example:
            # Example usage of the function
            result = my_get_next_imf(sensor_data)
    """
    amp_arr = [] #initialize array

    # Loop to process the data and perform STFT
    for i in range(7):
        for j in range(8):
            k = 8 * i + j
            y = data.iloc[:, k]  # Extracting data for FFT
            fs = 1 / 5e-7  # Sampling frequency

            # Perform Short-Time Fourier Transform (STFT)
            f, t, Zxx = signal.stft(y, fs, nperseg=250)
            amp = np.abs(Zxx)  # Get the absolute values of the STFT
            amp_arr.append(amp)  # Append amplitude data to the list

    # Flatten the amp_arr into a 1D list
    flat = [item for sublist in amp_arr for item in sublist]

    # Dimensions of the original 3D list
    dim1 = 56   # This is the number of elements in the outermost dimension (amount of signals 8*7, we dont include the actuator(would be 8*8))
    dim2 = 126  # This is the number of elements in the middle dimension (each segment has 17 frequencies)
    dim3 = 17   # This is the number of elements in the innermost dimension (number of segments, on which FFT is applied on seperatly)

    # Create an empty 3D list with the original dimensions
    unflattened_list = [[[0 for _ in range(dim3)] for _ in range(dim2)] for _ in range(dim1)]


    # #Unflattening example for future uses.
    # # Populate the unflattened_list with values from flat
    # index = 0
    # for i in range(dim1):
    #     for j in range(dim2):
    #             unflattened_list[i][j] = flat[index]
    #             index += 1

    # Convert the unflattened_list to a pandas DataFrame, this unflattened_df is not used here. It is only present to provide an example for an external user
    unflattened_df = pd.DataFrame(np.array(unflattened_list).reshape((dim1, dim2 * dim3)))

    return pd.DataFrame(flat)  #Flat is needed to run through main. After main it is unflattened again

