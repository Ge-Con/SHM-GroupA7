import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from time import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from Interpolating import scale_exact
from scipy.stats import pearsonr
from sklearn.preprocessing import Normalizer
from scipy.signal import resample_poly
import math
import scipy.interpolate as interp
import csv
from prognosticcriteria_v2 import Mo_single, Pr, Tr, Mo, fitness, Pr_single
import os
from PIL import Image

# This function is to create the big 5x6 graph in the paper
def plot_images(seed, file_type, dir):

    # Formatting
    filedir = os.path.join(dir, f"big_VAE_graph_{file_type}_seed_{seed}")
    nrows = 6
    ncols = 5
    panels = ("L103", "L105", "L109", "L104", "L123")
    freqs = ("050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz")
    fig, axs = plt.subplots(nrows, ncols, figsize=(40, 35))  # Adjusted figure size

    # Iterate over all folds
    for i, freq in enumerate(freqs):
        for j, panel in enumerate(panels):

            # Create the filename
            filename = f"HI_graph_{freq}_{panel}.png"

            # Check if the file exists
            if os.path.exists(os.path.join(dir, filename)):

                # Load the image
                img = mpimg.imread(os.path.join(dir, filename))

                # Display the image in the corresponding subplot
                axs[i, j].imshow(img)
                axs[i, j].axis('off')  # Hide the axes

            else:
                # If the image does not exist, print a warning and leave the subplot blank
                axs[i, j].text(0.5, 0.5, 'Image not found', ha='center', va='center', fontsize=12, color='red')
                axs[i, j].axis('off')

    freqs = ("050 kHz", "100 kHz", "125 kHz", "150 kHz", "200 kHz", "250 kHz")

    # Add row labels
    for ax, row in zip(axs[:, 0], freqs):
        ax.annotate(f'{row}', (-0.1, 0.5), xycoords='axes fraction', rotation=90, va='center', fontweight='bold',
                    fontsize=40)

    # Add column labels
    for ax, col in zip(axs[0], panels):
        ax.annotate(f'Test Sample {panels.index(col) + 1}', (0.5, 1), xycoords='axes fraction', ha='center',
                    fontweight='bold', fontsize=40)

    plt.tight_layout()  # Adjust spacing between subplots
    plt.savefig(filedir)  # Save figure

def store_hyperparameters(fitness_all, fitness_test, panel, freq, file_type, seed, dir):

    filename_test = os.path.join(dir, f"fitness-test-{file_type}-seed-{seed}.csv")
    filename_all = os.path.join(dir, f"fitness-all-{file_type}-seed-{seed}.csv")
    freqs = ["050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz"]

    if not os.path.exists(filename_test):
        df = pd.DataFrame(index=freqs)
    else:
        # Load the existing file if it exists
        df = pd.read_csv(filename_test, index_col=0)

    # Ensure that the panel column exists
    if panel not in df.columns:
        df[panel] = None

    # Update the dataframe with the new parameters
    df.loc[freq, panel] = str(fitness_test)

    # Save the dataframe back to the CSV
    df.to_csv(filename_test)

    if not os.path.exists(filename_all):
        df = pd.DataFrame(index=freqs)
    else:
        # Load the existing file if it exists
        df = pd.read_csv(filename_all, index_col=0)

    # Ensure that the panel column exists
    if panel not in df.columns:
        df[panel] = None

    # Update the dataframe with the new parameters
    df.loc[freq, panel] = str(fitness_all)

    # Save the dataframe back to the CSV
    df.to_csv(filename_all)