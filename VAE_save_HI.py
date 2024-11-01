import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os

def plot_images(seed, file_type, dir):
    """
    Plot 5x6 figure with graphs for all folds

    Parameters:
        - seed (int): Seed for reproducibility and filename
        - file_type (str): Indicates whether FFT or HLB data is being processed
        - dir (str): CSV root folder directory
    Returns: None
    """
    # Creating the 5x6 figure directory
    filedir = os.path.join(dir, f"big_VAE_graph_{file_type}_seed_{seed}")

    # List frequencies and panels
    panels = ("L103", "L105", "L109", "L104", "L123")
    freqs = ("050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz")

    # Initializing the figure
    nrows = 6
    ncols = 5
    fig, axs = plt.subplots(nrows, ncols, figsize=(40, 35))

    # Iterate over all folds of panel and frequency
    for i, freq in enumerate(freqs):
        for j, panel in enumerate(panels):

            # Create the filename for each individual graph
            filename = f"HI_graph_{freq}_{panel}.png"

            # Check if the file exists
            if os.path.exists(os.path.join(dir, filename)):

                # Load the individual graph
                img = mpimg.imread(os.path.join(dir, filename))

                # Display the image in the corresponding subplot and hide the axis
                axs[i, j].imshow(img)
                axs[i, j].axis('off')

            else:
                # If the image does not exist, print a warning and leave the subplot blank
                axs[i, j].text(0.5, 0.5, 'Image not found', ha='center', va='center', fontsize=12, color='red')
                axs[i, j].axis('off')

    # Change freqs for labelling
    freqs = ("050 kHz", "100 kHz", "125 kHz", "150 kHz", "200 kHz", "250 kHz")

    # Add row labels
    for ax, row in zip(axs[:, 0], freqs):
        ax.annotate(f'{row}', (-0.1, 0.5), xycoords='axes fraction', rotation=90, va='center', fontweight='bold',
                    fontsize=40)

    # Add column labels
    for ax, col in zip(axs[0], panels):
        ax.annotate(f'Test Sample {panels.index(col) + 1}', (0.5, 1), xycoords='axes fraction', ha='center',
                    fontweight='bold', fontsize=40)

    # Adjust spacing between subplots and save figure
    plt.tight_layout()
    plt.savefig(filedir)

def VAE_save_results(fitness_all, fitness_test, panel, freq, file_type, seed, dir):
    """
    Save VAE results to a CSV file

    Parameters:
        - fitness_all (float): Evaluation of fitness of all 5 HIs
        - fitness_test (float): Evaluation of fitness only for test HI
        - panel (str): Indicates test panel being processed
        - freq (str): Indicates frequency being processed
        - file_type (str): Indicates whether FFT or HLB data is being processed
        - seed (int): Seed for reproducibility and filename
        - dir (str): CSV root folder directory
    Returns: None
    """
    # Create filenames for the fitness-test and fitness-all CSV files
    filename_test = os.path.join(dir, f"fitness-test-{file_type}-seed-{seed}.csv")
    filename_all = os.path.join(dir, f"fitness-all-{file_type}-seed-{seed}.csv")

    # List frequencies
    freqs = ["050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz"]

    # Create the fitness-test file if it does not exist or load the existing fitness-test file
    if not os.path.exists(filename_test):
        df = pd.DataFrame(index=freqs)
    else:
        df = pd.read_csv(filename_test, index_col=0)

    # Ensure that the panel column exists in fitness-test
    if panel not in df.columns:
        df[panel] = None

    # Update the dataframe with the new results
    df.loc[freq, panel] = str(fitness_test)

    # Save the dataframe to a fitness-test CSV
    df.to_csv(filename_test)

    # Create the fitness-all file if it does not exist or load the existing fitness-all file
    if not os.path.exists(filename_all):
        df = pd.DataFrame(index=freqs)
    else:
        df = pd.read_csv(filename_all, index_col=0)

    # Ensure that the panel column exists in fitness-all
    if panel not in df.columns:
        df[panel] = None

    # Update the dataframe with the new results
    df.loc[freq, panel] = str(fitness_all)

    # Save the dataframe to a fitness-all CSV
    df.to_csv(filename_all)