import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import tensorflow as tf
import csv
import os
import VAE_save_hyperparameters as VAEparameters
from Interpolating import scale_exact
from prognosticcriteria_v2 import fitness, test_fitness

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
    
def VAE_train_run(dir):
    """
        Run VAE result generation main loop

        Parameters:
            - dir (str): Directory containing data
        Returns: None
        """
    # Connect global variable for seed
    global vae_seed

    # Set random seeds
    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)

    # List frequencies, filenames and samples
    freqs = ("050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz")
    filenames = ["FFT_FT_Reduced", "HLB_FT_Reduced"]
    panels = ("L103", "L105", "L109", "L104", "L123")

    # Determine dimensions of data
    time_steps = 30
    num_HIs = 5
    num_freqs = len(freqs)
    num_panels = len(panels)

    # Iterate over filenames
    for file_type in filenames:
        counter = 0
        result_dictionary = {}

        # Read hyperparameters from CSV and initialize array to save HIs
        hyperparameters_df = pd.read_csv(os.path.join(dir, f'hyperparameters-opt-{file_type}.csv'), index_col=0)
        hi_full_array = np.zeros((num_panels, num_freqs, num_HIs, time_steps))

        # Iterate over all panels and frequencies
        for panel_idx, panel in enumerate(panels):
            for freq_idx, freq in enumerate(freqs):

                train_filenames = []

                # Save name of train panels for current fold
                for i in tuple(x for x in panels if x != panel):
                    filename = os.path.join(dir, f"concatenated_{freq}_{i}_{file_type}.csv")
                    train_filenames.append(filename)

                result_dictionary[f"{panel}{freq}"] = []

                # Output progress
                counter += 1
                print("Counter: ", counter)
                print("Panel: ", panel)
                print("Freq: ", freq)
                print("SP Features: ", file_type)

                # Merge train data and delete last column
                vae_train_data = VAEparameters.VAE_merge_data(train_filenames)
                vae_train_data.drop(vae_train_data.columns[len(vae_train_data.columns) - 1], axis=1, inplace=True)

                # Read test data
                test_filename = os.path.join(dir, f"concatenated_{freq}_{panel}_{file_type}.csv")
                vae_test_data = pd.read_csv(test_filename, header=None).values.transpose()
                vae_test_data = np.delete(vae_test_data, -1, axis=1)

                # Normalize the train and test data, with respect to the train data
                vae_scaler = VAEparameters.StandardScaler()
                vae_scaler.fit(vae_train_data)
                vae_train_data = vae_scaler.transform(vae_train_data)
                vae_test_data = vae_scaler.transform(vae_test_data)

                # Apply PCA to the train and test data, fit to the train data
                vae_pca = VAEparameters.PCA(n_components=30)
                vae_pca.fit(vae_train_data)
                vae_train_data = vae_pca.transform(vae_train_data)
                vae_test_data = vae_pca.transform(vae_test_data)

                # Convert hyperparameter dataframe
                hyperparameters_str = hyperparameters_df.loc[freq, panel]
                hyperparameters = eval(hyperparameters_str)

                # Generate HIs with train_vae function
                health_indicators = VAEparameters.VAE_train(hyperparameters[0][0], hyperparameters[0][1],
                                                            hyperparameters[0][2], hyperparameters[0][3],
                                                            hyperparameters[0][4],
                                                            hyperparameters[0][5], hyperparameters[0][6],
                                                            vae_train_data, vae_test_data, vae_scaler, vae_pca,
                                                            vae_seed,
                                                            file_type, panel, freq, dir)

                # Evaluate and output fitness for all 5 HIs and only for the test HI
                fitness_all = fitness(health_indicators[0])
                fitness_test = test_fitness(health_indicators[2], health_indicators[1])
                print("Fitness all", fitness_all)
                print("Fitness test", fitness_test)

                # Append values to the result dictionary
                result_dictionary[f"{panel}{freq}"].append([fitness_all, fitness_test])

                # Generate directory for graphs
                graph_hi_filename = f"HI_graph_{freq}_{panel}_{file_type}_seed_{vae_seed}"
                graph_hi_dir = os.path.join(dir, graph_hi_filename)

                # Save train_panel names, and create x variable to scale y-axis from 0-100
                train_panels = [k for k in panels if k != panel]
                x = np.arange(0, health_indicators[2].shape[1], 1)
                x = x * (1 / (x.shape[0] - 1))
                x = x * 100

                # Initialize figure
                fig = plt.figure()

                # Iterate over train panels
                for i in range(len(train_panels)):
                    # Save HI and standard deviation
                    hi = health_indicators[1][i]
                    std_dev = health_indicators[4][i]

                    # Plot train HIs and test HI with error bars
                    plt.errorbar(x, hi, yerr=std_dev, label=f'Sample {i + 1}: Train', color=f'C{i}', ecolor='blue',
                                 elinewidth=2, capsize=5)
                plt.errorbar(x, health_indicators[2][0], yerr=health_indicators[5][0],
                             label=f'Sample {panels.index(panel) + 1}: Test', color='red', ecolor='salmon',
                             elinewidth=2, capsize=5)

                # Graph formatting
                plt.xlabel('Lifetime (%)')
                plt.ylabel('Health Indicators')
                plt.title('Train and Test Health Indicators over Time')
                plt.legend()

                # Saving and closing figure
                plt.savefig(graph_hi_dir)
                plt.close(fig)

                # Putting fitness values in parantheses
                fitness_test = (fitness_test)
                fitness_all = (fitness_all)

                # Save results to a CSV
                VAE_save_results(fitness_all, fitness_test, panel, freq, file_type, vae_seed, dir)

                # Scaling and reshaping HI through interpolation
                z_all_modified = np.array([scale_exact(row) for row in health_indicators[0][:num_HIs]])
                z_all_modified = z_all_modified.reshape(num_HIs, time_steps)

                # Assign z_all_modified to the correct position in hi_full_array
                hi_full_array[panel_idx, freq_idx] = z_all_modified

        # Saving array of HIs
        label = f"VAE_{file_type}_seed_{vae_seed}"
        savedir = dir + '\\' + label
        np.save(savedir, hi_full_array)

        # Plotting 5x6 graph with all folds
        plot_images(vae_seed, file_type, dir)

        # Saving results dictionary to a CSV
        with open(f"results_{file_type}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, result_dictionary.keys())
            w.writeheader()
            w.writerow(result_dictionary)