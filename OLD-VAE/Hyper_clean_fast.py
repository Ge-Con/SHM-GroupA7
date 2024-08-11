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
from Hyper_clean import mergedata, DCloss, test_fitness, find_largest_array_size

# Same as Hyper_clean, functions have been imported
tf.compat.v1.reset_default_graph()
seed = 140
tf.random.set_seed(seed)
np.random.seed(seed)
dir_root = r"C:\Users\pablo\Downloads\SHM_Concatenated_FFT_Features"
# C:\Users\pablo\Downloads\PZT Output folder

# This function is to create the big 5x6 graph in the paper
def plot_images(seed):
    
    # Directory, and formatting
    global dir_root
    filedir = os.path.join(dir_root, f"big_VAE_graph_seed_{seed}")
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
            if os.path.exists(os.path.join(dir_root, filename)):
                
                # Load the image
                img = mpimg.imread(os.path.join(dir_root, filename))

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
        ax.annotate(f'{row}', (-0.1, 0.5), xycoords = 'axes fraction', rotation = 90, va = 'center', fontweight = 'bold', fontsize = 40)

    # Add column labels
    for ax, col in zip(axs[0], panels):
        ax.annotate(f'Test Sample {panels.index(col)+1}', (0.5, 1), xycoords = 'axes fraction', ha = 'center', fontweight = 'bold', fontsize = 40)

    plt.tight_layout()  # Adjust spacing between subplots
    plt.savefig(filedir) # Save figure

# Same as the function in Hyper_clean but now includes the seed in the filename
# Also this function doesn't create a new file where hyperparameters are stored, instead the previously created file will be read
# So it only generates results
def store_hyperparameters(fitness_all, fitness_test, panel, freq, seed):
    global dir_root
    filename_test = os.path.join(dir_root, f"fitness-test-seed-{seed}.csv")
    filename_all = os.path.join(dir_root, f"fitness-all-seed-{seed}.csv")
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

def train_vae(hidden_1, batch_size, learning_rate, epochs, train_data, test_data, scaler, pca, seed):
   
    tf.random.set_seed(seed)
    n_input = data.shape[1]  # Number of features
    hidden_2 = 1
    display = 200

    # Xavier initialization for weights
    def xavier_init(fan_in, fan_out, constant=1, seed):
        
        tf.random.set_seed(seed)
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random.uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

    # Architecture + losses
    tf.compat.v1.disable_eager_execution()
    
    x = tf.compat.v1.placeholder(tf.float32, [None, n_input])
    
    w1 = tf.Variable(xavier_init(n_input, hidden_1), seed=seed)
    b1 = tf.Variable(tf.zeros([hidden_1, ]))
    
    mean_w = tf.Variable(xavier_init(hidden_1, hidden_2), seed=seed)
    mean_b = tf.Variable(tf.zeros([hidden_2, ]))
    
    logvar_w = tf.Variable(xavier_init(hidden_1, hidden_2), seed=seed)
    logvar_b = tf.Variable(tf.zeros([hidden_2, ]))
    
    dw1 = tf.Variable(xavier_init(hidden_2, hidden_1), seed=seed)
    db1 = tf.Variable(tf.zeros([hidden_1, ]))
    
    dw2 = tf.Variable(xavier_init(hidden_1, n_input), seed=seed)
    db2 = tf.Variable(tf.zeros([n_input, ]))
    
    l1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
    mean = tf.matmul(l1, mean_w) + mean_b
    logvar = tf.matmul(l1, logvar_w) + logvar_b
    eps = tf.random.normal(tf.shape(logvar), 0, 1, dtype=tf.float32)
    
    z = tf.multiply(tf.sqrt(tf.exp(logvar)), eps) + mean
    l2 = tf.nn.sigmoid(tf.matmul(z, dw1) + db1)
    pred = tf.matmul(l2, dw2) + db2

    reloss = tf.reduce_sum(tf.square(pred - x))
    klloss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), 1)
    fealoss = DCloss(z, batch_size)
    
    loss = tf.reduce_mean(0.1 * reloss + 0.6 * klloss + 10 * fealoss)

    optm = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)

    # Training
    begin_time = time()

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    
    print('Start training!!!')
    num_batch = int(train_data.shape[0] / batch_size)
    if num_batch == 0:
        raise ValueError("Batch size is too large for the given data.")

    for epoch in range(epochs):
        for i in range(num_batch):
            batch_xs = train_data[i * batch_size:(i + 1) * batch_size]
            _, cost = sess.run([optm, loss], feed_dict={x: batch_xs})

    print('Training finished!!!')
    end_time = time()
    print(f"Training time: {end_time - begin_time:.2f} seconds")

    z_test = sess.run(z, feed_dict={x: test})
    z_test = z_test.transpose()
    
    z_train = []
    
    for j in tuple(x for x in panels if x != panel):
        
        single_panel_data = pd.read_csv(dir_root + "\concatenated_" + freq + "_" + j + "_FFT_Features.csv", header=None).values.transpose()
        single_panel_data = np.delete(single_panel_data, -1, axis=1)
        
        single_panel_data = scaler.transform(single_panel_data)
        single_panel_data = pca.transform(single_panel_data)
        
        z_train_individual = sess.run(z, feed_dict={x: single_panel_data})
        z_train.append(z_train_individual)
        
    # Scaling and interpolating
    for i in range(len(z_train)):
        z_train[i] = z_train[i].transpose()
        
    max = find_largest_array_size(z_train)
    for i in range(len(z_train)):
        if z_train[i].size < max:
            interp_function = interp.interp1d(np.arange(z_train[i].size), z_train[i])
            arr_stretch = arr_interp(np.linspace(0, z_train[i].size - 1, max))
            z_train[i] = arr_stretch
            
    z_train = np.vstack(z_train)
    
    if z_test.size != z_train.shape[1]:
        interp_function = interp.interp1d(np.arange(z_test.size), z_test)
        arr_stretch = arr_interp(np.linspace(0, z_test.size - 1, z_train.shape[1]))
        z_test = arr_stretch
        
    z_all = np.append(z_train, z_test, axis=0)
    
    sess.close()
    
    return [z_all, z_train, z_test]

# Panels and freqs for iteration, result_dictionary, counter and colors
panels = ("L103", "L105", "L109", "L104", "L123")
freqs = ("050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz")
result_dictionary = {}
counter = 0
colors = ("b", "g", "y", "r", "m")

# Reading the hyperparameter file
# Hardcoded for FFT - need to change this in main
hyperparameters_df = pd.read_csv(dir_root + '/hyperparameters-opt-FFT.csv', index_col=0)

# This is relevant for later, time_steps is the length that we want the HI to have after scaling
time_steps = 30
num_HIs = 5  # Number of rows in f_all_array
num_freqs = len(freqs)
num_panels = len(panels)

# Initialize the final array with zeros. This final array will contain all HIs, for all folds. It will be 4 dimensional
# This array is something JJ asked for that is needed for the Ensemble model, thus needs to be saved in our code
hi_full_array = np.zeros((num_panels, num_freqs, num_HIs, time_steps))

for panel_idx, panel in enumerate(panels):
    for freq_idx, freq in enumerate(freqs):

        train_filenames = []

        for i in tuple(x for x in panels if x != panel):
            filename = os.path.join(dir_root, f"concatenated_{freq}_{i}_FFT_Features.csv")
            train_filenames.append(filename)

        result_dictionary[f"{panel}{freq}"] = []

        counter += 1
        print("Counter: ", counter)
        print("Panel: ", panel)
        print("Freq: ", freq)
        
        train_data, flags = mergedata(train_filenames)
        train_data.drop(train_data.columns[len(train_data.columns) - 1], axis=1, inplace=True)

        test_filename = os.path.join(dir_root, f"concatenated_{freq}_{panel}_FFT_Features.csv")
        test_data = pd.read_csv(test_filename, header=None).values.transpose()
        test_data = np.delete(test_data, -1, axis=1)

        scaler = StandardScaler()
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)

        pca = PCA(n_components=30)
        pca.fit(train_data)
        train_data = pca.transform(train_data)
        test_data = pca.transform(test_data)

        # Read hyperparameters from file
        # This is the MAIN difference between Hyper_clean_fast and Hyper_clean
        hyperparameters_str = hyperparameters_df.loc[freq, panel]
        hyperparameters = eval(hyperparameters_str)

        health_indicators = train_vae(hyperparameters[0][0], hyperparameters[0][1],
                                      hyperparameters[0][2], hyperparameters[0][3], train_data, test_data, scaler, pca, seed)


        fitness_all = fitness(health_indicators[0])
        print("Fitness all", fitness_all)

        fitness_test = test_fitness(health_indicators[2], health_indicators[1])
        print("Fitness test", fitness_test)

        result_dictionary[f"{panel}{freq}"].append([fitness_all, fitness_test])

        # Name and directory for the big 5x6 graph
        graph_hi_filename = f"HI_graph_{freq}_{panel}_seed_{seed}"
        graph_hi_dir = os.path.join(dir_root, graph_hi_filename)

        # Plotting all 5 HIs in one graph
        fig = plt.figure()

        # Finding train panels
        train_panels = [k for k in panels if k != panel]

        # Normalizing x-axis. First, create list from 0 to length of HI (e.g. 90 timesteps)
        # Then normalize it to go from 0 to 1, and *100.
        x = np.arange(0, health_indicators[2].shape[1], 1)
        x = x*(1/(x.shape[0]-1))
        x = x * 100

        # Plot the 4 train HIs
        for i, hi in enumerate(health_indicators[1]):
            plt.plot(x, hi, label=f'Sample {panels.index(train_panels[i]) + 1}: Train')

        # Plot the test HI
        for i, hi in enumerate(health_indicators[2]):
            plt.plot(x, hi, label=f'Sample {panels.index(panel) + 1}: Test')

        # Formatting
        plt.xlabel('Lifetime (%)')
        plt.ylabel('Health Indicators')
        plt.title('Train and Test Health Indicators over Time')
        plt.legend()

        # Save and close
        plt.savefig(graph_hi_dir)
        plt.close(fig)

        fitness_test = (fitness_test)
        fitness_all = (fitness_all)
        store_hyperparameters(fitness_all, fitness_test, panel, freq, seed)

        # Here we use the function scale_exact created by Martin in another file, and apply it to each row of z_all, such that all HIs have the correct dimension
        # Then reshape to the required 4D shape
        # Remember health_indicators[0] is z_all
        z_all_modified = np.array([scale_exact(row) for row in health_indicators[0][:num_HIs]])
        z_all_modified = z_all_modified.reshape(num_HIs, time_steps)

        # Assign z_all_modified to the correct position in hi_full_array
        hi_full_array[panel_idx, freq_idx] = z_all_modified

# Saving the HIs to the format that JJ wanted for the ensemble model. Will save to current directory
label = f"VAE_seed_{seed}"
savedir = dir_root + '\\' + label
np.save(savedir, hi_full_array)

# Plot big 5x6 graph (function at start of code)
plot_images(seed)

# Writing results -> I think this part is not even needed?
with open("results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, result_dictionary.keys())
    w.writeheader()
    w.writerow(result_dictionary)