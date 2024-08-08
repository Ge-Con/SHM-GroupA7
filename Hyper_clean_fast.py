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
def plot_images():
    global dir_root
    global seed
    filedir = os.path.join(dir_root, f"big_VAE_graph_seed_{seed}")
    nrows = 6
    ncols = 5
    panels = ("L103", "L105", "L109", "L104", "L123")
    freqs = ("050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz")
    fig, axs = plt.subplots(nrows, ncols, figsize=(40, 35))  # Adjusted figure size


    for i, freq in enumerate(freqs):
        for j, panel in enumerate(panels):
            # Generate the filename
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
    plt.savefig(filedir)

# Same as the function in Hyper_clean but now includes the seed in the filename
# Also this function doesn't create a new file where hyperparameters are stored, instead the previously created file will be read
# So it only generates results
def store_hyperparameters(params_test, params_hi_train, panel, freq):
    global dir_root
    global seed
    filename_test = os.path.join(dir_root, f"fitness-test-seed-{seed}.csv")
    filename_train = os.path.join(dir_root, f"fitness-all-seed-{seed}.csv")
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
    df.loc[freq, panel] = str(params_test)

    # Save the dataframe back to the CSV
    df.to_csv(filename_test)

    if not os.path.exists(filename_train):
        df = pd.DataFrame(index=freqs)
    else:
        # Load the existing file if it exists
        df = pd.read_csv(filename_train, index_col=0)

    # Ensure that the panel column exists
    if panel not in df.columns:
        df[panel] = None

    # Update the dataframe with the new parameters
    df.loc[freq, panel] = str(params_hi_train)

    # Save the dataframe back to the CSV
    df.to_csv(filename_train)

def train_vae(hidden_1, batch_size, learning_rate, epochs):
    global data
    global seed
    global test
    global valid
    global scaler
    global pca

    tf.random.set_seed(seed)
    n_input = data.shape[1]  # Number of features
    hidden_2 = 1
    display = 50

    # Xavier initialization for weights
    def xavier_init(fan_in, fan_out, constant=1):
        global seed
        tf.random.set_seed(seed)
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random.uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

    # Architecture + losses
    tf.compat.v1.disable_eager_execution()
    x = tf.compat.v1.placeholder(tf.float32, [None, n_input])
    w1 = tf.Variable(xavier_init(n_input, hidden_1))
    b1 = tf.Variable(tf.zeros([hidden_1, ]))
    mean_w = tf.Variable(xavier_init(hidden_1, hidden_2))
    mean_b = tf.Variable(tf.zeros([hidden_2, ]))
    logvar_w = tf.Variable(xavier_init(hidden_1, hidden_2))
    logvar_b = tf.Variable(tf.zeros([hidden_2, ]))
    dw1 = tf.Variable(xavier_init(hidden_2, hidden_1))
    db1 = tf.Variable(tf.zeros([hidden_1, ]))
    dw2 = tf.Variable(xavier_init(hidden_1, n_input))
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
    num_batch = int(data.shape[0] / batch_size)
    if num_batch == 0:
        raise ValueError("Batch size is too large for the given data.")

    for epoch in range(epochs):
        for i in range(num_batch):
            batch_xs = data[i * batch_size:(i + 1) * batch_size]
            _, cost = sess.run([optm, loss], feed_dict={x: batch_xs})

    print('Training finished!!!')
    end_time = time()
    print(f"Training time: {end_time - begin_time:.2f} seconds")


    z_arr = sess.run(z, feed_dict={x: test})
    z_arr = z_arr.transpose()
    HI_arr = []
    for j in tuple(x for x in panels if x != panel):
        graph_data = pd.read_csv(dir_root + "\concatenated_" + freq + "_" + j + "_FFT_Features.csv",
                                 header=None).values.transpose()
        graph_data = np.delete(graph_data, -1, axis=1)
        graph_data = scaler.transform(graph_data)
        graph_data = pca.transform(graph_data)
        y_pred = sess.run(z, feed_dict={x: graph_data})
        HI_arr.append(y_pred)
    # scale all arrays to the same length
    for i in range(len(HI_arr)):
        HI_arr[i] = HI_arr[i].transpose()
    max = find_largest_array_size(HI_arr)
    for i in range(len(HI_arr)):
        if HI_arr[i].size < max:
            arr_interp = interp.interp1d(np.arange(HI_arr[i].size), HI_arr[i])
            arr_stretch = arr_interp(np.linspace(0, HI_arr[i].size - 1, max))
            HI_arr[i] = arr_stretch
    HI_arr = np.vstack(HI_arr)
    if z_arr.size != HI_arr.shape[1]:
        arr_interp = interp.interp1d(np.arange(z_arr.size), z_arr)
        arr_stretch = arr_interp(np.linspace(0, z_arr.size - 1, HI_arr.shape[1]))
        z_arr = arr_stretch
    full = np.append(HI_arr, z_arr, axis=0)
    sess.close()
    return [full, HI_arr, z_arr]

# Panels and freqs for iteration
panels = ("L103", "L105", "L109", "L104", "L123")
freqs = ("050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz")
resdict = {}
counter = 0
colors = ("b", "g", "y", "r", "m")

# Reading the hyperparameter file
hyperparameters_df = pd.read_csv(dir_root + '/hyperparameters-opt-FFT.csv', index_col=0)

time_steps = 30
num_HIs = 5  # Number of rows in f_all_array
num_freqs = len(freqs)
num_panels = len(panels)

# Initialize the final array with zeros
hi_full_array = np.zeros((num_panels, num_freqs, num_HIs, time_steps))

for panel_idx, panel in enumerate(panels):
    for freq_idx, freq in enumerate(freqs):
        filenames = []
        for i in tuple(x for x in panels if x != panel):
            filename = os.path.join(dir_root, f"concatenated_{freq}_{i}_FFT_Features.csv")
            filenames.append(filename)
        resdict[f"{panel}{freq}"] = []
        for j in range(1):
            counter += 1
            train_data, flags = mergedata(filenames)
            test_filename = os.path.join(dir_root, f"concatenated_{freq}_{panel}_FFT_Features.csv")
            test = pd.read_csv(test_filename, header=None).values.transpose()
            train_data.drop(train_data.columns[len(train_data.columns)-1], axis=1, inplace=True)
            test = np.delete(test, -1, axis=1)
            scaler = StandardScaler()
            scaler.fit(train_data)
            data = scaler.transform(data)
            test = scaler.transform(test)
            pca = PCA(n_components=30)
            pca.fit(train_data)
            train_data = pca.transform(train_data)
            test = pca.transform(test)
            # Set hyperparameters and architecture details
            hyperparameters_str = hyperparameters_df.loc[freq, panel]
            hyperparameters = eval(hyperparameters_str)

            health_indicators = train_vae(hyperparameters[0][0], hyperparameters[0][1],
                                          hyperparameters[0][2], hyperparameters[0][3])
            f_all_array = health_indicators[1]
            f_all_array = np.append(f_all_array, health_indicators[2], axis=0)
            hi_train = fitness(f_all_array)
            print("Fitness all", hi_train)
            hi_test = test_fitness(health_indicators[2], health_indicators[1])
            print("Fitness test", hi_test)
            resdict[f"{panel}{freq}"].append([hi_train, hi_test])
            print("Counter: ", counter)
            print("Panel: ", panel)
            print("Freq: ", freq)
            graph_hi_filename = f"HI_graph_{freq}_{panel}_seed_{seed}"
            graph_hi_dir = os.path.join(dir_root, graph_hi_filename)
            fig = plt.figure()
            train_panels = [k for k in panels if k != panel]
            x = np.arange(0, health_indicators[2].shape[1], 1)
            x = x*(1/(x.shape[0]-1))
            x = x * 100
            for i, hi in enumerate(health_indicators[1]):
                plt.plot(x, hi, label=f'Sample {panels.index(train_panels[i]) + 1}: Train')
            for i, hi in enumerate(health_indicators[2]):
                plt.plot(x, hi, label=f'Sample {panels.index(panel) + 1}: Test')
            plt.xlabel('Lifetime (%)')
            plt.ylabel('Health Indicators')
            plt.title('Train and Test Health Indicators over Time')
            plt.legend()
            plt.savefig(graph_hi_dir)
            plt.close(fig)
            params_test = (hi_test)
            params_hitrain = (hi_train)
            store_hyperparameters(params_test, params_hitrain, panel, freq)

            # Apply scale_exact to each row in f_all_array and reshape to the required 4D shape
            f_all_array2 = np.array([scale_exact(row) for row in f_all_array[:num_HIs]])
            f_all_array2 = f_all_array2.reshape(num_HIs, time_steps)

            # Assign f_all_array2 to the correct position in hi_full_array
            hi_full_array[panel_idx, freq_idx] = f_all_array2
label = f"VAE_seed_{seed}"
savedir = dir_root + '\\' + label
np.save(savedir, hi_full_array)
plot_images()
