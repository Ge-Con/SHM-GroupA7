import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import Normalizer
from scipy.signal import resample_poly
import math
import scipy.interpolate as interp
import csv
from prognosticcriteria_v2 import Mo_single, Pr, Tr, Mo, fitness, Pr_single
import os

# Reset any previous graph and set seed for reproducibility
tf.compat.v1.reset_default_graph()
tf.random.set_seed(42)
dir_root = input("Enter directory of folder with data: ")
# C:\Users\pablo\Downloads\PZT Output folder

def mergedata(filenames):
    flags = tuple([0])
    data = pd.read_csv(filenames[0], header=None)
    if len(filenames) > 1:
        for i in range(len(filenames)-1):
            flags += tuple([len(data)])
            data = pd.concat([data, pd.read_csv(filenames[i+1])], axis=1)
    data = data.transpose()
    return data, flags

panels = ("L103", "L105", "L109", "L104", "L123")
freqs = ("050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz")
resdict = {}

def DCloss(feature, batch_size):
    s = 0
    for i in range(1, batch_size):
        s += tf.pow(feature[i] - tf.constant(10, dtype=tf.float32) - tf.random.normal([1], 0, 1) - feature[i - 1], 2)
    return s

def test_fitness(test_HI, X):
    test_HI = test_HI[0]
    monotonicity = Mo_single(test_HI)
    trendability = Tr(np.vstack([test_HI, X]))
    prognosability = Pr_single(test_HI, X)
    fitness_test = (monotonicity + trendability + prognosability), monotonicity, trendability , prognosability

    return fitness_test

def find_largest_array_size(array_list):
    max_size = 0

    for arr in array_list:
        if isinstance(arr, np.ndarray):
            size = arr.size
            if size > max_size:
                max_size = size

    return max_size

def store_hyperparameters(params_train, params_test, params_hi_train, panel, freq):
    global dir_root

    filename_opt = os.path.join(dir_root, "hyperparameters-opt.csv")
    filename_test = os.path.join(dir_root, "hyperparameters-test.csv")
    filename_train = os.path.join(dir_root, "hyperparameters-train.csv")
    freqs = ["050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz"]

    # Create an empty DataFrame with frequencies as the index if the file does not exist
    if not os.path.exists(filename_opt):
        df = pd.DataFrame(index=freqs)
    else:
        # Load the existing file if it exists
        df = pd.read_csv(filename_opt, index_col=0)

    # Ensure that the panel column exists
    if panel not in df.columns:
        df[panel] = None

    # Update the DataFrame with the new parameters
    df.loc[freq, panel] = str(params_train)

    # Save the DataFrame back to the CSV
    df.to_csv(filename_opt)

    # Create an empty DataFrame with frequencies as the index if the file does not exist
    if not os.path.exists(filename_test):
        df = pd.DataFrame(index=freqs)
    else:
        # Load the existing file if it exists
        df = pd.read_csv(filename_test, index_col=0)

    # Ensure that the panel column exists
    if panel not in df.columns:
        df[panel] = None

    # Update the DataFrame with the new parameters
    df.loc[freq, panel] = str(params_test)

    # Save the DataFrame back to the CSV
    df.to_csv(filename_test)

    if not os.path.exists(filename_train):
        df = pd.DataFrame(index=freqs)
    else:
        # Load the existing file if it exists
        df = pd.read_csv(filename_train, index_col=0)

    # Ensure that the panel column exists
    if panel not in df.columns:
        df[panel] = None

    # Update the DataFrame with the new parameters
    df.loc[freq, panel] = str(params_hi_train)

    # Save the DataFrame back to the CSV
    df.to_csv(filename_train)

def train_vae(hidden_1, batch_size, learning_rate, epochs):
    # Set hyperparameters and architecture details
    global data
    global test
    global valid
    global scaler
    global pca
    n_input = data.shape[1]  # Number of features
    hidden_2 = 1
    display = 50

    # Xavier initialization for weights
    def xavier_init(fan_in, fan_out, constant=1):
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random.uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

    tf.compat.v1.disable_eager_execution()
    # Input placeholder
    x = tf.compat.v1.placeholder(tf.float32, [None, n_input])

    # Encoder weights and biases
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

    # Loss function with additional KL divergence and custom loss
    reloss = tf.reduce_sum(tf.square(pred - x))
    klloss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), 1)

    # Total loss
    fealoss = DCloss(z, batch_size)
    loss = tf.reduce_mean(0.1 * reloss + 0.6 * klloss + 10 * fealoss)

    # Optimizer
    optm = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)

    # Training parameters
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
        graph_data = pd.read_csv(dir_root + "\concatenated_" + freq + "_" + j + "_HLB_Features.csv", header=None).values.transpose()
        graph_data = np.delete(graph_data, -1, axis=1)
        graph_data = scaler.transform(graph_data)
        graph_data = pca.transform(graph_data)
        y_pred = sess.run(z, feed_dict={x: graph_data})
        HI_arr.append(y_pred)
    #scale all arrays to the same length
    for i in range(len(HI_arr)):
        HI_arr[i] = resample_poly(HI_arr[i], find_largest_array_size(HI_arr), len(HI_arr[i]))

    scaler.fit(np.vstack([HI_arr, z_arr]))
    z_arr = scaler.transform(z_arr)
    HI_arr = scaler.transform(np.vstack(HI_arr))
    fitness_train = fitness(HI_arr, data)
    fitness_test = test_fitness(z_arr, test)
    print("Results:")
    print(f"Fitness train: {fitness_train}")
    print(f"Fitness test: {fitness_test}")
    return (hidden_1, batch_size, learning_rate, epochs), fitness_test[0]

def tune_vae(panel, freq):
    filename = f"{dir_root}/concatenated_{freq}_{panel}_HLB_Features.csv"
    global data
    global valid
    global test
    global scaler
    global pca

    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return

    data = pd.read_csv(filename, header=None).values
    data = np.delete(data, -1, axis=1)

    # Normalize data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Reduce dimensions using PCA
    pca = PCA(n_components=0.95)
    data = pca.fit_transform(data)

    valid = data[int(data.shape[0] * 0.7):int(data.shape[0] * 0.85)]
    test = data[int(data.shape[0] * 0.85):]

    params_list = []

    # Hyperparameter tuning
    best_fitness = float('-inf')
    best_params = None

    params_train = {
        'hidden_1': [16, 32, 64],
        'batch_size': [16, 32, 64],
        'learning_rate': [0.001, 0.01, 0.1],
        'epochs': [50, 100, 200]
    }

    # Define ranges for hyperparameters
    for hidden_1 in params_train['hidden_1']:
        for batch_size in params_train['batch_size']:
            for learning_rate in params_train['learning_rate']:
                for epochs in params_train['epochs']:
                    try:
                        params, fitness_value = train_vae(hidden_1, batch_size, learning_rate, epochs)
                        if fitness_value > best_fitness:
                            best_fitness = fitness_value
                            best_params = params
                    except Exception as e:
                        print(f"Error during training with parameters {hidden_1}, {batch_size}, {learning_rate}, {epochs}: {e}")
                        continue

    print(f"Best parameters for {panel} at {freq}: {best_params} with fitness {best_fitness}")

    params_list.append(best_params)
    store_hyperparameters(best_params, params_list, [], panel, freq)

def main():
    hyperparameters_df = pd.read_csv('/mnt/data/hyperparameters-opt-FFT.csv')
    for panel in panels:
        for freq in freqs:
            if not hyperparameters_df[(hyperparameters_df['Panel'] == panel) & (hyperparameters_df['Frequency'] == freq)].empty:
                print(f"Processing Panel: {panel}, Frequency: {freq}")
                params = hyperparameters_df[(hyperparameters_df['Panel'] == panel) & (hyperparameters_df['Frequency'] == freq)]
                hidden_1 = params['hidden_1'].values[0]
                batch_size = params['batch_size'].values[0]
                learning_rate = params['learning_rate'].values[0]
                epochs = params['epochs'].values[0]

                try:
                    train_vae(hidden_1, batch_size, learning_rate, epochs)
                except Exception as e:
                    print(f"Error during training with parameters {hidden_1}, {batch_size}, {learning_rate}, {epochs}: {e}")
                    continue

if __name__ == "__main__":
    main()
