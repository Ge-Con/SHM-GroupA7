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
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import csv
from prognosticcriteria_v2 import Mo_single, Pr, Tr, Mo, fitness, Pr_single
import os

panels = ("L103", "L105", "L109", "L104", "L123")

def mergedata(train_filenames):
    flags = tuple([0])
    data = pd.read_csv(train_filenames[0], header=None)
    if len(train_filenames) > 1:
        for i in range(len(train_filenames)-1):
            flags += tuple([len(data)])
            data = pd.concat([data, pd.read_csv(train_filenames[i+1])], axis = 1)
    data = data.transpose()
    return data, flags

def DCloss(feature, batch_size):
    s = 0
    for i in range(1, batch_size):
        s += tf.pow(feature[i] - tf.constant(10, dtype=tf.float32) - tf.random.normal([1], 0, 1) - feature[i - 1], 2)
    return s


def find_largest_array_size(array_list):
    max_size = 0

    for arr in array_list:
        if isinstance(arr, np.ndarray):
            size = arr.size
            if size > max_size:
                max_size = size

    return max_size

def simple_store_hyperparameters(hyperparameters, file, panel, freq, dir):

    filename_opt = os.path.join(dir, f"hyperparameters-opt-{file}.csv")
    freqs = ["050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz"]
    if not freq.endswith("_kHz"):
        freq = freq + "_kHz"

    # Create an empty dataframe with frequencies as the index if the file does not exist
    if not os.path.exists(filename_opt):
        df = pd.DataFrame(index=freqs)
    else:
        # Load the existing file if it exists
        df = pd.read_csv(filename_opt, index_col=0)

    # Ensure that the panel column exists
    if panel not in df.columns:
        df[panel] = None

    # Update the dataframe with the new parameters
    df.loc[freq, panel] = str(hyperparameters)

    # Save the dataframe back to the CSV
    df.to_csv(filename_opt)

def train_vae(hidden_1, batch_size, learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff, vae_train_data, vae_test_data, vae_scaler, vae_pca, vae_seed, file_type, panel, freq, csv_dir, valid = False, valid_data=None):

    n_input = vae_train_data.shape[1]  # Number of features
    hidden_2 = 1
    display = 50
    tf.random.set_seed(vae_seed)
    patience = 50    #early stopping criteria
    best_validation_loss = float('inf')  # Initialize with infinity

    # Xavier initialization for weights, as explained in Hyper_Qin_original.py
    def xavier_init(fan_in, fan_out, vae_seed, constant=1):
        tf.random.set_seed(vae_seed)
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random.uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

    # I don't remember what this line is, but we need it for some compatibility issues
    tf.compat.v1.disable_eager_execution()

    # Architecture and losses below as explained in Hyper_Qin_original.py
    x = tf.compat.v1.placeholder(tf.float32, [None, n_input])

    w1 = tf.Variable(xavier_init(n_input, hidden_1, vae_seed))
    b1 = tf.Variable(tf.zeros([hidden_1, ]))

    mean_w = tf.Variable(xavier_init(hidden_1, hidden_2, vae_seed))
    mean_b = tf.Variable(tf.zeros([hidden_2, ]))

    logvar_w = tf.Variable(xavier_init(hidden_1, hidden_2, vae_seed))
    logvar_b = tf.Variable(tf.zeros([hidden_2, ]))

    dw1 = tf.Variable(xavier_init(hidden_2, hidden_1, vae_seed))
    db1 = tf.Variable(tf.zeros([hidden_1, ]))

    dw2 = tf.Variable(xavier_init(hidden_1, n_input, vae_seed))
    db2 = tf.Variable(tf.zeros([n_input, ]))

    l1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
    mean = tf.matmul(l1, mean_w) + mean_b
    logvar = tf.matmul(l1, logvar_w) + logvar_b
    eps = tf.random.normal(tf.shape(logvar), 0, 1, dtype=tf.float32)

    std_dev = tf.sqrt(tf.exp(logvar))
    z = tf.multiply(std_dev, eps) + mean
    l2 = tf.nn.sigmoid(tf.matmul(z, dw1) + db1)
    pred = tf.matmul(l2, dw2) + db2

    reloss = tf.reduce_sum(tf.square(pred - x))
    klloss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), 1)
    fealoss = DCloss(z, batch_size)

    loss = tf.reduce_mean(reloss_coeff * reloss + klloss_coeff * klloss + moloss_coeff * fealoss)

    optm = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)

    # Training, as explained in Hyper_Qin_original.py
    begin_time = time()

    # In sess, we are saving everything from the training loop, the learnt weights and biases, etc
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    print('Start training!!!')
    num_batch = int(vae_train_data.shape[0] / batch_size)
    if num_batch == 0:
        raise ValueError("Batch size is too large for the given data.")

    for epoch in range(epochs):
        for i in range(num_batch):
            batch_xs = vae_train_data[i * batch_size:(i + 1) * batch_size]
            _, cost = sess.run([optm, loss], feed_dict={x: batch_xs})

            if valid:
                validation_loss = sess.run(loss, feed_dict={x: valid_data})

                # Early stopping condition
                if validation_loss < best_validation_loss * 0.995 and validation_loss > best_validation_loss:
                    best_validation_loss = validation_loss
                    patience_counter = 0  # Reset patience counter if we have an improvement
                else:
                    patience_counter += 1  # Increment if no improvement

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch} with validation loss: {validation_loss}")
                    break

        if epoch % display == 0:
            if valid:
                print(f"Epoch {epoch}, Cost = {cost}, validation loss = {validation_loss}")
            else:
                print(f"Epoch {epoch}, Cost = {cost}")

    print('Training finished!!!')
    end_time = time()
    print(f"Training time: {end_time - begin_time:.2f} seconds")

    std_dev_test, z_test = sess.run([std_dev, z], feed_dict={x: vae_test_data})
    z_test = z_test.transpose()
    std_dev_test = std_dev_test.transpose()

    # In this variable the train HIs will be saved
    z_train = []
    std_dev_train = []
    # This loops x over all panels EXCEPT the current test panel. j represents panel name
    for j in tuple(x for x in panels if x != panel):

        single_panel_data = pd.read_csv(csv_dir + "\concatenated_" + freq + "_" + j + "_" + file_type + ".csv", header=None).values.transpose()
        single_panel_data = np.delete(single_panel_data, -1, axis=1)

        single_panel_data = vae_scaler.transform(single_panel_data)
        single_panel_data = vae_pca.transform(single_panel_data)

        std_dev_train_individual, z_train_individual = sess.run([std_dev, z], feed_dict={x: single_panel_data})
        z_train.append(z_train_individual)
        std_dev_train.append(std_dev_train_individual)


    # Transpose
    for i in range(len(z_train)):
        z_train[i] = z_train[i].transpose()
        std_dev_train[i] = std_dev_train[i].transpose()

    # We find the longest HI and interpolate the other HIs so that they all have matching lengths
    max_size = find_largest_array_size(z_train)
    for i in range(len(z_train)):
        if z_train[i].size < max_size:

            # The interpolation function is found for the shorter HI, and applied to arr_stretch. Then the shorter HI z_train[i] is updated
            interp_function = interp.interp1d(np.arange(z_train[i].size), z_train[i])
            arr_stretch = interp_function(np.linspace(0, z_train[i].size - 1, max_size))
            z_train[i] = arr_stretch

            interp_function_std = interp.interp1d(np.arange(std_dev_train[i].size), std_dev_train[i])
            arr_stretch_std = interp_function_std(np.linspace(0, std_dev_train[i].size - 1, max_size))
            std_dev_train[i] = arr_stretch_std

    # Vertically stack the HIs
    z_train = np.vstack(z_train)
    std_dev_train = np.vstack(std_dev_train)

    # We do the same but to match the length of z_test with z_train
    if z_test.size != z_train.shape[1]:
        interp_function = interp.interp1d(np.arange(z_test.size), z_test)
        arr_stretch = interp_function(np.linspace(0, z_test.size - 1, z_train.shape[1]))
        z_test = arr_stretch

        interp_function_std = interp.interp1d(np.arange(std_dev_test.size), std_dev_test)
        arr_stretch_std = interp_function_std(np.linspace(0, std_dev_test.size - 1, z_train.shape[1]))
        std_dev_test = arr_stretch_std

    if valid:
        std_dev_valid, z_valid = sess.run([std_dev, z], feed_dict={x: valid_data})
        z_valid = z_valid.transpose()
        std_dev_valid = std_dev_valid.transpose()

        if z_valid.size != z_train.shape[1]:
            interp_function = interp.interp1d(np.arange(z_valid.size), z_valid)
            arr_stretch = interp_function(np.linspace(0, z_valid.size - 1, z_train.shape[1]))
            z_valid = arr_stretch

            interp_function_std = interp.interp1d(np.arange(std_dev_valid.size), std_dev_valid)
            arr_stretch_std = interp_function_std(np.linspace(0, std_dev_valid.size - 1, z_train.shape[1]))
            std_dev_valid = arr_stretch_std

    # Create an array that contains all HIs (train+test)
    z_all = np.append(z_train, z_test, axis = 0)
    std_dev_all = np.append(std_dev_train, std_dev_test, axis = 0)

    if valid:
        z_all = np.append(z_all, z_valid, axis=0)
        std_dev_all = np.append(std_dev_all, std_dev_valid, axis=0)

    # Close the TensorFlow session
    sess.close()

    if valid:
        return [z_all, z_train, z_test, std_dev_all, std_dev_train, std_dev_test, z_valid, std_dev_valid]

    else:
        return [z_all, z_train, z_test, std_dev_all, std_dev_train, std_dev_test]

# Hyperparameter optimization

# This function is just to print what call number you are on
def print_progress(res):
    n_calls = len(res.x_iters)
    print(f"Call number: {n_calls}")

space = [
    Integer(10, 100, name='hidden_1'),
    Integer(16, 128, name='batch_size'),
    Real(0.0001, 0.01, name='learning_rate'),
    Integer(500, 10000, name='epochs'),
    Real(0.05, 20, name='reloss_coeff'),
    Real(0.05, 20, name='klloss_coeff'),
    Real(0.05, 20, name='moloss_coeff')
]

# Use the decorator to automatically convert parameters to keyword arguments
@use_named_args(space)
def objective(hidden_1, batch_size, learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff):
    print(
        f"Trying parameters: hidden_1={hidden_1}, batch_size={batch_size}, learning_rate={learning_rate}, "
        f"epochs={epochs}, reloss_coeff={reloss_coeff}, klloss_coeff={klloss_coeff}, moloss_coeff={moloss_coeff}")

    health_indicators = train_vae(hidden_1, batch_size,
                                  learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff, vae_train_data, vae_test_data, vae_scaler, vae_pca, vae_seed, file_type, panel, freq, csv_dir)
    ftn, monotonicity, trendability, prognosability, error = fitness(health_indicators[1])
    print("Error: ", error)
    return error


def hyperparameter_optimisation(vae_train_data, vae_test_data, vae_scaler, vae_pca, vae_seed, file_type, panel, freq, csv_dir, n_calls,
                                random_state=42):
    # Define the space for hyperparameters
    space = [
        Integer(10, 100, name='hidden_1'),
        Integer(16, 128, name='batch_size'),
        Real(0.0001, 0.01, name='learning_rate'),
        Integer(500, 10000, name='epochs'),
        Real(0.05, 20, name='reloss_coeff'),
        Real(0.05, 20, name='klloss_coeff'),
        Real(0.05, 20, name='moloss_coeff')
    ]

    # Define the objective function with named args
    @use_named_args(space)
    def objective(hidden_1, batch_size, learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff):
        print(
            f"Trying parameters: hidden_1={hidden_1}, batch_size={batch_size}, learning_rate={learning_rate}, "
            f"epochs={epochs}, reloss_coeff={reloss_coeff}, klloss_coeff={klloss_coeff}, moloss_coeff={moloss_coeff}")
        health_indicators = train_vae(hidden_1, batch_size,
                                      learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff, vae_train_data, vae_test_data, vae_scaler, vae_pca,
                                      vae_seed, file_type, panel, freq, csv_dir)
        ftn, monotonicity, trendability, prognosability, error = fitness(health_indicators[1])
        print("Error: ", error)
        return error

    # Run the optimization
    res_gp = gp_minimize(objective, space, n_calls=n_calls, random_state=random_state, callback=[print_progress])
    opt_parameters = [res_gp.x, res_gp.fun]
    print("Best parameters found: ", res_gp.x)
    print("Error of best parameters: ", res_gp.fun)

    return opt_parameters