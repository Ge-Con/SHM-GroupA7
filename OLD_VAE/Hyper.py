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

# Reset any previous graph and set seed for reproducibility
tf.compat.v1.reset_default_graph()
tf.random.set_seed(42)
#dir_root = input("Enter directory of folder with data: ")
# C:\Users\pablo\Downloads\PZT Output folder
def mergedata(filenames):
    flags = tuple([0])
    data = pd.read_csv(filenames[0], header=None)
    if len(filenames) > 1:
        for i in range(len(filenames)-1):
            flags += tuple([len(data)])
            data = pd.concat([data, pd.read_csv(filenames[i+1])], axis = 1)
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

def simple_store_hyperparameters(params, file, panel, freq, dir):

    filename_opt = os.path.join(dir, file + "-hopt.csv")
    #filename_test = os.path.join(dir_root, "hyperparameters-test.csv")
    #filename_train = os.path.join(dir_root, "hyperparameters-train.csv")
    freqs = ["050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz"]
    freq = freq + "_kHz"

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
    df.loc[freq, panel] = str(params)

    # Save the DataFrame back to the CSV
    df.to_csv(filename_opt)

def train_vae_ensemble(hidden_1, batch_size, learning_rate, epochs):
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
            #validation_loss = sess.run(loss, feed_dict={x: valid})

        #if epoch % display == 0:
            #print(f"Epoch {epoch}, Cost = {cost}, validation loss = N/a")

    print('Training finished!!!')
    end_time = time()
    print(f"Training time: {end_time - begin_time:.2f} seconds")
    return sess, x, z

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
            #validation_loss = sess.run(loss, feed_dict={x: valid})

        #if epoch % display == 0:
            #print(f"Epoch {epoch}, Cost = {cost}, validation loss = N/a")

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
    #scale all arrays to the same lenght
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
    full = np.append(HI_arr, z_arr, axis = 0)
    sess.close()
    return [full, HI_arr, z_arr]

# Bayesian optimization

def print_progress(res):
    n_calls = len(res.x_iters)
    print(f"Call number: {n_calls}")

space = [
        Integer(10, 100, name='hidden_1'),
        Integer(16, 128, name='batch_size'),
        Real(0.0001, 0.01, name='learning_rate'),
        Integer(500, 10000, name='epochs')
    ]
@use_named_args(space)
def objective(**params):
    print(params)
    ftn, monotonicity, trendability, prognosability, error = fitness(train_vae(**params)[1])
    print("Error: ", error)
    return error

def hyperparameter_optimisation(n_calls, random_state=42):
    res_gp = gp_minimize(objective, space, n_calls=n_calls, random_state=random_state,
                         callback=[print_progress])
    opt_parameters = [res_gp.x, res_gp.fun]
    print("Best parameters found: ", res_gp.x)
    return opt_parameters

# You can create additional datasets if needed
# Example: Using the first few columns as one dataset and the rest as another
#data1 = data[:, :1]  # First column as one dataset
#data2 = data[:, 1:2]  # Second column as another dataset
'''
panels = ("L103", "L105", "L109", "L104", "L123")
freqs = ("050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz")
resdict = {}
counter = 0
for panel in panels:
    for freq in freqs:
        filenames = []
        for i in tuple(x for x in panels if x != panel):
            filename = os.path.join(dir_root, f"concatenated_{freq}_{i}_HLB_Features.csv")
            filenames.append(filename)
        resdict[f"{panel}{freq}"] = []
        for j in range(1):
            counter += 1
            data, flags = mergedata(filenames)
            test_filename = os.path.join(dir_root, f"concatenated_{freq}_{panel}_HLB_Features.csv")
            test = pd.read_csv(test_filename, header=None).values.transpose()
            data.drop(data.columns[len(data.columns)-1], axis=1, inplace=True)
            test = np.delete(test, -1, axis=1)
            scaler = StandardScaler()
            scaler.fit(data)
            data = scaler.transform(data)
            test = scaler.transform(test)
            pca = PCA(n_components=30)
            pca.fit(data)
            data = pca.transform(data)
            test = pca.transform(test)
            # Set hyperparameters and architecture details
            hyperparameters = hyperparameter_optimisation(n_calls=20)

            health_indicators = train_vae(hyperparameters[0][0], hyperparameters[0][1],
                                          hyperparameters[0][2], hyperparameters[0][3])
            hi_train = fitness(health_indicators[0])
            print("HI train", hi_train)
            hi_test, m, t, p = test_fitness(health_indicators[2], health_indicators[1])
            print("HI test", hi_test)
            resdict[f"{panel}{freq}"].append([hi_train, hi_test])
            print("Counter: ", counter)
            params_test = (hi_test, m, t, p)
            params_hitrain = (hi_train)
            store_hyperparameters(hyperparameters, params_test, params_hitrain, panel, freq)
with open("results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, resdict.keys())
    w.writeheader()
    w.writerow(resdict)
'''

        # # Xavier initialization for weights
        # def xavier_init(fan_in, fan_out, constant=1):
        #     low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        #     high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        #     return tf.random.uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
        #
        # tf.compat.v1.disable_eager_execution()
        # # Input placeholder
        # x = tf.compat.v1.placeholder(tf.float32, [None, n_input])
        #
        # # Encoder weights and biases
        # w1 = tf.Variable(xavier_init(n_input, hidden_1))
        # b1 = tf.Variable(tf.zeros([hidden_1, ]))
        #
        # mean_w = tf.Variable(xavier_init(hidden_1, hidden_2))
        # mean_b = tf.Variable(tf.zeros([hidden_2, ]))
        #
        # logvar_w = tf.Variable(xavier_init(hidden_1, hidden_2))
        # logvar_b = tf.Variable(tf.zeros([hidden_2, ]))
        #
        # dw1 = tf.Variable(xavier_init(hidden_2, hidden_1))
        # db1 = tf.Variable(tf.zeros([hidden_1, ]))
        #
        # dw2 = tf.Variable(xavier_init(hidden_1, n_input))
        # db2 = tf.Variable(tf.zeros([n_input, ]))
        #
        # l1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
        # mean = tf.matmul(l1, mean_w) + mean_b
        # logvar = tf.matmul(l1, logvar_w) + logvar_b
        # eps = tf.random.normal(tf.shape(logvar), 0, 1, dtype=tf.float32)
        # z = tf.multiply(tf.sqrt(tf.exp(logvar)), eps) + mean
        # l2 = tf.nn.sigmoid(tf.matmul(z, dw1) + db1)
        # pred = tf.matmul(l2, dw2) + db2
        #
        # # Loss function with additional KL divergence and custom loss
        # reloss = tf.reduce_sum(tf.square(pred - x))
        # klloss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), 1)
        #
        #
        #
        #
        # # Total loss
        # fealoss = DCloss(z, batch_size)
        # loss = tf.reduce_mean(0.1 * reloss + 0.6 * klloss + 10 * fealoss)
        #
        # # Optimizer
        # optm = tf.compat.v1.train.AdamOptimizer(0.003).minimize(loss)
        #
        # # Training parameters
        # epochs = 1200
        # display = 50
        # begin_time = time()

# Training the VAE


#resdict = {{}}
        # with tf.compat.v1.Session() as sess:
        #     sess.run(tf.compat.v1.global_variables_initializer())
        #     print('Start training!!!')
        #     num_batch = int(data.shape[0] / batch_size)
        #     if num_batch == 0:
        #         raise ValueError("Batch size is too large for the given data.")
        #
        #     for epoch in range(epochs):
        #         for i in range(num_batch):
        #             batch_xs = data[i * batch_size:(i + 1) * batch_size]
        #             _, cost = sess.run([optm, loss], feed_dict={x: batch_xs})
        #             #validation_loss = sess.run([loss], feed_dict={x: valid})
        #
        #         if epoch % display == 0:
        #             print(f"Epoch {epoch}, Cost = {cost}, validation loss = N/a")
        #
        #     print('Training finished!!!')
        #     end_time = time()
        #     print(f"Training time: {end_time - begin_time:.2f} seconds")
        #     z_arr = sess.run(z, feed_dict={x: test})
        #     plt.figure()
        #     plt.plot(z_arr, 'c-', label='Feature 1')
        #     HI_arr = [z_arr]
        #     for j in tuple(x for x in panels if x != panel):
        #         graph_data = pd.read_csv(j + freq + ".csv", header=None).values.transpose()
        #         graph_data = np.delete(graph_data, -1, axis=1)
        #         graph_data = scaler.transform(graph_data)
        #         graph_data = pca.transform(graph_data)
        #         y_pred = sess.run(z, feed_dict={x: graph_data})
        #         HI_arr.append(y_pred)
        #         plt.plot(y_pred, 'g-', label=f'{j}')
        #     #scale all arrays to the same lenght
        #     for i in range(len(HI_arr)):
        #         HI_arr[i] = HI_arr[i].transpose()
        #     max = find_largest_array_size(HI_arr)
        #     for i in range(len(HI_arr)):
        #         if HI_arr[i].size < max:
        #             arr_interp = interp.interp1d(np.arange(HI_arr[i].size), HI_arr[i])
        #             arr_stretch = arr_interp(np.linspace(0, HI_arr[i].size - 1, max))
        #             HI_arr[i] = arr_stretch
        #     HI_arr = np.vstack(HI_arr)
        #
        #     print(fitness(HI_arr))
        #     # plt.title("Health Index")
        #     # plt.xlabel("# state")
        #     # plt.ylabel("Health Index")
        #     # plt.show()
        #     #resdict[panel][freq] = prognostic_eval(z_arr)
