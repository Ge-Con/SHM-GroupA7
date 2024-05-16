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

# Reset any previous graph and set seed for reproducibility
tf.compat.v1.reset_default_graph()
tf.random.set_seed(42)

def mergedata(filenames):
    flags = tuple([0])
    data = pd.read_csv(filenames[0], header=None)
    if len(filenames) > 1:
        for i in range(len(filenames)-1):
            flags += tuple([len(data)])
            data = pd.concat([data, pd.read_csv(filenames[i+1])], axis = 1)
    data = data.transpose()
    return data, flags

def DCloss(feature, batch_size):
    s = 0
    for i in range(1, batch_size):
        s += tf.pow(feature[i] - tf.constant(10, dtype=tf.float32) - tf.random.normal([1], 0, 1) - feature[i - 1], 2)
    return s

def Pr(X):
    M = len(X)
    Nfeatures = X.shape[1]
    top = np.zeros((M, Nfeatures))
    bottom = np.zeros((M, Nfeatures))

    for j in range(M):
        top[j, :] = X[j, -1]
        bottom[j, :] = np.abs(X[j, 0] - X[j, -1])

    prognosability = np.exp(-np.std(top) / np.mean(bottom))

    return prognosability


def Tr(X):
    """
    Shape of 'X':  (m rows x n columns) where X is samples vs measurements array for a specific PC
    where m is the number of samples
    and n is the number of measurements (ie. cycles)
    """
    m, n = X.shape  # m = rows (ie. # of samples), n = columns (ie. # of cycles)
    trendability_feature = np.inf
    trendability_list = []

    # Finding normalised matrix
    scaler = Normalizer()
    X = scaler.fit_transform(X)

    for j in range(m):
        # Obtain (pre-normalised) first vector of measurements
        vector1 = X[j]

        for k in range(m):
            # Obtain (pre-normalised) second vector of measurements
            vector2 = X[k]

            # Check whether two vectors are the same length, ie. if they both experience failure at same cycle
            # If vectors are not same length, reshape/resample them to equal lengths
            if len(vector2) != len(vector1):
                if len(vector2) < len(vector1):
                    vector2 = resample_poly(vector2, len(vector1), len(vector2), window=('kaiser', 5))
                else:
                    vector1 = resample_poly(vector1, len(vector2), len(vector1), window=('kaiser', 5))

            rho = pearsonr(vector1, vector2)[0]
            if math.fabs(rho) < trendability_feature:  # ie. if less than infinity, give new value
                trendability_feature = math.fabs(rho)

            # Add math.fabs(rho) to list
            trendability_list.append(trendability_feature)

    # Return minimum value
    return min(trendability_list)


def Mo_single(PC_single) -> float:  # single column
    sum_samples = 0
    for i in range(len(PC_single)):
        sum_measurements = 0
        div_sum = 0
        for k in range(len(PC_single)):
            sub_sum = 0
            div_sub_sum = 0
            if k > i:
                sub_sum += (k - i) * np.sign(PC_single[k] - PC_single[i])
                div_sub_sum += k - i
            sum_measurements += sub_sum
            div_sum += div_sub_sum
        if div_sum == 0:
            print("for", i, "div_sum is zero")
            sum_samples += 0
        else:
            sum_samples += abs(sum_measurements / div_sum)
    return sum_samples / (len(PC_single) - 1)

def Pr_single(test_HIs, HIs):
    x_t = test_HIs[-1]
    deviation_basis = 0
    for i in range(HIs.shape[0]):
        deviation_basis += HIs[i, -1]
    deviation_basis = abs(deviation_basis/HIs.shape[0])
    scaling_factor = 0
    for i in range(HIs.shape[0]):
        scaling_factor += abs(HIs[i, 0] - HIs[i, -1])
    scaling_factor += abs(test_HIs[0] - test_HIs[-1])
    scaling_factor = scaling_factor/(HIs.shape[0]+1)

    prognosability = np.exp(-abs((x_t-deviation_basis)) / scaling_factor)

    return prognosability


def Mo(PC):
    sum_monotonicities = 0
    for i in range(len(PC)):
        monotonicity_i = Mo_single(PC[i, :])
        sum_monotonicities += monotonicity_i
    return sum_monotonicities / np.shape(PC)[0]


def fitness(X, Mo_a=1, Tr_b=1, Pr_c=1):
    monotonicity = Mo(X)
    trendability = Tr(X)
    prognosability = Pr(X)

    ftn = Mo_a * monotonicity + Tr_b * trendability + Pr_c * prognosability

    error = (Mo_a + Tr_b + Pr_c) / ftn
    return ftn, error

def test_fitness(test_HI, X):
    test_HI = test_HI[0]
    monotonicity = Mo_single(test_HI)
    trendability = Tr(np.vstack([test_HI, X]))
    prognosability = Pr_single(test_HI, X)
    fitness_test = monotonicity + trendability + prognosability

    return fitness_test

def find_largest_array_size(array_list):
    max_size = 0

    for arr in array_list:
        if isinstance(arr, np.ndarray):
            size = arr.size
            if size > max_size:
                max_size = size

    return max_size


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

    with tf.compat.v1.Session() as sess:
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
            graph_data = pd.read_csv(j + freq + ".csv", header=None).values.transpose()
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
        return full, HI_arr, z_arr

# Bayesian optimization



# You can create additional datasets if needed
# Example: Using the first few columns as one dataset and the rest as another
#data1 = data[:, :1]  # First column as one dataset
#data2 = data[:, 1:2]  # Second column as another dataset

panels = ("L03", "L05", "L09", "L04", "L23")
freqs = ("050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz")
resdict = {}
counter = 0
for panel in panels:
    for freq in freqs:
        filenames = []
        for i in tuple(x for x in panels if x != panel):
            filenames.append(i + freq + ".csv")
        resdict[f"{panel}{freq}"] = []
        for j in range(2):
            counter += 1
            data, flags = mergedata(filenames)
            test = pd.read_csv(panel + freq + ".csv", header=None).values.transpose()
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
            health_indicators = train_vae(20, 32, 0.03, 1200)
            hi_train = fitness(health_indicators[0])
            print(hi_train)
            hi_test = test_fitness(health_indicators[2], health_indicators[1])
            resdict[f"{panel}{freq}"].append([hi_train, hi_test])
            print(counter)
with open("results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, resdict.keys())
    w.writeheader()
    w.writerow(resdict)


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