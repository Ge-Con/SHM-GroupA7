import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math
from sklearn.preprocessing import Normalizer
from scipy.stats import pearsonr
from scipy.signal import resample_poly
from time import time

import tensorflow as tf

# Disable eager execution
tf.compat.v1.disable_eager_execution()


# Load data from CSV and preprocess
csv_path = 'resultalltimesteps.csv'
data = pd.read_csv(csv_path, header=None).values.transpose()
scaler = StandardScaler()
data = scaler.fit_transform(data)
pca = PCA(n_components = 32)
data_pca = pca.fit_transform(data)

# TensorFlow model
n_input = data_pca.shape[1]
hidden_1 = 10
hidden_2 = 1
batch_size = 2

tf.compat.v1.reset_default_graph()
tf.random.set_seed(19)

x = tf.compat.v1.placeholder(tf.float32, [None, n_input])

w1 = tf.Variable(tf.random.uniform([n_input, hidden_1], -1.0 / np.sqrt(n_input), 1.0 / np.sqrt(n_input)))
b1 = tf.Variable(tf.zeros([hidden_1]))

mean_w = tf.Variable(tf.random.uniform([hidden_1, hidden_2], -1.0 / np.sqrt(hidden_1), 1.0 / np.sqrt(hidden_1)))
mean_b = tf.Variable(tf.zeros([hidden_2]))

logvar_w = tf.Variable(tf.random.uniform([hidden_1, hidden_2], -1.0 / np.sqrt(hidden_1), 1.0 / np.sqrt(hidden_1)))
logvar_b = tf.Variable(tf.zeros([hidden_2]))

dw1 = tf.Variable(tf.random.uniform([hidden_2, hidden_1], -1.0 / np.sqrt(hidden_2), 1.0 / np.sqrt(hidden_2)))
db1 = tf.Variable(tf.zeros([hidden_1]))

dw2 = tf.Variable(tf.random.uniform([hidden_1, n_input], -1.0 / np.sqrt(hidden_1), 1.0 / np.sqrt(hidden_1)))
db2 = tf.Variable(tf.zeros([n_input]))

l1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
mean = tf.matmul(l1, mean_w) + mean_b
logvar = tf.matmul(l1, logvar_w) + logvar_b
eps = tf.random.normal(tf.shape(logvar), 0, 1, dtype=tf.float32)
z = tf.nn.sigmoid(tf.multiply(tf.sqrt(tf.exp(logvar)), eps) + mean)

l2 = tf.nn.sigmoid(tf.matmul(z, dw1) + db1)
pred = tf.matmul(l2, dw2) + db2

reloss = tf.reduce_sum(tf.square(pred - x))
klloss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), 1)

def DCloss(feature, batch_size):
    s = 0
    for i in range(1, batch_size):
        s += tf.pow(feature[i] - tf.constant(0.5, dtype=tf.float32) - feature[i - 1],2)
    return s

fealoss = DCloss(z, batch_size)
loss = tf.reduce_mean(0.1 * reloss + 0.6 * klloss + 11 * fealoss) # Originial coefficients: 0.1 ; 0.6 ; 11
optm = tf.compat.v1.train.AdamOptimizer(0.01).minimize(loss)

# Training parameters
epochs = 5000
display = 50
begin_time = time()


# Define fitness function
def fitness(X, Mo_a=1, Tr_b=1, Pr_c=1):
    def Pr(X):
        M = len(X)
        Nfeatures = X.shape[1]
        top = np.zeros((M, Nfeatures))
        bottom = np.zeros((M, Nfeatures))

        for j in range(M):
            top[j, :] = X[j, -1]
            bottom[j, :] = np.abs(X[j, 0] - X[j, -1])

        mean_bottom = np.mean(bottom)
        if mean_bottom == 0:
            prognosability = 0  # Handle division by zero gracefully
        else:
            prognosability = np.exp(-np.std(top) / mean_bottom)

        return prognosability

    def Tr(X):
        m, n = X.shape
        trendability_feature = np.inf
        trendability_list = []

        scaler = Normalizer()
        X = scaler.fit_transform(X)

        for j in range(m):
            vector1 = X[j]

            for k in range(m):
                vector2 = X[k]

                if len(vector1) >= 2 and len(vector2) >= 2:  # Check for minimum length
                    if len(vector2) != len(vector1):
                        if len(vector2) < len(vector1):
                            vector2 = resample_poly(vector2, len(vector1), len(vector2), window=('kaiser', 5))
                        else:
                            vector1 = resample_poly(vector1, len(vector2), len(vector1), window=('kaiser', 5))

                    rho = pearsonr(vector1, vector2)[0]
                    if math.fabs(rho) < trendability_feature:
                        trendability_feature = math.fabs(rho)

                    trendability_list.append(trendability_feature)

        if trendability_list:
            return min(trendability_list)
        else:
            return 0  # Return 0 if no valid correlation coefficients are calculated

    def Mo(PC) -> float:
        sum_samples = 0
        for i in range(len(PC)):
            sum_measurements = 0
            div_sum = 0
            for k in range(len(PC)):
                sub_sum = 0
                div_sub_sum = 0
                if k > i:
                    sub_sum += (k - i) * np.sign(PC[k] - PC[i])
                    div_sub_sum += k - i
                sum_measurements += sub_sum
                div_sum += div_sub_sum
            if div_sum == 0:
                sum_samples += 0
            else:
                sum_samples += abs(sum_measurements / div_sum)
        return sum_samples / (len(PC) - 1)

    monotonicity = Mo(X)
    print("Mo is", monotonicity)
    trendability = Tr(X)
    print("Tr is", trendability)
    prognosability = Pr(X)
    print("Pr is", prognosability)

    ftn = Mo_a * monotonicity + Tr_b * trendability + Pr_c * prognosability
    error = Mo_a + Tr_b + Pr_c - ftn

    return ftn, error

# Training the VAE
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print('Start training!!!')
    num_batch = int(data_pca.shape[0] / batch_size)
    if num_batch == 0:
        raise ValueError("Batch size is too large for the given data.")

    for epoch in range(epochs):
        for i in range(num_batch):
            batch_xs = data_pca[i * batch_size:(i + 1) * batch_size]
            _, cost = sess.run([optm, loss], feed_dict={x: batch_xs})

        if epoch % display == 0:
            print(f"Epoch {epoch}, Cost = {cost}")
            #print(sess.run(z, feed_dict={x: data_pca}))

    print('Training finished!!!')
    end_time = time()
    print(f"Training time: {end_time - begin_time:.2f} seconds")

    # Visualizing health index using latent representation (z)
    plt.figure()
    fea1 = sess.run(z, feed_dict={x: data_pca})
    #print(fea1)
    plt.plot(fea1, 'c-', label='Feature 1')
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 23}
    plt.legend(loc = 'upper left', prop = font1)
    plt.title("Health Index")
    plt.xlabel("Serial Number")
    plt.ylabel("Health Index")

    # Calculate fitness
    fitness_values, _ = fitness(fea1)
    #print(fitness_values)

    plt.show()