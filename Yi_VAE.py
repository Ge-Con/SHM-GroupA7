import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time

# Reset any previous graph and set seed for reproducibility
tf.compat.v1.reset_default_graph()
tf.random.set_seed(42)

# Load data from CSV and transpose to have timesteps as rows and features as columns
csv_path = 'resultalltimesteps.csv'  # Change to your CSV file path
data = pd.read_csv(csv_path, header=None).values  # Read and transpose the data

# You can create additional datasets if needed
# Example: Using the first few columns as one dataset and the rest as another
data1 = data[:, :1]  # First column as one dataset
data2 = data[:, 1:2]  # Second column as another dataset

# Set hyperparameters and architecture details
n_input = data.shape[1]  # Number of features
hidden_1 = 10  # Number of neurons in the hidden layer
hidden_2 = 1  # Number of neurons in the bottleneck layer
batch_size = 100  # Batch size


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
b1 = tf.Variable(tf.zeros([hidden_1]))

mean_w = tf.Variable(xavier_init(hidden_1, hidden_2))
mean_b = tf.Variable(tf.zeros([hidden_2]))

logvar_w = tf.Variable(xavier_init(hidden_1, hidden_2))
logvar_b = tf.Variable(tf.zeros([hidden_2]))

# Decoder weights and biases
dw1 = tf.Variable(xavier_init(hidden_2, hidden_1))
db1 = tf.Variable(tf.zeros([hidden_1]))

dw2 = tf.Variable(xavier_init(hidden_1, n_input))
db2 = tf.Variable(tf.zeros([n_input]))

# Encoder and bottleneck
l1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
mean = tf.matmul(l1, mean_w) + mean_b
logvar = tf.matmul(l1, logvar_w) + logvar_b
eps = tf.random.normal(tf.shape(logvar), 0, 1, dtype=tf.float32)
z = tf.multiply(tf.sqrt(tf.exp(logvar)), eps) + mean

# Decoder and prediction
l2 = tf.nn.sigmoid(tf.matmul(z, dw1) + db1)
pred = tf.matmul(l2, dw2) + db2

# Loss function with additional KL divergence and custom loss
reloss = tf.reduce_sum(tf.square(pred - x))
klloss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), 1)


# Define custom loss function
def DCloss(feature, batch_size):
    s = 0
    for i in range(1, batch_size):
        s += tf.pow(feature[i] - tf.constant(10, dtype=tf.float32) - tf.random.normal([1], 0, 1) - feature[i - 1], 2)
    return s


# Total loss
fealoss = DCloss(z, batch_size)
loss = tf.reduce_mean(0.1 * reloss + 0.6 * klloss + 10 * fealoss)

# Optimizer
optm = tf.compat.v1.train.AdamOptimizer(0.0003).minimize(loss)

# Training parameters
epochs = 500
display = 50
begin_time = time()

# Training the VAE
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

        if epoch % display == 0:
            print(f"Epoch {epoch}, Cost = {cost}")

    print('Training finished!!!')
    end_time = time()
    print(f"Training time: {end_time - begin_time:.2f} seconds")

    # Visualizing health index using latent representation (z)
    plt.figure()
    fea1 = sess.run(z, feed_dict={x: data})
    # fea2 = sess.run(z, feed_dict={x: data2})

    plt.plot(fea1, 'c-', label='Feature 1')
    # plt.plot(fea2, 'k-', label='Feature 2')

    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 23}
    plt.legend(loc='upper left', prop=font1)
    plt.title("Health Index")
    plt.xlabel("Serial Number")
    plt.ylabel("Health Index")
    plt.show()
