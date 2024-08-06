# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:46:11 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 10:54:59 2020

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from time import *

# This "resets the default graph" - esentially ensuring that there is no variable clashes
tf.compat.v1.reset_default_graph()

# This loads the data from their local directory. As you can see it is being imported from a MATLAB file, from a variable called "n" I believe. And then is transposed
path = r'E:\IEEEPHM2012\feature\norm_bearing1_2.mat'
matdata = scio.loadmat(path)
data2 = matdata['n']
data2 = np.transpose(data2)

# Same thing, but now a new dataset
path = r'E:\IEEEPHM2012\feature\norm_bearing1.mat'
matdata = scio.loadmat(path)
data6 = matdata['n']
data6 = np.transpose(data6)

# So now data = data6, which is a bit confusing. num_data = number of rows
data = data6
num_data = np.shape(data)[0]

# This function is used for weight initialization as explained in the paper (above eq. 10)
# It creates a uniform distribution for the weights
# fan_in = layer input neurons, fan_out = layer output neurons
# This prevents exploding gradients and should increase performance
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    # The line below indicates the weight matrix dimensions (W[i][j] = weight connecting input i with output j)
    # with the limit values minval and maxval being represented by low and high respectively
    return tf.random.uniform((fan_in, fan_out),
                             minval=low, maxval=high, dtype=tf.float32)

# number of columns
n_input = np.shape(data)[1]

#not really sure where this hidden_1=10 is coming from, in the paper I thought he specified 15->7->1
hidden_1 = 10
hidden_2 = 1
batch_size = 100

# MAIN CODE LOOP - replaces placeholder x with whatever feed_dict says, then does the operations, calculates z, finds loss

# This is important to remember: x is a PLACEHOLDER. It means that it is used as an "empty variable" which will be replaced later
# for example by batch_xs, data2 or data6
# its shape can be any rows, n_input columns
x = tf.compat.v1.placeholder(tf.float32, [None, n_input])

# weight for first layer, initialized by Xavier initialization
# this has shape input neurons x output neurons as explained in the comment on line 47
# W[i][j] = weight connecting input i with output j
# b1 is the bias of the first layer, one for each hidden neuron. Initialized as 0
w1 = tf.Variable(xavier_init(n_input, hidden_1))
b1 = tf.Variable(tf.zeros([hidden_1, ]))

# so the VAE creates distributions of how the data is distributed
# mean_w is a vector which has the mean values for all the distributions for the weights between the first hidden layer
# and the second, where the second hidden layer is the bottleneck z (what we want). Xavier initialized
# same goes for mean_b, just the bias of these weight mean values. Initialied as 0 again
# This distribution is only used for the latent space (the bottleneck) and not the output, so we won't use this in the decoder
mean_w = tf.Variable(xavier_init(hidden_1, hidden_2))
mean_b = tf.Variable(tf.zeros([hidden_2, ]))

# same as before, but now for the standard deviation (sigma) values of these distributions
logvar_w = tf.Variable(xavier_init(hidden_1, hidden_2))
logvar_b = tf.Variable(tf.zeros([hidden_2, ]))

# similar to before, except now for the decoder ("d"). So this is these are the weights and biases, same as w1 and b1
# Note that this is no longer a distribution of those weights but actual values, unlike mean_w and logvar_w
# This one is from bottleneck to first hidden layer of decoder (same dimensions as encoder, hidden_1)
dw1 = tf.Variable(xavier_init(hidden_2, hidden_1))
db1 = tf.Variable(tf.zeros([hidden_1, ]))

# Same, but now from first layer to output (same dimension as input, n_input)
dw2 = tf.Variable(xavier_init(hidden_1, n_input))
db2 = tf.Variable(tf.zeros([n_input, ]))

# From now on these are operations, calculating actual values
# l1 is the value at first hidden layer. This is input x*w1 (first weights) + bias1. Then passed through sigmoid
l1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)

# The mean of the latent space is l1*the mean values of weight + mean bias of weight
# same with logvar
mean = tf.matmul(l1, mean_w) + mean_b
logvar = tf.matmul(l1, logvar_w) + logvar_b

# eps is for epsilon, it represents noise. We are generating random noise with same shape as logvar, in a normal
# distribution with mean 0 and standard deviation 1. Adding this noise makes the model more diverse + variational
eps = tf.random.normal(tf.shape(logvar), 0, 1, dtype=tf.float32)

# logvar is actually ln(variance), so e^logvar is variance, and sqrt that is standard deviation. Here we are basically doing
# z = mean + standard deviation*epsilon. z is the value of the HI, the bottleneck. We save this later
z = tf.multiply(tf.sqrt(tf.exp(logvar)), eps) + mean

# This part is pretty simple now, same as before. l2 is value at 2nd hidden layer (first in decoder): z*weight decoder 1 + bias decoder 1
# and finally pred is the output of the model which should be a reconstruction of the original input x: pred = l2*weight decoder 2 + bias decoder 2
l2 = tf.nn.sigmoid(tf.matmul(z, dw1) + db1)
pred = tf.matmul(l2, dw2) + db2

# This is the end of the network operations themselves and now we will calculate losses

# This is the monotonicity loss term, as explained in the paper (eq. 7)
# basically doing (z_i - z_i-1 - r)^2, where z_i is value of HI at "time i"
# the r represents a constant that ranges from 9 to 10
# this is achieved by subtracting 10 and then subtracting a random number from 0 to 1 (normal distribution)
def DCloss(feature, batch_size):
    s = 0
    for i in range(1, batch_size):
        s += tf.pow(feature[i] - tf.constant(10, dtype=tf.float32)
                    - tf.random.normal([1], 0, 1) - feature[i - 1], 2)
    # All of this is done to avoid small fluctuations in the value of z having an effect
    # The value of 10 was determined "via a large number of experiments", we may need to find a value that works better for our data...
    # -> new hyperparameter
    return s

# This calculates monotonicity, reconstruction, and KL divergence losses as explained by the equations in the paper
# Note we are using pred for reconstruction loss which is the output of the decoder, comparing how similar it is to the input of the encoder
fealoss = DCloss(z, batch_size)
reloss = tf.reduce_sum(tf.pow(pred - x, 2))
klloss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), 1)

# This sums all 3, and again these 0.1, 0.6, 10 constants are determined from experiments
# We need to determine the optimal values for our data -> 3 new hyperparameters
# The bigger the number, the "more importance" you are telling the model it has
# Could be the cause of the straight line issue
loss = tf.reduce_mean(0.1 * reloss + 0.6 * klloss + 10 * fealoss)

# Adam optimizer, pretty standard. 0.0003 is the learning rate, in our code this is already a hyperparameter
optm = tf.compat.v1.train.AdamOptimizer(0.0003).minimize(loss)

# END OF MAIN CODE LOOP

epochs = 500
display = 50
begin_time = time()

# Starts a tensorflow session
with tf.compat.v1.Session() as sess:
    # Initializes variables
    # These are: w1, mean_w, logvar_w, dw1, dw2, b1, mean_b, logvar_b, db1, db2
    # The variables that have tf.Variable
    sess.run(tf.compat.v1.global_variables_initializer())

    # Training loop
    print('start tarining!!!')
    for epoch in range(epochs):
        # number of batches is an integer, same formula as always
        num_batch = int(num_data / batch_size)

        # Loop over all batches
        for i in range(num_batch):
            # Select the current batch data
            # In training, we use data, which before we set data=data6. So data6 is used for training here
            batch_xs = data[i * batch_size:(i + 1) * batch_size]

            # sess.run executes the session, basically running through the operations we have indicated
            # here the optimizer, and calculates the loss. _ is the variable returned by optm which is not needed
            # the part with feed_dict indicates to the code that any instances of "x" should be replaced with batch_xs
            # esentially x = batch_xs. As we saw before, x is a PLACEHOLDER. We can replace x and rerun the code
            _, cost = sess.run([optm, loss], feed_dict={x: batch_xs})

        # Print loss every 50 epochs
        if epoch % display == 0:
            print('cost =', cost)

    print('finish training!!!')
    end_time = time()
    print('training time is', end_time - begin_time)

    plt.figure()

    # So now, what this means is, run the session, and save "z" to fea2, when x = data2
    # Same for fea6
    # This second and third run of the session is no longer for training but for evaluating. So now it is like we are
    # using data=data2 and data=data6, with data6 as training data from before
    # As we saw before, x is a PLACEHOLDER. So now we are running the main session loop with data2, data6 in the placeholder "x"
    fea2 = sess.run(z, feed_dict={x: data2})
    fea6 = sess.run(z, feed_dict={x: data6})

    plt.plot(fea2, 'c-', label='bearing2')
    plt.plot(fea6, 'k-', label='bearing6')

    # Below is just formatting the graph, plotting HI vs "time"
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 23,
             }
    plt.legend(loc='upper left', prop=font1)
    plt.title("Health index")
    plt.xlabel("Serial number")
    plt.ylabel("Health index")
    plt.show()
