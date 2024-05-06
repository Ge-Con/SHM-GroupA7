# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:46:11 2020

@author: Administrator
"""
import pandas as pd

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

tf.compat.v1.reset_default_graph()

#path = r'E:\IEEEPHM2012\feature\norm_bearing1_2.mat'
#matdata = scio.loadmat(path)
#data2 = matdata['n']
#data2 = np.transpose(data2)

#path = r'E:\IEEEPHM2012\feature\norm_bearing1.mat'
matdata = pd.read_csv('resultVAEtest.csv')
data6 = matdata['n']
data6 = np.transpose(data6)

data = data6
num_data = np.shape(data)[0]


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random.uniform((fan_in, fan_out),
                             minval=low, maxval=high, dtype=tf.float32)


n_input = np.shape(data)[1]
hidden_1 = 10
hidden_2 = 1
batch_size = 100

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


def DCloss(feature, batch_size):
    s = 0
    for i in range(1, batch_size):
        s += tf.pow(feature[i] - tf.constant(10, dtype=tf.float32)
                    - tf.random.normal([1], 0, 1) - feature[i - 1], 2)
    return s


fealoss = DCloss(z, batch_size)
reloss = tf.reduce_sum(tf.pow(pred - x, 2))
klloss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), 1)
loss = tf.reduce_mean(0.1 * reloss + 0.6 * klloss + 10 * fealoss)

optm = tf.compat.v1.train.AdamOptimizer(0.0003).minimize(loss)

epochs = 500
display = 50
begin_time = time()

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print('start tarining!!!')
    for epoch in range(epochs):
        num_batch = int(num_data / batch_size)
        for i in range(num_batch):
            batch_xs = data[i * batch_size:(i + 1) * batch_size]
            _, cost = sess.run([optm, loss], feed_dict={x: batch_xs})
        if epoch % display == 0:
            print('cost =', cost)
    print('finish training!!!')
    end_time = time()
    print('training time is', end_time - begin_time)

    plt.figure()
    fea2 = sess.run(z, feed_dict={x: data2})
    fea6 = sess.run(z, feed_dict={x: data6})
    plt.plot(fea2, 'c-', label='bearing2')
    plt.plot(fea6, 'k-', label='bearing6')
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 23,
             }
    plt.legend(loc='upper left', prop=font1)
    plt.title("Health index")
    plt.xlabel("Serial number")
    plt.ylabel("Health index")
    plt.show()
