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

# Reset any previous graph (as explained in Hyper_Qin_original.py) and set seed for reproducibility
tf.compat.v1.reset_default_graph()
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Directory with data
dir_root = input("Enter directory of folder with data: ")
# C:\Users\pablo\Downloads\PZT Output folder

# This function takes the names of the csv's and returns the data concatenated horizontally and transposed
# The flags indicate where the data for a new file starts. We use this to combine the files later. We haven't used flags as far as I know?
def mergedata(train_filenames):
    flags = tuple([0])
    data = pd.read_csv(train_filenames[0], header=None)
    if len(train_filenames) > 1:
        for i in range(len(train_filenames)-1):
            flags += tuple([len(data)])
            data = pd.concat([data, pd.read_csv(train_filenames[i+1])], axis = 1)
    data = data.transpose()
    return data, flags

# As explained in Hyper_Qin_original.py
def DCloss(feature, batch_size):
    s = 0
    for i in range(1, batch_size):
        s += tf.pow(feature[i] - tf.constant(10, dtype=tf.float32) - tf.random.normal([1], 0, 1) - feature[i - 1], 2)
    return s

# As you might remember from the oral exam presentations, we need a different formula to calculate fitness with test HI, this is it
# Just uses different Pr and Mo functions. test_HI is self explanatory, X is an array with the training HIs (test_HI needs to be compared to them for trendability)6
def test_fitness(test_HI, X):
    test_HI = test_HI[0]
    monotonicity = Mo_single(test_HI)
    trendability = Tr(np.vstack([test_HI, X]))
    prognosability = Pr_single(test_HI, X)
    fitness_test = (monotonicity + trendability + prognosability), monotonicity, trendability , prognosability

    return fitness_test

# This function finds the size of the largest array in a list of arrays
# We use this to scale all HIs later to the same length
def find_largest_array_size(array_list):
    max_size = 0

    for arr in array_list:
        if isinstance(arr, np.ndarray):
            size = arr.size
            if size > max_size:
                max_size = size

    return max_size

# This function creates the csv's where hyperparameters will be stored and does the actual storing.
# It will create 3 files: hyperparameters-opt, and fitness-test/train
# After a given number of repetitions, the best-performing hyperparameters for each fold need to be stored
# hyperparameters-opt is a csv which contains the actual values for those hyperparameters e.g. hidden_1=15, learning_rate=0.001, etc...
# fitness-test contains the fitness-test value (only test HI) when running the VAE with those hyperparameters
# fitness-all is the same, but it saves the fitness of train+test HIs. These last 2 files are just to compare the performance of the found optimum hyperparameters
def store_hyperparameters(hyperparameters, fitness_all, fitness_test, panel, freq):
    global dir_root
    filename_opt = os.path.join(dir_root, "hyperparameters-opt.csv")
    filename_test = os.path.join(dir_root, "fitness-test.csv")
    filename_all = os.path.join(dir_root, "fitness-all.csv")
    freqs = ["050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz"]

    # Create an empty dataframe with frequencies as the index if the file does not exist
    if not os.path.exists(filename_opt):
        df = pd.DataFrame(index=freqs)
    else:
        # Load the existing file if it exists
        df = pd.read_csv(filename_opt, index_col=0)

    # Ensure that the panel column exists
    if panel not in df.columns:
        df[panel] = None

    # Update the dataframe with the new hyperparameters
    df.loc[freq, panel] = str(hyperparameters)

    # Save the dataframe back to the CSV
    df.to_csv(filename_opt)

    # Create an empty dataframe with frequencies as the index if the file does not exist
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

# Same function but only stores one file with the hyperparameters and not the other 2 with results
def simple_store_hyperparameters(hyperparameters, file, panel, freq, dir):

    filename_opt = os.path.join(dir, file + "-hopt.csv")
    freqs = ["050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz"]
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

################ MAIN FUNCTION: takes data and hyperparameters, generates train and test HIs
# Function to train VAE and generate results but with hyperparameters as function inputs
# NOTE: have removed global variables, that's why there's 5 extra inputs: train_data, test_data, scaler, pca, seed
def train_vae(hidden_1, batch_size, learning_rate, epochs, train_data, test_data, scaler, pca, seed):

    #global valid
    # commented, this is for validation data?

    n_input = train_data.shape[1]  # Number of features
    hidden_2 = 1
    display = 200

    # Xavier initialization for weights, as explained in Hyper_Qin_original.py
    def xavier_init(fan_in, fan_out, constant=1, seed):
        tf.random.set_seed(seed)
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random.uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

    # I don't remember what this line is, but we need it for some compatibility issues
    tf.compat.v1.disable_eager_execution()

    # Architecture and losses below as explained in Hyper_Qin_original.py
    x = tf.compat.v1.placeholder(tf.float32, [None, n_input])

    w1 = tf.Variable(xavier_init(n_input, hidden_1, seed=seed))
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

    # Training, as explained in Hyper_Qin_original.py
    begin_time = time()

    # In sess, we are saving everything from the training loop, the learnt weights and biases, etc
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
            #validation_loss = sess.run(loss, feed_dict={x: valid})
            # not sure why this line is here, it's for something with validation, but we can keep it as a comment

        if epoch % display == 0:
            print(f"Epoch {epoch}, Cost = {cost})

    print('Training finished!!!')
    end_time = time()
    print(f"Training time: {end_time - begin_time:.2f} seconds")

    # Here is where things start changing a lot compared to Hyper_Qin_original
    # So this z_test stores the bottleneck values for the test dataset (x: test -> x=test). We will see later where test comes from
    z_test = sess.run(z, feed_dict={x: test_data})
    z_test = z_test.transpose()

    # In this variable the train HIs will be saved
    z_train = []

    # This loops x over all panels EXCEPT the current test panel. j represents panel name
    for j in tuple(x for x in panels if x != panel):

        # dir_root comes from the input at the start of the code
        # single_panel_data is the data from the files, but only from one panel at a time, NOT merged
        # first the file is read, the final column is deleted (don't know why), then it is normalized and PCA is applied
        # Not sure why we must import scaler and pca to do this, when those are fitted with the merged train data
        single_panel_data = pd.read_csv(dir_root + "\concatenated_" + freq + "_" + j + "_HLB_Features.csv", header=None).values.transpose()
        single_panel_data = np.delete(single_panel_data, -1, axis=1)

        # Then, the train HI is calculated for each panel individually and appended to the array z_train
        single_panel_data = scaler.transform(single_panel_data)
        single_panel_data = pca.transform(single_panel_data)
        z_train_individual = sess.run(z, feed_dict={x: single_panel_data})
        z_train.append(z_train_individual)


    # Transpose
    for i in range(len(z_train)):
        z_train[i] = z_train[i].transpose()

    # We find the longest HI and interpolate the other HIs so that they all have matching lengths
    max = find_largest_array_size(z_train)
    for i in range(len(z_train)):
        if z_train[i].size < max:

            # The interpolation function is found for the shorter HI, and applied to arr_stretch. Then the shorter HI z_train[i] is updated
            interp_function = interp.interp1d(np.arange(z_train[i].size), z_train[i])
            arr_stretch = interp_function(np.linspace(0, z_train[i].size - 1, max))
            z_train[i] = arr_stretch

    # Vertically stack the HIs
    z_train = np.vstack(z_train)

    # We do the same but to match the length of z_test with z_train
    if z_test.size != z_train.shape[1]:
        interp_function = interp.interp1d(np.arange(z_test.size), z_test)
        arr_stretch = interp_function(np.linspace(0, z_test.size - 1, z_train.shape[1]))
        z_test = arr_stretch

    # Create an array that contains all HIs (train+test)
    z_all = np.append(z_train, z_test, axis = 0)

    # Close the TensorFlow session
    sess.close()

    return [z_all, z_train, z_test]

# Hyperparameter optimization

# This function is just to print what call number you are on
def print_progress(res):
    n_calls = len(res.x_iters)
    print(f"Call number: {n_calls}")

# This is the space over which we let hyperparameters be optimized. So for example hidden_1 can only be between 10 and 100
space = [
        Integer(10, 100, name='hidden_1'),
        Integer(16, 128, name='batch_size'),
        Real(0.0001, 0.01, name='learning_rate'),
        Integer(500, 10000, name='epochs')
    ]
# I'm not 100% sure how this works, but apparently it maps the parameters from objective to the gp.minimize below
@use_named_args(space)

# This function is the objective so what we want to minimize. That is the error (3/fitness).
# We give it all these inputs so that we can run train_vae and calculate fitness
def objective(hidden_1, batch_size, learning_rate, epochs, train_data, test_data, scaler, pca):
    ftn, monotonicity, trendability, prognosability, error = fitness(train_vae(hidden_1, batch_size, learning_rate, epochs, train_data, test_data, scaler, pca, seed)[1])
    print("Error: ", error)
    return error

# Now this function looks a little different than what it used to be
# As train_vae no longer uses global variables, there's now this objective_with_fixed_args function inside this function
# Essentially it's the same as what it used to be except that now we must specify which inputs to optimize and which are fixed
def hyperparameter_optimisation(n_calls, random_state=42):
    fixed_args = {
        'train_data': train_data,
        'test_data': test_data,
        'scaler': scaler,
        'pca': pca
    }
    def objective_with_fixed_args(**params):
        all_args = {**params, **fixed_args}
        return objective(**all_args)

    res_gp = gp_minimize(objective_with_fixed_args, space, n_calls=n_calls, random_state=random_state,
                         callback=[print_progress])
    opt_parameters = [res_gp.x, res_gp.fun]
    print("Best parameters found: ", res_gp.x)
    return opt_parameters

# This is what we will iterate over in 2 for loops, in this order.
# So first: panel l103 freq 050, l103 100, l103 125, l103 150, l103 200, l103 250, l105 050, etc...
# result_dictionary is an empty dictionary where results will be stored
# Counter is to know how many iterations you've completed, so one for every combination of freq and panel
panels = ("L103", "L105", "L109", "L104", "L123")
freqs = ("050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz")
result_dictionary = {}
counter = 0

# Here are the two for loops
# MAIN BLOCK OF CODE
for panel in panels:
    for freq in freqs:

        train_filenames = []

        # The array train_filenames saves the filenames of the data to be used for training
        # Note that here and a few lines below, we hardcode the name "HLB". This is to run the VAE on HLB data
        # When implementing VAE into main, this will need to be changed such that it's not hardcoded
        for i in tuple(x for x in panels if x != panel):
            filename = os.path.join(dir_root, f"concatenated_{freq}_{i}_HLB_Features.csv")
            train_filenames.append(filename)

        # Create an empty entry in the dictionary for this fold (e.g. L103050_kHz)
        result_dictionary[f"{panel}{freq}"] = []

        counter += 1
        print("Counter: ", counter)
        print("Panel: ", panel)
        print("Freq: ", freq)

        # For train_data, create the merged file with all 4 panels, delete the last column. I don't know why we delete the last column though
        train_data, flags = mergedata(train_filenames)
        train_data.drop(train_data.columns[len(train_data.columns) - 1], axis=1, inplace=True)

        # Same as with train data. Read the filename, and delete the last column
        # Also hardcoded
        test_filename = os.path.join(dir_root, f"concatenated_{freq}_{panel}_HLB_Features.csv")
        test_data = pd.read_csv(test_filename, header=None).values.transpose()
        test_data = np.delete(test_data, -1, axis=1)

        # Normalizing the train and test data, scaled with train_data
        scaler = StandardScaler()
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)

        # Applying PCA to the train and test data, fit with train_data
        pca = PCA(n_components=30)
        pca.fit(train_data)
        train_data = pca.transform(train_data)
        test_data = pca.transform(test_data)

        # Extract optimal hyperparameters, here we indicate how many calls for hyperparameter optimisation we want (min. 10)
        hyperparameters = hyperparameter_optimisation(n_calls=20)

        # Determine the test and train HIs from the VAE model
        # [0][0] is hidden_1, [0][1] is batch_size, [0][2] is learning_rate and [0][3] is epochs
        health_indicators = train_vae(hyperparameters[0][0], hyperparameters[0][1],
                                      hyperparameters[0][2], hyperparameters[0][3], train_data, test_data, scaler, pca, seed)

        # fitness_all value (includes train panel), health_indicators[0] is z_all (all HIs, train+test)
        fitness_all = fitness(health_indicators[0])
        print("Fitness all", fitness_all)

        #fitness_test value (only test panel), health_indicators[2] is z_test and health_indicators[1] is z_train
        # Remember for test_fitness we compare the test HI to the train HIs
        # health_indicators[2] is z_test, health_indicators[1] is z_train
        fitness_test, m, t, p = test_fitness(health_indicators[2], health_indicators[1])
        print("Fitness test", fitness_test)

        # For the current entry of result_dictionary, append the fitness_train and all values
        result_dictionary[f"{panel}{freq}"].append([fitness_all, fitness_test])

        # Writing to the csv's
        fitness_test = (fitness_test, m, t, p)
        fitness_all = (fitness_all)
        store_hyperparameters(hyperparameters, fitness_all, fitness_test, panel, freq)

# Write the result_dictionary to a csv file. To reach this line, the two for loops must be completed
# This takes extremely long for Hyper_clean, so realistically will only be reached with Hyper_clean_fast (as the hyperparameter optimization is skipped)
# Basically means we only need this file to generate the hyperparameters and not also generate results all at once
# We can implement into main a hyperparameter generation step and a results step, one "modelled" after Hyper_clean, the other after Hyper_clean_fast
with open("results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, result_dictionary.keys())
    w.writeheader()
    w.writerow(result_dictionary)

# CONCERNS:
# 1. I don't know yet why columns are deleted in the data 2. Don't know why we import pca and scaler to transform the individual panel data when this has
# been fitted to the merged data, seems weird 3.