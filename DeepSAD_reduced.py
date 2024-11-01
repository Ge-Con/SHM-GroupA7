# Import libraries
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import copy
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from Interpolating import scale_exact
from prognosticcriteria_v2 import fitness, test_fitness
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Suppress warnings - please comment out if doing any troubleshooting or changes to the model
import warnings

warnings.filterwarnings('ignore')

# Global variables necessary for passing data other than parameters during hyperparameter optimisation
global pass_train_data
global pass_semi_targets
global pass_fnwf
global pass_dir

# Random seed for repeatability
global ds_seed
ds_seed = 120
torch.manual_seed(ds_seed)


class NeuralNet(nn.Module):
    """
        Encoder network

        Attributes:
        - c (2D numpy array): Hypersphere centre
        - size (List of two integers): Dimensions of output
        - fc1 to fc 5 (Linear objects): Layers
        - m (LeakyReLU object): Activation function

        Methods:
        - forward: Forward pass through encoder
    """

    def __init__(self, size):
        super().__init__()  # Initialise parent torch module
        self.c = None  # Define c to be set later
        self.size = size  # Set size to an attribute

        # Create network layers
        self.fc1 = nn.Linear(size[0] * size[1], 512) #1024
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 16)
        # Create activation function
        self.m = torch.nn.LeakyReLU(0.01)

    def forward(self, x):
        x = torch.flatten(x, start_dim=0)  # Flatten matrix input
        x = x.to(next(self.parameters()).dtype)  # Ensure tensor is of correct datatype
        x = self.m(self.fc1(x))  # Forward pass through layers
        x = self.m(self.fc2(x))
        x = self.m(self.fc3(x))
        x = self.m(self.fc4(x))
        x = self.m(self.fc5(x))
        x = self.m(self.fc6(x))
        # Reshape output to same as c
        encoded = x.view(4, 4)
        return encoded


class NeuralNet_Decoder(nn.Module):
    """
        Decoder network for NeuralNet

        Attributes:
        - size (List of two integers): Dimensions of output
        - fc1 to fc 5 (Linear objects): Layers
        - m (LeakyReLU object): Activation function

        Methods:
        - forward: Forward pass through decoder
    """

    def __init__(self, size):
        super().__init__()  # Initialise parent torch module
        self.size = size  # Set size to an attribute
        # Create network layers
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, size[0] * size[1])
        # Create activation function
        self.m = torch.nn.LeakyReLU(0.01)

    def forward(self, x):
        x = torch.flatten(x)  # Flatten matrix input
        x = self.m(self.fc1(x))  # Run through network layers
        x = self.m(self.fc2(x))
        x = self.m(self.fc3(x))
        x = self.m(self.fc4(x))
        x = self.m(self.fc5(x))
        x = self.fc6(x)
        x = x.view(-1, self.size[0], self.size[1])  # Reconstruct matrix of original data dimenions
        return x


class NeuralNet_Autoencoder(nn.Module):
    """
        Autoencoder network

        Attributes:
        - encoder (NeuralNet object): Encoder network
        - decoder (NeuralNetObject): Decoder network

        Methods:
        - forward: Forward pass through autoencoder
    """

    def __init__(self, size):
        super().__init__()  # Initialise parent torch module
        # Create encoder and decoder and save as attributes
        self.encoder = NeuralNet(size)
        self.decoder = NeuralNet_Decoder(size)

    def forward(self, x):
        # Run encoder and decoder
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def init_c(model, train_loader, eps=0.1):
    """
        Initialise hypersphere center c as the mean from an initial forward pass on the data

        Parameters:
        - model (DeepSAD_net object): Untrained DeepSAD model
        - train_loader (DataLoader object): Training data
        - eps (float): Very small number to prevent zero errors

        Returns:
        - c (2D numpy array): Coordinates of hypersphere centre
    """

    n_samples = 0
    c = torch.zeros((4, 4))  # 16-dimensional coordinates, formatted into 2D array

    # Forward pass
    model.eval()
    with torch.no_grad():

        # Calculate network outputs for all data
        for train_data, train_target in train_loader:
            for index in range(len(train_data)):
                data = train_data[index]
                target = train_target[index]

                outputs = model(data)
                n_samples += 1

                c += outputs
    c /= n_samples  # Average outputs

    # If c_i is too close to 0, set to +-eps
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def train(model, train_loader, learning_rate, weight_decay, n_epochs, lr_milestones, gamma, eta, eps, reg=0.001):
    """
        Train the DeepSAD model from a semi-labelled dataset.

        Parameters:
        - model (NeuralNet object): DeepSAD model
        - train_loader (DataLoader object): Training data
        in addition to hyperparameters:
        - learning_rate (float): Learning rate
        - weight_decay (float): Factor to reduce LR by at milestones
        - n_epochs (int): Number of epochs for training
        - lr_milestones (list): Epoch milestones to reduce learning rate
        - gamma (float): Weighting of L2 regularisation and
        - eta (float): Weighting of labelled data points
        - eps (float): Small number to prevent zero errors
        - reg (float): Weighting of diversity loss function

        Returns:
        None
    """

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=gamma)

    # Initialise c if necessary
    if model.c is None:
        model.c = init_c(model, train_loader, eps)

    # Iterate epochs to train model
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for train_data, train_target in train_loader:
            loss = 0.0
            n_batches = 0

            for index in range(len(train_data)):
                data = train_data[index]
                target = train_target[index]

                # Forward and backward pass
                optimizer.zero_grad()
                outputs = model(data)

                # Calculating loss function
                Y = outputs - model.c
                dist = torch.sum(Y ** 2)
                loss_d = 0
                if target == 0:
                    losses = dist
                else:
                    losses = eta * ((dist + eps) ** (target))

                if reg != 0:  # If we want to diversify
                    C = torch.matmul(Y.T, Y)  # Gram Matrix
                    loss_d = -torch.log(torch.det(C)) + torch.trace(C)  # Diversity loss contribution

                losses += reg * loss_d

                #losses += (dist-(target-1)*-0.5)**2 #Fit to labels

                loss += losses
                n_batches += 1

            # Finish off training the network
            loss = loss / n_batches
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        scheduler.step()

    return model, epoch_loss


def AE_train(model, train_loader, learning_rate, weight_decay, n_epochs, lr_milestones, gamma):
    """
        Trains neural net model weights

        Parameters:
        - model (NeuralNet object): DeepSAD model
        - train_loader (DataLoader object): Data loader for training data and targets
        - learning_rate (float): Learning rate
        - weight_decay (float): Factor for L2 regularisation
        - n_epochs (int): Number of epochs for training
        - lr_milestones (list): Epoch milestones to reduce learning rate
        - gamma (float): Factor to reduce learning rate by at milestones

        Returns:
        - model (NeuralNet object): Trained neural network
    """

    # Set optimizer (Adam optimizer for now)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Set learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=gamma)

    model.train()
    for epoch in range(n_epochs):

        scheduler.step()
        epoch_loss = 0.0

        for train_data, train_target in train_loader:
            loss = 0.0
            n_batches = 0
            for index in range(len(train_data)):
                data = train_data[index]
                target = train_target[index]

                # Zero the network parameter gradients
                optimizer.zero_grad()
                outputs = model(data)
                loss_f = nn.MSELoss()
                losses = loss_f(outputs, data)
                loss += losses
                n_batches += 1
            loss = loss / n_batches
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    return model


def pretrain(model, train_loader, learning_rate, weight_decay, n_epochs, lr_milestones, gamma):
    """
        Pretrains neural net model weights using an autoencoder

        Parameters:
        - model (NeuralNet object): DeepSAD model
        - train_loader (DataLoader object): Data loader for training data and targets
        - learning_rate (float): Autoencoder learning rate
        - weight_decay (float): Weight decay for L2 regularisation and milestones
        - n_epochs (int): Number of epochs for pretraining
        - lr_milestones (list): Epoch milestones to reduce learning rate

        Returns:
        - model (NeuralNet object): Pretrained neural network
    """

    # Create and train autoencoder
    ae_model = NeuralNet_Autoencoder(model.size)
    ae_model = AE_train(ae_model, train_loader, learning_rate, weight_decay, n_epochs, lr_milestones, gamma)

    # Create dictionaries to store network states
    model_dict = model.state_dict()
    ae_model_dict = ae_model.state_dict()

    # Remove decoder network keys
    ae_model_dict = {k: v for k, v in ae_model_dict.items() if k in model_dict}
    # Update and reload network states
    model_dict.update(ae_model_dict)
    model.load_state_dict(model_dict)

    return model


def embed(X, model):
    """
        Returns a health indicator for a test dataset

        Parameters:
        - X (2D array): Array containing feature data
        - model (DeepSAD_net object): Trained DeepSAD model

        Returns:
        - y (float): Anomaly score to be used as a Health Indicator
    """

    model.eval()
    y = torch.norm(model(X) - model.c)  # Magnitude of the vector is anomaly score
    return y


def load_data(dir, filename, labelled_fraction, ignore):
    """
        Loads data from CSV files

        Parameters:
        - dir (string): Root directory of train/test data
        - filename (string): file name for train/test data
        - ignore (int): number of time steps from end to disregard

        Returns:
         - data (2D numpy array): list of training data vectors
         - labels (1D numpy array): artificial labels for training data
    """
    data = None
    labels = None
    first = True  # First sample flag

    # Walk directory
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name == filename:  # If correct file to be included in training data
                read_data = np.array(pd.read_csv(os.path.join(root, name)))

                # Set data and labels arrays to data from first sample
                if first:
                    data = np.array([read_data])
                    labels = np.array([1.0])
                    first = False

                # Concatenate additional samples
                else:
                    data = np.concatenate((data, [read_data]))
                    labels = np.append(labels, 0)  # Default label is 0

    if labels is not None and len(labels) > 0:

        if ignore != 0:
            data = data[0:-1 * ignore]
            labels = labels[0:-1 * ignore]

        # Add artificial labels
        teol = data.shape[0]

        x_values = np.arange(1, teol + 1)
        #health_indicators = ((x_values ** 2) / (teol ** 2)) * 2 - 1  # Equation scaled from -1 to 1
        health_indicators = -1+2*x_values/teol

        for i in range(int(len(labels) * labelled_fraction)):  # Originally 5
            labels[i] = health_indicators[-i - 1]  # Healthy

        for i in range(int(len(labels) * labelled_fraction)):  # Originally 3
            labels[-i - 1] = health_indicators[i]  # Unhealthy

        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float)
    else:
        raise ValueError("No data loaded or empty dataset found.")


def DeepSAD_train_run(dir, freq, file_name):
    """
       Trains and runs the DeepSAD model

       Parameters:
       - dir (string): Root directory of train/test data
       - freq (string): 3-digit frequency for train/test data
       - filename (string): file name for train/test data, excluding freq, "kHz_" and .csv

       Returns:
        - results (2D numpy array): 5x30 Array of health indicators with state for each panel
    """

    ### Hyperparamters ###
    #Fixed/background
    lr_milestones_AE = []#[8]  # Milestones when learning rate reduces
    lr_milestones = []#[20, 40, 60, 80]
    gamma = 0.1 # Factor to reduce LR by at milestones
    gamma_AE = 0.1  # "
    eps = 1 * 10 ** (-6)  # Very small number to prevent zero errors

    #Training
    batch_size = 128 # Include in HPO   - 50 to 150 (128 from paper)
    learning_rate_AE = 0.0005 # Include in HPO - 0.0001 to 0.001 (0.0005)
    learning_rate = 0.0005 # Include in HPO - 0.0001 to 0.001 (0.0005)
    n_epochs_AE = 10 # Include in HPO - 5 to 20 (10)
    n_epochs = 100 # Include in HPO   - 50 to 200 (100)

    #Loss function
    weight_decay = 10  # Nu | From paper - 1 or 10, doesn't make much difference
    weight_decay_AE = weight_decay  # Keep it the same
    eta = 10  # Weighting of LABELLED datapoints (unlabelled weighting 1)
    reg = 0.001  # Lambda - diversity weighting (from paper)

    #Additional (to do with labels, not in original model)
    labelled_fraction = 0.25  # Labelled points from each end (so strictly < 0.5) | Don't include in HPO? Do not set below 0.1. Very little difference in results from 0.25 up to 0.5
                            # Keep well below 0.5 to maintain gap in the middle to enable us to use straight line labels
    ignore = 0  #Number of timesteps from end to ignore - leave at 0, anything else was bad


    global pass_dir
    pass_dir = dir

    # Make string of filename for train/test data
    file_name_with_freq = freq + "kHz_" + file_name + ".csv"
    # print(f"Training with directory: {dir}, frequency: {freq}, filename: {file_name_with_freq}")

    samples = ["PZT-FFT-HLB-L1-03", "PZT-FFT-HLB-L1-04", "PZT-FFT-HLB-L1-05", "PZT-FFT-HLB-L1-09", "PZT-FFT-HLB-L1-23"]
    # Initialise results matrix
    results = np.empty((5, 5, 30-ignore))
    hps = []
    global pass_fnwf
    # Loop for each sample as test data
    for sample_count in range(len(samples)):
        print("--- ", freq, "kHz, Sample ", sample_count + 1, " as test ---")
        test_sample = samples[sample_count]

        # Make new list of samples excluding test data
        temp_samples = copy.deepcopy(samples)
        temp_samples.remove(test_sample)

        first = True  # Flag for first training sample
        # Iterate and retrieve each training sample
        for count in range(len(temp_samples)):
            sample = temp_samples[count]

            # Load training sample
            temp_data, temp_targets = load_data(os.path.join(dir, sample), file_name_with_freq, labelled_fraction, ignore)

            # Create new arrays for training data and targets
            if first:
                arr_data = copy.deepcopy(temp_data)
                arr_targets = copy.deepcopy(temp_targets)
                first = False

            # Concatenate data and targets from other samples
            else:
                arr_data = np.concatenate((arr_data, temp_data))
                arr_targets = np.concatenate((arr_targets, temp_targets))

        # Normalise training data
        normal_mn = np.mean(arr_data, axis=0)
        normal_sd = np.std(arr_data, axis=0)
        arr_data = (arr_data - normal_mn) / normal_sd

        #PCA
        #pca = PCA(n_components=0.95)
        #pca.fit(scaled_data)
        #arr_data = pca.transform(scaled_data)

        # Convert to pytorch tensors
        train_data = torch.tensor(arr_data)
        semi_targets = torch.tensor(arr_targets)

        # Create list of data dimensions to set number of input nodes in neural network
        size = [train_data.shape[1], train_data.shape[2]]

        # Convert to dataset and create loader
        train_dataset = TensorDataset(train_data, semi_targets)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Create, pretrain and train a model
        model = NeuralNet(size)
        model = pretrain(model, train_loader, learning_rate_AE, weight_decay=weight_decay_AE, n_epochs=n_epochs_AE,
                         lr_milestones=lr_milestones_AE, gamma=gamma_AE)
        model, loss = train(model, train_loader, learning_rate, weight_decay=weight_decay, n_epochs=n_epochs,
                            lr_milestones=lr_milestones, gamma=gamma, eta=eta, eps=eps, reg=reg)

        # Display first layer heatmap
        # fc1_weights = model.fc1.weight.data
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(fc1_weights.cpu().numpy(), cmap="viridis", cbar=True)
        # plt.title("Weights of the First Layer (fc1)")
        # plt.show()


        # Test for all panels
        # Load test sample data (targets not used)
        list = []
        for test_sample in samples:
            test_data, temp_targets = load_data(os.path.join(dir, test_sample), file_name_with_freq, labelled_fraction, ignore)
            test_data = (test_data - normal_mn) / normal_sd  # Normalise using test statistics
            #test_data = pca.transform(test_data)

            # Calculate HI at each state
            current_result = []
            for state in range(test_data.shape[0]):
                data = test_data[state]
                current_result.append(embed(data, model).item())

            # Interpolate
            list.append(scale_exact(np.array(current_result), 30-ignore))

        list = np.array(list)

        # Scale so on average starts at 0 and ends at 1, excluding test sample
        av_start = np.mean(np.concatenate((list[:sample_count, 0], list[sample_count:, 0])))
        av_end = np.mean(np.concatenate((list[:sample_count, -1], list[sample_count:, -1])))
        list = (list - av_start) / av_end

        # Plot and print fitness
        ftn = fitness(list)
        testftn = test_fitness([list[sample_count]], list)
        print("F-test:", testftn[0], "| Mo:", testftn[1], "| Tr:", testftn[2], "| Pr:", testftn[3])
        print("F-all: ", ftn[0], "| Mo:", ftn[1], "| Tr:", ftn[2], "| Pr:", ftn[3])
        #Graphs.HI_graph(list, dir, samples[sample_count] + " " + freq + "kHz")

        results[sample_count] = list

    return results


def plot_ds_images(dir, type):
    """
        Assemble grid of HI graphs

        Parameters:
        - dir (str): Directory of HI graph images
        - type (str): Seed of HIs generated

        Returns: None
    """

    # Define variables
    filedir = os.path.join(dir, f"big_VAE_graph_seed_{ds_seed}")
    nrows = 6
    ncols = 5
    panels = ("0", "1", "2", "3", "4")
    freqs = ("050", "100", "125", "150", "200", "250")
    samples = ["PZT-FFT-HLB-L1-03", "PZT-FFT-HLB-L1-04", "PZT-FFT-HLB-L1-05", "PZT-FFT-HLB-L1-09", "PZT-FFT-HLB-L1-23"]
    fig, axs = plt.subplots(nrows, ncols, figsize=(40, 35))  # Adjusted figure size

    # For each frequency and panel
    for i, freq in enumerate(freqs):
        for j, panel in enumerate(panels):
            # Generate the filename
            filename = f"{samples[int(panel)]} {freq}kHz.png"

            # Check if the file exists
            if os.path.exists(os.path.join(dir, filename)):
                # Load the image
                img = mpimg.imread(os.path.join(dir, filename))

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
        ax.annotate(f'{row}', (-0.1, 0.5), xycoords='axes fraction', rotation=90, va='center', fontweight='bold',
                    fontsize=40)

    # Add column labels
    for ax, col in zip(axs[0], panels):
        ax.annotate(f'Test Sample {panels.index(col) + 1}', (0.5, 1), xycoords='axes fraction', ha='center',
                    fontweight='bold', fontsize=40)

    # Adjust spacing between subplots and save
    plt.tight_layout()
    plt.savefig(filedir)

if True:
    frequencies = ["050", "100", "125", "150", "200", "250"]
    HIs = np.empty((6), dtype=object)
    #dir = "C:\\Users\\geort\\Desktop\\CSV-FFT-HLB-Reduced 2"
    #dir = "CSV-FFT-HLB-Reduced"
    #dir = "/Users/cornelie/Desktop/CSV-FFT-HLB-Reduced"
    dir = "C:\\Users\\Jamie\\Documents\\Uni\\Year 2\\Q3+4\\Project\\CSV-FFT-HLB-Reduced"
    type = "FFT"
    filename = "_FT_Reduced"
    for freq in range(len(frequencies)):
         print(f"Processing frequency: {frequencies[freq]} kHz for " + type)
         HIs[freq] = DeepSAD_train_run(dir, frequencies[freq], type + filename)
     # Save and plot results
    #save_evaluation(np.array(HIs), "DeepSAD", dir, filename)
    #plot_ds_images(dir, type)