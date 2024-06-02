#Just working in here for labels to make sure nothing breaks in original file

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import copy

#original without labels

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
        """
            Initialise decoder

            Parameters:
            - size (List of two integers): Input dimensions

            Returns:
            None
        """
        super().__init__()      #Initialise parent torch module
        self.c = None           #Define c to be set later
        self.size = size        #Set size to an attribute

        #Create network layers
        self.fc1 = nn.Linear(size[0]*size[1], 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 16)
        # Create activation function
        self.m = torch.nn.LeakyReLU(0.001)

    def forward(self, x):
        """
            Forward pass through encoder

            Parameters:
            - x (2D numpy array): Training data

            Returns:
            - x (2D numpy array): Network output
        """

        x = torch.flatten(x)    #Flatten matrix input
        x = x.to(next(self.parameters()).dtype) #Ensure tensor is of correct datatype
        x = self.m(self.fc1(x)) #Forward pass through layers
        x = self.m(self.fc2(x))
        x = self.m(self.fc3(x))
        x = self.m(self.fc4(x))
        x = self.m(self.fc5(x))
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
        """
            Initialise decoder

            Parameters:
            - size (List of two integers): Input dimensions

            Returns:
            None
        """
        super().__init__()      #Initialise parent torch module
        self.size = size        #Set size to an attribute

        #Create network layers
        self.fc1 = nn.Linear(1024, size[0]*size[1])
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(16, 64)
        #Create activation function
        self.m = torch.nn.LeakyReLU(0.001)

    def forward(self, x):
        """
            Forward pass through decoder

            Parameters:
            - x (2D numpy array): Training data

            Returns:
            - x (2D numpy array): Network output
        """

        x = torch.flatten(x)    #Flatten matrix input
        x = self.m(self.fc5(x)) #Run through network layers
        x = self.m(self.fc4(x))
        x = self.m(self.fc3(x))
        x = self.m(self.fc2(x))
        x = self.m(self.fc1(x))
        x = x.view(-1, self.size[0], self.size[1])  #Reconstruct matrix of original data dimenions
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
        """
            Initialise autoencoder

            Parameters:
            - size (List of two integers): Input dimensions

            Returns:
            None
        """
        super().__init__()      #Initialise parent torch module

        #Create encoder and decoder and save as attributes
        self.encoder = NeuralNet(size)
        self.decoder = NeuralNet_Decoder(size)

    def forward(self, x):
        """
            Autoencoder forward pass, through encoder and decoder

            Parameters:
            - x (2D numpy array): Training data

            Returns:
            - x (2D numpy array): Network output
        """

        #Run encoder and decoder
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
    c = torch.zeros((4, 4))     #16-dimensional coordinates, formatted into 2D array

    #Forward pass
    model.eval()
    with torch.no_grad():
        #Load data
        train_data, train_targets = enumerate(train_loader)

        #Calculate network outputs for all data
        for data in train_data:
            outputs = model(data)
            n_samples += outputs.shape[0]

            #Average outputs
            c += torch.sum(outputs, dim=0)
    c /= n_samples

    # If c_i is too close to 0, set to +-eps
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def train(model, train_loader, learning_rate, weight_decay, n_epochs, lr_milestones, gamma, eta, eps, reg=0):
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
    - model (NeuralNet object): Trained DeepSAD model
    - loss_history (list): List of epoch losses for analysis
    """

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=gamma)

    # Initialise c if necessary
    if model.c is None:
        model.c = init_c(model, train_loader, eps)

    # Track loss history for analysis
    loss_history = []

    # Iterate epochs to train model
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        scheduler.step()

        for index, (train_data, train_target) in enumerate(train_loader):
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
                if reg != 0:  # If we want to diversify
                    C = torch.matmul(Y, Y.T)  # Gram Matrix
                    loss_d = -torch.log(torch.det(C)) + torch.trace(C)  # Diversity loss contribution
                else:
                    loss_d = 0
                if target == 0:
                    losses = dist
                else:
                    losses = eta * ((dist + eps) ** target)
                losses += reg * loss_d

                loss += losses
                n_batches += 1

            # Finish off training the network
            loss = loss / n_batches
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        loss_history.append(epoch_loss)
        print(f"Epoch {epoch}, loss = {epoch_loss}")

    return model, loss_history


def AE_train(model, train_loader, learning_rate, weight_decay, n_epochs, lr_milestones, gamma):
    """
        Trains neural net model weights

        Parameters:
        - model (NeuralNet object): DeepSAD model
        - train_loader (DataLoader object): Data loader for training data and targets
        - learning_rate (float): Learning rate
        - weight_decay (float): Factor to reduce LR by at milestones
        - n_epochs (int): Number of epochs for training
        - lr_milestones (list): Epoch milestones to reduce learning rate
        - gamma (float): Weighting of L2 regularisation

        Returns:
        - model (NeuralNet object): Trained neural network
    """

    # Set loss
    criterion = nn.MSELoss(reduction='none')

    # Set optimizer (Adam optimizer for now)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Set learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=gamma)

    model.train()
    for epoch in range(n_epochs):
        print("Epoch " + str(epoch))
        scheduler.step()
        #if epoch in lr_milestones:
        #    print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

        epoch_loss = 0.0
        n_batches = 0
        for index, (train_data, target) in enumerate(train_loader):
            batch_loss = 0.0
            for data in train_data:
                data = data.float()
                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                rec = model(data).float()
                rec_loss = criterion(rec, data) #Not so sure about this
                loss = torch.mean(rec_loss.float())
                #print(loss)
                batch_loss += loss.item()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            #print("Batch loss: " + str(batch_loss))
        print(epoch_loss)

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

    #Create and train autoencoder
    ae_model = NeuralNet_Autoencoder(model.size)
    ae_model = AE_train(ae_model, train_loader, learning_rate, weight_decay, n_epochs, lr_milestones, gamma)

    #Create dictionaries to store network states
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
    y = torch.norm(model(X) - model.c)   #Magnitude of the vector is anomaly score
    return y


def load_data(dir, margin, filename):
    """
        Loads data from CSV files

        Parameters:
        - dir (string): Root directory of train/test data
        - margin (int): Number of data points at either end of lifespan to be labelled
        - filename (string): file name for train/test data

        Returns:
         - data (2D numpy array): list of training data vectors
         - labels (1D numpy array): artificial labels for training data
    """

    first = True    #First sample flag
    #Walk directory
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name == filename:    #If correct file to be included in training data
                read_data = np.array(pd.read_csv(os.path.join(root, name)))

                #Set data and labels arrays to data from first sample
                if first:
                    data = np.array([read_data])
                    labels = np.array([1])
                    first = False

                #Concatenate additional samples
                else:
                    data = np.concatenate((data, [read_data]))
                    labels = np.append(labels, 0)   #Default label is 0
    labels[-1*margin::] = -1    #Unhealthy labels
    labels[::margin] = 1        #Healthy labels
    return data, labels


def DeepSAD_train_run(dir, freq, filename):
    """
    Trains and runs the DeepSAD model

    Parameters:
    - dir (string): Root directory of train/test data
    - freq (string): 3-digit frequency for train/test data
    - filename (string): file name for train/test data, excluding freq, "kHz_" and .csv

    Returns:
    - results (2D numpy array): 5x30 Array of health indicators with state for each panel
    """

    # Hyperparameters
    learning_rate_AE = 0.001
    learning_rate = 0.00001
    weight_decay = 0.1
    n_epochs_AE = 10
    n_epochs = 15
    lr_milestones_AE = [20, 30, 40]  # Milestones when learning rate reduces
    lr_milestones = [5, 10, 50, 70, 90]
    gamma = 0.1  # Factor to reduce LR by at milestones
    gamma_AE = 0.1  # "
    eta = 3  # Weighting of labelled datapoints
    eps = 1 * 10 ** (-8)  # Very small number to prevent zero errors
    reg = 0.001  # Lambda - diversity weighting
    batch_size = 10
    margin = 5  # Number of samples labelled on each end
    n_iterations = 5  # Number of self-training iterations
    healthy_range = (1, 5)  # Healthy range
    unhealthy_range = (1, 3)  # Unhealthy range

    samples = ["PZT-CSV-L1-03", "PZT-CSV-L1-04", "PZT-CSV-L1-05", "PZT-CSV-L1-09", "PZT-CSV-L1-23"]

    # Make string of filename for train/test data
    filename = freq + "kHz_" + filename + ".csv"

    # Initialise results matrix
    results = np.empty((5, 30), dtype=object)

    # Loop for each sample as test data
    for sample_count in range(len(samples)):
        test_sample = samples[sample_count]

        # Make new list of samples excluding test data
        temp_samples = copy.deepcopy(samples)
        temp_samples.remove(test_sample)

        first = True  # Flag to set up final training set
        # Initialise training data lists
        for current_sample in temp_samples:
            current_data = np.loadtxt(dir + current_sample + "/" + filename, delimiter=',')
            if first:
                train_data = copy.deepcopy(current_data)
                first = False
            else:
                train_data = np.append(train_data, current_data, axis=0)

        # Load test data
        test_data = np.loadtxt(dir + test_sample + "/" + filename, delimiter=',')

        # Add columns to results matrix
        for j in range(30):
            results[sample_count, j] = [freq + "kHz", int(j + 1)]

        # Loop for each frequency band
        for i in range(30):
            train_array = copy.deepcopy(train_data)
            test_array = copy.deepcopy(test_data)
            train_array = np.delete(train_array, np.s_[0:i * 160:1], axis=1)
            train_array = np.delete(train_array, np.s_[160:30 * 160 - i * 160:1], axis=1)
            test_array = np.delete(test_array, np.s_[0:i * 160:1], axis=1)
            test_array = np.delete(test_array, np.s_[160:30 * 160 - i * 160:1], axis=1)

            # Initialize the DeepSAD model and create DataLoader
            model = NeuralNet(train_array.shape[1])
            train_loader = DataLoader(TensorDataset(torch.tensor(train_array, dtype=torch.float32),
                                                    torch.tensor(test_array, dtype=torch.float32)),
                                      batch_size=batch_size, shuffle=True)

            # Self-training the DeepSAD model
            model, label_history = self_train(model, train_loader, n_iterations, healthy_range, unhealthy_range,
                                              learning_rate, weight_decay, n_epochs, lr_milestones, gamma, eta, eps,
                                              reg)

            # Print the final best-fit labels
            print(f"Final best-fit labels for frequency band {i + 1}: {label_history[-1]}")

            # Evaluate the model
            test_loader = DataLoader(torch.tensor(test_array, dtype=torch.float32), batch_size=batch_size,
                                     shuffle=False)
            model.eval()
            for index, test_sample in enumerate(test_loader):
                output = embed(test_sample, model)
                # Save the output results (predictions) in the results array
                results[sample_count, i].append(output.detach().numpy())

    return results


def update_labels(model, train_loader, healthy_range, unhealthy_range):
    """
    Update labels based on model predictions.

    Parameters:
    - model (NeuralNet object): Trained DeepSAD model.
    - train_loader (DataLoader object): Training data loader.
    - healthy_range (tuple): Range of values considered healthy.
    - unhealthy_range (tuple): Range of values considered unhealthy.

    Returns:
    - updated_labels (1D numpy array): Updated labels for the training data.
    """
    model.eval()
    updated_labels = []
    with torch.no_grad():
        for data, target in train_loader:
            for x in data:
                score = embed(x, model).item()
                if healthy_range[0] <= score <= healthy_range[1]:
                    updated_labels.append(1)  # Healthy
                elif unhealthy_range[0] <= score <= unhealthy_range[1]:
                    updated_labels.append(-1)  # Unhealthy
                else:
                    updated_labels.append(0)  # Uncertain / Unlabeled
    return np.array(updated_labels)


def self_train(model, train_loader, n_iterations, healthy_range, unhealthy_range, learning_rate, weight_decay, n_epochs,
               lr_milestones, gamma, eta, eps, reg):
    """
    Self-training loop for DeepSAD model.

    Parameters:
    - model (NeuralNet object): DeepSAD model
    - train_loader (DataLoader object): Training data loader
    - n_iterations (int): Number of self-training iterations
    - healthy_range (tuple): Range of values considered healthy
    - unhealthy_range (tuple): Range of values considered unhealthy
    - learning_rate (float): Learning rate
    - weight_decay (float): Factor to reduce LR by at milestones
    - n_epochs (int): Number of epochs for training
    - lr_milestones (list): Epoch milestones to reduce learning rate
    - gamma (float): Weighting of L2 regularisation
    - eta (float): Weighting of labelled data points
    - eps (float): Small number to prevent zero errors
    - reg (float): Weighting of diversity loss function

    Returns:
    - model (NeuralNet object): Trained DeepSAD model
    """

    for iteration in range(n_iterations):
        print(f"Self-training iteration {iteration + 1}/{n_iterations}")

        # Train the model
        model, _ = train(model, train_loader, learning_rate, weight_decay, n_epochs, lr_milestones, gamma, eta, eps,
                         reg)

        # Update labels
        updated_labels = update_labels(model, train_loader, healthy_range, unhealthy_range)

        # Print the updated labels
        print(f"Updated labels after iteration {iteration + 1}: {updated_labels}")

        # Update train_loader with new labels
        train_data, _ = next(iter(train_loader))
        train_loader = DataLoader(TensorDataset(train_data, torch.tensor(updated_labels)),
                                  batch_size=train_loader.batch_size, shuffle=True)

    return model
