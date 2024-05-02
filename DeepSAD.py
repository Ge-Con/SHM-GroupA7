import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import copy

class FashionMNIST_LeNet(nn.Module):
    #Object for the neural network model for DeepSAD
    #mnist_LeNet implementation - this is phi in the equations

    def __init__(self):
        super().__init__()
        self.c = None           #Hypersphere centre

        #CNN
        self.fc1 = nn.Linear(56*71, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        #34 rows, 71 columns
        x = torch.flatten(x)
        x = x.to(next(self.parameters()).dtype)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

class FashionMNIST_LeNet_Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(512, 56*71)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(1, 64)

    def forward(self, x):
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc2(x))
        x = self.fc1(x)
        decoded = x.view(-1, 56, 71)
        return x

class FashionMNIST_LeNet_Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = FashionMNIST_LeNet()
        self.decoder = FashionMNIST_LeNet_Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def init_c(model, train_data, eps=0.1):
    """
        Initialise hypersphere center c as the mean from an initial forward pass on the data

        Parameters:
        - model (DeepSAD_net object): Untrained DeepSAD model
        - train_data (2D array): Training data
        - eps (float): Hyperparameter

        Returns:
        - c (1D array): Coordinates of hypersphere centre
    """

    n_samples = 0
    c = torch.zeros(model.rep_dim)

    #Forward pass
    model.eval()
    with torch.no_grad():
        for data in train_data:
            outputs = model(data)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)
    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def train(model, train_data, semi_targets, learning_rate, weight_decay, n_epochs, lr_milestones, gamma, eta, eps, reg=0):
    """
        Train the DeepSAD model from a semi-labelled dataset.

        Parameters:
        - model (DeepSAD_net object): DeepSAD model
        - train_data (2D array): Training data
        - semi_targets (yet to be implemented): Data labels
        - learning_rate (float): Learning rate
        - weight_decay (float): Weight decay for Adam optimizer
        - n_epochs (int): Number of epochs
        - lr_milestoned (int): Milestones for learning rate updates
        - gamma (float): Gamma
        - eta (float): eta
        - eps (float): eps <^- hyperparameters
        - reg(int): Lambda, weight of diversification in loss function (zero if none)

        Returns:
        None
    """

    #Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=gamma)

    #Initialise c if necessary
    if model.c == None:
        model.c = init_c(model, train_data, eps)

    #Iterate epochs to train model
    model.train()
    for epoch in range(n_epochs):
        scheduler.step()

        if epoch in lr_milestones:  #Report new learning rate on milestones
            print("\tNew learning rate is " + str(float(scheduler.get_lr()[0])))
        epoch_loss = 0.0
        n_batches = 0

        for data in train_data:  #Batches training data?

            #Forward and backward pass
            optimizer.zero_grad()
            outputs = model(data)

            #Calculating loss function
            Y = outputs - model.c
            dist = torch.sum(Y ** 2, dim=1)
            if reg != 0:  # If we want to diversify
                C = torch.matmul(Y, Y.T)  # Gram Matrix
                loss_d = -torch.log(torch.det(C)) + torch.trace(C)  # Diversity loss contribution
            else:
                loss_d = 0
            losses = torch.where(semi_targets == 0, dist, eta * ((dist + eps) ** semi_targets.float()))
            losses += reg * loss_d

            # Finish off training the network
            loss = torch.mean(losses)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        print("Epoch " + str(epoch) + ", loss = " + str(epoch_loss/n_batches))
    return model

def AE_train(model, train_data, learning_rate, weight_decay, n_epochs, lr_milestones):
    # Set loss
    criterion = nn.MSELoss(reduction='none')

    # Set optimizer (Adam optimizer for now)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Set learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

    model.train()
    for epoch in range(n_epochs):
        scheduler.step()
        if epoch in lr_milestones:
            print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_loss = 0.0
            n_batches = 0
            for batch in train_data:
                data = batch

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                rec = model(data)
                rec_loss = criterion(rec, data) #Not so sure about this
                loss = torch.mean(rec_loss)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

    return model

def pretrain(model, train_data, learning_rate, weight_decay, n_epochs, lr_milestones):
    ae_model = FashionMNIST_LeNet_Autoencoder()

    ae_model = AE_train(ae_model, train_data, learning_rate, weight_decay, n_epochs, lr_milestones)

    model_dict = model.state_dict()
    ae_model_dict = ae_model.state_dict()

    # Filter out decoder network keys
    ae_model_dict = {k: v for k, v in ae_model_dict.items() if k in model_dict}
    # Overwrite values in the existing state_dict
    model_dict.update(ae_model_dict)
    # Load the new state_dict
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

        Example:

        # Example usage of the function
        HI = Time_domain_features(feature_data, model)
    """

    model.eval()
    y = torch.norm(model(X) - model.c)   #Magnitude of the vector = anomaly score
    return y

learning_rate = 0.1
weight_decay = 0.1
n_epochs = 1000
lr_milestones = [500]
gamma = 1
eta = 1
eps = 1
reg = 0.2

def batch_data(data, batch_size):
    num_samples = len(data)
    num_batches = num_samples // batch_size
    batches = []

    for i in range(num_batches):
        batch = data[i * batch_size: (i + 1) * batch_size]
        batches.append(batch)

    # Handling the last batch which may have a different size
    if num_samples % batch_size != 0:
        batches.append(data[num_batches * batch_size:])

    return batches

def load_data(dir, margin):
    first = True
    count = 0
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name == '050_kHz-allfeatures.csv':
                read_data = np.array(pd.read_csv(os.path.join(root, name)))
                if first:
                    data = np.array([read_data])
                    labels = np.array([np.array([1])])  #Healthy
                    first = False
                else:
                    data = np.concatenate((data, [read_data]))
                    labels = np.concatenate((labels, [np.array([0])]))    #Unlabelled
                count += 1
    labels[-1*margin::] = np.array([-1])    #Unhealthy
    return data, labels

batches = 4 #Batch ->size<-!!
margin = 5 #Number of samples labelled on each end

samples = ["PZT-CSV L1-03", "PZT-CSV L1-05", "PZT-CSV L1-09"]

dir = input("CSV file location: ")

first = True
for sample in samples:
    temp_data, temp_targets = load_data(dir + "\\" + sample, margin)
    if first:
        train_data = copy.deepcopy(temp_data)
        semi_targets = copy.deepcopy(temp_targets)
        first = False
    else:
        train_data = np.concatenate((train_data, temp_data))
        semi_targets = np.concatenate((semi_targets, temp_targets))

#train_data = batch_data(train_data, batches)
batched_data = []
for batch in range(len(train_data)):
    batched_data.append(torch.from_numpy(train_data[batch]))   #x = x.detach().numpy()
model = FashionMNIST_LeNet()
model = pretrain(model, batched_data, learning_rate, weight_decay, n_epochs, lr_milestones)
model = train(model, batched_data, semi_targets, learning_rate, weight_decay, n_epochs, lr_milestones, gamma, eta, eps, reg)