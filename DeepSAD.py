import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os

class FashionMNIST_LeNet(nn.Module):
    #Object for the neural network model for DeepSAD
    #mnist_LeNet implementation - this is phi in the equations

    def __init__(self, rep_dim=64):
        super().__init__()
        self.c = None           #Hypersphere centre
        self.rep_dim = rep_dim  #Number of dimensions

        #CNN
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(int(x.size(0)), -1)
        x = self.fc1(x)
        return x

class FashionMNIST_LeNet_Decoder(nn.Module):
    def __init__(self, rep_dim=64):
        super().__init__()

        self.rep_dim = rep_dim

        self.fc3 = nn.Linear(self.rep_dim, 128, bias=False)
        self.bn1d2 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.deconv1 = nn.ConvTranspose2d(8, 32, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 5, bias=False, padding=3)
        self.bn2d4 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(16, 1, 5, bias=False, padding=2)

    def forward(self, x):
        x = self.bn1d2(self.fc3(x))
        x = x.view(int(x.size(0)), int(128 / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv3(x)
        x = torch.sigmoid(x)
        return x

class FashionMNIST_LeNet_Autoencoder(nn.Module):

    def __init__(self, rep_dim=64):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = FashionMNIST_LeNet(rep_dim=rep_dim)
        self.decoder = FashionMNIST_LeNet_Decoder(rep_dim=rep_dim)

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
    optimizer = nn.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Set learning rate scheduler
    scheduler = nn.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

    model.train()
    for epoch in range(n_epochs):

        scheduler.step()
        if epoch in lr_milestones:
            print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_loss = 0.0
            n_batches = 0
            for data in train_data:
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
semi_targets = []
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

def load_data(dir):
    data = np.empty((1), dtype=object)
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name == '050_kHz-allfeatures.csv':
                read_data = np.array(pd.read_csv(os.path.join(root, name)))
                if str(type(data[0])) == "<class 'NoneType'>":
                    data = np.array([read_data])
                else:
                    data= np.vstack([data, read_data])
    return data

dir = input("CSV file location: ")
train_data = load_data(dir)
print(train_data)
train_data = batch_data(train_data, 4)
model = FashionMNIST_LeNet()
model = pretrain(model, train_data, learning_rate, weight_decay, n_epochs, lr_milestones)
model = train(model, train_data, semi_targets, learning_rate, weight_decay, n_epochs, lr_milestones, gamma, eta, eps, reg)