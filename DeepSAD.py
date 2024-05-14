import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import copy
import matplotlib.pyplot as plt

from prognosticcriteria import Mo, Pr, Tr



samples = ["PZT-CSV L1-03", "PZT-CSV L1-04", "PZT-CSV L1-05", "PZT-CSV L1-09", "PZT-CSV L1-23"]


class FashionMNIST_LeNet(nn.Module):
    #Object for the neural network model for DeepSAD
    #mnist_LeNet implementation - this is phi in the equations

    def __init__(self, size):
        super().__init__()
        self.c = None           #Hypersphere centre
        self.size = size

        #CNN
        self.fc1 = nn.Linear(size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 16)
        self.m = torch.nn.LeakyReLU(0.001)

    def forward(self, x):
        #34 rows, 71 columns
        x = torch.flatten(x)
        x = x.to(next(self.parameters()).dtype)
        x = self.m(self.fc1(x))
        x = self.m(self.fc2(x))
        x = self.m(self.fc3(x))
        x = self.m(self.fc4(x))
        x = self.m(self.fc5(x))
        encoded = x.view(4, 4)
        return encoded

class FashionMNIST_LeNet_Decoder(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.fc1 = nn.Linear(1024, size)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(16, 64)
        self.m = torch.nn.LeakyReLU(0.001)

    def forward(self, x):
        x = torch.flatten(x)
        x = self.m(self.fc5(x))
        x = self.m(self.fc4(x))
        x = self.m(self.fc3(x))
        x = self.m(self.fc2(x))
        x = self.m(self.fc1(x))
        decoded = x.view(-1, 19, 56)
        return decoded

class FashionMNIST_LeNet_Autoencoder(nn.Module):

    def __init__(self, size):
        super().__init__()

        self.encoder = FashionMNIST_LeNet(size)
        self.decoder = FashionMNIST_LeNet_Decoder(size)

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
    c = torch.zeros((4, 4))

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


def train(model, train_data, train_loader, learning_rate, weight_decay, n_epochs, lr_milestones, gamma, eta, eps, reg=0):
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
        epoch_loss = 0.0
        scheduler.step()

        #if epoch in lr_milestones:  #Report new learning rate on milestones
        #    print("\tNew learning rate is " + str(float(scheduler.get_lr()[0])))

        for index, (train_data, train_target) in enumerate(train_loader):
            loss = 0.0
            n_batches = 0
            for index in range(len(train_data)):
                data = train_data[index]
                target = train_target[index]

                #Forward and backward pass
                optimizer.zero_grad()
                outputs = model(data)

                #Calculating loss function
                Y = outputs - model.c
                dist = torch.sum(Y ** 2)
                if reg != 0:  # If we want to diversify
                    C = torch.matmul(Y, Y.T)  # Gram Matrix
                    loss_d = -torch.log(torch.det(C)) + torch.trace(C)  # Diversity loss contribution
                else:
                    loss_d = 0
                if target == 0:#semi_targets[index] == 0:
                    losses = dist
                else:
                    losses = eta * ((dist + eps) ** target)#semi_targets[index])
                #losses = torch.where(semi_targets[index] == 0, dist, eta * ((dist + eps) ** semi_targets[index]))
                losses += reg * loss_d

                loss += losses
                n_batches += 1

            # Finish off training the network
            loss = loss/n_batches
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            #print("Batch loss: " + str(loss))

        print("Epoch " + str(epoch) + ", loss = " + str(epoch_loss))
    return model

def AE_train(model, train_loader, learning_rate, weight_decay, n_epochs, lr_milestones):
    # Set loss
    criterion = nn.MSELoss(reduction='none')

    # Set optimizer (Adam optimizer for now)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Set learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

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

def pretrain(model, train_loader, learning_rate, weight_decay, n_epochs, lr_milestones):
    ae_model = FashionMNIST_LeNet_Autoencoder(model.size)

    ae_model = AE_train(ae_model, train_loader, learning_rate, weight_decay, n_epochs, lr_milestones)

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
    #print(model(X))
    y = torch.norm(model(X) - model.c)   #Magnitude of the vector = anomaly score
    return y


def load_data(dir, margin, filename):
    first = True
    count = 0
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name == filename:
                read_data = np.array(pd.read_csv(os.path.join(root, name)))
                if first:
                    data = np.array([read_data])
                    #labels = np.array([np.array([1])])  #Healthy
                    labels = np.array([1])
                    first = False
                else:
                    data = np.concatenate((data, [read_data]))
                    #labels = np.concatenate((labels, [np.array([0])]))    #Unlabelled
                    labels = np.append(labels, 0)
                count += 1
    labels[-1*margin::] = -1 #np.array([-1])    #Unhealthy
    labels[::margin] = 1
    return data, labels



#train_data = []
#semi_targets = []
#for i in range(len(arr_data)):
#    train_data.append(torch.from_numpy(arr_data[i]))   #x = x.detach().numpy()
#    semi_targets.append(torch.from_numpy(np.array(arr_targets[i])))

#m = Mo(all)
#p = Pr(all)
#t = Tr(all)
#print(m, p, t)
#print(m+p+t)

"""scores = []

for reg10 in range(10):
    eta = 1+reg10/10

    model = FashionMNIST_LeNet()
    model = pretrain(model, train_loader, learning_rate_AE, weight_decay, n_epochs_AE, lr_milestones_AE)
    model = train(model, train_data, train_loader, learning_rate, weight_decay, n_epochs, lr_milestones, gamma, eta, eps, reg)

    all = np.empty((len(test_data), 30))
    count = 0

    for sample in test_data:
        results = []
        for state in range(sample.shape[0]):
            data = sample[state]
            results.append(embed(torch.from_numpy(data), model).item())
        #print(results)
        #plt.plot(range(sample.shape[0]), results)
        #plt.show()
        all[count] = results[-30::]
        count += 1
    m = Mo(all)
    p = Pr(all)
    t = Tr(all)
    print(m, p, t)
    print(m+p+t)

    scores.append(m+p+t)
plt.plot(range(10), scores)
plt.show()"""

def DeepSAD_train_run(dir):

    # Hyperparamters
    learning_rate_AE = 0.00001
    learning_rate = 0.00001
    weight_decay = 0.1
    n_epochs_AE = 10
    n_epochs = 50
    lr_milestones_AE = [10, 20, 30, 40]  # Milestones when learning rate reduces
    lr_milestones = [10, 30, 50, 70, 90]
    gamma = 0.4  # L2 weighting to prevent large nodes
    eta = 2  # Weighting of labelled datapoints
    eps = 1 * 10 ** (-8)  # Very small number to prevent zero errors
    reg = 0.001  # Lambda - diversity weighting
    batch_size = 10
    margin = 5  # Number of samples labelled on each end

    filename = input("Please enter file name for training: ")#e.g. '050kHz-Features.csv'

    test_data = np.empty((len(samples)), dtype=object)
    first = True
    count = 0
    for sample in samples:
        temp_data, temp_targets = load_data(dir + "\\" + sample, margin, filename)
        if first:
            arr_data = copy.deepcopy(temp_data)
            arr_targets = copy.deepcopy(temp_targets)
            first = False
        else:
            arr_data = np.concatenate((arr_data, temp_data))
            arr_targets = np.concatenate((arr_targets, temp_targets))
        test_data[count] = temp_data
        count += 1

    train_data = torch.tensor(arr_data)
    semi_targets = torch.tensor(arr_targets)

    size = train_data.shape[1] * train_data.shape[2]

    train_dataset = TensorDataset(train_data, semi_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = FashionMNIST_LeNet(size)
    model = pretrain(model, train_loader, learning_rate_AE, weight_decay, n_epochs_AE, lr_milestones_AE)
    model = train(model, train_data, train_loader, learning_rate, weight_decay, n_epochs, lr_milestones, gamma, eta, eps, reg)

    all = np.empty((len(test_data), 30))
    count = 0

    for sample in test_data:
        results = []
        for state in range(sample.shape[0]):
            data = sample[state]
            results.append(embed(torch.from_numpy(data), model).item())
        # print(results)
        plt.plot(range(sample.shape[0]), results)
        plt.show()
        all[count] = results[-30::]
        count += 1

    return all