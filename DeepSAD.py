import torch
from torch import nn

class DeepSAD_net(nn.Module):
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
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=gamma)

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

        for data in training_data:  #Batches training data?

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