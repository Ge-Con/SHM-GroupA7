import numpy as np
import torch
import scipy

def train_regressor_nn(n_features, n_hidden_neurons, learning_rate, n_epochs, X, Y):

    # Define the model:
    model = torch.nn.Sequential(
          torch.nn.Linear(n_features,n_hidden_neurons),
          torch.nn.Sigmoid(),
          torch.nn.Linear(n_hidden_neurons,n_hidden_neurons),
          torch.nn.Sigmoid(),
          torch.nn.Linear(n_hidden_neurons,1),
        )

    # MSE loss function:


    # optimizer:

    # Train the network:

    # return the trained model
    return model