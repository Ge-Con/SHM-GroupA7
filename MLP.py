import numpy as np
import torch
import scipy

def train_regressor_nn(n_features, n_hidden_neurons, learning_rate, n_epochs, X):

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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the network:
    for count in range(n_epochs):
        loss = loss_fn(model(X))
        # Backward pass and optimization:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # return the trained model
    return model