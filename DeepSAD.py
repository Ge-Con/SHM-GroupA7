import torch
from torch import nn

class model(nn.Module):
    #mnist_LeNet implementation - this is phi in the equations

    def __init__(self, rep_dim=64):
        super().__init__()
        self.c = None
        self.rep_dim = rep_dim
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
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = torch.zeros(model.rep_dim)

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

def train(model, train_data, semi_targets, learning_rate, weight_decay, lr_milestones, n_epochs, gamma, eta, eps, reg=0):
    #reg is 0 if no diversification

    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=gamma)
    if model.c == None:
        model.c = init_c(model, train_data, eps)
    model.train()
    for epoch in range(n_epochs):
        scheduler.step()

        if epoch in lr_milestones:
            print("\tNew learning rate is " + str(float(scheduler.get_lr()[0])))
        epoch_loss = 0.0
        n_batches = 0

        for data in training_data:

            optimizer.zero_grad()
            outputs = model(data)

            Y = outputs - model.c
            dist = torch.sum(Y ** 2, dim=1)
            if reg != 0:  # If we want to diversify
                C = torch.matmul(Y, Y.T)  # Gram Matrix
                loss_d = -torch.log(torch.det(C)) + torch.trace(C)  # Diversity loss contribution
            else:
                loss_d = 0
            losses = torch.where(semi_targets == 0, dist, eta * ((dist + eps) ** semi_targets.float()))
            losses += reg * loss_d

            loss = torch.mean(losses)  # Finish off training the network
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        print("Epoch " + str(epoch) + ", loss = " + str(epoch_loss/n_batches))

def embed(self, X, model):
    #Get Health Indicator
    model.eval()
    y = torch.norm(model(X) - model.c)   #Magnitude of the vector = anomaly score
    return y