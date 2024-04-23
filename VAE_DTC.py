import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(
          self, 
          x_dim,
          hidden_dim,
          z_dim=1
        ):
        super(VAE, self).__init__()

        self.prev = 0

        # Define autoencoding layers
        self.enc_layer1 = nn.Sigmoid(x_dim, hidden_dim)
        self.enc_layer2_mu = nn.Sigmoid(hidden_dim, z_dim)
        self.enc_layer2_logvar = nn.Sigmoid(hidden_dim, z_dim)

        # Define autoencoding layers
        self.dec_layer1 = nn.Sigmoid(z_dim, hidden_dim)
        self.dec_layer2 = nn.Sigmoid(hidden_dim, x_dim) 

    def encoder(self, x):
        x = F.relu(self.enc_layer1(x))
        mu = F.relu(self.enc_layer2_mu(x))
        logvar = F.relu(self.enc_layer2_logvar(x))
        return mu, logVar

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def decoder(self, z):
        # Define decoder network
        output = F.relu(self.dec_layer1(z))
        output = F.relu(self.dec_layer2(output))
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        output = self.decoder(z)
        z_i1 = self.prev
        self.prev = z
        return output, z, mu, logvar, z_i1 

# Define the loss function
def loss_function(output, x, mu, logvar, z, z_i1):
    recon_loss = F.mse_loss(output, x, reduction='sum') / batch_size
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    DTC = math.sqrt(z - z_i1 - 9)
        
    return recon_loss + 0.002  * kl_loss + DTC


def train_model(
    X, 
    learning_rate=1e-4, 
    batch_size=128, 
    num_epochs=15,
    hidden_dim=20,
    latent_dim=1
  ):
  # Define the VAE model
  model = VAE(x_dim=X.shape[1], hidden_dim=hidden_dim, z_dim=latent_dim)

  # Define the optimizer
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  
  # Convert X to a PyTorch tensor
  X = torch.tensor(X).float()

  # Create DataLoader object to generate minibatches
  dataset = TensorDataset(X)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
  
  # Train the model
  for epoch in range(num_epochs):
      model.prev = 0
      epoch_loss = 0
      for batch in dataloader:
          # Zero the gradients
          optimizer.zero_grad()

          # Get batch
          x = batch[0]

          # Forward pass
          output, z, mu, logvar, z_i1 = model(x)

          # Calculate loss
          loss = loss_function(output, x, mu, logvar, z, z_i1)

          # Backward pass
          loss.backward()

          # Update parameters
          optimizer.step()

          # Add batch loss to epoch loss
          epoch_loss += loss.item()

      # Print epoch loss
      print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(X)}")
      
  return model
