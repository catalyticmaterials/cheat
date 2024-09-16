import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
     
        # Encoder
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean of the latent space
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Log variance of the latent space
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4_mu = nn.Linear(hidden_dim, output_dim)       # Mean of the output
        self.fc4_logvar = nn.Linear(hidden_dim, output_dim)   # Log variance of the output
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
 
    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return self.fc4_mu(h3), self.fc4_logvar(h3)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_mu, recon_logvar, y_true, mu, logvar):
        # Convert log variance to variance
        recon_var = torch.exp(recon_logvar)

        # GaussianNLLLoss expects variance, not log variance
        nll_loss = nn.GaussianNLLLoss(full=True, reduction='sum')
        reproduction_loss = nll_loss(recon_mu, y_true, recon_var)

        # KL divergence loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return reproduction_loss + KLD

    def train4epochs(self, trainloader, valloader, optimizer, epochs, verbosity=10):
        
        best_state, best_loss, n_epoch = None, np.inf, 0 
        for epoch in range(epochs):
            self.train()
            overall_loss = 0
            for batch_idx, (x, y) in enumerate(trainloader):
                batchsize = len(x)
                x = x.view(batchsize, self.input_dim).to(self.device)
                (recon_mu, recon_logvar), mu, logvar = self.forward(x)

                # Calculate loss
                loss = self.loss_function(recon_mu, recon_logvar, y, mu, logvar)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                overall_loss += loss.item()
            
            self.eval()
            val_loss = 0
            for x, y in valloader:
                (recon_mu, recon_logvar), mu, logvar = self.forward(x)
                loss = self.loss_function(recon_mu, recon_logvar, y, mu, logvar)         
                val_loss += loss

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = self.state_dict()
                n_epoch = epoch
            
            if (epoch+1) % verbosity == 0:
                print(f"\tEpoch {epoch+1 :4d}:   {val_loss/(batch_idx*batchsize)}")
        
        print(f"Best validation loss was achieved after epoch {n_epoch+1} with a loss of {best_loss/(batch_idx*batchsize):.2f}")
 
        return best_state

    def predict(self, dataloader):
        self.eval()
        
        meanArr, stdArr, trueArr = [], [], []
        for x, y in dataloader:
            (mean, log_var), _, _ = self.forward(x)
            meanArr.append(mean.detach().numpy()[0])
            stdArr.append(torch.exp(0.5 * log_var).detach().numpy()[0])
            trueArr.append(y.detach().numpy()[0])

        return np.array(meanArr), np.array(stdArr), np.array(trueArr)
