import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn.aggr import AttentionalAggregation
from torch.nn import Linear
from tqdm import tqdm

class lGNN(torch.nn.Module):
    """
    Lean Graph Neural Network (lGNN) -> see https://doi.org/10.1007/s44210-022-00006-4
    -------
    If architecture is not supplied and trained state is, the architecture is loaded from the trained state.
    """
    def __init__(self,arch=None, trained_state=None):
        super(lGNN, self).__init__() 
        device = torch.device('cpu')
        if arch == None and trained_state != None:
            arch = trained_state['arch']
            del trained_state['arch']
 
        self.out_dim = 1
        self.act = arch['act']
        self.att = AttentionalAggregation(Linear(arch['conv_dim'], 1))

        ## Set up gated graph convolution
        self.conv = GatedGraphConv(out_channels=arch['conv_dim'], aggr='add', num_layers=arch['n_conv_layers'])

        ## Set up hidden layers
        self.fc_list, self.fcbn_list = torch.nn.ModuleList(), torch.nn.ModuleList()
        if arch['n_hidden_layers'] > 0:
             for i in range(arch['n_hidden_layers']):
                self.fc_list.append(Linear(arch['conv_dim'], arch['conv_dim']))
    
        ## Final output layer
        self.lin_out = Linear(arch['conv_dim'], self.out_dim)
        
        device = torch.device('cpu')
        self.to(device)

        # if trained state is provided fetch onehot labels (required in surface.py)
        if trained_state != None:
            self.onehot_labels = trained_state['onehot_labels']
            del trained_state['onehot_labels']
            self.load_state_dict(trained_state)

    def forward(self, data):
        """
        Forward pass of the model.
        """
        out = data.x

        # Gated Graph Convolutions
        out = self.conv(out, data.edge_index)

        ## Global attention pooling
        out = self.att.forward(out, data.batch)

        ## Hidden layers
        for i in range(0, len(self.fc_list)):
            out = self.fc_list[i](out)
            out = getattr(torch.nn.functional, self.act)(out)

        # Output layer
        out = self.lin_out(out)

        if out.shape[1] == 1:
            return out.view(-1)
        else:
            return out

    def train4epoch(self, loader, batch_size, optimizer):
        """
        Train the model for a single epoch.
        ------
        Model weights are updated according to the mean squared error but the mean absolute error is returned.
        """

        self.train()

        pred, target = [], []

        for data in loader:

            optimizer.zero_grad()

            # predict
            predicted = self(data)

            # compute mean squared error
            loss = torch.nn.MSELoss()(predicted.reshape([batch_size]), data.y)

            # update step
            loss.backward()
            optimizer.step()

            pred += predicted.reshape([batch_size]).tolist()
            target += data.y.tolist()

        # return mean absolute error
        L1Loss = np.mean(abs(np.array(pred) - np.array(target)))

        return L1Loss 

    def test(self, loader, batch_size):
        """
        Predict on provided dataloader.
        
        Returns
        ------
        L1Loss
            Mean absolute error
        pred
            Predicted values
        target
            True values
        ads
            Adsorbate type
        """
        self.eval()
        pred = []
        target = []
        site = []
        ads = []

        for data in loader:
            pred += self(data).reshape([batch_size]).tolist()
            target += data.y.tolist()
            ads += data.ads

        L1Loss = abs(np.array(pred) - np.array(target)).mean()
        return L1Loss, pred, target, ads

    def predict(self, graphs, tqdm_bool=True):
        """
        Predict on provided list of graphs.
        """
        self.eval()
        loader = DataLoader(graphs, batch_size=256)
        pred = []
        for data in tqdm(loader,total=len(loader), disable=not tqdm_bool):
            pred += self(data).reshape([len(data)]).tolist()
        return np.array(pred)

