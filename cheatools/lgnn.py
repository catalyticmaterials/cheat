import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn.aggr import AttentionalAggregation
from torch.nn import Linear
from tqdm import tqdm

class lGNN(torch.nn.Module):
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
    
        ## Final output layer?
        self.lin_out = Linear(arch['conv_dim'], self.out_dim)
        
        device = torch.device('cpu')
        self.to(device)
        if trained_state != None:
            self.onehot_labels = trained_state['onehot_labels']
            del trained_state['onehot_labels']
            self.load_state_dict(trained_state)

    def forward(self, data):
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
        self.train()

        loss_all = 0
        count = 0
        pred, target = [], []

        for data in loader:

            optimizer.zero_grad()

            # predict
            predicted = self(data)

            # compute loss
            loss = torch.nn.MSELoss()(predicted.reshape([batch_size]), data.y)

            loss_all += loss
            count += 1

            # update step
            loss.backward()
            optimizer.step()

            pred += predicted.reshape([batch_size]).tolist()
            target += data.y.tolist()

        L1Loss = np.mean(abs(np.array(pred) - np.array(target)))

        return L1Loss 

    def test(self, loader, batch_size):
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
        self.eval()
        loader = DataLoader(graphs, batch_size=256)
        pred = []
        for data in tqdm(loader,total=len(loader), disable=not tqdm_bool):
            pred += self(data).reshape([len(data)]).tolist()
        return np.array(pred)

