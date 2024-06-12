import numpy as np
import torch
import itertools
import copy
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GatedGraphConv #, GlobalAttention
from torch_geometric.nn.aggr import AttentionalAggregation as GlobalAttention
from torch.nn import Linear
import torch.nn.functional as F
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
        self.att = GlobalAttention(Linear(arch['conv_dim'], 1))

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
                out = getattr(F, self.act)(out)

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


def lGNNold(arch=None, trained_state=None):
    class lGNN(torch.nn.Module):
        def __init__(self,
                     n_conv_layers,
                     n_hidden_layers,
                     conv_dim,
                     act,
                    ):
            super(GCNN, self).__init__()
            
            self.out_dim = 1
            self.act = act
            self.att = GlobalAttention(Linear(conv_dim, 1))

            ## Set up gated graph convolution
            self.conv = GatedGraphConv(out_channels=conv_dim, aggr='add', num_layers=n_conv_layers)

            ## Set up hidden layers
            self.fc_list, self.fcbn_list = torch.nn.ModuleList(), torch.nn.ModuleList()
            if n_hidden_layers > 0:
                for i in range(n_hidden_layers):
                    self.fc_list.append(Linear(conv_dim, conv_dim))

            ## Final output layer int?
            self.lin_out = Linear(conv_dim, self.out_dim)

            self.onehot_labels = []

        def forward(self, data):
            out = data.x

            # Gated Graph Convolutions
            out = self.conv(out, data.edge_index)

            ## Global attention pooling
            out = self.att.forward(out, data.batch)

            ## Hidden layers
            for i in range(0, len(self.fc_list)):
                out = self.fc_list[i](out)
                out = getattr(F, self.act)(out)

            # Output layer
            out = self.lin_out(out)

            if out.shape[1] == 1:
                return out.view(-1)
            else:
                return out
    
    device = torch.device('cpu')
    
    if arch == None and trained_state != None:
        arch = trained_state['arch']
        del trained_state['arch']
        model = lGNN(**arch).to(device)
        model.onehot_labels = trained_state['onehot_labels']
        del trained_state['onehot_labels']
        model.load_state_dict(trained_state)
    
    else: model = lGNN(**arch).to(device)
    
    return model

def train(model, loader, batch_size, optimizer):
	model.train()

	loss_all = 0
	count = 0
	pred, target = [], []

	for data in loader:

		optimizer.zero_grad()

		# predict
		predicted = model(data)

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

def test(model, loader, batch_size):
	model.eval()
	pred = []
	target = []
	site = []
	ads = []

	for data in loader:
		pred += model(data).reshape([batch_size]).tolist()
		target += data.y.tolist()
		#site += data.site
		ads += data.ads

	L1Loss = abs(np.array(pred) - np.array(target)).mean()
	return L1Loss, pred, target, ads

def predict(model, loader, batch_size):
	model.eval()
	pred = []
	site = []
	ads = []

	for data in loader:
		pred += model(data).reshape([batch_size]).tolist()
		#site += data.site
		#ads += data.ads

		return pred
