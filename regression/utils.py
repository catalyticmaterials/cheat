import numpy as np
import torch
from torch_geometric.nn import GatedGraphConv, GlobalAttention
from torch.nn import Linear
import torch.nn.functional as F

def load_GCN(kwargs_dict, trained_state=None):
	class GCNN(torch.nn.Module):
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

			## Final output layer
			self.lin_out = Linear(int(conv_dim), self.out_dim)

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

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = GCNN(**kwargs_dict).to(device)

	if trained_state != None:
		model.load_state_dict(trained_state)
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
		site += data.site
		ads += data.ads

	L1Loss = abs(np.array(pred) - np.array(target)).mean()
	return L1Loss, pred, target, site, ads
def predict(model, loader, batch_size):
	model.eval()
	pred = []
	site = []
	ads = []

	for data in loader:
		pred += model(data).reshape([batch_size]).tolist()
		site += data.site
		ads += data.ads

		return pred, site, ads