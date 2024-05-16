import numpy as np
import torch
import itertools
import copy
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
def split_ensembles(dataset,n_metals):
	"""Splits dataset into ensembles.
	Accepts numpy array of zone counts and
	needs integer of number of different metals

	Returns numpy array of arrays containing samples of each unique ensemble configuration"""

	# Placeholder for unique ensemble vectors
	ens_vector = []

	# Find number of atoms in ensemble zone (ontop=1 / bridge=2 / fcc or hcp=3)
	n_ens_atoms = sum(dataset[0, :][:n_metals])

	# Find all possible combinations, remove vectors with more than n_ens_atoms, permutate and find unique vectors.
	# This will give all unique ensemble zone configurations.
	combs = np.array(list(itertools.combinations_with_replacement(np.arange(n_ens_atoms + 1), n_metals)))
	mask = np.sum(combs, axis=1) == n_ens_atoms
	for comb in combs[mask]:
		for unique in np.unique(np.array(list(itertools.permutations(comb))), axis=0):
			# Append to list of unique ensemble vectors for comparison
			ens_vector.append(unique)
	ens_vector = np.array(ens_vector)
	ens_vector = ens_vector[np.argsort(ens_vector[:,0])]

	# Prepare list of lists for sorted samples.
	split_samples = [[] for _ in range(len(ens_vector))]

	for row in dataset:
		# Match the metal counts of the ensemble zone to the possible unique configurations.
		for i, vector in enumerate(ens_vector):
			if np.all(row[:n_metals] == vector):
				# If succesfully matched it is appended to corresponding list
				split_samples[i].append(row)

	return ens_vector, split_samples
def train_PWL(regr_object,ensemble_array,n_metals):
	"""Trains SKLearn regressor object with the .fit() method to each ensemble
	and subsequently saves the objects in a dict with ensemble vector tuples as keys
	eg. (1,0,2) for a ternary alloy fcc-site composed of one atom metal 1 and two atom metal 3"""

	# Prepare dict of trained regressor objects
	regressor_dict = {}

	# Iterate through ensembles
	for ensemble in ensemble_array:
		# Numpy conversion
		array = np.array(ensemble)

		# Placeholder for the ensemble vector
		ens_vector = []

		# Interpret feature columns that have identical entries as the ensemble features.
		for i in range(n_metals):
			if np.all(array[:,i] == array[0,i]):
				# Append to ensemble vector
				ens_vector.append(array[0,i])

		# Define training features and targets (disposing row ids)
		training_features = array[:,n_metals:-1]
		training_targets = array[:,-1]

		# Train regressor object
		regr_object.fit(training_features,training_targets)

		# Save trained copy of regressor in dict with ensemble vector tuple as key
		regressor_dict[tuple(ens_vector)] = copy.deepcopy(regr_object)

	return regressor_dict
