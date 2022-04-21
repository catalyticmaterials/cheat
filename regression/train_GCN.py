import numpy as np
import pickle
from torch_geometric.data import DataLoader
import torch
import matplotlib.pyplot as plt
from utils import load_GCN, train, test
from copy import deepcopy

# set random seeds
np.random.seed(42)
torch.manual_seed(42)

# set Dataloader batch size, learning rate and max epochs
batch_size = 64
max_epochs = 3000
learning_rate = 1e-3

# early stopping is evaluated based on rolling validation error.
# if the val error has not decreased 1% during the prior *patience* number of epochs early stopping is invoked.
roll_val_width = 20  # mean of [current_epoch-roll_val_width/2 : current_epoch+roll_val_width/2 +1]
patience = 100
report_every = 100

# set grid of search parameters
kwargs = {'n_conv_layers': 3,  # number of gated graph convolution layers
		  'n_hidden_layers': 0,  # number of hidden layers
		  'conv_dim': 18,  # number of fully connected layers
		  'act': 'relu', # activation function in hidden layers.
		  }

# load training set
set_list = ['AgIrPdPtRu']
train_graphs, val_graphs, test_graphs = [], [], []
for alloy in set_list:
	with open(f'../features/{alloy.lower()}.graphs', 'rb') as input:
		graphs = pickle.load(input)
	np.random.shuffle(graphs)
	train_graphs += graphs[:int(len(graphs)*0.8)]
	val_graphs += graphs[int(len(graphs)*0.8):int(len(graphs)*0.9)]
	test_graphs += graphs[int(len(graphs) * 0.9):]

print(f'Training model with: \n\
			n_conv_layers={kwargs["n_conv_layers"]} \n\
			n_hidden_layers={kwargs["n_hidden_layers"]} \n\
			conv_dim={kwargs["conv_dim"]} \n\
			act={kwargs["act"]}')

filename = f'GC{kwargs["n_conv_layers"]}H{kwargs["n_hidden_layers"]}{kwargs["act"]}dim{kwargs["conv_dim"]}BS{batch_size}lr{learning_rate:.5}'

# initialize arrays for training and validation error
train_loss, val_loss = [], []

# initialize model and Dataloader
train_loaders = []
val_loaders = []
model = load_GCN(kwargs)
opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
train_loader = DataLoader(train_graphs, batch_size=batch_size, drop_last=True, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=len(val_graphs), drop_last=True, shuffle=False)

model_states = []

# epoch loop
for epoch in range(max_epochs):
	# train and validate for this epoch
	train_loss.append(train(model, train_loader, batch_size, opt))
	val_err, _, _, _, _ = test(model, val_loader, len(val_graphs))
	val_loss.append(val_err)
	model_states.append(deepcopy(model.state_dict()))

	# evaluate rolling mean of validation error and check for early stopping
	if epoch >= roll_val_width+patience:

		roll_val = np.convolve(val_loss, np.ones(int(roll_val_width+1)), 'valid') / int(roll_val_width+1)
		min_roll_val = np.min(roll_val[:-patience+1])
		improv = (roll_val[-1] - min_roll_val) / min_roll_val

		if improv > - 0.01:
			print('Early stopping invoked.')
			best_epoch = np.argmin(val_loss)
			best_state = model_states[best_epoch]
			break

	# report progress
	if epoch % report_every == 0:
		print(f'Epoch {epoch} train and val L1Loss: {train_loss[-1]:.3f} / {val_loss[-1]:.3f} eV')

# save final validation error.
print(f'Finished training sequence. Best epoch was {best_epoch} with val. L1Loss {np.min(val_loss):.3f} eV')

# save best model
with open(f'model_states/{filename}.state', 'wb') as output:
	pickle.dump(best_state, output)

# plot epoch curve
fig, main_ax = plt.subplots(1, 1, figsize=(8, 5))
color = ['steelblue','green']
label = [r'Training set  L1Loss',r'Validation set L1Loss']

for i, results in enumerate([train_loss, val_loss]):
	main_ax.plot(range(len(results)), results, color=color[i], label=label[i])
	if i == 1:
		main_ax.scatter(best_epoch, val_loss[best_epoch], facecolors='none', edgecolors='maroon', label='Best epoch', s=50, zorder=10)

main_ax.set_xlabel(r'Epoch', fontsize=16)
main_ax.set_ylabel(r'L1Loss [eV]', fontsize=16)
main_ax.set(ylim=(0.025,0.125))
main_ax.legend()
plt.savefig(f'epoch_curves/{filename}_curve.png')

## load trained state
#regressor = load_GCN(kwargs,best_state)

test_loader = DataLoader(test_graphs, batch_size=len(test_graphs))
_, test_pred, test_true, test_site, test_ads = test(regressor, test_loader, len(test_graphs))

start, stop = -2, 1.5
ontop_mask = np.array(test_site) == 'ontop'
fcc_mask = np.array(test_site) == 'fcc'

colors = ['steelblue', 'maroon']
color_list = []
for entry in test_site:
	if entry == 'ontop':
		color_list.append(colors[0])
	elif entry == 'fcc':
		color_list.append(colors[1])

for i, site in enumerate(['*OH ontop', '*O fcc']):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    if i == 0:
        ax.set_xlabel(r'$\Delta \mathrm{E}^{\mathrm{DFT}}_{\mathrm{*OH}}-\Delta \mathrm{E}_{\mathrm{*OH}}^\mathrm{Pt}}$ [eV]',fontsize=16)
        ax.set_ylabel(r'$\Delta \mathrm{E}^{\mathrm{pred}}_{\mathrm{*OH}} \, [\mathrm{eV}]$', fontsize=16)
    elif i == 1:
        ax.set_xlabel(r'$\Delta \mathrm{E}^{\mathrm{DFT}}_{\mathrm{*O}}-\Delta \mathrm{E}_{\mathrm{*O}}^\mathrm{Pt}}$ [eV]',fontsize=16)
        ax.set_ylabel(r'$\Delta \mathrm{E}^{\mathrm{pred}}_{\mathrm{*O}} \, [\mathrm{eV}]$', fontsize=16)
    ax.set_xlim(start, stop)
    ax.set_ylim(start, stop)
    ax.text(0.01, 0.98, f'GCN model on testset', family='monospace', fontsize=18, transform=ax.transAxes,
            verticalalignment='top', color='k')
    if i == 0:
        ax.scatter(np.array(test_true)[ontop_mask], np.array(test_pred)[ontop_mask], s=2, c='steelblue', alpha=0.75)
    elif i == 1:
        ax.scatter(np.array(test_true)[fcc_mask], np.array(test_pred)[fcc_mask], s=2, c='maroon', alpha=0.75)

    # plot solid diagonal line
    ax.plot([start, stop], [start, stop], 'k-', linewidth=1.0,
            label=r'$\Delta \mathrm{E}^{\mathrm{pred}} = \Delta \mathrm{E}^{\mathrm{DFT}}$')

    # plot dashed diagonal lines 0.1 eV above and below solid diagonal line
    pm = 0.1
    ax.plot([start, stop], [start + pm, stop + pm], 'k--', linewidth=1.0, label=r'$\pm %.2f \mathrm{eV}$' % pm)
    ax.plot([start + pm, stop], [start, stop - pm], 'k--', linewidth=1.0)
    if i == 0:
        ontop_L1loss = np.array(test_pred)[ontop_mask] - np.array(test_true)[ontop_mask]
        ax.text(0.01, 0.93,
                f'ontop *OH MAE: {np.mean(np.abs(ontop_L1loss)):.3f} eV',
                family='monospace', fontsize=18, transform=ax.transAxes,
                verticalalignment='top', color='steelblue')
    elif i == 1:
        fcc_L1loss = np.array(test_pred)[fcc_mask] - np.array(test_true)[fcc_mask]
        ax.text(0.01, 0.93,
                f'fcc *O MAE:    {np.mean(np.abs(fcc_L1loss)):.3f} eV',
                family='monospace', fontsize=18, transform=ax.transAxes,
                verticalalignment='top', color='maroon')

    axins = ax.inset_axes([0.65, 0.1, 0.3, 0.3])
    axins.patch.set_alpha(0)
    if i == 0:
        axins.hist(ontop_L1loss, bins=20, range=(-3 * pm, 3 * pm), color='steelblue', alpha=0.5)
    elif i == 1:
        axins.hist(fcc_L1loss, bins=20, range=(-3 * pm, 3 * pm), color='maroon', alpha=0.5)
    axins.axvline(0.0, linestyle='-', linewidth=0.5, color='black')
    axins.axvline(-pm, linestyle='--', linewidth=0.5, color='black')
    axins.axvline(pm, linestyle='--', linewidth=0.5, color='black')
    axins.get_yaxis().set_visible(False)
    for spine in ['right', 'left', 'top']:
        axins.spines[spine].set_visible(False)
    plt.savefig(f'parity_{site[1:].replace(" ", "")}_GCN.png')

