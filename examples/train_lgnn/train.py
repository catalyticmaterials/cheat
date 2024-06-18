import pickle, torch
import numpy as np
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
from cheatools.lgnn import lGNN
from copy import deepcopy

# load train and validation set
for s in ['train','val']:
    with open(f'graphs/{s}.graphs', 'rb') as input:
        globals()[f'{s}_graphs'] = pickle.load(input)

filename = 'lGNN'
torch.manual_seed(42)

# set Dataloader batch size, learning rate and max epochs
batch_size = 64
max_epochs = 1000
learning_rate = 1e-3

# early stopping is evaluated based on rolling validation error.
# if the val error has not decreased 1% during the prior *patience* number of epochs early stopping is invoked.
roll_val_width = 20  # mean of [current_epoch-roll_val_width/2 : current_epoch+roll_val_width/2 +1]
patience = 100
report_every = 25

# set lGNN architecture
arch = {'n_conv_layers': 3,  # number of gated graph convolution layers
        'n_hidden_layers': 0,  # number of hidden layers
        'conv_dim': 18,  # feature dimension w. zero-padding
        'act': 'relu', # activation function in hidden layers.
       }

# load model, optimizer, and dataloaders
model = lGNN(arch=arch)
opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
train_loader = DataLoader(train_graphs, batch_size=batch_size, drop_last=True, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=len(val_graphs), drop_last=True, shuffle=False)

# initialize arrays for training and validation error
train_loss, val_loss = [], []
model_states = []

# epoch loop
for epoch in range(max_epochs):
    # train and validate for this epoch
    train_loss.append(model.train4epoch(train_loader, batch_size, opt))
    val_err, _, _, _ = model.test(val_loader, len(val_graphs))
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

# add arch and onehot labels for future reference
best_state['onehot_labels'] = train_graphs[0]['onehot_labels']
best_state['arch'] = arch

# save best model
with open(f'{filename}.state', 'wb') as output:
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

plt.savefig(f'{filename}_curve.png')


