import numpy as np
import pickle
import torch
from torch_geometric.data import DataLoader
from aux_scripts.GCN_model import load_GCN, test
import matplotlib.pyplot as plt

# set random seeds
np.random.seed(42)
torch.manual_seed(42)

# load validation set
set_list = ['AgIrPdPtRu',
            'IrPdPtRu',
            'AgPdPtRu',
            'AgIrPtRu',
            'AgIrPdRu',
            'AgIrPdPt',
            'AgPd',
            'IrPd',
            'IrPt',
            'PdRu',
            'PtRu']
test_sets = []
for alloy in set_list:
    with open(f'data/{alloy.lower()}.graphs', 'rb') as input:
        graphs = pickle.load(input)
        np.random.shuffle(graphs)
    test_sets.append(graphs[int(len(graphs)/2):])

kwargs = {'n_conv_layers': 3,  # number of gated graph convolution layers
          'n_hidden_layers': 0,  # number of hidden layers
          'conv_dim': 18,  # number of fully connected layers
          'act': 'relu' # activation function in hidden layers.
         }

# load trained state
with open(f'model_states/GC3H0reludim18BS64lr0.001.state', 'rb') as input:
    regressor = load_GCN(kwargs,pickle.load(input))

"""
all_total = []

# test loop
for i, test_graphs in enumerate(test_sets):
    # test model and concatenate L1loss
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

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlabel(r'$\Delta \mathrm{E}^{\mathrm{DFT}}_{\mathrm{ads}}-\Delta \mathrm{E}_{\mathrm{ads}}^\mathrm{Pt}}$ [eV]',fontsize=16)
    ax.set_ylabel(r'$\Delta \mathrm{E}^{\mathrm{pred}}_{\mathrm{ads}} \, [\mathrm{eV}]$', fontsize=16)
    ax.set_xlim(start, stop)
    ax.set_ylim(start, stop)
    ax.text(0.01, 0.98, f'GCN model on {set_list[i]}', family='monospace', fontsize=14, transform=ax.transAxes,
            verticalalignment='top', color='k')
    ax.scatter(test_true, test_pred, s=5, c=color_list, alpha=0.75)

    # plot solid diagonal line
    ax.plot([start, stop], [start, stop], 'k-', linewidth=1.0,
            label=r'$\Delta \mathrm{E}^{\mathrm{pred}} = \Delta \mathrm{E}^{\mathrm{DFT}}$')

    # plot dashed diagonal lines 0.1 eV above and below solid diagonal line
    pm = 0.1
    ax.plot([start, stop], [start + pm, stop + pm], 'k--', linewidth=1.0, label=r'$\pm %.2f \mathrm{eV}$' % pm)
    ax.plot([start + pm, stop], [start, stop - pm], 'k--', linewidth=1.0)

    ontop_L1loss = np.array(test_pred)[ontop_mask] - np.array(test_true)[ontop_mask]
    ax.text(0.01, 0.93,
            f'ontop OH L1loss: {np.mean(np.abs(ontop_L1loss)):.3f} eV ({len(np.array(test_pred)[ontop_mask])} samples)',
            family='monospace', fontsize=14, transform=ax.transAxes,
            verticalalignment='top', color='steelblue')
    fcc_L1loss = np.array(test_pred)[fcc_mask] - np.array(test_true)[fcc_mask]
    ax.text(0.01, 0.88,
            f'fcc O L1loss:    {np.mean(np.abs(fcc_L1loss)):.3f} eV ({len(np.array(test_pred)[fcc_mask])} samples)',
            family='monospace', fontsize=14, transform=ax.transAxes,
            verticalalignment='top', color='maroon')

    total_L1loss = np.array(test_pred) - np.array(test_true)
    all_total += list(total_L1loss)

    ax.text(0.01, 0.83, f'total L1loss:    {np.mean(np.abs(total_L1loss)):.3f} eV ({len(test_graphs)} samples)',
            family='monospace', fontsize=14,
            transform=ax.transAxes, verticalalignment='top', color='black')

    print(f'{np.mean(np.abs(total_L1loss)):.3f}')

    axins = ax.inset_axes([0.65, 0.1, 0.3, 0.3])
    axins.patch.set_alpha(0)
    axins.hist(ontop_L1loss, bins=20, range=(-3 * pm, 3 * pm), color='steelblue', alpha=0.5)
    axins.hist(fcc_L1loss, bins=20, range=(-3 * pm, 3 * pm), color='maroon', alpha=0.5)
    axins.axvline(0.0, linestyle='-', linewidth=0.5, color='black')
    axins.axvline(-pm, linestyle='--', linewidth=0.5, color='black')
    axins.axvline(pm, linestyle='--', linewidth=0.5, color='black')
    axins.get_yaxis().set_visible(False)
    for spine in ['right', 'left', 'top']:
        axins.spines[spine].set_visible(False)
    plt.savefig(f'parity/{set_list[i]}_GCN.png')
"""

# TOTAL ======
test_graphs = []
for set in test_sets:
    for sample in set:
        test_graphs.append(sample)
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
    plt.savefig(f'parity/{site[1:].replace(" ", "")}_GCN.png')

print(np.mean(np.abs(np.array(test_pred) - np.array(test_true))))