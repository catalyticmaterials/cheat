from utils.regression import split_ensembles, train_PWL
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
import pickle

np.random.seed(42)
n_metals = 5
regressor = Ridge()
site_ads_list = ['ontop_OH','fcc_O']
reg_dict = {}
preds, targets = {}, {}

for i, site_ads in enumerate(site_ads_list):
    # load training set
    set_list = ['AgIrPdPtRu']
    for alloy in set_list:
        with open(f'../features/{alloy.lower()}_{site_ads}.zonefeats', 'rb') as input:
            feats = pickle.load(input)
        np.random.shuffle(feats)
        train_feats = feats[:int(len(feats) * 0.9)]
        test_feats = feats[int(len(feats) * 0.9):]

    _, ensembles = split_ensembles(train_feats, n_metals)

    reg_dict[site_ads] = train_PWL(regressor,ensembles,n_metals)

    _, ensembles = split_ensembles(np.array(test_feats), n_metals)

    pred, true = [], []

    for ensemble in ensembles:
        if len(ensemble) == 0:
            continue
        ensemble = np.array(ensemble)
        skip = []
        for j, p in enumerate(reg_dict[site_ads][tuple(ensemble[0, :5])].predict(ensemble[:, n_metals:-1])):
            if np.abs(p) > 3.0:
                skip.append(j)
                print(ensemble[j], p)
                continue
            pred.append(p)
        for j, t in enumerate(ensemble[:, -1]):
            if j in skip:
                continue
            true.append(t)

    start, stop = -2, 1.5
    colors = ['steelblue', 'maroon']
    labels = ['*OH ontop', '*O fcc']

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    if i == 0:
        ax.set_xlabel(
            r'$\Delta \mathrm{E}^{\mathrm{DFT}}_{\mathrm{*OH}}-\Delta \mathrm{E}_{\mathrm{*OH}}^\mathrm{Pt}}$ [eV]',
            fontsize=16)
        ax.set_ylabel(r'$\Delta \mathrm{E}^{\mathrm{pred}}_{\mathrm{*OH}} \, [\mathrm{eV}]$', fontsize=16)
    elif i == 1:
        ax.set_xlabel(
            r'$\Delta \mathrm{E}^{\mathrm{DFT}}_{\mathrm{*O}}-\Delta \mathrm{E}_{\mathrm{*O}}^\mathrm{Pt}}$ [eV]',
            fontsize=16)
        ax.set_ylabel(r'$\Delta \mathrm{E}^{\mathrm{pred}}_{\mathrm{*O}} \, [\mathrm{eV}]$', fontsize=16)
    ax.set_xlim(start, stop)
    ax.set_ylim(start, stop)
    ax.text(0.01, 0.98, f'PWR model on testset', family='monospace', fontsize=18, transform=ax.transAxes,
            verticalalignment='top', color='k')
    ax.scatter(true, pred, s=2, color=colors[i], alpha=0.75)

    # plot solid diagonal line
    ax.plot([start, stop], [start, stop], 'k-', linewidth=1.0,
            label=r'$\Delta \mathrm{E}^{\mathrm{pred}} = \Delta \mathrm{E}^{\mathrm{DFT}}$')

    # plot dashed diagonal lines 0.1 eV above and below solid diagonal line
    pm = 0.1
    ax.plot([start, stop], [start + pm, stop + pm], 'k--', linewidth=1.0, label=r'$\pm %.2f \mathrm{eV}$' % pm)
    ax.plot([start + pm, stop], [start, stop - pm], 'k--', linewidth=1.0)
    L1loss = np.array(true) - np.array(pred)

    if i == 0:
        ax.text(0.01, 0.93,
                f'ontop *OH MAE: {np.mean(np.abs(L1loss)):.3f} eV',
                family='monospace', fontsize=18, transform=ax.transAxes,
                verticalalignment='top', color='steelblue')
    elif i == 1:
        ax.text(0.01, 0.93,
                f'fcc *O MAE:    {np.mean(np.abs(L1loss)):.3f} eV',
                family='monospace', fontsize=18, transform=ax.transAxes,
                verticalalignment='top', color='maroon')

    axins = ax.inset_axes([0.65, 0.1, 0.3, 0.3])
    axins.patch.set_alpha(0)
    axins.hist(L1loss, bins=20, range=(-3 * pm, 3 * pm), color=colors[i], alpha=0.5)

    axins.axvline(0.0, linestyle='-', linewidth=0.5, color='black')
    axins.axvline(-pm, linestyle='--', linewidth=0.5, color='black')
    axins.axvline(pm, linestyle='--', linewidth=0.5, color='black')
    axins.get_yaxis().set_visible(False)
    for spine in ['right', 'left', 'top']:
        axins.spines[spine].set_visible(False)
    plt.savefig(f'parity_{labels[i][1:].replace(" ", "")}_PWR.png')


with open(f'model_states/AgIrPdPtRu_PWR.obj', 'wb') as output:
    pickle.dump(reg_dict, output)

