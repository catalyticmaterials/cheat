from aux_scripts.PWL_model import split_ensembles, train_PWL
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

train_list = ['AgIrPdPtRu_dirichlet']

for i, site_ads in enumerate(site_ads_list):
    data = []
    for alloy in train_list:
        # load test set
        with open(f'data/{alloy.lower()}_{site_ads}.zonefeats', 'rb') as input:
            dataset = pickle.load(input)
            for row in dataset:
                data.append(row)
    data = np.array(data)

    _, ensembles = split_ensembles(data, n_metals)

    reg_dict[site_ads] = train_PWL(regressor,ensembles,n_metals)

with open(f'model_states/AgIrPdPtRu_dirichlet_PWR.obj', 'wb') as output:
    pickle.dump(reg_dict, output)

test_list = ['AgIrPdPtRu',
             'IrPdPtRu',
             'AgPdPtRu',
             'AgIrPtRu',
             'AgIrPdRu',
             'AgIrPdPt',
             'AgPd',
             'IrPd',
             'IrPt',
             'PdRu',
             'PtRu'
             ]

for alloy in test_list:
    preds, targets = {}, {}
    for i, site_ads in enumerate(site_ads_list):
        with open(f'data/{alloy.lower()}_{site_ads}.zonefeats', 'rb') as input:
            data = pickle.load(input)
            np.random.shuffle(data)
        data = data[int(len(data)/2):]

        uniq_ensembles, ensembles = split_ensembles(data, n_metals)

        preds[site_ads], targets[site_ads] = [], []

        for ensemble in ensembles:
            if len(ensemble) == 0:
                continue
            ensemble = np.array(ensemble)
            skip = []
            for j, p in enumerate(reg_dict[site_ads][tuple(ensemble[0,:5])].predict(ensemble[:,n_metals:-1])):
                if np.abs(p) > 3.0:
                    skip.append(j)
                    print(ensemble[j], p)
                    continue
                preds[site_ads].append(p)
            for j, t in enumerate(ensemble[:,-1]):
                if j in skip:
                    continue
                targets[site_ads].append(t)

        #preds[site_ads] = [sample for list in p for sample in list]
        #targets[site_ads] = [sample for list in t for sample in list]

    start, stop = -2,1.5
    pm = 0.1

    colors = ['steelblue','maroon']
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlabel(r'$\Delta \mathrm{E}^{\mathrm{DFT}}_{\mathrm{ads}}-\Delta \mathrm{E}_{\mathrm{ads}}^\mathrm{Pt}}$ [eV]', fontsize=16)
    ax.set_ylabel(r'$\Delta \mathrm{E}^{\mathrm{pred}}_{\mathrm{ads}} \, [\mathrm{eV}]$', fontsize=16)
    ax.set_xlim(start,stop)
    ax.set_ylim(start,stop)
    ax.plot([start,stop],[start,stop], 'k-', linewidth=1.0, label=r'$\Delta \mathrm{E}_{\mathrm{test}} = \Delta \mathrm{E}_{\mathrm{DFT}}$')
    ax.plot([start,stop],[start+pm, stop+pm], 'k--', linewidth=1.0,label=r'$\pm %.2f \mathrm{eV}$'%pm)
    ax.plot([start+pm,stop],[start,stop-pm], 'k--', linewidth=1.0)
    axins = ax.inset_axes([0.65, 0.1, 0.3, 0.3])
    axins.patch.set_alpha(0)
    axins.axvline(0.0, linestyle='-', linewidth=0.5, color='black')
    axins.axvline(-pm, linestyle='--', linewidth=0.5, color='black')
    axins.axvline(pm, linestyle='--', linewidth=0.5, color='black')
    axins.get_yaxis().set_visible(False)
    for spine in ['right', 'left', 'top']:
        axins.spines[spine].set_visible(False)

    n_samples = 0
    for a in targets.values():
        n_samples += len(a)
    ax.text(0.01, 0.98, f'PWR model on {alloy}', family='monospace', fontsize=14, transform=ax.transAxes, verticalalignment='top', color='k')

    total_err = []
    for i, site_ads in enumerate(site_ads_list):
        ax.scatter(targets[site_ads], preds[site_ads], s=5, c=colors[i], alpha=0.75)
        err = np.array(preds[site_ads]) - np.array(targets[site_ads])
        if i == 0:
            total_err = err
            insert = ''
        else:
            insert = '   '
            total_err = np.concatenate((total_err, err))
        axins.hist(err, bins=20, range=(-3 * pm, 3 * pm), color=colors[i], alpha=0.5)
        ax.text(0.01, 0.93 - 0.05 * i,
                f'{site_ads.replace("_", " ")} L1loss:{insert} {np.mean(np.abs(err)):.3f} eV ({len(targets[site_ads])} samples)',
                family='monospace', fontsize=14,
                transform=ax.transAxes, verticalalignment='top', color=colors[i])

    ax.text(0.01, 0.83, f'total L1loss:    {np.mean(np.abs(total_err)):.3f} eV ({n_samples} samples)',
            family='monospace', fontsize=14, transform=ax.transAxes, verticalalignment='top', color='k')

    print(f'{np.mean(np.abs(total_err)):.3f}')

    plt.savefig(f'parity/{alloy}_PWR.png')

np.random.seed(42)
all_preds, all_targets = [], []

# TOTAL ======
for i, site_ads in enumerate(site_ads_list):
    all_data = []
    for alloy in test_list:
        with open(f'data/{alloy.lower()}_{site_ads}.zonefeats', 'rb') as input:
            data = pickle.load(input)
            np.random.shuffle(data)
            for sample in data[int(len(data) / 2):]:
                all_data.append(sample)
    uniq_ensembles, ensembles = split_ensembles(np.array(all_data), n_metals)

    pred, true = [], []

    for ensemble in ensembles:
        if len(ensemble) == 0:
            continue
        ensemble = np.array(ensemble)
        skip = []
        for j, p in enumerate(reg_dict[site_ads][tuple(ensemble[0, :5])].predict(ensemble[:, n_metals:-1])):
            if np.abs(p) > 3.0:
                skip.append(j)
                print(ensemble[j],p)
                continue
            pred.append(p)
            all_preds.append(p)
        for j, t in enumerate(ensemble[:, -1]):
            if j in skip:
                continue
            true.append(t)
            all_targets.append(t)


    start, stop = -2, 1.5
    colors = ['steelblue', 'maroon']
    labels = ['*OH ontop', '*O fcc']

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    if i == 0:
        ax.set_xlabel(r'$\Delta \mathrm{E}^{\mathrm{DFT}}_{\mathrm{*OH}}-\Delta \mathrm{E}_{\mathrm{*OH}}^\mathrm{Pt}}$ [eV]',fontsize=16)
        ax.set_ylabel(r'$\Delta \mathrm{E}^{\mathrm{pred}}_{\mathrm{*OH}} \, [\mathrm{eV}]$', fontsize=16)
    elif i == 1:
        ax.set_xlabel(r'$\Delta \mathrm{E}^{\mathrm{DFT}}_{\mathrm{*O}}-\Delta \mathrm{E}_{\mathrm{*O}}^\mathrm{Pt}}$ [eV]',fontsize=16)
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
    plt.savefig(f'parity/{labels[i][1:].replace(" ", "")}_PWR.png')

L1loss = np.array(all_targets) - np.array(all_preds)
print(np.mean(np.abs(L1loss)))