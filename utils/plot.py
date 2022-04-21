from matplotlib.ticker import AutoMinorLocator
import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#from .misc import uncertainty

def format_ax(ax,xlabel,ylabel,ticklabel_size=10,axlabel_size=12, put_minor=True):
    ax.yaxis.set_tick_params(labelsize=ticklabel_size)
    ax.xaxis.set_tick_params(labelsize=ticklabel_size)
    if put_minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='minor', length=3)
        ax.tick_params(which='major', length=6)
    ax.set_xlabel(xlabel, fontsize=axlabel_size)
    ax.set_ylabel(ylabel, fontsize=axlabel_size)

def plot_cv(output, pm, metal_labels, regressor_label, colormap=True,no_color=False):
    train_score, train_std = output[0],output[1]
    test_score, test_std = output[2], output[3]
    train_pred, train_targets = output[4],output[6]
    test_pred, test_targets = output[5], output[7]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    n_samples = 0
    temp_min, temp_max = [], []
    for i in range(len(train_pred)):
        if no_color:
            ax.scatter(train_targets[i], train_pred[i], marker='o', s=5, alpha=0.5,
                       cmap=get_cmap('gist_rainbow'), c=np.arange(len(train_pred[i])))
            ax.scatter(test_targets[i], test_pred[i], marker='x', s=50, alpha=0.5,
                       cmap=get_cmap('gist_rainbow'), c=np.arange(len(test_pred[i])))
        elif colormap:
            cmap = get_colormap(get_color(metal_labels[0][:2]), get_color(metal_labels[1][:2]))
            ax.scatter(train_targets[i],train_pred[i],marker='o',s=5, alpha=0.75,
                       label= metal_labels[i], color=cmap(float(i/(len(train_pred)-1))))
            ax.scatter(test_targets[i], test_pred[i], marker='x', s=50, alpha=0.75,
                       color=cmap(float(i / (len(test_pred) - 1))))
        else:
            ax.scatter(train_targets[i], train_pred[i], marker='o', s=5, alpha=0.75,
                       label= metal_labels[i][:2], color=get_color(metal_labels[i][:2]))
            ax.scatter(test_targets[i], test_pred[i], marker='x', s=50, alpha=0.75,
                       color=get_color(metal_labels[i][:2]))

        n_samples += len(train_targets[i])
        n_samples += len(test_targets[i])
        temp_min.append(np.min(train_targets[i]))
        temp_min.append(np.min(train_pred[i]))
        temp_min.append(np.min(test_targets[i]))
        temp_min.append(np.min(test_pred[i]))
        temp_max.append(np.max(train_targets[i]))
        temp_max.append(np.max(train_pred[i]))
        temp_max.append(np.max(test_targets[i]))
        temp_max.append(np.max(test_pred[i]))

    min = np.min(temp_min) - 0.1
    max = np.max(temp_max) + 0.1

    ax.plot([min,max], [min,max], 'k-', linewidth=1.0)
    ax.plot([min,max], [min+pm,max+pm], 'k--', linewidth=1.0)
    ax.plot([min,max], [min-pm,max-pm], 'k--', linewidth=1.0)
    format_ax(ax, r'E$_{\mathrm{DFT}}$-E$_{\mathrm{DFT}}^{\mathrm{Pt}}$ [eV]', r'E$_{\mathrm{pred}}$-E$_{\mathrm{DFT}}^{\mathrm{Pt}}$ [eV]')
    ax.set(xlim=(min,max),ylim=(min,max))
    ax.text(0.02,0.98,f'{regressor_label}\nTrain MAE = {train_score:.3f}({uncertainty(train_std,3)})eV\nTest MAE = {test_score:.3f}({uncertainty(test_std,3)})eV\n' + r'N$_{\mathrm{samples}}$= '+ str(n_samples), family='monospace', transform=ax.transAxes,
              fontsize=14, verticalalignment='top', horizontalalignment='left', color='black')
    if not no_color:
        ax.legend(loc='lower right', fontsize=14, markerscale=3)
    plt.tight_layout()

    return fig

"""
def plot_parity():
    start, stop = -2,1.5
	ontop_mask = np.array(test_site) == 'ontop'
	fcc_mask = np.array(test_site) == 'fcc'

	colors = ['steelblue','maroon']
	color_list = []
	for entry in test_site:
		if entry == 'ontop':
			color_list.append(colors[0])
		elif entry == 'fcc':
			color_list.append(colors[1])

	fig, ax = plt.subplots(1, 1, figsize=(12, 6))
	ax.set_xlabel(r'$\Delta \mathrm{E}_{\mathrm{DFT}} \, [\mathrm{eV}]$', fontsize=16)
	ax.set_ylabel(r'$\Delta \mathrm{E}_{\mathrm{pred}} \, [\mathrm{eV}]$', fontsize=16)
	ax.set_xlim(start,stop)
	ax.set_ylim(start,stop)
	ax.text(0.01, 0.98, f'Best GCN model on test set ({len(test_graphs)} samples)', family='monospace', fontsize=14, transform=ax.transAxes,
			verticalalignment='top', color='k')
	ax.text(0.01, 0.93, f'Epoch {best_val_loss[1]}, CV-fold {best_val_loss[2]}, Val. L1Loss {best_val_loss[0]:.3f} eV', family='monospace', fontsize=14, transform=ax.transAxes, verticalalignment='top', color='k')
	ax.scatter(test_targets, test_pred, s=5, c=color_list, alpha=0.75)
	# plot solid diagonal line
	ax.plot([start,stop],[start,stop], 'k-', linewidth=1.0, label=r'$\Delta \mathrm{E}_{\mathrm{test}} = \Delta \mathrm{E}_{\mathrm{DFT}}$')

	# plot dashed diagonal lines 0.1 eV above and below solid diagonal line
	pm = 0.1
	ax.plot([start,stop],[start+pm, stop+pm], 'k--', linewidth=1.0,label=r'$\pm %.2f \mathrm{eV}$'%pm)
	ax.plot([start+pm,stop],[start,stop-pm], 'k--', linewidth=1.0)

	ontop_L1loss = np.array(test_pred)[ontop_mask] - np.array(test_targets)[ontop_mask]
	ax.text(0.01, 0.88, f'ontop OH L1loss: {np.mean(np.abs(ontop_L1loss)):.3f} eV', family='monospace', fontsize=14, transform=ax.transAxes,
			verticalalignment='top', color='steelblue')
	fcc_L1loss = np.array(test_pred)[fcc_mask] - np.array(test_targets)[fcc_mask]
	ax.text(0.01, 0.83, f'fcc O L1loss:    {np.mean(np.abs(fcc_L1loss)):.3f} eV', family='monospace', fontsize=14, transform=ax.transAxes,
			verticalalignment='top', color='maroon')

	total_L1loss = np.array(test_pred) - np.array(test_targets)
	ax.text(0.01, 0.78, f'total L1loss:    {np.mean(np.abs(total_L1loss)):.3f} eV', family='monospace', fontsize=14,
			transform=ax.transAxes, verticalalignment='top', color='black')

	axins = ax.inset_axes([0.65, 0.1, 0.3, 0.3])
	axins.patch.set_alpha(0)
	axins.hist(ontop_L1loss, bins=20, range=(-3 * pm, 3 * pm), color='steelblue', alpha=0.5)
	axins.hist(fcc_L1loss, bins=20, range=(-3*pm, 3*pm), color='maroon',alpha=0.5)
	axins.axvline(0.0, linestyle='-', linewidth=0.5, color='black')
	axins.axvline(-pm, linestyle='--', linewidth=0.5, color='black')
	axins.axvline(pm, linestyle='--', linewidth=0.5, color='black')
	axins.get_yaxis().set_visible(False)
	for spine in ['right', 'left', 'top']:
		axins.spines[spine].set_visible(False)
	plt.savefig(f'GCN_benchmark/{filename}_parity.png')
"""

def get_color(metal_label, whiteout_param=0):

    color_dict = {'Ag':np.array([192,192,192]) / 256,
                  'Ir': np.array([0,85,138]) / 256,
                  'Pd': np.array([0,107,136]) / 256,
                  'Pt': np.array([208,208,224]) / 256,
                  'Ru': np.array([0,146,144]) / 256,
                  }

    return color_dict[metal_label] * (1 - whiteout_param) + whiteout_param

def get_dark_color(metal_label):
    color_dict = {'Ag': np.array([192, 192, 192])/2 / 256,
                  'Ir': np.array([0, 85, 138])/2 / 256,
                  'Pd': np.array([0, 107, 136])/2 / 256,
                  'Pt': np.array([208, 208, 224])/2 / 256,
                  'Ru': np.array([0, 146, 144])/2 / 256,
                  }
    return color_dict[metal_label]

def get_colormap(color1,color2):
    vals = np.ones((256, 3))
    vals[:, 0] = np.linspace(color1[0], color2[0], 256)
    vals[:, 1] = np.linspace(color1[1], color2[1], 256)
    vals[:, 2] = np.linspace(color1[2], color2[2], 256)
    return ListedColormap(vals)

def find_maxmin(list):
    all_max, all_min = None, None
    for ens in list:
        ens = np.array(ens)
        if all_max != None and all_min != None:
            if max(ens[:,-4]) > all_max:
                all_max = max(ens[:,-4])
            if min(ens[:,-4]) < all_min:
                all_min = min(ens[:,-4])
        else:
            all_max, all_min = max(ens[:,-4]), min(ens[:,-4])
    return all_min-0.2, all_max+0.2

def plot_histogram(ensemble_array,alloy_label,sites,adsorbate,bin_width,pure_eads, min_E, max_E):
    #min_E, max_E = find_maxmin(ensemble_array)

    bins = int((max_E-min_E)/bin_width)

    metals = []
    for i in range(int(len(alloy_label)/2)):
        metals.append(alloy_label[i*2:i*2+2])

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.hist(np.array([item for sublist in ensemble_array for item in sublist])[:, -4], bins=bins,
            range=(min_E, max_E), color='black', histtype='bar', alpha=0.3, label='Total')
    ax.hist(np.array([item for sublist in ensemble_array for item in sublist])[:, -4], bins=bins,
            range=(min_E, max_E), color='black', histtype='step', alpha=0.5)

    vert_list = [0.83, 0.77, 0.71, 0.65, 0.59]

    for i, ensemble in enumerate(ensemble_array):
        ens = np.array(ensemble)
        if adsorbate == 'OH':
            color, darkcolor = get_color(sites[i]), get_dark_color(sites[i])
        elif adsorbate == 'O' and len(metals) == 2:
            cmap = get_colormap(get_color(sites[0][:2]), get_color(sites[3][:2]))
            color = cmap(float(i/(len(ensemble_array)-1)))
        else:
            color = get_cmap('gist_rainbow')(float(i/(2)))

        ax.hist(ens[:, -4], bins=bins, range=(min_E, max_E), color=color, histtype='bar', alpha=0.5)
        ax.hist(ens[:, -4], bins=bins, range=(min_E, max_E), color=color, histtype='step')

        if len(metals) == 2 or not adsorbate == 'O':
            print(len(sites[i]))
            if len(sites[i]) > 6:
                if np.mean(ens[:, -4]) < 0:
                    d = sites[i] + r' {:.3f} ({:.3f})'.format(np.mean(ens[:, -4]), np.std(ens[:, -4]))
                else:
                    d = sites[i] + r'  {:.3f} ({:.3f})'.format(np.mean(ens[:, -4]), np.std(ens[:, -4]))
            else:
                if np.mean(ens[:, -4]) < 0:
                    d = sites[i] + r'   {:.3f} ({:.3f})'.format(np.mean(ens[:, -4]), np.std(ens[:, -4]))
                else:
                    d = sites[i] + r'    {:.3f} ({:.3f})'.format(np.mean(ens[:, -4]), np.std(ens[:, -4]))

            ax.text(0.02, vert_list[i], d, family='monospace', transform=ax.transAxes,
                        fontsize=14, color=color, verticalalignment='top')

            ylim = ax.get_ylim()[1]*1.1

            if adsorbate == 'O' and len(metals) < 2:
                pass
            else:
                ax.text(pure_eads[sites[i][:2]], ylim / 12, sites[i][:2], family='monospace', fontsize=14,
                        verticalalignment='bottom', horizontalalignment='center',zorder=10)
                ax.arrow(pure_eads[sites[i][:2]], ylim  / 12, 0, -ylim  / 12 + 0.2,
                             head_width=(max_E - min_E) / 100, head_length=ylim / 30, length_includes_head=True,
                             ec='black', fill=False,zorder=10)


    ax.set(xlim=(min_E, max_E), ylim=(0,ax.get_ylim()[1]*1.3))
    ax.set_xlabel(r'$\Delta \mathrm{E}_{\mathrm{OH}}-\Delta \mathrm{E}_{\mathrm{OH}}^\mathrm{Pt}}$ [eV]', fontsize=20)
    ax.set_ylabel('Frequency [binsize: {:.3f} eV]'.format((max_E - min_E) / bins), fontsize=20)
    ax.text(0.01, 0.98, f'$^*${adsorbate} ' + alloy_label, family='monospace', transform=ax.transAxes, fontsize=18,
            color='black', verticalalignment='top')
    if len(metals) == 2 or not adsorbate == 'O':
        ax.text(0.01, 0.90, r'Ens.     $\mu_{\Delta E}$   ($\sigma_{\Delta E}$)  [eV]', family='monospace',
                transform=ax.transAxes, fontsize=14, color='black', verticalalignment='top')
    ax.yaxis.set_tick_params(labelsize=16)
    ax.xaxis.set_tick_params(labelsize=16, size=6, width=1.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_tick_params(which='minor', size=3, width=1)
    plt.tight_layout()

    number_of_samples = len(np.array([item for sublist in ensemble_array for item in sublist])[:, -1])
    ax.text(0.98, 0.98, str(number_of_samples) + r' samples', family='monospace', transform=ax.transAxes, fontsize=16,
            color='black', verticalalignment='top', horizontalalignment='right')

    return fig

def plot_histogram_onehot(ensemble_array,alloy_label,sites,adsorbate,bin_width,pure_eads):
    min_E, max_E = find_maxmin(ensemble_array)

    bins = int((max_E-min_E)/bin_width)

    metals = []
    for i in range(int(len(alloy_label)/2)):
        metals.append(alloy_label[i*2:i*2+2])

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.hist(np.array([item for sublist in ensemble_array for item in sublist])[:, -4], bins=bins,
            range=(min_E, max_E), color='black', histtype='bar', alpha=0.3, label='Total')
    ax.hist(np.array([item for sublist in ensemble_array for item in sublist])[:, -4], bins=bins,
            range=(min_E, max_E), color='black', histtype='step', alpha=0.5)

    vert_list = [0.83, 0.77, 0.71, 0.65, 0.59]

    for i, ensemble in enumerate(ensemble_array):
        ens = np.array(ensemble)
        if adsorbate == 'OH':
            color, darkcolor = get_color(sites[i]), get_dark_color(sites[i])
        elif adsorbate == 'O' and len(metals) == 2:
            cmap = get_colormap(get_color(sites[0][:2]), get_color(sites[3][:2]))
            color = cmap(float(i/(len(ensemble_array)-1)))
        else:
            color = get_cmap('gist_rainbow')(float(i/(len(ensemble_array)-1)))

        ax.hist(ens[:, -4], bins=bins, range=(min_E, max_E), color=color, histtype='bar', alpha=0.5)
        ax.hist(ens[:, -4], bins=bins, range=(min_E, max_E), color=color, histtype='step')

        if len(metals) == 2 or not adsorbate == 'O':
            print(len(sites[i]))
            if len(sites[i]) > 6:
                if np.mean(ens[:, -4]) < 0:
                    d = sites[i] + r' {:.3f} ({:.3f})'.format(np.mean(ens[:, -4]), np.std(ens[:, -4]))
                else:
                    d = sites[i] + r'  {:.3f} ({:.3f})'.format(np.mean(ens[:, -4]), np.std(ens[:, -4]))
            else:
                if np.mean(ens[:, -4]) < 0:
                    d = sites[i] + r'   {:.3f} ({:.3f})'.format(np.mean(ens[:, -4]), np.std(ens[:, -4]))
                else:
                    d = sites[i] + r'    {:.3f} ({:.3f})'.format(np.mean(ens[:, -4]), np.std(ens[:, -4]))

            ax.text(0.02, vert_list[i], d, family='monospace', transform=ax.transAxes,
                        fontsize=14, color=color, verticalalignment='top')

            ylim = ax.get_ylim()[1]*1.1

            if adsorbate == 'O' and len(metals) < 2:
                pass
            else:
                ax.text(pure_eads[sites[i][:2]], ylim / 12, sites[i][:2], family='monospace', fontsize=14,
                        verticalalignment='bottom', horizontalalignment='center',zorder=10)
                ax.arrow(pure_eads[sites[i][:2]], ylim  / 12, 0, -ylim  / 12 + 0.2,
                             head_width=(max_E - min_E) / 100, head_length=ylim / 30, length_includes_head=True,
                             ec='black', fill=False,zorder=10)


    ax.set(xlim=(min_E, max_E), ylim=(0,ax.get_ylim()[1]*1.3))
    if adsorbate == 'OH':
        ax.set_xlabel(r'$\Delta \mathrm{E}_{\mathrm{OH}}-\Delta \mathrm{E}_{\mathrm{OH}}^\mathrm{Pt}}$ [eV]', fontsize=20)
    if adsorbate == 'O':
        ax.set_xlabel(r'$\Delta \mathrm{E}_{\mathrm{O}}-\Delta \mathrm{E}_{\mathrm{O}}^\mathrm{Pt}}$ [eV]', fontsize=20)
    ax.set_ylabel('Frequency [binsize: {:.3f} eV]'.format((max_E - min_E) / bins), fontsize=20)
    ax.text(0.01, 0.98, f'$^*${adsorbate} ' + alloy_label, family='monospace', transform=ax.transAxes, fontsize=18,
            color='black', verticalalignment='top')
    if len(metals) == 2 or not adsorbate == 'O':
        ax.text(0.01, 0.90, r'Ens.     $\mu_{\Delta E}$   ($\sigma_{\Delta E}$)  [eV]', family='monospace',
                transform=ax.transAxes, fontsize=14, color='black', verticalalignment='top')
    ax.yaxis.set_tick_params(labelsize=16)
    ax.xaxis.set_tick_params(labelsize=16, size=6, width=1.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_tick_params(which='minor', size=3, width=1)
    plt.tight_layout()

    number_of_samples = len(np.array([item for sublist in ensemble_array for item in sublist])[:, -1])
    ax.text(0.98, 0.98, str(number_of_samples) + r' samples', family='monospace', transform=ax.transAxes, fontsize=16,
            color='black', verticalalignment='top', horizontalalignment='right')

    return fig

