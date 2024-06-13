from matplotlib.ticker import AutoMinorLocator
import numpy as np
#from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ase.data import atomic_numbers
from ase.data.colors import jmol_colors
import matplotlib as mpl
import itertools as it

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

def get_color(metal_label, whiteout_param=0):
    return jmol_colors[atomic_numbers[metal_label]] * (1 - whiteout_param) + whiteout_param

def get_dark_color(metal_label):
    return jmol_colors[atomic_numbers[metal_label]] / 2

def get_colormap(color1,color2):
    vals = np.ones((256, 3))
    vals[:, 0] = np.linspace(color1[0], color2[0], 256)
    vals[:, 1] = np.linspace(color1[1], color2[1], 256)
    vals[:, 2] = np.linspace(color1[2], color2[2], 256)
    return ListedColormap(vals)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

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

def plot_parity(OH_pred,O_pred,OH_true,O_true,string):
    
    start, stop = -0.5, 2.5

    colors = ['royalblue', 'firebrick']
    color_list = []

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    #ax.set_xlabel(r'$\Delta \mathrm{E}^{\mathrm{DFT}}_{\mathrm{ads}} - \Delta \mathrm{E}^{\mathrm{DFT}}_{\mathrm{Pt}} $ [eV]',fontsize=18)
    ax.set_xlabel(r'$\Delta \mathrm{E}^{\mathrm{DFT}}_{\mathrm{ads}}$ [eV]',fontsize=22, labelpad=10)
    #ax.set_ylabel(r'$\Delta \mathrm{E}^{\mathrm{pred}}_{\mathrm{ads}} - \Delta \mathrm{E}^{\mathrm{pred}}_{\mathrm{Pt}} \, [\mathrm{eV}]$', fontsize=18)
    ax.set_ylabel(r'$\Delta \mathrm{E}^{\mathrm{model}}_{\mathrm{ads}}$ [eV]', fontsize=22, labelpad=-7)
    ax.set_xlim(start, stop)
    ax.set_ylim(start, stop)
    
    #if fmax == None:
    ax.text(0.01, 0.99, string, family='monospace', fontsize=16, transform=ax.transAxes,va='top', color='k')
    #elif frame_num == None:
    #    ax.text(0.01, 0.98, f'GemNet-OC-large S2EF fmax = {fmax:.2f} eV/Å', family='monospace', fontsize=14, transform=ax.transAxes,verticalalignment='top', color='k')
    #else: 
    #    ax.text(0.01, 0.98, f'GemNet-OC-large S2EF fmax = {fmax:.2f} eV/Å at frame {frame_num}', family='monospace', fontsize=14, transform=ax.transAxes,verticalalignment='top', color='k')
    ax.scatter(OH_true, OH_pred, s=15, c=colors[0], alpha=0.5,zorder=1)
    ax.scatter(O_true, O_pred, s=15, c=colors[1], alpha=0.5,zorder=0)

    # plot solid diagonal line
    ax.plot([start, stop], [start, stop], 'k-', linewidth=1.0,
            label=r'$\Delta \mathrm{E}^{\mathrm{pred}} = \Delta \mathrm{E}^{\mathrm{DFT}}$')

    # plot dashed diagonal lines 0.1 eV above and below solid diagonal line
    pm = 0.1
    ax.plot([start, stop], [start + pm, stop + pm], 'k--', linewidth=1.0, label=r'$\pm %.2f \mathrm{eV}$' % pm)
    ax.plot([start + pm, stop], [start, stop - pm], 'k--', linewidth=1.0)

    ontop_L1loss = OH_pred - OH_true
    ax.text(0.01, 0.95,
            f'*OH MAE (ME):   {np.mean(np.abs(ontop_L1loss)):.3f} ({np.mean(ontop_L1loss):.3f}) eV',
            family='monospace', fontsize=16, transform=ax.transAxes,
            va='top', color=colors[0])
    fcc_L1loss = O_pred - O_true
    ax.text(0.01, 0.91,
            f'*O MAE (ME):    {np.mean(np.abs(fcc_L1loss)):.3f} ({np.mean(fcc_L1loss):.3f}) eV',
            family='monospace', fontsize=16, transform=ax.transAxes,
            va='top', color=colors[1])
    
    total_L1loss = np.concatenate((OH_pred,O_pred)) - np.concatenate((OH_true,O_true))

    ax.text(0.01, 0.87, f'total MAE (ME): {np.mean(np.abs(total_L1loss)):.3f} ({np.mean(total_L1loss):.3f}) eV',
            family='monospace', fontsize=16,
            transform=ax.transAxes, va='top', color='black')

    axins = ax.inset_axes([0.55, 0.1, 0.4, 0.4])
    ax.tick_params(axis='both', which='major', labelsize=16)
    axins.patch.set_alpha(0)
    axins.hist(ontop_L1loss, bins=20, range=(-3 * pm, 3 * pm), color=colors[0], alpha=0.5)
    axins.hist(fcc_L1loss, bins=20, range=(-3 * pm, 3 * pm), color=colors[1], alpha=0.5)
    axins.axvline(0.0, linestyle='-', linewidth=0.5, color='black')
    axins.axvline(-pm, linestyle='--', linewidth=0.5, color='black')
    axins.axvline(pm, linestyle='--', linewidth=0.5, color='black')
    axins.tick_params(axis='both', which='major', labelsize=14)
    axins.get_yaxis().set_visible(False)
    for spine in ['right', 'left', 'top']:
        axins.spines[spine].set_visible(False)
    
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)    

    return fig

def plot_parity_single(true,pred,string,color,pr=[-0.5,2.5]):
    true, pred = np.array(true), np.array(pred)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlabel(r'$\Delta \mathrm{E}^{\mathrm{DFT}}_{\mathrm{ads}}$ [eV]',fontsize=22, labelpad=10)
    ax.set_ylabel(r'$\Delta \mathrm{E}^{\mathrm{model}}_{\mathrm{ads}}$ [eV]', fontsize=22, labelpad=-7)
    ax.set(xlim=(pr[0],pr[1]),ylim=(pr[0],pr[1]))
    ax.text(0.01, 0.99, string, family='monospace', fontsize=16, transform=ax.transAxes,va='top', color='k')
    ax.scatter(true, pred, s=15, c=color, alpha=0.5,zorder=0)
    
    # plot diagonal lines
    ax.plot(pr, pr, 'k-', linewidth=1.0,label=r'$\Delta \mathrm{E}^{\mathrm{pred}} = \Delta \mathrm{E}^{\mathrm{DFT}}$')
    pm = 0.1
    ax.plot(pr, [pr[0] + pm, pr[1] + pm], 'k--', linewidth=1.0, label=r'$\pm %.2f \mathrm{eV}$' % pm)
    ax.plot([pr[0] + pm, pr[1]], [pr[0], pr[1] - pm], 'k--', linewidth=1.0)

    diff = pred - true
    ax.text(0.01, 0.95,
            f'MAE (ME):   {np.mean(np.abs(diff)):.3f} ({np.mean(diff):.3f}) eV',
            family='monospace', fontsize=16, transform=ax.transAxes,
            va='top', color=color)

    axins = ax.inset_axes([0.55, 0.1, 0.4, 0.4])
    ax.tick_params(axis='both', which='major', labelsize=16)
    axins.patch.set_alpha(0)
    axins.hist(diff, bins=20, range=(-3 * pm, 3 * pm), color=color, alpha=0.5)
    axins.axvline(0.0, linestyle='-', linewidth=0.5, color='black')
    axins.axvline(-pm, linestyle='--', linewidth=0.5, color='black')
    axins.axvline(pm, linestyle='--', linewidth=0.5, color='black')
    axins.tick_params(axis='both', which='major', labelsize=14)
    axins.get_yaxis().set_visible(False)
    for spine in ['right', 'left', 'top']:
        axins.spines[spine].set_visible(False)

    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)

    return fig

def plot_parity_array(arr,string,colors,ads,pr=[-0.5,2.5]):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlabel(r'$\Delta \mathrm{E}^{\mathrm{DFT}}_{\mathrm{ads}}$ [eV]',fontsize=22, labelpad=10)
    ax.set_ylabel(r'$\Delta \mathrm{E}^{\mathrm{model}}_{\mathrm{ads}}$ [eV]', fontsize=22, labelpad=0)
    ax.set(xlim=(pr[0],pr[1]),ylim=(pr[0],pr[1]))
    ax.text(0.01, 0.99, string, family='monospace', fontsize=16, transform=ax.transAxes,va='top', color='k')
    
    # plot diagonal lines 
    ax.plot(pr, pr, 'k-', linewidth=1.0,label=r'$\Delta \mathrm{E}^{\mathrm{pred}} = \Delta \mathrm{E}^{\mathrm{DFT}}$')
    pm = 0.1
    ax.plot(pr, [pr[0] + pm, pr[1] + pm], 'k--', linewidth=1.0, label=r'$\pm %.2f \mathrm{eV}$' % pm)
    ax.plot([pr[0] + pm, pr[1]], [pr[0], pr[1] - pm], 'k--', linewidth=1.0)
 
    axins = ax.inset_axes([0.55, 0.1, 0.4, 0.4])
    ax.tick_params(axis='both', which='major', labelsize=16)
    axins.patch.set_alpha(0)
    axins.axvline(0.0, linestyle='-', linewidth=0.5, color='black')
    axins.axvline(-pm, linestyle='--', linewidth=0.5, color='black')
    axins.axvline(pm, linestyle='--', linewidth=0.5, color='black')
    axins.tick_params(axis='both', which='major', labelsize=14)
    axins.get_yaxis().set_visible(False)
    
    for spine in ['right', 'left', 'top']:
        axins.spines[spine].set_visible(False)
    
    tot_diff = np.array([])
    for i, pair in enumerate(arr):
        true, pred = np.array(pair[0]), np.array(pair[1])
        ax.scatter(true, pred, s=15, c=colors[i], alpha=0.5,zorder=0)
        diff = pred - true
        tot_diff = np.append(tot_diff,diff)
        axins.hist(diff, bins=20, range=(-3 * pm, 3 * pm), color=colors[i], alpha=0.5) 
        ax.text(0.01, 0.95-0.04*(i+1),
            f'{ads[i]:3}: {np.mean(np.abs(diff)):.3f} ({np.mean(diff):.3f}) eV',
            family='monospace', fontsize=16, transform=ax.transAxes,
            va='top', color=colors[i])

    ax.text(0.01, 0.95,
            f'Total MAE (ME): {np.mean(np.abs(tot_diff)):.3f} ({np.mean(tot_diff):.3f}) eV',
            family='monospace', fontsize=16, transform=ax.transAxes,
            va='top', color='k')

    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)

    return fig

def plot_3Dsimplex(ax, elements, verts, label_offset=0.05):
    lines = it.combinations(verts, 2)
    pairs = it.combinations(elements, 2)
    for x, p in zip(lines,pairs):
        if p == ('',''):
            continue
        line = np.transpose(np.array(x))
        ax.plot3D(line[0], line[1], line[2], c='0', alpha=0.5, linewidth=0.5, linestyle='--')

    c3d = barycentric2cartesian([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], verts)
    for i, coord in enumerate(c3d):
        # Calculate the midpoint of each edge
        midpoint = np.mean([verts[j] for j in [[1, 2], [0, 2], [0, 1], [0, 1, 2]][i]], axis=0)
        # Calculate the vector from the midpoint to the vertex and normalize it
        offset_vector = (coord - midpoint) / np.linalg.norm(coord - midpoint)
        # Adjust label position by adding the offset
        new_coord = coord + label_offset * offset_vector
        ax.text(new_coord[0], new_coord[1], new_coord[2], elements[i], size=11, va='center', ha='center')

def barycentric2cartesian(b, verts):
    t = np.transpose(np.array(verts))
    t_array = np.array([t.dot(x) for x in b])
    return t_array

class simplex2D():
    #def __init__(self):     

    def barycentric2cartesian(self, b):
        t = barycentric2cartesian(b, [[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
        return t
    
    def make_triangle_ticks(self, ax, start, stop, tick, n_ticks, offset=(0., 0.),
                        fontsize=12, ha='center', tick_labels=True):
        r = np.linspace(0, 1, n_ticks+1)
        x = start[0] * (1 - r) + stop[0] * r
        x = np.vstack((x, x + tick[0]))
        y = start[1] * (1 - r) + stop[1] * r
        y = np.vstack((y, y + tick[1]))
        ax.plot(x, y, 'black', lw=1., zorder=0)

        if tick_labels:

            # Add tick labels
            for xx, yy, rr in zip(x[0], y[0], r):
                ax.text(xx+offset[0], yy+offset[1], f'{rr*100.:.0f}',
                    fontsize=fontsize, ha=ha)

    def prepare_ax(self, ax, labels=None, fontsize=10, n_ticks=5, show_symbols=True, show_edges=True, show_ticklabels=True):
        # Set axis limits
        vertices = self.barycentric2cartesian(np.identity(3))
        h = 3**0.5/2
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, h+0.05)
            
        # Remove spines
        for direction in ax.spines.keys():
            ax.spines[direction].set_visible(False)

        # Remove tick and tick labels
        ax.tick_params(which='both', bottom=False, labelbottom=False, left=False, labelleft=False)
        ax.set_aspect('equal')

        if show_edges:
            # Plot triangle edges
            ax.plot([0., 0.5], [0., h], '-', color='black', zorder=0)
            ax.plot([0.5, 1.], [h, 0.], '-', color='black', zorder=0)
            ax.plot([0., 1.], [0., 0.], '-', color='black', zorder=0)

        if n_ticks !=0 and labels != None:
            
            # Make ticks and tick labels on the triangle axes
            left, right, top = vertices
            tick_size = 0.035
            bottom_ticks = 0.8264*tick_size * (right - top)
            right_ticks = 0.8264*tick_size * (top - left)
            left_ticks = 0.8264*tick_size * (left - right)

            # Make ticks on triangle edges..
            tick_kw = dict(n_ticks=n_ticks, tick_labels=show_ticklabels, fontsize=fontsize)

            # ..for the bottom axis 
            self.make_triangle_ticks(ax, right, left, bottom_ticks, offset=(0.03, -0.12), ha='center', **tick_kw)

            # ..for the left-hand axis
            self.make_triangle_ticks(ax, left, top, left_ticks, offset=(-0.05, -0.02), ha='right', **tick_kw)

            # ..and for the right-hand axis
            self.make_triangle_ticks(ax, top, right, right_ticks, offset=(0.02, 0.02), ha='left', **tick_kw)

            # Show axis labels along edges..
            kwargs = dict(fontsize=fontsize, ha='center', va='center')

            # ..for the bottom axis
            ax.text(0.50, -0.17, f'{labels[0]} content [at.%]', rotation=0., **kwargs)

            # ..for the right-hand axis
            ax.text(0.94,  0.54, f'{labels[1]} content [at.%]', rotation=-60., **kwargs)

            # ..and for the left-hand axis
            ax.text(0.02,  0.54, f'{labels[2]} content [at.%]', rotation=60., **kwargs)

        if show_symbols and labels != None:
            # Define padding to put the vertex text neatly
            pad = np.array([[-0.07, -0.06], # lower left vertex
                            [ 0.07, -0.06], # lower right vertex
                            [ 0.00,  0.12]])# upper vertex
            has = ['right', 'left', 'center']
            vas = ['top', 'top', 'bottom']
            # Show the element symbol at each vertex
            
            for idx, (r, dpad) in enumerate(zip(vertices, pad)):
                ax.text(*(r+dpad), s=labels[idx], fontsize=fontsize, ha=has[idx], va=vas[idx])#, color='none')

        # Return axis abject
        return ax
 


def curr_parity(trp, trt, tep, tet, string, limits, comp=[]):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.set_xlabel(r'Exp. current density [mA/cm$^2$]',fontsize=18)
    ax.set_ylabel(r'Pred. current density [mA/cm$^2$]', fontsize=18)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[0], limits[1])

    ax.scatter(np.array(trt), np.array(trp), c='grey', s=10, alpha=0.20)
    if len(comp) == 0:
        ax.scatter(np.array(tet), np.array(tep), c='crimson', s=10, alpha=0.80)
    else:
        ax.scatter(np.array(tet), np.array(tep), c=comp, cmap=cmap, s=10, alpha=0.8, vmin=0.0, vmax=0.75)

    # plot solid diagonal line
    ax.plot([limits[0], limits[1]], [limits[0], limits[1]], 'k-', linewidth=1.0)

    # plot dashed diagonal lines 0.1 eV above and below solid diagonal line
    pm = 0.1
    ax.plot([limits[0], limits[1]], [limits[0] + pm, limits[1] + pm], 'k--', linewidth=1.0, label=r'$\pm %.2f \mathrm{eV}$' % pm)
    ax.plot([limits[0] + pm, limits[1]], [limits[0], limits[1] - pm], 'k--', linewidth=1.0)

    ax.text(0.01, 0.99, string, family='monospace', fontsize=18, transform=ax.transAxes, va='top', color='k')
    ax.tick_params(labelsize=14)

    return fig
