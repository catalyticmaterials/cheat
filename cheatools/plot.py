import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools as it
from matplotlib.colors import ListedColormap
from ase.data import atomic_numbers
from ase.data.colors import jmol_colors

def get_color(element, whiteout_param=0):
    """Get jmol color with optional whiteout parameter."""
    return jmol_colors[atomic_numbers[element]] * (1 - whiteout_param) + whiteout_param

def get_dark_color(element):
    """Get darkened jmol color."""
    return jmol_colors[atomic_numbers[element]] / 2

def get_colormap(color1,color2):
    """Get colormap between two colors."""
    vals = np.ones((256, 3))
    vals[:, 0] = np.linspace(color1[0], color2[0], 256)
    vals[:, 1] = np.linspace(color1[1], color2[1], 256)
    vals[:, 2] = np.linspace(color1[2], color2[2], 256)
    return ListedColormap(vals)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Truncate colormap to a range of values."""
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval), # naming the new colormap e.g. 'trunc(viridis, 0.2, 0.8)'
        cmap(np.linspace(minval, maxval, n))) # sampling the new colormap at n points between minval and maxval
    return new_cmap

def plot_parity(true_dict, pred_dict, colors, header, limit=[-0.5,2.5]):
    """ 
    Parity plot function for multiple adsorbates
    ------
    true_dict and pred_dict should contain adsorbate keys and lists of energies e.g. {'OH':[list of energies]}
    """
    # create figure and set axis labels
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlabel(r'$\Delta \mathrm{E}^{\mathrm{DFT}}_{\mathrm{ads}}$ [eV]',fontsize=22, labelpad=10)
    ax.set_ylabel(r'$\Delta \mathrm{E}^{\mathrm{model}}_{\mathrm{ads}}$ [eV]', fontsize=22, labelpad=0)
    
    # set range, ticks, and add top text
    ax.set(xlim=limit,ylim=limit)
    ax.text(0.01, 0.99, header, family='monospace', fontsize=16, transform=ax.transAxes,va='top', color='k')
    ax.tick_params(axis='both', which='major', labelsize=16)

    # plot diagonal lines
    ax.plot(limit, limit, 'k-', linewidth=1.0,label=r'$\Delta \mathrm{E}^{\mathrm{pred}} = \Delta \mathrm{E}^{\mathrm{DFT}}$')
    pm = 0.1
    ax.plot(limit, [limit[0] + pm, limit[1] + pm], 'k--', linewidth=1.0, label=r'$\pm %.2f \mathrm{eV}$' % pm)
    ax.plot([limit[0] + pm, limit[1]], [limit[0], limit[1] - pm], 'k--', linewidth=1.0)
 
    # inset transparent histogram ax
    axins = ax.inset_axes([0.55, 0.1, 0.4, 0.4])
    axins.patch.set_alpha(0)
    for spine in ['right', 'left', 'top']:
        axins.spines[spine].set_visible(False)
    
    # plot vertical lines in histogram
    axins.axvline(0.0, linestyle='-', linewidth=0.5, color='black')
    axins.axvline(-pm, linestyle='--', linewidth=0.5, color='black')
    axins.axvline(pm, linestyle='--', linewidth=0.5, color='black')
    axins.tick_params(axis='both', which='major', labelsize=14)
    axins.get_yaxis().set_visible(False)
    
    # loop through adsorbates and plot
    tot_diff = np.array([])
    for i, ads in enumerate(true_dict.keys()):
        true, pred = np.array(true_dict[ads]), np.array(pred_dict[ads])
        diff = pred - true
        tot_diff = np.append(tot_diff, diff) # differences to total difference array

        ax.scatter(true, pred, s=15, c=colors[i], alpha=0.5,zorder=0) # plot scatter
        axins.hist(diff, bins=20, range=(-3 * pm, 3 * pm), color=colors[i], alpha=0.5) # plot histogram

        # add MAE and ME for adsorbate
        ax.text(0.01, 0.95-0.04*(i+1),
            f'{ads:3}: {np.mean(np.abs(diff)):.3f} ({np.mean(diff):.3f}) eV',
            family='monospace', fontsize=16, transform=ax.transAxes,
            va='top', color=colors[i])

    # add total MAE and ME
    ax.text(0.01, 0.95,
            f'Total MAE (ME): {np.mean(np.abs(tot_diff)):.3f} ({np.mean(tot_diff):.3f}) eV',
            family='monospace', fontsize=16, transform=ax.transAxes,
            va='top', color='k')

    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)

    return fig

class simplex2D():
    """
    A class for plotting a 2D simplex
    ------
    Methods:
        plot(): Plot the simplex with optional labels and ticks
        make_triangle_ticks(): Used by plot() to place ticks
        comps2coords(): Transform compositions to coordinates
    """
    def __init__(self):
        self.verts = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])

    def plot(self, labels=None, n_ticks=5, show_edges=True, show_cornerlabels=True, show_axlabels=True, show_ticklabels=True):
        """Plot the simplex with optional labels and ticks."""
        # Create blank figure with limits
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set(xlim = (-0.05, 1.05), ylim = (-0.05,self.verts[2][1]+0.05))
        for direction in ax.spines.keys():
            ax.spines[direction].set_visible(False)
        ax.tick_params(which='both', bottom=False, labelbottom=False, left=False, labelleft=False)
        ax.set_aspect('equal')

        # Plotting the simplex borders
        if show_edges:
            for i in range(3):
                j = (i + 1) % 3  # Index of the next point, wrapping around to 0 at the end
                ax.plot([self.verts[i][0], self.verts[j][0]], [self.verts[i][1], self.verts[j][1]],'-', color='black')

        if show_ticklabels:
            # Make ticks on triangle edges
            l, r, t = self.verts
            tick_size = 0.025
            tick_kw = dict(n_ticks=n_ticks, tick_labels=show_ticklabels)

            self.make_triangle_ticks(ax, r, l, 0.8264*tick_size * (r - t), offset=(0.025, -0.065), ha='center', **tick_kw) # bottom axis
            self.make_triangle_ticks(ax, l, t, 0.8264*tick_size * (l - r), offset=(-0.03, -0.015), ha='right', **tick_kw) # left-hand axis
            self.make_triangle_ticks(ax, t, r, 0.8264*tick_size * (t - l), offset=(0.01, 0.035), ha='left', **tick_kw) # right-hand axis

        if show_axlabels:
            # Show axis labels along edges
            label_kw = dict(fontsize=18, ha='center', va='center')
            ax.text(0.50, -0.13, f'{labels[0]} content [at.%]', rotation=0., **label_kw) # bottom axis
            ax.text(0.87,  0.51, f'{labels[1]} content [at.%]', rotation=-60., **label_kw) # right-hand axis
            ax.text(0.13,  0.51, f'{labels[2]} content [at.%]', rotation=60., **label_kw) # left-hand axis

        if show_cornerlabels:
            # Define padding to put the vertex text neatly
            pad = np.array([[-0.07, -0.06], # lower left vertex
                            [ 0.07, -0.06], # lower right vertex
                            [ 0.00,  0.10]])# upper vertex
            has = ['right', 'left', 'center']
            vas = ['top', 'top', 'bottom']

            # Show the element symbol at each vertex
            for idx, (r, dpad) in enumerate(zip(self.verts, pad)):
                ax.text(*(r+dpad), s=labels[idx], fontsize=32, ha=has[idx], va=vas[idx])

        return fig

    def comps2coords(self,comps):
        """Transform compositions to coordinates"""
        return np.dot(np.array(comps),self.verts).T
    
    def make_triangle_ticks(self, ax, start, stop, tick, n_ticks, offset=(0., 0.), fontsize=15, ha='center', tick_labels=True):
        """Make ticks on the simplex edges."""
        r = np.linspace(0, 1, n_ticks+1)
        x = start[0] * (1 - r) + stop[0] * r
        x = np.vstack((x, x + tick[0]))
        y = start[1] * (1 - r) + stop[1] * r
        y = np.vstack((y, y + tick[1]))
        ax.plot(x, y, 'black', lw=1., zorder=0)

        if tick_labels:
            for xx, yy, rr in zip(x[0], y[0], r):
                ax.text(xx+offset[0], yy+offset[1], f'{rr*100.:.0f}',
                    fontsize=fontsize, ha=ha)
 
class simplex3D():
    """
    A class for plotting a 3D simplex
    ------
    Methods:
        plot(): Plot the simplex with optional labels and ticks
        comps2coords(): Transform compositions to coordinates
    """
    def __init__(self):
        self.verts = np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3)/2, 0], [0.5, 0.28867513, 0.81649658]])

    def plot(self,labels=None):
        """Plot the simplex with optional labels."""
        # Create blank figure and center 3D simplex
        fig, ax = plt.subplots(1,1,figsize=(5.5, 5),subplot_kw=dict(projection='3d'))
        ax.view_init(elev=15., azim=300)
        ax.set_axis_off()
        fig.tight_layout(rect=[-0.25, -0.25, 1.55, 1.35])

        # Plot the simplex edges
        lines = it.combinations(self.verts, 2)
        for l in lines:
            line = np.transpose(np.array(l))
            ax.plot3D(line[0], line[1], line[2], c='0', alpha=0.5, linewidth=0.5, linestyle='--')
        
        # Put labels on the simplex vertices
        c3d = self.comps2coords([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]).T
        for i, coord in enumerate(c3d):
            midpoint = np.mean([self.verts[j] for j in [[1, 2], [0, 2], [0, 1], [0, 1, 2]][i]], axis=0) # Calculate the midpoint of each edge
            offset_vector = (coord - midpoint) / np.linalg.norm(coord - midpoint) # Calculate the vector from the midpoint to the vertex and normalize it
            new_coord = coord + 0.075 * offset_vector # Adjust label position by adding the offset
            ax.text(new_coord[0], new_coord[1], new_coord[2], labels[i], size=16, va='center', ha='center')

        return fig
    
    def comps2coords(self,comps):
        """Transform compositions to coordinates"""
        return np.dot(np.array(comps),self.verts).T

class orthographic_projection():
    """
    A class to plot orthographic projections
    ------
    Initilize with a list elements 

    Methods:
        plot(): Get projection of the requested elements as nodes connected by edges
        comps2coords(): Transform compositions to projection coordinates
    """
    def __init__(self, elements, radius=6):

        self.elements = elements

        theta = np.linspace(0, 2*np.pi, len(self.elements), endpoint=False) + np.pi/2
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        self.verts = list(zip(x,y))

    def plot(self):
        """Plot projection as nodes connected by edges"""
        fig = plt.figure(figsize=(8,8))
        N = len(self.elements)
        for i, xy in enumerate(self.verts):
            plt.plot(1.1*xy[0], 1.1*xy[1], marker='o', markeredgecolor=get_dark_color(self.elements[i]), markerfacecolor=get_color(self.elements[i]), markersize=25, markeredgewidth=1, zorder=1)
            plt.text(1.25*xy[0], 1.25*xy[1], self.elements[i], fontsize=18, ha='center', va='center', color='black', zorder=2)

        for i in range(N):
            j = (i + 1) % N  # Index of the next point, wrapping around to 0 at the end
            plt.plot([self.verts[i][0], self.verts[j][0]], [self.verts[i][1], self.verts[j][1]], 'darkslategrey', lw=0.5, alpha=0.5, zorder=0)  # 'k-' for black lines

        for i in range(N):
            for j in range(i+1, N):
                plt.plot([self.verts[i][0], self.verts[j][0]], [self.verts[i][1], self.verts[j][1]], 'darkslategrey', lw=0.5, alpha=0.5, zorder=0)  # 'k-' for black lines

        plt.gca().set_aspect('equal', adjustable='box')  # Equal aspect ratio
        for side in ['top','right','bottom','left']:
            plt.gca().spines[side].set_visible(False)

        plt.xticks([])
        plt.yticks([])

        return fig

    def comps2coords(self,comps):
        """Transform compositions to coordinates"""
        return np.dot(np.array(comps),self.verts).T