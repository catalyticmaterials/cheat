import numpy as np
import itertools as it
import iteround
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
#from .dftsampling import write_hea_slab, add_ads
from .plot import get_color, get_dark_color
from .graphtools import templater
import matplotlib.pyplot as plt
import ase
from copy import deepcopy

class SurrogateSurface():
    def __init__(self, composition, adsorbates, sites, regressor, template='ocp', facet='fcc111', size=(96,96), 
                 displace_e=None, scale_e=None, direct_e_input=None):
        
        if not isinstance(composition,list):
            composition = [composition] 

        # Graph templates
        self.template = template
        if template == 'lgnn':
            for comp in composition:
                ok = [m for m in regressor.onehot_labels if m not in ''.join(adsorbates)]
                if len(list(set(comp.keys()) - set(ok))) != 0:
                    raise Exception(f"To match lGNN state the composition should contain only {ok}")
            from .graphtools import lGNNtemplater
            self.templater = lGNNtemplater(facet,adsorbates,sites,regressor.onehot_labels)
       
        elif template == 'ocp':
            from .ocp_utils import OCPtemplater 
            self.templater = OCPtemplater(facet,adsorbates,sites)
        
        # Metal parameters
        self.metals = list(set([k for k in comp.keys() for comp in composition]))
        self.n_metals = len(self.metals)

        # Adsorbates parameters
        self.adsorbates = adsorbates
        self.sites = sites
        self.displace_e = np.zeros(len(adsorbates)) if displace_e ==  None else displace_e
        self.scale_e = np.ones(len(adsorbates)) if scale_e ==  None else scale_e
        self.direct_e_input = direct_e_input
    
        # Load regressor and construct edge-templates of all ads/site combinations
        self.regressor = regressor

        # Size and surface parameters
        self.facet = facet
        self.size = size
        self.nrows, self.ncols = size

        # Number of layers scaled to what rank neighbors used in regression
        self.n_layers = 5 if template == 'ocp' else 3 if template == 'lgnn' else None
        
        # Assign elements to grid (New version)
        n_atoms_surface = np.prod(size)
        if len(composition) == 1:
            n_atoms = n_atoms_surface * self.n_layers
            metal_ids = np.random.choice(range(self.n_metals),size=n_atoms,p=[composition[0][metal] for metal in self.metals])
        
        elif len(composition) > 1:
            assert len(composition) == self.n_layers, f"For template {template} please provide a list of {self.n_layers} compositions (one for each layer) or a single composition."
            
            metal_id_arr = []
            for comp in composition:
                metal_id_arr.append(np.random.choice(range(self.n_metals),size=n_atoms_surface,p=[comp[m] if m in comp else 0.0 for m in self.metals]))
            metal_ids = []
            for l in zip(*metal_id_arr):
                metal_ids.extend([*l])
            
        self.grid = np.reshape(metal_ids, (*size, self.n_layers))

    def get_gross_energies(self):
        """Regression of adsorption energies of all possible surface sites of the chosen types."""

        # Prep dicts for gross ads. energy grids and adsorption bool grid
        self.grid_dict_gross, self.ads_dict = {}, {}

        # Loop through chosen ads/site combinations and regress gross ads. energy grid
        for i, adsorbate in enumerate(self.adsorbates):
            # Grid for tracking gross ads. energy of ads/site comb.
            self.grid_dict_gross[(adsorbate, self.sites[i])] = np.zeros(self.size)
            # Grid for tracking occupation of sites
            self.ads_dict[(adsorbate, self.sites[i])] = np.zeros(self.size, dtype=bool)

            # Collect graph features for ads/site comb. on surface
            feat_list = []
            for row in range(self.nrows):
                for col in range(self.ncols):
                    coords = (row, col, 0)
                    feat_list.append(self.get_cell(coords, adsorbate, self.sites[i]))
                    #if self.template in ['ocp','shallow_ocp']:
                    #    feat_list.append(self.get_ocp_cell(len(feat_list), coords, adsorbate, self.sites[i]))
                    #elif self.template == 'lgnn':
                    #    feat_list.append(self.get_lgnn_cell(coords, adsorbate, self.sites[i]))
            
            # NB! Order has been changed to displace before scale. This is different from previous versions.
            if self.direct_e_input != None:
                pred = (np.ones(np.prod(self.size)) * self.direct_e_input[i] + self.displace_e[i]) * self.scale_e[i]
            else:
                pred = (self.regressor.predict(feat_list,tqdm_bool=True) + self.displace_e[i]) * self.scale_e[i]
            #elif self.template in ['ocp','shallow_ocp']:
            #    pred = (self.regressor.direct_prediction(feat_list,tqdm_bool=False) + self.displace_e[i]) * self.scale_e[i]
            #elif self.template == 'lgnn':
            #    pred_loader = DataLoader(feat_list, batch_size=len(feat_list))
            #    pred, _ = predict(self.regressor, pred_loader, len(feat_list))
            #    pred = (np.array(pred) + self.displace_e[i]) * self.scale_e[i]   
        
            self.grid_dict_gross[(adsorbate, self.sites[i])] = np.reshape(pred, self.size)

        return self

    def get_net_energies(self):
        """Determine net ads. energies after mutual blocking of neighboring sites"""

        # Establish gross ads. energy grids if not already present
        try:
            isinstance(self.grid_dict_gross, dict)
        except:
            self.get_gross_energies()

        # Prep dict with maskable copies of gross ads. energy grids
        # NB! Important that we make an explicit copy, because imposed energy contribution otherwise
        # will alter the masked gross energies
        self.grid_dict_net = {}
        for key in self.grid_dict_gross.keys():
            self.grid_dict_net[key] = np.ma.masked_array(np.copy(self.grid_dict_gross[key]))

        # append figure of clean surface
        #if self.surf_images:
        #    self.surf_image_list.append(self.plot_surface())

        # Occupy most stable site and block neighbors until surface is filled
        while True:
            min_e = np.inf
            # Loop through ads/site combs
            for key in self.grid_dict_net.keys():
                # Determine if most stable site is the most stable of all ads/site combs
                if np.min(self.grid_dict_net[key]) < min_e:
                    min_e, min_e_key = self.grid_dict_net[key].min(), key

            # if multiple sites are equal in adsorption energy => pick first one with most neighbors already block to maximize coverage
            if 1 < np.count_nonzero(min_e == self.grid_dict_net[min_e_key]):
                ids = np.arange(0, np.prod(self.size[:2])).reshape(self.grid_dict_net[min_e_key].shape)[min_e == self.grid_dict_net[min_e_key]]
                neighbor_dict = self.get_neighbor_ids(self.facet, min_e_key[1])
                overlap_list = []
                count = 0
                best_id = False
                for id in ids:
                    count += 1
                    id_coords = np.unravel_index(id, self.grid_dict_net[min_e_key].shape)
                    overlap_count = 0
                    for block_key in neighbor_dict.keys():
                        #if block_key[0] != min_e_key[1] and block_key[1] == 1:
                        #    overlap_count += np.count_nonzero(self.grid_dict_net[min_e_key].mask[tuple(((id_coords + neighbor_dict[block_key]) % self.size).T)])
                        if block_key[0] == min_e_key[1] and block_key[1] == 1:
                            if min_e_key[1] == 'fcc':
                                try:
                                    overlap_count += np.count_nonzero(self.grid_dict_net[min_e_key].mask[tuple(((id_coords + neighbor_dict[block_key]) % self.size).T)])
                                except IndexError:
                                    pass

                            elif min_e_key[1] == 'ontop':
                                taken_sites = np.where(self.ads_dict[min_e_key][tuple(
                                    ((id_coords + neighbor_dict[block_key]) % self.size).T)] == True)[0]
                                if len(taken_sites) != 0:
                                    # Block shared neighbors
                                    for i in taken_sites:
                                        overlap_count += np.count_nonzero(self.grid_dict_net[min_e_key].mask[
                                            tuple(((id_coords + neighbor_dict[('ontop', 1)][
                                                (i - 6) + 1]) % self.size).T)])

                                        overlap_count += np.count_nonzero(self.grid_dict_net[min_e_key].mask[
                                            tuple(((id_coords + neighbor_dict[('ontop', 1)][
                                                i - 1]) % self.size).T)])
                    #overlap_list.append(overlap_count)

                    if min_e_key[1] == 'fcc' and overlap_count > 1:
                        best_id = id
                        break
                    elif min_e_key[1] == 'ontop' and overlap_count > 0:
                        best_id = id
                        break

                if not best_id:
                    min_e_coords = np.unravel_index(ids[0], self.grid_dict_net[min_e_key].shape)
                else:
                    min_e_coords = np.unravel_index(best_id, self.grid_dict_net[min_e_key].shape)

            else:
                min_e_coords = np.unravel_index(self.grid_dict_net[min_e_key].argmin(),self.grid_dict_net[min_e_key].shape)

            # Break if no available sites left
            if min_e == np.inf:
                break

            # Mark site as occupied and mask ads energy
            self.ads_dict[min_e_key][min_e_coords] = True

            # append figure of added adsorbate
            #if self.surf_images:
            #    self.surf_image_list.append(self.plot_surface())

            self.block_sites(min_e_key,min_e_coords)

        return self

    def block_sites(self,min_e_key,min_e_coords):
        self.grid_dict_net[min_e_key][min_e_coords] = np.ma.masked

        # Get neighbor coordinates dictionary
        neighbor_dict = self.get_neighbor_ids(self.facet, min_e_key[1])

        # Loop through site keys block correspondingly
        for block_key in neighbor_dict.keys():
            for key in self.grid_dict_net.keys():
                if block_key[0] == key[1] and block_key[1] == 0:
                    # Block adjoining opposite site neighbors
                    self.grid_dict_net[key][
                        tuple(((min_e_coords + neighbor_dict[block_key]) % self.size).T)] = np.ma.masked

                elif block_key[0] == key[1] and block_key[1] == 1:
                    # Separate the different situations and block neighbor atoms or impose energy accordingly.
                    if min_e_key[1] == 'fcc':
                        if block_key[0] == 'fcc':
                            self.grid_dict_net[key][
                                tuple(((min_e_coords + neighbor_dict[block_key]) % self.size).T)] = np.ma.masked

                    elif min_e_key[1] == 'ontop':
                        if block_key[0] == 'ontop':
                            # Examine if there is any adsorbed *OH on 1st rank neighbor sites.
                            taken_sites = np.where(self.ads_dict[key][tuple(
                                ((min_e_coords + neighbor_dict[block_key]) % self.size).T)] == True)[0]
                            if len(taken_sites) != 0:
                                # Block shared neighbors
                                for i in taken_sites:
                                    self.grid_dict_net[key][tuple(((min_e_coords + neighbor_dict[('ontop', 1)][
                                        (i - 6) + 1]) % self.size).T)] = np.ma.masked

                                    self.grid_dict_net[key][tuple(((min_e_coords + neighbor_dict[('ontop', 1)][
                                        i - 1]) % self.size).T)] = np.ma.masked

    def get_cell(self, coords, adsorbate, site):
        # Get ordered list of coordinates of atoms included in graph
        if self.facet == 'fcc111':
            t = [-1,0,1]
            site_ids = np.array([(a,b,c) for c in range(self.n_layers)[::-1] for a in t for b in t])

        # Get ordered list of element labels of atoms in graph
        site_labels = self.grid[tuple(((site_ids + coords) % [*self.size, self.n_layers + 1]).T)]
        site_symbols = [self.metals[t] for t in site_labels]
        cell = self.templater.fill_template(site_symbols,adsorbate,site)
        return cell
        
    """
    def get_ocp_cell(self, sid, coords, adsorbate, site):
        Get the OCP-compatible graph of the requested site from the cell-templates

        NB! Graphs use torch edge-templates from adjacency matrix of ASE model system.
        Hence site_ids are listed in the order matching the edge-template and will result in
        mismatch between node-list and edge-list if changed.

        Coordinates are structured as (row,coloumn,layer) with surface layer being 0, subsurface 1 etc.
        
        # Get ordered list of coordinates of atoms included in graph
        if self.facet == 'fcc111':
            t = [-1,0,1]
            site_ids = np.array([(a,b,c) for c in range(self.n_layers)[::-1] for a in t for b in t])
        
        # Get ordered list of element labels of atoms in graph
        site_labels = self.grid[tuple(((site_ids + coords) % [*self.size, self.n_layers + 1]).T)]
        cell = deepcopy(self.templates[(adsorbate, site)])
        cell.atomic_numbers[:int(self.n_layers*9)] = torch.tensor([ase.data.atomic_numbers[self.metals[t]] for t in site_labels])
        cell.sid = sid
        cell.fid = 0

        return cell

    def get_lgnn_cell(self, coords, adsorbate, site):
        # Get ordered list of coordinates of atoms included in graph
        if self.facet == 'fcc111':
            t = [-1,0,1]
            site_ids = np.array([(a,b,c) for c in range(self.n_layers)[::-1] for a in t for b in t])
        
        site_labels = self.grid[tuple(((site_ids + coords) % [*self.size, self.n_layers + 1]).T)]
        cell = deepcopy(self.templates[(adsorbate, site)])
        for i, label in enumerate(site_labels):
            cell.x[i, cell.onehot_labels.index(self.metals[label])] = 1
        return cell        
    """

    def get_neighbor_ids(self, facet, site):
        """Return neighboring coordinates for blocking or for energy contribution"""
        if facet == 'fcc111':
            if site == 'ontop':
                neighbor_dict = {('fcc', 0): np.array([(-1, 0), (0, -1), (0, 0)]),
                                 ('ontop', 1): np.array([(-1, 0), (0, -1), (1, -1), (1, 0), (0, 1), (-1, 1)]),
                                 ('fcc', 1): np.array([(-1, 1), (1, -1), (-1, -1)]),
                                 ('ontop', 2): np.array([(1, 1), (-1, 2), (-2, 1), (-1, -1), (1, -2), (2, -1)]),
                                 ('fcc', 2): np.array([(1, 0), (0, 1), (-2, 1), (-2, 0), (0, -2), (1, -2)])
                                 }

            elif site == 'fcc':
                neighbor_dict = {('ontop',0): np.array([(1, 0), (0, 1), (0, 0)]),
                                ('fcc',1): np.array([(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]),
                                ('ontop',1): np.array([(1, -1), (-1, 1), (1, 1)]),
                                ('fcc',2): np.array([(-1, -1), (1, -2), (2, -1), (1, 1), (-1, 2), (-2, 1)]),
                                ('ontop',2): np.array([(-1, 0), (0, -1), (2, -1), (2, 0), (0, 2), (-1, 2)])
                                }

        return neighbor_dict

    def plot_surface(self, show_ads = True):
        """Plot surface and adsorbates. Currently only implemented for:
        facets: fcc111
        adsorbates: OH, O
        sites: ontop, fcc
        """

        if self.size[0] != self.size[1]:
            raise ValueError("Set surface side lengths equal before plotting surface.")

        # Initialize figure
        fig, ax = plt.subplots(1, 1, figsize=(self.size[0], self.size[0] * 0.85))

        # Loop through layers
        for i in np.arange(3)[::-1]:

            # Shift layers to emulate fcc packing
            if i == 2:
                layer_xshift = 0.5
                layer_yshift = 0.25
                alpha = 1
                grey = 0.6
            elif i == 1:
                layer_xshift = 0
                layer_yshift = 0.5
                alpha = 1
                grey = 0.3
            elif i == 0:
                layer_xshift = 0.000
                layer_yshift = 0.000
                alpha = 1
                grey = 0

            # Plot each atom and shift atoms to make orthogonal surface
            for j, row in enumerate(self.grid[:, :, i]):
                for k, atom in enumerate(row):
                    # This if/else-statement will shift the atoms "beyond the base of the parallelogram" to the other side
                    if k > len(row) - 1 - np.floor(j / 2):
                        row_shift = 0.5 * j - len(row)
                    else:
                        row_shift = 0.5 * j

                    ax.scatter(k + row_shift + layer_xshift,
                               j + layer_yshift,
                               s=1500,
                               color=get_color(list(self.metals)[atom], whiteout_param=grey),
                               edgecolor='black',
                               alpha=alpha)
        if show_ads:
            # Plot occupied OH ontop sites
            for j, row in enumerate(self.ads_dict[('OH', 'ontop')]):
                for k, bool in enumerate(row):
                    if k > len(row) - 1 - np.floor(j / 2):
                        row_shift = 0.5 * j - len(row)
                    else:
                        row_shift = 0.5 * j
                    if bool:
                        ax.scatter(k + row_shift,
                                   j,
                                   s=500,
                                   color='red',
                                   edgecolor='black')
                        ax.scatter(k + row_shift + 0.15,
                                   j + 0.15,
                                   s=200,
                                   color='white',
                                   edgecolor='black')

            # Plot occupied O fcc sites
            for j, row in enumerate(self.ads_dict[('O', 'fcc')]):
                layer_xshift = 0.5
                layer_yshift = 0.25
                for k, bool in enumerate(row):
                    if k > len(row) - 1 - np.floor(j / 2):
                        row_shift = 0.5 * j - len(row)
                    else:
                        row_shift = 0.5 * j
                    if bool:
                        ax.scatter(k + row_shift + layer_xshift,
                                   j + layer_yshift,
                                   s=500,
                                   color='red',
                                   edgecolor='black')
        
        # Set figure parameters
        ax.set(xlim=(-1, self.size[1] + 1), ylim=(-1, self.size[0] + 1))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for spine in ['right', 'left', 'top', 'bottom']:
            ax.spines[spine].set_visible(False)

        return fig
