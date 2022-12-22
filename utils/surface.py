import numpy as np
import itertools as it
import iteround
import torch
from torch_geometric.data import Data, DataLoader
from .regression import predict
from .plot import get_color, get_dark_color
from .edges import get_edges
import matplotlib.pyplot as plt

class BruteForceSurface():
    def __init__(self, composition, adsorbates, ads_atoms, sites, coordinates, height,
                 regressor, feat_type, n_neighbors, facet, size, displace_e, scale_e, surf_images=False):

        self.surf_images = surf_images
        self.surf_image_list = []

        # Metal parameters
        self.metals = sorted(list(composition.keys()))
        self.n_metals = len(self.metals)
        self.alloy = ''.join(self.metals)

        # Adsorbates parameters
        self.adsorbates = adsorbates
        self.sites = sites
        self.all_elem = self.metals + ads_atoms
        self.displace_e = displace_e
        self.scale_e = scale_e

        # Load regressor and construct edge-templates of all ads/site combinations
        self.regressor = regressor
        self.edge_dict = get_edges(facet, adsorbates, sites, n_neighbors, coordinates, height)
        self.feat_type = feat_type

        # Size and surface parameters
        self.facet = facet
        self.size = size
        self.nrows, self.ncols = size

        # Number of layers scaled to what rank neighbors used in regression
        self.n_neighbors = n_neighbors
        if n_neighbors == 0:
            self.n_layers = 1
        elif n_neighbors == 1:
            self.n_layers = 2
        elif n_neighbors == 2:
            self.n_layers = 3

        # Get the number of atoms of each surface element (as close to comp as possible)
        n_atoms_surface = np.prod(size)
        n_atoms = n_atoms_surface * self.n_layers
        comp = np.asarray([composition[metal] for metal in self.metals])
        n_each_metal = {metal_idx: comp[metal_idx] * n_atoms for metal_idx in range(self.n_metals)}
        n_each_metal = iteround.saferound(n_each_metal, 0)

        # Shuffle list of surface element ids and set up 3D grid
        metal_ids = list(it.chain.from_iterable([[metal_idx] * int(n) for metal_idx, n in n_each_metal.items()]))
        np.random.shuffle(metal_ids)
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
                    if self.feat_type == 'zonefeats':
                        feat_list.append(self.get_zonefeat(coords, self.sites[i]))
                    elif self.feat_type == 'graphs':
                        feat_list.append(self.get_graph(coords, adsorbate, self.sites[i]))

            if self.feat_type == 'zonefeats':
                pred = []
                for site in feat_list:
                    pred.append(self.regressor[f'{self.sites[i]}_{adsorbate}'][tuple(site[:self.n_metals])].predict(site[self.n_metals:].reshape(1,-1)) * self.scale_e[i] + self.displace_e[i])

            elif self.feat_type == 'graphs':
                # Prep DataLoader, predict ads energies and store in gross ads. energy grid
                pred_loader = DataLoader(feat_list, batch_size=len(feat_list))
                pred, _, _ = predict(self.regressor, pred_loader, len(feat_list))
                pred = np.array(pred) * self.scale_e[i] + self.displace_e[i]

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
        if self.surf_images:
            self.surf_image_list.append(self.plot_surface())

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
                ids = np.arange(0, np.product(self.size[:2])).reshape(self.grid_dict_net[min_e_key].shape)[min_e == self.grid_dict_net[min_e_key]]
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
            if self.surf_images:
                self.surf_image_list.append(self.plot_surface())

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

    def get_graph(self, coords, adsorbate, site):
        """Construct graph feature of requested site

        NB! Graphs use torch edge-templates from adjacency matrix of ASE model system.
        Hence site_ids are listed in the order matching the edge-template and will result in
        mismatch between node-list and edge-list if changed.

        Coordinates are structured as (row,coloumn,layer) with surface layer being 0, subsurface 1 etc."""

        # Get ordered list of coordinates of atoms included in graph
        if self.facet == 'fcc111':
            if site == 'ontop':
                if self.n_neighbors == 0:
                    site_ids = np.array([(0, 0, 0)])
                elif self.n_neighbors == 1:
                    site_ids = np.array([(-1, 0, 1),
                                         (-1, 1, 1),
                                         (0, 0, 1),
                                         (-1, 0, 0),
                                         (-1, 1, 0),
                                         (0, -1, 0),
                                         (0, 0, 0),
                                         (0, 1, 0),
                                         (1, -1, 0),
                                         (1, 0, 0)
                                         ])
                elif self.n_neighbors == 2:
                    site_ids = np.array([(-1, -1, 2),  #aoi
                                         (-1, 0, 2),
                                         (-1, 1, 2),  #aoi
                                         (0, -1, 2),
                                         (0, 0, 2),
                                         (1, -1, 2),  #aoi
                                         (-2, 0, 1),
                                         (-2, 1, 1),
                                         (-2, 2, 1),
                                         (-1, -1, 1),
                                         (-1, 0, 1),
                                         (-1, 1, 1),
                                         (-1, 2, 1),
                                         (0, -1, 1),
                                         (0, 0, 1),
                                         (0, 1, 1),
                                         (1, -1, 1),
                                         (1, 0, 1),
                                         (-2, 0, 0),  #aoi
                                         (-2, 1, 0),
                                         (-2, 2, 0),  #aoi
                                         (-1, -1, 0),
                                         (-1, 0, 0),
                                         (-1, 1, 0),
                                         (-1, 2, 0),
                                         (0, -2, 0),  #aoi
                                         (0, -1, 0),
                                         (0, 0, 0),
                                         (0, 1, 0),
                                         (0, 2, 0),  #aoi
                                         (1, -2, 0),
                                         (1, -1, 0),
                                         (1, 0, 0),
                                         (1, 1, 0),
                                         (2, -2, 0),  #aoi
                                         (2, -1, 0),
                                         (2, 0, 0)  #aoi
                                         ])
                    aoi_ids = [0, 2, 5, 18, 20, 25, 29, 34, 36]


            if site == 'fcc':
                if self.n_neighbors == 0:
                    site_ids = np.array([(0, 0, 0),
                                         (0, 1, 0),
                                         (1, 0, 0)
                                         ])
                elif self.n_neighbors == 1:
                    site_ids = np.array([(-1, 0, 1),
                                         (-1, 1, 1),
                                         (-1, 2, 1),
                                         (0, 0, 1),
                                         (0, 1, 1),
                                         (1, 0, 1),
                                         (-1, 0, 0),
                                         (-1, 1, 0),
                                         (-1, 2, 0),
                                         (0, -1, 0),
                                         (0, 0, 0),
                                         (0, 1, 0),
                                         (0, 2, 0),
                                         (1, -1, 0),
                                         (1, 0, 0),
                                         (1, 1, 0),
                                         (2, -1, 0),
                                         (2, 0, 0)
                                         ])
                elif self.n_neighbors == 2:
                    site_ids = np.array([(-1, -1, 2),  #aoi
                                         (-1, 0, 2),
                                         (-1, 1, 2),
                                         (-1, 2, 2),  #aoi
                                         (0, -1, 2),
                                         (0, 0, 2),
                                         (0, 1, 2),
                                         (1, -1, 2),
                                         (1, 0, 2),
                                         (2, -1, 2),  #aoi
                                         (-2, 0, 1),
                                         (-2, 1, 1),
                                         (-2, 2, 1),
                                         (-2, 3, 1),
                                         (-1, -1, 1),
                                         (-1, 0, 1),
                                         (-1, 1, 1),
                                         (-1, 2, 1),
                                         (-1, 3, 1),
                                         (0, -1, 1),
                                         (0, 0, 1),
                                         (0, 1, 1),
                                         (0, 2, 1),
                                         (1, -1, 1),
                                         (1, 0, 1),
                                         (1, 1, 1),
                                         (2, -1, 1),
                                         (2, 0, 1),
                                         (-2, 0, 0),  #aoi
                                         (-2, 1, 0),
                                         (-2, 2, 0),
                                         (-2, 3, 0),  #aoi
                                         (-1, -1, 0),
                                         (-1, 0, 0),
                                         (-1, 1, 0),
                                         (-1, 2, 0),
                                         (-1, 3, 0),
                                         (0, -2, 0),  #aoi
                                         (0, -1, 0),
                                         (0, 0, 0),
                                         (0, 1, 0),
                                         (0, 2, 0),
                                         (0, 3, 0),  #aoi
                                         (1, -2, 0),
                                         (1, -1, 0),
                                         (1, 0, 0),
                                         (1, 1, 0),
                                         (1, 2, 0),
                                         (2, -2, 0),
                                         (2, -1, 0),
                                         (2, 0, 0),
                                         (2, 1, 0),
                                         (3, -2, 0),  #aoi
                                         (3, -1, 0),
                                         (3, 0, 0)  #aoi
                                         ])
                    aoi_ids = [0, 3, 9, 28, 31, 37, 42, 52, 54]

        # Get ordered list of element labels of atoms in graph
        site_labels = self.grid[tuple(((site_ids + coords) % [*self.size, self.n_layers + 1]).T)]

        # Onehot encoding of elements in nodelist taking the possible ads atoms into account.
        node_onehot = np.zeros((len(site_labels) + len(adsorbate), len(self.all_elem) + 2))
        for i, label in enumerate(site_labels):
            node_onehot[i, label] = 1
            node_onehot[i, -2] = site_ids[i][2] + 1
            if self.n_neighbors == 2:
                if i in aoi_ids:
                    node_onehot[i, -1] = 1

            # Append ads atoms to the node list
        ### THIS IS A WEAK POINT. Make sure that ads atoms are added in correct order that matches edge-template!
        for i, atom in enumerate(adsorbate[::-1]):
            node_onehot[-(i + 1), self.all_elem.index(atom)] = 1

        # Initiate torch Data object
        torch_nodes = torch.tensor(node_onehot, dtype=torch.float)
        site_graph = Data(x=torch_nodes, edge_index=self.edge_dict[(adsorbate, site)], site=site, ads=adsorbate)

        return site_graph

    def get_zonefeat(self, coords, site):
        """Insert description"""

        # Get ordered list of coordinates of atoms included in graph
        if self.facet == 'fcc111':
            if site == 'ontop':
                site_id_dict = {'ens':[(0, 0, 0)],
                                '1A':[(-1,0,0),(-1,1,0),(0,-1,0),(0,1,0),(1,-1,0),(1,0,0)],
                                '2A':[(-1,0,1),(-1,1,1),(0,0,1)],
                                '3B':[(-1,-1,2),(-1,1,2),(1,-1,2)]
                                }

            if site == 'fcc':
                site_id_dict = {'ens': [(0, 0, 0),(0, 1, 0),(1, 0, 0)],
                                '1A': [(-1, 1, 0), (1, -1, 0), (1, 1, 0)],
                                '1B': [(-1, 0, 0), (-1, 2, 0), (0, -1, 0),(0, 2, 0),(2, -1, 0),(2, 0, 0)],
                                '2A': [(-1, 1, 1), (0, 0, 1), (0, 1, 1)],
                                '2B': [(-1, 0, 1), (-1, 2, 1), (1, 0, 1)],
                                '3C': [(-1, -1, 2), (-1, 2, 2), (2, -1, 2)],
                                }

        # Get ordered list of element labels of atoms in graph

        site_zonefeat = []
        for key in site_id_dict:
            zone_labels = self.grid[tuple(((np.array(site_id_dict[key]) + coords) % [*self.size, self.n_layers + 1]).T)]
            site_zonefeat = np.concatenate([site_zonefeat,[sum(zone_labels == elem) for elem in range(self.n_metals)]])

        return site_zonefeat

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

    def plot_surface(self):
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
            if i == 2 and self.n_layers == 3:
                layer_xshift = 0.5
                layer_yshift = 0.25
                alpha = 1
                grey = 0.6
            elif i == 1 and self.n_layers >= 2:
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

    def plot_hist(self, bin_width=0.01, G_opt=0.1, eU=0.82, T=298.15):
        """
        Version specific function for plotting the adsorption energies in terms of gross and net.
        Two different plots are constructed in this function. One plot containing the brutto gross and net distribution.
        and one plot containing the ensemble specific gross and net distribution.
        Color scheme for the ensemble specific ontop adsorption is implemented for OH* only.
        """

        gross, net = [], []
        for key in self.grid_dict_gross.keys():
            gross.append(self.grid_dict_gross[key].flatten())
            net.append(self.grid_dict_net[key].data[self.ads_dict[key]])

        all_min, all_max = np.min(gross), np.max(gross)

        gross_hist, net_hist = [], []
        for ens in gross:
            counts, bin_edges = np.histogram(ens, bins=int((all_max - all_min) / bin_width),
                                             range=(all_min, all_max), density=False)
            gross_hist.append(counts)

        for ens in net:
            counts, bin_edges = np.histogram(ens, bins=int((all_max - all_min) / bin_width),
                                             range=(all_min, all_max), density=False)
            net_hist.append(counts)

        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        hist_max = 0
        for counts in gross_hist:
            if hist_max < max(counts):
                hist_max = max(counts)

        kb = 8.617e-5
        j_ki = np.exp(-(np.abs(-G_opt) - 0.86 + eU) / (kb * T))
        pt_act = 2 / 3 * np.sum(1 / (1 + 1 / j_ki))
        act = self.get_activity(G_opt,eU,T)/pt_act * 100

        fig, axes = plt.subplots(2, 1, figsize=(20, 16))
        twin_axes = []
        for ax in axes:
            twin_axes.append(ax.twinx())
        for j, ax in enumerate(axes):
            ax.clear()
            ax.step(bin_centers, gross_hist[j] / hist_max, where='mid', linewidth=2, alpha=0.5, color='black')
            my_cmap = plt.get_cmap("viridis")
            rescale = lambda bin_centers: (bin_centers - np.min(bin_centers))/(np.max(bin_centers) - np.min(bin_centers))
            ax.bar(bin_centers, net_hist[j] / hist_max, width=bin_width * 1.01, color=my_cmap(rescale(bin_centers)), alpha=0.7)
            ax.set(xlim=(all_min, all_max))
            ax.xaxis.set_tick_params(labelsize=20)
            ax.yaxis.set_ticks([])
            kb = 8.617e-5
            dist_linspace = np.linspace(all_min, all_max, 1000)
            max_act = np.exp((0.86 - eU) / (kb * T))
            ax.plot(dist_linspace, np.exp((-np.abs(dist_linspace - G_opt) + 0.86 - eU) / (kb * T)) / max_act,
                    color='black', linewidth=1.5, label=r'$j_{k_i}$', linestyle='--')
            if j == 0:
                s = f'{act:.0f}%'.rjust(4)
                ax.text(0.01, 1.02,
                        f'Activity relative to Pt(111): ' + s,
                        family='monospace', fontsize=32, transform=ax.transAxes,
                        va='bottom', ha='left')

        for j, twin_ax in enumerate(twin_axes):
            twin_ax.clear()
            cov = np.cumsum(net_hist[j]) / np.sum(gross_hist[j])
            twin_ax.plot(bin_centers, cov, color='green', linewidth=4, alpha=0.7)
            twin_ax.arrow(bin_centers[-1], cov[-1], 0.001, 0.000,
                      head_width=0.040, head_length=0.030, color='green', alpha=0.7)
            twin_ax.set(xlim=(all_min-0.05,all_max+0.05), ylim=(0, 1.1))
            twin_ax.yaxis.set_tick_params(labelsize=20)
            twin_ax.tick_params(axis='y', colors='green')
            if j == 0:
                twin_ax.set_ylabel('OH coverage', fontsize=32, color='green', labelpad=20)
            elif j == 1:
                twin_ax.set_ylabel('O coverage', fontsize=32, color='green', labelpad=20)

        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)
        plt.xlabel(r'$\Delta \mathrm{E}^{\mathrm{pred}}_{\mathrm{*OH}\,equivalent}$ [eV]', fontsize=40, labelpad=20)
        plt.ylabel('Density', fontsize=48, labelpad=50)

        return fig

    def plot_enshist(self, G_opt=0.1, eU=0.82, T=298.15):
        fig2, ax = plt.subplots(2, 1, figsize=(15, 10))
        """Ensemble specific adsorption energy distribution"""

        # Make shorthand notation for ensemble, gross and net adsorption energy
        ens = self.grid[:, :, 0][self.ads_dict[('OH', 'ontop')]]
        ens_gross = self.grid[:, :, 0].flatten()
        net = self.grid_dict_net[('OH', 'ontop')].data[self.ads_dict[('OH', 'ontop')]]
        gross = self.grid_dict_gross[('OH', 'ontop')].flatten()
        
        # Define bin width
        bin_width = 0.01
        
        # Define the maximum and minimum of all data. Also used for O.
        all_max, all_min = None, None
        for dist in [gross, net]:
            if all_max != None and all_min != None:
                if max(dist) > all_max:
                    all_max = max(dist)
                if min(dist) < all_min:
                    all_min = min(dist)
            else:
                all_max, all_min = max(dist), min(dist)
            all_max, all_min = all_max + 0.1, all_min - 0.1
        
        # Construct an element dictionary of gross and net adsorption energies
        elements = np.array(['Ag', 'Ir', 'Pd', 'Pt', 'Ru'])

        e_dict_net = {el: [] for el in elements}
        for i, ads_e in enumerate(net):
            e_dict_net[elements[ens[i]]].append(ads_e)

        e_dict_gross = {el: [] for el in elements}
        for i, ads_e in enumerate(gross):
            e_dict_gross[elements[ens_gross[i]]].append(ads_e)

        # Obtain the total energy counts.
        counts_gross, bin_edges = np.histogram(gross, bins=int((all_max - all_min) / bin_width),
                                       range=(all_min, all_max), density=False)
        
        #Define bin centers
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])        

        # Construct the ensemble specific adsorption energy histogram.
        for i, dict in enumerate([e_dict_gross, e_dict_net]):

            ax[i].step(bin_centers, counts_gross / max(counts_gross), where='mid', linestyle='--',
                       linewidth=1, alpha=0.5, color='black')

            last_cov_gross = [0] * len(bin_centers)
            for el in elements:
                counts, _ = np.histogram(dict[el], bins=int((all_max - all_min) / bin_width),
                                         range=(all_min, all_max), density=False)

                ax[i].bar(bin_centers, counts / max(counts_gross), width=bin_width * 1.005, color=get_color(el),
                          alpha=0.7)
                ax[i].step(bin_centers, counts / max(counts_gross), where='mid', linewidth=1, alpha=1,
                           color=get_dark_color(el))

                ax[i].text(np.mean(dict[el]), np.max(counts) / max(counts_gross) + 0.05, el, family='monospace',
                           fontsize=22,
                           va='bottom', ha='center', color=get_dark_color(el))

                element_cov_gross = np.cumsum(counts) / np.sum(counts_gross)
                cov_gross = last_cov_gross[:] + element_cov_gross[:]
                last_cov_gross = cov_gross

            twin_ax = ax[i].twinx()
            twin_ax.plot(bin_centers, cov_gross, color='green', linewidth=1.5, alpha=0.5)

            ax[i].tick_params(axis='x', labelsize=16)
            twin_ax.tick_params(axis='y', colors='green', labelsize=16)
            ax[i].set_ylabel('Density', fontsize=26, labelpad=20)
            twin_ax.set_ylabel('Coverage [ML]', fontsize=26, color='green', labelpad=20)
            ax[i].set(ylim=(0, 1.1))
            twin_ax.set(ylim=(0, 1.1))
            ax[i].set_yticks([])

        ax[1].set_xlabel(r'$\Delta \mathrm{G} _{\mathrm{*OH}}$', fontsize=36, labelpad=20)
        ax[1].set_ylabel('Density', fontsize=36, labelpad=20)

        plt.tight_layout()
        
        return fig2

    def get_activity(self, G_opt=0.10, eU=0.82, T=298.15, j_d=1):
        kb = 8.617e-5
        j_ki = np.array([])
        for i, key in enumerate(self.grid_dict_gross.keys()):
            e = self.grid_dict_gross[key][self.ads_dict[key]]
            a = np.exp(-(np.abs(e - G_opt) - 0.86 + eU) / (kb*T))
            j_ki = np.concatenate([j_ki, a])

        j = 1/np.product(self.size) * np.sum(1 / (1/j_d + 1/j_ki))

        return j
