from ase.build import fcc111, add_adsorbate
from ase.geometry import analysis
from ase import Atoms
import numpy as np
import itertools as it
import torch
from ase.visualize import view

def get_site_ids(facet, site, size):
    if site == 'ontop':
        ads_ids = np.arange(np.product(size))[-int(np.product(size[:2])/2)-1]

    elif site == 'fcc':
        if facet in ['fcc111']:
            ads_ids = [np.arange(np.product(size))[-int(np.product(size[:2])/2)-1],
                    np.arange(np.product(size))[-int(np.product(size[:2])/2)],
                    np.arange(np.product(size))[-int(np.product(size[:2])/2)-1+size[0]]]

    try:
        iterable = iter(ads_ids)
        return ads_ids

    except:
        return [ads_ids]


def get_edges(facet,adsorbates,sites,neighbors,coordinates,height):
    edge_dict = {}
    for set in list(it.product(*[adsorbates,sites])):
        atoms = globals()[facet]('Pt', size=(7,7,3), vacuum=10, a=3.9936)
        ads_ids = get_site_ids(facet,set[1],(7,7,3))
        x = np.mean(np.array([atom.position for atom in atoms if atom.index in ads_ids])[:, 0])
        y = np.mean(np.array([atom.position for atom in atoms if atom.index in ads_ids])[:, 1])
        if len(set[0]) == 1:
            add_adsorbate(atoms, set[0], height[set[1] == np.array(sites)][0], position=(x,y))
        elif len(set[0]) > 1:
            ads = Atoms(set[0], coordinates[adsorbates.index(set[0])])
            add_adsorbate(atoms, ads, height[set[1] == np.array(sites)][0], position=(x,y))

        ana_object = analysis.Analysis(atoms, bothways=True)
        all_edges = np.c_[np.array(list(ana_object.adjacency_matrix[0].keys()), dtype=np.dtype('int,int'))['f0'],
                          np.array(list(ana_object.adjacency_matrix[0].keys()), dtype=np.dtype('int,int'))['f1']]
        all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1]]

        ads_ids = [atom.index for atom in atoms if atom.tag == 0]

        ads_edges = all_edges[np.isin(all_edges[:, 0], ads_ids)]
        neighbor_edges = all_edges[np.isin(all_edges[:, 0], ads_edges[:, 1])]
        nextneighbor_edges = all_edges[np.isin(all_edges[:, 0], neighbor_edges[:, 1])]

        if neighbors == 0:
            incl_ids = np.unique(ads_edges)
        elif neighbors == 1:
            incl_ids = np.unique(np.concatenate((ads_edges, neighbor_edges)))
        elif neighbors == 2:
            incl_ids = np.unique(np.concatenate((ads_edges, neighbor_edges, nextneighbor_edges)))
        all_edges = all_edges[np.all(np.isin(all_edges[:, :], incl_ids), axis=1)]
        for i, edge in enumerate(all_edges):
            for j, id in enumerate(edge):
                all_edges[i, j] = np.where(incl_ids == id)[0][0]

        torch_edges = torch.tensor(np.transpose(all_edges), dtype=torch.long)

        edge_dict[(set[0],set[1])] = torch_edges

    return edge_dict