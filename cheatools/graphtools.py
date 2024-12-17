import torch, ase.build
import numpy as np
from ase.neighborlist import build_neighbor_list, natural_cutoffs, neighbor_list, NeighborList
from itertools import combinations
from .dftsampling import add_ads
from copy import deepcopy
from collections import Counter, deque, defaultdict
from torch_geometric.data import Data
from .utils import get_adslabel
from ase.visualize import view
from ase.data import atomic_numbers, covalent_radii

def ase2ocp_tags(atoms):
    """
    Converts ASE tag format to OCP tag format
    -------
    ASE format: 0 = Adsorbate, 1 = Surface, 2 = Subsurface, 3 = Third layer, ..
    OCP format: 2 = Adsorbate, 1 = Surface, 0 = Everything else, ...
    """
    atoms.set_tags([0 if t >= 2 else 2 if t == 0 else 1 for t in atoms.get_tags()])
    
    return atoms

def get_ensemble(atoms):
    """
    Get ensemble (surface elements bound to adsorbate) as well as site for fcc(111) surfaces.
    ------
    Only implemented for monodentate adsorbates on fcc(111) surfaces. 
    Will categorize sites in ontop, bridge_{0,1,2}, fcc, or hcp.  

    Returns
    -------
    dict
        Keys are elements found in the ensemble and values are the number of atoms found
    list
        IDs of ensemble atoms
    string
        Site category  
    """
     
    atoms = deepcopy(atoms)  # copy to not change tags on original atoms object 
    if np.any(np.isin([3,4,5], atoms.get_tags())):  # the following operations uses the ocp tag format
        atoms = ase2ocp_tags(atoms)
    
    # center adsorbate and wrap cell to handle adsorbates close to the periodic boundaries
    ads_ids = [a.index for a in atoms if a.tag == 2]
    cell_xy = 0.5*np.sum(atoms.get_cell()[:2],axis=0)
    ads_xy = atoms[ads_ids[0]].position
    adjust = -ads_xy+cell_xy
    atoms.translate([adjust[0],adjust[1],0])
    atoms.wrap() 
    
    # build neighborlist to assert bonds - uses 110% natural cutoffs (+0.3Å skin) to ensure connectivity in distorted structures
    nl = build_neighbor_list(atoms, cutoffs=natural_cutoffs(atoms, mult=1.1), self_interaction=False, bothways=True) # skin=0.3 Å is default
    
    
    ads_neighbors = np.array([i for i in nl.get_neighbors(ads_ids[0])[0] if i not in ads_ids])
    if len(ads_neighbors) == 0:
        raise Exception("Adsorbate has no neighboring atoms.")  
    
    # we consider the three nearest atoms to be the potential ensemble  
    dist = atoms.get_distances(ads_ids[0],ads_neighbors)
    ens_ids = ads_neighbors[np.argsort(dist)][:3]
    
    # all possible ensembles given the three nearest atoms 
    ens = [[i] for i in ens_ids]+[[*i] for i in list(combinations(ens_ids, 2))]+[list(ens_ids)]
     
    # assert horizontal distances to all ensemble midpoint e.g. mean position of two atoms for bridge site etc. 
    pos = atoms.get_positions()
    dist = []
    for e in ens:        
        mean = np.mean(pos[e],axis=0)
        delta = pos[ads_ids[0]][:2] - mean[:2]
        dist.append(np.sqrt(np.sum(delta**2)))  
    closest_ens = ens[np.argsort(dist)[0]]
    
    # categorize ensemble
    if len(closest_ens) == 1:
        site = 'ontop'
    
    # bridge sites are subcategorized based on direction
    elif len(closest_ens) == 2:
        # directions (both ways)
        directions = np.array([[1.0,0.0,0.0],[-1.0,0.0,0.0], # horizontal
                        [0.5,0.866,0.0],[-0.5,-0.866,0.0], # lower left to upper
                        [-0.5,0.866,0.0],[0.5,-0.866,0.0]]) # lower right to upper
        
        delta = pos[closest_ens[0]] - pos[closest_ens[1]]  # vector between ensemble atoms
        delta = delta/np.sqrt(np.sum(delta**2))  # normalize to unit vector
        direction_id = int(np.floor(np.argmin(np.sum(np.abs(delta - directions),axis=1))/2))  # closest direction is chosen
        site = f'bridge_{direction_id}'
    
    elif len(closest_ens) == 3:
        # top-heavy triangle determined by y-coordinate is categorized as hcp and vice versa for fcc - NB! This assumes geometries as made by ASEs default surface functions
        meanY = np.mean(pos[closest_ens,1])
        diff = pos[closest_ens,1]-meanY
        site = 'hcp' if (diff > 0).sum() == 2 else 'fcc'

    # get elements of ensemble
    ensemble = np.array(atoms.get_chemical_symbols())[closest_ens]     
    ensemble = dict(Counter(ensemble))
    
    return ensemble, closest_ens, site

def atoms2template(atoms, tag_style='ocp'):
    """
    Transforms atoms object to template (set structure for each adsorbate/site combination)
    -------
    Only implemented for 3x3x5 atom-sized fcc(111) objects. Incompatible with any other size/surface.  
    Templates are structured based on a 3x3x5 atom-sized fcc(111) surface with lattice parameter (a) 3.9 and 10Å vacuum added above and below.     
    Bonds lengths for sites are {ontop:2.0, bridge:1.8, fcc:1.3, hcp:1.5}

    Returns
    -------
    Atoms object
        Template object  
    """
    # get ensemble ids and site to assert rolls and rotations
    _, ids, site = get_ensemble(atoms)
    
    # get adsorbate to add to template
    ads = get_adslabel(atoms)    

    # ontop roll/rotate scheme
    if site == 'ontop':
        ll = ids[0]
        rotate_scheme = 0
    
    # bridge roll/rotate schemes (depending on direction)
    elif site[:-2] == 'bridge':
        rotate_scheme = int(site[-1])
            
        if np.any(np.isin([36,39,42],ids)) and np.any(np.isin([38,41,44],ids)): # right side
            if rotate_scheme == 2:
                ll = np.min([id for id in ids if id in [36,39,42]])
            else: 
                ll = np.min([id for id in ids if id in [38,41,44]])
            
        elif np.any(np.isin([42,43,44],ids)) and np.any(np.isin([36,37,38],ids)): # upper side
            ll = np.min([id for id in ids if id in [42,43,44]])
        
        else:
            ll = np.min(ids)

    # fcc roll/rotate scheme
    elif site == 'fcc': 
        if np.all(np.isin([44,42,38],ids)):
            ll = 44
        elif np.any(np.isin([36,39,42],ids)) and np.any(np.isin([38,41,44],ids)): 
            ll = np.min([id for id in ids if id in [38,41,44]])
        elif np.any(np.isin([42,43,44],ids)) and np.any(np.isin([36,37,38],ids)):
            ll = np.min([id for id in ids if id in [42,43,44]])
        else:
            ll = np.min(ids)

        rotate_scheme = 0
    
    # hcp roll/rotate scheme
    elif site == 'hcp':
        if np.all(np.isin([36,38,42],ids)):
            ll = 42
        elif np.any(np.isin([36,39,42],ids)) and np.any(np.isin([38,41,44],ids)): 
            ll = np.min([id for id in ids if id in [36,39,42]])
        elif np.any(np.isin([42,43,44],ids)) and np.any(np.isin([36,37,38],ids)):
            ll = np.min([id for id in ids if id in [42,43,44]])
        else:
            ll = np.min(ids)

        rotate_scheme = 0
    
    # roll template ids
    roll_1 = 1 if ll < 39 else -1 if ll > 41 else 0
    roll_2 = 1 if ll in [36,39,42] else -1 if ll in [38,41,44] else 0
    tpl = np.arange(45).reshape((5,3,3))
    tpl = np.roll(tpl,roll_1,1)
    tpl = np.roll(tpl,roll_2,2)
    
    # if necessary rotate and roll template ids
    if rotate_scheme == 1:
        tpl = np.rot90(tpl, k=-1, axes=(1, 2))
        tpl[:,1,:] = np.roll(tpl[:,1,:],1,1)
        tpl[:,0,:] = np.roll(tpl[:,0,:],2,1)
        tpl[-2,:,:] = np.roll(tpl[-2,:,:],-1,0)
        tpl[-3,:,:] = np.roll(tpl[-3,:,:],-1,1)
        tpl[-5,:,:] = np.roll(tpl[-5,:,:],-1,0)

    elif rotate_scheme == 2:
        tpl = np.rot90(tpl, k=1, axes=(1, 2))
        tpl[:,:,0] = np.roll(tpl[:,:,0],1,1)
        tpl[:,:,2] = np.roll(tpl[:,:,2],2,1)
        tpl[-2,:,:] = np.roll(tpl[-2,:,:],-1,0)
        tpl[-2,:,:] = np.roll(tpl[-2,:,:],1,1)
        tpl[-3,:,:] = np.roll(tpl[-3,:,:],-1,0)
        tpl[-5,:,:] = np.roll(tpl[-5,:,:],-1,0)
        tpl[-5,:,:] = np.roll(tpl[-5,:,:],1,1)
    
    # make template atoms object and assign symbols from rolled ids 
    template = ase.build.fcc111('Au', size=(3,3,5), vacuum=10, a=3.9)
    template.set_chemical_symbols(np.array(atoms.get_chemical_symbols())[tpl.ravel()])
    
    # add adsorbate based on site details
    site = 'bridge' if site[:-2] == 'bridge' else site
    ads_id = 3 if site == 'hcp' else 4
    height = {'ontop':2.0,'bridge':1.8,'fcc':1.3,'hcp':1.5}
    add_ads(template, 'fcc111', (3,3,5), site, ads, height[site], ads_id)
    
    # adjust tag style
    if tag_style == 'ocp':
        template = ase2ocp_tags(template)

    return template

def BFS(edges, start_node):
    """
    Breadth-first search to find the distances between all nodes and the specified start node
    ------
    Returns
    -------
    list
        Distances from the start node to all other nodes in number of edges
    """
    # create an adjacency list
    adjacency_list = defaultdict(list)
    for u, v in edges:
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
    
    # initialize the distances list with -1
    distances = [-1] * np.unique(edges)
    distances[start_node] = 0
    
    # initialize BFS
    queue = deque([(start_node, 0)])  # (current_node, current_distance)
    visited = {start_node}
    
    while queue:
        current_node, current_distance = queue.popleft()  # dequeue the front element
        
        for neighbor in adjacency_list[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = current_distance + 1
                queue.append((neighbor, current_distance + 1))  # enqueue neighbors
    
    return distances 

def atoms2graph(atoms, onehot_labels):
    """
    Converts atoms object to torch data object in the lGNN graph format
    ------
    This function initially converts the 3x3 atom-sized atoms object to the appropriate template and then scales it up to a 5x5x3 cell.
    Then a graph is extracted containing atoms up to the N nearest neighbors to the bonded adsorbate atoms with N being:
    1 for adsorbate atoms, 2 for surface atoms, 2 for subsurface atoms, and 3 for third layer atoms.
    The nodes are one-hot encoded according to the labels plus the tag and the AtomOfInterest tag -> See https://doi.org/10.1002/advs.202003357
    The edges are featureless and only contain the node indices.
    
    For downpipe reference the onehot_labels of the elements are stored in the data object.
    In addition the ids of the nodes (referred to a 5x5x3 cell) is stored. This is used to fill the template with the correct symbols
    when cutting out a 5x5x3 cell from the surrogate surface.
    Adsorbate is also stored for easy categorization when testing.
    
    Returns
    -------
    Pytorch data object
    """
    # get template and scale up cell
    ads = get_adslabel(atoms)
    _, _, site = get_ensemble(atoms)
    atoms = atoms2template(atoms, tag_style='ase')
    del atoms[[a.index for a in atoms if a.tag > 3]]
    atoms_3x3 = atoms.repeat((3, 3, 1))
    cell = atoms_3x3.get_cell()
    
    #cut to 5x5x3 cell
    atoms_3x3.translate([-1.7/9 * np.sum(cell[:,0]),-2/9 * np.sum(cell[:,1]),0.0])
    atoms_3x3.set_cell([cell[0] * 5/9, cell[1] * 5/9, cell[2]])
    pos = atoms_3x3.get_positions()
    cell = atoms_3x3.get_cell()
    inverse_cell = np.linalg.inv(cell)
    fractional_positions = np.dot(pos, inverse_cell)
    inside = np.all((fractional_positions > -0.001) & (fractional_positions < 0.999), axis=1)
    atoms_3x3 = atoms_3x3[inside]
    
    # catching stray adsorbate atoms -> only keep N ads atoms closest to the center with N = number of atoms in adsobate 
    if len([a for a in atoms_3x3 if a.tag == 0]) > len(ads):
        positions = atoms_3x3.get_positions()
        surface_center = np.mean(positions[[a.index for a in atoms_3x3 if a.tag == 1]],axis=0)
        ads_ids = np.array([a.index for a in atoms_3x3 if a.tag == 0])
        ads_positions = positions[ads_ids]
        dists = np.sqrt(np.sum((ads_positions-surface_center)**2,axis=1))
        ranked = ads_ids[np.argsort(dists)[::-1]]
        del atoms_3x3[ranked[:(len(ranked)-len(ads))]]
    
    # rename ids
    id_sort = [a.index for a in sorted(atoms_3x3, key=lambda a: (-a.tag, a.position[1], a.position[0]))]
    atoms_3x3 = atoms_3x3[id_sort]
    
    # get edges 
    nl = build_neighbor_list(atoms_3x3, cutoffs=natural_cutoffs(atoms_3x3, mult=1.1), self_interaction=False, bothways=True)
    edges = np.array([[a.index,i] for a in atoms_3x3 for i in nl.get_neighbors(a.index)[0]])
   
    ads_id = [i for i in nl.get_neighbors(62)[0] if atoms_3x3[i].tag == 0]
     
    # breadth first search
    edge_dists = BFS(edges,ads_id[0])        

    gIds = ads_id
    for t, n in [(0,1),(1,2),(2,2),(3,3)]: # if adsorbate contains more than 2 atoms adjust this line
        gIds += [a.index for a in atoms_3x3 if a.tag == t and edge_dists[a.index] <= n and a.index not in gIds]
    gIds = np.sort(gIds) # included nodes in graph
 
    edges = edges[np.all(np.isin(edges,gIds),axis=1)] # only includes edges betw. incl. nodes
    
    # manually set atoms of interest
    if site == 'ontop':
        aoi = [6,8,16]
    elif site == 'fcc':
        aoi = [6,9,21]
    elif site == 'hcp':
        aoi = [6,8,10,13,20,21]
    elif 'bridge' in site:
        aoi = [6,9,16,17]
    
    # onehot encoding of the node list
    node_onehot = np.zeros((len(gIds), len(onehot_labels) + 2))
    for i, j in enumerate(gIds):
        node_onehot[i, onehot_labels.index(atoms_3x3[j].symbol)] = 1
        node_onehot[i, -2] = atoms_3x3[j].tag
        if j in aoi:
            node_onehot[i, -1] = 1
    
    # rename edges to match node list 
    id_map = {g: i for i, g in enumerate(gIds)}
    edges = np.array([[id_map[e[0]], id_map[e[1]]] for e in edges])
    
    # make torch data object
    torch_edges = torch.tensor(np.transpose(edges), dtype=torch.long)
    torch_nodes = torch.tensor(node_onehot, dtype=torch.float)

    graph = Data(x=torch_nodes, edge_index=torch_edges, onehot_labels=onehot_labels, gIds=gIds, ads=ads)
    return graph

class lGNNtemplater():
    """
    Template class for the surrogate surface used with the lGNN model
    """
    def __init__(self,facet,adsorbates,sites,onehot_labels):
        """
        Fetches the templates for the specified site/adsorbate combinations.
        """
        # initialize template dictionary and parent slab
        self.template_dict = {}
        height = {'ontop':2.0,'bridge':1.8,'fcc':1.3,'hcp':1.5}
        atoms = ase.build.fcc111(onehot_labels[0], size=(3,3,5), vacuum=10, a=3.9)

        # loop through site/adsorbate combinations
        for ads, site in zip(adsorbates,sites):
            ads_id = 3 if site == 'hcp' else 4
            temp_atoms = deepcopy(atoms)
            add_ads(temp_atoms, 'fcc111', (3,3,5), site, ads, height[site], ads_id)
            data_object = atoms2graph(temp_atoms, onehot_labels)
            data_object.x[:,0] = 0

            self.template_dict[(ads,site)] = data_object   
            
    def fill_template(self,symbols,adsorbate,site):
        """
        Fills the template with the specified symbols (from 5x5x3 cell)
        """
        cell = deepcopy(self.template_dict[(adsorbate,site)])
        for i, j in enumerate(cell.gIds):
            if j > 74:
                continue
            s = symbols[j]
            cell.x[i, cell.onehot_labels.index(s)] = 1
            
        return cell
