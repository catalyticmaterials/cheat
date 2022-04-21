import numpy as np
import torch
from torch_geometric.data import Data
from ase.geometry import analysis


def db_to_zonedfeats(surface_elements, site, adsorbate, fmax, db, ref_dict):
	counter = 0
	n_rejected = 0
	feat_list, target_list = [], []

	# iterate through the desired database entries
	for row in db.select(relaxed=1, site=site, ads=adsorbate):
		counter += 1

		# skip relaxed slabs
		if row.site == 'N/A':
			continue

		# check that the maximum force is not too large
		if row.fmax > fmax:
			print('Rejecting slab based on fmax.')
			n_rejected += 1
			continue

		# get atoms object and repeat
		atoms = db.get_atoms(row.id)
		atoms_3x3 = atoms.repeat((3, 3, 1))
		labels_3x3 = np.array(atoms_3x3.get_chemical_symbols())

		# get all bonds/edges in the atoms object
		ana_object = analysis.Analysis(atoms_3x3, bothways=True)
		all_edges = np.c_[np.array(list(ana_object.adjacency_matrix[0].keys()), dtype=np.dtype('int,int'))['f0'],
						  np.array(list(ana_object.adjacency_matrix[0].keys()), dtype=np.dtype('int,int'))['f1']]

		# remove self-to-self edges
		all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1]]

		# adsorbing atom symbol (interpreted as the first adsorbate atom in the list of atoms)
		n_ads_atoms = len([atom.index for atom in atoms if atom.tag == 0])
		ads_atoms_ids = [atom.index for atom in atoms_3x3 if atom.tag == 0]

		# ids of adsorbate in the middle of the 3x3 atoms object
		ads_ids = ads_atoms_ids[n_ads_atoms * 4:n_ads_atoms * 5]

		# find all unique atom ids included in adsorbate, bond to surface, first neighbors and second neighbors
		ads_edges = all_edges[np.isin(all_edges[:, 0], ads_ids)]
		neighbor_edges = all_edges[np.isin(all_edges[:, 0], ads_edges[:, 1])]
		nextneighbor_edges = all_edges[np.isin(all_edges[:, 0], neighbor_edges[:, 1])]

		ens_ids = [id for id in np.unique(ads_edges) if id not in ads_ids]
		n_ids = [id for id in np.unique(neighbor_edges) if id not in ens_ids + ads_ids]
		nn_ids = [id for id in np.unique(nextneighbor_edges) if id not in ens_ids + ads_ids + n_ids]

		aoi = []
		for nn in nn_ids:
			n = n_ids[np.argmin(atoms_3x3.get_distances(nn,[n_ids]))]
			e = ens_ids[np.argmin(atoms_3x3.get_distances(nn,[ens_ids]))]
			a = atoms_3x3.get_angle(e,n,nn)
			if a > 150:
				aoi.append(nn)

		# remove sample if site not expected or unidentifiable site
		found_n_firstshell = len([atom.symbol for atom in atoms_3x3[np.unique(neighbor_edges)] if atom.tag != 0])
		site_dict = {10: 'ontop',
					 18: 'fcc',
					 19: 'hcp',
					 15: 'bridge'}

		try:
			if site_dict[found_n_firstshell] != row.site:
				#print(f'Expecting {row.site} site but found {site_dict[found_n_firstshell]} site. Rejecting slab.')
				n_rejected += 1
				continue
		except:
			#print(f'Could not identify type of site. Rejecting slab.')
			n_rejected += 1
			continue

		zone_counts = []

		if site == 'ontop':
			zone_counts = zone_counts + [sum(labels_3x3[ens_ids] == elem)
										 for elem in surface_elements]  # ensemble

			zone_counts = zone_counts + [sum(labels_3x3[[id for id in n_ids if atoms_3x3[id].tag == 1]] == elem)
										 for elem in surface_elements]  # zone 1A

			zone_counts = zone_counts + [sum(labels_3x3[[id for id in n_ids if atoms_3x3[id].tag == 2]] == elem)
										 for elem in surface_elements]  # zone 2A

			zone_counts = zone_counts + [sum(labels_3x3[[id for id in nn_ids if atoms_3x3[id].tag == 3 and id in aoi]] == elem)
										 for elem in surface_elements]  # zone 3B

		elif site == 'fcc':
			zone_counts = zone_counts + [sum(labels_3x3[ens_ids] == elem)
										 for elem in surface_elements]  # ensemble

			zone_counts = zone_counts + [sum(labels_3x3[[id for id in n_ids if atoms_3x3[id].tag == 1 and np.count_nonzero(np.any(np.isin(neighbor_edges,id),axis=1)) == 2]] == elem)
										 for elem in surface_elements]  # zone 1A

			zone_counts = zone_counts + [sum(labels_3x3[[id for id in n_ids if atoms_3x3[id].tag == 1 and np.count_nonzero(np.any(np.isin(neighbor_edges, id),axis=1)) == 1]] == elem)
										 for elem in surface_elements]  # zone 1B

			zone_counts = zone_counts + [sum(labels_3x3[[id for id in n_ids if atoms_3x3[id].tag == 2 and np.count_nonzero(np.any(np.isin(neighbor_edges, id), axis=1)) == 2]] == elem)
									 	for elem in surface_elements]  # zone 2A

			zone_counts = zone_counts + [sum(labels_3x3[[id for id in n_ids if atoms_3x3[id].tag == 2 and np.count_nonzero(np.any(np.isin(neighbor_edges, id), axis=1)) == 1]] == elem)
									 	for elem in surface_elements]  # zone 2B

			zone_counts = zone_counts + [sum(labels_3x3[[id for id in nn_ids if atoms_3x3[id].tag == 3 and id in aoi]] == elem)
									 	for elem in surface_elements]  # zone 3C

		# get the corresponding slab without the adsorbate
		for slab_row in db.select(slabId=row.slabId, ads='N/A', site='N/A', relaxed=1):
			# check that the maximum force is not too large
			if slab_row.fmax > fmax:
				continue

			# check that the slabs surface atoms matches
			slab_atoms = db.get_atoms(slab_row.id)
			slab_symbols = slab_atoms.get_chemical_symbols()
			if np.all([atom.symbol for atom in atoms if atom.tag != 0] == slab_symbols):
				break

		else:
			#print(f'[WARNING] No slab match found in for row {row.id}. Rejecting slab')
			n_rejected += 1
			continue

		# get adsorption energy
		ads_energy = row.energy - slab_row.energy - ref_dict[f'{row.site}_{row.ads}'] + ref_dict[f'slab']

		feat_list.append(zone_counts)
		target_list.append(ads_energy)

	joined = np.c_[feat_list, target_list]

	print(f'Rejected {n_rejected} slabs in the zonefeat construction process.')

	return joined

def db_to_graphs(surface_elements, adsorbate_elements, n_neighbors, fmax, db, ref_dict):
	counter = 0
	n_rejected = 0
	graph_list = []
	# iterate through the desired database entries
	for row in db.select(relaxed=1):
		counter += 1

		# skip relaxed slabs
		if row.site == 'N/A':
			continue

		# check that the maximum force is not too large
		if row.fmax > fmax:
			print('Rejecting slab based on fmax.')
			n_rejected += 1
			continue

		# get atoms object and repeat
		atoms = db.get_atoms(row.id)
		atoms_3x3 = atoms.repeat((3, 3, 1))
		labels_3x3 = atoms_3x3.get_chemical_symbols()

		# get all bonds/edges in the atoms object
		ana_object = analysis.Analysis(atoms_3x3, bothways=True)
		all_edges = np.c_[np.array(list(ana_object.adjacency_matrix[0].keys()), dtype=np.dtype('int,int'))['f0'],
						  np.array(list(ana_object.adjacency_matrix[0].keys()), dtype=np.dtype('int,int'))['f1']]

		# remove self-to-self edges
		all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1]]

		# adsorbing atom symbol (interpreted as the first adsorbate atom in the list of atoms)
		n_ads_atoms = len([atom.index for atom in atoms if atom.tag == 0])
		ads_atoms_ids = [atom.index for atom in atoms_3x3 if atom.tag == 0]

		# ids of adsorbate in the middle of the 3x3 atoms object
		ads_ids = ads_atoms_ids[n_ads_atoms * 4:n_ads_atoms * 5]

		# find all unique atom ids included in adsorbate, bond to surface, first neighbors and second neighbors
		ads_edges = all_edges[np.isin(all_edges[:, 0], ads_ids)]
		neighbor_edges = all_edges[np.isin(all_edges[:, 0], ads_edges[:, 1])]
		nextneighbor_edges = all_edges[np.isin(all_edges[:, 0], neighbor_edges[:, 1])]

		ens_ids = [id for id in np.unique(ads_edges) if id not in ads_ids]
		n_ids = [id for id in np.unique(neighbor_edges) if id not in ens_ids + ads_ids]
		nn_ids = [id for id in np.unique(nextneighbor_edges) if id not in ens_ids + ads_ids + n_ids]

		aoi = []
		for nn in nn_ids:
			n = n_ids[np.argmin(atoms_3x3.get_distances(nn,[n_ids]))]
			e = ens_ids[np.argmin(atoms_3x3.get_distances(nn,[ens_ids]))]
			a = atoms_3x3.get_angle(e,n,nn)
			if a > 150:
				aoi.append(nn)

		# remove sample if site not expected or unidentifiable site
		found_n_firstshell = len([atom.symbol for atom in atoms_3x3[np.unique(neighbor_edges)] if atom.tag != 0])
		site_dict = {10: 'ontop',
					 18: 'fcc',
					 19: 'hcp',
					 15: 'bridge'}

		try:
			if site_dict[found_n_firstshell] != row.site:
				#print(f'Expecting {row.site} site but found {site_dict[found_n_firstshell]} site. Rejecting slab.')
				n_rejected += 1
				continue
		except:
			#print(f'Could not identify type of site. Rejecting slab.')
			n_rejected += 1
			continue

		# include nodes according to rank of neighbors
		if n_neighbors == 0:
			incl_ids = np.array(ads_ids+ens_ids)
		elif n_neighbors == 1:
			incl_ids = np.array(ads_ids+ens_ids+n_ids)
		elif n_neighbors == 2:
			incl_ids = np.array(ads_ids+ens_ids+n_ids+nn_ids)

		# shuffle the node list
		np.random.shuffle(incl_ids)

		# isolate all edges between included atom ids (both ways / undirected)
		all_edges_reduced = all_edges[np.all(np.isin(all_edges[:, :], incl_ids), axis=1)]

		# node feature lists (element label and onehot indexed)
		elements = surface_elements + adsorbate_elements

		# onehot encoding of the node list
		node_onehot = np.zeros((len(incl_ids), len(elements) + 2))
		for i, id in enumerate(incl_ids):
			node_onehot[i, elements.index(labels_3x3[id])] = 1
			node_onehot[i, -2] = atoms_3x3[id].tag

			if id in aoi:
				node_onehot[i, -1] = 1

			# rename all atom ids to the index in the node list
		for i, edge in enumerate(all_edges_reduced):
			for j, id in enumerate(edge):
				all_edges_reduced[i, j] = np.where(incl_ids == id)[0][0]

		# get the corresponding slab without the adsorbate
		for slab_row in db.select(slabId=row.slabId, ads='N/A', site='N/A', relaxed=1):
			# check that the maximum force is not too large
			if slab_row.fmax > fmax:
				continue

			# check that the slabs surface atoms matches
			slab_atoms = db.get_atoms(slab_row.id)
			slab_symbols = slab_atoms.get_chemical_symbols()
			if np.all([atom.symbol for atom in atoms if atom.tag != 0] == slab_symbols):
				break

		else:
			#print(f'[WARNING] No slab match found in for row {row.id}. Rejecting slab')
			n_rejected += 1
			continue

		# get adsorption energy
		ads_energy = row.energy - slab_row.energy - ref_dict[f'{row.site}_{row.ads}'] + ref_dict[f'slab']

		# make torch data object
		torch_edges = torch.tensor(np.transpose(all_edges_reduced), dtype=torch.long)

		# torch_edges_feat = torch.tensor(np.transpose(edge_feat), dtype=torch.long)
		torch_nodes = torch.tensor(node_onehot, dtype=torch.float)

		# get ensemble (surface elements directly involved in the bond)
		ensemble = np.array(labels_3x3)[[id for id in ens_ids]]
		ensemble = {el: sum(ensemble == el) for el in surface_elements}

		graph_list.append(Data(x=torch_nodes, edge_index=torch_edges, y=torch.tensor([ads_energy], dtype=torch.float),
				 site=row.site, ads=row.ads, ens=ensemble))

	print(f'Rejected {n_rejected} slabs in the graph construction process.')

	return graph_list
