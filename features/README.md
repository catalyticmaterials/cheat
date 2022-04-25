Construction of features for regression
---------------------
After obtaining a joined ASE database use *construct_feats.py* to prepare all slabs and corresponing adsorbates as features for regression of adsorption energies. Currently, two types of features are available:

*Zone features* applies a template of counting the number of elements in each zone of equivalent atomic positions as described in previous [publications](https://doi.org/10.1002/anie.202108116). Per default, this includes the nearest neighbors of the binding site and select next-nearest neighbors deemed important to the adsorption energy. This feature scheme requires a separate regression models since the feature vector will vary in length for different sites. These features are saved as a numpy-array with each row representing a feature with the corresponding adsorption energies in the last column.

*Graph features* converts the structures to graphs consisting of a list of node features and a sparse edge matrix and per default all next-nearest neightbors are included. The element of each node/atom is one-hot encoded and the atomic layer is noted (0 for adsorbates, 1 for surface-layer, 2 for subsurface-layer etc.). Addtionally, a feature called *aoi* tracks important atomic positions with favourable [long-ranged interactions](https://doi.org/10.1002/advs.202003357). All edge pair are These features are saved as a list of PyTorch Geometric Data-objects from which following information can be accessed: 
'x': Node features
'y': Adsorbtion energy
'edge_index': Sparse edge matrix
'site': Adsorbtion site
'ads': Adsorbate
'ens': Element composition of binding site

Reference energies are loaded from individual databases and all adsorbtion energies are normalized relative to adsorption on pure Pt(111).
```python
ref_dict = {'ontop_OH':ase.db.connect('../data/3x3x5_pt111_ontop_OH.db').get().energy,
			'fcc_O':ase.db.connect('../data/3x3x5_pt111_fcc_O.db').get().energy,
			'slab':ase.db.connect('../data/3x3x5_pt111_slab.db').get().energy
			 }
```
site_list = ['ontop','fcc']
ads_list = ['OH','O']

surface_elements = ['Ag','Ir','Pd','Pt','Ru']
adsorbate_elements = ['O','H']

for i in range(len(site_list)):

	## load joined ASE datebase
	db = ase.db.connect(f'../data/agirpdptru.db')

	## filename used for pickling
	filename = f'agirpdptru_{site_list[i]}_{ads_list[i]}'

	## Construct zoned features and pickle
	print(f'Performing zonefeat construction of agirpdptru_{site_list[i]}_{ads_list[i]}')
	samples = db_to_zonedfeats(surface_elements, site_list[i], ads_list[i], 0.1, db, ref_dict)

	with open(filename + '.zonefeats', 'wb') as output:
		pickle.dump(samples, output)

## Construct graphs and pickle
print(f'Performing graph construction of agirpdptru')
samples = db_to_graphs(surface_elements, adsorbate_elements, 2, 0.1, db, ref_dict)
with open(f'agirpdptru.graphs', 'wb') as output:
	pickle.dump(samples, output)
