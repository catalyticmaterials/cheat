import ase.db
import pickle
from utils.features import db_to_zonedfeats, db_to_graphs

# Reference energies
ref_dict = {'ontop_OH':ase.db.connect('../data/3x3x5_pt111_ontop_OH.db').get().energy,
			'fcc_O':ase.db.connect('../data/3x3x5_pt111_fcc_O.db').get().energy,
			'slab':ase.db.connect('../data/3x3x5_pt111_slab.db').get().energy
			 }

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