import re, glob, torch, pickle, ase
from cheatools.graphtools import atoms2graph, get_ensemble
from tqdm import tqdm

# import gasphase reference
e_h20 = ase.io.read('../gpaw/refs/h2o.traj',-1).get_potential_energy()
e_h2 = ase.io.read('../gpaw/refs/h2.traj',-1).get_potential_energy()
ref_dict = {'O':e_h20-e_h2, 'OH':e_h20-e_h2/2}

# set onehot labels for node vectors -> stored in data_object.onehot_labels
onehot_labels = ['Ag','Ir','Pd','Pt','Ru','H','O']

# loop through sets
for s in ['train','val','test']:     
    slab_paths = glob.glob(f'../gpaw/{s}/*_slab.traj')
    graph_list = []
    
    # slab loop
    for sp in tqdm(slab_paths,total=len(slab_paths)):
        slabId = re.findall(r'\d{4}', sp)[0]
        slab = ase.io.read(sp,'-1')
        slab_e = slab.get_potential_energy() # slab energy
        
        # adsorbates on current slabId 
        ads_paths = glob.glob(f'../gpaw/{s}/{str(slabId).zfill(4)}_ads*.traj')
        
        # adsorbate loop
        for ap in ads_paths:
            atoms = ase.io.read(ap,'-1')

            ads_e = atoms.get_potential_energy() # ads+slab energy
            ads = ''.join([a.symbol for a in atoms if a.tag == 0])
            e = ads_e - slab_e - ref_dict[ads] # adsorbtion energy
            
            # get graph
            g = atoms2graph(atoms,onehot_labels)
            g.y= e
            graph_list.append(g)

    with open(f'graphs/{s}.graphs', 'wb') as output:
        pickle.dump(graph_list, output)


