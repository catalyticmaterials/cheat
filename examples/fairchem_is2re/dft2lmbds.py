import re, glob, pickle, ase, lmdb, torch
from tqdm import tqdm
from fairchem.core.preprocessing import AtomsToGraphs
from cheatools.graphtools import atoms2template

# import gasphase reference
e_h20 = ase.io.read('../gpaw/refs/h2o.traj',-1).get_potential_energy()
e_h2 = ase.io.read('../gpaw/refs/h2.traj',-1).get_potential_energy()
ref_dict = {'O':e_h20-e_h2, 'OH':e_h20-e_h2/2}

a2g = AtomsToGraphs() # OCP graph maker

for s in ['train','val','test']:
    # initialize LMDB
    db = lmdb.open(
        path=f'lmdbs/{s}.lmdb',
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
        )
     
    slab_paths = glob.glob(f'../gpaw/{s}/*_slab.traj')
    counter = 0 # counter for sample ids
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

            templated_atoms = atoms2template(atoms) # reduce atoms object to fixed template structure
            
            # create OCP data object 
            d = a2g.convert_all([templated_atoms], disable_tqdm=True)[0]
            d.sid = torch.LongTensor([counter])
            d.fid = torch.LongTensor([0])
            d.tags = torch.LongTensor(templated_atoms.get_tags())
            d.ads = ads
            d.y_relaxed = e # label with adsorption energy

            # write to LMDB
            txn = db.begin(write=True)
            txn.put(key = f"{counter}".encode("ascii"), value = pickle.dumps(d, protocol=-1))
            txn.commit()
            db.sync()
            counter += 1 

    db.close()



