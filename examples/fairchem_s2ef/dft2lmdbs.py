import re, glob, pickle, ase, lmdb, torch
from tqdm import tqdm
from fairchem.core.preprocessing import AtomsToGraphs
from cheatools.graphtools import ase2ocp_tags

# import gasphase reference
e_h20 = ase.io.read('../gpaw/refs/h2o.traj',-1).get_potential_energy()
e_h2 = ase.io.read('../gpaw/refs/h2.traj',-1).get_potential_energy()
ref_dict = {'O':e_h20-e_h2, 'OH':e_h20-e_h2/2}

a2g = AtomsToGraphs() # OCP graph maker

for s in ['train','val','test']:
    # initialize LMDB
    db = lmdb.open(
        path=f'lmdbs/{s}_test.lmdb',
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
            traj = ase.io.read(ap,':') # load trajectory
            traj_dataobj = a2g.convert_all(traj, disable_tqdm=True) # convert the trajectory to data objects

            # iterate over the data objects
            for fid, atoms in enumerate(traj): 
                ads_e = atoms.get_potential_energy() # ads+slab energy
                ads = ''.join([a.symbol for a in atoms if a.tag == 0])
                e = ads_e - slab_e - ref_dict[ads] # adsorbtion energy
                
                # add attributes to the data object
                atoms = ase2ocp_tags(atoms)
                d = traj_dataobj[fid]
                d.sid = counter
                d.fid = fid
                d.tags = atoms.get_tags()
                d.ads = ads
                d.y = e
                d.force = torch.Tensor(atoms.get_forces())            
 
                # write to LMDB
                txn = db.begin(write=True)
                txn.put(key = f"{counter}".encode("ascii"), value = pickle.dumps(d, protocol=-1))
                txn.commit()
                db.sync()
                counter += 1 

    db.close()



