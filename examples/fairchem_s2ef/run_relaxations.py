import os, tqdm, ase
#import numpy as np
from cheatools.dftsampling import make_slab, add_ads
from cheatools.graphtools import ase2ocp_tags
from fairchem.core.common.tutorial_utils import generate_yml_config, fairchem_main

# adsorbate parameters
adsorbates = ['OH','O']
sites = ['ontop','fcc']
init_bonds = [2.0,1.3]

# initialize ASE database if not present already
if not os.path.isfile('initial_structures.db'):
    
    db = ase.db.connect('initial_structures.db')

    # loop writing initial structures
    for i in tqdm.tqdm(range(50),total=50):
        j = i%len(adsorbates) # alternating between adsorbates

        # generate slab, add adsorbate and save -> refer to cheat/examples/run_dft/submitDFT.py for more details on this
        atoms = make_slab('fcc111', {'Ag':0.25,'Au':0.25,'Cu':0.25,'Pd':0.25}, (3,3,5), 'surface_adjusted', 10, 2)
        atoms = add_ads(atoms, 'fcc111', (3,3,5), sites[j], adsorbates[j], init_bonds[j], 4)
        atoms = ase2ocp_tags(atoms)
        db.write(atoms)

# path to model checkpoint
checkpoint_path = 'checkpoints/AI2PR-dft-S2EF153M.pt'

# fetch config file from checkpoint, delete irrelevant entries and add relaxation info
config = generate_yml_config(checkpoint_path, 'configs/relax.yml',
                   delete=['cmd', 'logger', 'task', 'model_attributes',
                           'dataset', 'slurm','evaluation_metrics',
                           'val_dataset','test_dataset'],
                   
                   update={'dataset.relax.format': 'ase_db',
                           'dataset.relax.src': 'initial_structures.db',
                           'task.write_pos': 'True',
                           'task.relaxation_steps': 50,
                           'task.relaxation_fmax': 0.1,
                           'task.relax_opt.maxstep': 0.04,
                           'task.relax_opt.memory': 50,
                           'task.relax_opt.damping': 1.0,
                           'task.relax_opt.alpha': 70.0,
                           'task.relax_opt.traj_dir': 'relaxed_trajectories/',
                           'task.save_full_traj': False,
                           'logger': 'tensorboard',
                           'optim.eval_batch_size': 6,
                          })

# initialize relaxations
os.system(f'python {fairchem_main()} --mode run-relaxations --config-yml {config} --checkpoint {checkpoint_path}')
