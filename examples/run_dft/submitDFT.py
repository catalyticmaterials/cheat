import sys, os, subprocess
import numpy as np
from time import sleep
from ase.db import connect
from cheatools.dftsampling import make_slab, relax_slab, relax_ads, SLURM_script

### Filename
project_name = 'example_project'

### Ids of unique slabs to generate
start_id = 0 
end_id = 1

### Slab parameters
facet = 'fcc111'                    # facet (see cheatools.dftsampling.make_slab for supported facets)
size = (3,3,5)                      # slab size 
composition = {'Ag': 0.2,           # alloy composition given as dictionary with {'element': fraction} 
               'Ir': 0.1,
               'Pd': 0.3,
               'Pt': 0.35,
               'Ru': 0.05,
              }
dirichlet = False                   # use uniform Dirichlet sampling and override composition [True,False]
surf_adj_lat = True                 # Use surface adjusted lattice instead of the weighted mean of composition [True,False]
vacuum = 10                         # vacuum added above and below slab in Å
fix_bottom = 2                      # number of bottom layers to fix
distort_limit = None                # set distortion limit of the relaxed slab e.g. 1.1 = 10%
skin = None                         # set surface layer to given element
spin_polarized = False              # toggle spin polarization [True,False]

### Adsorbate parameters
adsorbates = ['O','H','OH']         # adsorbates to add to the relaxed slab (see cheatools.utils.get_ads for supported species)
sites = ['fcc','fcc','ontop']       # sites to add adsorbates to (see cheatools.dftsampling.get_site_ids for supported sites)
init_bonds = [1.3,1.3,2.0]          # set initial bond lenghts in Å
ads_per_slab = 2                    # number of sites sampled on each slab (for each adsorbate/site combination)
multiple_adsId = None               # add multiple adsorbates to the same slab as a list of lists of site ids e.g. [[1,2],[2,3]] else None

### GPAW kwargs
fmax = 0.1                          # force threshold for relaxations
GPAW_kwargs = {'mode':"PW(400)",
               'xc':"'RPBE'",
               'kpts':(4,4,1),
               'eigensolver':'Davidson(3)',
               'parallel':{'augment_grids':True,'sl_auto':True}
              }

### SLURM kwargs                    # !DISCLAMER! These kwargs are highly personalized and should be modified accordingly to fit your own HPC protocols.
SLURM_kwargs = {'partition': 'katla_day',
				'nodes': '1-1',
				'ntasks': 24,
				'ntasks-per-core': 2,
				'mem-per-cpu': '2G',
				'constraint': '[v1|v2|v3|v4|v5]',
				'nice': 0,
			   }

# Make folders for output
for f in ['sl','py','traj','txt','err','log']:
    os.makedirs(f'{f}', exist_ok=True)

# Initiate preview database for slabs 
preview_db = connect(f'{project_name}_preview.db')

# Slab loop
for i in np.arange(start_id,end_id+1):
    np.random.seed(i)
    
    # override composition if dirichlet sampling is chosen
    if dirichlet:
        rnd_comp = np.random.dirichlet(np.ones(len(composition.keys())), 1)[0]
        composition = dict(zip(composition.keys(),rnd_comp))

    # initiate slab
    atoms = make_slab(facet, composition, size, surf_adj_lat, vacuum, fix_bottom, skin, spin_polarized)

    # write to preview db
    while True:
        try:
            if len(list(preview_db.select(slabId=i))) != 0:
                print(f'Skibbing slabId {i} already written to {project_name}_preview.db')
                break
            else:
                preview_db.write(atoms, slabId=i)
                break
        except:
            sleep(1)

    # write slab relaxation script
    filename = project_name + '_' + str(i).zfill(4)
    relax_slab(filename, i, fmax, distort_limit, GPAW_kwargs)
    
    # adsorbate loop
    adsId_counter, arrayId_counter = 0, 0
    for j, ads in enumerate(adsorbates):
    
        # iterate through site ids for single adsorbates
        if multiple_adsId == None:
            for k in range(ads_per_slab):
                relax_ads(filename, i, adsId_counter, facet, size, sites[j], ads, init_bonds[j], arrayId_counter, fmax, GPAW_kwargs)
                adsId_counter += 1
                arrayId_counter += 1
                if adsId_counter == np.product(size[:2]):
                    adsId_counter = 0
       
        # iterate through sets of site ids for multiple adsorbates
        elif isinstance(multiple_adsId, list):
            for adsIds in multiple_adsId:
                relax_ads(filename, i, adsIds, facet, size, sites[j], ads, init_bonds[j], arrayId_counter, fmax, GPAW_kwargs)
                arrayId_counter += 1
    
    # initiate SLURM submission if "submit" flag has been given
    try:
        if sys.argv[1] == 'submit':
            SLURM_script(f'{filename}_slab', SLURM_kwargs, dependency=None, jobarray=None)
            out = subprocess.run(f'cd sl/ && sbatch {filename}_slab.sl' , shell=True, check=True, capture_output=True)
            job_id = re.findall(r'\d+', out.stdout.decode('utf-8'))[0]
            
            # adsorbate calculations will be submitted as an SLURM array with dependency on the slab relaxation
            SLURM_script(f'{filename}_ads', SLURM_kwargs, dependency=job_id, jobarray=[0,arrayId_counter])
            out = subprocess.run(f'cd sl/ && sbatch {filename}_ads.sl' , shell=True, check=True, capture_output=True)
            print(f'Submitted jobs for slabId {i}')
    
    except IndexError:
        pass
