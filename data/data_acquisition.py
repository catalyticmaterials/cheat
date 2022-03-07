# Made by Christian Møgelberg Clausen
import os, sys
from ase.db import connect
import numpy as np
from utils import write_slab, relax_slab_script, relax_ads_script, SLURM_script
import subprocess
from time import sleep

### see readme.txt for documentation of flags and parameters

### Filename
project_name = 'example_project'

### Ids of unique slabs to generate
start_id = 0
end_id = 2

### Slab parameters
facet = 'fcc111'
size = (3,3,5)
elements = ['Ag','Au','Cu','Pd','Pt']
lattice = 'surface_adjusted'
comp_sampling = 'dirichlet'
vacuum = 10
fix_bottom = 2
distort_limit = None

### Adsorbate parameters
adsorbates = ['OH','O']
sites = ['ontop','fcc']
init_bonds = [2.0,1.3]
ads_per_slab = 2
multiple_adsId = None

# GPAW parameters
GPAW_kwargs = {'xc':'RPBE',
			   'ecut':400,
			   'kpts':(4,4,1),
			   'max_force':0.1}

# Cluster parameters
SLURM_kwargs = {'partition': 'katla_day',
				'nodes': '1-2',
				'ntasks': 16,
				'ntasks_per_core': 2,
				'mem_per_cpu': '2G',
				'constraint': '[v1|v2|v3|v4]',
				'nice': 0,
				'exclude': None}

# Make folders for output
os.system("mkdir sl")
os.system("mkdir py")
os.system("mkdir traj")
os.system("mkdir txt")
os.system("mkdir err")
os.system("mkdir log")

preview_db = connect(f'{project_name}_preview.db')

for i in np.arange(start_id,end_id+1):
	np.random.seed(i)

	if comp_sampling == 'dirichlet':
		rnd_comp = np.random.dirichlet(np.ones(len(elements)), 1)[0]

	elif isinstance(comp_sampling,list):
		rnd_comp = comp_sampling

	atoms = write_slab(facet, elements, rnd_comp, size, lattice, vacuum, fix_bottom)

	while True:
		try:
			if len(list(preview_db.select(slabId=i))) != 0:
				print(f'Skibbing slabId {i} already written to preview.db')
				break
			else:
				preview_db.write(atoms, slabId=i)
			break
		except:
			sleep(1)

	filename = project_name + '_' + str(i).zfill(4)

	relax_slab_script(filename, i, distort_limit, **GPAW_kwargs)

	for j, ads in enumerate(adsorbates):
		if multiple_adsId == None:
			for k in range(ads_per_slab):
				relax_ads_script(filename, i, [k], facet, size, sites[j], ads, init_bonds[j], **GPAW_kwargs)
		elif isinstance(multiple_adsId, list):
			for id_set in multiple_adsId:
				relax_ads_script(filename, i, id_set, facet, size, sites[j], ads, init_bonds[j], **GPAW_kwargs)

	try:
		if sys.argv[1] == 'submit':
			SLURM_script(filename + '_slab', **SLURM_kwargs, dependency=None)
			os.system(f"(cd sl/ && sbatch {filename + '_slab.sl'})")
			job_id = None
			print('Fetching SLURM jobid of slab optimization. Please wait...')
			while job_id == None:
				try:
					job_id = int(subprocess.run(['sacct', '-n', '-X', '--state=R,PD', '--format=jobid', f'--name={filename}_slab'],
												stdout=subprocess.PIPE).stdout.decode('utf-8'))
				except:
					pass
			
			for j, ads in enumerate(adsorbates):
				if multiple_adsId == None:
					for k in range(ads_per_slab):
						SLURM_script(filename + f'_{sites[j]}_{ads}_{k}', **SLURM_kwargs, dependency=job_id)
						os.system(f"(cd sl/ && sbatch {filename}_{sites[j]}_{ads}_{k}.sl)")
				elif isinstance(multiple_adsId, list):
					for id_set in multiple_adsId:
						ads_id_str = "+".join([str(Id) for Id in id_set])		
						SLURM_script(filename + f'_{sites[j]}_{ads}_{ads_id_str}', **SLURM_kwargs, dependency=job_id)
						os.system(f"(cd sl/ && sbatch {filename}_{sites[j]}_{ads}_{ads_id_str}.sl)")




	except IndexError:
		pass
