Setting up density functional theory geometry optimizations
---------------------
Edit the simulation parameters of *data_acquisition.py* to suit your task (see explanations below). After adjusting the parameters, run the script and preview the slabs in the *\*_preview.db*-file. Subsequently, the simulations are submitted by running the script from the terminal with the "submit" argument (e.g. "python3 data_acquisition.py submit").

A  *\*_slab.db*-file with relaxed slabs will be created as the slab optimizations finish and subsequently *\*_site_adsorbate.db*-files are created for each site adsorbate combination. Once calculations have finished use *join_dbs.py*

Choosing a project name that will function as a prefix for all files.
```python
project_name = 'agirpdptru'
```

A range of unique slabs with individual slabIds will be generated. This includes both the start_id and end_id.
```python
start_id = 0
end_id = 2
```

Desired surface facet. Choose from ['fcc100','fcc110','fcc111','bcc100','bcc110','bcc111','hcp0001']
```python
facet = 'fcc111'
```
Number of atoms in the lateral dimensions and number of layers -> NB! Minimum 2 atoms in each lateral dimension.
```python
size = (3,3,5)
```

List of surface elements. Lattice parameters will be loaded from dict in hea_aux.py
```python
elements = ['Ag','Ir','Pd','Pt','Ru']
```

Lattice constant for each slab. Choose from:
1. 'mean' = Slab lattices will be determined by the composition weighted mean lattice parameters. 
2. 'surface_adjusted' = In addition to 'mean' the lateral dimensions will be scaled to the weighted mean lattice parameter of the atoms constituting the surface -> See https://doi.org/10.1007/s12274-021-3544-3
```python
lattice = 'surface_adjusted'
```

Alloy composition sampled. Choose from:
1. 'dirichlet' = Uniform random sampling of the composition space. Each slab has new set of probabilities for each element.
2. *list of fractions summing to 1* = Each slab is generated from the same set of probabilities e.g. [0.2,0.3,0.1,0.05,0.35] for 20%, 30%, 10%, 5% and 35% of the elements, respectively.
```python
comp_sampling = 'dirichlet'
```

Vacuum layer added above and below slab (measured from the surface) in angstrom.
```python
vacuum = 10
```

Number of bottom layers to fix during geometry optimizations.
```python
fix_bottom = 2
```

Fraction the relaxed slab are allowed to have distorted e.g. if distort_limit = 1.1 the slab are allowed a 10% increase in height. If it exceeds the limit after relaxation an error will be raised which causes the adsorbate jobs not to commence.
```python
distort_limit = None  # 
```

Adsorbates. Slab will first be optimized and subsequently each adsorbate will be placed and optimized on the slab. Choose from ['OH','O','H']
```python
adsorbates = ['OH','O']
```

Corresponding sites on which to place adsorbates. Choose from ['ontop','bridge','shortbridge','longbridge','hollow','fcc','hcp'] 
```python
sites = ['ontop','fcc']
```

Initial bond lengths of the adsorbates in angstrom.
```python
init_bonds = [2.0,1.3]
```

Adsorbates to add sequentially to a single slab e.g. a combination of 'OH', 'ontop' and ads_per_slab = 2 will relax OH on two different ontop sites (position 0 and 1) resulting in two adsorbate calculations on the same 'parent-slab', thus greatly reducing computational load. NB! If multiple_adsId is not None this is not used.
```python
ads_per_slab = 9
```

Used to adsorb multiple adsorbates on the same slab. Manually specify the combinations of adsorbate ids e.g. a combination of 'OH', 'ontop' and multiple_adsId = [[0,1],[0,2]] will give a slab with OH on ontop sites 0 and 1 and another simulation of OH on ontop sites 0 and 2.
```python
multiple_adsId = None
```

GPAW calculation parameters in order: 
1. Exchange-correlation functional to be used.
2. Plane wave energy cutoff in eV
3. Monkhorst-Pack k-point grid
4. Force threshhold for optimization in eV/angstrom

```python
GPAW_kwargs = {'xc':'RPBE',
			   'ecut':400,
			   'kpts':(4,4,1),
			   'max_force':0.1}
```

SLURM scheduling parameters in order:
1. Partition name
2. Number of nodes allowed for parallelization
3. Number of allocated CPUs
4. Enable multithreading
5. Amount of memory allocated per CPU
6. Node types included.
7. Set Nice-factor to directly down-prioritize jobs if needed
8. Exclude faulty nodes if necessary

```python
SLURM_kwargs = {'partition': 'katla_short',
				'nodes': '1-2',
				'ntasks': 16,
				'ntasks_per_core': 2,
				'mem_per_cpu': '2G',
				'constraint': '[v1|v2|v3|v4]',
				'nice': 0,
				'exclude': None}
```

