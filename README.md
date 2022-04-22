# High-Entropy Alloy Catalysis Simulation
HEACS is a set of Python modules for modeling catalysis on high-entropy alloys.

Requirements
------------
* [ase](https://wiki.fysik.dtu.dk/ase/index.html)
* [gpaw](https://wiki.fysik.dtu.dk/gpaw/)
* [SLURM](https://slurm.schedmd.com)
* [scikit-learn](https://scikit-learn.org/stable/)
* [torch](https://pypi.org/project/torch/)
* [torch-geometric](https://pypi.org/project/torch-geometric/)
* [torch-sparse](https://pypi.org/project/torch-sparse/)
* [torch-scatter](https://pypi.org/project/torch-scatter/)
* [iteround](https://pypi.org/project/iteround/)

Setting up density functional theory geometry optimizations
---------------------
In the **data** directory edit the simulation parameters of *data_acquisition.py* to suit your task (see comments below). After adjusting the parameters, run the script and preview the slabs in the *\*_preview.db*-file. Subsequently, the simulations are submitted by running the script from the terminal with the "submit" argument (e.g. "python3 data_acquisition.py submit")

Start by choosing a project name that will function as a prefix for all files.
```python
### Filename
project_name = 'agirpdptru'
```

Next adjust slab parameters. A range of unique slabs with integer slabIds will be generated. A
```python
### Ids of unique slabs to generate
start_id = 0
end_id = 2

### Slab parameters
facet = 'fcc111'  # choose from ['fcc100','fcc110','fcc111','bcc100','bcc110','bcc111','hcp0001']
size = (3,3,5)  # minimum 2 atoms in each lateral direction
elements = ['Ag','Ir','Pd','Pt','Ru']  # list of alloy elements
lattice = 'surface_adjusted'  # choose from 'mean' or 'surface_adjusted'
comp_sampling = 'dirichlet'  # choose 'dirichlet' or list of fractions summing to 1.0
vacuum = 10  # vacuum in angstroms added above and below the slab
fix_bottom = 2  # number of bottom slablayers to fix during geometry optimization
distort_limit = None  # 

### Adsorbate parameters
adsorbates = ['OH','O']
sites = ['ontop','fcc']
init_bonds = [2.0,1.3]
ads_per_slab = 9
multiple_adsId = None

# GPAW parameters
GPAW_kwargs = {'xc':'RPBE',
			   'ecut':400,
			   'kpts':(4,4,1),
			   'max_force':0.1}

# Cluster parameters
SLURM_kwargs = {'partition': 'katla_short',
				'nodes': '1-2',
				'ntasks': 16,
				'ntasks_per_core': 2,
				'mem_per_cpu': '2G',
				'constraint': '[v1|v2|v3|v4]',
				'nice': 0,
				'exclude': None}
```

Calculating surface properties
------------------------------

```python
from someDFTprogram import DFTCalculator
from heacs.calculate import AdsorptionEnergy
energies = DFTCalculator('hea_slabs.db')
slabs = read('hea_slabs.db')
properties = AdsorptionEnergy().get(slabs, energies, ref='Pt111')
slabs.add_property(properties)
```

Extrapolating properties
------------------------

```python
from heacs.extrapolate import LinearRegression
from heacs.features import NeighborCounting
from heacs.io import read
known_data = read('hea_slabs.db')
unknown_data = known_data.get_all_surfaces()
reg = LinearRegression(known_data, NeighborCounting).predict(unknown_data)
reg.parity_plot('parity_plot.png')
reg.save_regressor('regressor.pickle')
```

Optimizing alloy composition
----------------------------

```python
from heacs.optimize import BayesianOptimizer
from heacs.io import read
known_data = read('hea_slabs.db')
reg = read('regressor.pickle')
opt = BayesianOptimizer(known_data, reg)
optimum = opt.find_optimum(acquisition_function='expected improvement')
```
