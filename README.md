# High-Entropy Alloy Catalysis Simulations
HEACS is a set of Python modules for modeling catalysis on high-entropy alloys.


Requirements
------------
[ase](https://wiki.fysik.dtu.dk/ase/index.html)
[scikit-learn](https://scikit-learn.org/stable/)

Constructing a system
---------------------

The workflow could be something like the following:

```python
from heacs.build import fcc111, OH, uniform_sampling
molar_fractions = {'Ag': 0.2, 'Ir': 0.2, 'Pd': 0.2, 'Pt': 0.2, 'Ru': 0.2}
slabs = fcc111(molar_fractions, size=(3,3,4), n=100, sampling=uniform_sampling, lattice_parameter='default')
slabs.add_adsorbate(OH, location='default')
slabs.save_to_database('hea_slabs.db')
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
