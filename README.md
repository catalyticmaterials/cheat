# High-Entropy Alloy Catalysis Simulations
HEACS is a set of Python modules for modeling catalysis on high-entropy alloys.


Requirements
------------
ase


Example
-------

The workflow could be something like the following.

Constructing a system:

´´´python
from heacs.build import fcc111, OH, uniform_sampling
molar_fractions = {'Ag': 0.2, 'Ir': 0.2, 'Pd': 0.2, 'Pt': 0.2, 'Ru': 0.2}
slabs = fcc111(molar_fractions, size=(3,3,4), n=100, sampling=uniform_sampling, lattice_parameter='default')
slabs.add_adsorbate(OH, location='default')
slabs.save_to_database('hea_slabs.db')
´´´

Doing DFT simulation:
Probably easiest with GPAW.

