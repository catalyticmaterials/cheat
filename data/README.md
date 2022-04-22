This is a test

Line-by-line documentation for the flags and parameters:

project_name = *str* 
# name displayed on all files and jobs
# individual files will be named "{filename}_{slabId}_{site}_{adsorbate}_[{adsId}]"

start_id = 0
# slabId of first unique slab

end_id = 1
# slabId of last included unique slab

facet = *str* 
# desired surface facet. Choose from ['fcc100','fcc110','fcc111','bcc100','bcc110','bcc111','hcp0001']

elements = *list* 
# list of surface elements(*str*) e.g. ['Ir','Pt','Ru']
# lattice parameters will be loaded from dict in hea_aux.py

size = *tuple* 
# tuple of number(*int*) of atoms in the lateral dimensions and number of layers e.g. (3x3x4) -> NB! Minimum 2 atoms in each lateral dimension.

lattice = *str*
# lattice constant for each slab. Choose from:
1. 'mean' = Slab lattices will be determined by the composition weighted mean lattice parameters. 
2. 'surface_adjusted' = In addition to 'mean' the lateral dimensions will be scaled to the weighted mean lattice parameter of the atoms constituting the surface -> See https://doi.org/10.1007/s12274-021-3544-3

comp_sampling = *str*
# alloy composition sampled. Choose from:
1. 'dirichlet' = Uniform random sampling of the composition space. Each slab has new set of probabilities for each element.
2. *list of fractions summing to 1* = Each slab is generated from the same set of probabilities e.g. [0.3,0.3,0.4] for 30%, 30% and 40% of the elements, respectively.

vacuum = *int*
# vacuum layer added above and below slab (measured from the surface) in angstrom.

fix_bottom = *int*
# number of bottom layers to fix during optimization procedure

distort_limit = *float* or None
# fraction the relaxed slab are allowed to have distorted e.g. if distort_limit = 1.1 the slab are allowed a 10% increase in height. If it exceeds the limit after relaxation an error will be raised which causes the adsorbate jobs not to commence.

adsorbates = *list*
# list of adsorbates e.g. ['OH','O','O']  Slab will first be optimized and subsequently each adsorbate will be placed and optimized on the slab. Choose from ['OH','O','H']
