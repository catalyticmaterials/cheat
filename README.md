# High-Entropy Alloy Catalysis Simulation
HEACS is a set of Python modules for modeling catalysis on high-entropy alloys (HEAs). <br />
This repository is closely linked to the article <br />
"Ab Initio to activity: Machine learning assisted optimization of high-entropy alloy catalytic activity." <br />
DOI: https://doi.org/10.26434/chemrxiv-2022-vvrrf-v2

If this repository is utilized please cite: <br />
Clausen, C. M., Nielsen, M. L. S., Pedersen, J. K., & Rossmeisl, J. (2022). Ab Initio to activity: Machine learning assisted optimization of high-entropy alloy catalytic activity.

We hope you find this page helpful :)


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
In the 
[data](data)
 directory all calculations of DFT simulated slabs are set up. Each HEA slab can be used to sample multiple adsorption sites saving towards half of computational resources. All simulations are stores in ASE database-files for easy overview and subsequently joined into a single database to construct regression features. Instructions are found in the 
[data](data)
 directory's README.md-file.

Feature construction
------------------------------
In the 
[features](features)
 directory you can find already constructed features for a working example. 
Furthermore, you can find a comprehensive guide in the 
[features](features)
 directory's README-file, for how to construct features for both our Piecewies Linear regression (PWL) as well as our Graph Convoluted network (GCN).

The construction of features is needed for the regression step of the workflow. Therefore, ensure that the code provided is understood and that you have obtained features for your project before moving on.

Regression of adsorption energies
------------------------------
In the 
[regression](regression)
 folder you will find the neccessary code to utilize our proposed methods of regression (PWL and GCN). 
The regression requires features as well as at least a single database of adsorption energies (see "Setting up DFT geometry optimization" or the 
[data](data)
 directory). As in all other steps of this repository you will also find a regression specific README-file containing code examples in the 
[regression](regression)
 directory.


Extrapolating properties
------------------------


Optimizing alloy composition
----------------------------

