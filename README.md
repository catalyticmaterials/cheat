# Computational High-Entropy Alloy Tools
CHEAT is a set of Python modules for regression of adsorption energies and modeling catalysis on high-entropy alloys.
This modeling procedure is described in detail here:

"Ab Initio to activity: Machine learning assisted optimization of high-entropy alloy catalytic activity." <br />
DOI: https://doi.org/10.26434/chemrxiv-2022-vvrrf-v2

If this repository is utilized please cite: <br />
Clausen, C. M., Nielsen, M. L. S., Pedersen, J. K., & Rossmeisl, J. (2022). Ab Initio to activity: Machine learning assisted optimization of high-entropy alloy catalytic activity.

It is the hope of the authors that this repository will be used, copied and modified by groups interested in doing computational studies on high-entropy alloys.

#### Required packages
* [ase](https://wiki.fysik.dtu.dk/ase/index.html) 
* [gpaw](https://wiki.fysik.dtu.dk/gpaw/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [torch](https://pypi.org/project/torch/)
* [torch-geometric](https://pypi.org/project/torch-geometric/)
* [torch-sparse](https://pypi.org/project/torch-sparse/)
* [torch-scatter](https://pypi.org/project/torch-scatter/)
* [iteround](https://pypi.org/project/iteround/)

The data acquisition module utilizes [SLURM](https://slurm.schedmd.com) for computational workload management but this can be omitted.

## CHEAT modules
All modules contain further explanation and instructions within each subdirectory. Data have been provides so that each module contains a working example.

The [data](data) module assists setting up DFT calculations. Optimized geometries are stored in ASE databases and can subsequently be joined into a single database to construct regression features.

The [features](features) modules will reduce optimized geometries to features suitable for regression of adsorption energies. Currently two types of feature schemes are available: a zone-reduced schemed based on equivalent atomic positions relative to the adsorption site and a graph-based feature scheme.

The [regression](regression) modules trains the corrensponding regression model, Piecewise Linear regression (PWL) or Graph Convolutional Neural Network (GCN), depending on the chosen feature scheme and benchmarks adsorption energy prediction accuracy.

The [surface](surface) module simulates a high-entropy alloy surface of a given size, predicts the available adsorption energies and simulates adsorbate coverage including competitive co-adsorption of \*O and \*OH. Based on established theory a catalytic activity can be estimated.

The [search](search) module apply the above step in a Bayesian optimization procedure to maximize the catalytic activity within the given composition space.

## Data
All DFT calculations required to reproduce the results of the paper is available [here](https://nano.ku.dk/english/research/theoretical-electrocatalysis/katladb/ab-initio-to-activity/)
