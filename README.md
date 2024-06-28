# Computational High-Entropy Alloy Tools
CHEAT is a suite of modules for inference of adsorption energies and modeling catalytic reactions on high-entropy and solid-solution alloys. This workflow was originally published in [*High Entropy Alloys & Materials*](https://doi.org/10.1007/s44210-022-00006-4) and the original version of this repository can be found in the v1.0-legacy branch.

If this repository is utilized please cite: <br />
Clausen, C. M., Nielsen, M. L. S., Pedersen, J. K., & Rossmeisl, J. (2022). Ab Initio to activity: Machine learning assisted optimization of high-entropy alloy catalytic activity.

#### Installation
The required packages can be installed into a conda environment running:
```terminal
conda env create -f env.yml
```
however if you intend to use inference models from the Open Catalyst Project follow the install guide [here](https://fair-chem.github.io/core/install.html).

After environment creatio navigate to this folder and install *cheatools*:
```terminal
pip install -e .
```

#### Examples
The *examples* folder contains working examples of 
All modules contain further explanation and instructions within each subdirectory. Data have been provided so that each module contains a working example.

The [run_dft](examples/run_dft) demonstrates querying DFT calculations used to train the inference algorithms. This aids in sampling multiple binding sites on the same slab to minimize compute per adsorption energy optained.

The [train_lgnn](examples/train_lgnn) reduces the optimized geometries from the DFT calculations to graph features and subsequently trains a lean graph neural network (lGNN) to perform adsorption energy inference.

The [surface_simulation](examples/surface_simulation) emulates a solid-solution alloy surface via a grid-based approach. This surrogate surface is used in conjunction with the lGNN to infer the distribution of adsorption energies on the surface. Additionally, competitive co-adsorption of different species can be included for certain sites.

The bayesian_optimization(Coming Soon!) applies the above step in a Bayesian optimization procedure to maximize a catalytic activity by sampling surfaces within a specified composition space.
