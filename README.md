# Computational High-Entropy Alloy Tools
CHEAT is a suite of modules for inference of adsorption energies and modeling catalytic reactions on high-entropy and solid-solution alloys. This workflow was originally published in [*High Entropy Alloys & Materials*](https://doi.org/10.1007/s44210-022-00006-4) and the original version of this repository can be found in the v1.0-legacy branch.

#### Installation
It is recommended to fetch the latest version of the main branch using:
```terminal
git clone https://github.com/catalyticmaterials/cheat.git
```
as the newest implemented features might not be included in the latest release yet.

The required packages can be installed into a conda environment running:
```terminal
conda env create -f env.yml
```
Note the different installation procedure if you intend to use inference models from [FAIR Chemistry](examples/fairchem_is2re).

After environment creation, activate the environment and navigate to this folder to install *cheatools*:
```terminal
conda activate cheat
pip install -e .
```

#### Examples
The *examples* folder contains working examples of different applications with further explanation and instructions within each subdirectory. Start by unzipping the files in the [gpaw](examples/gpaw) folder. These contain pre-calculated DFT trajectories (GPAW 22.1.0) of *OH and *O on Ag-Ir-Pd-Pt-Ru high-entropy alloy surfaces which will form the basis for these examples.

[run_dft](examples/run_dft) demonstrates querying your own DFT calculations used to train the inference algorithms. This aids in sampling multiple binding sites on the same slab to minimize compute per adsorption energy optained. Note that this requires [installing GPAW](https://wiki.fysik.dtu.dk/gpaw/install.html) and some additional setup to conform to whatever high-performance cluster you are using.

[train_lgnn](examples/train_lgnn) reduces the optimized geometries from the DFT calculations to graph features and subsequently trains a lean graph neural network (lGNN) to perform adsorption energy inference.

[surface_simulation](examples/surface_simulation) emulates a solid-solution alloy surface via a grid-based approach. This surrogate surface is used in conjunction with the lGNN to infer the distribution of adsorption energies on the surface. Additionally, competitive co-adsorption of different species can be included for certain sites.

[bayesian_optimization](examples/bayesian_optimization) applies the above step in a Bayesian optimization procedure to maximize a catalytic activity by sampling surfaces within a specified composition space.

[fairchem_is2re](examples/fairchem_is2re) showcases the finetuning and application of a more advanced adsorption energy inference model from FAIR Chemistry and it's implementation in the surrogate surface simulation.

[fairchem_s2ef](examples/fairchem_s2ef) demonstrates how to finetune a machine learning potential from FAIR Chemistry, providing a much(!) faster alternative to a DFT calculator.

[plots](examples/plots) contains examples of a few plotting functions that can be handy to visualize results in high-dimensional composition space.

