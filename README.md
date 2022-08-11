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


Extended surface investigation
------------------------
To employ the bruteforce adsorption on the extended surface, please follow the README-file in the 
[surface](surface)
 directory. This guide enables you to simulate the competetive co-adsorption of *O and *OH on an extended HEA surface. The adsorption is bruteforcce in the manner, that the strongest binding adsorption site, regardless of nature, is occupied first, followed by an adsorption to the second strongest binding site. When an adsorption has happende a blocking scheme is utilized to ensure our proposed adsorption isotherm. For more information go to the 
[surface](surface)
 directory or take a look in our article (DOI: https://doi.org/10.26434/chemrxiv-2022-vvrrf-v2) under the section "Net adsorption".


Optimizing alloy composition
----------------------------
After the completion of all above steps, you now have the tools to optimize the composition of the HEA through an Bayesian Optimization shceme.
This is simply done by following the guide in the README-file, located in the 
[optimization](optimization)
 directory.
 

Author's note
---------------------------
We hope you enjoyed this repository and have optained a better understanding of the infrastructure of our proposed workflow.
If more information is needed we urge you to take a look in the individuel directories README-file or in our associated article: <br />
"Ab Initio to activity: Machine learning assisted optimization of high-entropy alloy catalytic activity." <br />
DOI: https://doi.org/10.26434/chemrxiv-2022-vvrrf-v2

And once again, if this repository is utilized please cite: <br />
Clausen, C. M., Nielsen, M. L. S., Pedersen, J. K., & Rossmeisl, J. (2022). Ab Initio to activity: Machine learning assisted optimization of high-entropy alloy catalytic activity.

Kind regards
Authors
