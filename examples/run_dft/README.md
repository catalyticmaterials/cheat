#### Setting up DFT geometry optimizations

This folder contains an example of how to submit a batch of DFT optimizations of solid-solution slabs with randomized atomic structures. Each slab (identified by slabId) will be relaxed to the specified force threshold and then adsorbates can automatically be added to any number of binding sites on the surface and optimized. Thus, each slab can serve as basis for probing a large number of unique binding sites.

By running *submitDFT.py* you can preview the slabs in the *\*_preview.db* using the *ase gui* command. Subsequently, the run the script with the "submit" argument to submit the jobs to SLURM (e.g. "python submitDFT.py submit").

A  *\*_slab.db*-file with relaxed slabs will be created as the slab optimizations finish and subsequently *\*_site_adsorbate.db*-files are created for each site/adsorbate combination.

Most options are explained directly in *submitDFT.py* however a few warrants additional comments:

As it is often desirable to create a dataset of diverse slab compositions, this option allows for sampling slab compositions with a uniform Dirichlet distribution. This will override the otherwise specified slab composition and each slab will have a new set of probabilities for each element.
```python
dirichlet = False # [True,False]
```

The lattice will be determined by the composition weighted mean lattice parameters according to Vegards law. However, the lateral cell dimension can be scaled to the weighted mean lattice parameter of the atoms constituting the surface -> See https://doi.org/10.1007/s12274-021-3544-3
```python
surf_adj_lat = True # [True,False]
```

If the relaxed slab distorts beyond a set threshold an error will be raised which causes the adsorbate jobs not to commence e.g. distort_limit = 1.1 allows the slab a 10% increase in height. Use None for no limit.
```python
distort_limit = None
```

Site/adsorbate combinations will be added sequentially to a single slab e.g. a combination of 'OH', 'ontop' and ads_per_slab = 2 will relax OH on two different ontop sites (position 0 and 1) resulting in two adsorbate calculations on the same slabId.
```python
ads_per_slab = 2
```

Used to adsorb multiple adsorbates on the same slab. Manually specify the combinations of adsorbate ids e.g. a combination of 'OH', 'ontop' and multiple_adsId = [[0,1],[0,2]] will give a slab with OH on ontop sites 0 and 1 and another simulation of OH on ontop sites 0 and 2. This option overrides ads_per_slab.
```python
multiple_adsId = None
```

This example uses GPAW as DFT calculation software of choice and kwargs for this is supplied as a dict.
```python
GPAW_kwargs = {'mode':"PW(400)",
               'xc':"'RPBE'",
               'kpts':(4,4,1),
               'eigensolver':'Davidson(3)',
               'parallel':{'augment_grids':True,'sl_auto':True}
              }
```

The SLURM kwargs are highly personalized and should be modified accordingly to fit your own HPC protocols.
```python
SLURM_kwargs = {'partition': 'your_partition_here',
				'nodes': '1-1',
				'ntasks': 24,
				'ntasks_per_core': 2,
				'mem_per_cpu': '2G',
				}
```

