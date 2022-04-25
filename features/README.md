Construction of features for regression
---------------------
Edit the simulation parameters of *data_acquisition.py* to suit your task (see explanations below). After adjusting the parameters, run the script and preview the slabs in the *\*_preview.db*-file. Subsequently, the simulations are submitted by running the script from the terminal with the "submit" argument (e.g. "python3 data_acquisition.py submit").

A  *\*_slab.db*-file with relaxed slabs will be created as the slab optimizations finish and subsequently *\*_site_adsorbate.db*-files are created for each site adsorbate combination. Once calculations have edit *join_dbs.py* to suit the including combinations and run from the terminal with the project_name as argument (e.g. "python3 join_dbs project_name"). This will result in a joined ASE database which is subsequently used in feature construction.

Choosing a project name that will function as a prefix for all files.
```python
project_name = 'agirpdptru'
```
