## Surface simulation
To simulate the adsorbate coverage on the HEA surface two example scripts have been provided with two different regression algorithms, *bruteforcesurface_PWR* and *bruteforcesurface_GCN*. Both scripts contain the same initial flags:

Set element composition of the surface.
```python
composition = {'Ag': 0.20,'Ir': 0.20,'Pd': 0.20,'Pt': 0.20,'Ru': 0.20}
```

Set adsorbate information to match the regressor.
```python
ads_atoms = ['O','H']  # adsorbate elements included
adsorbates = ['OH','O']  # adsorbates included
sites = ['ontop','fcc']  # sites of adsorption
coordinates = [([0,0,0],[0.65,0.65,0.40]),None]  # coordinates of multi-atom adsorbates
height = np.array([2,1.3])  # length of bond to surface
```

If required the adsorption energy of the adsorbates can be displaced or scaled.
```python
displace_e = [0.0, 0.0]
scale_e = [1, 0.5]
```

Then either the trained PWR model or GCN model is loaded:
```python
# PWR model
with open(f'../regression/model_states/AgIrPdPtRu_PWR.obj', 'rb') as input:
    regressor = pickle.load(input)
    
# GCN model   
with open(f'../regression/model_states/GC3H0reludim18BS64lr0.001.state', 'rb') as input:
    trained_state = pickle.load(input)

kwargs = {'n_conv_layers': 3,  # number of gated graph convolution layers
          'n_hidden_layers': 0,  # number of hidden layers
          'conv_dim': 18,  # number of fully connected layers
          'act': 'relu' # activation function in hidden layers.
         }
regressor = load_GCN(kwargs,trained_state=trained_state)
```

The surface object can now be initiated with either 'zonefeats' or 'graphs' to match the regressor. It is also possible to change the rank of atomic neighboring positions to be considered per site (default=2 -> Next-nearest neighbors). Here the size of the surface is also chosen.
```python
surface_obj = BruteForceSurface(composition, adsorbates, ads_atoms, sites, coordinates, height,
                                regressor, *insert feature*, 2, 'fcc111', (96,96), displace_e, scale_e)
```

Calling .get_net_energies() will predict binding energies on all surface sites and subsequently fill the surface with adsorbates taking mutual blocking into account. List of adsorption energies can be returned and histograms showing the adsorption energy distribution and coverage can be plotted.
```python
surface_obj.get_net_energies()

OH_ontop_energies = surface_obj.grid_dict_gross[('OH','ontop')][surface_obj.ads_dict[('OH','ontop')]]
O_fcc_energies = surface_obj.grid_dict_gross[('O','fcc')][surface_obj.ads_dict[('O','fcc')]]

fig = surface_obj.plot_hist()
fig.savefig('Ehist_GCN.png')
```
