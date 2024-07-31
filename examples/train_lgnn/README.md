#### Training and testing the lean graph neural network (lGNN)

This folder contains an example of how to train and test the [lGNN model](https://doi.org/10.1002/advs.202003357).

`dft2graphs.py` will pull the supplied trajectory files in the *gpaw* folder and perform graph feature construction to form train, validation and test sets from the relaxed slabs with adsorbates. Adsorbtion energies, $\Delta E_{ads}$ are calculated as:

$$\Delta E_{ads} = E_{slab+ads} - E_{slab} - E_{ads}$$

where $E_{slab+ads}$ and $E_{slab}$ are the slab with and without adsorbate respectively and $E_{ads}$ is the gas-phase reference energy of the adsorbate.

The graph includes the adsorbate and the nearest neighboring atoms to the adsorbing atom(s) (ensemble) as well as the next nearest neighbors in the third surface layer.

The graph node features are onehot encoded to denote element. In addition, the layer tag and an AtomOfInterest feature tracking important atomic positions with favourable [long-ranged interactions](https://doi.org/10.1002/advs.202003357) are included. As no positional information is included in the nodes and because the edges only denote connectivity, the resulting graphs retains equivariant properties.

The graphs are PyTorch Geometric data-objects from which following information can be accessed: 
'x': Node features
'y': Adsorbtion energy
'edge_index': Edge pairs
'onehot_labels': Element list used for onehot encoding (does not include tag or AoI feature)
'ads': Adsorbate
'gIds': "graph Ids" used for translating a 5x5x3 sized surface to a graph (used in conjunction with templates and the surrogate surface. See the *surface_simulation* folder for further info)

The lGNN model is trained by running train.py where you will also find a few adjustable parameters regarding the GNN architecture and training. The architecture will be saved in the *.state*-file. Therefore, after training, the model can be loaded with
```python
with open(f'{filename}.state', 'rb') as input:
	model_state = pickle.load(input)
model = lGNN(trained_state=model_state)
```

The lGNN class supports two methods: `model.predict(graphlist)` will return the predicted adsorbtion energies from a list of graphs. `model.test(data_loader,batch_size)` will return additional information
```python
pred, true, ads = model.test(data_loader, batch_size)
```
with *pred* and *true* being the predicted and true energies, respectively, and *ads* denoting the adsorbate for easy categorization.

Running `test.py` will create a *.results* file. Use it as argument to `plot_parity.py` to obtain a parity plot of the test results.
