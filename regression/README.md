# Traning and testing the adsorption energy regression algorithm
---------------------
## Piecewise linear model
In the piecewise linear regression(PWR) a multilinear regression algorithm is trained for each ensemble in the constructed zone features. To train the regressor state the number of elements in the surface, the desired linear regression model supported by scikit-learn and make sure that the adsorbate/site-combinations match the set of features:

```python
n_metals = 5
regressor = Ridge()
site_ads_list = ['ontop_OH','fcc_O']
```

The PWR will be trained on 90% and tested on 10% of the available data. Samples are chosen randomly and are not split evenly among the ensembles. Extreme outliers in the predictions will be reported. A dict composed of regressors for all ensembles is saved in *model_states* along with parity plots showing test accuracy.

## Graph convolutional neural network model
The graph convolutional neural network (GCN) is a single algorithm trained on the available graph features. The model has the following adjustable parameters:

Batch size, learning rate and maximum number of training epochs.
```python
batch_size = 64
max_epochs = 3000
learning_rate = 1e-3
```

Early stopping and reporting frequency. The stopping criteria is based on rolling validation error: If the validation error has not decreased 1% during the prior *patience* number of epochs early stopping is invoked.
```python
roll_val_width = 20
patience = 100
report_every = 100
```

Model architecture. Specify number of convolution layers, hidden layers, dimension(width) of the layers and activation function of the hidden layers.
```python
kwargs = {'n_conv_layers': 3,
		  'n_hidden_layers': 0,
		  'conv_dim': 18,
		  'act': 'relu',
		  }
```

The GCN will be trained with 80% of the data and use 10% for validation. The remaining 10% are used for testing. The best performing set of model parameters across all epochs are saved in *model_states* along with parity plots showing test accuracy.
