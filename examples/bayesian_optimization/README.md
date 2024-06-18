## Bayesian optimization of high-entropy alloy composition

In this section we employ the earlier surface simulation in an iterative manner with Bayesian optimization to find the most active catalyst composition.

Input composition space and adsorbate details to match the trained regression model which is then loaded.

```python
elements = ['Ag', 'Ir', 'Pd', 'Pt', 'Ru']  # surface elements

ads_atoms = ['O','H']  # adsorbate elements included
adsorbates = ['OH','O']  # adsorbates included
sites = ['ontop','fcc']  # sites of adsorption
coordinates = [([0,0,0],[0.65,0.65,0.40]), None]  # coordinates of multi-atom adsorbates
height = np.array([2,1.3])  # length of bond to surface

# displacement and scaling of adsorption energies
displace_e = [0.0, 0.0]
scale_e = [1, 0.5]

# load trained state
with open(f'../regression/model_states/GC3H0reludim18BS64lr0.001.state', 'rb') as input:
    trained_state = pickle.load(input)

# set model parameters and load trained model
kwargs = {'n_conv_layers': 3,  # number of gated graph convolution layers
          'n_hidden_layers': 0,  # number of hidden layers
          'conv_dim': 18,  # number of fully connected layers
          'act': 'relu' # activation function in hidden layers.
         }
regressor = load_GCN(kwargs,trained_state=trained_state
```

Gaussian process regression will be used as surrogate function during the optimization procedure and the default acquisition function is expected improvement. Both can be adjusted accordingly.

```python
# Define kernel to use for Gaussian process regressors
kernel = C(constant_value=0.05, constant_value_bounds=(1e-5, 1e1))\
        *RBF(length_scale=0.5, length_scale_bounds=(1e-5, 1e1))\

# Define Gaussian process regressor
# alpha = 1e-5 is the approximate variance in the predicted current densities
gpr = GPR(kernel=kernel, alpha=1e-5, n_restarts_optimizer=10)

# Define exploration-exploitation trade-off
# The higher this number, the more exploration
xi = 0.01

# Number of steps to run the optimization for
n_steps = 50
```

Define size of the simulated surface, the optimum adsorption energy and potential at which the catalytic activity should be evaluated.

```python
# Define slab size (no. of atom x no. of atoms)
surf_size = (96, 96)

# Define optimal OH adsorption energy (relative to Pt(111))
E_opt = 0.100  # eV

# Define potential at which to evaluate the activity
eU = 0.820  # eV
```
