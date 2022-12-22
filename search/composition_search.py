from utils.bruteforcesurface import BruteForceSurface
from utils.GCN_model import load_GCN
from utils.bayesian import expected_improvement, append_to_file, random_comp
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from time import time
import pickle
import numpy as np

# Objective function
def comp2act(comp):#, filename=None):
    # Construct surface
    surface = BruteForceSurface(comp, adsorbates, ads_atoms, sites, coordinates, height,
                                    regressor, 'graphs', 2, 'fcc111', surf_size, displace_e, scale_e)

    # Determine gross energies of surface sites
    surface.get_gross_energies()

    # Get net adsorption energy distributions upon filling the surface
    surface.get_net_energies()

    # Get activity of the net distribution of *OH adsorption energies
    activity = surface.get_activity(G_opt=E_opt, eU=eU, T=298.15, j_d=1)
    
    # Print sampled composition
    f_str = ' '.join(f"{k}({v + 1e-5:.2f})" for k, v in comp.items())
    print(f'{f_str}     A = {activity / pt_act * 100:.0f} %')
    
    return activity

# set random seed
np.random.seed(42)

# Define elements to use
elements = ['Ag', 'Ir', 'Pd', 'Pt', 'Ru']

# set adsorbate information
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
regressor = load_GCN(kwargs,trained_state=trained_state)

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

# Define slab size (no. of atom x no. of atoms)
surf_size = (96, 96)

# Define optimal OH adsorption energy (relative to Pt(111))
E_opt = 0.100  # eV

# Define potential at which to evaluate the activity
eU = 0.820  # eV

# Define filename where sampled molar fraction are written to
filename = 'sampled_compositions.csv'
with open(filename, 'w') as filetowrite:
    filetowrite.write(','.join(elements) + ',Activity rel. to Pt(111)\n')

# Generate N molar fractions at random to begin the search
# Random compositions are limited to maximum 80% of a single element
n_random = 5

# Reference activity of 2/3 *OH coverage on pure Pt(111)
j_ki = np.exp(-(np.abs(-E_opt) - 0.86 + eU) / (8.617e-5 * 298.15))
pt_act = 2 / 3 * np.sum(1 / (1 + 1 / j_ki))

# Initiate activities list
fs, activities = [], []

# Sample random points to fit the GPR with
for i in range(n_random):
    temp = random_comp(elements)
    fs.append(list(temp.values()))
    activities.append(comp2act(temp))

    f_str = ','.join(f"{e+1e-5:.2f}" for e in temp.values())
    
    with open(filename, 'a') as file_:
        file_.write(f'{f_str},{activities[-1] / pt_act * 100:.0f}%\n')

# Initial fit of the Gaussian process regressor
gpr.fit(np.array(fs), activities)

# Iterate through the optimization steps
for n_samples in range(len(fs), n_steps):

    # Select the best composition for activity sampling using acquisition function. 
    f_opt = opt_acquisition(fs, gpr, elements, acq_func=expected_improvement, xi=xi, step_size=0.01, n_random=1000)

    # Sample the composition
    activity = comp2act(dict(zip(elements, f_opt)))
    
    # Write result to file
    f_str = ','.join(f"{e+1e-5:.2f}" for e in f_opt)
    with open(filename, 'a') as file_:
        file_.write(f'{f_str},{activity / pt_act * 100:.0f}%\n')

    # Add the sampled composition and activity to the dataset and update the Gaussian process regressor
    fs = np.vstack((fs, [f_opt]))
    activities += [activity]
    gpr.fit(fs, activities)


