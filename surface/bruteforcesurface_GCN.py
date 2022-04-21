from utils.surface import BruteForceSurface
from utils.regression import load_GCN
import pickle
import numpy as np

# set random seed and surface composition
np.random.seed(42)
composition = {'Ag': 0.20,'Ir': 0.20,'Pd': 0.20,'Pt': 0.20,'Ru': 0.20}

# set adsorbate information
ads_atoms = ['O','H']  # adsorbate elements included
adsorbates = ['OH','O']  # adsorbates included
sites = ['ontop','fcc']  # sites of adsorption
coordinates = [([0,0,0],[0.65,0.65,0.40]),None]  # coordinates of multi-atom adsorbates
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

# initialize BFS and get net adsorption
surface_obj = BruteForceSurface(composition, adsorbates, ads_atoms, sites, coordinates, height,
                                regressor, 'graphs', 2, 'fcc111', (96,96), displace_e, scale_e)
surface_obj.get_net_energies()

# mask gross adsE grid with net adsorption
OH_ontop_energies = surface_obj.grid_dict_gross[('OH','ontop')][surface_obj.ads_dict[('OH','ontop')]]
O_fcc_energies = surface_obj.grid_dict_gross[('O','fcc')][surface_obj.ads_dict[('O','fcc')]]

# plot surface
fig = surface_obj.plot_hist()
fig.savefig('Ehist_GCN.png')