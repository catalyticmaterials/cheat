import ase.db, pickle
import numpy as np
from cheatools.surface import SurrogateSurface
from cheatools.lgnn import lGNN
#from cheatools.ocp_utils import BatchOCPPredictor
from cheatools.fairchem import BatchOCPPredictor
## load trained lGNN model
#with open(f'../train_lgnn/lGNN.state', 'rb') as input:
#    regressor = lGNN(trained_state=pickle.load(input))

# load OCP batch predictor
#config = "/groups/kemi/clausen/ocp/configs/is2re/all/equiformer_v2/equiformer_v2_N@8_L@4_M@2_31M.yml"
checkpoint = f"/groups/kemi/clausen/ocp/checkpoints/2A2CINO2P3R-dft-IS2RE31M/best_checkpoint.pt"
regressor = BatchOCPPredictor(None, checkpoint, batch_size=16, cpu=False)

# set random seed and surface composition
np.random.seed(42)
composition = {'Ag':0.2,'Pd':0.8} # elements not compatible with lGNN model will raise an error

# set adsorbate information e.g. *OH on on-top sites and *O on fcc sites
adsorbates = ['OH','O']
sites = ['ontop','fcc']

# displacement and scaling of adsorption energies
pure_ref_db = ase.db.connect('../gpaw/pure_refs.db')
pt_OH = pure_ref_db.get(element='Pt',ads='OH').e
pt_O = pure_ref_db.get(element='Pt',ads='O').e

displace_e = [-pt_OH,-pt_O]
scale_e = [1, 0.5]

# initialize surface and get net adsorption
surface = SurrogateSurface(composition, adsorbates, sites, regressor, template='ocp', size=(12,12), displace_e=displace_e, scale_e=scale_e, direct_e_input=None)
surface.get_net_energies() # if interadsorbate blocking should not be performed call .get_gross_energies() instead

# gross adsorption energies can be accessed here:
gross_OH = surface.grid_dict_gross[('OH','ontop')].flatten()
gross_O = surface.grid_dict_gross[('O','fcc')].flatten()

# if interadsorbate blocking is performed the net adsorption energies can be accessed here:
net_OH = surface.grid_dict_net[('OH', 'ontop')][surface.ads_dict[('OH', 'ontop')]]
net_O = surface.grid_dict_net[('O','fcc')][surface.ads_dict[('O','fcc')]]

# surface can be plotted w./w.o. adsorbates
fig = surface.plot_surface(show_ads=True)
fig.savefig('12x12_surface.png')
