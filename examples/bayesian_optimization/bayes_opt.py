import ase.db, pickle
import numpy as np
from cheatools.surface import SurrogateSurface
from cheatools.lgnn import lGNN
from cheatools.bayesian import expected_improvement, random_comp, opt_acquisition
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

def get_activity(surface, G_opt=0.10, eU=0.82, T=298.15, j_d=1):
    """Measure of ORR catalytic acitivity based on a kinetic "volcano" expression"""
    kb = 8.617e-5
    j_k = np.array([])
    for key in surface.grid_dict_gross.keys():
        e = surface.grid_dict_gross[key][surface.ads_dict[key]]
        j_ki = np.exp(-(np.abs(e - G_opt) - 0.86 + eU) / (kb*T))
        j_k = np.concatenate([j_k, j_ki])
    j = 1/np.prod(surface.size) * np.sum(1 / (1/j_d + 1/j_k))
    return j

def comp2act(comp):
    """Wrapper function for running a surrogate surface to predict ORR activity of a given alloy composition"""

    # run surrogate surface
    surface = SurrogateSurface(comp, adsorbates, sites, regressor, template='lgnn', size=(96,96), displace_e=displace_e, scale_e=scale_e)
    surface.get_net_energies()

    activity = get_activity(surface) # get ORR activity 
    return activity


# set random seed
np.random.seed(42)

# Define elements to use
elements = ['Ag', 'Ir', 'Pd', 'Pt', 'Ru']

# set adsorbate information e.g. *OH on on-top sites and *O on fcc sites
adsorbates = ['OH','O']
sites = ['ontop','fcc']

# displacement and scaling of adsorption energies
pure_ref_db = ase.db.connect('../gpaw/pure_refs.db')
pt_OH = pure_ref_db.get(element='Pt',ads='OH').e
pt_O = pure_ref_db.get(element='Pt',ads='O').e

displace_e = [-pt_OH,-pt_O]
scale_e = [1, 0.5]

# load trained lGNN model
with open(f'../train_lgnn/lGNN.state', 'rb') as input:
    regressor = lGNN(trained_state=pickle.load(input))

# Define kernel and Gaussian process regression
kernel = C(constant_value=0.05, constant_value_bounds=(1e-5, 1e1))\
        *RBF(length_scale=0.5, length_scale_bounds=(1e-5, 1e1))
gpr = GPR(kernel=kernel, alpha=1e-5, n_restarts_optimizer=10)

# Define exploration-exploitation trade-off parameter for the acquisition function (higher = more exploration)
xi = 0.01

# Number of steps to run the optimization for and number of initial random samples
n_steps = 50
n_random = 5

# Define filename where sampled molar fraction are written to
filename = 'sampled.csv'
with open(filename, 'w') as filetowrite:
    filetowrite.write(','.join(elements) + ',Activity rel. to Pt(111)\n')

# Reference activity of 2/3 *OH coverage on pure Pt(111)
pt_surface = SurrogateSurface({'Pt':1.0}, adsorbates, sites, regressor, template='lgnn', size=(12,12), displace_e=displace_e, scale_e=scale_e, direct_e_input=[pt_OH,pt_O])
pt_surface.get_net_energies()
pt_act = get_activity(pt_surface)

# Initiate activities list
fs, acts = [], []

# Sample random points to make initial GPR fit
for i in range(n_random):
    temp = random_comp(elements, max=0.9) # impose maximum molar fraction to avoid sampling pure elements
    fs.append(list(temp.values()))
    acts.append(comp2act(temp))
    
    # Write result to file
    with open(filename, 'a') as file_:
        file_.write(','.join(f"{e:.2f}" for e in temp.values()) + f',{acts[-1] / pt_act * 100:.0f}%\n')

# Initial fit of the Gaussian process regressor
gpr.fit(np.array(fs), acts)

# Iterate through the optimization steps
for n_samples in range(n_random, n_steps):

    # Select the best composition for activity sampling using acquisition function. 
    f_opt = opt_acquisition(fs, gpr, elements, acq_func=expected_improvement, xi=xi, step_size=0.01, n_random=1000)

    # Sample the composition
    act = comp2act(dict(zip(elements, f_opt)))

    # Write result to file
    with open(filename, 'a') as file_:
        file_.write(','.join(f"{e:.2f}" for e in f_opt) + f',{act / pt_act * 100:.0f}%\n')

    # Add the sampled composition and activity to the dataset and update the Gaussian process regressor
    fs = np.vstack((fs, [f_opt]))
    acts += [act]
    gpr.fit(fs, acts)
