import numpy as np
from .__init__ import get_random_molar_fractions, maximize_molar_fraction, expected_improvement

class BayesianSampler():

	def __init__(self, n_elems):
		self.n_elems = n_elems

	def get_molar_fraction_samples(self, fs_train, activities, gpr, **kwargs):
		'Return an updated array of molar fractions to sample'
	
		# If no molar fractions have been sampled so far
		if len(fs_train) == 0:
		
			# Get two random molar fractions to start with
			fs_train = get_random_molar_fractions(self.n_elems, n_molar_fractions=2)
		
		else:	
			# Optimize acquisition function to select the next,
			# most optimal molar fraction
			f_next = maximize_molar_fraction(expected_improvement,
											 self.n_elems,
											 grid_step=0.05,
											 func_args=(gpr, max(activities)))
		
			# Append molar fraction to the training set
			fs_train = np.append(fs_train, f_next, axis=0)
	
		return fs_train
