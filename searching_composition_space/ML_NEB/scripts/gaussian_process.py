from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
from sklearn.utils.optimize import _check_optimize_result
import scipy

def gpr_optimizer(obj_func, initial_theta, bounds):
	opt_res = scipy.optimize.minimize(
				obj_func,
				initial_theta,
				method='L-BFGS-B',
				jac=True,
				bounds=bounds,
				options=dict(maxiter=50000)
				)
	_check_optimize_result('lbfgs', opt_res)
	return opt_res.x, opt_res.fun

class GPR(GaussianProcessRegressor):
	
	def __init__(self, kernel=None, alpha=1e-10, n_restarts_optimizer=25):
		
		if kernel is None:
			# Define kernel to use for Gaussian process regressors
			kernel = C(constant_value=0.05, constant_value_bounds=(1e-5, 1e1))\
					 *RBF(length_scale=0.5, length_scale_bounds=(1e-5, 1e1))
		
		super().__init__(kernel=kernel,
						 alpha=alpha,
						 n_restarts_optimizer=n_restarts_optimizer,
						 optimizer=gpr_optimizer)
