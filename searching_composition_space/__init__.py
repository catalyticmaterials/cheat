__version__ = '1.0'

from .Simplex import Simplex
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import numpy as np
import itertools as it
from scipy import special, stats
from copy import deepcopy

# Define Boltzmann's constant
kB = 1.380649e-4 / 1.602176634  # eV K-1 (exact)
kBT = kB*300 # eV

def get_activity(energies, E_opt, n_surface_atoms, eU, jD=1.):
    '''
    Return the activity per surface atom calculated using the
    Angewandte Chemie equations 2-4 (doi: 10.1002/anie.202014374)
    '''
    jki = np.exp((-np.abs(energies - E_opt) + 0.86 - eU) / kBT)
    return np.sum(1. / (1. / jki + 1./jD)) / n_surface_atoms

def get_simulated_activities(fs, E_pure, params, n_atoms_zones, n_sites=1000):
	'''
	Return catalytic activities assuming the linear parameters for each element given.
	This is equivalent to a simple *OH adsorption model for ORR
	'''
	# Ensure that molar fractions are numpy
	fs = np.asarray(fs)
	
	# If 1D arrays, then make into 2D
	if np.ndim(fs) == 1:
		fs = fs.reshape(1, -1)
	
	# Get the number of molar fractions
	n_fs = fs.shape[0]
	
	# Get the number of elements
	n_elems = fs.shape[1]
	
	# Initiate activities list
	activities = np.zeros(n_fs)
	
	# Get the number of atoms in each adsorption site
	n_atoms_site = sum(n_atoms_zones)
	
	# Get the  oms_zones)
	
	# Get repeated colum indices
	cols = np.array(list(it.chain.from_iterable([[idx]*n_atoms for idx, n_atoms in enumerate(n_atoms_zones[1:])]))).reshape(1, -1)
	#print('cols:')
	#print(cols)
	
	# Iterate through molar fractions
	for f_idx, f in enumerate(fs):
		#print('f:', f, 'sum:', np.sum(f))
		
		# Get atoms in sites
		atom_ids = np.random.choice(range(n_elems), size=(n_sites, n_atoms_site), p=f)
		#print('atom_ids:')
		#print(atom_ids)
		
		#print('params[atom_ids[:, 0].reshape(n_sites, 1):')
		#print(params[atom_ids[:, 0].reshape(n_sites, 1)])
		#exit()
		
		#print('params[atom_ids[:, 0].reshape(n_sites, 1), atom_ids]:')
		#print(params[atom_ids[:, 0].reshape(n_sites, 1), atom_ids])
		#exit()
		
		#print('params[atom_ids[:, 0].reshape(n_sites, 1), atom_ids[:, 1:], cols]:')
		#print(params[atom_ids[:, 0].reshape(n_sites, 1), atom_ids[:, 1:], cols])
		#exit()
		
		# Get adsorption energies of sites
		E_ads = E_pure[atom_ids[:, 0]]
		E_ads += np.sum(params[atom_ids[:, 0].reshape(n_sites, 1), atom_ids[:, 1:], cols], axis=1)
		
		# Get activity of molar fraction
		activities[f_idx] = get_activity(E_ads, E_opt=0.86, n_surface_atoms=n_sites, eU=0.82)
		#print('activities[f_idx]:')
		#print(activities[f_idx])
		
	# Return activities, one for each molar fraction
	return activities
	
#def get_simulated_activities(fs, fs_peaks, std=0.15):
#	'''
#	Function for getting activities from summed normal distributions.
#	´fs´ are the grid of molar fractions to return activities from.
#	´fs_peaks´ are the molar fractions of the normal distribution peaks.
#	´std´ is the standard deviation along the diagonal in the covariance matrix.
#	'''
#	if np.ndim(fs) == 1:
#		fs = fs.reshape(1, -1)
#	
#	# Get cartesian coordinates of molar fractions
#	r = molar_fractions_to_cartesians(fs)

#	# Get cartesian coordinates of peak positions
#	r_peaks = molar_fractions_to_cartesians(fs_peaks)
#	
#	# Get number of samples in the grid
#	n_grid_samples = r.shape[0]
#	
#	# Initialize output activity values
#	activities = np.zeros(n_grid_samples)
#	
#	# Get the dimensionality
#	n_dim = r.shape[1]
#	
#	# Define simple independent covariance matrix and its inverse
#	cov = np.eye(n_dim) * std**2
#	inv_cov = np.eye(n_dim) / std**2

#	# Iterate through peaks
#	for r_peak in r_peaks:
#		
#		# Make temporary variable
#		z = r - r_peak
#		
#		# Get diagonal elements of the matrix product z.T @ inv_cov @ z
#		diagonal = np.asarray([row @ col for row, col in zip(z @ inv_cov, z)])

#		# Update activities with the current peak
#		activities += np.exp(-0.5*diagonal)
#	
#	# Return activities
#	return activities

def get_time_stamp(dt):
	'''
	Return the elapsed time in a nice format.
	
	Parameters
	----------
	dt: float
		Elapsed time in seconds.
		
	Return
	------
	string
		Elapsed time in a neat human-radable format.
	'''
	dt = int(dt)
	if dt < 60:
		return '{}s'.format(dt)
	elif 60 < dt < 3600:
		mins = dt//60
		secs = dt%60
		return '{:d}min{:d}s'.format(mins, secs)
	else:
		hs = dt//3600
		mins = dt//60 - hs*60
		secs = dt%60
		return '{:d}h{:d}min{:d}s'.format(hs, mins, secs)

def count_elements(elements, n_elems):
	count = np.zeros(n_elems, dtype=int)
	for elem in elements:
	    count[elem] += 1
	return count

def get_molar_fractions(step_size, n_elems, total=1., return_number_of_molar_fractions=False):
	'Get all molar fractions with the given step size'
	
	interval = int(total/step_size)
	n_combs = special.comb(n_elems+interval-1, interval, exact=True)
	
	if return_number_of_molar_fractions:
		return n_combs
		
	counts = np.zeros((n_combs, n_elems), dtype=int)

	for i, comb in enumerate(it.combinations_with_replacement(range(n_elems), interval)):
		counts[i] = count_elements(comb, n_elems)

	return counts*step_size

def get_random_molar_fractions(n_elems, n_molar_fractions=1, random_state=None):
	'Get ´size´ random molar fractions of ´n_elems´ elements'
	if random_state is not None:
		np.random.seed(random_state)

	fs = np.random.rand(n_molar_fractions, n_elems)
	return fs / np.sum(fs, axis=1)[:, None]

def get_molar_fractions_around(f, step_size, total=1., eps=1e-10):
	'Get all molar fractions with the given step size around the given molar fraction'	
	fs = []	
	n_elems = len(f)
	for pair, ids in zip(it.permutations(f, 2), it.permutations(range(n_elems), 2)):
	
		# Get molar fractions and their ids
		f0, f1 = pair
		id0, id1 = ids
		
		# Increment one molar fraction and decrement the other
		f0_new = f0 + (step_size - eps)
		f1_new = f1 - (step_size - eps)
		
		# Ignore if the new molar fractions are not between 0 and 1
		if f0_new <= total and f1_new >= 0.:
			
			# Make new molar fraction
			f_new = deepcopy(f)
			f_new[id0] = f0_new + eps
			f_new[id1] = f1_new - eps
			
			# Append to the output
			assert np.isclose(sum(f_new), 1.), "Molar fractions do not sum to unity : {}. Sum : {:.4f}".format(f_new, sum(f_new))
			fs.append(f_new)
			
	return np.array(fs)

def maximize_molar_fraction(func, n_elems, grid_step=0.05, step_threshold=0.005, func_args=()):
	'''
	Optimize ´func´ by first searching a regular grid with step size ´grid_step´.
	The maximum from the grid search is then used and optimized further on a grid,
	until the step size	has reached ´step_threshold´.
	Arguments to to ´func´ is passed via ´func_args´.
	'''
	
	# Get grid of molar fractions
	fs_grid = get_molar_fractions(grid_step, n_elems)
	
	# Convert to cartesian coordinates
	rs_grid = molar_fractions_to_cartesians(fs_grid)
	
	# Get function values on the grid
	ys_grid = func(rs_grid, *func_args)
	
	# Pick the point with the largest value on the grid
	idx_max = np.argmax(ys_grid)
	r_max = rs_grid[idx_max]
	f_max = cartesians_to_molar_fractions(r_max.reshape(1, -1))
	
	# Half the molar fraction step size
	step_size = grid_step / 2.
	
	# Optimize around the found maximum in ever decreasing steps
	while step_size > step_threshold:
		
		# Get molar fractions around the preliminary optimum
		fs_around = get_molar_fractions_around(f_max[0], step_size)
		
		# Convert to cartesian coordinates
		rs_around = molar_fractions_to_cartesians(fs_around)
		
		# Get function values of these coordinates
		ys_around = func(rs_around, *func_args)
		
		# Get index of the maximum value
		idx_max = np.argmax(ys_around)
		
		# Get cartesian coordinates of maximum
		r_max = rs_around[idx_max]
		
		# Convert maximum to molar fractions
		f_max = cartesians_to_molar_fractions(r_max.reshape(1, -1))
		
		# Half the molar fraction step size
		step_size /= 2.

	# Return molar fractions that maximize the function
	return f_max

def get_local_maxima(fs, func, step_size=0.01, func_args=(), converter=None):
	
	# Initiate list containers
	fs_max = []
	funcs_max = []

	for f in fs:
		
		# Convert molar fraction to another input (e.g. cartesian coordinates)
		# if a converter is given
		if converter is None:
			func_input = f.reshape(1, -1)
		else:	
			func_input = converter(f).reshape(1, -1)
		

		# Get the function value of the current molar fraction	
		func_max = func(func_input, *func_args)
	
		# Get molar fractions around the current molar fraction in the given step size
		fs_around = get_molar_fractions_around(f, step_size=step_size)
		
		# Convert molar fraction to another input (e.g. cartesian coordinates)
		# if a converter is given
		if converter is None:
			func_input = fs_around
		else:
			func_input = converter(fs_around)
		
		# Get function values
		func_vals = func(func_input, *func_args)

		# If none of the neighbors have a higher function value,
		# then the current molar fractions is a local maximum
		if np.all(func_vals < func_max):
			
			# Append the found maximum
			fs_max.append(f)
			funcs_max.append(func_max)
		
	return np.asarray(fs_max), np.asarray(funcs_max)

def check_optima(fs_optima, fs_peaks, f_threshold=0.05):	
	'''
	Return True if the molar fractions in ´fs_peaks´ correspond to those
	in ´fs_optima´ to within a threshold
	'''
	# Make into numpy
	fs_optima = np.asarray(fs_optima)
	fs_peaks = np.asarray(fs_peaks)
	
	# Initiate array, stating that no peaks have been identified yet
	found_peaks = np.zeros_like(fs_peaks)
	
	# Make into masked array
	fs_peaks = np.ma.masked_array(fs_peaks, mask=False)
	
	# Define small parameter to resolve near-equalities as equalities
	eps = 1e-6
	
	# Iterate through optima
	for f_optimum in fs_optima:

		# Get differences in molar fractions
		diff = np.abs(fs_peaks - f_optimum)
		
		# Sum the differences
		diff_sum = np.sum(diff, axis=1)
		
		# If the sum of the differences in molar fractions is below two times the threshold,
		# so that e.g. Ag20Ir5Pd75 is identified as the same as Ag20Pd80 for ´f_threshold´ = 0.05
		if np.any(diff_sum < 2*f_threshold+eps):
			
			# Get the index of the matched optima
			idx_match = np.argmin(diff_sum)
			
			# Mask entry
			fs_peaks.mask[idx_match] = True
			
			# Identify this optimum as identified
			found_peaks[idx_match] = f_optimum
	
	# Return array of peaks that have been identified successfully
	return found_peaks
	
def get_composition(f, metals, return_latex=False, saferound=True):
	
	# Make into numpy and convert to atomic percent
	f = np.asarray(f)*100
	
	if saferound:
		# Round while maintaining the sum, the iteround module may need
		# to be installed manually from pypi: "pip3 install iteround"
		import iteround
		f = iteround.saferound(f, 0)
	
	if return_latex:
		# Return string in latex format with numbers as subscripts
		return ''.join(['$\\rm {0}_{{{1}}}$'.format(m,f0) for m,f0 in\
			zip(metals, map('{:.0f}'.format, f)) if float(f0) > 0.])
	else:
		# Return composition as plain text
		return ''.join([''.join([m, f0]) for m,f0 in\
			zip(metals, map('{:.0f}'.format, f)) if float(f0) > 0.])

def molar_fractions_to_cartesians(fs):
	
	# Make into numpy
	fs = np.asarray(fs)

	if fs.ndim == 1:
		fs = np.reshape(fs, (1, -1))

	# Get vertices of the multidimensional simplex
	n_elems = fs.shape[1]
	vertices = Simplex(n_elems).get_vertices()
	
	# Get cartesian coordinates corresponding to the molar fractions
	return fs @ vertices

def cartesians_to_molar_fractions(rs):
	
	# Make into numpy
	rs = np.asarray(rs)
	
	# Add column of ones to ´rs´ to account for the restriction
	# that the molar fractions must sum to unity
	rs = np.concatenate((rs, np.ones((rs.shape[0], 1))), axis=1)
	
	# Get vertices of the multidimensional simplex
	n_elems = rs.shape[1]
	vertices = Simplex(n_elems).get_vertices()
	
	# Add column of ones to ´vertices´ to account for the restriction
	# that the molar fractions must sum to unity
	vertices = np.concatenate((vertices, np.ones((vertices.shape[0], 1))), axis=1)
	
	# Get molar fractions corresponding to the cartesian coordinates
	# r = fV <=> r^T = (fV)^T = V^T f^T (i.e. on the form Ax = b that np.linalg.solve takes as input)
	return np.linalg.solve(vertices.T, rs.T).T

def expected_improvement(x, gpr, y_max):
	'Return the expected improvement for the sample x'
	
	# Reshape feature vector
	if x.ndim == 1:
		x = x.reshape(1, -1)
	
	# Get best estimate and uncertainty for x with the GP surrogate model
	mu, std = gpr.predict(x, return_std=True)
	
	# Get difference between best estimate and the highest value found so far
	diff = mu - y_max
	
	# Get quotient between the difference and the uncertainty
	Z = diff / std
	
	# Return the expected improvement
	return diff*stats.norm.cdf(Z) + std*stats.norm.pdf(Z)
	
	
def make_triangle_ticks(ax, start, stop, tick, n, offset=(0., 0.),
						fontsize=12, ha='center', tick_labels=True):
	r = np.linspace(0, 1, n+1)
	x = start[0] * (1 - r) + stop[0] * r
	x = np.vstack((x, x + tick[0]))
	y = start[1] * (1 - r) + stop[1] * r
	y = np.vstack((y, y + tick[1]))
	ax.plot(x, y, 'black', lw=1., zorder=0)
	
	if tick_labels:
	
		# Add tick labels
		for xx, yy, rr in zip(x[0], y[0], r):
			ax.text(xx+offset[0], yy+offset[1], f'{rr*100.:.0f}',
					fontsize=fontsize, ha=ha)

def mscatter(x, y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    ax = ax or plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
	new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
		'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
		cmap(np.linspace(minval, maxval, n)))
	return new_cmap

def prepare_triangle_plot(ax, elems):
	
	# Set the number of ticks to make
	n_ticks = 5
	tick_labels = True

	# Specify vertices as molar fractions
	fs_vertices = [[1., 0., 0.],
				   [0., 1., 0.],
				   [0., 0., 1.]]
	
	# Get cartesian coordinates of vertices
	xs_vertices, ys_vertices = molar_fractions_to_cartesians(fs_vertices).T
	
	# Get height of triangle
	h = 3**0.5/2
	
	# Define padding to put the vertex text neatly
	pad = [[-0.06, -0.06],
		   [ 0.06, -0.06],
		   [ 0.00,  0.08]]
	has = ['right', 'left', 'center']
	vas = ['top', 'top', 'bottom']

	# Make ticks and tick labels on the triangle axes
	left, right, top = np.concatenate((xs_vertices.reshape(-1,1), ys_vertices.reshape(-1,1)), axis=1)

	tick_size = 0.035
	bottom_ticks = 0.8264*tick_size * (right - top)
	right_ticks = 0.8264*tick_size * (top - left)
	left_ticks = 0.8264*tick_size * (left - right)

	# Set axis limits
	ax.set_xlim(-0.05, 1.05)
	ax.set_ylim(-0.05, h+0.05)

	# Plot triangle edges
	ax.plot([0., 0.5], [0., h], '-', color='black', zorder=0)
	ax.plot([0.5, 1.], [h, 0.], '-', color='black', zorder=0)
	ax.plot([0., 1.], [0., 0.], '-', color='black', zorder=0)
	
	# Remove spines
	for direction in ['right', 'left', 'top', 'bottom']:
		ax.spines[direction].set_visible(False)
	
	# Remove tick and tick labels
	ax.tick_params(which='both', bottom=False, labelbottom=False, left=False, labelleft=False)
	ax.set_aspect('equal')
		
	make_triangle_ticks(ax, right, left, bottom_ticks, n_ticks, offset=(0.03, -0.08), ha='center', tick_labels=tick_labels)
	make_triangle_ticks(ax, left, top, left_ticks, n_ticks, offset=(-0.03, -0.015), ha='right', tick_labels=tick_labels)
	make_triangle_ticks(ax, top, right, right_ticks, n_ticks, offset=(0.015, 0.02), ha='left', tick_labels=tick_labels)

	# Show axis labels (i.e. atomic percentages)
	ax.text(0.5, -0.14, f'{elems[0]} content (at.%)', rotation=0., fontsize=12, ha='center', va='center')
	ax.text(0.9, 0.5, f'{elems[1]} content (at.%)', rotation=-60., fontsize=12, ha='center', va='center')
	ax.text(0.1, 0.5, f'{elems[2]} content (at.%)', rotation=60., fontsize=12, ha='center', va='center')
	
	# Show the chemical symbol as text at each vertex
	for idx, (x, y, (dx, dy)) in enumerate(zip(xs_vertices, ys_vertices, pad)):
		ax.text(x+dx, y+dy, s=elems[idx], fontsize=14, ha=has[idx], va=vas[idx])
	
	return ax

def get_pseudo_ternary_molar_fractions(fs):
	'Return molar fractions array with all but the two first columns added together'
	# Make into numpy
	fs = np.asarray(fs)
	
	# If one molar fractions is specified then make it into a two-dimensional array
	if np.ndim(fs) == 1:	
		fs = fs.reshape(1, -1)
	
	# Concatenate the molar fractions row-wise
	return np.concatenate((
			  fs[:, 0].reshape(-1, 1),
			  fs[:, 1].reshape(-1, 1),
			  np.sum(fs[:, 2:], axis=1).reshape(-1, 1)), axis=1)
			  
def prepare_tetrahedron_plot(ax, elems):
	
	# Set the number of ticks to make
	n_ticks = 5
	tick_labels = True

	# Specify vertices as molar fractions
	fs_vertices = [[1., 0., 0., 0.],
				   [0., 1., 0., 0.],
				   [0., 0., 1., 0.],
				   [0., 0., 0., 1.]]
	
	# Get cartesian coordinates of vertices
	rs_vertices = molar_fractions_to_cartesians(fs_vertices)

	# Set axis limits
	ax.set_xlim(-0.05, np.max(rs_vertices[:, 0]) + 0.05)
	ax.set_ylim(-0.05, np.max(rs_vertices[:, 1]) + 0.05)
	ax.set_zlim(-0.05, np.max(rs_vertices[:, 2]) + 0.05)

	# Get all combinations of pairs of vertices
	for edge in it.combinations(rs_vertices, 2):
	
		# Plot simplex edge
		ax.plot(*np.array(edge).T, ls='solid', color='black')
	
	# Remove spines
	for direction in ['right', 'left', 'top', 'bottom']:
		ax.spines[direction].set_visible(False)
	
	# Remove spines etc.
	ax.set_axis_off()
	
	# Define padding to put the vertex text neatly
	pad = [[-0.03, -0.03, 0.00],
		   [ 0.03, -0.03, 0.00],
		   [ 0.03,  0.03, 0.00],
		   [ 0.00,  0.00, 0.06]]
	has = ['right', 'left', 'center', 'center']
	vas = ['top', 'top', 'bottom', 'center']
	
	# Show the chemical symbol as text at each vertex
	for idx, (r, dr) in enumerate(zip(rs_vertices, pad)):
		ax.text(*(r+dr).T, s=elems[idx], fontsize=14, ha=has[idx], va=vas[idx])
	
	return ax

def get_number_of_samples_for_finding_optima(func, sampler_func, n_elems, gpr=None,
											 func_args=(), step_size=0.05, max_samples=200):
	'Return the number of samples necessary for find the optima of the function'
	
	# If gaussian process regressor is not given then use the default
	if gpr is None:
		from .gaussian_process import GPR
		gpr = GPR()
	
	# Initiate variables and containers
	fs_train = []
	activities = []
	
	# Define grid of molar fractions to find optima for
	fs_grid = get_molar_fractions(step_size=step_size, n_elems=n_elems)
	
	# Get corresponding cartesian coordinates
	rs_grid = molar_fractions_to_cartesians(fs_grid)
	
	# Get activities on a fine grid
	target_func = func(fs_grid, *func_args)
	
	# Get maxima of surrogate function (i.e. the actual peaks after the summation of the distribution,
	# on a the grid)
	fs_peaks, activity_maxima = get_local_maxima(fs_grid, func, step_size=step_size,
												 converter=None, func_args=func_args)
	print(f'fs_peaks (actual rounded to {step_size*100}% steps):')
	print(fs_peaks)
	
	# Keep iterating until the optima have all been located
	while True:
	
		# Get molar fractions to sample by updating the fs_train array
		fs_train = sampler_func(fs_train, activities=activities, gpr=gpr)
		n_samples = len(fs_train)

		# Get corresponding cartesian coordinates of molar fractions				
		rs_train = molar_fractions_to_cartesians(fs_train)

		# Get corresponding activity values
		activities = func(fs_train, *func_args)

		# Train regressor
		gpr.fit(rs_train, activities)
		print(f'[KERNEL] ({n_samples}) {gpr.kernel_}')

		# Predict on the fine grid
		activity_pred = gpr.predict(rs_grid)

		# Get maxima of surrogate function
		fs_maxima, activity_maxima = get_local_maxima(fs_grid, gpr.predict, step_size=step_size,
													  converter=molar_fractions_to_cartesians)
	
		# Check if the known optima have been located to within a threshold distance
		found_peaks = check_optima(fs_maxima, fs_peaks, f_threshold=step_size)
		
		# If all rows of ´found_peaks´ has an element that is different from zero,
		# then all peaks have been located successfully and the search can stop
		if np.all(np.any(found_peaks, axis=1)):
			print(f'[INFO] Correct optima located at {n_samples} samples')
			return n_samples
		
		if n_samples >= max_samples:
			msg = f'[WARNING] Number of samples is above the maximum threshold ({n_samples})'
			print(msg)
			raise ValueError(msg)
