import numpy as np
from .Simplex import Simplex
from .__init__ import molar_fractions_to_cartesians, cartesians_to_molar_fractions, get_random_molar_fractions

class PointRepulsion():	
	
	def __init__(self, n_elems, max_iter=10000):
		self.n_elems = n_elems
		self.vertices = Simplex(n_elems).get_vertices()
		self.max_iter = max_iter

	def within_simplex(self, r):
		'''
		Return True if the points are within the simplex
		and False if at least one is not
		'''
		# Make position into molar fractions
		f = cartesians_to_molar_fractions(r)
	
		# If all molar fractions are between 0 and 1,
		# then the point is inside the simplex
		if np.all(0. < f) and np.all(f < 1.):
			return True
		else:
			return False

	def repel_points(self, r, dt=0.001, f_max=1.):
	
		# Make into numpy
		r = np.asarray(r)
	
		# Get number of points
		n_points = r.shape[0]
	
		# Set iteration counter to 0 initially
		n_iter = 1
	
		# Relax repelling 'forces' on points until a relaxed state is obtained
		while True:
		
			# Initiate repulsion vectors
			repulsion = np.zeros((n_points, self.n_elems-1))
	
			# Get vector from the points and perpendicular to the plane that is
			# one dimensional lower than the dimensionality of the vector to the point
			for vertex_idx, vertex in enumerate(self.vertices):
			
				# Reshape vertex vector
				vertex = vertex.reshape((1, -1))
			
				# Get the n_elems - 1 next vertex vectors that together span
				# a plane that is the 'edge' of the simplex
				indices = [idx % self.n_elems for idx in range(vertex_idx+1, vertex_idx + (self.n_elems - 1))]
				vertices_next = self.vertices[indices]
			
				# Make neighboring vertex vectors relative to the current vertex
				vertices_next = vertices_next - vertex
			
				# Get coefficients in the projections onto the current edge plane of the simplex
				coefficients = np.linalg.solve(vertices_next @ vertices_next.T, vertices_next @ (r-vertex).T).T
			
				# Get projection of the points onto the current edge plane
				# of the simplex as cartesian coordinates
				projection = coefficients @ vertices_next
			
				# Get vectors from points to the current simplex edge plane
				d = (r - vertex) - projection
			
				# Get inverse square repulsion between points and edge
				# Ensure correct division by first transposing d,
				# and then the resulting vector
				force = (d.T / (np.sum(d**2, axis=1))**(3/2)).T
				repulsion += force
			
			# Get repulsive force between points
			for r_idx, r0 in enumerate(r):
			
				# Get vectors to the other points.
				# The vectors are split into two arrays to avoid self-interaction of a point
				d_before = r[:r_idx] - r0
				d_after = r[r_idx+1:] - r0
			
				# Get distances to the other points
				dists_before = np.sum(d_before**2, axis=1)
				dists_after = np.sum(d_after**2, axis=1)
			
				# Get inverse square forces
				force_before = (d_before.T / dists_before**(3/2)).T
				force_after = (d_after.T / dists_after**(3/2)).T
			
				# Update cummulated repulsions
				repulsion[:r_idx] += force_before
				repulsion[r_idx+1:] += force_after
		
			# If the forces are all below a threshold value, then stop relaxation
			norm_repulsion = np.sum(repulsion**2, axis=1)**0.5
			repulsion_max = max(norm_repulsion)
		
			# Move points a little step in the direction of the net repelling force
			dt_min = 1e-2 / n_iter
			dt = min(1e-1 / repulsion_max, dt_min)

			while True:
		
				# Update positions
				r_new = r + repulsion*dt
		
				# Check that all points are still within the simplex.
				# If not, then decrease the time step and check again
				if self.within_simplex(r_new):
					r = r_new
					break
				else:
					dt /= 2.
					print(f'[INFO] Decreasing time step to {dt:.1e}, because a point was outside the simplex')
			
			# Increment iteration counter
			n_iter += 1
		
			# Return positions if forces have relaxed below the threshold
			if repulsion_max < f_max:
				print(f'[CONVERGED] {n_iter} iterations')
				return r
		
			# If maximum number of iterations is reached, then terminate
			if n_iter == self.max_iter:	
				print(f'Maximum number of iterations ({self.max_iter}) reached')
				raise ValueError(f'Maximum number of iterations ({self.max_iter}) reached')

	def get_molar_fraction_samples(self, fs_train, **kwargs):
		'Return an updated array of molar fractions to sample'
		'the kwargs are ignored, but is here for consistency with other sampling methods'
	
		# Get the number of samples
		n_samples = len(fs_train)
	
		# If no samples yet
		if n_samples == 0:
		
			# Get two random molar fractions to start with
			fs_train = get_random_molar_fractions(self.n_elems, n_molar_fractions=2)
	
		# If there are samples already
		else:
	
			# Get ´n_samples´+1 random molar fractions to start with
			fs_train = get_random_molar_fractions(self.n_elems, n_molar_fractions=n_samples+1)
	
		# Get corresponding cartesian coordinates
		xy_train = molar_fractions_to_cartesians(fs_train)
	
		# Apply a repelling force between points and the edges of the simplex
		# to create as uniform a sampling as possible
		xy_train = self.repel_points(xy_train)
	
		# Get corresponding molar fractions
		fs_train = cartesians_to_molar_fractions(xy_train)

		return fs_train
