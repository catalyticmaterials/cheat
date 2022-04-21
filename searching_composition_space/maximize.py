import numpy as np
from .Simplex import Simplex

class SimplexOptimizer():
	
	def __init__(self, n_vertices):
		
		self.n_vertices = n_vertices
		
		# Get simplex vertex coordinates
		simplex = Simplex(n_vertices)
		self.vertices = simplex.get_vertices()
		self.vertex_ids = np.array([list(range(0, idx)) + list(range(idx+1, n_vertices)) for idx in range(n_vertices)])
		
		# Add column of ones to ´vertices´ to account for the restriction
		# that the molar fractions must sum to unity
		self.vertices_padded = np.concatenate((self.vertices, np.ones((n_vertices, 1))), axis=1)
		
	
	def to_cartesian(self, fs):	
		'''
		Return cartesian coordinates of the barycentric coordinates
		'''
		fs = np.asarray(fs)
		
		# Make into 2D array, if a 1D array
		if np.ndim(fs) == 1:
			fs = fs.reshape(1, -1)
		
		# Take matrix product with the vertex coordinates
		# to get the cartesian coordinates
		return fs @ self.vertices
	
	
	def to_barycentric(self, rs, eps=1e-15):
		'''
		Return barycentric coordinates of the cartesian coordinates
		'''
		rs = np.asarray(rs)
		
		# Make into 2D array, if a 1D array
		if np.ndim(rs) == 1:
			rs = rs.reshape(1, -1)
		
		# Get the number of samples
		n_samples = rs.shape[0]
		
		# Add column of ones to the array of cartesian coordinates
		rs_padded = np.concatenate((rs, np.ones((n_samples, 1))), axis=1)
		
		# Get molar fractions corresponding to the cartesian coordinates
		# r = fV <=> r^T = (fV)^T = V^T f^T (i.e. on the form Ax = b that np.linalg.solve takes as input)
		fs = np.linalg.solve(self.vertices_padded.T, rs_padded.T).T
		
		# Set any negative barycentric coordinates to zero
		# (negative molar fractions could occur in the numerical solution of
		# the matrix equation wit np.linalg.solve)
		mask = (fs < 0.) * (fs > -eps)
		if np.any(mask):
#			print(f'[SimplexOptimizer] Small negative barycentric coordinates found. Setting {fs[mask]} to 0.')
			fs[mask] = 0.
		
		# Return barycentric coordinates
		return fs
		
	
	def get_gradient(self, fun, f0=None, r0=None, eps=1e-7, use_cartesian=False, func_args=(), func_kwargs={}):
		'''
		Return the (cartesian) gradient of the function in the specified point
		
		Parameters
		----------
		fun: callable
			Function takings barycentric coordinates as input
			(or cartesian coordinates if ´use_cartesian´ is set to True
		f0: array-like
			Barycentric coordinates to evaluate the gradient at
		eps: float
			Step taken along each unit vector when computing the directional derivatives
		use_cartesian: bool
			If True, then use cartesian coordinates as input to ´fun´, else barycentric coordinates
		func_args: tuple
			Arguments to ´fun´
		func_kwargs: dict
			Keyword arguments to ´fun´
		
		Return
		------
		gradient: array
			The gradient in cartesian coordinates evaluated at the point ´f0´	
		'''
		if f0 is None and r0 is not None:	
			f0 = self.to_barycentric(r0)
		else:
			# Ensure that f0 is a numpy array
			f0 = np.asarray(f0)
			
			# Ensure that f0 is 2D
			if np.ndim(f0) == 1:
				f0 = f0.reshape(1, -1)
		
			# Get cartesian coordinates of point
			r0 = self.to_cartesian(f0)
		
		# Initiate gradients
		gradients = np.zeros_like(r0)
		
		# Iterate through points
		for point_idx, (r, f) in enumerate(zip(r0, f0)):
			
			# Get index of largest molar fraction
			f_idx = np.argmax(f)
#			print('f_idx: ', f_idx)
			
			# Get vectors to take steps along so that no step taken goes outside the simplex
			# Get vertices, except the one corresponding to the largest molar fraction
			vectors = np.delete(self.vertices, f_idx, axis=0)
			vertex_ids = self.vertex_ids[f_idx]
#			print('vertex_ids')
#			print(vertex_ids)
			
			# Subtract the vertex coordinate corresponding to the largest molar fraction
			# to get the simplex edge vectors pointing away from this vertex
			vectors -= self.vertices[f_idx]
#			print('vectors')
#			print(vectors)

			if use_cartesian:
			
				# Get cartesian coordinates of point after stepping along directional vectors
				point_step = r + eps*vectors
				point = r.reshape(1, -1)
			else:
			
				# Get barycentric coordinates of the points obtained from stepping along the directional vectors
				point_step = self.to_barycentric(r + eps*vectors)
				point = f.reshape(1, -1)

#			print('point_step:')
#			for v in point_step:
#				print(','.join(map('{:.10e}'.format, v)))

#			print('point:')
#			for v in point:
#				print(','.join(map('{:.10e}'.format, v)))
			
			# Get the directional derivatives by stepping a small amount along the edge vectors
			directional_derivatives = (fun(point_step, *func_args, **func_kwargs) - fun(point, *func_args, **func_kwargs)) / eps
#			print('directional_derivatives:')
#			print(directional_derivatives)
			
			# Get the gradient from the directional derivatives
			#     V      @  grad_f.T  =   D_f.T     <=>    Ax = b, x=np.linalg.solve(A, b)
			#(m-1 x m-1)   (m-1 x 1)    (m-1 x 1)
			gradient = np.linalg.solve(vectors, directional_derivatives.T).T
#			print('gradient (1):')
#			print(gradient)
			
			# If any molar fraction is zero, then the point is on an edge of the simplex
			molar_fraction_is_zero = np.isclose(f, 0.)
#			print('molar_fraction_is_zero')
#			print(molar_fraction_is_zero)
			if np.any(molar_fraction_is_zero):
				
				# Test if the gradient points outside the simplex
#				print('r')
#				print(r)
#				print('1e-10*gradient')
#				print(1e-10*gradient)
				f_step = self.to_barycentric(r + 1e-10*gradient)
#				print('f_step')
#				print(f_step)
				
				# If gradient points outside the simplex
				if np.any(f_step < 0.) or np.any(f_step > 1.):
					
					# Get indices of the barycentric coordinates that are non-zero
					f_edge_ids = np.nonzero(molar_fraction_is_zero)[0]
#					print('f_edge_ids')
#					print(f_edge_ids)
				
					# Get indices of the directional vectors not pointing along this simplex edge
					vector_ids = [np.nonzero(vertex_ids == f_edge_idx)[0][0] for f_edge_idx in f_edge_ids]
#					print('vector_ids:')
#					print(vector_ids)
				
					# Get indices of the directional vectors pointing along this simplex edge
#					vector_ids = [[idx for idx in range(self.n_vertices-1) if idx != vector_idx] for vector_idx in vector_ids]
#					vector_ids = np.unique(vector_ids)
#					print('vector_ids:')
#					print(vector_ids)
#				
#					print('vectors[vector_ids]')
#					print(vectors[vector_ids])
					
					# Project the gradient onto the directional vectors
					projection_coefficients = self.get_projection_coefficient(gradient, vectors)
#					print('projection_coefficients')
#					print(projection_coefficients)
				
					# If any of the projection coefficients of the vectors not pointing along the simplex edge
					# are negative, then set them to zero to avoid that the gradient points outside the simplex
#					print('projection_coefficients[vector_ids] < 0.')
#					print(projection_coefficients[vector_ids] < 0.)
					
					# Iterate through vector indices of vectors not pointing along the simplex edge
					for vector_idx in vector_ids:
						
						# If this vector's projection coefficient is negative, then set the projection
						# coefficient to zero, as a step along this vector would be a step outside the simplex
						if projection_coefficients[vector_idx] < 0.:
							projection_coefficients[vector_idx] = 0.
							
#					print('projection_coefficients')
#					print(projection_coefficients)
					
					# Update gradient
#					print('vectors')
#					print(vectors)
					gradient = projection_coefficients @ vectors
#					print('gradient (2):')
#					print(gradient)
				
					#exit()

			# Set gradient of the current point
			gradients[point_idx] = gradient
		
		# Return gradients
		return gradients
	
	def get_projection_coefficient(self, vector, projection_vectors):
		'''
		Return projection coefficients of a vector onto the given projection vectors
		
		Parameters
		----------
		vector: array-like
			Vector to get projections for
		projection_vectors: array-like
			Vector(s) to project onto
		
		Return
		------
		projection coefficients: array
		'''
		n_projection_vectors = len(projection_vectors)
		
		return vector @ projection_vectors.T
		
		
	def maximize(self, fun, f0, tol=1e-3, step=0.01, max_iter=10000, eps=1e-7, func_args=(), func_kwargs={}, **kwargs):
		'''
		Maximize a function on a simplex.
	
		Parameters
		----------
		fun: callable
			Function to be maximized
		f0: array-like
			Initial barycentric coordinated to start maximization at.
			The coordinates must sum to unity
		tol: float
			Tolerance at which to terminate the maximization.
			The maximization terminates when the L2 norm of the gradient is less than ´tol´.
		step: float
			Maximum step size to take along the normalized gradient
		max_iter: int
			Maximum number of iterations
		eps: float
			Step taken along each unit vector when computing the directional derivatives
		func_args: tuple
			Arguments to ´fun´
		func_kwargs: dict
			Keyword arguments to ´fun´
	
		Return
		------
		f1: array
			(Local) maximum resulting from the specified initial input
		'''
		# Make initial molar fraction into numpy
		f0 = np.asarray(f0)
		
		# Store the original step size
		step_orig = step
		eps_orig = eps
		
		# Initiate iteration count to zero
		n_iter = 0
		
		# Keep looping until the gradients are small enough
		while True:
			
			# Increment iteration count
			n_iter += 1
			
			# Terminate if the maximum number of iterations has been reached
			if n_iter > max_iter:
				raise ValueError(f'[ERROR] Maximum number of iterations reached ({max_iter})')
			
			# Get gradient at the initial point
#			print('f0:')
#			print(','.join(map('{:.4f}'.format, f0.ravel())))
			gradient = self.get_gradient(fun, f0, eps=eps, func_args=func_args, func_kwargs=func_kwargs, **kwargs)
#			print('gradient:')
#			print(gradient)
			
			# Take a step along the direction of the gradient
			norm_gradient = np.sum(gradient**2)**0.5
#			print('norm_gradient', f'{norm_gradient:.4f}')
			
			# Stop optimization when the length of the gradient is smaller than the tolerance
			if norm_gradient < tol:
				if n_iter == 1:
					f1 = np.asarray(f0).reshape(1, -1)
				break
			
			# Get gradient with unit length
			gradient /= norm_gradient
			
			# Get cartesian coordinates of point
			r0 = self.to_cartesian(f0)
			
			# Set step size to half the length of the gradient or the original step size,
			# whichever is smaller. Empirically, this seems to give good convergence..
			#step = min(0.5*norm_gradient, step_orig)
			step = min(np.random.random_sample()*norm_gradient, np.random.random_sample()*step_orig)
#			print('step: ', step)
			
			# Keep looping with smaller and smaller steps along the gradient so that 
			# a step does not go outside the simplex edges
#			c = 0
			while True:
#				c += 1
#				print('r0:', r0)
				# Get updated cartesian coordinates after taking a step along the gradient
				r1 = r0 + step*gradient
#				print('r1:', r1)
				
				# Convert cartesian coordinates to barycentric coordinates
				f1 = self.to_barycentric(r1)
#				print('f1:', f1)
				
#				if c == 3:
#					exit()
				
				# If all barycentric coordinates are all positive then proceed
				if np.all(f1 >= 0.):
					break
				
				# If any of the barycentric coordinates are negative, then reduce the step taken
				else:
					step /= 2.
					
			# Update point
			f0 = f1
#			print('f1:', ', '.join(map('{:.4f}'.format, f1.ravel())))
			
#		print(f'[SimplexOptimizer] Maximum found after {n_iter} iterations')
		
		# Return the point having a gradient with length
		# below the threshold
		return f1
