import numpy as np
from .Simplex import Simplex

class SimplexOptimizer():
	
	def __init__(self, n_vertices):
		
		self.n_vertices = n_vertices
		
		# Get simplex vertex coordinates
		simplex = Simplex(n_vertices)
		self.vertices = simplex.get_vertices()
		
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
	
	
	def to_barycentric(self, rs):
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
		return np.linalg.solve(self.vertices_padded.T, rs_padded.T).T	
		
	
	def get_gradient(self, fun, f0=None, r0=None, eps=1e-7, *args, **kwargs):
		'''
		Return the (cartesian) gradient of the function in the specified point
		
		Parameters
		----------
		fun: callable
			Function takings barycentric coordinates as input
		f0: array-like
			Barycentric coordinates to evaluate the gradient at
		eps: float
			Step taken along each unit vector when computing the directional derivatives
		args: tuple
			Arguments to ´fun´
		kwargs: dict
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
		
		# Get index of largest molar fraction
		f_idx = np.argmax(f0)
		
		# Get vectors to take steps along so that no step taken goes outside the simplex
		# Get vertices, except the one corresponding to the largest molar fraction
		vectors = np.delete(self.vertices, f_idx, axis=0)
		
		# Subtract the vertex coordinate corresponding to the largest molar fraction
		# to get the simplex edge vectors pointing away from this vertex
		vectors -= self.vertices[f_idx]
		
		# Initiate gradients
		gradients = np.zeros_like(r0)
		
		# Iterate through points
		for point_idx, (r, f) in enumerate(zip(r0, f0)):
			
			# Get barycentric coordinates of the points obtained from stepping along the edge vectors
			fs_step = self.to_barycentric(r + eps*vectors) + 1e-20
			#print('fs_step:')
			#print(fs_step)
			
			# Get the directional derivatives by stepping a small amount along the edge vectors
			directional_derivatives = (fun(fs_step, *args, **kwargs) - fun(f, *args, **kwargs)) / eps
			
			# Get the gradient from the directional derivatives
			#     V      @  grad_f.T  =   D_f.T     <=>    Ax = b, x=np.linalg.solve(A, b)
			#(m-1 x m-1)   (m-1 x 1)    (m-1 x 1)
			gradients[point_idx] = np.linalg.solve(vectors, directional_derivatives.T).T
		
		# Return gradients
		return gradients
		
		
	def maximize(self, fun, f0, tol=1e-3, step=0.01, max_iter=10000, eps=1e-7, *args, **kwargs):
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
			Step
		max_iter: int
			Maximum number of iterations
		args: tuple
			Arguments to ´fun´
		kwargs: dict
			Keyword arguments to ´fun´
	
		Return
		------
		f1: array
			(Local) maximum resulting from the specified initial input
		'''
		print('f0:', ', '.join(map('{:.4f}'.format, f0.ravel())))
		
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
			if n_iter == max_iter:
				raise ValueError(f'[ERROR] Maximum number of iterations reached ({max_iter})')
			
			# Get gradient at the initial point
			gradient = self.get_gradient(fun, f0, eps=eps, *args, **kwargs)
		
			# Take a step along the direction of the gradient
			norm_gradient = np.sum(gradient**2)**0.5
			print('norm_gradient', f'{norm_gradient:.4f}')
			
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
			step = min(0.5*norm_gradient, step_orig)
			
			# Keep looping with smaller and smaller steps along the gradient so that 
			# a step does not go outside the simplex edges
			while True:
			
				# Get updated cartesian coordinates after taking a step along the gradient
				r1 = r0 + step*gradient
			
				# Convert cartesian coordinates to barycentric coordinates
				f1 = self.to_barycentric(r1)
				
				# If all barycentric coordinates are all positive then proceed
				if np.all(f1 >= 0.):
					break
				
				# If any of the barycentric coordinates are negative, then reduce the step taken
				else:
					step /= 2.
					
			# Update point
			f0 = f1
			print('f1:', ', '.join(map('{:.4f}'.format, f1.ravel())))
			
		print(f'[INFO] Maximum found after {n_iter} iterations')
		
		# Return the point having a gradient with length
		# below the threshold
		return f1
