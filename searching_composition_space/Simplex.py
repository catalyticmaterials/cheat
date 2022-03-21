import numpy as np
from math import factorial

class Simplex():
	
	def __init__(self, n_vertices):
		self.n_vertices = n_vertices
		self.n_dim = n_vertices - 1
		self.get_vertices()
	
	def get_vertices(self):
		'Return cartesian coordinates of vertices'
		# Initiate array of vertex coordinates
		vertices = np.zeros((self.n_vertices, self.n_vertices-1))
		
		# Iterate through vertices, putting the first vertex at the origo
		for idx in range(1, self.n_vertices):
		
			# Get coordinate of the existing dimensions as the 
			# mean of the existing vertices
			vertices[idx] = np.mean(vertices[:idx], axis=0)
		
			# Get the coordinate of the new dimension by ensuring it has a unit 
			# distance to the first vertex at the origin 
			vertices[idx][idx-1] = (1 - np.sum(vertices[idx][:-1]**2))**0.5
		
		self.vertices = vertices
		return vertices
	
	def get_volume(self):
		'Return volume of simplex'
		return abs(np.linalg.det(self.vertices[1:].T)) / factorial(self.n_dim)
