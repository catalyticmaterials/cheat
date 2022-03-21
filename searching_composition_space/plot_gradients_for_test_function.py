import sys
sys.path.append('..')
from scripts import get_molar_fractions
from scripts.maximize import SimplexOptimizer
import numpy as np
import matplotlib.pyplot as plt

#def fun(rs):
#	rs = np.asarray(rs)
#	if np.ndim(rs) == 1:
#		rs = rs.reshape(1, -1)
#	return -(rs[:, 0] - 3/4)**2 - (rs[:, 1] - 3**0.5/4)**2

def fun(fs):
	fs = np.asarray(fs)
	if np.ndim(fs) == 1:
		fs = fs.reshape(1, -1)
	return -(fs[:, 1] + (1/2)*fs[:, 2] - 1/2)**2 - ((3**0.5/2)*fs[:, 2] - (1/(2*3**0.5)))**2

opt = SimplexOptimizer(n_vertices=3)

fig, ax = plt.subplots()
fs = get_molar_fractions(step_size=0.05, n_elems=3)

gradients = opt.get_gradient(fun, fs)

activities = fun(fs)

rs = opt.to_cartesian(fs)
ax.scatter(*rs.T, c=activities, marker='o', s=150)
ax.quiver(*rs.T, *gradients.T)

ax.set_aspect('equal')
plt.show()
