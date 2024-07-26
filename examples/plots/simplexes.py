import numpy as np
import matplotlib.pyplot as plt
from cheatools.plot import simplex2D, simplex3D

# load in Bayesian search results
search_results = np.genfromtxt('../bayesian_optimization/sampled.csv', skip_header=1, delimiter=',',dtype=str)
compositions, activities = search_results[:,:-1].astype(float), np.array([int(a[:-1]) for a in search_results[:,-1]])

# a 3D simplex can only depict 4 elements so we squash Ir and Ru together
compositions[:,1] = compositions[:,1] + compositions[:,4]
compositions = np.delete(compositions,4,1)

# initialize 3D simplex class and plot
simplex = simplex3D()
fig = simplex.plot(labels=['Ag','IrRu','Pd','Pt'])
ax = fig.get_axes()[0]

coordinates = simplex.comps2coords(compositions) # transform compositions to coordinates
ax.scatter(*coordinates, c=activities, s=50, vmin=0, vmax=250) # plot scatter colored according to activity 

# adjust and save figure
fig.set_size_inches(10, 8)
ax.view_init(elev=10, azim=0)
ax.set_position([-0.15, -0.15, 1.3, 1.3])
fig.savefig('3Dsimplex.png',dpi=200)
plt.close()

# a 2D simplex can only depict 3 elements so now Ir, Ru, and Pt are collapsed to a single vertex 
compositions[:,1] = compositions[:,1] + compositions[:,3]
compositions = np.delete(compositions,3,1)

# initialize 3D simplex class and plot
simplex = simplex2D()
fig = simplex.plot(labels=['Ag','IrRuPt','Pd'], show_edges=True, show_cornerlabels=True, show_axlabels=True, show_ticklabels=True)
ax = fig.get_axes()[0]

coordinates = simplex.comps2coords(compositions) # transform compositions to coordinates
ax.scatter(*coordinates, c=activities, s=50, vmin=0, vmax=250) # plot scatter colored according to activity

# save figure
ax.set_position([0.2, 0.2, 0.6, 0.6])
fig.savefig('2Dsimplex.png',dpi=200)
plt.close()

