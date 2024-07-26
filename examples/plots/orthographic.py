import numpy as np
import matplotlib.pyplot as plt
from cheatools.plot import orthographic_projection

# load in sampled compositions and activities
elements = ['Ag','Ir','Pd','Pt','Ru']
search_results = np.genfromtxt('../bayesian_optimization/sampled.csv', skip_header=1, delimiter=',', dtype=str)
compositions, activities = search_results[:,:-1].astype(float), np.array([int(a[:-1]) for a in search_results[:,-1]])

# calculate the Euclidian distance from equimolar composition for each sample
distances = []
for comp in compositions:
    diff = np.array(comp) - np.ones(len(elements))/len(elements)
    dist = np.sqrt(np.sum(diff**2))
    distances.append(dist)

# initialize orthographic projection and plot figure
ort_prj = orthographic_projection(elements)
fig = ort_prj.plot()

coordinates = ort_prj.comps2coords(compositions) # transform compositions to coordinates
normalized_activities = (activities / np.max(activities)) * (250 - 5) + 5 # normalize activities to scatter point sizes between 5 and 250

# plot scatter points sized by activity and colored according to distance, then save figure
plt.scatter(*coordinates, s=normalized_activities, c=distances, cmap='viridis', vmin=0, vmax=0.9, alpha=0.6)
fig.savefig('orthographic.png',dpi=200,bbox_inches='tight')
