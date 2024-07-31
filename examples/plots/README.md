#### Plotting alloy composition space

This folder showcases a few of plotting tools for use in a high-dimensional composition spaces.

As an example, we will plot the sampled alloy compositions from the [Bayesian optimization](../bayesian_optimization/).

*orthographic.py* plots the points in an orthographic projection. This can plot any number of dimensions but can be quite confusing to view. To alleviate this the samples are colored according to their Euclidian distance to the center of the composition space. Furthermore, size is used to display catalytic activity.

*simplexes.py* showcases the 3Dsimplex and 2Dsimplex classes which will plot alloy spaces of 4 and 3 elements, respectively. To plot the Bayesian optimization in AgIrPdPtRu composition space, we collapse selected elements onto the same vertex. In these plots, color is used to display catalytic activity.
