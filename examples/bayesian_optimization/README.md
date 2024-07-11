#### Bayesian optimization of solid-solution alloy composition

This folder contains an example of how to iteratively employ the surrogate surface simulation in a Bayesian optimization protocol to find the most catalyst composition given a specified activity expression. 

Running *bayes_opt.py* will initially sample a few randomly selected alloy compositions as input to a Gaussian Process Regression algorithm to fit a surrogate function. In conjunction with the Expected Improvement acquisition function the next alloy composition to sample as a surrogate surface is selected. This protocol is described in detail in [Pedersen et al. *Angew. Chem.* 2021](https://doi.org/10.1002/ange.202108116).
