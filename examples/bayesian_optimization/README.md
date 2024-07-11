#### Bayesian optimization of solid-solution alloy composition

This folder contains an example of how to iteratively employ the surrogate surface simulation in a Bayesian optimization protocol to find the most catalyst composition given a specified activity expression. 

Running *bayes_opt.py* will initially sample a few randomly selected alloy compositions as input to a Gaussian Process Regression (GPR) algorithm to fit a surrogate function. In conjunction with the Expected Improvement (EI) acquisition function the next alloy composition to sample as a surrogate surface is selected. This protocol is described in detail in [Pedersen et al. *Angew. Chem.* 2021](https://doi.org/10.1002/ange.202108116).

There is a few adjustable parameters such as the $\xi$-parameter of the EI acquisition function and the kernel for the GPR. Should you not wish to sample alloys too close to the vertices of composition space, it is possible to impose a limit on fractions of single elements in randomly chosen compositions by editing the default limit in random_comp function in *cheatools/bayesian.py*:
```python
random_comp(elements, max=1.0)
```
