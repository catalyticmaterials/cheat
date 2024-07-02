import scipy
import numpy as np
import itertools as it
from .utils import saferound
from scipy.stats import norm
from copy import deepcopy

def expected_improvement(X, X_known, gpr, xi=0.01):
    '''
    Adaptation of http://krasserm.github.io/2018/03/21/bayesian-optimization/

    Args:
        X: Points at which EI shall be computed (m x d).
        X_known: Sample locations (n x d).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.

    Returns:
        Expected improvements at points X.
    '''

    mu, std = gpr.predict(X, return_std=True)
    mu_known = gpr.predict(X_known)

    std = np.ravel(std)

    mu_known_opt = np.max(mu_known)

    temp = mu - mu_known_opt - xi
    Z = temp / std
    EI = temp * norm.cdf(Z) + std * norm.pdf(Z)

    EI[std == 0.0] = 0.0

    return EI

def random_comp(elements, max=1.0):
    """Get random dirichlet distributed composition from list of elements"""
    m = 1.01
    while m >= max:
        rnd = np.random.dirichlet(np.ones(len(elements)), 1)[0]
        m = np.max(rnd)
    rnd = saferound(rnd, decimals=2)
    comp = dict(zip(elements, rnd))
    return comp

def get_molar_fractions_around(f, step_size, total=1., eps=1e-10):
    """Get all molar fractions with the given step size around the given molar fraction"""
    fs = []
    n_elems = len(f)
    for pair, ids in zip(it.permutations(f, 2), it.permutations(range(n_elems), 2)):

        # Get molar fractions and their ids
        f0, f1 = pair
        id0, id1 = ids

        # Increment one molar fraction and decrement the other
        f0_new = f0 + (step_size - eps)
        f1_new = f1 - (step_size - eps)

        # Ignore if the new molar fractions are not between 0 and 1
        if f0_new <= total and f1_new >= 0.:
            # Make new molar fraction
            f_new = deepcopy(f)
            f_new[id0] = f0_new + eps
            f_new[id1] = f1_new - eps

            # Append to the output
            assert np.isclose(sum(f_new), 1.), "Molar fractions do not sum to unity : {}. Sum : {:.4f}".format(f_new,
                                                                                                               sum(
                                                                                                                   f_new))
            fs.append(saferound(f_new))

    return np.array(fs)

def optimize_molar_fraction(f, func, func_args=[], n_iter_max=1000, step_size=0.01):
    '''
    Return the molar fractions and their function value that locally
    maximizes the specified function starting from the molar fractions ´f´
    '''

    # Get the number of decimals to round molar fractions to
    # from the number of decimals of the molar fraction step size
    n_decimals = len(str(step_size).split('.')[1])

    # Get the function value of the specified molar fraction
    func_max = float(func(f.reshape(1, -1), *func_args))

    # Initiate the number of iterations to zero
    n_iter = 0

    while True:
        # Raise error if the number of iterations reaches the threshold
        if n_iter == n_iter_max:
            raise ValueError(f'No maximum has been found after {n_iter} iterations,\
							 so convergence is unlikely to happen.')

        # Get molar fractions around the current molar fraction in the given step size
        fs_around = get_molar_fractions_around(f, step_size=step_size)

        # Get function values
        func_vals = func(fs_around, *func_args)

        # Get the largest function value
        func_max_around = np.max(func_vals)

        # If the new function value is higher, then repeat for that molar fraction
        if func_max_around > func_max:

            # Get the index of largest function value
            idx_max = np.argmax(func_vals)

            # Get molar fraction of the maximum
            f = fs_around[idx_max]

            # Set the new function maximum
            func_max = func_max_around

        # If the function did now improve around the current molar fraction,
        # then the found molar fraction is a maximum and is returned
        else:
            # Round the molar fractions to the given number of decimals and
            # use a trick of adding 0. to make all -0. floats positive
            return np.around(f, decimals=n_decimals) + 0., func_max

        # Increment iteration count
        n_iter += 1

def get_local_maxima(fs, func, step_size=0.01, func_args=[]):
    # Initiate list containers
    fs_max = []
    funcs_max = []

    for f in fs:
        # Get the function value of the current molar fraction
        func_max = func(f.reshape(1, -1), *func_args)

        # Get molar fractions around the current molar fraction in the given step size
        fs_around = get_molar_fractions_around(f, step_size=step_size)

        # Get function values
        func_vals = func(fs_around, *func_args)

        # Get the largest function value
        func_max_around = np.max(func_vals)

        # If none of the neighbors have a higher function value,
        # then the current molar fractions is a local maximum
        if func_max_around < func_max:
            # Append the found maximum
            fs_max.append(f)
            funcs_max.append(func_max)

    return np.asarray(fs_max), np.asarray(funcs_max)

def get_molar_fractions(step_size, elements, total=1., return_number_of_molar_fractions=False):
    'Get all molar fractions with the given step size'

    n_elem = len(elements)

    interval = int(total / step_size)
    n_combs = scipy.special.comb(n_elem, interval, exact=True, repetition=True)

    if return_number_of_molar_fractions:
        return n_combs

    counts = np.zeros((n_combs, n_elem), dtype=int)

    for i, comb in enumerate(it.combinations_with_replacement(range(n_elem), interval)):
        for j in range(n_elem):
            counts[i, j] = np.count_nonzero(np.array(comb) == j)

    return counts * step_size

def opt_acquisition(X_known, gpr, elements, acq_func='EI', xi=0.01, n_iter_max=1000, n_random=1000, step_size=0.01):
    # Define the acquisition function
    if callable(acq_func):
        acquisition = acq_func
    elif acq_func == 'EI':
        acquisition = expected_improvement
    else:
        raise NotImplementedError(f"The acquisition function '{acq_func}' has not been implemented")

    random_samples = []
    for i in range(n_random):
        temp = random_comp(elements)
        random_samples.append(list(temp.values()))
    random_samples = np.array(random_samples)

    # Calculate the acquisition function for each sample
    acq_vals = acquisition(random_samples, X_known, gpr, xi)

    # Get the index of the largest acquisition function
    #ids_acq_sorted = np.argsort(acq_vals)[::-1]
    idx_max = np.argmax(acq_vals)

    # Get the molar fraction with the largest acquisition value
    f_max = random_samples[idx_max]
    #f_max = random_samples[ids_acq_sorted[0]]
    acq_max = np.max(acq_vals)

    # Optimize the aquisition function starting from this molar fraction
    n_iter = 0
    while True:
        if n_iter == n_iter_max:
            raise ValueError(f'No maximum has been found after {n_iter} iterations,\
							 so convergence is unlikely to happen.')

        # Get molar fractions around the found maximum in the given step size
        fs_around = get_molar_fractions_around(f_max, step_size=step_size)

        # Get acquisition values
        acq_vals = acquisition(fs_around, X_known, gpr, xi)

        # Get the largest acquisition value
        acq_max_around = np.max(acq_vals)

        # If the new aquisition value is higher, then repeat for that molar fraction
        if acq_max_around > acq_max and np.max(fs_around[np.argmax(acq_vals)]) <= 0.9:

            # Get the index of largest acquisition value
            idx_max = np.argmax(acq_vals)

            # Get molar fraction
            f_max = fs_around[idx_max]
            
            # Set the new acquisition maximum
            acq_max = acq_max_around

        # If the acquisition function did now improve around the molar fraction,
        # then return the found molar fraction
        else:
            return f_max

        n_iter += 1

