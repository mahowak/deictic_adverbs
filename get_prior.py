import numpy as np
import scipy.optimize

"""
numbers for FINN_COUNTS from finnish_freqs.csv
FINN_WORDS = ["täällä", "siellä", "tuolla",
                "tänne", "sinne", "tuonne",
                "täältä", "sieltä", "tuolta"]
"""

FINN_COUNTS = {"place":  [232946, 94887, 38402],
            "goal": [109576, 42923, 10006],
            "source": [43016, 17587, 3850],
            "unif": [1, 1, 1]}

def get_exp_fit(prior, distal_levels):
    """Take exponential fit from Finnish to get distribution over other levels"""
    def func(x, a, b):
        return a*np.exp(-b*x)
    a, b = scipy.optimize.curve_fit(func, range(len(prior)), prior)[0]
    dist = func(range(distal_levels), a, b)
    return dist

def get_exp_prior(distal_levels, prior_spec):
    """Get the prior distribution over distal levels"""
    x = np.concatenate([get_exp_fit(FINN_COUNTS[i], distal_levels) for i in
                        prior_spec])
    return x/np.sum(x)