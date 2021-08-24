
import numpy as np
import scipy.optimize

FINN_WORDS = ["täällä", "siellä", "tuolla",
              "tänne", "sinne", "tuonne",
              "täältä", "sieltä", "tuolta"]

FINN_COUNTS = {"place":  [39, 16, 6],
"goal": [18, 7, 1.6],
"source": [7, 2, .6],
"unif": [1, 1, 1]}

def get_exp_fit(prior, distal_levels):
    def func(x, a, b):
        return a*np.exp(-b*x)
    a, b = scipy.optimize.curve_fit(func, range(len(prior)), prior)[0]
    dist = func(range(distal_levels), a, b)
    return dist

def get_exp_prior(distal_levels, prior_spec):
    x = np.concatenate([get_exp_fit(FINN_COUNTS[i], distal_levels) for i in
                        prior_spec])
    return x/np.sum(x)

def exp_fit_place(distal_levels, loc):
    x =  get_exp_fit(FINN_COUNTS[loc], distal_levels)
    return x/np.sum(x)
