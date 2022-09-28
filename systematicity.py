import numpy as np
import einops
import pandas as pd

def count_patterns(arr):
    # arr: an np array
    arr_new = einops.rearrange(arr, "(a b) -> a b", b=3) # convert to standard paradigm table

    coded_r_patterns = [pd.factorize(arr_new[i])[0] for i in range(arr_new.shape[0])]
    r_patterns = np.unique(np.array(coded_r_patterns), axis = 0).shape[0]

    coded_theta_patterns = [pd.factorize(arr_new.T[i])[0] for i in range(arr_new.T.shape[0])]
    theta_patterns = np.unique(np.array(coded_theta_patterns), axis = 0).shape[0]

    #print(f'r_patterns: {r_patterns}; theta_patterns: {theta_patterns}')
    return r_patterns + theta_patterns

arr = np.array([1,0,4,5,2,6,7,8,3])
print(count_patterns(arr))