
import numpy as np

FINN_WORDS = ["täällä", "siellä", "tuolla",
              "tänne", "sinne", "tuonne",
              "täältä", "sieltä", "tuolta"]
    
def get_prior_finnish():
    """
    Return the prior prob distribution over universe, using Finnish
    """
    x = np.array([39, 16, 6, 18, 7, 1.6, 7, 2, .6])
    return x/np.sum(x)
