from ib import ib
import numpy as np
import pandas as pd

import argparse



# first number in tuple is D1/D2/D3, second is place/goal/source
DEICTIC_MAP = {
    0: (0, 1), 
    1: (1, 1),
    2: (2, 1),
    3: (0, 0), 
    4: (1, 0),
    5: (2, 0),
    6: (0, 2), 
    7: (1, 2),
    8: (2, 2)
}

DEICTIC_INDEX = {
    "D1_place": 0,
    "D2_place": 1,
    "D3_place": 2,
    "D1_goal": 3,
    "D2_goal": 4,
    "D3_goal": 5,
    "D1_source": 6,
    "D2_source": 7,
    "D3_source": 8,
}

FINN_WORDS = ["täällä", "siellä", "tuolla",
           "tänne", "sinne", "tuonne",
           "täältä", "sieltä", "tuolta"]

def get_prior_finnish():
    """
    Return the prior prob distribution over deictics, using Finnish
    """
    fin = pd.read_csv("~/deictic_adverbs/finnish_freqs.csv",
                    error_bad_lines=False)
    fin = fin.set_index("word").loc[FINN_WORDS].groupby("word").sum()
    fin["prob"] = fin["count"]/fin["count"].sum()
    df = pd.DataFrame(FINN_WORDS, columns=["word"]).set_index(["word"])
    df["prob"] = fin["prob"]
    return fin["prob"]

def get_prob_u_given_m(mu):
    u_m = np.zeros([len(DEICTIC_MAP), len(DEICTIC_MAP)])
    for i in range(len(DEICTIC_MAP)):
        distal, place = DEICTIC_MAP[i]
        for num in DEICTIC_MAP:
            costs = DEICTIC_MAP[num]
            u_m[i][num] = 1 * (mu ** (np.abs(costs[0] - distal) + np.abs(costs[1] - place)))
    return u_m/u_m.sum(axis=1)[:, None]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='get ib distribution.')
    parser.add_argument('--mu',  type=float, help='set mu', default=.1)
    parser.add_argument('--gamma', type=float, default=2)
    parser.add_argument('--Z', type =int, default=5)

    args = parser.parse_args()

    df = pd.DataFrame(ib(get_prior_finnish(), get_prob_u_given_m(args.mu), args.Z, args.gamma))
    df["deictic"] = list(DEICTIC_INDEX.keys())

    print(df)
