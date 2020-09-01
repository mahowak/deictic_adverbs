from enumerate_lexicons import enumerate_possible_lexicons
from ib import ib, mi

import argparse
import numpy as np
import pandas as pd
import random

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


def distal2equalsdistal3(a):
    return all([all(a[DEICTIC_INDEX["D2_{}".format(str(i))]] ==
                    a[DEICTIC_INDEX["D3_{}".format(str(i))]])
                    for i in ["place", "goal", "source"]])

def get_pgs_match(a):
    assert a.shape[0] == 3
    return (np.argmax(a[0]) == np.argmax(a[1]),
            np.argmax(a[1]) == np.argmax(a[2]),
            np.argmax(a[0]) == np.argmax(a[2]))

def get_pgs_complexity(a):
    return len(set([get_pgs_match(np.stack([a[DEICTIC_INDEX[i]]for i in
            DEICTIC_INDEX if j in i])) for j in ["D1", "D2", "D3"]]))

def get_complexity_of_paradigm(a):
    return "_".join([str(i) for i in [2 + distal2equalsdistal3(a),
                    get_pgs_complexity(a),
                    np.linalg.matrix_rank(a)]])

def get_prior_finnish():
    """
    Return the prior prob distribution over universe, using Finnish
    """
    fin = pd.read_csv("finnish_freqs.csv",
                    error_bad_lines=False)
    fin = fin.set_index("word").loc[FINN_WORDS].groupby("word").sum()
    fin["prob"] = fin["count"]/fin["count"].sum()
    df = pd.DataFrame(FINN_WORDS, columns=["word"]).set_index(["word"])
    df["prob"] = fin["prob"]
    return df["prob"]

def get_prob_u_given_m(mu):
    u_m = np.zeros([len(DEICTIC_MAP), len(DEICTIC_MAP)])
    for i in DEICTIC_MAP:
        distal, place = DEICTIC_MAP[i]
        for num in DEICTIC_MAP:
            costs = DEICTIC_MAP[num]
            u_m[i][num] = 1 * (mu ** (np.abs(costs[0] - distal) + np.abs(costs[1] - place)))
    return u_m/u_m.sum(axis=1)[:, None]

def get_mi_meaning_word(lexicon, prior):
    """Take lexicon of size U (num elements in world) x W (num words),
    multiply by prior [length=num meanings] to get I[M;W]"""
    return mi(prior[:,None] * lexicon)

def get_mi_u_meaning(lexicon, mu, prior):
    """Take lexicon of size U (num elements in world) x W (num words),
    get I[U;W] by summing over words"""
    num_u = lexicon.shape[0]
    lexicon_size = lexicon.shape[1]
    p_u_given_m = prior[:, None] * get_prob_u_given_m(mu)
    p_u_given_w = np.zeros([num_u, lexicon_size])
    word_max = np.argmax(lexicon, axis=1)
    for u in range(len(p_u_given_m)):
        for m in range(len(p_u_given_m[u])):
            p_u_given_w[u, word_max[m]] += p_u_given_m[u, m]
    return mi(p_u_given_w)

def get_mi_for_all(lexicon_size_range=range(2, 10), mu=.1, num_meanings=9, gamma=2):
    prior = get_prior_finnish()
    assert (len(prior) == num_meanings)
    dfs = []
    lexicons = []
    for lexicon_size in lexicon_size_range:
        print(lexicon_size)
        all_lex = list(enumerate_possible_lexicons(num_meanings, lexicon_size))
        if len(all_lex) > 1000:
            all_lex = random.choices(all_lex, k=1000)
        lexicons += [("simulated", l) for l in all_lex]
        
        x = ib(prior, get_prob_u_given_m(mu), lexicon_size, gamma)
        optimal_for_size = np.zeros((x.shape[0], x.shape[1]))
        optimal_for_size[np.arange(x.shape[0]), np.argmax(x, axis=1)] = 1
        lexicons += [("optimal", optimal_for_size)]

    # add real lexicons
    lexicons += get_real_langs()

    df = pd.DataFrame([{dm: l[1].argmax(axis=1)[dm_num]
                    for dm_num, dm in enumerate(DEICTIC_INDEX)}for l in lexicons])
    df["I[M;U]"] = [get_mi_u_meaning(l[1], mu, prior) for l in lexicons]
    df["I[M;W]"] = [get_mi_meaning_word(l[1], prior) for l in lexicons]
    df["grammar_complexity"] = ["_".join(get_complexity_of_paradigm(l[1])) for l in lexicons]
    df["Language"] = [l[0] for l in lexicons]
    dfs += [df]
    return pd.concat(dfs).sort_values(["I[M;U]"], ascending=False)


def get_real_langs(num_meanings=9):
    df = pd.read_csv("processed_datasheets/europe.csv")
    real_lexicon_arrays = []
    num_meanings = 9
    for lang in set(df.Language):
        langsubset = df.loc[df.Language == lang]
        real_lexicon = np.zeros([num_meanings, max(langsubset.uid) + 1])
        real_lexicon[langsubset.deictic_index, langsubset.uid] = 1
        real_lexicon_arrays += [(lang, real_lexicon)]
    return real_lexicon_arrays


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='get ib distribution.')
    parser.add_argument('--mu',  type=float, help='set mu', default=.1)
    parser.add_argument('--gamma', type=float, default=2)
    parser.add_argument('--Z', type =int, default=5)
    parser.add_argument('--outfile', type=str, default="mi_test_1.csv")
    args = parser.parse_args()

    #df = pd.DataFrame(ib(get_prior_finnish(), get_prob_u_given_m(args.mu), args.Z, args.gamma))
    #df["deictic"] = list(DEICTIC_INDEX.keys())
    get_mi_for_all().to_csv(args.outfile)
    #print(df)
