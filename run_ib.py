from collections import defaultdict
from get_lang_data import get_lang_dict
from get_prior import get_exp_prior
from ib import ib, mi
from enumerate_lexicons import get_random_lexicon

import argparse
import numpy as np
import pandas as pd
import random
import re

AREAS = ["europe", "asia", "americas", "africa"]

class RunIB:

    def __init__(self, mu, gamma, distal_levels):
        self.deictic_map = {}
        self.deictic_index = {}
        self.mu = mu
        self.gamma=gamma
        self.distal_levels = distal_levels
        c = 0
        for i in [("place", 1), ("goal", 0), ("source", 2)]:
            for j in range(distal_levels):
                self.deictic_map[c] = (j, i[1])
                self.deictic_index["D{}_{}".format(str(j + 1), i[0])] = c
                c += 1
        print (self.deictic_map, self.deictic_index)

    def distal2equalsdistal3(self, a):
        return all([all(a[self.deictic_index["D2_{}".format(str(i))]] ==
                        a[self.deictic_index["D3_{}".format(str(i))]])
                        for i in ["place", "goal", "source"]])

    def get_pgs_match(self, a):
        assert a.shape[0] == 3
        return (np.argmax(a[0]) == np.argmax(a[1]),
                np.argmax(a[1]) == np.argmax(a[2]),
                np.argmax(a[0]) == np.argmax(a[2]))

    def get_pgs_complexity(self, a):
        return len(set([self.get_pgs_match(np.stack([a[self.deictic_index[i]]for i in
                self.deictic_index if j in i])) for j in ["D1", "D2", "D3"]]))

    def get_complexity_of_paradigm(self, a):
        return "_".join([str(i) for i in [2 + self.distal2equalsdistal3(a),
                        self.get_pgs_complexity(a),
                        np.linalg.matrix_rank(a)]])

    def get_prob_u_given_m(self):
        u_m = np.zeros([len(self.deictic_map), len(self.deictic_map)])
        for i in self.deictic_map:
            distal, place = self.deictic_map[i]
            for num in self.deictic_map:
                costs = self.deictic_map[num]
                u_m[i][num] = 1 * (self.mu ** (np.abs(costs[0] - distal) + np.abs(costs[1] - place)))
        return u_m/u_m.sum(axis=1)[:, None]

    def get_mi_meaning_word(self, lexicon, prior):
        """Take lexicon of size U (num elements in world) x W (num words),
        multiply by prior [length=num meanings] to get I[M;W]"""
        return mi(prior[:,None] * lexicon)

    def get_mi_u_meaning(self, lexicon, prior):
        """Take lexicon of size U (num elements in world) x W (num words),
        get I[U;W] by summing over words"""
        num_u = lexicon.shape[0]
        lexicon_size = lexicon.shape[1]
        p_u_m = prior[:, None] * self.get_prob_u_given_m()
        p_u_w = np.zeros([num_u, lexicon_size]) # num_meanings by lexicon size
        word_max = np.argmax(lexicon, axis=1)
        for u in range(len(p_u_m)):
            for m in range(len(p_u_m[u])):
                p_u_w[u, word_max[m]] += p_u_m[u, m]
        return mi(p_u_w)

    def get_mi_for_all(self):
        num_meanings = self.distal_levels * 3
        lexicon_size_range = range(2, num_meanings + 1)
        prior = get_exp_prior(self.distal_levels)
        assert (len(prior) == num_meanings)
        dfs = []
        lexicons = []
        for lexicon_size in lexicon_size_range:
            all_lex = [get_random_lexicon(num_meanings, lexicon_size) for i in range(1000)]
            lexicons += [("simulated", l[1], "simulated") for l in all_lex]
            
            x = ib(prior, self.get_prob_u_given_m(), lexicon_size, self.gamma)
            optimal_for_size = np.zeros((x.shape[0], x.shape[1]))
            optimal_for_size[np.arange(x.shape[0]), np.argmax(x, axis=1)] = 1
            lexicons += [("optimal", optimal_for_size, "optimal")]

        # add real lexicons
        lexicons += self.get_real_langs(num_meanings)
        df = pd.DataFrame([{dm: l[1].argmax(axis=1)[dm_num]
                        for dm_num, dm in enumerate(self.deictic_index)}for l in lexicons])
        df["I[M;U]"] = [self.get_mi_u_meaning(l[1], prior) for l in lexicons]
        df["I[M;W]"] = [self.get_mi_meaning_word(l[1], prior) for l in lexicons]
        df["grammar_complexity"] = ["_".join(self.get_complexity_of_paradigm(l[1])) for l in lexicons]
        df["Language"] = [l[0] for l in lexicons]
        df["Area"] = [l[2] for l in lexicons]
        dfs += [df]
        return pd.concat(dfs).sort_values(["I[M;U]"], ascending=False)


    def get_real_langs(self, num_meanings):
        real_lexicon_arrays = []
        for area in AREAS:
            real_lexicon_arrays += self.get_real_arrays(area)
        return real_lexicon_arrays

    def get_real_arrays(self, area):
        """
        TODO: handle different ways of counting multiple occurences of words
            # in a single cell

        Take in an area, return a list of
        (lang, dict, area)
        where the lang_array is a typical lexicon of size M x W
        where M is number of meanings and W is number of words in the lexicon.
        """
        d = get_lang_dict(area)
        lang_arrays = []
        permissible_distals = ["D{}".format(str(i)) for i in range(1, self.distal_levels + 1)]
        for lang in d:
            # filter to just distal levels we recognize
            dd = {i: d[lang][i] for i in d[lang] if i[0] in permissible_distals}
            a = ["_".join(sorted(word)) for word in dd.values()]
            keys = [k[0] + "_" + k[2].lower() for k in list(dd.keys())]
            setx = list(set(a))
            # make a dict with unique ids for each word
            unique_word_d = {keys[k]: setx.index(a[k]) for k in range(len(a))}
            
            # fill in missing distal levels by passing previous distal level to next
            for distal_num in range(1, self.distal_levels + 1):
                for pgs in ["place", "goal", "source"]:
                    curkey = "D{}_{}".format(str(distal_num), pgs)
                    if curkey not in unique_word_d:
                        prevkey = "D{}_{}".format(str(distal_num - 1), pgs)
                        unique_word_d[curkey] = unique_word_d[prevkey]

            lang_array = np.zeros([len(self.deictic_index), len(set(unique_word_d.values()))])
            for spot in unique_word_d:
                lang_array[self.deictic_index[spot]][unique_word_d[spot]] = 1
            lang_arrays += [(lang, lang_array, area)]

        return lang_arrays


    def get_prob_u_given_m_mini(self, mu, mininum):
        u_m = np.zeros([mininum, mininum])
        for i in range(mininum):
            for num in range(mininum):
                u_m[i][num] = 1 * (mu ** (np.abs(num - i)))
        return u_m/u_m.sum(axis=1)[:, None]

    def get_optimal_lexicon_mini(self, mu, minimum, gamma):
        """For one of P/G/S, return the optimal distribution of W|M.
        When gamma is large, this is deterministic in the expected way."""
        x = self.get_prob_u_given_m_mini(mu, minimum)
        p = get_exp_prior(3)[:3]
        return ib(p, x, minimum, gamma)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='get ib distribution.')
    parser.add_argument('--mu',  type=float, help='set mu', default=.1)
    parser.add_argument('--gamma', type=float, default=2)
    parser.add_argument('--outfile', type=str, default="mi_test_1.csv")
    parser.add_argument('--distal', type=float, default=6)

    args = parser.parse_args()
    RunIB(args.mu, args.gamma, args.distal).get_mi_for_all().to_csv(args.outfile)
