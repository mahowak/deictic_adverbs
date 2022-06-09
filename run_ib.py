from get_lang_data import get_lang_dict
from get_prior import get_exp_prior
from ib import information_plane
from enumerate_lexicons import enumerate_possible_lexicons
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import euclidean_distances

from scipy.spatial import ConvexHull
from scipy.stats import linregress

import argparse
import itertools
import numpy as np
import os
import pandas as pd

AREAS = ["europe", "asia", "americas", "africa", "oceania"]

def getEquidistantPoints(p1, p2, parts):
    return zip(np.linspace(p1[0], p2[0], parts+1),
               np.linspace(p1[1], p2[1], parts+1))

def classify_lang(lang):
    if lang not in ["simulated", "optimal"]:
        return "real"
    return lang
        

class RunIB:

    def __init__(self,
    mu,
    gamma,
    distal_levels,
    pgs_dists,
    prior_spec=["place", "goal", "source"]):
    #prior_spec = ["goal", "place", "source"]):
        self.deictic_map = {}
        self.deictic_index = {}
        self.mu = mu
        self.gamma=gamma
        self.distal_levels = distal_levels
        self.prior_spec = prior_spec
        self.prior = get_exp_prior(self.distal_levels, prior_spec)  # p(x)
        self.pgs_dists = pgs_dists
        c = 0
        for i in [("place", self.pgs_dists[0]),
        ("goal", self.pgs_dists[1]),
        ("source", self.pgs_dists[2])]:
            for j in range(distal_levels):
                self.deictic_map[c] = (j, i[1])
                self.deictic_index["D{}_{}".format(str(j + 1), i[0])] = c
                c += 1
        self.prob_u_given_m = self.get_prob_u_given_m()

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

    def get_mi_for_all(self, sim_lex_dict={}, outfile="default"):
        num_meanings = self.distal_levels * 3
        lexicon_size_range = range(2, int(num_meanings) + 1)
        assert (len(self.prior) == num_meanings)
        dfs = []
        lexicons = []
        for lexicon_size in lexicon_size_range:
            all_lex = sim_lex_dict[lexicon_size]
            lexicons += [("simulated", l[1], "simulated") for l in all_lex]

        # add real lexicons
        lexicons += self.get_real_langs(num_meanings) # p(z|x)

        # turn lexicon into df
        df = pd.DataFrame([{dm: l[1].argmax(axis=1)[dm_num]
                        for dm_num, dm in enumerate(self.deictic_index)}for l in lexicons])

        information_plane_list = [information_plane(self.prior, self.prob_u_given_m, l[1]) for l in lexicons]                        
        df["I[U;W]"] = [l[0] for l in information_plane_list]
        df["I[M;W]"] = [l[1] for l in information_plane_list]
        df["MI_Objective"] = df["I[M;W]"] - self.gamma * df["I[U;W]"] # complexity - beta * informativity
        df["grammar_complexity"] = ["_".join(self.get_complexity_of_paradigm(l[1])) for l in lexicons]
        df["Language"] = [l[0] for l in lexicons]
        df["Area"] = [l[2] for l in lexicons]
        df["LangCategory"] = [classify_lang(lang) for lang in df["Language"]]
        dfs += [df]
        x =  pd.concat(dfs).sort_values(["I[U;W]"], ascending=False)
        x.to_csv(outfile + '_mu_' + str(args.mu) + '_gamma_' + str(args.gamma) + "_" + "_".join([str(pgs) for pgs in self.pgs_dists]) + "_" + ".csv")
        
        mi_obj_simulated = df.groupby("LangCategory").mean()["MI_Objective"]["simulated"]
        mi_obj_real = df.groupby("LangCategory").mean()["MI_Objective"]["real"]
        
        mi_obj_simulated_std = df.groupby("LangCategory").std()[
            "MI_Objective"]["simulated"]
        mi_obj_real_std = df.groupby("LangCategory").std()["MI_Objective"]["real"]

        sim_only = df.loc[df["LangCategory"] == "simulated"]
        real_only = df.loc[df["LangCategory"] == "real"]
        sim_only_plane = np.array(df.loc[df["LangCategory"] == "simulated", ["I[U;W]", "I[M;W]"]])
        real_only_plane = np.array(df.loc[df["LangCategory"] == "real", ["I[U;W]", "I[M;W]"]])

        # for simulated
        X = np.array(sim_only["I[U;W]"]).reshape(-1, 1)
        y = np.array(sim_only["I[M;W]"])
        model = LinearRegression()
        model.fit(X, y)
        predictions_for_real = model.predict(
            np.array(real_only["I[U;W]"]).reshape(-1, 1))
        residuals_sim = np.array(real_only["I[M;W]"]) - predictions_for_real

        vertices = ConvexHull(sim_only_plane).vertices
        pv = sim_only_plane[vertices] # this is the hull
        resid = pv[:, 1] - (linregress(sim_only_plane).intercept +
                            pv[:, 0] * linregress(sim_only_plane).slope)
        vertices = vertices[resid < 0]  # just the bottom part of hull

        # fill in between the points
        verts = []
        for i, j in zip(pv, np.array(list(pv[1:, :]) + list([pv[0, :]]))):
            if i[0] < j[0]:
                verts += list(getEquidistantPoints(i, j, 100))
        verts = np.array(verts)

        dists = euclidean_distances(real_only_plane, verts).min(axis=1)
        dist_to_hull = np.sum(dists)
        dist_to_hull_2 = np.sum(dists * dists)

        df_grid = pd.DataFrame({"place": [self.pgs_dists[0]],
        "goal": [self.pgs_dists[1]],
        "source": [self.pgs_dists[2]],
        "real": [mi_obj_real],
        "simulated": [mi_obj_simulated],
        "real_sd": [mi_obj_real_std],
        "simulated_sd": [mi_obj_simulated_std],
        "real_minus_sim_resid": [np.sum(residuals_sim)],
        "dist_to_hull": [dist_to_hull],
        "dist_to_hull_sq": [dist_to_hull_2],
        "prior_spec": ["_".join(self.prior_spec)],
        "mu": [str(self.mu)],
        "gamma": [str(self.gamma)]
        })

        df_grid.to_csv(outfile + "_gridsearch.csv", mode='a',
                  header=not os.path.exists(outfile + "_gridsearch.csv"))

        return x


    def get_real_langs(self, num_meanings):
        real_lexicon_arrays = []
        for area in AREAS:
            real_lexicon_arrays += self.get_real_arrays(area)
        return real_lexicon_arrays

    def get_real_arrays(self, area):
        """
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
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='get ib distribution.')
    parser.add_argument('--mu',  type=float, help='set mu', default=.1)
    parser.add_argument('--gamma', type=float, default=2)
    parser.add_argument('--outfile', type=str, default="mi_test_1.csv")
    parser.add_argument('--distal', type=int, default=3)
    parser.add_argument('--grid_search', action='store_true')
    parser.add_argument('--prior_search', action='store_true')
    parser.add_argument('--mu_search', action='store_true')
    parser.add_argument('--total_search', action='store_true')
    parser.add_argument('--total_search_mini', action='store_true') # search within a smaller range of parameters
    parser.add_argument('--pgs', help='the relative location for PLACE / GOAL / SOURCE (e.g. 0, -1, 1)', 
        type=lambda s: [float(item) for item in s.split(',')], default = '0, -0.789, 1.316')

    args = parser.parse_args()
    
    num_meanings = args.distal * 3
    lexicon_size_range = range(2, num_meanings + 1)

    sim_lex_dict = {lexicon_size: [lexicon for lexicon in enumerate_possible_lexicons(num_meanings, lexicon_size)] for 
        lexicon_size in lexicon_size_range}
    
    if args.grid_search:
        for i in np.append(np.linspace(-10, 10, 8), np.array(0)):
            for j in np.append(np.linspace(-10, 10, 8), np.array(0)):
                if np.abs(i) > .5 and np.abs(j) > .5:
                    RunIB(args.mu, args.gamma, args.distal,
                        [0, i, j]).get_mi_for_all(sim_lex_dict=sim_lex_dict,
                                                outfile = args.outfile + str(args.mu) + str(args.gamma))

    elif args.prior_search:
        for perm in (list(itertools.permutations(["place", "goal", "source"])) + 
        [["unif", "unif", "unif"], ["place", "place", "place"]]):
            RunIB(args.mu, args.gamma, args.distal, args.pgs,
                  prior_spec=perm).get_mi_for_all(sim_lex_dict=sim_lex_dict,
                                            outfile=args.outfile)
    elif args.mu_search:
        for mu in np.append(np.linspace(0.05, 0.99, 19), np.array(0.99)):
            RunIB(mu, args.gamma, args.distal, args.pgs, prior_spec =["place", "goal", "source"] ).get_mi_for_all(get_opt=args.get_opt,
                                            sim_lex_dict=sim_lex_dict,
                                            outfile=args.outfile)


    elif args.total_search:
        for perm in (list(itertools.permutations(["place", "goal", "source"])) +
                     [["unif", "unif", "unif"], ["place", "place", "place"]]):
            for mu in [.1, .2, .3]:
                for i in np.append(np.linspace(-5, 5, 20), np.array(0)):
                    for j in np.append(np.linspace(-5, 5, 20), np.array(0)):
                        pgs = [0, i, j]
                        RunIB(mu, args.gamma, args.distal,
                            pgs,
                            prior_spec=perm).get_mi_for_all(sim_lex_dict=sim_lex_dict,
                                                            outfile=args.outfile)
    elif args.total_search_mini:
        for perm in (list(itertools.permutations(["place", "goal", "source"])) +
                     [["unif", "unif", "unif"]]):
            for mu in [.2, .3]:
                for i in [0.8, 0.9, 1, 1.1, 1.2, 1.3]:
                    for j in [-0.8, -0.9, -1, -1.1, -1.2, -1.3]:
                        RunIB(mu, args.gamma, args.distal,
                            [0, i, j],
                            prior_spec=perm).get_mi_for_all(sim_lex_dict=sim_lex_dict,
                                                            outfile=args.outfile)

    else:
        RunIB(args.mu, args.gamma, args.distal, args.pgs).get_mi_for_all(sim_lex_dict=sim_lex_dict,
            outfile = args.outfile)