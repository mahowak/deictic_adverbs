from collections import defaultdict
from get_lang_data import get_lang_dict
from get_prior import get_exp_prior, exp_fit_place
from ib import ib, mi, information_plane
from enumerate_lexicons import get_random_lexicon
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import euclidean_distances

from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.stats import linregress

import argparse
import matplotlib.pyplot as plt
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

    def __init__(self, mu, gamma, distal_levels, pgs_dists=[0, 1.3, -1.7]):
        self.deictic_map = {}
        self.deictic_index = {}
        self.mu = mu
        self.gamma=gamma
        self.distal_levels = distal_levels
        self.prior = get_exp_prior(self.distal_levels)  # p(x)
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

    def get_optimal_lexicon(self, lexicon_size):
        return ib(self.prior, self.prob_u_given_m, lexicon_size,
                  self.gamma, num_iter=100, outer_iter=100)

    def get_mi_for_all(self, get_opt=True, sim_lex_dict={}, outfile="default"):
        num_meanings = self.distal_levels * 3
        lexicon_size_range = range(2, num_meanings + 1)
        assert (len(self.prior) == num_meanings)
        dfs = []
        lexicons = []
        for lexicon_size in lexicon_size_range:
            all_lex = sim_lex_dict[lexicon_size]
            lexicons += [("simulated", l[1], "simulated") for l in all_lex]
            if get_opt:
                lexicons += [("optimal", self.get_optimal_lexicon(lexicon_size),
                         "optimal")]

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
        x.to_csv(outfile + "_" + "_".join([str(pgs) for pgs in self.pgs_dists]) + "_" + ".csv")


        
        # GRID SEARCH PART 
        standardized = False
        if standardized:
            df["I[U;W]"] = (df["I[U;W]"] - df["I[U;W]"].mean())/df["I[U;W]"].std()
            df["I[M;W]"] = (df["I[M;W]"] - df["I[M;W]"].mean()) / \
                df["I[M;W]"].std()

        mi_obj_simulated = df.groupby("LangCategory").mean()["MI_Objective"]["simulated"]
        mi_obj_real = df.groupby("LangCategory").mean()["MI_Objective"]["real"]
        if get_opt:
            mi_obj_optimal = df.groupby("LangCategory").mean()[
            "MI_Objective"]["optimal"]
        else:
            mi_obj_optimal = 0
        mi_obj_simulated_std = df.groupby("LangCategory").std()[
            "MI_Objective"]["simulated"]
        mi_obj_real_std = df.groupby("LangCategory").std()["MI_Objective"]["real"]
        if get_opt:
            mi_obj_optimal_std = df.groupby("LangCategory").std()[
            "MI_Objective"]["optimal"]
        else:
            mi_obj_optimal_std = 0

        sim_only = df.loc[df["LangCategory"] == "simulated"]
        real_only = df.loc[df["LangCategory"] == "real"]
        sim_only_plane = np.array(df.loc[df["LangCategory"] == "simulated", ["I[U;W]", "I[M;W]"]])
        real_only_plane = np.array(df.loc[df["LangCategory"] == "real", ["I[U;W]", "I[M;W]"]])

        if get_opt:
            optimal_only = df.loc[df["LangCategory"] == "optimal"]
            X = np.array(optimal_only["I[U;W]"]).reshape(-1, 1)
            y = np.array(optimal_only["I[M;W]"])
            model = LinearRegression()
            model.fit(X, y)
            predictions_for_real = model.predict(
                                np.array(real_only["I[U;W]"]).reshape(-1, 1))
            residuals_optimal = np.array(real_only["I[M;W]"]) - predictions_for_real
        else:
            residuals_optimal = 0

        # for simulated
        X = np.array(sim_only["I[U;W]"]).reshape(-1, 1)
        y = np.array(sim_only["I[M;W]"])
        model = LinearRegression()
        model.fit(X, y)
        predictions_for_real = model.predict(
            np.array(real_only["I[U;W]"]).reshape(-1, 1))
        residuals_sim = np.array(real_only["I[M;W]"]) - predictions_for_real


        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis

        plt.plot(sim_only_plane[:, 0], sim_only_plane[:, 1], 'go')
        plt.plot(real_only_plane[:, 0], real_only_plane[:, 1], 'wo')

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


        plt.plot(verts[:, 0], verts[:, 1], 'r--', lw=2)

        fig.savefig(outfile + "_plot.png")   # save the figure to file
        plt.close(fig)    # close the figure window

        df_grid = pd.DataFrame({"place": [self.pgs_dists[0]],
        "goal": [self.pgs_dists[1]],
        "source": [self.pgs_dists[2]],
        "real": [mi_obj_real],
        "simulated": [mi_obj_simulated],
        "optimal": [mi_obj_optimal],
        "real_sd": [mi_obj_real_std],
        "simulated_sd": [mi_obj_simulated_std],
        "optimal_sd": [mi_obj_optimal_std],
        "real_minus_opt_resid": [np.sum(residuals_optimal)],
        "real_minus_opt_resid_sq": [np.sum(residuals_optimal * residuals_optimal)],
        "real_minus_sim_resid": [np.sum(residuals_sim)],
        "real_minus_sim_resid_sq": [np.sum(residuals_sim * residuals_sim)],
        "dist_to_hull": [dist_to_hull],
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

def print_optimal_lexicons_for_ggplot(df):
    newds = []
    for i in range(df.shape[0]):
        row = df.iloc[i]
        for item_num, item in enumerate(row["lexicon"]):
            for word_num, word in enumerate(item):
                distal, loc = row["index"][item_num].split("_")
                newd = {"mu": row["mu"],
                        "gamma": row["gamma"],
                        "loc": loc,
                        "total_num_words": row["num_words"],
                        "total_distal": row["distal"],
                        "distal": distal,
                        "word": word_num + 1,
                        "value": word}
                newds += [newd]

    pd.DataFrame(newds).to_csv("optimal_lexicons_for_plot.csv")

def get_optimal_lexicons():
    d = []
    for mu in [.1, .2]:
        for gamma in [2, 10]:
            for distal_levels in [3, 4, 5]:
                runib = RunIB(mu, gamma, distal_levels)
                for num_words in [3, 4, 5, 6, 7, 8, 9]:
                    if num_words <= distal_levels * 3:
                        z_given_x = runib.get_optimal_lexicon(num_words)
                        mi_xz, mi_yz = information_plane(
                            runib.prior, runib.prob_u_given_m, z_given_x)
                        z_given_x = np.round(z_given_x, 2)
                        d += [{"mu": mu,
                                "gamma": gamma,
                                "distal": distal_levels,
                                "num_words": num_words,
                                "mi_xz": mi_xz,
                                "mi_yz": mi_yz,
                                "u_given_m": runib.prob_u_given_m,
                                "prior": runib.prior,
                                "map": runib.deictic_map,
                                "index": {runib.deictic_index[i]: i for i in runib.deictic_index},
                                "lexicon": z_given_x[:, z_given_x.argmax(axis=0).argsort()]}]
    df = pd.DataFrame(d)
    df.to_pickle("optimal_lexicons.pkl")
    print_optimal_lexicons_for_ggplot(df)
    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='get ib distribution.')
    parser.add_argument('--mu',  type=float, help='set mu', default=.1)
    parser.add_argument('--gamma', type=float, default=2)
    parser.add_argument('--outfile', type=str, default="mi_test_1.csv")
    parser.add_argument('--distal', type=float, default=6)
    parser.add_argument('--get_opt', action='store_true')
    parser.add_argument('--grid_search', action='store_true')

    args = parser.parse_args()
    
    num_meanings = args.distal * 3
    lexicon_size_range = range(2, num_meanings + 1)

    # pre-populate and fix the simulated lexicons
    sim_lex_dict = {lexicon_size: [get_random_lexicon(
        num_meanings, lexicon_size, seed=i) for i in range(10000)] for 
        lexicon_size in lexicon_size_range}
    
    if args.grid_search:
        for i in np.append(np.linspace(-5, 5, 10), np.array(0)):
            for j in np.append(np.linspace(-5, 5, 10), np.array(0)):
                RunIB(args.mu, args.gamma, args.distal,
                    [0, i, j]).get_mi_for_all(get_opt=args.get_opt,
                                              sim_lex_dict=sim_lex_dict,
                                              outfile = args.outfile)
    else:
        RunIB(args.mu, args.gamma, args.distal).get_mi_for_all(
            get_opt=args.get_opt, sim_lex_dict=sim_lex_dict,
            outfile = args.outfile)

    if args.get_opt == True:
        print("getting optimal lexicons")
        get_optimal_lexicons()
