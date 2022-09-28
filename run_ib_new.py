from this import d
from get_lang_data import get_lang_dict
from get_prior import get_exp_prior
from ib import information_plane, mi
from enumerate_lexicons import enumerate_possible_lexicons
from sklearn.linear_model import LinearRegression
from helper_functions import *


import joblib
import einops
import scipy
import argparse
from tqdm import tqdm
import itertools
import numpy as np
import os
import pandas as pd

AREAS = ["europe", "asia", "americas", "africa", "oceania"]

def classify_lang(lang):
    if lang not in ["simulated", "optimal"]:
        return "real"
    return lang

DEFAULT_NUM_ITER=10
PRECISION = 1e-16

# codes from Zaslavsky et al. (2018)
def xlogx(v):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(v > PRECISION, v * np.log2(v), 0)
    
def H(p, axis=None):
    """ Entropy """
    return -xlogx(p).sum(axis=axis)

def MI(pXY):
    """ mutual information, I(X;Y) """
    return H(pXY.sum(axis=0)) + H(pXY.sum(axis=1)) - H(pXY)

# function to calculate gNID (Zaslavsky et al., 2018)
def gNID(pW_X, pV_X, pX):
    if len(pX.shape) == 1:
        pX = pX[:, None]
    elif pX.shape[0] == 1 and pX.shape[1] > 1:
        pX = pX.T
    pXW = pW_X * pX
    pWV = pXW.T.dot(pV_X)
    pWW = pXW.T.dot(pW_X)
    pVV = (pV_X * pX).T.dot(pV_X)
    score = 1 - MI(pWV) / (np.max([MI(pWW), MI(pVV)]))
    return score

# modified _ib function to implement the reverse deterministic annealing algorithm (Zaslavsky & Tishby, 2019)
def _ib(p_x, p_y_x, Z, gamma, init, num_iter=DEFAULT_NUM_ITER, temperature = 1):
    """ Find encoder q(Z|X) to minimize J = I[X:Z] - gamma * I[Y:Z].
    
    Input:
    p_x : Distribution on X, of shape X.
    p_y_x : Conditional distribution on Y given X, of shape X x Y.
    gamma : A non-negative scalar value.
    Z : Support size of Z.

    Output: 
    Conditional distribution on Z given X, of shape X x Z.

    """
    # Support size of X
    X = p_x.shape[-1]

    # Support size of Y
    Y = p_y_x.shape[-1]

    # Randomly initialize the conditional distribution q(z|x)
    q_z_x = init #scipy.special.softmax(np.random.randn(X, Z), -1) # shape X x Z
    p_y_x = p_y_x[:, None, :] # shape X x 1 x Y
    p_x = p_x[:, None] # shape X x 1

    # Blahut-Arimoto iteration to find the minimizing q(z|x)
    for _ in range(num_iter):
        q_xz = p_x * q_z_x # Joint distribution q(x,z), shape X x Z
        q_z = q_xz.sum(axis=0, keepdims=True) # Marginal distribution q(z), shape 1 x Z
        q_y_z = ((q_xz / q_z)[:, :, None] * p_y_x).sum(axis=0, keepdims=True) # Conditional decoder distribution q(y|z), shape 1 x Z x Y
        d = ( 
            scipy.special.xlogy(p_y_x, p_y_x)
            - scipy.special.xlogy(p_y_x, q_y_z) # negative KL divergence -D[p(y|x) || q(y|z)]
        ).sum(axis=-1) # expected distortion over Y; shape X x Z
        q_z_x = scipy.special.softmax((np.log(q_z) - gamma*d)/temperature, axis=-1) # Conditional encoder distribution q(z|x) = 1/Z q(z) e^{-gamma*d}

    return q_z_x


class RunIB:

    def __init__(self,
    mu,
    distal_levels,
    pgs_dists,
    prior_spec=["place", "goal", "source"]):
    #prior_spec = ["goal", "place", "source"]):
        self.deictic_map = {}
        self.deictic_index = {}
        self.mu = mu
        self.distal_levels = distal_levels
        self.prior_spec = prior_spec
        self.prior = get_exp_prior(self.distal_levels, prior_spec)  # p(x)
        self.pgs_dists = pgs_dists
        self.logsp = np.logspace(2, 0, num=1500)
        c = 0
        for i in [("place", self.pgs_dists[0]),
        ("goal", self.pgs_dists[1]),
        ("source", self.pgs_dists[2])]:
            for j in range(distal_levels):
                self.deictic_map[c] = (j, i[1])
                self.deictic_index["D{}_{}".format(str(j + 1), i[0])] = c
                c += 1
        self.prob_u_given_m = self.get_prob_u_given_m()
        self.optimal_lexicons, self.optimal_lexicon_informativity, self.optimal_lexicon_complexity, self.optimal_lexicon_score = self.get_ib_curve()

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

    def get_ib_curve(self):
        num_words = 3 * self.distal_levels # maximal number of words
        init = np.identity(num_words) 

        qW_M = []
        informativity = []
        complexity = []

        for gamma in self.logsp:
            q_w_m = _ib(self.prior, self.prob_u_given_m, num_words, gamma, init, num_iter=20)
            informativity_temp, complexity_temp = information_plane(self.prior, self.prob_u_given_m, q_w_m)

            qW_M.append(q_w_m)
            informativity.append(informativity_temp)
            complexity.append(complexity_temp)
            init = q_w_m

        return qW_M, informativity, complexity, complexity - self.logsp*informativity
         
    def get_prob_u_given_m(self):
        u_m = np.zeros([len(self.deictic_map), len(self.deictic_map)])
        for i in self.deictic_map:
            distal, place = self.deictic_map[i]
            for num in self.deictic_map:
                costs = self.deictic_map[num]
                u_m[i][num] = 1 * (self.mu ** (np.abs(costs[0] - distal) + np.abs(costs[1] - place)))
        return u_m/u_m.sum(axis=1)[:, None]

    # function to find the objective function
    def get_objective(self, q_w_m, gamma):
        informativity, complexity = information_plane(self.prior, self.prob_u_given_m, q_w_m)
        return complexity - gamma * informativity

    # function to find gamma minimizing the objective function
    def find_gamma_index(self, q_w_m):
        objs = np.array([self.get_objective(q_w_m, gamma) for gamma in self.logsp])
        diff = objs - np.array(self.optimal_lexicon_score)
        return(diff.argmin())

    # function to calculate efficiency loss (Zaslavsky et al., 2018)
    def find_epsilon(self, q_w_m):
        objs = np.array([self.get_objective(q_w_m, gamma) for gamma in self.logsp])
        diff = objs - np.array(self.optimal_lexicon_score)
        return(diff.min()/self.logsp[diff.argmin()])

    # function to calculate efficiency loss (Zaslavsky et al., 2018)

    # @numba.jit(nopython=True)
    def find_everything(self, q_w_m):
        objs = np.array([self.get_objective(q_w_m, gamma) for gamma in self.logsp])
        diff = objs - np.array(self.optimal_lexicon_score)
        return diff.argmin(), self.logsp[diff.argmin()], diff.min()/self.logsp[diff.argmin()]

    # compute systematicity
    def count_patterns(self,arr):
        # arr: an np array
        arr_new = einops.rearrange(arr, "(a b) -> a b", b=3) # convert to standard paradigm table

        coded_r_patterns = [pd.factorize(arr_new[i])[0] for i in range(arr_new.shape[0])]
        r_patterns = np.unique(np.array(coded_r_patterns), axis = 0).shape[0]

        coded_theta_patterns = [pd.factorize(arr_new.T[i])[0] for i in range(arr_new.T.shape[0])]
        theta_patterns = np.unique(np.array(coded_theta_patterns), axis = 0).shape[0]

        #print(f'r_patterns: {r_patterns}; theta_patterns: {theta_patterns}')
        return r_patterns + theta_patterns



    def get_mi_for_all(self, sim_lex_dict={}, outfile="default"):
        num_meanings = self.distal_levels * 3
        lexicon_size_range = range(2, int(num_meanings) + 1)
        assert (len(self.prior) == num_meanings)
        dfs = []
        lexicons = []
        print('Compiling lexicons...')

        for lexicon_size in lexicon_size_range:
            all_lex = sim_lex_dict[lexicon_size]
            lexicons += [("simulated", l[1], "simulated") for l in all_lex]

        # add real lexicons
        lexicons += self.get_real_langs(num_meanings) # p(z|x)

        print("Making dataframe...")

        # turn lexicon into df
        df = pd.DataFrame([{dm: l[1].argmax(axis=1)[dm_num]
                        for dm_num, dm in enumerate(self.deictic_index)}for l in lexicons])

        information_plane_list = [information_plane(self.prior, self.prob_u_given_m, l[1]) for l in lexicons]  

        print("Calculating informativity and complexity...")                      
        df["I[U;W]"] = [l[0] for l in information_plane_list]
        df["I[M;W]"] = [l[1] for l in information_plane_list]

        print("Fitting tradeoff parameters...")
        
        # this takes the most time (20min)
        # temp = [self.find_everything(l[1]) for l in tqdm(lexicons)]

        temp = joblib.Parallel(n_jobs=-1)(joblib.delayed(self.find_everything)(l[1]) for l in tqdm(lexicons))
        # temp = map(self.find_everything, tqdm([l[1] for l in lexicons]))
        #

        idxs = [tup[0] for tup in temp]
        optimal_lexicons_ordered = [self.optimal_lexicons[int(idx)] for idx in idxs]

        df["gamma_fit"] = [t[1] for t in temp]

        print("Calculating epsilon and gNID...")
        df["epsilon"] = [t[2] for t in temp]
        df["gNID"] = [gNID(lexicons[i][1], optimal_lexicons_ordered[i], self.prior) for i in range(len(lexicons))]
       
        print("logging language data...")
        df["Language"] = [l[0] for l in lexicons]
        df["Area"] = [l[2] for l in lexicons]
        df["LangCategory"] = [classify_lang(lang) for lang in df["Language"]]

        print('Calculating systematicity...')
        lst = list(self.deictic_index.keys())
        lexicon_list = df[lst].values

        df["systematicity"] = [self.count_patterns(arr) for arr in lexicon_list]

        # df = systematicity(df) 

        dfs += [df]
        x =  pd.concat(dfs).sort_values(["I[U;W]"], ascending=False)
        #x.to_csv(outfile + '_mu_' + str(args.mu) + '_gamma_' + str(args.gamma) + "_" + "_".join([str(pgs) for pgs in self.pgs_dists]) + "_" + ".csv")
        x.to_csv(f'{outfile}_mu_{args.mu}_pgs'+"_".join([str(pgs) for pgs in self.pgs_dists]) + f"num_dists_{self.distal_levels}" + ".csv")
    
        # add average epsilon

        df_grid = pd.DataFrame({"place": [self.pgs_dists[0]],
        "goal": [self.pgs_dists[1]],
        "source": [self.pgs_dists[2]],
        "prior_spec": ["_".join(self.prior_spec)],
        "mean_gNID": [x["gNID"].loc[(x["LangCategory"] == 'real')].values.mean()],
        "mean_epsilon": [x["epsilon"].loc[(x["LangCategory"] == 'real')].values.mean()],
        "mu": [str(self.mu)]
        })

        df_grid.to_csv(outfile + "_gridsearch.csv", mode='a',
                  header=not os.path.exists(outfile + "_gridsearch.csv"))

        curve = pd.DataFrame({"gamma": self.logsp,
        "informativity": self.optimal_lexicon_informativity,
        "complexity": self.optimal_lexicon_complexity,
        "J": self.optimal_lexicon_score})

        #curve.to_csv(outfile + "ib_curve.csv")

        curve.to_csv(f'{outfile}_mu_{args.mu}_pgs'+"_".join([str(pgs) for pgs in self.pgs_dists]) + f"num_dists_{self.distal_levels}" + '_ib_curve.csv')

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
    parser.add_argument('--outfile', type=str, default="sheets/mi_test_1.csv")
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

    print('Initialization completed. Generating simulated lexicons...')

    sim_lex_dict = {lexicon_size: [lexicon for lexicon in enumerate_possible_lexicons(num_meanings, lexicon_size)] for 
        lexicon_size in lexicon_size_range}

    print('Simulated lexicon generated.')
    
    if args.grid_search:
        for i in np.append(np.linspace(-10, 10, 8), np.array(0)):
            for j in np.append(np.linspace(-10, 10, 8), np.array(0)):
                if np.abs(i) > .5 and np.abs(j) > .5:
                    RunIB(args.mu, args.distal,
                        [0, i, j]).get_mi_for_all(sim_lex_dict=sim_lex_dict,
                                                outfile = args.outfile + str(args.mu) + str(args.gamma))

    elif args.prior_search:
        for perm in (list(itertools.permutations(["place", "goal", "source"])) + 
        [["unif", "unif", "unif"], ["place", "place", "place"]]):
            RunIB(args.mu, args.distal, args.pgs,
                  prior_spec=perm).get_mi_for_all(sim_lex_dict=sim_lex_dict,
                                            outfile=args.outfile)
    elif args.mu_search:
        for mu in np.append(np.linspace(0.05, 0.99, 19), np.array(0.99)):
            RunIB(mu, args.distal, args.pgs, prior_spec =["place", "goal", "source"] ).get_mi_for_all(get_opt=args.get_opt,
                                            sim_lex_dict=sim_lex_dict,
                                            outfile=args.outfile)


    elif args.total_search:
        for perm in (list(itertools.permutations(["place", "goal", "source"])) +
                     [["unif", "unif", "unif"], ["place", "place", "place"]]):
            for mu in [.1, .2, .3]:
                for i in np.append(np.linspace(-5, 5, 20), np.array(0)):
                    for j in np.append(np.linspace(-5, 5, 20), np.array(0)):
                        pgs = [0, i, j]
                        RunIB(mu, args.distal,
                            pgs,
                            prior_spec=perm).get_mi_for_all(sim_lex_dict=sim_lex_dict,
                                                            outfile=args.outfile)
    elif args.total_search_mini:
        for perm in (list(itertools.permutations(["place", "goal", "source"])) +
                     [["unif", "unif", "unif"]]):
            for mu in [.2, .3]:
                for i in [0.8, 0.9, 1, 1.1, 1.2, 1.3]:
                    for j in [-0.8, -0.9, -1, -1.1, -1.2, -1.3]:
                        RunIB(mu, args.distal,
                            [0, i, j],
                            prior_spec=perm).get_mi_for_all(sim_lex_dict=sim_lex_dict,
                                                            outfile=args.outfile)

    else:
        RunIB(args.mu, args.distal, args.pgs).get_mi_for_all(sim_lex_dict=sim_lex_dict,
            outfile = args.outfile)