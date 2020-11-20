import numpy as np
from operator import itemgetter
import scipy.special
import math
import argparse
import pandas as pd
from collections import Counter
# Toy example:
# R:        |  0       1
# theta: -1 |  A       B
# theta:  0 |  C       D
# theta:  1 |  C       D
# R: distal level - 0: here / 1: there
# theta: direction - -1 = from somewhere / 0 = at somewhere / 1 = to somewhere


def calculate_systematicity_score(direction, distal_level, word):

    # direction: np.array with values -1 0 1 (-1: source / 0: location / 1: goal)
    # distal_level: np.array with values 0, 1, 2, ...
    # word: np.array for the word choice associated with the direction + distal_level combination

    # Toy example:
    # R:        |  0       1
    # theta: -1 |  A       B
    # theta:  0 |  C       D
    # theta:  1 |  C       D
    # R: distal level - 0: here / 1: there
    # theta: direction - -1 = from somewhere / 0 = at somewhere / 1 = to somewhere

    # The inputs will be coded as follows given the example above:
    # direction = np.array([-1, -1, 0, 0, 1, 1])
    # distal_level = np.array([0, 1, 0, 1, 0, 1])
    # word = np.array([0, 1, 2, 3, 2, 3])


    num_direction = list(set(direction))
    num_distal_level = list(set(distal_level))
    num_words = list(set(word))

    # checkpoints
    # print("num_direction = ", num_direction)
    # print("num_distal_level = ", num_distal_level)
    # print("num_words = ", num_words)
    # print("---")

    # building a back reference list num_words -> words
    word_ind_from_num_words = [0]*len(num_words)
    for i in num_words:
        word_ind_from_num_words[i] = [j for j in range(len(word)) if word[j] == i]

    # print("word_ind_from_num_words = ", word_ind_from_num_words)

    # --- w_R
    # the R attribute of each word: words -> w_R of that word
    w_R_words = [list(set(distal_level[word_ind_from_num_words[i]])) for i in num_words]
    # unique w_R
    w_R = [list(i) for i in set(map(tuple, w_R_words))]
    # print("w_R = ", w_R)
    # print("w_R_words = ", w_R_words)

    # --- w_theta
    # the theta attribute of each word: words -> w_theta of that word
    w_theta_words = [list(set(direction[word_ind_from_num_words[i]])) for i in num_words]
    # unique w_theta
    w_theta = [list(i) for i in set(map(tuple, w_theta_words))]
    # print("w_theta = ", w_theta)
    # print("w_theta_words = ", w_theta_words)
    # print("---")

    # --- build a back - reference list: w_theta -> words
    word_given_theta = [0] * len(w_theta)
    k = 0
    for i in w_theta:
        word_given_theta[k] = [j for j in range(len(w_theta_words)) if w_theta_words[j] == i]
        k += 1
    # print("word_given_theta = ", word_given_theta)

    # --- build a back - reference list: w_R -> words
    word_given_R = [0] * len(w_R)
    k = 0
    for i in w_R:
        word_given_R[k] = [j for j in range(len(w_R_words)) if w_R_words[j] == i]
        k += 1
    # print("word_given_R = ", word_given_R)

    # print("---")

    # reference list: w_theta -> m_R
    # print("len(w_theta) = ", len(w_theta))
    # print("len(num_distal_level) = ", len(num_distal_level))

    m_R_given_w_theta = {}
    for i in range(len(w_theta)):
        word_temp = word_given_theta[i]
        distal_temp = []
        for j in range(len(word_temp)):
            ind = word_ind_from_num_words[word_temp[j]]
            distal_temp.append(list(set(distal_level[ind])))
            # print(w_theta[i], word_temp[j], list(set(distal_level[ind])))
        # print(w_theta[i], distal_temp)
        m_R_given_w_theta[str(w_theta[i])] = distal_temp
    # print("m_R_given_w_theta = ", m_R_given_w_theta)

    # joint probability distribution p(m_R, w_theta)
    # print("--")
    count = [[0 for i in range(len(num_distal_level))] for j in range(len(w_theta))]
    for i in range(len(w_theta)):
        temp = []
        # get words from the given theta value
        word_temp = word_given_theta[i]
        # for each word in word_temp, grab their distal level
        distal_temp = [distal_level[word_ind_from_num_words[w]] for w in range(len(word_temp))]
        # print("w_theta = ", w_theta[i], ", word_given_theta = ", word_temp, ", distal_level_given_words = ", distal_temp)
        for j in range(len(num_distal_level)):
            temp = len([ind for ind in range(len(word)) if (distal_level[ind] == num_distal_level[j] and w_theta_words[word[ind]] == w_theta[i])])
            count[i][j] = temp
    count = np.array(count)
    p_m_R_w_theta = count / np.sum(count)
    # print("p_m_R_w_theta = ", p_m_R_w_theta)

    # marginal distribution p(m_R)
    p_m_R = p_m_R_w_theta.sum(axis=0)
    # print("p_m_R = ", p_m_R)
    #
    # marginal distribution p(w_theta)
    p_w_theta = p_m_R_w_theta.sum(axis=1)
    # print("p_w_theta = ", p_w_theta)
    #
    #
    # reference list: w_R -> m_theta
    # print("len(w_theta) = ", len(w_R))
    # print("len(num_distal_level) = ", len(num_direction))
    #
    m_theta_given_w_R = {}
    for i in range(len(w_R)):
        word_temp = word_given_R[i]
        direction_temp = []
        for j in range(len(word_temp)):
            ind = word_ind_from_num_words[word_temp[j]]
            direction_temp.append(list(set(direction[ind])))
            # print(w_R[i], word_temp[j], list(set(direction[ind])))
        # print(w_R[i], direction_temp)
        m_theta_given_w_R[str(w_R[i])] = direction_temp
    # print("m_theta_given_w_R = ", m_theta_given_w_R)
    #
    # joint probability distribution p(m_theta, w_R)
    # print("--")
    count = [[0 for i in range(len(num_direction))] for j in range(len(w_R))]
    for i in range(len(w_R)):
        temp = []
        # get words from the given theta value
        word_temp = word_given_R[i]
        # for each word in word_temp, grab their distal level
        direction_temp = [direction[word_ind_from_num_words[w]] for w in range(len(word_temp))]
        # print("w_theta = ", w_R[i], ", word_given_theta = ", word_temp, ", distal_level_given_words = ", direction_temp)
        for j in range(len(num_direction)):
            temp = len([ind for ind in range(len(word)) if (direction[ind] == num_direction[j] and w_R_words[word[ind]] == w_R[i])])
            count[i][j] = temp
    count = np.array(count)
    p_m_theta_w_R = count / np.sum(count)
    # print("p_m_theta_w_R = ", p_m_theta_w_R)
    #
    # marginal distribution p(m_theta)
    p_m_theta = p_m_theta_w_R.sum(axis=0)
    # print("p_m_theta = ", p_m_theta)
    #
    # marginal distribution p(w_R)
    p_w_R = p_m_theta_w_R.sum(axis=1)
    # print("p_w_R = ", p_w_R)

    # goal: calculate I(M_R ; W_theta) = H(M_R) + H(W_theta) - H(M_R, W_theta)
    # goal: calculate I(M_theta ; W_R) = H(M_theta) + H(W_R) - H(M_theta, W_R)
    # print("==== outputs ====")
    # entropy H(M_R,W_theta)
    H_M_R_W_theta = 0
    for i in [item for sublist in p_m_R_w_theta for item in sublist]:
        H_M_R_W_theta = H_M_R_W_theta - scipy.special.xlogy(i, i)
    # print("H_M_R_W_theta = ", H_M_R_W_theta)

    # entropy H(M_theta,W_R)
    H_M_theta_W_R = 0
    for i in [item for sublist in p_m_theta_w_R for item in sublist]:
        H_M_theta_W_R = H_M_theta_W_R - scipy.special.xlogy(i, i)
    # print("H_M_theta_W_R = ", H_M_theta_W_R)

    # entropy H(M_R)
    H_M_R = 0
    for i in p_m_R:
        H_M_R = H_M_R - scipy.special.xlogy(i, i)
    # print("H_M_R = ", H_M_R)

    # entropy H(M_theta)
    H_M_theta = 0
    for i in p_m_theta:
        H_M_theta = H_M_theta - scipy.special.xlogy(i, i)
    # print("H_M_theta = ", H_M_theta)

    # entropy H(W_R)
    H_W_R = 0
    for i in p_w_R:
        H_W_R = H_W_R - scipy.special.xlogy(i, i)
    # print("H_W_R = ", H_W_R)

    # entropy H(W_theta)
    H_W_theta = 0
    for i in p_w_theta:
        H_W_theta = H_W_theta - scipy.special.xlogy(i, i)
    # print("H_W_theta = ", H_W_theta)

    # I(M_R ; W_theta)
    I_M_R_W_theta = H_M_R + H_W_theta - H_M_R_W_theta
    # print("I_M_R_W_theta = ", I_M_R_W_theta, " nats / ", I_M_R_W_theta / math.log(2), " bits")
    # I(M_theta ; W_R)
    I_M_theta_W_R = H_M_theta + H_W_R - H_M_theta_W_R
    # print("I_M_theta_W_R = ", I_M_theta_W_R, " nats / ", I_M_theta_W_R / math.log(2), " bits")

    # print("==== final output ====")
    # systmaticity score:
    score = I_M_R_W_theta + I_M_theta_W_R
    return score / math.log(2)

score = calculate_systematicity_score(np.array([-1,-1,0,0,1,1]), np.array([0,1,0,1,0,1]), np.array([0,1,2,3,2,3]))

print(score)
