""" This script calculates the systematicity score of a discrete paradigm (part of the ib_with_systematicity_score_v2.py script)

 Systematicity score S[X;Z] :
 S[X;Z] = I(Z_theta ; X_R) + I(Z_R ; X_theta)

 """

import numpy as np
import math
import scipy
import pandas as pd
import torch
from scipy import special
import torch.optim as optim
import torch.nn as nn
import itertools

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(torch.__version__)
print(device)

""" 
    Sample paradigm table: [the setup is different from that in run_ib.py]
    Notice that this is a JOINT probability table: all the entries must sum up to 1.
                    word
    R   theta  0    1    2    3      4
    1   -1     0.25 0    0    0      0
    1   0      0    0    0.4  0      0
    1   1      ...
    2   -1
    2   0
    2   1
    3   -1
    3   0
    3   1
    """


def systematicity_score(q_xz, num_R):
    # Systematicity score (formula to be used here):
    #    S[Z;X] = \sum_{x_R} \sum_{z_theta} p(z_theta | x_R) * p(x_R) * log(p(z_theta | x_R) / p(z_theta)) +
    #             \sum_{x_theta} \sum_{z_R} p(z_R | x_theta) * p(x_theta) * log(p(z_R | x_theta) / p(z_R))

    # nonzero = torch.heaviside(q_xz, torch.tensor([0], dtype=torch.float64))

    # step 1: calculate I[Z_theta; X_R]:
    p_z_R = calculate_p_z_R(q_xz, num_R)
    # print("p_z_R = ", p_z_R)
    p_z_theta = calculate_p_z_theta(q_xz, num_R) # TODO: fix this function
    # print("p_z_R = ", p_z_R)
    # print("p_z_theta = ", p_z_theta)
    i_x_r_z_theta = torch.zeros(1, dtype=torch.float64)
    for i in range(num_R):
        R = i + 1
        p_z_theta_x_R = calculate_p_z_theta_x_R(segment_q_xz_by_R(q_xz, R), True)
        # print("i = ", i, "p_z_theta_x_R = ", p_z_theta_x_R)
        # print("i = ", i, "p_z_theta = ", p_z_theta)
        p_x_R = calculate_p_x_R(q_xz, R)
        for j in range(2**3):
            if p_z_theta_x_R[0][j] != torch.tensor([0]):
                # print(p_z_theta_x_R[0][j]/p_z_theta[0][j])
                i_x_r_z_theta += p_z_theta_x_R[0][j] * p_x_R * torch.log((p_z_theta_x_R[0][j] +
                                                                          torch.tensor([1e-6], dtype=torch.float64)) / (p_z_theta[0][j] + torch.tensor([1e-6], dtype=torch.float64)))
    i_x_r_z_theta.flatten()

    # step 2: calculate I[Z_R; X_theta]:
    i_x_theta_z_R = torch.zeros(1, dtype=torch.float64)
    for theta in np.array([-1, 0, 1]):
        p_z_R_x_theta = calculate_p_z_R_x_theta(segment_q_xz_by_theta(q_xz, theta, num_R), num_R, True)
        # print("theta = ", theta, "p_z_R_x_theta = ", p_z_R_x_theta)
        # print("theta = ", theta, "p_z_R = ", p_z_R)
        p_x_theta = calculate_p_x_theta(q_xz, theta, num_R)
        for j in range(2**num_R):
            if p_z_R_x_theta[0][j] != torch.tensor([0]):
                i_x_theta_z_R += p_z_R_x_theta[0][j] * p_x_theta * torch.log((p_z_R_x_theta[0][j] +
                                                                              torch.tensor([1e-6], dtype = torch.float64)) / (p_z_R[0][j] + torch.tensor([1e-6], dtype = torch.float64)))
    i_x_theta_z_R.flatten()
    # systematicity score:
    score = i_x_theta_z_R + i_x_r_z_theta
    # print(score)
    return score/0.6931 # convert to bits



def calculate_p_z_theta(q_xz, num_R):
    # input: q_zx - the joint distribution q(z,x) - torch tensor
    # input: num_R - total number of distal levels - number
    # print(theta)
    p_z_theta = torch.zeros(1, 2**3)
    for i in range(num_R):
        R = i + 1
        segmented_q_xz = segment_q_xz_by_R(q_xz, R)
        p_z_theta += calculate_p_z_theta_x_R(segmented_q_xz, False)
    return p_z_theta / torch.sum(p_z_theta)


def calculate_p_z_R(q_xz, num_R):
    # input: q_zx - the joint distribution q(z,x) - torch tensor
    # input: num_R - total number of distal levels - number
    # print(theta)
    p_z_R = torch.zeros(1, 2**num_R)
    for theta in np.array([-1, 0, 1]):
        segmented_q_xz = segment_q_xz_by_theta(q_xz, theta, num_R)
        p_z_R += calculate_p_z_R_x_theta(segmented_q_xz, num_R, False)
    return p_z_R / torch.sum(p_z_R)


def calculate_p_z_theta_x_R(segmented_q_xz, norm):
    p_z_theta_x_R = torch.zeros(1, 2**3)#, requires_grad=True)
    mask = [i for i in itertools.product(range(2), repeat=3)]
    mask = torch.transpose(torch.tensor(mask), 0, 1) # each column stands for a possible z_theta attribute
    masked_segmented_q_xz = torch.heaviside(segmented_q_xz, torch.tensor([0], dtype = torch.float64))
    for i in range(2**3):
        places = torch.ones_like(segmented_q_xz) - torch.pow(masked_segmented_q_xz - mask[:,i][:,None], 2)
        ind = torch.prod(places, 0) # which column fits the pattern
        p_z_theta_x_R[0, i] = torch.sum(torch.sum(torch.mul(ind, segmented_q_xz), 1))
    if norm == True:
        return p_z_theta_x_R / torch.sum(p_z_theta_x_R)
    else:
        return p_z_theta_x_R


def calculate_p_z_R_x_theta(segmented_q_xz, num_R, norm):
    p_z_R_x_theta = torch.zeros(1, 2**num_R)
    mask = [i for i in itertools.product(range(2), repeat=num_R)]
    mask = torch.transpose(torch.tensor(mask), 0, 1) # each column stands for a possible z_theta attribute
    masked_segmented_q_xz = torch.heaviside(segmented_q_xz, torch.tensor([0], dtype = torch.float64))
    for i in range(2**num_R):
        places = torch.ones_like(segmented_q_xz) - torch.pow(masked_segmented_q_xz - mask[:,i][:,None], 2)
        ind = torch.prod(places, 0) # which column fits the pattern
        p_z_R_x_theta[0, i] = torch.sum(torch.sum(torch.mul(ind, segmented_q_xz), 1))
    if norm == True:
        return p_z_R_x_theta / torch.sum(p_z_R_x_theta)
    else:
        return p_z_R_x_theta


def calculate_p_x_R(q_xz, R):
    # input: q_xz - the joint distributions
    # input: R - the R attribute to be summed
    p_z_x_R = segment_q_xz_by_R(q_xz, R)
    p_x_R = torch.sum(p_z_x_R)
    return p_x_R


def calculate_p_x_theta(q_xz, theta, num_R):
    # input: q_sz - the joint distributions
    # input: theta - the theta attribute to be summed
    # input: num_R - the total number of distal levels
    p_z_x_theta = segment_q_xz_by_theta(q_xz, theta, num_R)
    p_x_theta = torch.sum(p_z_x_theta)
    return p_x_theta


def segment_q_xz_by_R(q_xz, R):
    # input: R - the R attributes
    # input: q_xz - the joint distribution q(x,z) - pytorch tensor
    # output: q(z, x_theta |x_R) - the conditional distribution - pytorch tensor
    indices = torch.tensor(R_to_indices(R), dtype=torch.int64).flatten()
    q_z_xR = torch.index_select(q_xz, 0, indices)
    return q_z_xR


def segment_q_xz_by_theta(q_xz, theta, num_R):
    # input: theta - the theta attributes
    # input: q_xz - the joint distribution q(x,z) - pytorch tensor
    # output: q(z, x_R |x_theta) - the conditional distribution - pytorch tensor
    indices = torch.tensor(theta_to_indices(theta, num_R), dtype=torch.int64).flatten()
    # print(indices)
    q_z_xtheta = torch.index_select(q_xz, 0, indices)
    return q_z_xtheta

def R_to_indices(R):
    return np.array([[3 * R - 3], [3 * R - 2], [3 * R - 1]])


def theta_to_indices(theta, num_R):
    index = np.zeros((num_R, 1))
    if theta == -1:
        num = 3
    if theta == 0:
        num = 2
    if theta == 1:
        num = 1

    for i in range(num_R):
        index[i] = int(3 * (i + 1) - num)
    return index


def get_discrete_paradigm(q_xz):
    # input: q_xz - joint distribution (pytorch tensor)
    # this script selects the largest entry of each row, replace its value to the row sum
    # and make every other entry in that row 0
    new_q_xz = torch.zeros_like(q_xz, requires_grad=True, dtype=torch.float64)
    replacement = torch.sum(q_xz, 1)
    for i in range(q_xz.shape[0]):
        loc = torch.max(q_xz[i, ], 0).indices.item()
        new_q_xz[i][loc] = replacement[i]
    return new_q_xz


# non-systematic paradigm
A = torch.tensor([[1/6, 0, 0, 0],
                  [0, 0, 1/6, 0],
                  [0, 0, 1/6, 0],
                  [0, 1/6, 0, 0],
                  [0, 1/6, 0, 0],
                  [0, 0, 0, 1/6]], dtype = torch.float64)

print("the systematicity score is ", systematicity_score(A, 2).item(), " bits")

# systematic paradigm
B = torch.tensor([[1/6, 0, 0, 0],
                  [0, 0, 1/6, 0],
                  [0, 0, 1/6, 0],
                  [0, 1/6, 0, 0],
                  [0, 0, 0, 1/6],
                  [0, 0, 0, 1/6]], dtype = torch.float64)

print("the systematicity score is ", systematicity_score(B, 2).item(), " bits")

# English paradigm (Finnish prior)
C = torch.tensor([[0, 0, 7/97.2, 0],
                 [39/97.2, 0, 0, 0],
                 [18/97.2, 0, 0, 0],
                 [0, 0, 0, 2.6/97.2],
                 [0, 22/97.2, 0, 0],
                 [0, 8.6/97.2, 0, 0]], dtype = torch.float64)
print("the systematicity score is ", systematicity_score(C, 2).item(), " bits")

# Icelandic paradigm (Finnish prior)
D = torch.tensor([[0, 0, 7/97.2, 0, 0, 0, 0, 0],
                 [19.5/97.2, 0, 0, 0, 0, 0, 19.5/97.2, 0],
                 [0, 9/97.2, 0, 0, 0, 0, 9/97.2, 0],
                 [0, 0, 0, 0, 0, 2.6/97.2, 0, 0],
                 [0, 0, 0, 11/97.2, 0, 0, 0, 11/97.2],
                 [0, 0, 0, 0, 8.6/97.2, 0, 0, 8.6/97.2]], dtype = torch.float64)
print("the systematicity score is ", systematicity_score(D, 2).item(), " bits")
