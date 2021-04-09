""" This script minimizes J = I[X;Z] - gamma * I[Y;Z] + eta * S, where S is the systematicity score of a paradigm
 Variables:
 Z: word [W in Zaslavsky et al. (2018)]
 Y: universe [U in Zaslavsky et al. (2018)]
 X: meaning [M in Zaslavsky et al. (2018)]
 q(Z|X): encoder to be optimized

 Inputs:
 p_x: distribution on X, of shape X
 p_y_x: conditional distribution of Y given X, of shape X * Y
 gamma: a non-negative scalar value
 eta: a non-negative scalar value
 Z: support size of Z (number of words in a paradigm)


 Systematicity score S[X;Z] :
 S[X;Z] = I(Z_theta ; X_R) + I(Z_R ; X_theta)

 Deviation from v1: try to do everything in torch and make sure things are differentiable (so that we can do gradient)

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
x = np.array([7, 39, 18, 2, 16, 7, 0.6, 6, 1.6]) # frequency data in Finnish


def ib_sys(p_x, p_y_x, Z, gamma, eta, num_epoch, lr):
    # num epoch: number of epochs
    # lr: learning rate

    # Support size of X
    X = p_x.shape[-1]

    # Support size of Y
    Y = p_y_x.shape[-1]

    # convert p_x and p_y_x into torch tensors
    p_x = torch.from_numpy(p_x)
    p_x.unsqueeze_(1)

    """ we randomly initialize an torch tensor of size X by Z (X should be the number of distal levels * 3, whereas Z should be
    the number of words in the paradigm.
    Convention: the theta value of the (3*n)th row should be 1, (3*n - 1) th row be 0, and (3*n - 2)th row be -1, where n
    corresponds to the distal level (i.e. n = 1, 2, 3, ...)


    Sample q(z|x) table: [the setup is different from that in run_ib.py]
                    word
    R   theta  0    1    2    3      4
    1   -1     0.1  0.2  0.3  0.2    0.2
    1   0      0.05 0.05 0.2  0.25   0.45
    1   1      ...
    2   -1
    2   0
    2   1
    3   -1
    3   0
    3   1
    """

    # initialize q(z|x)
    m = torch.nn.Softmax(dim=-1)
    input = torch.randn(X, Z, dtype=torch.float, requires_grad=True, device=device)
    # number of distal levels
    num_R = int(X / 3)

    for epoch in range(num_epoch):
        q_z_x = m(input)
        # q(x,z) = q(z|x)*p(x)
        q_xz = torch.mul(q_z_x, p_x)  # obtain joint table
        # test (systematic case):
        # q_xz = torch.tensor([[1/6, 0, 0, 0], [0, 0, 1/6, 0], [0, 0, 1/6, 0], [0, 1/6, 0, 0], [0, 0, 0, 1/6], [0, 0, 0, 1/6]], dtype= torch.float64)
        # test (unsystematic case):
        # q_z_x = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype= torch.float64)
        # q_xz = torch.tensor([[1/6, 0, 0, 0], [0, 0, 1/6, 0], [0, 0, 1/6, 0], [0, 1/6, 0, 0], [0, 1/6, 0, 0], [0, 0, 0, 1/6]], dtype= torch.float64).detach()

        # component 1: systematicity socre S[X;Z]
        score = systematicity_score(q_xz, num_R).flatten()
        # component 2: mutual information I[X;Z]:
        i_x_z = mi(q_xz)
        # print(i_x_z)
        # component 3: mutual information I[Y;Z]:
        # calculate I[Y;Z] from p(y,z)
        # formula: p(y,z) = sum_{x} p(x,y,z) = sum_{z} p(z|x) * p(y|x) * p(x)
        # [assuming y and z are conditionally independent from x]
        i_y_z = information_plane(q_xz, p_y_x)
        # print(i_y_z)

        # objective function:
        J = i_x_z - gamma * i_y_z + eta * score

        # print("J= ", J)

        J.backward()

        with torch.no_grad():
            input -= lr * input.grad
            # print(input.grad)

        input.grad.zero_()

        # print("epoch = ", epoch, "q_xz =", torch.mul(m(input), p_x))

    return J, torch.mul(m(input), p_x)




def mi(p_xy):
    """ Calculate mutual information of a distribution P(x,y)

    Input:
    p_xy: An X x Y array giving p(x,y)

    Output:
    The mutual information I[X:Y], a nonnegative scalar,
    """
    p_x = torch.sum(p_xy, -1)
    p_y = torch.sum(p_xy, -2)
    return torch.sum(xlogy(p_xy, p_xy)) - torch.sum(xlogy(p_x, p_x)) - torch.sum(xlogy(p_y, p_y))


def information_plane(p_xz, p_y_x):
    # this is a modified version of the information_plane function in ib.py
    # inputs:
    # conditional distributions p(z|x) [size: X * Z] - torch tensor
    #                           p(y|x) [size: X * Y] - torch tensor
    #             distribution  p(x)   [size: X * 1] - torch tensor

    # Step 1: we need to build the joint distribution p(x,y,z) [size: X * Y * Z]
    # print(p_xz[:, None, :].size())
    # print(p_y_x[:, :, None].size())
    p_xyz = torch.mul(p_xz[:, None, :], p_y_x[:, :, None])
    p_yz = torch.sum(p_xyz, 0)
    return mi(p_yz)


def xlogy(x,y):
    return x * torch.log(y) # it's fine for now but we need to fix this later


def systematicity_score(q_xz, num_R):
    # Systematicity score (formula to be used here):
    #    S[Z;X] = \sum_{x_R} \sum_{z_theta} p(z_theta | x_R) * p(x_R) * log(p(z_theta | x_R) / p(z_theta)) +
    #             \sum_{x_theta} \sum_{z_R} p(z_R | x_theta) * p(x_theta) * log(p(z_R | x_theta) / p(z_R))

    # nonzero = torch.heaviside(q_xz, torch.tensor([0], dtype=torch.float64))

    # step 1: calculate I[Z_theta; X_R]:
    p_z_R = calculate_p_z_R(q_xz, num_R)
    p_z_theta = calculate_p_z_theta(q_xz, num_R)
    # print("p_z_R = ", p_z_R)
    # print("p_z_theta = ", p_z_theta)
    i_x_r_z_theta = torch.zeros(1, dtype=torch.float64)
    for i in range(num_R):
        R = i + 1
        p_z_theta_x_R = calculate_p_z_theta_x_R(segment_q_xz_by_R(q_xz, R), True)
        p_x_R = calculate_p_x_R(q_xz, R)
        for j in range(2**3):
            if p_z_theta_x_R[0][j] != torch.tensor([0]):
                i_x_r_z_theta += p_z_theta_x_R[0][j] * p_x_R * torch.log(p_z_theta_x_R[0][j] / p_z_theta[0][j])
    i_x_r_z_theta.flatten()

    # step 2: calculate I[Z_R; X_theta]:
    i_x_theta_z_R = torch.zeros(1, dtype=torch.float64)
    for theta in np.array([-1, 0, 1]):
        p_z_R_x_theta = calculate_p_z_R_x_theta(segment_q_xz_by_theta(q_xz, theta, num_R), num_R, True)
        p_x_theta = calculate_p_x_theta(q_xz, theta, num_R)
        for j in range(2**num_R):
            if p_z_R_x_theta[0][j] != torch.tensor([0]):
                i_x_theta_z_R += p_z_R_x_theta[0][j] * p_x_theta * torch.log(p_z_R_x_theta[0][j] / p_z_R[0][j])
    i_x_theta_z_R.flatten()
    # systematicity score:
    score = i_x_theta_z_R + i_x_r_z_theta
    return score



def calculate_p_z_theta(q_xz, num_R):
    # input: q_zx - the joint distribution q(z,x) - torch tensor
    # input: num_R - total number of distal levels - number
    # print(theta)
    p_z_theta = torch.zeros(1, 2**3)
    for i in range(num_R):
        R = i + 1
        segmented_q_xz = segment_q_xz_by_R(q_xz, R)
        p_z_theta += calculate_p_z_theta_x_R(segmented_q_xz, False)
    return p_z_theta


def calculate_p_z_R(q_xz, num_R):
    # input: q_zx - the joint distribution q(z,x) - torch tensor
    # input: num_R - total number of distal levels - number
    # print(theta)
    p_z_R = torch.zeros(1, 2**num_R)
    for theta in np.array([-1, 0, 1]):
        segmented_q_xz = segment_q_xz_by_theta(q_xz, theta, num_R)
        p_z_R += calculate_p_z_R_x_theta(segmented_q_xz, num_R, False)
    return p_z_R


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


def get_prob_u_given_m_mini(mu, distal_levels):
    u_m = torch.zeros([distal_levels, distal_levels])
    for i in range(distal_levels):
        for num in range(distal_levels):
            u_m[i][num] = 1 * (mu ** (np.abs(num - i)))
    u_m_norm = u_m / torch.sum(u_m, 1, keepdim=True)
    return torch.repeat_interleave(u_m_norm, 3*torch.ones([distal_levels]).long(), dim=0)


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



p_x = x / np.sum(x)
gamma = 1.5
eta = 2
mu = 0.2
num_R = int(len(p_x) / 3)
p_y_x = get_prob_u_given_m_mini(mu, num_R)
J, p = ib_sys(p_x, p_y_x, 6, gamma, eta, 100000, 0.02)

print(p)
print(J)


