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
import sys
import numpy as np
import math
import scipy
import pandas as pd
import torch
from scipy import special
import torch.optim as optim
import torch.nn as nn
import argparse
import itertools
import tabulate
import string

device = 'cuda' if torch.cuda.is_available() else 'cpu'

EPSILON = 10 ** -10
DEFAULT_NUM_EPOCHS = 10000
DEFAULT_LR = 0.001
DEFAULT_PRINT_EVERY = 500
DEFAULT_INIT_TEMPERATURE = 5
DEFAULT_BETA = 1
DEFAULT_GAMMA = 10
DEFAULT_ETA = 0.2
DEFAULT_MU = 0.2
DEFAULT_NUM_Z_R = 2
DEFAULT_NUM_Z_THETA = 2

print(torch.__version__)
print(device)
x = np.array([7, 39, 18, 2, 16, 7, 0.6, 6, 1.6])  # frequency data in Finnish

def softmax2(x):
    *initials, a, b = x.shape
    coalesced = initials + [a * b]
    return torch.softmax(x.reshape(coalesced), -1).reshape(x.shape)


def ib_sys2(p_x,
            p_y_x,
            num_Z_R=DEFAULT_NUM_Z_R,
            num_Z_theta=DEFAULT_NUM_Z_THETA,
            beta=DEFAULT_BETA,
            gamma=DEFAULT_GAMMA,
            eta=DEFAULT_ETA,
            num_epochs=DEFAULT_NUM_EPOCHS,
            print_every=DEFAULT_PRINT_EVERY,
            init_temperature=DEFAULT_INIT_TEMPERATURE,
            **kwds):

    
    # Input:
    # p_x, a tensor of shape X_R x X_theta giving P_X(R, theta)
    # p_y_x, a tensor of shape X_R x X_theta x Y giving P(Y | R, theta)
    p_x = torch.from_numpy(p_x)  # shape X_R x X_theta
    *_, num_X_R, num_X_theta = p_x.shape

    num_X = num_X_R * num_X_theta
    num_Z = num_Z_R * num_Z_theta

    p_x = p_x.reshape(tuple(p_x.shape) + (1, 1))  # shape X_R x X_theta x 1 x 1
    p_y_x = p_y_x.reshape(num_X, p_y_x.shape[-1])  # shape X x Y

    # initialize q(z|x)
    energies = (1 / init_temperature * torch.randn(num_X_R, num_X_theta, num_Z_R, num_Z_theta)).detach().to(
        device).requires_grad_(True)
    opt = torch.optim.Adam(params=[energies], **kwds)

    for i in range(num_epochs):
        opt.zero_grad()
        q_z_x = softmax2(energies)  # shape X_R x X_theta x Z_R x Z_theta
        q_xz = p_x * q_z_x
        q_xz_flat = q_xz.reshape(num_X, num_Z)  # shape X x Z
        i_xz = mi(q_xz_flat)
        i_zy = information_plane(q_xz_flat, p_y_x)
        s = lexicon_systematicity(p_x, q_z_x)

        J = beta * i_xz - gamma * i_zy + eta * s

        J.backward()
        opt.step()

        if i % print_every == 0:
            print(i, " loss = ", J.item(), " I[X:Z] = ", i_xz.item(), " I[Z:Y] = ", i_zy.item(), " S = ", s.item(), file=sys.stderr)

    # output: q(z_R, z_theta | x_R, x_theta)
    return softmax2(energies)

def lexicon_stats(source, lexicon):
    if len(source.shape) < len(lexicon.shape):
        source = source[:, :, None, None]
        
    q = source * lexicon

    X_R_axis = -4
    X_theta_axis = -3
    Z_R_axis = -2
    Z_theta_axis = -1    

    mi_xtheta_ztheta = mi(q.sum((X_R_axis, Z_R_axis)))
    mi_xtheta_zr = mi(q.sum((X_R_axis, Z_theta_axis))) # shape X_theta x Z_R
    mi_xr_ztheta = mi(q.sum((X_theta_axis, Z_R_axis))) # shape X_R x Z_theta
    mi_xr_zr = mi(q.sum((X_theta_axis, Z_theta_axis)))

    return mi_xtheta_ztheta, mi_xtheta_zr, mi_xr_ztheta, mi_xr_zr
    

def lexicon_systematicity(source, lexicon):
    # source is a distribution on (X_R, X_theta) of shape X_R x X_theta
    # lexicon is a conditional distribution on (Z_R, Z_theta) given (X_R, X_theta) of shape X_R x X_theta x Z_R x Z_theta
    mi_xtheta_ztheta, mi_xtheta_zr, mi_xr_ztheta, mi_xr_zr = lexicon_stats(source, lexicon)
    return mi_xr_ztheta + mi_xtheta_zr

def full_systematicity(source, lexicon): # higher = more systematic?
    mi_xtheta_ztheta, mi_xtheta_zr, mi_xr_ztheta, mi_xr_zr = lexicon_stats(source, lexicon)
    return mi_xr_ztheta + mi_xtheta_zr + mi_xtheta_ztheta + mi_xr_zr

def lexicon_systematicity2(source, lexicon):
    mi_xtheta_ztheta, mi_xtheta_zr, mi_xr_ztheta, mi_xr_zr = lexicon_stats(source, lexicon)
    return mi_xr_zr + mi_xtheta_ztheta    
    

def xlogy(x, y):
    return x * (y+EPSILON).log()


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


def get_prob_u_given_m_mini(mu, distal_levels):
    u_m = torch.zeros([distal_levels, distal_levels])
    for i in range(distal_levels):
        for num in range(distal_levels):
            u_m[i][num] = 1 * (mu ** (np.abs(num - i)))
    u_m_norm = u_m / torch.sum(u_m, 1, keepdim=True)
    return torch.repeat_interleave(u_m_norm, 3 * torch.ones([distal_levels]).long(), dim=0)

def get_prob_u_given_m(mu, gamma, distal_levels):
    deictic_map = {} # initialize the dictionary
    deictic_index = {} # initialize the dictionary
    c = 0 # create a dictionary to map a (distal_level, direction) pair to an index
    for i in [("place", 1), ("goal", 0), ("source", 2)]:
        for j in range(distal_levels):
            deictic_map[c] = (j, i[1])
            deictic_index["D{}_{}".format(str(j + 1), i[0])] = c
            c += 1

    u_m = np.zeros([len(deictic_map), len(deictic_map)])
    for i in deictic_map:
        distal, place = deictic_map[i]
        for num in deictic_map:
            costs = deictic_map[num]
            u_m[i][num] = 1 * (mu ** (np.abs(costs[0] - distal) + np.abs(costs[1] - place)))
    return torch.from_numpy(u_m/u_m.sum(axis=1)[:, None])


def main(gamma=DEFAULT_GAMMA,
         beta=DEFAULT_BETA,
         eta=DEFAULT_ETA,
         mu=DEFAULT_MU,
         num_R=3,
         num_theta=3,
         num_Z_R=DEFAULT_NUM_Z_R,
         num_Z_theta=DEFAULT_NUM_Z_THETA,
         num_epochs=DEFAULT_NUM_EPOCHS,
         init_temperature=DEFAULT_INIT_TEMPERATURE,
         **kwds):
    p_x = x.reshape(num_R, num_theta) / x.sum()  # TODO: is this reshape correct?
    p_y_x = get_prob_u_given_m(mu, gamma, num_R)
    q = ib_sys2(p_x, p_y_x, gamma=gamma, eta=eta, beta=beta, num_Z_R=num_Z_R, num_Z_theta=num_Z_theta,
                num_epochs=num_epochs, init_temperature=init_temperature, **kwds)
    return q


def decode_lexicon(q):
    # convert the 4D matrix q to readable lexicon tables
    nWords = q.size(-2) * q.size(-1)
    num_z_R = q.size(0)
    num_z_theta = q.size(1)
    letter = [[0]*num_z_theta for i in range(num_z_R)]

    for i in range(num_z_R):
        for j in range(num_z_theta):
            loc = torch.max(q[i][j].flatten(),0).indices.item()
            letter[i][j] = string.ascii_uppercase[loc]

    return letter

def print_lexicon(q):
    letter = decode_lexicon(q)
    print(tabulate.tabulate(letter))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run IB optimization for bimorphemic deictic words using Finnish meaning frequencies.')
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA, help="Coefficient for I[X:Z]")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="Coefficient for -I[Z:Y]")
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA, help="Coefficient for nonsystematicity S")
    parser.add_argument("--mu", type=float, default=DEFAULT_MU, help="mu in p(y|z)")
    parser.add_argument("--num_Z_R", type=int, default=DEFAULT_NUM_Z_R, help="Number of distinct R morphemes")
    parser.add_argument("--num_Z_theta", type=int, default=DEFAULT_NUM_Z_THETA,
                        help="Number of distinct theta morphemes")
    parser.add_argument("--init_temperature", type=float, default=DEFAULT_INIT_TEMPERATURE,
                        help="temperature of initialization")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="starting learning rate for Adam")
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--print_every", type=int, default=DEFAULT_PRINT_EVERY, help="print results per x epochs")
    args = parser.parse_args()
    # q(z_R, z_theta | x_R, x_theta)
    q = main(gamma=args.gamma, eta=args.eta, mu=args.mu, beta=args.beta, num_epochs=args.num_epochs,
         print_every=args.print_every, lr=args.lr, init_temperature=args.init_temperature, num_Z_R=args.num_Z_R,
         num_Z_theta=args.num_Z_theta)
    # print(q)
    print_lexicon(q)



