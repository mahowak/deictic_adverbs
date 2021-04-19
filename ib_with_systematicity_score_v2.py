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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

EPSILON = 10 ** -10
DEFAULT_NUM_EPOCHS = 10000
DEFAULT_LR = 0.001
DEFAULT_PRINT_EVERY = 500
DEFAULT_INIT_TEMPERATURE = 1
DEFAULT_BETA = 1
DEFAULT_GAMMA = 2
DEFAULT_ETA = 0
DEFAULT_MU = 0.2
DEFAULT_NUM_Z_R = 2
DEFAULT_NUM_Z_THETA = 2


print(torch.__version__)
print(device)
x = np.array([7, 39, 18, 2, 16, 7, 0.6, 6, 1.6]) # frequency data in Finnish

def softmax2(x):
    *initials, a, b = x.shape
    coalesced = initials + [a*b]
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
    p_x = torch.from_numpy(p_x) # shape X_R x X_theta
    *_, num_X_R, num_X_theta = p_x.shape

    num_X = num_X_R * num_X_theta
    num_Z = num_Z_R * num_Z_theta

    p_x = p_x.reshape(tuple(p_x.shape) + (1,1)) # shape X_R x X_theta x 1 x 1
    p_y_x = p_y_x.reshape(num_X, p_y_x.shape[-1]) # shape X x Y 
    
    # initialize q(z|x)
    energies = (1/init_temperature*torch.randn(num_X_R, num_X_theta, num_Z_R, num_Z_theta)).detach().to(device).requires_grad_(True)
    opt = torch.optim.Adam(params=[energies], **kwds)

    X_R_axis = -4
    X_theta_axis = -3
    Z_R_axis = -2
    Z_theta_axis = -1

    for i in range(num_epochs):
        opt.zero_grad()
        q_z_x = softmax2(energies) # shape X_R x X_theta x Z_R x Z_theta
        q_xz = p_x * q_z_x
        q_xz_flat = q_xz.reshape(num_X, num_Z) # shape X x Z
        i_xz = mi(q_xz_flat)
        i_zy = information_plane(q_xz_flat, p_y_x)

        q_xrztheta = q_xz.sum((X_theta_axis, Z_R_axis)) # shape X_R x Z_theta
        q_xthetazr = q_xz.sum((X_R_axis, Z_theta_axis)) # shape X_theta z Z_R

        mi_xr_ztheta = mi(q_xrztheta)
        mi_xtheta_zr = mi(q_xthetazr)
        
        s = mi_xr_ztheta + mi_xtheta_zr

        J = beta*i_xz - gamma*i_zy + eta*s

        J.backward()
        opt.step()

        if i % print_every == 0:
            print(i, " loss = ", J.item(), " I[X:Z] = ", i_xz.item(), " I[Z:Y] = ", i_zy.item(), " I[X_theta : Z_R] = ", mi_xtheta_zr.item(), " I[X_R : Z_theta] = ", mi_xr_ztheta.item(), " S = ", s.item(), file=sys.stderr)

    return softmax2(energies)
    

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
        # print("i_x_z =", i_x_z)
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

def xlogy(x, y, eps=EPSILON):
    return x * torch.log(y+eps)

# def xlogy(x,y):
#     #if x == torch.tensor([0]) and y == torch.tensor([0]):
#     #    return 0
#     #else:
#     print("x=", x)
#     print("y=", y)
#     return torch.mul(x , torch.log(y)) # it's fine for now but we need to fix this later
# # TODO: see if there's a nice xlogy function in torch


def systematicity_score(q_xz, num_R):
    # Systematicity score (formula to be used here):
    #    S[Z;X] = \sum_{x_R} \sum_{z_theta} p(z_theta | x_R) * p(x_R) * log(p(z_theta | x_R) / p(z_theta)) +
    #             \sum_{x_theta} \sum_{z_R} p(z_R | x_theta) * p(x_theta) * log(p(z_R | x_theta) / p(z_R))

    # nonzero = torch.heaviside(q_xz, torch.tensor([0], dtype=torch.float64))

    # step 1: calculate I[Z_theta; X_R]:
    p_z_R = calculate_p_z_R(q_xz, num_R)
    p_z_theta = calculate_p_z_theta(q_xz, num_R) # TODO: fix this function
    # print("p_z_R = ", p_z_R)
    # print("p_z_theta = ", p_z_theta)
    i_x_r_z_theta = torch.zeros(1, dtype=torch.float64)
    for i in range(num_R):
        R = i + 1
        p_z_theta_x_R = calculate_p_z_theta_x_R(segment_q_xz_by_R(q_xz, R), True)
        p_x_R = calculate_p_x_R(q_xz, R)
        for j in range(2**3):
            if p_z_theta_x_R[0][j] != torch.tensor([0]):
                i_x_r_z_theta += p_z_theta_x_R[0][j] * p_x_R * torch.log((p_z_theta_x_R[0][j] +
                                                                          torch.tensor([1e-6], dtype=torch.float64)) / (p_z_theta[0][j] + torch.tensor([1e-6], dtype=torch.float64)))
    i_x_r_z_theta.flatten()

    # step 2: calculate I[Z_R; X_theta]:
    i_x_theta_z_R = torch.zeros(1, dtype=torch.float64)
    for theta in np.array([-1, 0, 1]):
        p_z_R_x_theta = calculate_p_z_R_x_theta(segment_q_xz_by_theta(q_xz, theta, num_R), num_R, True)
        p_x_theta = calculate_p_x_theta(q_xz, theta, num_R)
        for j in range(2**num_R):
            if p_z_R_x_theta[0][j] != torch.tensor([0]):
                i_x_theta_z_R += p_z_R_x_theta[0][j] * p_x_theta * torch.log((p_z_R_x_theta[0][j] +
                                                                              torch.tensor([1e-6], dtype = torch.float64)) / (p_z_R[0][j] + torch.tensor([1e-6], dtype = torch.float64)))
    i_x_theta_z_R.flatten()
    # systematicity score:
    score = i_x_theta_z_R + i_x_r_z_theta
    # print(score)
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



#p_x = x / np.sum(x)
#gamma = 1.5
#eta = 2
#mu = 0.2
#num_R = int(len(p_x) / 3)
#p_y_x = get_prob_u_given_m_mini(mu, num_R)
#J, p = ib_sys(p_x, p_y_x, 4, gamma, eta, 500, 0.02)

#print("p=",p)
#print("J= ",J)

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
    p_x = x.reshape(num_R, num_theta) / x.sum() # TODO: is this reshape correct?
    p_y_x = get_prob_u_given_m_mini(mu, num_R)
    q = ib_sys2(p_x, p_y_x, gamma=gamma, eta=eta, beta=beta, num_Z_R = num_Z_R, num_Z_theta=num_Z_theta, num_epochs=num_epochs, init_temperature=init_temperature, **kwds)
    return q

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run IB optimization for bimorphemic deictic words using Finnish meaning frequencies.')
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA, help="Coefficient for I[X:Z]")    
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="Coefficient for -I[Z:Y]")
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA, help="Coefficient for nonsystematicity S")
    parser.add_argument("--mu", type=float, default=DEFAULT_MU, help="mu in p(y|z)")
    parser.add_argument("--num_Z_R", type=int, default=DEFAULT_NUM_Z_R, help="Number of distinct R morphemes")
    parser.add_argument("--num_Z_theta", type=int, default=DEFAULT_NUM_Z_THETA, help="Number of distinct theta morphemes")    
    parser.add_argument("--init_temperature", type=float, default=DEFAULT_INIT_TEMPERATURE, help="temperature of initialization")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="starting learning rate for Adam")
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS)    
    parser.add_argument("--print_every", type=int, default=DEFAULT_PRINT_EVERY, help="print results per x epochs")
    args = parser.parse_args()    
    main(gamma=args.gamma, eta=args.eta, mu=args.mu, beta=args.beta, num_epochs=args.num_epochs, print_every=args.print_every, lr=args.lr, init_temperature=args.init_temperature, num_Z_R=args.num_Z_R, num_Z_theta=args.num_Z_theta)



    # benchmark test (passed)
# non-systematic paradigm
# A = torch.tensor([[1/6, 0, 0, 0],
#                   [0, 0, 1/6, 0],
#                   [0, 0, 1/6, 0],
#                   [0, 1/6, 0, 0],
#                   [0, 1/6, 0, 0],
#                   [0, 0, 0, 1/6]], dtype = torch.float64)
#
# print(systematicity_score(A, 2))

# systematic paradigm
# B = torch.tensor([[1/6, 0, 0, 0],
#                   [0, 0, 1/6, 0],
#                   [0, 0, 1/6, 0],
#                   [0, 1/6, 0, 0],
#                   [0, 0, 0, 1/6],
#                   [0, 0, 0, 1/6]], dtype = torch.float64)
#
# print(systematicity_score(B, 2))



