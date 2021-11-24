from ib import *
import numpy as np
import scipy.special

# deterministic information bottleneck implementation
def dib(p_x, p_y_x, Z, gamma, num_iter=100, outer_iter=100):
    # for a given gamma, choose the q(z|x) with the lowest J score
    q_z_xs = [_dib(p_x, p_y_x, Z, gamma, num_iter) for i in range(outer_iter)]
    iplanes = [information_plane_dib(p_x, p_y_x, q) for q in q_z_xs]
    J = np.array([i[1] - gamma * i[0] for i in iplanes]) # TODO: change this!
    #print(J[J.argmin()])
    return q_z_xs[J.argmin()]


def information_plane_dib(p_x, p_y_x, p_z_x):
    """ Given p(x), p(y|x), and p(z|x), calculate I[Y:Z] and H[Z] """
    p_xz = p_x[:, None] * p_z_x # Joint p(x,y), shape X x Y    
    p_xyz = p_x[:, None, None] * p_y_x[:, :, None] * p_z_x[:, None, :] # Joint p(x,y,z), shape X x Y x Z
    p_yz = p_xyz.sum(axis=0) # Joint p(y,z), shape Y x Z
    p_z = p_z_x.sum(axis = 0) # marginal p(z), shape Z
    h_z = scipy.special.xlogy(p_z, p_z).sum()
    return  mi(p_yz), h_z


def _dib(p_x, p_y_x, Z, gamma, num_iter=DEFAULT_NUM_ITER):
    """ Find encoder q(Z|X) to minimize J = I[X:Z] - gamma * I[Y:Z].
    
    Input:
    p_x : Distribution on X, of shape X. (meaning - M)
    p_y_x : Conditional distribution on Y given X, of shape X x Y.
    gamma : A non-negative scalar value. (world state - U) 
    Z : Support size of Z. (words - W; T in the DIB paper)

    Output: 
    Conditional distribution on Z given X, of shape X x Z. [words given meaning]

    """
    # Support size of X
    X = p_x.shape[-1]

    # Support size of Y
    Y = p_y_x.shape[-1]

    p_xy = p_y_x * p_x

    # step 1: initialize f0(x)
    f_x = np.random.randint(low = 0, high = Z, size = X) # shape X 
    
    while np.unique(f_x).shape[0] < Z:
        f_x = np.random.randint(low = 0, high = Z, size = X) # shape X 

    # step 2: initialize q0(z)
    q_z = np.array([p_x[np.where(f_x == z)].sum() for z in range(Z)]) # shape Z

    # step 3: initialize q(y|z)
    q_y_z = np.array([p_xy[np.where(f_x == z), :].sum(axis = 1).flatten() / p_x[np.where(f_x == z)].sum() for z in range(Z)]) # shape Z x Y

    # change the shape 
    p_xy = p_xy[:, None, :] # shape X x 1 x Y
    p_y_x = p_y_x[:, None, :] # shape X x 1 x Y
    q_y_z = q_y_z[None, :, :] # shape 1 x Z x Y

    for _ in range(num_iter):
        d_x_z = (
            scipy.special.xlogy(p_y_x, p_y_x)
            - scipy.special.xlogy(p_y_x, q_y_z)
        ).sum(axis = -1) # shape X x Z
        l_x_z = np.log2(q_z) - gamma * d_x_z # shape X x Z
        f_x = np.argmax(l_x_z, axis = 1) # shape X
        q_z = np.array([p_x[np.where(f_x == z)].sum() for z in range(Z)]) # shape Z
        q_y_z = np.array([p_xy[np.where(f_x == z), :].sum(axis = 1).flatten() / p_x[np.where(f_x == z)].sum() for z in range(Z)])
    
    q_z_x = np.zeros([X, Z])
    for i, j in enumerate(f_x):
            q_z_x[i,j] = 1
    return q_z_x
        

def dib_example():
    p_x = scipy.special.softmax(np.random.randn(9), -1)
    Y = 9
    X = p_x.shape[-1]
    p_y_x = scipy.special.softmax(np.random.randn(X, Y), -1)

    return dib(p_x, p_y_x, 6, 1.3)     

    # # Randomly initialize the conditional distribution q(z|x)
    # q_z_x = scipy.special.softmax(np.random.randn(X, Z), -1) # shape X x Z
    # p_y_x = p_y_x[:, None, :] # shape X x 1 x Y
    # p_x = p_x[:, None] # shape X x 1

    # # Blahut-Arimoto iteration to find the minimizing q(z|x)
    # for _ in range(num_iter):
    #     q_xz = p_x * q_z_x # Joint distribution q(x,z), shape X x Z
    #     q_z = q_xz.sum(axis=0, keepdims=True) # Marginal distribution q(z), shape 1 x Z
    #     q_y_z = ((q_xz / q_z)[:, :, None] * p_y_x).sum(axis=0, keepdims=True) # Conditional decoder distribution q(y|z), shape 1 x Z x Y
    #     d = ( 
    #         scipy.special.xlogy(p_y_x, p_y_x)
    #         - scipy.special.xlogy(p_y_x, q_y_z) # negative KL divergence -D[p(y|x) || q(y|z)]
    #     ).sum(axis=-1) # expected distortion over Y; shape X x Z
    #     q_z_x = scipy.special.softmax(np.log(q_z) - gamma*d, axis=-1) # Conditional encoder distribution q(z|x) = 1/Z q(z) e^{-gamma*d}

    # return q_z_x

