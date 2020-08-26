import numpy as np
import scipy.special

DEFAULT_NUM_ITER = 10

def ib(p_x, p_y_x, Z, gamma, num_iter=DEFAULT_NUM_ITER):
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
    q_z_x = scipy.special.softmax(np.random.randn(X, Z), -1) # shape X x Z
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
        q_z_x = scipy.special.softmax(np.log(q_z) - gamma*d, axis=-1) # Conditional encoder distribution q(z|x) = 1/Z q(z) e^{-gamma*d}

    return q_z_x

def mi(p_xy):
    """ Calculate mutual information of a distribution P(x,y) 

    Input: 
    p_xy: An X x Y array giving p(x,y)
    
    Output:
    The mutual information I[X:Y], a nonnegative scalar,
    """
    p_x = p_xy.sum(axis=-1, keepdims=True)
    p_y = p_xy.sum(axis=-2, keepdims=True)
    return scipy.special.xlogy(p_xy, p_xy).sum() - scipy.special.xlogy(p_x, p_x).sum() - scipy.special.xlogy(p_y, p_y).sum()

def zipf_mandelbrot(N, s, q=0):
    """ Return a Zipf-Mandelbrot distribution over N items """
    k = np.arange(N) + 1
    p = 1/(k+q)**s
    Z = p.sum()
    return p/Z

def test_mi():
    p_xy = np.array([[1,0],[0,1]])/2
    assert mi(p_xy) == np.log(2)

    p_x = scipy.special.softmax(np.random.randn(5))
    p_y = scipy.special.softmax(np.random.randn(5))
    p_xy = p_x[:, None] * p_y[None, :]
    assert mi(p_xy) == 0

def test_ib():
    # Suppose we have three Xs, named x1 x2 and x3, and two Y's y1 and y2,
    # And we have p(y1|x1)=1, p(y2|x2)=1, p(y2|x3)=1,
    # And we want to find a lossy encoding q(z|x) of x that minimizes the KL divergence
    # D[ p(y|x) || q(y|z) ]
    # Then if we have two values of Z, z1 and z2, the solution is to map x1 to z1 and {x2,x3} to z2
    # or to map x1 to z2 and {x2,x3} to z1.
    p_x = np.ones(3)/3
    p_y_x = np.array([
        [1,0],  # p(y|x1)
        [0,1],  # p(y|x2)
        [0,1],  # p(y|x3)
    ])
    q_z_x = ib(p_x, p_y_x, 2, 2, num_iter=100)
    assert (
        (np.round(q_z_x, 1) == np.round(np.array([[1,0],[0,1],[0,1]]))).all()
        or (np.round(q_z_x, 1) == np.round(np.array([[0,1],[1,0],[1,0]]))).all()
    )

    # Now suppose |Z|=3, what do we get?
    
if __name__ == '__main__':
    import nose
    nose.runmodule()
