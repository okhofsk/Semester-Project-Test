import numpy as np
import time

## --------------------------------------------------------------------------------##

def create_randMat(name_, size_):
    """
    
    A function that creates a random matrix of a defined size 
    usind different random distributions.
    
    
    Parameters
    ----------
    name_: string
       name of random operator to be used:
           - "uniform" : uniform distirbution from [0,1]
           - "minusuniform" : uniform distirbution from [-1,1]
           - "plusuniform" : uniform distirbution from [-10,10]
       
    size_: tuple
       shape of matrix 
    
    Raises
    ------
    ValueError: if the name specified is not known
        
    Returns
    -------
    out: ndarray
        the matrix
    """
    if name_ == "uniform":
        return np.random.random(size_)
    elif name_ == "minusuniform":
        return np.random.random(size_)*2-1
    if name_ == "plusuniform":
        return (np.random.random(size_)*2-1)*10
    else:
        raise ValueError('name_ not known. Please refer to documentation.') 

## --------------------------------------------------------------------------------##

def get_projector(name_):
    """
    
    A function that fetches a proximal operator
    
    
    Parameters
    ----------
    name_: string
       name of projector wanted:
           -"simplex": simplex using projsplx() funciton
           -"l1ball": euclidean projection onto l1 ball using projsplx() funciton
           -"rplus": projection onto positive orthant
           -else : projection onto real space
        
    Returns
    -------
    out: function
        the projection operator
    """ 
    if name_ == "simplex":
        return projsplx
    elif name_ == "l1ball":
        return euclidean_proj_l1ball
    elif name_ == "rplus":
        operator = lambda q: np.fmax(q,eps)
        return operator
    else:
        operator = lambda q: q
        return operator

## --------------------------------------------------------------------------------##

def projsplx(v, s=1):
    """
    
    Computes the projection onto a simplex using the algorithm described here[1].
    
    
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
        
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    
    Raises
    ------
        
    Notes
    -----
    
    References
    ----------
    .. [1] https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246
    
    Examples
    --------
    """ 
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w

## --------------------------------------------------------------------------------##

def euclidean_proj_l1ball(v, s=1):
    """ 
    Compute the Euclidean projection on a L1-ball
        
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
       
    s: int, optional, default: 1,
       radius of the L1-ball
       
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s
    
    References
    ----------
    .. [1] https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246
       
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = projsplx(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w