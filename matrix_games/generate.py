# Import Libraries and Dependencies
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import numpy.linalg as LA
import networkx as nx
from random import uniform
from time import perf_counter
from scipy.sparse import random

def create_A(name_, size_="small"):
    """
    External function for matrix games library'
    
    Computes A matrix using stored entries:
    - "rock_paper_scissor"      : payoff matrix for the game rock, paper, scissor
    - "hide_and_seek"           : Hide and Seek Matrix
    - "normandy"                : Matrix from “Normandy: Game and Reality”[1]
    - "diag"                    : random diagonal game matrix; if custom size, 
                                   second dim will be considered to make nxn matrix
    - "triangular"              : random upper triangular matrix
    - "rand"                    : a random matrix
    - "rand_sparse"             : a random sparse matrix [2]
    
    Parameters
    ----------
    name_ : string
        refernce string of the matrix to be returned
    size_ : tuple/string 
        define shape of matrix only -----, default "small"
        
    Returns
    -------
    out : ndarray or a sparse matrix in csr format.[3]
    
    Raises
    ------
    ValueError: unknown string
        If name_ is not one of the stored strings and defined A matices.
        
    Notes
    -----
    
    References
    ----------
    .. [1] “Normandy: Game and Reality” by W. Drakert in Moves, No. 6 (1972)
    .. [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.random.html#scipy.sparse.random
    .. [3] https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix
    
    Examples
    --------
    """
    if isinstance(name_, str):
        if isinstance(size_, tuple) :
            if size_ is (0,0):
                raise ValueError('Input a tuple as the size of your A matrix. Please refer to documentation.') 
            else : 
                size = size_
        elif isinstance (size_, str) :
            if size_  is "large":
                size = (1000,1000)
            elif size_ is "small": 
                size = (10,10)
            else:
                raise ValueError('Input string size_ unknown. Please refer to documentation.') 
        else:
            raise ValueError('Input "size_" unknown input format. Please refer to the documentation.')
        if name_ == "rock_paper_scissor":
            return np.array([[0, -1, 1],[1, 0, -1],[-1, 1, 0]])            
        elif name_ == "normandy":
            return np.array([[13, 29, 8, 12, 16, 23],[18, 22, 21, 22, 29, 31],[18, 22, 31, 31, 27, 37],[11, 22, 12, 21, 21, 26],[18, 16, 19, 14, 19, 28],[23, 22, 19, 23, 30, 34]])
        elif name_ == "diag": 
            return np.diagflat(create_A_rand(1, size[1], False))
        elif name_ == "triangular":
            return np.triu(create_A_rand(size[0], size[1], False)) 
        elif name_ == "hide_and_seek":
            return np.random.randint(2, size=size)
        elif name_ == "one_plus":
            ones = np.ones((size[0], size[0]))
            return (-2)*ones + np.diag(np.diag(ones, k=1), k=1) + np.diag(np.diag(ones, k=-1), k=-1) + np.diag(np.diag(ones))*2
        elif name_ == "matching_pennies":
            return create_rand_pennies_mat(size[0])
        elif name_ == "rand":
            return create_A_rand(size[0], size[1],False)
        elif name_ == "rand_sparse":
            return create_A_rand(size[0], size[1],True)
        else:
            raise ValueError('Input string "name_" unknown. Please refer to the documentation.') 
    else:
        raise ValueError('Input "name_" unknown input format. Please refer to the documentation.')    

## --------------------------------------------------------------------------------##
  
def create_A_rand(n, m, sparsity):
    """
    External function for matrix games library'
    
    Computes A matrix with random entries and size (n,m)
    
    Parameters
    ----------
    n : int
        number of columns.
    m : int
        number of rows.
    sparsity : boolean
        if "True" it returns a random sparse matrix using the scipy.sparse
        library[1], otherwise it computes a full matrix with random
        entries from 0 to 1. 
        
    Returns
    -------
    out : ndarray or a sparse matrix in csr format.[2]
    
    Raises
    ------
        
    Notes
    -----
    
    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.random.html#scipy.sparse.random
    .. [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix
    
    Examples
    --------
    """
    if sparsity:
        return sp.sparse.random(n, m)
    else:
        return np.random.random((n, m))

## --------------------------------------------------------------------------------##

def create_F(A_):
    """
    External function for matrix games library'
    
    Computes the F operator explicitly for the matrix games problem which is
    given in a matrix form by F = [[0, A^T];[-A, 0]] where 0 indicates a zero
    matrix of the appropriate size (for A a nxm matrix we have a mxm matrix 
    in the top left corner and a nxn matrix on the bottom right corner.
    
    Parameters
    ----------
    A_ : ndarray/sparse.csr[1]
        the matrix A from the matrix games problem in sparse or ndarray form
        
    Returns
    -------
    out : ndarray 
        the operator F in matrix form as given by theory
    
    Raises
    ------
        
    Notes
    -----
    
    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.random.html#scipy.sparse.random
    
    Examples
    --------
    """
    if sp.sparse.issparse(A_):
        mat_a = np.copy(a.toarray())
    else:
        mat_a = np.copy(A_)
    zero_matrix_top = np.zeros((mat_a.shape[1], mat_a.shape[1]))
    zero_matrix_bot = np.zeros((mat_a.shape[0], mat_a.shape[0]))
    left_F = np.concatenate((zero_matrix_top, -mat_a))
    right_F = np.concatenate((mat_a.transpose(), zero_matrix_bot))
    return np.concatenate((left_F, right_F), axis=1)

## --------------------------------------------------------------------------------##

def create_F_rand(n_, m_, sparsity_):
    """
    External function for matrix games library'
    
    Computes the F operator explicitly for the matrix games problem using a matrix A
    with random variables of size nxm which is given in a matrix form by 
    F = [[0, A^T];[-A, 0]] where 0 indicates a zero matrix of the appropriate size 
    (for A a nxm matrix we have a mxm matrix in the top left corner and a nxn matrix 
    on the bottom right corner.
    
    Parameters
    ----------
    n : int
        number of columns.
    m : int
        number of rows.
    sparsity : boolean
        if "True" it returns a random sparse matrix using the scipy.sparse
        library[1], otherwise it computes a full matrix with random
        entries from 0 to 1. 
        
    Returns
    -------
    out : ndarray 
        the operator F in matrix form as given by theory
    
    Raises
    ------
        
    Notes
    -----
    
    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.random.html#scipy.sparse.random
    .. [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix
    
    Examples
    --------
    """
    if sparsity_:
        mat_a = create_A_rand(n_,m_,sparsity).toarray()
    else:
        mat_a = create_A_rand(n_,m_,sparsity)
    return create_F(mat_a)

## --------------------------------------------------------------------------------##

def Fx_product(A_, x_):
    """
    External function for matrix games library'
    
    Computes the Fx product using the matrix a and the vector x. By using the nature of the 
    F matrix we can make the operation more efficient since we only have to multiply the a
    matrix part with part of the vector x and not the whole F matrix. 
    
    Parameters
    ----------
    A_ : ndarray/sparse.csr[1]
        the matrix A from the matrix games problem in sparse or ndarray form
    x_ : ndarray
        the vector x 
        
    Returns
    -------
    out : ndarray 
        the product Fx as a vector
    
    Raises
    ------
        
    Notes
    -----
    
    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix
    
    Examples
    --------
    """
    x = np.reshape(x_, (len(x_),1))
    (dimN, dimM) = A_.shape
    x_top = A_.T@x[dimM:]
    x_bot = -A_@x[:dimM]
    self.F_temp = np.append(x_top, x_bot)
    return np.append(x_top, x_bot)

## --------------------------------------------------------------------------------##

def Fx(A_):
    """
    External function for matrix games library'
    
    Computes the Fx operator using the matrix a. By using the function "Fx_product".
    
    Parameters
    ----------
    A_ : ndarray/sparse.csr[1]
        the matrix A from the matrix games problem in sparse or ndarray form
        
    Returns
    -------
    out : function 
        a function that descibes the Fx operator 
    
    Raises
    ------
        
    Notes
    -----
    
    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix
    
    Examples
    --------
    """
    def F(q):
        return Fx_product(A_, q)
    return F
    
## --------------------------------------------------------------------------------##

def J_operator(A_, name_, prox_g_): 
    """
    External function for matrix games library'

    Computes the J operator

    Parameters
    ----------
    A_ : ndarray/sparse.csr[1]
        the matrix A from the matrix games problem in sparse or ndarray form
    name_ : string
        the name of the J operator to use
    prox_g_ : function
        the proximal operator function to use

    Returns
    -------
    out : function 
        a function that descibes the J operator 

    Raises
    ------

    Notes
    -----

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix

    Examples
    --------
    """
    dimN, dimM = A_.shape
    if name_ == 'simplex':
        def J(q):
            Fx = Fx_product(A_, q)
            ATy = Fx[:dimM]
            Ax = -Fx[dimM:]
            return np.amax(Ax) - np.amin(ATy) 
        def J_complete(q, ax, ay):
            return np.amax(ax) - np.amin(ay)
        return J, J_complete
    else :
        def J(q):
            F = Fx(A_)
            return LA.norm(q - prox_g_(q - F(q), 1))
    return J, None

## --------------------------------------------------------------------------------##

def proxg_operator(name_, A_):
    """
    External function for matrix games library'
    
    Computes the proxima operator. There are __ predefined operators available:
        - "simplex" : uses the simplex projection see the function 'projsplx'
        - "fmax"         : returns the (q)+ version of the vector 
        - "none"         : just returns q itself
    
    Parameters
    ----------
    name_ : string
        the name of the proximal operator to use
        
    Returns
    -------
    out : function 
        a function that descibes the prox_g operator 
    
    Raises
    ------
    ValueError: unknown string
        If name_ is not one of the stored strings and defined proxg_operators.   
        
    Notes
    -----
    
    References
    ----------
    
    Examples
    --------
    """
    dimN, dimM = A_.shape
    if name_ == "simplex":
        def prox_g(q, eps): 
            x = q[:dimM]
            y = q[dimM:]
            return np.concatenate((projsplx(x),projsplx(y)))
    elif  name_ == "fmax":
        def prox_g(q, eps):
            return np.fmax(q,0)
    elif  name_ == "none":
        def prox_g(q, eps):
            return q 
    else:
        raise ValueError('Input string unknown. Please refer to documentation.') 
    return prox_g

## --------------------------------------------------------------------------------##

def projsplx(v, s=1):
    """
    External function for matrix games library'
    
    Computes the projection onto a simplex using the algorithm described here[1].
    
    Parameters
    ----------
    q : ndarray
        the input vector 
        
    Returns
    -------
    out : ndarray 
        the input vector projected onto a simplex 
    
    Raises
    ------
        
    Notes
    -----
    
    References
    ----------
    .. [1] https://arxiv.org/abs/1101.6081
    
    Examples
    --------
    n = len(q_)
    q_sorted = np.sort(q_)
    t = -1
    zeros = np.zeros(n)
    for i in reversed(range(n-1)):
        t = np.sum(q_[i+1:]-1)/(n-i)
        if t >= q_[i]:
            return np.fmax(q_-t,zeros)
    return np.fmax(q_-np.sum(q_-1)/n,zeros) 
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


def create_rand_pennies_mat(size0_):
    G = nx.fast_gnp_random_graph(size0_, 1.0, 78456, True)
    W = np.zeros(size0_)
    u0 = -1
    for (u, v) in G.edges():
        wij = np.random.randint(0,10)
        G.edges[u,v]['weight'] = wij
        if (u == u0):
            W[u] += wij
        else :
            u0 = u
            W[u] = wij


    a = np.zeros((size0_,size0_))
    for (u,v) in G.edges():
        a[u,v] = G.edges[u,v]['weight']

    for i in range(size0_):
        wii = np.random.randint(0,10)
        a[i,i] = wii - W[i]
    return a
