# Import Libraries and Dependencies
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import numpy.linalg as LA
from random import uniform
from time import perf_counter
from scipy.sparse import random

def create_A(_name, _large=False, _custom=False, _tulpe=(0,0)):
    """
    External function for matrix games library'
    
    Computes A matrix using stored entries:
    - "rock_paper_scissor"      : payoff matrix for the game rock, paper, scissor
    - "marriage_problem_small"  : Hall's marriage problem as a 10x10 matrix
    - "normandy"                : Matrix from “Normandy: Game and Reality”[1]
    - "diag"                    : random diagonal game matrix; if custom size, 
                           second dim will be considered to make nxn matrix
    - "triangular"              : random upper triangular matrix
    - "rand"                    : a random 10x10 matrix
    - "rand_sparse"             : a random 10x10 sparse matrix [2]
    
    Parameters
    ----------
    _name : string
        refernce string of the matrix to be returned
    _large : string
        either small (10x10) or large (1000x1000) matrix, default False
    _custom : string
        define own matrix shape set to true and add the shape as a tulpe, default False
    _tulpe : tulpe 
        define shape of matrix only if _custom is True, default (0,0)
        
    Returns
    -------
    out : ndarray or a sparse matrix in csr format.[3]
    
    Raises
    ------
    ValueError: unknown string
        If _name is not one of the stored strings and defined A matices.
        
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
    if _custom == True :
        if _tulpe == (0,0):
            raise ValueError('Input a tulpe as the size of your A matrix. Please refer to documentation.') 
        else : 
            if _name is "rock_paper_scissor":
                return np.array([[0, -1, 1],[1, 0, -1],[-1, 1, 0]])
            elif _name is "normandy":
                return np.array([[13, 29, 8, 12, 16, 23],[18, 22, 21, 22, 29, 31],[18, 22, 31, 31, 27, 37],[11, 22, 12, 21, 21, 26],[18, 16, 19, 14, 19, 28],[23, 22, 19, 23, 30, 34]])
            elif _name is "diag": 
                return np.diagflat(create_A_rand(1, _tulpe[1], False))
            elif _name is "triangular":
                return np.triu(create_A_rand(_tulpe[0], _tulpe[1], False)) 
            elif _name is "marriage_problem":
                return np.random.randint(2, size=_tulpe)
            elif _name is "rand":
                return create_A_rand(_tulpe[0], _tulpe[1],False)
            elif _name is "rand_sparse":
                return create_A_rand(_tulpe[0], _tulpe[1],True)
            else:
                raise ValueError('Input string unknown. Please refer to documentation.') 
    else :
        if _large == True:
            tulpe = (1000,1000)
        else : 
            tulpe = (10,10)
        if _name is "rock_paper_scissor":
            return np.array([[0, -1, 1],[1, 0, -1],[-1, 1, 0]])
        elif _name is "normandy":
            return np.array([[13, 29, 8, 12, 16, 23],[18, 22, 21, 22, 29, 31],[18, 22, 31, 31, 27, 37],[11, 22, 12, 21, 21, 26],[18, 16, 19, 14, 19, 28],[23, 22, 19, 23, 30, 34]])
        elif _name is "diag": 
            return np.diag(create_A_rand(1, tulpe[1], False))
        elif _name is "triangular":
            return np.triu(create_A_rand(tulpe[0], tulpe[1], False)) 
        elif _name is "marriage_problem":
            return np.random.randint(2, size=tulpe)
        elif _name is "rand":
            return create_A_rand(tulpe[0], tulpe[1],False)
        elif _name is "rand_sparse":
            return create_A_rand(tulpe[0], tulpe[1],True)
        else:
            raise ValueError('Input string unknown. Please refer to documentation.') 
    

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

def create_F(a):
    """
    External function for matrix games library'
    
    Computes the F operator explicitly for the matrix games problem which is
    given in a matrix form by F = [[0, A^T];[-A, 0]] where 0 indicates a zero
    matrix of the appropriate size (for A a nxm matrix we have a mxm matrix 
    in the top left corner and a nxn matrix on the bottom right corner.
    
    Parameters
    ----------
    a : ndarray/sparse.csr[1]
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
    if sp.sparse.issparse(a):
        mat_a = np.copy(a.toarray())
    else:
        mat_a = np.copy(a)
    zero_matrix_top = np.zeros((mat_a.shape[1], mat_a.shape[1]))
    zero_matrix_bot = np.zeros((mat_a.shape[0], mat_a.shape[0]))
    left_F = np.concatenate((zero_matrix_top, -mat_a))
    right_F = np.concatenate((mat_a.transpose(), zero_matrix_bot))
    return np.concatenate((left_F, right_F), axis=1)

## --------------------------------------------------------------------------------##

def create_F_rand(n, m, sparsity):
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
    if sparsity:
        mat_a = create_A_rand(n,m,sparsity).toarray()
    else:
        mat_a = create_A_rand(n,m,sparsity)
    return create_F(mat_a)

## --------------------------------------------------------------------------------##

def Fx_product(a, x):
    """
    External function for matrix games library'
    
    Computes the Fx product using the matrix a and the vector x. By using the nature of the 
    F matrix we can make the operation more efficient since we only have to multiply the a
    matrix part with part of the vector x and not the whole F matrix. 
    
    Parameters
    ----------
    a : ndarray/sparse.csr[1]
        the matrix A from the matrix games problem in sparse or ndarray form
    x : ndarray
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
    x_ = np.reshape(x, (len(x),1))
    (dimN, dimM) = a.shape
    x_top = a.T@x_[dimM:dimN+dimM]
    x_bot = -a@x_[:dimM]
    return np.append(x_top, x_bot)

## --------------------------------------------------------------------------------##

def Fx(a):
    """
    External function for matrix games library'
    
    Computes the Fx operator using the matrix a. By using the function "Fx_product".
    
    Parameters
    ----------
    a : ndarray/sparse.csr[1]
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
        return Fx_product(a, q)
    return F

## --------------------------------------------------------------------------------##

def J_operator(a, _name, prox_g): 
    """
    External function for matrix games library'
    
    Computes the J operator
    
    Parameters
    ----------
    a : ndarray/sparse.csr[1]
        the matrix A from the matrix games problem in sparse or ndarray form
    _name : string
        the name of the J operator to use
    prox_g : function
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
    if sp.sparse.issparse(a):
        _A = a.toarray()
    else :
        _A = a
    
    (dimN, dimM) = _A.shape;
    if _name == 'simplex':
        def J(q):
            Fx = Fx_product(_A, q)
            Ax = Fx[:dimM]
            ATx = Fx[dimM:dimN+dimM]
            return np.amax(Ax) - np.amin(ATx)
    else :
        def J(q):
            return LA.norm(q - prox_g(q - Fx(q), 1))
    return J

## --------------------------------------------------------------------------------##

def proxg_operator(_name):
    """
    External function for matrix games library'
    
    Computes the proxima operator. There are __ predefined operators available:
        - "simplex" : uses the simplex projection see the function 'projsplx'
        - "fmax"         : returns the (q)+ version of the vector 
        - "none"         : just returns q itself
    
    Parameters
    ----------
    _name : string
        the name of the proximal operator to use
        
    Returns
    -------
    out : function 
        a function that descibes the prox_g operator 
    
    Raises
    ------
    ValueError: unknown string
        If _name is not one of the stored strings and defined proxg_operators.   
        
    Notes
    -----
    
    References
    ----------
    
    Examples
    --------
    """
    if _name is "simplex":
        def _prox_g(q, eps): 
            return projsplx(q)
    elif  _name is "fmax":
        def _prox_g(q, eps):
            return np.fmax(q,0)
    elif  _name is "none":
        def _prox_g(q, eps):
            return q 
    else:
        raise ValueError('Input string unknown. Please refer to documentation.') 
    return _prox_g

## --------------------------------------------------------------------------------##

def projsplx(q):
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
    """
    _n = len(q)
    q_sorted = np.sort(q)
    _t = -1
    _zeros = np.zeros(_n)
    for i in reversed(range(len(q))):
        _t = np.sum(q[i+1:]-1)/(_n-i)
        if _t >= q[i]:
            return np.fmax(q-_t,_zeros)
    return np.fmax(q-np.sum(q-1)/_n,_zeros)  
