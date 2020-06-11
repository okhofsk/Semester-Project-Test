# Import Libraries and Dependencies
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import numpy.linalg as LA
from random import uniform
from time import perf_counter
from scipy.sparse import random

from matrix_games.generate import*

class mg_problem:
    """
    Class definition of matrix games library
    """
    def __init__(self, A_=None, proxg_str_=None, size_=None):
        """
        Initialization of mg_problem class that containts:
        - A :       A matrix describing the problem
        - J :       J matrix describing the way to classifiy the distance to optimum
        - proxg :   proximal operator

        Parameters
        ----------
        A_ : ndarray/string
            either directly give the matrix A as a numpy array or use a string to 
            choose one of the predefined matricies. For more information see the 
            documenation on "create_A()".
        
        size_ : tuple/string
            either define size of the A matrix as a tuple or use string "small" for
            a 10x10 matrix size or "large" to have a matrix size of 1000x1000 

        prox_str_ : string
            string describing the proximal operator wanted. If "None" gives simplex
            Possibilities are:
            - "simplex"      : uses the simplex projection see the function 'projsplx'
            - "fmax"         : returns the (q)+ version of the vector 
            - "none"         : just returns q itself
            default "None"

        Returns
        -------
        out : mg_problem object

        Notes
        -----

        References
        ----------

        Examples
        --------
        """
        
        if isinstance(A_, (np.ndarray, np.generic)) or isinstance(A_, sp.sparse.coo_matrix):
            self.A = A_
        elif isinstance(A_, str):
            if isinstance(size_, str) or isinstance(size_, tuple):
                self.A = create_A(A_, size_)
            else :
                self.A = create_A(A_)
        else:
            self.A = create_A('rand')
        (self.dimN, self.dimM) = self.A.shape
        self.F = intern_Fx(self)
        self.Fx = None
        if proxg_str_ == None:
            self.is_simplex = True
            self.proxg = proxg_operator('simplex')
            self.J, self.J_complete = J_operator(self.A, 'simplex', None)
        elif isinstance(proxg_str_, str):
            self.name = proxg_str_
            self.proxg = intern_proxg(self)
            if proxg_str_ == 'simplex':
                self.is_simplex = True
                self.J, self.J_complete = intern_J(self)
            else :
                self.is_simplex = False
                self.J, self.J_complete = intern_J(self)

## --------------------------------------------------------------------------------##

    def __str__(self):
        q0 = np.ones(self.A.shape[0] + self.A.shape[1])
        return "Matrix A: " + str(self.A) + "\n Matrix F: " + str(create_F(self.A)) + "\n Proximal Operator: " + str(self.proxg(q0,0)) + "\n Simplex?: " + str(self.is_simplex) + "\n J Operator: " + str(self.J(q0)) + "\n"
        

## --------------------------------------------------------------------------------##

    def get_parameters(self):
        """
        returns all parameters necessary to test solvers

        Parameters
        ----------

        Returns
        -------
        out :
            - Fx :      function that takes x vector and return Fx product
            - J :       J matrix describing the way to classifiy the distance to optimum
            - proxg :   proximal operator

        Notes
        -----

        References
        ----------

        Examples
        --------
        """
        if self.is_simplex:
            return self.F, self.J, self.J_complete, self.proxg
        else:
            return self.F, self.J, None, self.proxg

## --------------------------------------------------------------------------------##

    def get_all(self):
        """
        returns all parameters 

        Parameters
        ----------

        Returns
        -------
        out :
            - A :       Matrix games matrix (ndarray)
            - F :       Using A matrix construct F matrix for VI
            - Fx :      function that takes x vector and return Fx product
            - J :       J matrix describing the way to classifiy the distance to optimum
            - proxg :   proximal operator

        Notes
        -----

        References
        ----------

        Examples
        --------
        """
        if self.is_simplex:
            return self.A, self.F, create_F(self.A), self.J, self.J_complete, self.proxg
        else: 
            return self.A, self.F, create_F(self.A), self.J, None,  self.proxg

## --------------------------------------------------------------------------------##

def intern_Fx(self):
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
    def operator(q):
        q = np.reshape(q, (len(q),1))
        q_top = self.A.T@q[self.dimM:]
        q_bot = -self.A@q[:self.dimM]
        self.Fx = np.append(q_top, q_bot)
        return np.append(q_top, q_bot)
    return operator
    
## --------------------------------------------------------------------------------##

def intern_J(self): 
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
    if self.name == 'simplex':
        def J(q):
            if self.Fx is None:
                Fx = self.F(q)
            else:
                Fx = self.Fx
            ATy = Fx[:self.dimM]
            Ax = -Fx[self.dimM:]
            return np.amax(Ax) - np.amin(ATy) 
        def J_complete(q, ax, ay):
            return np.amax(ax) - np.amin(ay)
        return J, J_complete
    else :
        def J(q):
            if self.Fx is None:
                Fx = self.F(q)
            else:
                Fx = self.Fx
            return LA.norm(q - prox_g_(q - Fx, 1))
    return J, None

## --------------------------------------------------------------------------------##

def intern_proxg(self):
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
    if self.name == "simplex":
        def prox_g(q, eps): 
            x = q[:self.dimM]
            y = q[self.dimM:]
            return np.concatenate((projsplx(x),projsplx(y)))
    elif  self.name == "fmax":
        def prox_g(q, eps):
            return np.fmax(q,0)
    elif  self.name == "none":
        def prox_g(q, eps):
            return q 
    else:
        raise ValueError('Input string unknown. Please refer to documentation.') 
    return prox_g


        


    