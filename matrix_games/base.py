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
    def __init__(self, A=None, size=None, proxg_str=None):
        """
        Initialization of mg_problem class that containts:
        - A :       A matrix describing the problem
        - J :       J matrix describing the way to classifiy the distance to optimum
        - proxg :   proximal operator

        Parameters
        ----------
        A : ndarray/string
            the matrix A for the MG problem. If "None" gives a random 10x10 matrix
            default "None". Can load a type one of the following matrices using the string:
            - "rock_paper_scissor"      : payoff matrix for the game rock, paper, scissor
            - "marriage_problem_small"  : Hall's marriage problem as a 10x10 matrix
            - "normandy"                : Matrix from “Normandy: Game and Reality”[1]
            - "diag"                    : random diagonal game matrix
                           second dim will be considered to make nxn matrix
            - "triangular"              : random upper triangular matrix
            - "rand"                    : a random 10x10 matrix
            - "rand_sparse"             : a random 10x10 sparse matrix [2]
            Or supply own matrix A 
        
        size : tulpe/string
            either define size of the A matrix as a tulpe or use string "small" for
            a 10x10 matrix size or "large" to have a matrix size of 1000x1000 

        prox_str : string
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
        if A == None:
            self.A = create_A('rand')
        elif isinstance(A, str):
            if isinstance(size, str):
                if size == "large":
                    self.A = create_A(A, True)
                elif size == "small": 
                    self.A = create_A(A, False)
                else :
                    raise ValueError('Input string size unknown. Please refer to documentation.') 
            self.A = create_A(A, False, True, size)
        else:
            self.A = A
        self.F = create_F(self.A)
        if proxg_str == None:
            self.proxg = proxg_operator('simplex')
            self.J = J_operator(self.A, 'simplex', None)
        else:
            self.proxg = proxg_operator(proxg_str)
            if proxg_str == 'simplex':
                self.J = J_operator(self.A, proxg_str, None)
            else :
                self.J = J_operator(self.A, 'norm', self.proxg)

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
        return Fx(self.A), self.J, self.proxg

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
        return self.A, self.F, Fx(self.A), self.J, self.proxg
    
## --------------------------------------------------------------------------------##

    def J_operator(self, _name, prox_g): 
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


    