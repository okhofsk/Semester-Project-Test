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
        self.F = create_F(self.A)
        if proxg_str_ == None:
            self.is_simplex = True
            self.proxg = proxg_operator('simplex')
            self.J, self.J_complete = J_operator(self.A, 'simplex', None)
        elif isinstance(proxg_str_, str):
            self.proxg = proxg_operator(proxg_str_)
            if proxg_str_ == 'simplex':
                self.is_simplex = True
                self.J, self.J_complete = J_operator(self.A, proxg_str_, self.proxg)
            else :
                self.is_simplex = False
                self.J, self.J_complete = J_operator(self.A, 'norm', self.proxg)

## --------------------------------------------------------------------------------##

    def __str__(self):
        q0 = np.ones(self.A.shape[0] + self.A.shape[1])
        return "Matrix A: " + str(self.A) + "\n Matrix F: " + str(self.F) + "\n Proximal Operator: " + str(self.proxg(q0,0)) + "\n Simplex?: " + str(self.is_simplex) + "\n J Operator: " + str(self.J(q0)) + "\n"
        

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
            return Fx(self.A), self.J, self.J_complete, self.proxg
        else:
            return Fx(self.A), self.J, None, self.proxg

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
            return self.A, self.F, Fx(self.A), self.J, self.J_complete, self.proxg
        else: 
            return self.A, self.F, Fx(self.A), self.J, None,  self.proxg
        


    