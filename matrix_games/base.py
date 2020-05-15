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
    def __init__(self, A=None, proxg_str=None, size=None):
        """
        Initialization of mg_problem class that containts:
        - A :       A matrix describing the problem
        - J :       J matrix describing the way to classifiy the distance to optimum
        - proxg :   proximal operator

        Parameters
        ----------
        A : ndarray
        
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
        if A is None:
            self.A = create_A('rand')
        #elif isinstance(A, str):
        #    if isinstance(size, str):
        #        if size == "large":
        #            self.A = create_A(A, True)
        #        elif size == "small": 
        #            self.A = create_A(A, False)
        #        else :
        #            raise ValueError('Input string size unknown. Please refer to documentation.') 
        #    else:
        #        self.A = create_A(A, False, True, size)
        else:
            self.A = A
        self.F = create_F(self.A)
        if proxg_str == None:
            self.is_simplex = True
            self.proxg = proxg_operator('simplex')
            self.J, self.J_complete = J_operator(self.A, 'simplex', None)
        else:
            self.proxg = proxg_operator(proxg_str)
            if proxg_str == 'simplex':
                self.is_simplex = True
                self.J, self.J_complete = J_operator(self.A, proxg_str, None)
            else :
                self.is_simplex = False
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
        if self.is_simplex:
            return Fx(self.A), self.J, self.J_complete, self.proxg
        else:
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
        if self.is_simplex:
            return self.A, self.F, Fx(self.A), self.J, self.J_complete, self.proxg
        else: 
            return self.A, self.F, Fx(self.A), self.J, self.proxg
        


    