import numpy as np
import matplotlib.pyplot as plt
from random import uniform
from time import perf_counter
import scipy as sp
from scipy import linalg
from scipy.sparse import random
import numpy.linalg as LA

import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad    # The only autograd function you may ever need

from constrained_optimization.generate import*


class co_problem:
    """
    Class definition of constrained optimization library
    """
    def __init__(self, fx_, hxs_=None, proj_=None, gradf_=None, gradhs_=None, A_=None, b_=None): #
        """
        Initialization of qcqp_problem class that containts:

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
            
        Returns
        -------
        out : qcqp_problem object

        Notes
        -----

        References
        ----------

        Examples
        --------
        """
        ### Check "F" ###
        if isinstance(fx_, str):
            if fx_ is "name1":
                #stuff
                x = 0
        elif callable(fx_):
            self.fx = fx_                                   #var
            ### Check "H" ###
            if hxs_ is None:
                self.hx = None
                self.gradh = None
            elif isinstance(hxs_, list):
                for i in range(len(hxs_)):
                    if not callable(hxs_[i]):
                        raise ValueError('"hx"; Input of wrong format. Please refer to the documentation.') 
                self.hx = hxs_                               #var
                self.M = len(hxs_)                           #var
                ### Check "gradh" ###
                if gradhs_ is None:
                    self.gradh = []
                    for i in range(len(self.hx)):
                        gradhi = grad(self.hx[i])
                        self.gradh.append(gradhi)
                elif isinstance(gradhs_, list):
                    for i in range(len(gradhs_)):
                        if not callable(gradhs_[i]):
                            raise ValueError('"gradh"; Input of wrong format. Please refer to the documentation.') 
                    self.gradh = gradhs_                         #var
                else :
                    raise ValueError('"gradh"; Input of wrong format. Please refer to the documentation.')
            else :
                raise ValueError('"hx"; Input of wrong format. Please refer to the documentation.')
            ### Check "gradf" ###
            if gradf_ is None:
                self.gradf = grad(self.fx)
            elif callable(gradf_):
                self.gradf = gradf_                          #var
            else:
                raise ValueError('"gradf"; Input of wrong format. Please refer to the documentation.')
            ### Check "A" ###
            if A_ is None:
                self.A = None                            
            elif isinstance(A_, (np.ndarray, np.generic)):
                self.A = A_                                  #var
                if isinstance(b_, (np.ndarray, np.generic)):
                    self.b = b_                                  #var
                elif b_ is None:
                    raise ValueError('"b": Cannot input matrix A without vector b. Please refer to the documentation.')
                else:
                    raise ValueError('"b"; Input of wrong format. Please refer to the documentation.')
            else:
                raise ValueError('"A"; Input of wrong format. Please refer to the documentation.')
        else:
            raise ValueError('"fx"; Input of wrong format. Please refer to the documentation.') 
        if proj_ is None:
            self.proj = None
        elif callable(proj_):
            self.proj = proj_
        else:
            raise ValueError('"proj"; Input of wrong format. Please refer to the documentation.') 
            
        if self.A is None:
            self.F = create_F(self.gradh, self.hx, self.gradh)
            self.J = create_J(self.gradh, self.hx, self.gradh, self.proj)
        else:
            self.F = create_F(self.gradh, self.hx, self.gradh, self.A, self.b)
            self.J = create_J(self.gradh, self.hx, self.gradh, self.proj, self.A, self.b)

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

        Notes
        -----

        References
        ----------

        Examples
        --------
        """
        return self.F, self.J

      