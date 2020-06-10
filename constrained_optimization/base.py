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
    def __init__(self, fx_, N_, hxs_=None, proj_=None, gradf_=None, gradhs_=None, A_=None, b_=None): #
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
        self.U = None
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
            elif isinstance(hxs_, dict):
                for i in range(1,len(hxs_)+1):
                    if not callable(hxs_[i]):
                        raise ValueError('"hx"; Input of wrong format. Please refer to the documentation.') 
                self.hx = hxs_                               #var
                self.M = len(hxs_)                           #var
                ### Check "gradh" ###
                if gradhs_ is None:
                    self.gradh = []
                    for i in range(1,len(self.hx)+1):
                        gradhi = grad(self.hx[i])
                        self.gradh.append(gradhi)
                elif isinstance(gradhs_, dict):
                    for i in range(1,len(gradhs_)+1):
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
                    self.U = len(b_)
                elif b_ is None:
                    raise ValueError('"b": Cannot input matrix A without vector b. Please refer to the documentation.')
                else:
                    raise ValueError('"b"; Input of wrong format. Please refer to the documentation.')
            else:
                raise ValueError('"A"; Input of wrong format. Please refer to the documentation.')
        else:
            raise ValueError('"fx"; Necessary Input. Please refer to the documentation.') 
        if N_ is None:
            raise ValueError('"N"; Necessary Input. Please refer to the documentation.') 
        elif isinstance(N_, int):
            self.N = N_
        else:
            raise ValueError('"N"; Input of wrong format. Please refer to the documentation.')
            
            
        if proj_ is None:
            self.proj = None
        elif callable(proj_):
            self.proj = proj_
        else:
            raise ValueError('"proj"; Input of wrong format. Please refer to the documentation.') 
            
        if self.A is None:
            self.F = create_F(self.gradf, self.hx, self.gradh)
            self.J = create_J_int(self)
            #self.J = create_J(self.gradf, self.hx, self.gradh, self.proj)
        else:
            self.F = create_F(self.gradf, self.hx, self.gradh, self.A, self.b)
            self.J = create_J_int(self)
            #self.J = create_J(self.gradf, self.hx, self.gradh, self.proj, self.A, self.b)
            
        self.Fz = set_z(self, self.F)
        self.Jz = set_z(self, self.J)
        self.prox = proximal(self)

## --------------------------------------------------------------------------------##

    def __str__(self):
        q0 = np.ones(self.A.shape[0] + self.A.shape[1])
        return "string of class to be added"

## --------------------------------------------------------------------------------##

    def get_parameters(self, separate_):
        """
        returns all parameters necessary to test solvers

        Parameters
        ----------
            - separate_: boolean that checks if input vector z is in parts or together

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
        if separate_:
            return self.F, self.J
        else :
            return self.Fz, self.Jz
    
## --------------------------------------------------------------------------------##

def set_z(self, operator_):
    nbr_params = len(signature(operator_).parameters)
    if nbr_params is 2:
        def oper(z):
            if len(z) == self.N+self.M:
                x = z[0:self.N]
                y = z[self.N:self.N+self.M]
                fx, fy, _ = operator_(x,y)
                if fy is None:
                    return fx
                return np.concatenate((fx,fy))
            else:
                raise ValueError('set_z(operator_); len of z not correct. Please refer to the documentation.')  
        return oper
    elif nbr_params is 3:
        def oper(z):
            if len(z) == self.N+self.M+self.U:
                x = z[0:self.N]
                y = z[self.N:self.N+self.M]
                u = z[self.N+self.M:self.N+self.M+self.U]
                fx, fy, fu = operator_(x,y,u)
                if fy is None:
                    return fx
                return np.concatenate((fx,fy,fu))
            else:
                raise ValueError('set_z(operator_); len of z not correct. Please refer to the documentation.') 
        return oper
    else : 
        raise ValueError('set_z(operator_); operator can only have either 2 or 3 inputs. Please refer to the documentation.') 
    
## --------------------------------------------------------------------------------##

def proximal(self):
    if self.A is None:
        def prox(x,y):
            return operator_P(self.proj, x, y, None)
    else:
        def prox(x,y,u):
            return operator_P(self.proj, x, y, u)
    return set_z(self,prox)

      