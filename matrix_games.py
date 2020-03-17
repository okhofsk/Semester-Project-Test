# Import Libraries and Dependencies
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from time import perf_counter
import scipy as sp
from scipy.sparse import random
import numpy.linalg as LA

# create a random A matrix that is either sparse or non-sparse
def create_A(n, m, sparsity):
    if sparsity:
        return sp.sparse.random(n, m)
    else:
        a = np.zeros([n,m])
        for i in range(n):
            for j in range(m):
                a[i,j] = randrange(10)
        return a
    
# create the F operator, only needed if wanted explicitly
def create_F(a):
    zero_matrix_top = np.zeros((a.shape[1], a.shape[1]))
    zero_matrix_bot = np.zeros((a.shape[0], a.shape[0]))
    left_F = np.concatenate((zero_matrix_top, -a))
    right_F = np.concatenate((a.transpose(), zero_matrix_bot))
    return np.concatenate((left_F, right_F), axis=1)

# multiplies the input vector with the F operator
def Fx_operator(a, x):
    x_ = np.reshape(x, (len(x),1))
    (dimN, dimM) = a.shape
    x_top = a.T.dot(x_[dimM-1:dimN+dimM-1])
    x_bot = -a.dot(x_[:dimM])
    return np.append(x_top, x_bot)