import numpy as np
import matplotlib.pyplot as plt
from random import uniform
from time import perf_counter
import scipy as sp
from scipy import linalg
from scipy.sparse import random
import numpy.linalg as LA
from inspect import signature

## --------------------------------------------------------------------------------##

def randQCQP(N_, M_, convex_, equality_, L_):
    p = []
    q = []
    r = []
    x_rand = np.random.rand(N_)
    if convex_:
        p_temp = np.random.rand(N_, N_)
        p.append(np.dot(p_temp, p_temp.transpose()))
    else:
        p_temp = np.random.rand(N_, N_)
        p.append(np.maximum(p_temp, p_temp.transpose()))
    q.append(np.random.rand(N_))
    r.append(-1)
    for i in range(1,M_+1):
        if convex_:
            p_temp = np.random.rand(N_, N_)
            p_temp = np.dot(p_temp, p_temp.transpose())
        else:
            p_temp = np.random.rand(N_, N_)
            p_temp = np.maximum(p_temp, p_temp.transpose())
        
        p.append(p_temp)
        q_temp = np.random.rand(N_)
        q.append(q_temp)
        r.append(-0.5*x_rand.T@p_temp@x_rand - q_temp@x_rand)
    if equality_:
        A = np.random.rand(L_,N_)
        b = A@x_rand
        return p, q, r, A, b
    else:
        return p, q, r, None, None

## --------------------------------------------------------------------------------##

def toQCQP(P_, q_, r_):
    fx = lambda x: 0.5*x.T@P_[0]@x + q_[0]@x
    gradf = lambda x: x.T@P_[0] + q_[0]
    hxs = []
    gradhs = []
    for i in range(1,len(P_)):
        hx = lambda x: 0.5*x.T@P_[i]@x + q_[i]@x+r_[i]
        gradh = lambda x: x.T@P_[i] + q_[i]
        hxs.append(hx)
        gradhs.append(gradh)
    return fx, hxs, gradf, gradhs

## --------------------------------------------------------------------------------##

def minus(x1_, x2_, y1_=None, y2_=None, u1_=None, u2_=None):
    if u1_ is None and u2_ is None:
        if y1_ is None and y2_ is None:
            return x1_ - x2_, None, None
        else:
            return x1_ - x2_, y1_ - y2_, None
    else:
        return x1_ - x2_, y1_ - y2_, u1_ - u2_

## --------------------------------------------------------------------------------##

def create_F(gradf_, hx_, gradh_, A_=None, b_=None):
    if A_ is None:
        def Fx(x,y):
            if gradh_ is None or hx_ is None:
                return gradf_(x), None, None
            else:
                vec_prod = np.zeros(len(x))
                fy = np.zeros(len(y))
                for i in range(len(y)):
                    vec_prod_ += y[i] * gradh[i](x)
                    fy[i] = -hx[i](x_)
                fx = gradf_(x)+ sum(vec_prod)
                return fx, fy, None
    else:
        def Fx(x,y,u):
            if gradh_ is None or hx_ is None:
                fx = gradf_(x)
                fu = b_-A_@x
                return fx, None, fu
            else:
                vec_prod = np.zeros(len(x))
                fy = np.zeros(len(y))
                for i in range(len(y)):
                    vec_prod += y[i] * gradh_[i](x)
                    fy[i] = -hx_[i](x)
                
                #print(sum(vec_prod).shape)
                #print(u.shape)
                #print(A_.T@u.shape)
                    
                fx = gradf_(x)+ sum(vec_prod)+ A_.T@u
                fu = b_-A_@x
                return fx, fy, fu
    return Fx

## --------------------------------------------------------------------------------##

def create_J(gradf_, hx_, gradh_, proj_, A_=None, b_=None):
    if A_ is None:
        Fx = create_F(gradf_, hx_, gradh_)
        def J(x,y):
            if hx_ is None or gradh_ is None:
                fx, _, _ = Fx(x,y)
                xp, yp, _ = minus(x, fx)
                xp, yp, _ = operator_P(proj_, xp)
                xp, yp, _ = minus(x, xp)
                return np.linalg.norm(xp),None,None
            else:
                fx, fy, _ = Fx(x,y)
                xp, yp, _ = minus(x, fx, y, fy)
                xp, yp, _ = operator_P(proj_, xp, yp)
                xp, yp, _ = minus(x, xp, y, yp)
                return np.linalg.norm(xp) + np.linalg.norm(yp),None,None
    else:
        Fx = create_F(gradf_, hx_, gradh_, A_, b_)
        def J(x,y,u):
            if hx_ is None or gradh_ is None:
                something
            else:
                fx, fy, fu = Fx(x,y,u)
                xp, yp, up = minus(x, fx, y, fy, u, fu)
                xp, yp, up = operator_P(proj_, xp, yp, up)
                xp, yp, up = minus(x, xp, y, yp, u, up)
                return np.linalg.norm(xp) + np.linalg.norm(yp) + np.linalg.norm(up),None,None
    return J

## --------------------------------------------------------------------------------##

def operator_P(proj_, x_, y_=None, u_=None):
    if proj_ is None:
        if y_ is None:
            return x_, y_, u_
        else:
            return x_, np.maximum(y_,0), u_
    else:
        if y_ is None:
            return proj_(x_), y_, u_
        else:
            return proj_(x_), np.maximum(y_,0)

        
        