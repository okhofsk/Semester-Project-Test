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

def randQCQP(N_, M_, convex_, equality_, L_=1):
    p = []
    q = []
    r = []
    x_rand = np.random.rand(N_)
    q0 = np.concatenate((x_rand, np.ones(M_)))
    print(x_rand)
    if convex_:
        p_temp = np.random.rand(N_, N_)
        #print("p_temp(0): ", p_temp)
        p.append(np.dot(p_temp, p_temp.transpose()))
    else:
        p_temp = np.random.rand(N_, N_)
        p.append(np.maximum(p_temp, p_temp.transpose()))
    q.append(np.random.rand(N_))
    print("q_temp(0): ", q[0])
    r.append(0)
    for i in range(1,M_+1):
        if convex_:
            p_temp = np.random.rand(N_, N_)
            #print("p_temp","(",i, "): ", p_temp)
            p_temp = np.dot(p_temp, p_temp.transpose())
        else:
            p_temp = np.random.rand(N_, N_)
            #p_temp = np.maximum(p_temp, p_temp.transpose())
        
        p.append(p_temp)
        q_temp = np.random.rand(N_)
        #print("q_temp","(",i, "): ", q_temp)
        q.append(q_temp)
        r.append(-0.5*x_rand.T@p_temp@x_rand - q_temp@x_rand)
        #print("r_temp","(",i, "): ", r[i])
    if equality_:
        A = np.random.rand(L_,N_)
        b = A@x_rand
        return q0, p, q, r, A, b
    else:
        return q0, p, q, r, None, None
    
def randQCQPmat(N_, M_, convex_, equality_, L_=1):
    p = []
    q = []
    r = []
    x_rand = np.random.rand(N_)
    q_temp = np.random.rand(M_+1, N_)
    p_temp = np.random.rand(M_+1, N_, N_)
    import time
    start_time = time.time()
    if convex_:
        p_temp = np.matmul(p_temp, np.transpose(p_temp, (0,2,1)))
    else :
        p_temp = np.maximum(p_temp, np.transpose(p_temp, (0,2,1)))
    r_temp = -0.5*np.transpose(x_rand)@p_temp@x_rand - q_temp@x_rand
    r_temp[0] = 0
    
    if equality_:
        A = np.random.rand(L_,N_)
        b = A@x_rand
        
        print("--- %s seconds ---" % (time.time() - start_time))
        return p_temp, q_temp, r_temp, A, b
    else:
        return p_temp, q_temp, r_temp, None, None
    
## --------------------------------------------------------------------------------##

def toQCQP(P_, q_, r_):
    fx = lambda x: 0.5*x.T@P_[0]@x + q_[0]@x
    gradf = lambda x: x.T@P_[0] + q_[0]
    hxs = []
    gradhs = []
    for i in range(1,len(P_)):
        hx = lambda x,i: 0.5*x.T@P_[i]@x + q_[i]@x+r_[i]
        gradh = lambda x,i: x.T@P_[i] + q_[i]
        hxs.append(hx)
        gradhs.append(gradh)
    return fx, hxs, gradf, gradhs

"""def toQCQPmat(P_,q_,r_):
    full = 
    x,i: 0.5*x.T@P_[i]@x + q_[i]@x + r_[i]
    gradfull = lambda x,i: x.T@P_[i] + q_[i]
    fx = lambda x: full(x,i)
    [lambda x: full(x,i) for i in range(1,len(P_))]
    hx = lambda x: 1
    fx = lambda x: full(x,i)
    fx = lambda x: full(x,i)
    return full"""

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
                    vec_prod += y[i] * gradh_[i+1](x,i+1)
                    fy[i] = -hx_[i+1](x, i+1)
                fx = gradf_(x)+ vec_prod
                #print("gradf_(x): ", gradf_(x))
                #print("vec_prod: ", vec_prod)
                #print("fx: ", fx)
                #print("fy: ", fy)
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
                    #print("gradh difference: ", gradh_[i](x,i+1))
                    #print("y: ", y[i])
                    #print("vec-prod(should): ", y * gradh_[i](x,i+1))
                    #print("vec-prod(is): ", vec_prod)
                    vec_prod += y[i] * gradh_[i+1](x,i+1)
                    #print("vec-prod(is): ", vec_prod)
                    fy[i] = -hx_[i+1](x,i+1)
                
                #print(vec_prod.shape)
                #print(u.shape)
                #print((A_.T@u).shape)
                    
                fx = gradf_(x)+ vec_prod + A_.T@u
                fu = b_-A_@x
                return fx, fy, fu
    return Fx

## --------------------------------------------------------------------------------##

def create_J_int(self):
    if self.A is None:
        Fx = self.F
        def J(x,y):
            if self.hx is None or self.gradh is None:
                fx, _, _ = Fx(x,y)
                xp, _, _ = minus(x, fx)
                xp, _, _ = operator_P(self.proj, xp)
                xp, _, _ = minus(x, xp)
                return np.linalg.norm(xp),None,None
            else:
                fx, fy, _ = Fx(x,y)
                xp, yp, _ = minus(x, fx, y, fy)
                xp, yp, _ = operator_P(self.proj, xp, yp)
                xp, yp, _ = minus(x, xp, y, yp)
                total = np.concatenate((xp, yp))
                return np.linalg.norm(total),None,None
    else:
        Fx = self.F
        def J(x,y,u):
            if self.hx is None or self.gradh is None:
                fx, _, fu = Fx(x,y,u)
                xp, up, _ = minus(x, fx, u, fu)
                xp, _, up = operator_P(self.proj, xp, None, up)
                xp, up, _ = minus(x, xp, u, up)
                total = np.concatenate((xp, up))
                return np.linalg.norm(total),None,None
            else:
                fx, fy, fu = Fx(x,y,u)
                xp, yp, up = minus(x, fx, y, fy, u, fu)
                xp, yp, up = operator_P(self.proj, xp, yp, up)
                xp, yp, up = minus(x, xp, y, yp, u, up)
                total = np.concatenate((xp, yp, up))
                #return np.linalg.norm(total),None,None
                return np.linalg.norm(xp)+np.linalg.norm(yp)+np.linalg.norm(up),None,None
    return J

## --------------------------------------------------------------------------------##

def create_J(gradf_, hx_, gradh_, proj_, A_=None, b_=None):
    if A_ is None:
        Fx = create_F(gradf_, hx_, gradh_)
        def J(x,y):
            if hx_ is None or gradh_ is None:
                fx, _, _ = Fx(x,y)
                xp, _, _ = minus(x, fx)
                xp, _, _ = operator_P(proj_, xp)
                xp, _, _ = minus(x, xp)
                return np.linalg.norm(xp),None,None
            else:
                fx, fy, _ = Fx(x,y)
                xp, yp, _ = minus(x, fx, y, fy)
                xp, yp, _ = operator_P(proj_, xp, yp)
                xp, yp, _ = minus(x, xp, y, yp)
                total = np.concatenate((xp, yp))
                return np.linalg.norm(xp)+np.linalg.norm(yp),None,None
    else:
        Fx = create_F(gradf_, hx_, gradh_, A_, b_)
        def J(x,y,u):
            if hx_ is None or gradh_ is None:
                fx, _, fu = Fx(x,y,u)
                xp, up, _ = minus(x, fx, u, fu)
                xp, _, up = operator_P(proj_, xp, None, up)
                xp, up, _ = minus(x, xp, u, up)
                total = np.concatenate((xp, up))
                return np.linalg.norm(total),None,None
            else:
                fx, fy, fu = Fx(x,y,u)
                xp, yp, up = minus(x, fx, y, fy, u, fu)
                xp, yp, up = operator_P(proj_, xp, yp, up)
                xp, yp, up = minus(x, xp, y, yp, u, up)
                total = np.concatenate((xp, yp, up))
                return np.linalg.norm(total),None,None
    return J

## --------------------------------------------------------------------------------##

def operator_P(proj_, x_, y_=None, u_=None):
    if proj_ is None:
        if y_ is None:
            return x_, y_, u_
        else:
            return x_, np.clip(y_,0,None), u_
    else:
        if y_ is None:
            return proj_(x_), y_, u_
        else:
            return proj_(x_), np.clip(y_,0,None), u_

        
        