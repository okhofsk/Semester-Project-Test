from vilib.utility.utility import*

## --------------------------------------------------------------------------------##

def randQCQP(N_, M_,proj_=None, convex_=True, distrib_="uniform", equality_=False, L_=0):
    """
    External function for matrix games library'
    
    Creates the P matrices and q vectors for a QCQP problem using creat_randMat()
    from the utility package. Based on the quadratic contrained quadratic program[1]
    Stores the matrices and vectors in lists.
    
    Parameters
    ----------
    N_: int
        size of x vector as defined here [1]
        
    M_: int
        number of inequality constraintes
        
    proj_: function
        projection operator for the domain of x
        if "None" projects onto real plane
        default: "None"
        
    convex_: boolean
        if the problem should be convex or not
        default: True
            
    distrib_ : string
       name of random operator to be used:
           - "uniform" : uniform distirbution from [0,1]
           - "minusuniform" : uniform distirbution from [-1,1]
           - "plusuniform" : uniform distirbution from [-10,10]
        default "uniform"
    
    equality_:
        if there should be equality constraints in which
        case L_ should be non-zero
        default: False
    
    L_: int
        the number of equality constaints
        default: 0
    
        
    Returns
    -------
    q0 : ndarray 
        random vector in the feasible set
    p : list 
        list of P matrices
    q : list 
        list of q vectors
    r : list 
        list of r values
    A : ndarray 
        A matrix combined with the b vector define the equality constraints
    b : ndarray 
        b vector combined with the A matrix define the equality constraints
        
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Quadratically_constrained_quadratic_program
    
    Examples
    --------
    """
    p = []
    q = []
    r = []
    if proj_ == None:
        x_rand = create_randMat(distrib_, (N_))
    else:
        x_rand = proj_(create_randMat(distrib_, (N_)))
    q0 = np.concatenate((x_rand, np.ones(M_)))
    print(x_rand)
    if convex_:
        p_temp = create_randMat(distrib_, (N_, N_))
        p.append(np.dot(p_temp, p_temp.transpose()))
    else:
        p_temp = create_randMat(distrib_, (N_, N_))
        p.append(np.maximum(p_temp, p_temp.transpose()))
    q.append(create_randMat(distrib_, (N_)))
    r.append(0)
    for i in range(1,M_+1):
        if convex_:
            p_temp = create_randMat(distrib_, (N_, N_))
            p_temp = np.dot(p_temp, p_temp.transpose())
        else:
            p_temp = create_randMat(distrib_, (N_, N_))
        
        p.append(p_temp)
        q_temp = create_randMat(distrib_, (N_))
        q.append(q_temp)
        r.append(-0.5*x_rand.T@p_temp@x_rand - q_temp@x_rand)
    if equality_:
        A = create_randMat(distrib_, (L_,N_))
        b = A@x_rand
        return q0, p, q, r, A, b
    else:
        return q0, p, q, r, None, None
    
## --------------------------------------------------------------------------------##

def randQCQPmat(N_, M_, proj_=None, convex_=True, distrib_="uniform", equality_=False, L_=0):
    """
    External function for matrix games library'
    
    Creates the P matrices and q vectors for a QCQP problem using creat_randMat()
    from the utility package. Based on the quadratic contrained quadratic program[1]
    Stores the matrices and vectors in ndarrays.
    
    Parameters
    ----------
    N_: int
        size of x vector as defined here [1]
        
    M_: int
        number of inequality constraintes
        
    proj_: function
        projection operator for the domain of x
        if "None" projects onto real plane
        default: "None"
        
    convex_: boolean
        if the problem should be convex or not
        default: True
            
    distrib_ : string
       name of random operator to be used:
           - "uniform" : uniform distirbution from [0,1]
           - "minusuniform" : uniform distirbution from [-1,1]
           - "plusuniform" : uniform distirbution from [-10,10]
        default "uniform"
    
    equality_:
        if there should be equality constraints in which
        case L_ should be non-zero
        default: False
    
    L_: int
        the number of equality constaints
        default: 0
    
        
    Returns
    -------
    q0 : ndarray 
        random vector in the feasible set
    p : ndarray 
        three dimensional array of P matrices
    q : ndarray 
        two dimensional array of P matrices
    r : ndarray 
        one dimensional array of r values
    A : ndarray 
        A matrix combined with the b vector define the equality constraints
    b : ndarray 
        b vector combined with the A matrix define the equality constraints
        
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Quadratically_constrained_quadratic_program
    
    Examples
    --------
    """
    p = []
    q = []
    r = []
    if proj_ == None:
        x_rand = create_randMat(distrib_, (N_))
    else:
        x_rand = proj_(create_randMat(distrib_, (N_)))
    q0 = np.concatenate((x_rand, np.ones(M_)))
    q_temp = create_randMat(distrib_, (M_+1, N_))
    p_temp = create_randMat(distrib_, (M_+1, N_, N_))
    import time
    start_time = time.time()
    if convex_:
        p_temp = np.matmul(p_temp, np.transpose(p_temp, (0,2,1)))
    else :
        p_temp = np.maximum(p_temp, np.transpose(p_temp, (0,2,1)))
    r_temp = -0.5*np.transpose(x_rand)@p_temp@x_rand - q_temp@x_rand
    r_temp[0] = 0
    
    if equality_:
        A = create_randMat(distrib_, (L_,N_))
        b = A@x_rand
        
        print("--- %s seconds ---" % (time.time() - start_time))
        return q0, p_temp, q_temp, r_temp, A, b
    else:
        return q0, p_temp, q_temp, r_temp, None, None
    
## --------------------------------------------------------------------------------##

def toQCQP(P_, q_, r_):
    """
    External function for matrix games library'
    
    Using either a list or ndarray generates the functions defining the QCQP problem
    as defined here [1]
    
    Parameters
    ----------
    p : list/ndarray 
        either list or three dimensional array of P matrices
    q : list/ndarray 
        either list or two dimensional array of P matrices
    r : list/ndarray 
        either list or one dimensional array of r values
        
    Returns
    -------
    fx : function 
        objective function
    hxs : list 
        list of functions that define inequality constraints
    gradf : function 
        gradient of objective function
    gradhs : list 
        list of functions that define gradients of hxs
    hxs_opt : list 
        list of functions that define gradients of hxs but using the 
        gradient as an input to save redundant calculations. 
    
    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.random.html#scipy.sparse.random
    
    Examples
    --------
    """
    fx = lambda x: 0.5*x.T@P_[0]@x + q_[0]@x
    gradf = lambda x: x.T@P_[0] + q_[0]
    hxs = {}
    gradhs = {}
    hxs_opt = {}
    for i in range(1,len(P_)):
        hx = lambda x,i: 0.5*x.T@P_[i]@x + q_[i]@x+r_[i]
        gradh = lambda x,i: x.T@P_[i] + q_[i]
        hx_opt = lambda x,i,gh: 0.5*gh@x+0.5*q_[i]@x+r_[i]
        hxs[i] = hx
        gradhs[i] = gradh
        hxs_opt[i] = hx_opt
    return fx, hxs, gradf, gradhs, hxs_opt

## --------------------------------------------------------------------------------##

def minus(x1_, x2_, y1_=None, y2_=None, u1_=None, u2_=None):
    """
    External function for matrix games library'
    
    Separately subtracts xs, ys and us form each other.
    
    Parameters
    ----------
    x1_ : ndarray
        vector on left side of subtraction
    x2_ : ndarray
        vector on right side of subtraction
    y1_ : ndarray
        vector on left side of subtraction
        default: None
    y2_ : ndarray
        vector on right side of subtraction
        default: None
    u1_ : ndarray
        vector on left side of subtraction
        default: None
    u2_ : ndarray
        vector on right side of subtraction
        default: None
        
    Returns
    -------
    _ : ndarray 
        x1-x2
    _ : ndarray 
        y1-y2 or None
    _ : ndarray 
        u1-u2 or None
    """
    if u1_ is None and u2_ is None:
        if y1_ is None and y2_ is None:
            return x1_ - x2_, None, None
        else:
            return x1_ - x2_, y1_ - y2_, None
    else:
        return x1_ - x2_, y1_ - y2_, u1_ - u2_

## --------------------------------------------------------------------------------##

def create_F(constraints_, equality_):
    """
    External function for matrix games library'

    This function is the counterpart to the internal_F operator that is used in the 
    constrained optimization class. It is meant to support custom version of the 
    operators where certain parts of the calculation can be passed as inputs to 
    speed up the execution. Otherwise the internal functions automatically do the same job.
    
    Parameters
    ----------
    constraints_ : boolean
        if there are inequality constraints True
        
    equality_ : boolean
        if there are equality constraints True
        
    Returns
    -------
    out : function 
        depending on the inputs we have four different operators:
        (1) = False, False; (2) = True, False; 
        (3) = False, True;  (4) = True, True; 
            (1) Fx(gradfx): 
                - where gradfx is the gradient of f evaluated at x
                
            (2) Fx(y, hxs, gradhsx, gradfx):
                - where y is lagrangian coeffiecent of the inequality constraints
                - where hxs is a list of h functions evaluated at x
                - where gradhsx is a list of gradh functions evaluated at x
                - where gradfx is the gradient of f evaluated at x
                
            (3) Fx(gradfx,Ax,b):
                - where gradfx is the gradient of f evaluated at x
                - where Ax the vector output of the matrix multiplication Ax
                - where b is the vector b for equality constraints
                
            (4) Fx(y, hxs, gradhs, gradf,Ax,b):
                - where y is lagrangian coeffiecent of the inequality constraints
                - where hxs is a list of h functions evaluated at x
                - where gradhsx is a list of gradh functions evaluated at x
                - where gradfx is the gradient of f evaluated at x
                - where Ax the vector output of the matrix multiplication Ax
                - where b is the vector b for equality constraints
    """
    if equality_ is False:
        if constraints_ is False:
            def Fx(gradfx):
                return gradfx
        else:
            def Fx(y, hxs, gradhsx, gradfx):
                vec_prod = np.zeros(len(x))
                fy = np.zeros(len(y))
                for i in range(len(y)):
                    vec_prod += y[i] * gradhsx[i+1]
                    fy[i] = -hxs[i+1]
                fx = gradfx+ vec_prod
                return np.concatenate((fx,fy))
    else:
        if constraints_ is False:
            def Fx(gradf,Ax,b):
                fx = gradf
                fu = b_-A_@x
                return p.concatenate((fx,fu))
        else:
            def Fx(y, hxs, gradhs, gradf,Ax,b):
                vec_prod = np.zeros(len(x))
                fy = np.zeros(len(y))
                for i in range(len(y)):
                    vec_prod += y[i] * gradhs[i+1]
                    fy[i] = -hxs[i+1]
                fx = gradf+ vec_prod
                fu = b-Ax
                return p.concatenate((fx,fy,fu))
    return Fx

## --------------------------------------------------------------------------------##

def create_J(constraints_):
    """
    External function for matrix games library'

    This function is the counterpart to the internal_J operator that is used in the 
    constrained optimization class. It is meant to support custom version of the 
    operators where certain parts of the calculation can be passed as inputs to 
    speed up the execution. Otherwise the internal functions automatically do the same job.
    
    Parameters
    ----------
    constraints_ : boolean
        if there are inequality constraints True
            (False) J(x, projFx): 
                - where x is the vector x 
                - where projFx is projected F operator evaluated at x
                
            (True) J(x, projFx, y):
                - where x is the vector x 
                - where projFx is projected F operator evaluated at x
                - where y is lagrangian coeffiecent of the inequality constraints
        
    Returns
    -------
    out : function 
        depending on the inputs we have two different operators:
    """
    if constraints_ is False:
        def J(x, projFx):
            return np.linalg.norm(x-projFx)
    else:
        def J(x, projFx, y):
            return np.linalg.norm(x-projFx) + np.linalg.norm(y-np.clip(y_,0,None))
    return J

## --------------------------------------------------------------------------------##

def operator_P(proj_, x_, y_=None, u_=None):
    """
    External function for matrix games library'
    
    Computes the projection of the x,y and u values. The x_ vector is projected
    using the proj_ function if this is not None. y is always projected onto R+ 
    and u onto R.
    
    Parameters
    ----------
    proj_: function
        projector onto domain of x
    x_: ndarray
        the vector x of the CO problem
    y_: ndarray
        the vector y representing the lagrangian coefficients of the inequality 
        constraints of the CO problem
    u_: ndarray
        the vector u representing the lagrangian coefficients of the equality 
        constraints of the CO problem
        
    Returns
    -------
    _: ndarray
        the projected vector x 
    _: ndarray
        the projected vector y or None
    _: ndarray
        the projected vector u or None
    
    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.random.html#scipy.sparse.random
    
    Examples
    --------
    """
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

        
        