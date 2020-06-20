# Import Libraries and Dependencies

from vilib.utility.utility import*
import scipy.sparse
import networkx as nx

def create_A(name_, size_="small", distrib_="uniform"):
    """
    External function for matrix games library'
    
    Computes A matrix using stored entries:
    - "rock_paper_scissor"      : payoff matrix for the game rock, paper, scissor
    - "normandy"                : Matrix from “Normandy: Game and Reality”[1]
    - "diag"                    : random diagonal game matrix; if custom size, 
                                   second dim will be considered to make nxn matrix
    - "triangular"              : random upper triangular matrix
    - "hide_and_seek"           : Hide and Seek Matrix as defined in [2]  
                                  '3.2. Hide and Seek games'
    - "one_plus"                : One plus matrix as defined in [2] 
                                  'Example 2.4.2 (Plus One)'
    - "matching_pennies"        : Matching pennies as defined in [2]
                                  '2.17. Generalized Matching Pennies' using the function
                                  create_rand_pennies_mat() that creates the graph and 
                                  matrix based on its definition in [2]
    - "rand"                    : a random matrix using uniform distibution from [0,1]
    - "randminus"               : a random matrix using uniform distirbution from [-1,1]
    - "randplus"                : a random matrix using uniform distirbution from [-10,10]
    - "rand_sparse"             : a random sparse matrix [3]
    
    Parameters
    ----------
    name_ : string
        refernce string of the matrix to be returned
        
    size_ : tuple/string 
        define shape of matrix only -----, default "small"
            
    distrib_ : string
       name of random operator to be used:
           - "uniform" : uniform distirbution from [0,1]
           - "minusuniform" : uniform distirbution from [-1,1]
           - "plusuniform" : uniform distirbution from [-10,10]
        default "uniform"
        
    Returns
    -------
    out : ndarray or a sparse matrix in csr format.[4]
    
    Raises
    ------
    ValueError: unknown string
        If name_ is not one of the stored strings and defined A matices.
    
    References
    ----------
    .. [1] “Normandy: Game and Reality” by W. Drakert in Moves, No. 6 (1972)
    .. [2] "Game Theory, Alive" by Anna R. Karlin and Yuval Peres (2016)
    .. [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.random.html#scipy.sparse.random
    .. [3] https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix
    
    Examples
    --------
    """
    if isinstance(name_, str):
        if isinstance(size_, tuple) :
            if size_ is (0,0):
                raise ValueError('Input a tuple as the size of your A matrix. Please refer to documentation.') 
            else : 
                size = size_
        elif isinstance (size_, str) :
            if size_  is "large":
                size = (1000,1000)
            elif size_ is "small": 
                size = (10,10)
            else:
                raise ValueError('Input string size_ unknown. Please refer to documentation.') 
        else:
            raise ValueError('Input "size_" unknown input format. Please refer to the documentation.')
        if name_ == "rock_paper_scissor":
            return np.array([[0, -1, 1],[1, 0, -1],[-1, 1, 0]])            
        elif name_ == "normandy":
            return np.array([[13, 29, 8, 12, 16, 23],[18, 22, 21, 22, 29, 31],[18, 22, 31, 31, 27, 37],[11, 22, 12, 21, 21, 26],[18, 16, 19, 14, 19, 28],[23, 22, 19, 23, 30, 34]])
        elif name_ == "diag": 
            return np.diagflat(create_A_rand(1, size[1], False))
        elif name_ == "triangular":
            return np.triu(create_A_rand(size[0], size[1], False)) 
        elif name_ == "hide_and_seek":
            return np.random.randint(2, size=size)
        elif name_ == "one_plus":
            ones = np.ones((size[0], size[0]))
            return (-2)*ones + np.diag(np.diag(ones, k=1), k=1) + np.diag(np.diag(ones, k=-1), k=-1) + np.diag(np.diag(ones))*2
        elif name_ == "matching_pennies":
            return create_rand_pennies_mat(size[0])
        elif name_ == "rand":
            return create_randMat(distrib_, size)
        elif name_ == "rand_sparse":
            return scipy.sparse.random(size[0], size[1])
        else:
            raise ValueError('Input string "name_" unknown. Please refer to the documentation.') 
    else:
        raise ValueError('Input "name_" unknown input format. Please refer to the documentation.')    

## --------------------------------------------------------------------------------##
  
def create_F(A_):
    """
    External function for matrix games library'
    
    Computes the F operator in a matrix form explicitly for the matrix games 
    problem. Its shape is F = [[0, A^T];[-A, 0]] where 0 indicates a zero
    matrix of the appropriate size (for A a nxm matrix we have a mxm matrix 
    in the top left corner and a nxn matrix on the bottom right corner).
    
    Parameters
    ----------
    A_ : ndarray/sparse.csr[1]
        the matrix A from the matrix games problem in sparse or ndarray form
        
    Returns
    -------
    out : ndarray 
        the operator F in matrix form as given by theory
    
    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.random.html#scipy.sparse.random
    
    Examples
    --------
    """
    if scipy.sparse.issparse(A_):
        mat_a = np.copy(a.toarray())
    else:
        mat_a = np.copy(A_)
    zero_matrix_top = np.zeros((mat_a.shape[1], mat_a.shape[1]))
    zero_matrix_bot = np.zeros((mat_a.shape[0], mat_a.shape[0]))
    left_F = np.concatenate((zero_matrix_top, -mat_a))
    right_F = np.concatenate((mat_a.transpose(), zero_matrix_bot))
    return np.concatenate((left_F, right_F), axis=1)

## --------------------------------------------------------------------------------##

def create_F_rand(n_, m_, sparsity_, distrib_="uniform"):
    """
    External function for matrix games library'
    
    Computes the F operator explicitly for the matrix games problem using a matrix A
    with random variables of size nxm which is given in a matrix form by 
    F = [[0, A^T];[-A, 0]] where 0 indicates a zero matrix of the appropriate size 
    (for A a nxm matrix we have a mxm matrix in the top left corner and a nxn matrix 
    on the bottom right corner.
    
    Parameters
    ----------
    n_ : int
        number of columns.
    m_ : int
        number of rows.
    sparsity_ : boolean
        if "True" it returns a random sparse matrix using the scipy.sparse
        library[1], otherwise it computes a full matrix with random
        entries from 0 to 1.
    distrib_ : string
        should be same as distribution used in create_A()
        default "uniform"
        
    Returns
    -------
    out : ndarray 
        the operator F in matrix form
    
    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.random.html#scipy.sparse.random
    """
    if sparsity_:
        mat_a = scipy.sparse.random(n_,m_).toarray()
    else:
        mat_a = create_randMat(distrib_, (n_,m_))
    return create_F(mat_a)

## --------------------------------------------------------------------------------##

def external_Fx():
    """
    External function for matrix games library'

    This function is the counterpart to the internal_F operator that is used in the 
    matrix game class. It is meant to support custom version of the operators where
    certain parts of the calculation can be passed as inputs to speed up the execution.
    Otherwise the internal functions automatically do the same job.
    
    Computes the Fx product using the vectors Ax and A^Ty.
    
    Parameters
    ----------
        
    Returns
    -------
    out : function 
        a function taking the vectors Ax and A^Ty as inputs and
        generating the output of Fx for matrix games
    """
    def operator(ax_, ay_):
        x_top = ay_
        x_bot = -ax_
        return np.append(x_top, x_bot)
    return operator

## --------------------------------------------------------------------------------##

def external_J(name_=None, prox_g_=None): 
    """
    External function for matrix games library'

    This function is the counterpart to the internal_J operator that is used in the 
    matrix game class. It is meant to support custom version of the operators where
    certain parts of the calculation can be passed as inputs to speed up the execution.
    Otherwise the internal functions automatically do the same job.
    
    Computes the J operator depending on what type of proximal operator is used.
        -For simplex it calculates the dual-primal gap. Generating an operator that 
         takes the matrix product Ax and A^Ty as inputs. 
        -Otherwise the residual function is used, here there are two possible operators:
            - if prox_g_ is a function then it calculates the residual while taking the 
              vector x and Fx as inputs.
            - otherwise then it calculates the residual while only taking x and the value
              of prox_g_(q - Fx) as inputs.
       

    Parameters
    ----------
    name_ : string
        either "simplex" or otherwise will return the residual operator
    prox_g_ : function
        the proximal operator function

    Returns
    -------
    out : function 
        a function that descibes the J operator 
    """
    dimN, dimM = A_.shape
    if name_ == 'simplex':
        def J(ax, ay):
            return np.amax(ax) - np.amin(ay)
    else :
        if callable(prox_g_):
            def J(q_, Fx_, eps=1):
                return LA.norm(q_ - prox_g_(q_ - Fx_, eps))
        else:
            def J(q_, proxg_):
                return LA.norm(q_ - proxg_)
    return J

## --------------------------------------------------------------------------------##

def create_rand_pennies_mat(size0_):
    """
    As based on the description in [1] this function uses the networkx 
    package to generate a random graph of a defined size. Using this 
    graph the matrix A is generated. 
       

    Parameters
    ----------
    size0_: the size of the nxn matrix 

    Returns
    -------
    out : ndarray
        the matrix A
    
    References
    ----------
    .. [1] "Game Theory, Alive" by Anna R. Karlin and Yuval Peres (2016)
    """
    G = nx.fast_gnp_random_graph(size0_, 1.0, 78456, True)
    W = np.zeros(size0_)
    u0 = -1
    for (u, v) in G.edges():
        wij = np.random.randint(0,10)
        G.edges[u,v]['weight'] = wij
        if (u == u0):
            W[u] += wij
        else :
            u0 = u
            W[u] = wij


    a = np.zeros((size0_,size0_))
    for (u,v) in G.edges():
        a[u,v] = G.edges[u,v]['weight']

    for i in range(size0_):
        wii = np.random.randint(0,10)
        a[i,i] = wii - W[i]
    return a