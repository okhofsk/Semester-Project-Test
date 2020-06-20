# Import Libraries and Dependencies
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import numpy.linalg as LA
from time import perf_counter

from vilib.matrix_games.generate import*
from vilib.matrix_games.big_mat import*
import vilib.utility as ut
#from utility.utility import
#from utility.global_const import*

class mg_problem:
    """
    Class definition of matrix games library
    """
    def __init__(self, A_=None, proxg_str_=None, size_=None, distrib_="uniform", filepaths_=None):
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
            documenation on "create_A()" in the generate.py module. 
            If the matrix is very large (bigger than 20000x20000) there exist two 
            possibilites:
                - either load matrices saved in .h5 files in either the row format
                  or in reccurently halved cubes as described in big_mat.py
                  for this A_ = "load_harddisk" and filepaths should be of the format
                  marked below. 
                - or create them using the functions defined in big_mat.py in which 
                  case A_ = "harddisk" 

        prox_str_ : string
            string describing the proximal operator wanted. If "None" gives simplex
            Possibilities are:
            - "simplex"      : uses the simplex projection see the function 'projsplx'
            - "fmax"         : returns the (q)+ version of the vector 
            - "none"         : just returns q itself
            default "None"
        
        size_ : tuple/string
            either define size of the A matrix as a tuple or use string "small" for
            a 10x10 matrix size or "large" to have a matrix size of 1000x1000. If loading
            from the hard drive this size must be the same as the saved matrix. 
            
        distrib_ : string
           name of random operator to be used:
               - "uniform" : uniform distirbution from [0,1]
               - "minusuniform" : uniform distirbution from [-1,1]
               - "plusuniform" : uniform distirbution from [-10,10]
            default "uniform"
        
        filepaths_ : list of strings
            if A_ is "load_harddisk"; pass the path to the matrix as well as its transpose 
            as the first two elements of the list in string form. As the third element 
            a boolean has to be set, if the matrices are saved in a row format or recursively 
            as smaller cube matrices as they are generated in big_mat.py. 
            
            For example: filepaths = ["matA.h5", "matAT.h5", True]

        Returns
        -------
        out : mg_problem object
    
        Raises
        ------
        ValueError: if the matrix is too big it cannot be stored in the RAM. Either the size
                    has to be reduced or saved on the hard drive using the function

        References
        ----------

        Examples
        --------
        
        prob = mg_problem("rand", "simplex", (10, 10))
        F_mg, J_mg, prox_g_mg = prob.get_parameters()
        
        """   
        self.harddisk = False
        self.filepathA = None
        self.filepathAT = None
        if isinstance(A_, (np.ndarray, np.generic)) or isinstance(A_, scipy.sparse.coo_matrix):
            self.A = A_
        elif isinstance(A_, str):
            if isinstance(size_, tuple): 
                if A_ == "harddisk" and size_[0] == size_[1]:
                    self.A = None
                    self.harddisk = True
                    create_harddisk_mat(size_[0], distrib_)
                    (self.dimN, self.dimM) = size_
                elif A_ == "load_harddisk" and isinstance(filepaths_, list):
                    if len(filepaths_) == 3:
                        self.A = None
                        self.harddisk = True
                        self.filepathA = filepaths_[0]
                        self.filepathAT = filepaths_[1]
                        self.cube = filepaths_[2]
                        (self.dimN, self.dimM) = size_
                    else:
                        raise ValueError('filepaths_ should have filepath to A and AT. Please refer to documentation.') 
                elif size_[0] <= ut.global_const.LARGEST_ARRAY:
                    self.A = create_A(A_, size_, distrib_)
                else:
                    raise ValueError('Matrices of this size are only available using A = "harddisk_mat" and as a square matrix or load matrix. Please refer to documentation.') 
            elif isinstance(size_, str):
                self.A = create_A(A_, size_, distrib_)
            else :
                self.A = create_A(A_)
        else:
            self.A = create_A('rand')
        if self.A is not None:
            (self.dimN, self.dimM) = self.A.shape
        self.F = intern_Fx(self)
        self.Fx = None
        if isinstance(proxg_str_, str):
            self.name = proxg_str_
            self.proxg = intern_proxg(self)
            if proxg_str_ == 'simplex':
                self.is_simplex = True
            else :
                self.is_simplex = False
            self.J = intern_J(self)
        else:
            self.is_simplex = True
            self.name = 'simplex'
            self.proxg = intern_proxg(self)
            self.J = intern_J(self)

## --------------------------------------------------------------------------------##

    def __str__(self):
        """
        Internal function for matrix games library'
        
        Overridden print operator that defines what happens when the class is printed.

        Parameters
        ----------
        self : instance (object) of the given class

        Returns
        -------
        out : string
        """
        q0 = np.ones(self.dimN + self.dimM)
        return "Matrix A: " + str(self.A) + "\n Matrix F: " + str(self.F(q0)) + "\n Proximal Operator: " + str(self.proxg(q0,0)) + "\n Simplex?: " + str(self.is_simplex) + "\n J Operator: " + str(self.J(q0)) + "\n"
        

## --------------------------------------------------------------------------------##

    def get_parameters(self):
        """
        Internal function for matrix games library'
        
        returns all parameters necessary to test solvers

        Parameters
        ----------
        self : instance (object) of the given class

        Returns
        -------
        out :
            - Fx :        Fx operator for matrix games
            - J :         J operator measuring distance to optimum
            - proxg :     proximal operator
        """
        return self.F, self.J, self.proxg

## --------------------------------------------------------------------------------##

    def get_all(self):
        """
        Internal function for matrix games library'
        
        returns all parameters 

        Parameters
        ----------
        self : instance (object) of the given class

        Returns
        -------
        out :
            - A :       Matrix games matrix (not available if loaded
                        from hard drive)
            - Fx :      Fx operator for matrix games
            - F :       Using A matrix construct F matrix for VI (not available if loaded
                        from hard drive)
            - J :       J operator measuring distance to optimum
            - proxg :   proximal operator
        """
        if self.harddisk:
            return self.A, self.F, None, self.J, self.proxg
        else: 
            return self.A, self.F, create_F(self.A), self.J, self.proxg

## --------------------------------------------------------------------------------##

def intern_Fx(self):
    """
    Internal function for matrix games library'
    
    Computes the Fx product using the matrix a and the vector x. Depending on if the
    matrix is loaded or stored in the hard drive or the RAM the matrix multiplication 
    Ax and A^Ty is done via separate functions defined in big_mat.py or using the @ operator. 
    The result is saved in the problem object for the J operator.
    
    Parameters
    ----------
    self : instance (object) of the given class
        
    Returns
    -------
    out : function 
        the Fx operator
    """
    if self.harddisk:
        if self.filepathA is None:
            def operator(q):
                q = np.reshape(q, (len(q),1))
                q_top = mult_A(q[self.dimM:], True)
                q_bot = -mult_A(q[:self.dimM], False)
                self.Fx = np.append(q_top, q_bot)
                return np.append(q_top, q_bot)
        else:
            def operator(q):
                q = np.reshape(q, (len(q),1))
                q_top = mult_harddisk_mat(self.filepathAT, q[self.dimM:],self.cube, self.dimM)
                q_bot = -mult_harddisk_mat(self.filepathA, q[:self.dimM],self.cube, self.dimN)
                self.Fx = np.append(q_top, q_bot)
                return np.append(q_top, q_bot) 
    else:
        def operator(q):
            q = np.reshape(q, (len(q),1))
            q_top = self.A.T@q[self.dimM:]
            q_bot = -self.A@q[:self.dimM]
            self.Fx = np.append(q_top, q_bot)
            return np.append(q_top, q_bot)
    return operator
    
## --------------------------------------------------------------------------------##

def intern_J(self): 
    """
    External function for matrix games library'

    Computes the J operator depending on which proximal operator was specified on creation
    of the object. For simplex it calculated the primal-dual gap, otherwise is uses the 
    natural residual function to calculated the error. 

    Parameters
    ----------
    self : instance (object) of the given class

    Returns
    -------
    out : function 
        a function that descibes the J operator 
    """
    if self.name == 'simplex':
        def J(q):
            if self.Fx is None:
                Fx = self.F(q)
            else:
                Fx = self.Fx
            ATy = Fx[:self.dimM]
            Ax = -Fx[self.dimM:]
            return np.amax(Ax) - np.amin(ATy) 
        return J
    else :
        def J(q):
            if self.Fx is None:
                Fx = self.F(q)
            else:
                Fx = self.Fx
            return LA.norm(q - self.proxg(q - Fx, 1))
    return J

## --------------------------------------------------------------------------------##

def intern_proxg(self):
    """
    External function for matrix games library'
    
    Computes the proxima operator. There are __ predefined operators available:
        - "simplex"      : uses the simplex projection see the function 'projsplx'
        - "fmax"         : returns the (q)+ version of the vector 
        - "none"         : just returns q itself
    
    Parameters
    ----------
    self : instance (object) of the given class
        
    Returns
    -------
    out : function 
        a function that descibes the prox_g operator 
    
    Raises
    ------
    ValueError: unknown string
        If name_ is not one of the stored strings and defined proxg_operators.   
    """
    if self.name == "simplex":
        def prox_g(q, eps): 
            x = q[:self.dimM]
            y = q[self.dimM:]
            return np.concatenate((ut.utility.projsplx(x),ut.utility.projsplx(y)))
    elif  self.name == "fmax":
        def prox_g(q, eps):
            return np.fmax(q,0)
    elif  self.name == "none":
        def prox_g(q, eps):
            return q 
    else:
        raise ValueError('Input string unknown. Please refer to documentation.') 
    return prox_g

    