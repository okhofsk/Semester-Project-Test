import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad    # The only autograd function you may ever need
from inspect import signature
import numpy.linalg as LA

from vilib.constrained_optimization.generate import*
from vilib.utility.utility import*

class co_problem:
    """
    Class definition of constrained optimization library
    """
    def __init__(self, fx_, size_, hxs_=None, proj_=None, gradf_=None, gradhs_=None, optimized_=True, A_=None, b_=None): #
        """
        Initialization of qcqp_problem class that containts:
        - F :       F operator 
        - J :       J operator as a measurement of distance to optimum
        - proxg :   proximal operator

        Parameters
        ----------
        fx_: function
            the function f as defined for constrained optimization problems [1]
        
        size_: tuple
            defines the sive of your problem 
                size_[0] should be the size of your x vector
                size_[1] should be the number of inequality constraints, 
                         if there are none this should be 0
                size_[2] should be the number of equality constraints
            
        hxs_:list
            list of functions hx that represent the inequality constraintes as
            defined for constrained optimization problems [1]
            default: "None"
        
        proj_: function
            projection operator for the domain of the variable x as defined for 
            constrained optimization problems [1]
            default: "None" - projection onto real plane
        
        gradf_: function
            the gradient of fx_, if "None" the gradient is automatically calculated.
            default: "None"
        
        gradhs_: list
            list of functions that represent the gradients of hx, if "None" the 
            gradient is automatically calculated.
            default: "None"
        
        optimized_: boolean
            if the optimized version of hxs should be used or not where the gradient
            of hx is required as input
        
        A_: ndarray
            combined with b_ represent the equality constraintes
        
        b_: ndarray
            combined with A_ represent the equality constraintes
            
            
        Returns
        -------
        out : qcqp_problem object

        Notes
        -----

        References
        ----------
        ..[1] https://en.wikipedia.org/wiki/Constrained_optimization

        Examples
        --------
        
        fx = lambda x: x^2
        hx = [lambda x: x+1, lambda x: -x-1]
        proj = vilib.utility.utility.get_projector("simplex")
        gradf = lambda x: 2*x
        gradh = [lambda x: 1, lambda x: -1]
        
        prob = co_problem(fx, (1, 2), hx, proj, gradf, gradh, True)
        F_co, J_co = prob.get_parameters(False)

        """
        self.autograd = False
        if size_ is None:
            raise ValueError('"size"; Must input the size of your problem. Please refer to the documentation.')
        elif isinstance(size_, tuple):
            if len(size_) == 2:
                self.N = size_[0]
                self.M = size_[1]
                self.U = None
            elif len(size_) == 3:
                self.N = size_[0]
                self.M = size_[1]
                self.U = size_[2]
            else:
                raise ValueError('"size"; Tuple must be either of len 2 or 3. Please refer to the documentation.')
        else:
            raise ValueError('"size"; Input of wrong format. Please refer to the documentation.')     
        ### Check "F" ###
        if callable(fx_):
            self.fx = fx_                                   #var
            ### Check "H" ###
            if hxs_ is None:
                self.hx = None
                self.gradh = None
            elif isinstance(hxs_, dict):
                for i in range(1,self.M+1):
                    if not callable(hxs_[i]):
                        raise ValueError('"hx"; Input of wrong format. Please refer to the documentation.') 
                self.hx = hxs_                               #var
                if len(hxs_) != self.M:
                        raise ValueError('"hx"; size_ tuple[1] and len of hx does not match. Please refer to the documentation.') 
                ### Check "gradh" ###
                if gradhs_ is None:
                    self.autograd = True
                    self.gradh = {}
                    for i in range(1,self.M+1):
                        self.gradh[i] = grad(self.hx[i])
                elif isinstance(gradhs_, dict):
                    self.autogard = False
                    if len(gradhs_) != self.M:
                            raise ValueError('"gradhs"; size tuple[1] and len of gradh does not match. Please refer to the documentation.') 
                    for i in range(1,self.M+1):
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
                if self.U != self.A.shape[1]:
                    raise ValueError('"A"; size tuple[2] and columns of A do not match. Please refer to the documentation.')
                if isinstance(b_, (np.ndarray, np.generic)):
                    self.b = b_                                  #var
                    if self.U != len(self.b):
                        raise ValueError('"b"; size tuple[2] and len of b do not match. Please refer to the documentation.')
                elif b_ is None:
                    raise ValueError('"b": Cannot input matrix A without vector b. Please refer to the documentation.')
                else:
                    raise ValueError('"b"; Input of wrong format. Please refer to the documentation.')
            else:
                raise ValueError('"A"; Input of wrong format. Please refer to the documentation.')
        else:
            raise ValueError('"fx"; Necessary Input. Please refer to the documentation.') 
        if proj_ is None:
            self.proj = None
        elif callable(proj_):
            self.proj = proj_
        else:
            raise ValueError('"proj"; Input of wrong format. Please refer to the documentation.') 
        if isinstance(optimized_, bool):
            self.optimized = optimized_
        else:
            raise ValueError('"optimized"; Input of wrong format. Please refer to the documentation.') 
            
        self.F = intern_F(self)
        self.Fz = None
        self.J = intern_J(self)
        self.Fone = set_z(self, self.F)
        self.Jone = set_z(self, self.J)
        self.prox = proximal(self)

## --------------------------------------------------------------------------------##

    def __str__(self):
        """
        Internal function for constrained optimization library'
        
        Overridden print operator that defines what happens when the class is printed.

        Parameters
        ----------
        self : instance (object) of the given class

        Returns
        -------
        out : string
        """
        if self.U is None:
            q0 = np.ones(self.N + self.M)
        else :
            q0 = np.ones(self.N + self.M + self.U)
            
        return "Evaluated point q0: " + str(q0) + "\n F operator evaluated at q0: " + str(self.Fone(q0)) + "\n Proximal Operator evaluated at q0: " + str(self.prox(q0)) + "\n optimized?: " + str(self.optimized) + "\n J Operator evaluated at q0: " + str(self.Jone(q0)) + "\n"

## --------------------------------------------------------------------------------##

    def get_parameters(self, separate_):
        """
        Internal function for constrained optimization library'
        
        returns all parameters necessary to test solvers

        Parameters
        ----------
        self : instance (object) of the given class

        Returns
        -------
        out :
            - Fx :      function that takes x vector and return Fx product
            - J :       J matrix describing the way to classifiy the distance to optimum
        """
        if separate_:
            return self.F, self.J
        else :
            return self.Fone, self.Jone
    
## --------------------------------------------------------------------------------##

def set_z(self, operator_):
    """
    Internal function for constrained optimization library'
    
    creates a version of the operator_ that takes only one input instead of 
    2-3 by splicing, using the operator and then concatenating.

    Parameters
    ----------
    self : instance (object) of the given class
    
    operator_ : function
        either the F or J operator that takes 2-3 inputs 

    Returns
    -------
    out : function
        the operator_ but with only one input
    """
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
    """
    Internal function for constrained optimization library'
    
    creates a version of the proximal operator using the projection operators and 
    the external function operator_P() defined in "generate.py".

    Parameters
    ----------
    self : instance (object) of the given class

    Returns
    -------
    out : function
        the proximal operator but with only one input
    """
    if self.A is None:
        def prox(x,y):
            return operator_P(self.proj, x, y, None)
    else:
        def prox(x,y,u):
            return operator_P(self.proj, x, y, u)
    return set_z(self,prox)

## --------------------------------------------------------------------------------##

def intern_F(self):
    """
    Internal function for constrained optimization library'
    
    creates the variation inequality equivalent operator F of the 
    constrained optimization problem. 

    Parameters
    ----------
    self : instance (object) of the given class

    Returns
    -------
    out : function
        the F operator
    """
    if self.A is None:
        def Fx(x,y):
            if self.hx is None:
                fx = self.gradf(x)
                self.Fz = fx, None, None
                return fx, None, None
            else:
                vec_prod = np.zeros(len(x))
                fy = np.zeros(len(y))
                for i in range(len(y)):
                    gh = self.gradh[i+1](x,i+1)
                    vec_prod += y[i] * gh
                    if self.optimized:
                        fy[i] = -self.hx[i+1](x, i+1, gh)
                    else:
                        fy[i] = -self.hx[i+1](x, i+1)
                fx = self.gradf(x)+ vec_prod
                self.Fz = fx, fy, None
                return fx, fy, None
    else:
        def Fx(x,y,u):
            if self.hx is None:
                fx = self.gradf(x)
                fu = self.b-self.A@x
                self.Fz = fx, None, fu
                return fx, None, fu
            else:
                vec_prod = np.zeros(len(x))
                fy = np.zeros(len(y))
                for i in range(len(y)):
                    gh = self.gradh[i+1](x,i+1)
                    vec_prod += y[i] * gh
                    if self.optimized:
                        fy[i] = -self.hx[i+1](x, i+1, gh)
                    else:
                        fy[i] = -self.hx[i+1](x, i+1)
                fx = self.gradf(x)+ vec_prod
                fu = self.b-self.A@x
                self.Fz = fx, fy, fu
                return fx, fy, fu
    return Fx

## --------------------------------------------------------------------------------##

def intern_J(self):
    """
    Internal function for constrained optimization library'

    Computes the J operator. Uses the natural residual function to calculated the error. 

    Parameters
    ----------
    self : instance (object) of the given class

    Returns
    -------
    out : function
        the J operator but with only one input
    """
    if self.Fz is None:
        fz_none = True
    else:
        fx, fy, fu = self.Fz
        fz_none = False
    if self.A is None:
        def J(x,y):
            if self.hx is None or self.gradh is None:
                if fz_none:
                    fx, _, _ = self.F(x,y)
                xp, _, _ = minus(x, fx)
                xp, _, _ = operator_P(self.proj, xp)
                xp, _, _ = minus(x, xp)
                return LA.norm(xp),None,None
            else:
                if fz_none:
                    fx, fy, _ = self.F(x,y)
                xp, yp, _ = minus(x, fx, y, fy)
                xp, yp, _ = operator_P(self.proj, xp, yp)
                xp, yp, _ = minus(x, xp, y, yp)
                total = np.concatenate((xp, yp))
                return LA.norm(xp)+LA.norm(yp),None,None
    else:
        def J(x,y,u):
            if self.hx is None or self.gradh is None:
                if fz_none:
                    fx, _,fu = self.F(x,y,u)
                xp, up, _ = minus(x, fx, u, fu)
                xp, _, up = operator_P(self.proj, xp, None, up)
                xp, up, _ = minus(x, xp, u, up)
                total = np.concatenate((xp, up))
                return LA.norm(xp)+LA.norm(up),None,None
            else:
                if fz_none:
                    fx, fy, fu = self.F(x,y,u)
                xp, yp, up = minus(x, fx, y, fy, u, fu)
                xp, yp, up = operator_P(self.proj, xp, yp, up)
                xp, yp, up = minus(x, xp, y, yp, u, up)
                total = np.concatenate((xp, yp, up))
                return LA.norm(xp)+LA.norm(yp)+LA.norm(up),None,None
    return J

      