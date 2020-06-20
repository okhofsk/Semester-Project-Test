from vilib.utility.global_const import*
from vilib.utility.utility import*
 
import tables as tb      

## --------------------------------------------------------------------------------##
  
def mult_harddisk_mat(filepath_, x_, cube_, output_size_ = None):
    """
    'big_mat support module'
    
    Generic function for matrix multiplication when the A matrix is saved on 
    the hard drive. Calls either mult_mat or mult_mat_row depending on the value
    of cube_. If cube_ is true then mult_A is called. See their description
    for more information.
    
    Parameters
    ----------
    filepath_ : string
        the filepath of the matrix A saved on the hard drive
        
    x_ : ndarray
        the vector x used for the matrix multiplication Ax
        
    cube_ : boolean
        depending in which format the matrix A is saved;
            - True: matrix is saved as small recurrent cube matrices
                    until sub matrices are small enough to be stored 
                    in the RAM. For more information see "recur_save()"
            - False: matrix saved in row format
        
    output_size_ : int
        size of the output vector
        
    Returns
    -------
    out : ndarray 
        the Ax matrix multiplication product
    """
    if cube_:
        h5file = tb.open_file(filepath_, mode='r', title="MatrixGame Amat")
        output = recur_mat(x_.shape[0], h5file, COR_NAME, x_)   
        h5file.close()
        return output
    else:
        return mult_mat_row(filepath_, x_, output_size_) 

## --------------------------------------------------------------------------------##
  
def mult_mat_row(filepath_, x_, out_size_):
    """
    'big_mat support module'
    
    This function uses the walk_nodes() [1] to pass through every 'Array' element 
    in order of creation and performs the vector product of x_ and the row. 
    
    Parameters
    ----------
    filepath_ : string
        the filepath of the matrix A saved on the hard drive
        
    x_ : ndarray
        the vector x used for the matrix multiplication Ax
        
    output_size_ : int
        size of the output vector
        
    Returns
    -------
    out : ndarray 
        the Ax matrix multiplication product
    
    References
    ----------
    .. [1] https://www.pytables.org/
    """
    h5file = tb.open_file(filepath_, mode='r')
    size = x_.shape[0]
    output = np.zeros(out_size_)
    i = 0
    for array in h5file.walk_nodes("/", "Array"):
        output[i] = array.read().reshape(size)@x_
        i += 1
    h5file.close()
    return output
    
## --------------------------------------------------------------------------------##
  
def create_harddisk_mat(size_, distrib_):
    """
    'big_mat support module'
    
    Generic function for creating two cube matrices, starting with the A matrix 
    saved in the file "matA.h5" in the same directory as the executing file. Then
    creating the transpose matrix in the file "matAT.h5" in the same directory as above.
    This matrix is created in a recurrent manner using in-place transposition.
    
    For more information on the generation of the matrices see 'recur_save'
    
    Parameters
    ----------
    size_ : int
        the size of the nxn matrix to be created
            
    distrib_ : string
       name of random operator to be used:
           - "uniform" : uniform distirbution from [0,1]
           - "minusuniform" : uniform distirbution from [-1,1]
           - "plusuniform" : uniform distirbution from [-10,10]
        default "uniform"
        
    Returns
    -------
    """
    h5file = tb.open_file("matA.h5", mode='w', title="MatrixGame Amat")
    recur_save(h5file, size_, COR_NAME, 0.0, 25.0, distrib_)
    print("---Matrix A saved in 'matA.h5'---")
    h5file.close()
    
    h5file = tb.open_file("matA.h5", mode='r', title="MatrixGame Amat")
    h5fileT = tb.open_file("matAT.h5", mode='w', title="MatrixGame Amat")
    recur_transpos(size_, h5file, h5fileT, COR_NAME, COR_NAME, 0.0, 25.0)
    print("---Matrix A transpose saved in 'matAT.h5'---")
    h5file.close()
    h5fileT.close()

## --------------------------------------------------------------------------------##
  
def recur_save(filename_, size_, name_, percentage_, ratio_, distrib_):
    """
    'big_mat support module'
    
    The matrices are created using this recursive function. If size_/2 is not even 
    the algorithm is stopped. If the size_/2 is smaller than LARGEST_ARRAY value, the
    matrix is separated into four earrays [1] using the create_randMat function 
    defined in the utility package. Otherwise the algorithm is called again with 
    size_ = size_/2 until the conditon is verified. These matrices are all given a name
    based on the COR_NAME value defined in the utility package. For every time this
    function is called, a number (1,2,3 or 4) is added to the end of COR_NAME. Therefore
    if the function is called twice the matrices will be labeled in the following form:
    
                            COR_NAME11 COR_NAME12  COR_NAME21 COR_NAME22
                            COR_NAME13 COR_NAME14  COR_NAME23 COR_NAME24
                            COR_NAME31 COR_NAME32  COR_NAME41 COR_NAME42
                            COR_NAME33 COR_NAME34  COR_NAME43 COR_NAME44
                            
    generating 16 smaller submatrices.
    
    Parameters
    ----------
    filepath_ : object
        the opened object by tb.open_file(...)[1]
        
    size_ : int
        size_ of the nxn (sub)matrix
        
    name_ : string
        name of the matrices based on the above mentioned naming convention.
        
    percentage_ : int
        for showing the percentage of progress made creating the matrix
        
    ratio_ : int
        for showing the percentage of progress made creating the matrix
            
    distrib_ : string
       name of random operator to be used:
           - "uniform" : uniform distirbution from [0,1]
           - "minusuniform" : uniform distirbution from [-1,1]
           - "plusuniform" : uniform distirbution from [-10,10]
        default "uniform"
        
    Returns
    -------
    out : int 
        percentage passed to show progress
    
    References
    ----------
    .. [1] https://www.pytables.org/
    """
    root = filename_.root
    if size_/2.0%2 != 0:
        tb.file._open_files.close_all()
        raise ValueError('Matrix or submatrix not even. Please refer to documentation.') 
    if size_ <= LARGEST_ARRAY:
        for i in range(4):
            arr = filename_.create_earray(root, name_+str(i), tb.Float32Atom(), (0,size_/2),"test", filters=tb.Filters(complevel=1, complib='blosc', fletcher32=True))
            temp = create_randMat(distrib_, ((int(size_/2),int(size_/2))))
            arr.append(temp)
            percentage_ += ratio_
            print_percentage(percentage_, False)
    else:
        for i in range(4):
            percentage_ = recur_save(filename_, size_/2, name_+str(i),percentage_, ratio_/4, distrib_) 
    return percentage_

## --------------------------------------------------------------------------------##
  
def recur_transpos(size_, filename_, filenameT_, name_, nameT_, percentage_, ratio_):
    """
    'big_mat support module'
    
    The matrices are created using this recursive function. The steps are the same as 
    for recur_save() except when size_ <= LARGEST_ARRAY the submatrix is loaded from
    the A matrix file and then the inplace transposition is done:
    
                             [a1, a2]            [a1^T, a3^T]
                         A = [a3, a4] ====> AT = [a2^T, a4^T]
    keeping the same naming convention as for recur_save().
    
    Parameters
    ----------
    size_ : int
        size_ of the nxn (sub)matrix 
    
    filepath_ : object
        the opened object matrix A by tb.open_file(...)[1]
    
    filepathT_ : object
        the opened object matrix A^T by tb.open_file(...)[1]
        
    name_ : string
        name of the matrices based on the above mentioned naming convention.
        
    percentage_ : int
        for showing the percentage of progress made creating the matrix
        
    ratio_ : int
        for showing the percentage of progress made creating the matrix
        
    Returns
    -------
    out : int 
        percentage passed to show progress
    
    References
    ----------
    .. [1] https://www.pytables.org/
    """
    atom = tb.Float32Atom()
    filters = tb.Filters(complevel=1, complib='blosc', fletcher32=True)
    A_temp = None
    shape_size = (int(size_/2),int(size_/2))
    if size_ <= LARGEST_ARRAY:
        # Top corner just transposed
        A_temp = filename_.get_node(filename_.root, name_+"0", "Array").read().reshape(shape_size)
        arr = filenameT_.create_earray(filenameT_.root, nameT_+"0", atom, (0,size_/2),"test", filters=filters)
        arr.append(A_temp.T)
        percentage_ += ratio_
        print_percentage(percentage_, True)
    else:
        percentage_ = recur_transpos(size_/2, filename_, filenameT_, name_+"0", nameT_+"0",percentage_, ratio_/4)
        
    if size_ <= LARGEST_ARRAY:
        # Bot left corner switched to Top right and transposed
        A_temp = filename_.get_node(filename_.root, name_+"2", "Array").read().reshape(shape_size)
        arr = filenameT_.create_earray(filenameT_.root, nameT_+"1", atom, (0,size_/2),"test", filters=filters)
        arr.append(A_temp.T)
        percentage_ += ratio_
        print_percentage(percentage_, True)
    else:
        percentage_ = recur_transpos(size_/2, filename_, filenameT_, name_+"2", nameT_+"1",percentage_, ratio_/4)
        
    if size_ <= LARGEST_ARRAY:
        # Bot left corner just transposed
        A_temp = filename_.get_node(filename_.root, name_+"1", "Array").read().reshape(shape_size)
        arr = filenameT_.create_earray(filenameT_.root, nameT_+"2", atom, (0,size_/2),"test", filters=filters)
        arr.append(A_temp.T)
        percentage_ += ratio_
        print_percentage(percentage_, True)
    else:
        percentage_ = recur_transpos(size_/2, filename_, filenameT_, name_+"1", nameT_+"2",percentage_, ratio_/4)
        
    if size_ <= LARGEST_ARRAY:
        # Bot corner just transposed
        A_temp = filename_.get_node(filename_.root, name_+"3", "Array").read().reshape(shape_size)
        arr = filenameT_.create_earray(filenameT_.root, nameT_+"3", atom, (0,size_/2),"test", filters=filters)
        arr.append(A_temp.T)
        percentage_ += ratio_
        print_percentage(percentage_, True)
    else:
        percentage_ = recur_transpos(size_/2, filename_, filenameT_, name_+"3", nameT_+"3",percentage_, ratio_/4)
    return percentage_   

## --------------------------------------------------------------------------------##
  
def recur_mat(size_, filename_, name_, x_):
    """
    'big_mat support module'
    
    The matrix multiplication is implemented using this recursive function. The steps 
    are the same as for recur_save() except when size_ <= LARGEST_ARRAY the submatrix 
    is mulitplied by a spliced version of the x vector. Then once all four submatrices 
    have been multiplied they are added :
        out[:shape_size[0]] = (out0 + out1)
        out[shape_size[0]:] = (out2 + out3)
    
    Parameters
    ----------
    size_ : int
        size_ of the nxn (sub)matrix 
    
    filepath_ : object
        the opened object matrix A by tb.open_file(...)[1]
        
    name_ : string
        name of the matrices based on the above mentioned naming convention.
        
    x_ : ndarray
        x vector or a spliced version
        
    Returns
    -------
    out : ndarray 
        the Ax matrix multiplication product
    """
    shape_size = (int(size_/2),int(size_/2))
    out0 = np.zeros(shape_size[0])
    out1 = np.zeros(shape_size[0])
    out2 = np.zeros(shape_size[0])
    out3 = np.zeros(shape_size[0])
    if size_ <= LARGEST_ARRAY:
        # Top corner
        A_temp = filename_.get_node(filename_.root, name_+"0", "Array").read().reshape(shape_size)
        out0 = A_temp@x_[:shape_size[0]]
    else:
        out0 = recur_mat(size_/2, filename_, name_+"0", x_[:shape_size[0]])
        
    if size_ <= LARGEST_ARRAY:
        # Bot left corner switched to Top right and transposed        
        A_temp = filename_.get_node(filename_.root, name_+"1", "Array").read().reshape(shape_size)
        out1 = A_temp@x_[:shape_size[0]]
    else:
        out1 = recur_mat(size_/2, filename_, name_+"1", x_[:shape_size[0]])
        
    if size_ <= LARGEST_ARRAY:
        # Bot left corner just transposed
        A_temp = filename_.get_node(filename_.root, name_+"2", "Array").read().reshape(shape_size)
        out2 = A_temp@x_[shape_size[0]:]
    else:
        out2 = recur_mat(size_/2, filename_, name_+"2", x_[shape_size[0]:])
        
    if size_ <= LARGEST_ARRAY:
        # Bot corner just transposed        
        A_temp = filename_.get_node(filename_.root, name_+"3", "Array").read().reshape(shape_size)
        out3 = A_temp@x_[shape_size[0]:]
    else:
        out3 = recur_mat(size_/2, filename_, name_+"3", x_[shape_size[0]:])
    A_temp = None
    out = np.zeros(int(size_))
    out[:shape_size[0]] = (out0 + out1).reshape((int(size_/2)))
    out[shape_size[0]:] = (out2 + out3).reshape((int(size_/2)))
    return out    

## --------------------------------------------------------------------------------##

def print_percentage(percentage_, transpose_):
    if transpose_:
        print ("Storing matrix A: " + str("{:.2f}".format(percentage_)) + "%", end="\r")
        time.sleep(1)
    else:
        print ("Storing matrix A^T: " + str("{:.2f}".format(percentage_)) + "%", end="\r")
        time.sleep(1)