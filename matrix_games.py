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
    """
    Internal function for matrix games library'
    
    Computes a matrix with random entries
    
    Parameters
    ----------
    n : int
        number of columns.
    m : int
        number of rows.
    sparsity : boolean
        if "True" it returns a sparse matrix using the scipy.sparse
        library[1], otherwise it computes a full matrix with random
        entries from 0 to 
        
    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
    Raises
    ------
    IndexError
        if `axes` is larger than the last axis of `x`.
    See Also
    --------
    ifft : The inverse of `fft`.
    fft2 : The 2-D FFT.
    fftn : The N-D FFT.
    rfftn : The N-D FFT of real input.
    fftfreq : Frequency bins for given FFT parameters.
    next_fast_len : Size to pad input to for most efficient transforms
    Notes
    -----
    FFT (Fast Fourier Transform) refers to a way the discrete Fourier Transform
    (DFT) can be calculated efficiently, by using symmetries in the calculated
    terms. The symmetry is highest when `n` is a power of 2, and the transform
    is therefore most efficient for these sizes. For poorly factorizable sizes,
    `scipy.fft` uses Bluestein's algorithm [2]_ and so is never worse than
    O(`n` log `n`). Further performance improvements may be seen by zero-padding
    the input using `next_fast_len`.
    If ``x`` is a 1d array, then the `fft` is equivalent to ::
        y[k] = np.sum(x * np.exp(-2j * np.pi * k * np.arange(n)/n))
    The frequency term ``f=k/n`` is found at ``y[k]``. At ``y[n/2]`` we reach
    the Nyquist frequency and wrap around to the negative-frequency terms. So,
    for an 8-point transform, the frequencies of the result are
    [0, 1, 2, 3, -4, -3, -2, -1]. To rearrange the fft output so that the
    zero-frequency component is centered, like [-4, -3, -2, -1, 0, 1, 2, 3],
    use `fftshift`.
    Transforms can be done in single, double, or extended precision (long
    double) floating point. Half precision inputs will be converted to single
    precision and non-floating-point inputs will be converted to double
    precision.
    If the data type of ``x`` is real, a "real FFT" algorithm is automatically
    used, which roughly halves the computation time. To increase efficiency
    a little further, use `rfft`, which does the same calculation, but only
    outputs half of the symmetrical spectrum. If the data are both real and
    symmetrical, the `dct` can again double the efficiency, by generating
    half of the spectrum from half of the signal.
    When ``overwrite_x=True`` is specified, the memory referenced by ``x`` may
    be used by the implementation in any way. This may include reusing the
    memory for the result, but this is in no way guaranteed. You should not
    rely on the contents of ``x`` after the transform as this may change in
    future without warning.
    The ``workers`` argument specifies the maximum number of parallel jobs to
    split the FFT computation into. This will execute independent 1-D
    FFTs within ``x``. So, ``x`` must be at least 2-D and the
    non-transformed axes must be large enough to split into chunks. If ``x`` is
    too small, fewer jobs may be used than requested.
    References
    ----------
    .. [1] Cooley, James W., and John W. Tukey, 1965, "An algorithm for the
           machine calculation of complex Fourier series," *Math. Comput.*
           19: 297-301.
    .. [2] Bluestein, L., 1970, "A linear filtering approach to the
           computation of discrete Fourier transform". *IEEE Transactions on
           Audio and Electroacoustics.* 18 (4): 451-455.
    Examples
    --------
    >>> import scipy.fft
    >>> scipy.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))
    array([-2.33486982e-16+1.14423775e-17j,  8.00000000e+00-1.25557246e-15j,
            2.33486982e-16+2.33486982e-16j,  0.00000000e+00+1.22464680e-16j,
           -1.14423775e-17+2.33486982e-16j,  0.00000000e+00+5.20784380e-16j,
            1.14423775e-17+1.14423775e-17j,  0.00000000e+00+1.22464680e-16j])
    In this example, real input has an FFT which is Hermitian, i.e., symmetric
    in the real part and anti-symmetric in the imaginary part:
    >>> from scipy.fft import fft, fftfreq, fftshift
    >>> import matplotlib.pyplot as plt
    >>> t = np.arange(256)
    >>> sp = fftshift(fft(np.sin(t)))
    >>> freq = fftshift(fftfreq(t.shape[-1]))
    >>> plt.plot(freq, sp.real, freq, sp.imag)
    [<matplotlib.lines.Line2D object at 0x...>, <matplotlib.lines.Line2D object at 0x...>]
    >>> plt.show()
    """
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