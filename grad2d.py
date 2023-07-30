import numpy as np
from numpy import sqrt, sum, abs, max, maximum, logspace, exp, log, log10, zeros
from numpy.random import normal, randn, choice
from numpy.linalg import norm
from scipy.signal import convolve2d


def check_adjoint(A,At,dims):
    # start with this line - create a random input for A()
    x = normal(size=dims)+1j*normal(size=dims)
    Ax = A(x)
    y = normal(size=Ax.shape)+1j*normal(size=Ax.shape)
    Aty = At(y)
    # compute the Hermitian inner products
    inner1 = np.sum(np.conj(Ax)*y)
    inner2 = np.sum(np.conj(x)*Aty)
    # report error
    rel_error = np.abs(inner1-inner2)/np.maximum(np.abs(inner1),np.abs(inner2))
    if rel_error < 1e-10:
        print('Adjoint Test Passed, rel_diff = %s'%rel_error)
        return True
    else:
        print('Adjoint Test Failed, rel_diff = %s'%rel_error)
        return False

# For total-variation

kernel_h = [[1,-1,0]]
kernel_v = [[1],[-1],[0]]

# Do not modify ANYTHING in this cell.
def gradh(x):
    """Discrete gradient/difference in horizontal direction"""
    return convolve2d(x,kernel_h, mode='same', boundary='wrap')
def gradv(x):
    """Discrete gradient/difference in vertical direction"""
    return convolve2d(x,kernel_v, mode='same', boundary='wrap')
def grad2d(x):
    """The full gradient operator: compute both x and y differences and return them all.  The x and y
    differences are stacked so that rval[0] is a 2D array of x differences, and rval[1] is the y differences."""
    return np.stack([gradh(x),gradv(x)])

def gradht(x):
    """Adjoint of gradh"""
    kernel_ht = [[0,-1,1]]
    return convolve2d(x,kernel_ht, mode='same', boundary='wrap')
def gradvt(x):
    """Adjoint of gradv"""
    kernel_vt = [[0],[-1],[1]]
    return convolve2d(x,kernel_vt, mode='same', boundary='wrap')
def divergence2d(x):
    "The methods is the adjoint of grad2d."
    return gradht(x[0])+gradvt(x[1])

