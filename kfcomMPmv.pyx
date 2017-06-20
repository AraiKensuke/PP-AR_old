import numpy as _N
cimport numpy as _N
cimport cython
import time as _tm

"""
c functions
"""
cdef extern from "math.h":
    double sqrt(double)

#Ik = _N.identity(9)
#IkN= _N.tile(Ik, (750, 1, 1))

###  Most expensive operation here is the SVD
@cython.boundscheck(False)
@cython.wraparound(False)
def BSvec(F, iF, fx, fV, smXN):
    for n in xrange(N-1, -1, -1):
        xnN = (I-AF)*fx[n] + A * x 
        #   (I-AF)*fx     -  I-AF has a non-zero last row.  (I-AF)fx has only last component != 0.
        #   Ax
        VnN = (I - AF)*fV          
        #   -  I-AF has a V^{-1} on the RHS.  (I-AF)fV has 1 non-zero elmt (N,N)
        
