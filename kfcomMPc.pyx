import numpy as _N
cimport numpy as _N
cimport cython
import time as _tm


dDTYPE = _N.double
ctypedef _N.double_t dDTYPE_t

cdef extern from "math.h":
    double sqrt(double)

#@cython.boundscheck(False)
#@cython.wraparound(False)

def BSvec(_N.ndarray[dDTYPE_t, ndim=2] F, int N, int k, _N.ndarray[dDTYPE_t, ndim=2]GQGT, _N.ndarray[dDTYPE_t, ndim=3] fx, _N.ndarray[dDTYPE_t, ndim=3] fV, _N.ndarray[dDTYPE_t, ndim=2] smXN):
    #t1 = _tm.time()
    #print "BSvec"
    Ik      = _N.identity(k)    
    IkN  =  _N.tile(Ik, (N+1, 1, 1))
    smX = _N.empty((N+1, k))   #  where to store our samples
    smX[N] = smXN[:]

    fFT    = _N.dot(fV, F.T)
    #  sum F_{il} V_{lm} F_{mj}
    FfFT   = _N.einsum("il,nlj->nij", F, fFT)

    iv     = _N.linalg.inv(FfFT + _N.tile(GQGT, (N+1,1,1)))
    A      = _N.einsum("nik,nkj->nij", fFT, iv)
    INAF   = IkN - _N.dot(A, F)
    PtN    = _N.einsum("nik,nkj->nij", INAF, fV)  #  covarainces
    ##  NOW multivariate normal
    mvn1   = _N.random.multivariate_normal(_N.zeros(k), Ik, size=(N+1))
    S,V,D  = _N.linalg.svd(PtN)
    Vs     = _N.sqrt(V)
    VsRn2  =  Vs*mvn1
    cdef _N.ndarray[dDTYPE_t, ndim=2] zrmn  = _N.einsum("njk,nk->nj", S, VsRn2)
    #zrmn  = _N.einsum("njk,nk->nj", S, VsRn2)
    #zrmn   = _N.einsum("njk,nk->nj", S, VsRn2).reshape((N+1), k, 1)

    #  out of order calculation.  one of the terms can be calculated
    INAFfx = _N.einsum("nj,nj->n", INAF[:, k-1], fx[:, :, 0])
    last   = _N.zeros(k)
    last[k-1] = 1

    #  temp storage
    cdef _N.ndarray[dDTYPE_t, ndim=1] Asx  = _N.empty(k, dtype=dDTYPE)
    cdef _N.ndarray[dDTYPE_t, ndim=1] INAFfxpAsx  = _N.empty(k, dtype=dDTYPE)
    #Asx = _N.empty(k)
    #INAFfxpAsx = _N.empty(k)
    cdef int n, ik
    for n in xrange(N - 1, -1, -1):
        #  (I - AF) x
        #smX[n] = zrmn[n, :, 0] + INAFfx[n]*last + _N.dot(A[n], smX[n+1])
        _N.dot(A[n], smX[n+1], out=Asx)
        _N.add(INAFfx[n]*last, Asx, out=INAFfxpAsx)
        _N.add(zrmn[n], INAFfxpAsx, out=smX[n])

    return smX

