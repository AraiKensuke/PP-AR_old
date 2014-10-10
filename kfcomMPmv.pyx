import numpy as _N
cimport numpy as _N
cimport cython

"""
c functions
"""
cdef extern from "math.h":
    double sqrt(double)


@cython.boundscheck(False)
@cython.wraparound(False)
def BSvec(F, N, _N.intp_t k, GQGT, fx, fV, smXN):
    Ik      = _N.identity(k)    
    IkN   =  _N.tile(Ik, (N+1, 1, 1))
    smX   = _N.empty((N+1, k))   #  where to store our samples
    cdef double[:, ::1] smXmv = smX   #  memory view
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
    zrmn   = _N.einsum("njk,nk->nj", S, VsRn2)
    cdef double[:, ::1] zrmnmv = zrmn

    #  out of order calculation.  one of the terms can be calculated
    INAFfx = _N.einsum("nj,nj->n", INAF[:, k-1], fx[:, :, 0])
    cdef double[::1] INAFfxmv = INAFfx
    last   = _N.zeros(k)
    last[k-1] = 1

    #  temp storage
    Asx = _N.empty(k)
    cdef double[::1] Asxmv = Asx

    cdef _N.intp_t t, i, n

    for n in xrange(N - 1, -1, -1):
        #smX[n] = zrmn[n, :, 0] + INAFfx[n]*last + _N.dot(A[n], smX[n+1])
        _N.dot(A[n], smX[n+1], out=Asx)
        smXmv[n, k-1] = zrmnmv[n, k-1] + Asxmv[k-1] + INAFfxmv[n]
        for i from 0 <= i < k-1:
            smXmv[n, i] = zrmnmv[n, i] + Asxmv[i]

    return smX

