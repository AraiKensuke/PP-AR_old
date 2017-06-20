import numpy as _N
cimport numpy as _N
import kfcomMPmv as _kfcom
import time as _tm
cimport cython

import warnings
warnings.filterwarnings("error")

dDTYPE = _N.double
ctypedef _N.double_t dDTYPE_t

"""
c functions
"""
cdef extern from "math.h":
    double exp(double)
    double sqrt(double)
    double log(double)
    double abs(double)
"""
p        AR order
Ftrgt    Ftrgt[0]  noise amp.  Ftrgt[1:]  AR(p) coeffs
f        freq.
f0       bandpass
f1
zr       amp. at band stop
"""

cdef double H       = _N.zeros((1, k))          #  row vector
H[0, 0] = 1

########################   FFBS
#def armdl_FFBS_1itrMP(y, Rv, F, q2, N, k, fx00, fV00):   #  approximation
@cython.boundscheck(False)
@cython.wraparound(False)

#  need a GQGT  M x k x k

def armdl_FFBS_1itrOMP(double[:, ::1] y, double[:, ::1] Rv, double[:, ::1] Fs, double q2, int Ns, int k, double[:, :, :, ::1] fx, double[:, :, :, ::1] fV, double[:, ::1] randns):   #  approximation
    """
    y  is TR x (N+1)
    Rv is TR x (N+1)
    Fs is k x k
    fx is TR x (N+1) x k x 1
    fV is TR x (N+1) x k x k
    """
    cdef int tr
    cdef double *p_y   = &y[0, 0]
    cdef double *p_Rv  = &Rv[0, 0]
    cdef double *p_Fs  = &Fs[0, 0]
    cdef double *p_fx  = &fx[0, 0, 0, 0]
    cdef double *p_fV  = &fV[0, 0, 0, 0]
    cdef double *p_GQGT= &GQGT[0, 0]

    with nogil, parallel(num_threads=nthrds):
        for tr in prange(TR):
            FFdv(double *y, double *Rv, int N, int k, double *F, double *GQGT, double *fx, double *fV) nogil:   #  approximate KF    #  k==1,dynamic variance
            FFdv(p_y[tr], p_Rv[tr], Ns, k, p_Fs, p_GQGT, p_fx, p_fV)
            smXN = 
            smXN = gsl_ran_multivariate_gaussian(fx[N,:,0], fV[N], size=1)
            BSvecSVD_OMP(F, N, k, GQGT, fx, fV, smXN)
            
    #fx[0, :, 0] = fx00
    GQGT   = _N.zeros((k, k))
    GQGT[0, 0] = q2

    ##########  FF
    #t1 = _tm.time()
    FFdv(y, Rv, N, k, F, GQGT, fx, fV)
    #t2 = _tm.time()
    #print "FFdv  %.3f" % (t2-t1)
    ##########  BS
    smXN =     gsl_ran_multivariate_gaussian
_N.random.multivariate_normal(fx[N,:,0], fV[N], size=1)
    #t1 = _tm.time()
    #smpls = _kfcom.BSvecChol(F, N, k, GQGT, fx, fV, smXN)
    smpls = _kfcom.BSvecSVD_new(F, N, k, GQGT, fx, fV, smXN)
    #t2 = _tm.time()
    #print (t2-t1)
    return [smpls, fx, fV]

#  reuse Ik, px, pv
#  K  Kalman gain



cdef FFdv(double *y, double *Rv, int N, int k, double *F, double *GQGT, double *fx, double *fV) nogil:   #  approximate KF    #  k==1,dynamic variance
    """
    """
    #print "FFdv"
    #  do this until p_V has settled into stable values
    cdef double q2 = GQGT[0, 0]

    px = _N.empty((N + 1, k, 1))
    pV = _N.empty((N + 1, k, k))

    K     = _N.empty((N + 1, k, 1))
    """
    temporary storage
    """
    IKH   = _N.eye(k)        #  only contents of first column modified
    VFT   = _N.empty((k, k))
    FVFT  = _N.empty((k, k))
    KyHpx = _N.empty((k, 1))

    #  need memory views for these
    #  F, fx, px need memory views
    #  K, KH
    #  IKH
    
    cdef double[:, ::1] Fmv       = F
    cdef double[:, :, ::1] fxmv   = fx
    cdef double[:, :, ::1] pxmv   = px
    cdef double[:, :, ::1] pVmv   = pV
    cdef double[::1] Rvmv   = Rv
    cdef double[:, :, ::1] Kmv    = K
    cdef double[:, ::1] IKHmv     = IKH

    cdef _N.intp_t n, i

    cdef double dd = 0
    for n from 1 <= n < N + 1:
        dd = 0
        for i in xrange(1, k):#  use same loop to copy and do dot product
            dd += Fmv[0, i]*fxmv[n-1, i, 0]
            pxmv[n, i, 0] = fxmv[n-1, i-1, 0]
        pxmv[n, 0, 0] = dd + Fmv[0, 0]*fxmv[n-1, 0, 0]

        _N.dot(fV[n - 1], F.T, out=VFT)
        _N.dot(F, VFT, out=pV[n])
        pVmv[n, 0, 0]    += q2
        mat  = 1 / (pVmv[n, 0, 0] + Rvmv[n])  #  scalar

        K[n, :, 0] = pV[n, :, 0] * mat

        _N.multiply(K[n], y[n] - pxmv[n, 0, 0], out=KyHpx)
        _N.add(px[n], KyHpx, out=fx[n])

        # (I - KH), KH is zeros except first column
        IKHmv[0, 0] = 1 - Kmv[n, 0, 0]
        for i in xrange(1, k):
            IKHmv[i, 0] = -Kmv[n, i, 0]
        # (I - KH)
        #  k x k
        _N.dot(IKH, pV[n], out=fV[n])



###  Most expensive operation here is the SVD
@cython.boundscheck(False)
@cython.wraparound(False)
cdef BSvecSVD_new(F, N, _N.intp_t k, GQGT, fx, fV, smXN) nogil:
    #print "SVD"
    #global Ik, IkN
    Ik      = _N.identity(k)
    IkN   =  _N.tile(Ik, (N+1, 1, 1))
    smX   = _N.empty((N+1, k))   #  where to store our samples
    cdef double[:, ::1] smXmv = smX   #  memory view
    smX[N] = smXN[:]

    fFT     = _N.empty((N+1, k, k))    
    _N.dot(fV, F.T, out=fFT)  # dot([N+1 x k x k], [k, k])
    FfFTr     = _N.empty((k, k, N+1))
    _N.dot(F, fFT.T, out=FfFTr)  #  FfFTr[:, :, n]  is symmetric

    ##  was doing B^{-1}, but we only want result of operation on B^{-1}.
    B      = FfFTr.T + _N.tile(GQGT, (N+1,1,1))
    A = _N.transpose(_N.linalg.solve(B, _N.transpose(fFT, axes=(0, 2, 1))), axes=(0, 2, 1))

    INAF   = IkN - _N.dot(A, F)
    PtN = _N.einsum("nj,nj->n", INAF[:, k-1], fV[:,k-1])

    #print PtN

    mvn1   = _N.random.randn(N+1)  #  
    zrmn     = _N.sqrt(PtN)*mvn1

    #  out of order calculation.  one of the terms can be calculated
    INAFfx = _N.einsum("nj,nj->n", INAF[:, k-1], fx[:, :, 0])  #  INAF last row only
    cdef double[::1] INAFfxmv = INAFfx
    last   = _N.zeros(k)
    last[k-1] = 1

    #  temp storage
    Asx = _N.empty(k)
    cdef double[::1] Asxmv = Asx

    cdef _N.intp_t t, i, n

    for n in xrange(N - 1, -1, -1):
        _N.dot(A[n], smX[n+1], out=smX[n])
        smXmv[n, k-1] += zrmn[n] + INAFfxmv[n]


    return smX

