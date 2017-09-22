import numpy as _N
cimport numpy as _N
#import kfcomMPmv_ram as _kfcom
#import ram as _kfcom
import kfcomMPmv_ram as _kfcom
import time as _tm
cimport cython

import warnings
warnings.filterwarnings("error")

dDTYPE = _N.double
ctypedef _N.double_t dDTYPE_t

"""
p        AR order
Ftrgt    Ftrgt[0]  noise amp.  Ftrgt[1:]  AR(p) coeffs
f        freq.
f0       bandpass
f1
zr       amp. at band stop
"""

########################   FFBS
#def armdl_FFBS_1itrMP(y, Rv, F, q2, N, k, fx00, fV00):   #  approximation
@cython.boundscheck(False)
@cython.wraparound(False)
def armdl_FFBS_1itrMP(args):   #  approximation
    """
    for Multiprocessor, aguments need to be put into a list.
    """
    y  = args[0]
    Rv = args[1]
    F  = args[2]
    iF  = args[3]
    q2 = args[4]
    N  = args[5] 
    cdef int k  = args[6]
    fx00 = args[7]
    fV00 = args[8]

    fx = _N.empty((N + 1, k, 1))
    fV = _N.empty((N + 1, k, k))
    fx[0] = fx00
    fV[0] = fV00

    ##########  FF
    FFdv(y, Rv, N, k, F, q2, fx, fV)
    ##########  BS
    smXN = _N.random.multivariate_normal(fx[N,:,0], fV[N], size=1)
    smpls = _kfcom.BSvec(iF, N, k, q2, fx, fV, smXN)
    return [smpls, fx, fV]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def FFdv(double[::1] y, double[::1] Rv, N, long k, double[:, ::1] F, double q2, _fx, _fV):   #  approximate KF    #  k==1,dynamic variance
    #  do this until p_V has settled into stable values

    px = _N.empty((N + 1, k, 1))    #  naive and analytic calculated same way
    pV = _N.empty((N + 1, k, k))

    cdef double[:, :, ::1] fx = _fx
    cdef double[:, :, ::1] fV = _fV
    cdef double* p_y  = &y[0]
    cdef double* p_Rv  = &Rv[0]
    K     = _N.empty((N + 1, k, 1))
    cdef double[:, :, ::1] Kmv   = K  # forward filter
    cdef double* p_K              = &Kmv[0, 0, 0]

    #  need memory views for these
    #  F, fx, px need memory views
    #  K, KH
    #  IKH
    
    cdef double* p_F              = &F[0, 0]
    cdef double* p_fx              = &fx[0, 0, 0]
    cdef double* p_fV              = &fV[0, 0, 0]

    cdef double[:, :, ::1] pxmv   = px
    cdef double* p_px             = &pxmv[0, 0, 0]
    cdef double[:, :, ::1] pVmv   = pV
    cdef double* p_pV             = &pVmv[0, 0, 0]
    cdef int n, i, j, ii, jj, nKK, nK, ik, n_m1_KK, n_m1_K, i_m1_K, iik

    cdef double dd = 0, val, Kfac

    for n from 1 <= n < N + 1:
        nKK = n * k * k
        nK  = n*k
        n_m1_KK = (n-1) * k * k
        n_m1_K = (n-1) * k
        dd = 0
        #  prediction mean  (naive and analytic method are the same)
        for i in xrange(1, k):#  use same loop to copy and do dot product
            dd             += p_F[i]*p_fx[n_m1_K + i]
            p_px[nK + i] = p_fx[n_m1_K + (i-1)] # shift older state
        #p_px[nKK]          = dd + p_F[0]*p_fx[n_m1_KK]  #  1-step prediction 
        #p_px[nKK]          = dd + p_F[0]*p_fx[(n-1)*k]  #  1-step prediction 
        p_px[nK]          = dd + p_F[0]*p_fx[n_m1_K]  #  1-step prediction 


        #####  covariance, 1-step prediction
        ####  upper 1x1
        val = 0
        for ii in xrange(k):   
            iik = ii*k
            val += p_F[ii]*p_F[ii]*p_fV[n_m1_KK + iik + ii]
            for jj in xrange(ii+1, k):
                val += 2*p_F[ii]*p_F[jj]*p_fV[n_m1_KK + iik+jj]
        p_pV[nKK]  = val + q2
        ####  lower k-1 x k-1
        for ii in xrange(1, k):
            for jj in xrange(ii, k):
                p_pV[nKK+ ii*k+ jj] = p_pV[nKK+ jj*k+ ii] = p_fV[n_m1_KK + (ii-1)*k + jj-1]
        ####  (1 x k-1) and (k-1 x 1)
        for j in xrange(1, k):
            val = 0
            for ii in xrange(k):
                val += p_F[ii]*p_fV[n_m1_KK+ ii*k + j-1]
            p_pV[nKK + j] = val
            p_pV[nKK + j*k] = val
        ######  Kalman gain
        Kfac  = 1. / (p_pV[nKK] + p_Rv[n])  #  scalar
        for i in xrange(k):
            p_K[nK + i] = p_pV[nKK + i*k] * Kfac
        #################  filter mean
        for i in xrange(k):
            p_fx[nK+i] = p_px[nK+ i] + p_K[nK+ i]*(p_y[n] - p_px[nK])

            for j in xrange(i, k):
                p_fV[nKK+i*k+ j] = p_pV[nKK+ i*k+ j] - p_pV[nKK+j]*p_K[nK+i]
                p_fV[nKK+j*k + i] = p_fV[nKK+i*k+ j]

