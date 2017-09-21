import numpy as _N
cimport numpy as _N
import kfcomMPmv_ram as _kfcom
#import kfcomMPmv as _kfcom
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
    q2 = args[3]
    N  = args[4] 
    cdef int k  = args[5]
    fx00 = args[6]
    fV00 = args[7]

    fx = _N.empty((N + 1, k, 1))
    fV = _N.empty((N + 1, k, k))
    fx[0] = fx00
    fV[0] = fV00
    GQGT   = _N.zeros((k, k))
    GQGT[0, 0] = q2

    ##########  FF
    #t1 = _tm.time()
    print "doing FFdv_orig"
    FFdv_orig(y, Rv, N, k, F, GQGT, fx, fV)
    #t2 = _tm.time()
    ##########  BS
    smXN = _N.random.multivariate_normal(fx[N,:,0], fV[N], size=1)
    #t1 = _tm.time()
    #smpls = _kfcom.BSvec(F, N, k, GQGT, fx, fV, smXN)
    smpls = _kfcom.BSvec_cmp_various(F, N, k, GQGT, fx, fV, smXN)
    #t2 = _tm.time()
    #print (t2-t1)
    return [smpls, fx, fV]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def FFdv(double[::1] y, double[::1] Rv, N, long k, F, GQGT, fx, fV):   #  approximate KF    #  k==1,dynamic variance
    #print "FFdv"
    #  do this until p_V has settled into stable values
    H       = _N.zeros((1, k))          #  row vector
    H[0, 0] = 1
    cdef double q2 = GQGT[0, 0]

    Ik      = _N.identity(k)
    px = _N.empty((N + 1, k, 1))    #  naive and analytic calculated same way
    fx_ram   = _N.empty((N+1, k, 1))
    pV = _N.empty((N + 1, k, k))
    pV_ram = _N.empty((N+1, k, k))
    fV_ram = _N.empty((N+1, k, k))


    cdef double* p_y  = &y[0]
    cdef double* p_Rv  = &Rv[0]

    K     = _N.empty((N + 1, k, 1))
    K_ram     = _N.empty((N + 1, k, 1))
    cdef double[:, :, ::1] K_rammv   = K_ram  # forward filter
    cdef double* p_K_ram              = &K_rammv[0, 0, 0]

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
    cdef double* p_F              = &Fmv[0, 0]
    cdef double[:, :, ::1] fxmv   = fx  # forward filter
    cdef double* p_fx              = &fxmv[0, 0, 0]
    cdef double[:, :, ::1] fVmv   = fV  # forward filter
    cdef double* p_fV              = &fVmv[0, 0, 0]

    cdef double[:, :, ::1] pxmv   = px
    cdef double* p_px             = &pxmv[0, 0, 0]
    cdef double[:, :, ::1] pVmv   = pV
    cdef double* p_pV             = &pVmv[0, 0, 0]


    cdef double[:, :, ::1] fx_ram_mv  = fx_ram
    cdef double* p_fx_ram  = &fx_ram_mv[0, 0, 0]

    cdef double[:, :, ::1] pV_ram_mv = pV_ram
    cdef double* p_pV_ram  = &pV_ram_mv[0, 0, 0]
    cdef double[:, :, ::1] fV_ram_mv = fV_ram
    cdef double* p_fV_ram  = &fV_ram_mv[0, 0, 0]

    cdef double[:, :, ::1] Kmv    = K
    cdef double[:, ::1] IKHmv     = IKH


    cdef int n, i, j, ii, jj, nKK, nK, ik, n_m1_KK, i_m1_K, iik

    cdef double dd = 0, val, Kfac

    for n from 1 <= n < N + 1:
        t2t1 = 0
        t3t2 = 0
        t1 = _tm.time()
        nKK = n * k * k
        nK  = n*k
        n_m1_KK = (n-1) * k * k
        dd = 0
        #  prediction mean  (naive and analytic method are the same)
        for i in xrange(1, k):#  use same loop to copy and do dot product
            ik             = i*k
            dd             += p_F[i]*p_fx[n_m1_KK + ik]
            p_px[nKK + ik] = p_fx[n_m1_KK + (i-1)*k] # shift older state
        p_px[nKK]          = dd + p_F[0]*p_fx[n_m1_KK]  #  1-step prediction 


        #####  covariance, 1-step prediction
        ####  upper 1x1

        val = 0
        for ii in xrange(k):   
            iik = ii*k
            val += p_F[ii]*p_F[ii]*p_fV[n_m1_KK + iik + ii]
            for jj in xrange(ii+1, k):
                val += 2*p_F[ii]*p_F[jj]*p_fV[n_m1_KK + iik+jj]
        p_pV_ram[nKK]  = val + q2
        ####  lower k-1 x k-1
        for ii in xrange(1, k):
            for jj in xrange(ii, k):
                p_pV_ram[nKK+ ii*k+ jj] = p_pV_ram[nKK+ jj*k+ ii] = p_fV[n_m1_KK + (ii-1)*k + jj-1]
                #                 = p_fV[n_m1_KK + (ii-1)*k + jj]
        ####  (1 x k-1) and (k-1 x 1)
        for j in xrange(1, k):
            val = 0
            for ii in xrange(k):
                val += p_F[ii]*p_fV[n_m1_KK+ ii*k + j-1]
            p_pV_ram[nKK + j] = val
            p_pV_ram[nKK + j*k] = val

        t2 = _tm.time()
        #  naive method
        _N.dot(fV[n - 1], F.T, out=VFT)
        _N.dot(F, VFT, out=pV[n])          #  prediction
        pVmv[n, 0, 0]    += q2
        t3 = _tm.time()
        t2t1 += t2-t1
        t3t2 += t3-t2

        # print "----------%%%%%%%%%%%%%%%%%%%%%%%%"
        # print (t2-t1)
        # print (t3-t2)
        # print pV_ram[n]
        # print pV[n]
        # print "----------"


        t1 = _tm.time()

        ################################################   ANALYTIC

        ######  Kalman gain
        Kfac  = 1. / (p_pV_ram[nKK] + p_Rv[n])  #  scalar
        for i in xrange(k):
            p_K_ram[nK + i] = p_pV_ram[nKK + i*k] * Kfac

        #################  filter mean
        for i in xrange(k):
            p_fx_ram[nK+i] = p_px[nK+ i] + p_K_ram[nK+ i]*(p_y[n] - p_px[nK])

            for j in xrange(i, k):
                p_fV_ram[nKK+i*k+ j] = p_pV_ram[nKK+ i*k+ j] - p_pV_ram[nKK+j]*p_K_ram[nK+i]
                p_fV_ram[nKK+j*k + i] = p_fV_ram[nKK+i*k+ j]


        ###############################################   NAIVE
        t2 = _tm.time()
        ######  Kalman gain
        mat  = 1 / (pVmv[n, 0, 0] + Rv[n])  #  scalar
        K[n, :, 0] = pV[n, :, 0] * mat

        #################  filter mean
        _N.multiply(K[n], y[n] - pxmv[n, 0, 0], out=KyHpx)
        _N.add(px[n], KyHpx, out=fx[n])

        # (I - KH), KH is zeros except first column
        IKHmv[0, 0] = 1 - Kmv[n, 0, 0]
        for i in xrange(1, k):
            IKHmv[i, 0] = -Kmv[n, i, 0]
        # (I - KH)
        #################  filter covariance  naive
        _N.dot(IKH, pV[n], out=fV[n])
        t3 = _tm.time()

        t2t1 += t2-t1
        t3t2 += t3-t2


        # print "!!!!!!!!!!!!!!!!!----------------"
        # print "t2t1   %.3e" % t2t1
        # print "t3t2   %.3e" % t3t2
        # print fx[n]
        # print fx_ram[n]

        # print fV[n]
        # print fV_ram[n]









def FFdv_orig(double[::1] y, Rv, N, k, F, GQGT, fx, fV):   #  approximate KF    #  k==1,dynamic variance
    #print "FFdv"
    #  do this until p_V has settled into stable values
    H       = _N.zeros((1, k))          #  row vector
    H[0, 0] = 1
    cdef double q2 = GQGT[0, 0]

    Ik      = _N.identity(k)
    px = _N.empty((N + 1, k, 1))
    pV = _N.empty((N + 1, k, k))
    pV_ram = _N.empty((N+1, k, k))
    cdef double[:, :, ::1] pV_ram_mv = pV_ram
    cdef double* p_pV_ram  = &pV_ram_mv[0, 0, 0]

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
    cdef double[:, :, ::1] fxmv   = fx  # forward filter
    cdef double[:, :, ::1] pxmv   = px
    cdef double[:, :, ::1] pVmv   = pV
    #cdef double[::1] Rvmv   = Rv
    cdef double[:, :, ::1] Kmv    = K
    cdef double[:, ::1] IKHmv     = IKH

    cdef _N.intp_t n, i

    cdef double dd = 0
    for n from 1 <= n < N + 1:
        dd = 0
        #  prediction mean
        for i in xrange(1, k):#  use same loop to copy and do dot product
            dd += Fmv[0, i]*fxmv[n-1, i, 0]
            pxmv[n, i, 0] = fxmv[n-1, i-1, 0] # shift older state
        pxmv[n, 0, 0] = dd + Fmv[0, 0]*fxmv[n-1, 0, 0]  #  1-step prediction 


        #  covariance, 1-step prediction
        ####  upper 1x1
        pV_ram[n, 0, 0] = 0
        for ii in xrange(k):   
            pV_ram[n,0,0] += F[0,ii]*F[0,ii]*fV[n-1,ii,ii]
            for jj in xrange(ii+1, k):
                pV_ram[n,0,0] += 2*F[0,ii]*F[0,jj]*fV[n-1,ii,jj]
        pV_ram[n,0,0]  += q2
        ####  lower k-1 x k-1
        for ii in xrange(1, k):
            for jj in xrange(ii, k):
                pV_ram[n, ii, jj] = fV[n-1,ii-1,jj-1]
                pV_ram[n, jj, ii] = fV[n-1,ii-1,jj-1]
        ####  (1 x k-1) and (k-1 x 1)
        for j in xrange(1, k):
            val = 0
            for ii in xrange(k):
                val += F[0, ii]*fV[n-1, ii, j-1]
            pV_ram[n, 0, j] = val
            pV_ram[n, j, 0] = val

        
        #  naive method
        _N.dot(fV[n - 1], F.T, out=VFT)
        _N.dot(F, VFT, out=pV[n])          #  prediction
        pVmv[n, 0, 0]    += q2

        """
        print "----------"
        print pV_ram[n]
        print pV[n]
        print "----------"
        """

        ######  Kalman gain
        mat  = 1 / (pVmv[n, 0, 0] + Rv[n])  #  scalar
        K[n, :, 0] = pV[n, :, 0] * mat

        #################  filter mean
        _N.multiply(K[n], y[n] - pxmv[n, 0, 0], out=KyHpx)
        _N.add(px[n], KyHpx, out=fx[n])

        # (I - KH), KH is zeros except first column
        IKHmv[0, 0] = 1 - Kmv[n, 0, 0]
        for i in xrange(1, k):
            IKHmv[i, 0] = -Kmv[n, i, 0]
        # (I - KH)
        #################  filter covariance
        _N.dot(IKH, pV[n], out=fV[n])


def FF1dv(_d, offset=0):   #  approximate KF    #  k==1,dynamic variance
    GQGT    = _d.G[0,0]*_d.G[0, 0] * _d.Q
    k     = _d.k
    px    = _d.p_x
    pV    = _d.p_V
    fx    = _d.f_x
    fV    = _d.f_V
    Rv    = _d.Rv
    K     = _d.K

    #  do this until p_V has settled into stable values

    for n from 1 <= n < _d.N + 1:
        px[n,0,0] = _d.F[0,0] * fx[n - 1,0,0]
#        pV[n,0,0] = _d.F[0,0] * fV[n - 1,0,0] * _d.F.T[0,0] + GQGT
        pV[n,0,0] = _d.F[0,0] * fV[n - 1,0,0] * _d.F[0,0] + GQGT
        #_d.p_Vi[n,0,0] = 1/pV[n,0,0]

#        mat  = 1 / (_d.H[0,0]*pV[n,0,0]*_d.H[0,0] + Rv[n])
        mat  = 1 / (pV[n,0,0] + Rv[n])
#        K[n,0,0] = pV[n]*_d.H[0,0]*mat
        K[n,0,0] = pV[n,0,0]*mat
#        fx[n,0,0]    = px[n,0,0] + K[n,0,0]*(_d.y[n] - offset[n] - _d.H[0,0]* px[n,0,0])
#        fx[n,0,0]    = px[n,0,0] + K[n,0,0]*(_d.y[n] - _d.H[0,0]* px[n,0,0])
        fx[n,0,0]    = px[n,0,0] + K[n,0,0]*(_d.y[n] - px[n,0,0])
#        fV[n,0,0] = (1 - K[n,0,0]* _d.H[0,0])* pV[n,0,0]
        fV[n,0,0] = (1 - K[n,0,0])* pV[n,0,0]


