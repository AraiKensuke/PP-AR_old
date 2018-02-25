import numpy as _N
cimport numpy as _N
import kfcomMPmv as _kfcom
import time as _tm
import cython

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

@cython.boundscheck(False)
@cython.wraparound(False)

########################   FFBS
#def armdl_FFBS_1itrMP(y, Rv, F, q2, N, k, fx00, fV00):   #  approximation
def armdl_FFBS_1itrMP(args):   #  approximation
    """
    for Multiprocessor, aguments need to be put into a list.
    """
    y  = args[0]
    Rv = args[1]
    F  = args[2]
    q2 = args[3]
    N  = args[4] 
    k  = args[5]
    fx00 = args[6]
    fV00 = args[7]

    fx = _N.empty((N + 1, k, 1))
    fV = _N.empty((N + 1, k, k))
    #fx[0, :, 0] = fx00
    fx[0] = fx00
    fV[0] = fV00
    GQGT   = _N.zeros((k, k))
    GQGT[0, 0] = q2

    ##########  FF
    FFdv(y, Rv, N, k, F, GQGT, fx, fV)
    ##########  BS
    smXN = _N.random.multivariate_normal(fx[N,:,0], fV[N], size=1)
    #t1 = _tm.time()
    smpls = _kfcom.BSvec(F, N, k, GQGT, fx, fV, smXN)
    #t2 = _tm.time()
    #print (t2-t1)
    return [smpls, fx, fV]

def FFdv(y, Rv, N, k, F, GQGT, fx, fV):   #  approximate KF    #  k==1,dynamic variance
    #print "FFdv"
    #  do this until p_V has settled into stable values
    H       = _N.zeros((1, k))          #  row vector
    H[0, 0] = 1

    Ik      = _N.identity(k)
    px = _N.empty((N + 1, k, 1))
    pV = _N.empty((N + 1, k, k))
    #cdef _N.ndarray[dDTYPE_t, ndim=3] px = _N.empty((N + 1, k, 1))
    #cdef _N.ndarray[dDTYPE_t, ndim=3] pV = _N.empty((N + 1, k, k))

    K     = _N.empty((N + 1, k, 1))
    """
    temporary storage
    """
    #Hpx   = _N.empty((1, 1))
    KH    = _N.empty((k, k))
    IKH   = _N.empty((k, k))
    VFT   = _N.empty((k, k))
    FVFT  = _N.empty((k, k))

    for n from 1 <= n < N + 1:   #  n is not a trial
        _N.dot(F, fx[n - 1], out=px[n])
        _N.dot(fV[n - 1], F.T, out=VFT)
        _N.dot(F, VFT, out=FVFT)
        _N.add(FVFT, GQGT, out=pV[n])
        mat  = 1 / (pVmv[n, 0, 0] + Rv[n])   #  mat is a scalar
        _N.dot(pV[n], mat*H.T, out=K[n])   #  vector

        # px + K(y - o - Hpx)  K column vec, (y-o-Hpx) is scalar
        #_N.dot(H, px[n], out=Hpx)
        #KyHpx = K[n]* (y[n] - Hpx[0, 0])
        KyHpx = K[n]* (y[n] - px[n, 0, 0])
        _N.add(px[n], KyHpx, out=fx[n])

        # (I - KH)pV
        _N.dot(K[n], H, out=KH)
        _N.subtract(Ik, KH, out=IKH)
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


