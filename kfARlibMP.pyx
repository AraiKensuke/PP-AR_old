import numpy as _N
cimport numpy as _N   #  need to add include path  /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include
import kfcomMP as _kfcom

import scipy.optimize as _sco

import time as _tm

import warnings
warnings.filterwarnings("error")

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
def armdl_FFBS_1itrMP(args):   #  approximation
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
    smpls = _kfcom.BSvec(F, N, k, GQGT, fx, fV, smXN)
    return [smpls, fx, fV]

def FFdv(y, Rv, N, k, F, GQGT, fx, fV):   #  approximate KF    #  k==1,dynamic variance
    #print "FFdv"
    #  do this until p_V has settled into stable values
    H       = _N.zeros((1, k))          #  row vector
    H[0, 0] = 1

    Ik      = _N.identity(k)
    px = _N.empty((N + 1, k, 1))
    pV = _N.empty((N + 1, k, k))

    K     = _N.empty((N + 1, k, 1))
    """
    temporary storage
    """
    Hpx   = _N.empty((1, 1))
    KH    = _N.empty((k, k))
    IKH   = _N.empty((k, k))
    VFT   = _N.empty((k, k))
    FVFT  = _N.empty((k, k))

    for n from 1 <= n < N + 1:
        _N.dot(F, fx[n - 1], out=px[n])
        _N.dot(fV[n - 1], F.T, out=VFT)
        _N.dot(F, VFT, out=FVFT)
        _N.add(FVFT, GQGT, out=pV[n])
        mat  = 1 / (pV[n, 0, 0] + Rv[n])
        _N.dot(pV[n], mat*H.T, out=K[n])   #  vector

        # px + K(y - o - Hpx)  K column vec, (y-o-Hpx) is scalar
        _N.dot(H, px[n], out=Hpx)
        KyHpx = K[n]* (y[n] - Hpx[0, 0])
        _N.add(px[n], KyHpx, out=fx[n])

        # (I - KH)pV
        _N.dot(K[n], H, out=KH)
        _N.subtract(Ik, KH, out=IKH)
        _N.dot(IKH, pV[n], out=fV[n])
