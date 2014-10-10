import numpy as _N
import time as _tm

import warnings
warnings.filterwarnings("error")

"""
p        AR order
Ftrgt    Ftrgt[0]  noise amp.  Ftrgt[1:]  AR(p) coeffs
f        freq.
f0       bandpass
f1
zr       amp. at band stop
"""
cdef extern from "math.h":
    double exp(double)
    double sqrt(double)
    double log(double)
    double abs(double)

########################   FFBS
def armdl_FFBS_1itr(_d):   #  approximation
    ##########  FF
    FF1dv(_d)
    return BS1(_d)

def FF1dv(_d):   #  approximate KF    #  k==1,dynamic variance
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
        pV[n,0,0] = _d.F[0,0] * fV[n - 1,0,0] * _d.F[0,0] + GQGT

        mat  = 1 / (pV[n,0,0] + Rv[n])
        K[n,0,0] = pV[n,0,0]*mat
        fx[n,0,0]    = px[n,0,0] + K[n,0,0]*(_d.y[n] - px[n,0,0])
        fV[n,0,0] = (1 - K[n,0,0])* pV[n,0,0]

def BS1(_d):
    k     = _d.k
    fx    = _d.f_x
    fV    = _d.f_V

    nrands = _N.random.randn(_d.N+1)
    smX = _N.empty(_d.N+1)
    smX[_d.N] = fx[_d.N,0,0] + sqrt(fV[_d.N,0,0]) * nrands[_d.N]
    F0   = _d.F[0,0]
    F02  = F0*F0

    ##########  SMOOTHING step
    GQGT = _d.Q * _d.G[0,0]* _d.G[0,0]
    iGQGT= 1./GQGT

    FTiGQGTF = iGQGT*F02
    #for n in xrange(_d.N - 1, -1, -1):
    for n from _d.N - 1 >= n > -1:
        Sig  = 1/(FTiGQGTF + 1/(fV[n,0,0]))
        p1   = fV[n,0,0]* F0
        p2   = 1/(F02*fV[n,0,0] +GQGT)
        p3   = smX[n+1] - F0*fx[n,0,0]
        M    = fx[n,0,0] + p1* p2* p3
        smX[n]= M + sqrt(Sig) * nrands[n]
    return smX
