#  Fit PSTH with spike history
import patsy
import scipy.optimize as _sco
import os
import numpy as _N
import matplotlib.pyplot as _plt
from utildirs import setFN
from shutil import copyfile
from kstat import ranFromPercentile

from fitPSTHhistlib import h_dL, h_d2L, h_L, h_L_func, mkBounds

nbs1  = None;  knts1 = None
nbs2  = None;  knts2 = None
TM    = None;  M0    = None;    M1    = None
Gm    = None;  B     = None    #  spline basis
doPh  = True;  doAl  = True
_dat  = None;  dat   = None
dt    = 0.001
dCols = 3
N     = None;  M     = None    #  number of data points per trial, # trials
aSi   = None;  phiSi = None
mL    = -1;
pctl  = None;
sts   = None;  itvs  = None

######  INITIALIZE
def init(ebf):
    copyfile("%s.py" % ebf, "%(s)s/%(s)s.py" % {"s" : ebf, "to" : setFN("%s.py" % ebf, dir=ebf, create=True)})
    global nbs1, knts1, nbs2, knts2, TM, Gm, B, N, M, _dat, dat, aSi, phiSi, M0, M1, pctl
    _dat   = _N.loadtxt("xprbsdN.dat")

    N     = _dat.shape[0]            #  how many bins per trial
    M     = _dat.shape[1] / dCols    #  TRIALS
    if (M0 != None) and (M0 < 0):
        raise Exception("M0 if set needs to be >= 0")
    if (M1 != None) and ((M1 > M) or (M1 < 0)):
        raise Exception("M1 if set needs to be <= M and >= 0")
    if (M0 != None) or (M1 != None):
        if (M0 == None):
            M0 = 0
        if (M1 == None):
            M1 = M
        dat = _dat[:, M0*dCols:M1*dCols]
        M   = M1 - M0
    else:
        dat = _dat
        M0  = 0
        M1  = M

    B     = patsy.bs(_N.linspace(0., dt*(N-1), N), df=nbs1, knots=knts1, include_intercept=True)
    nbs1  = B.shape[1]
    B     = B.T

    _Gm   = patsy.bs(_N.linspace(0, dt*(TM-1), TM), knots=knts2, df=nbs2, include_intercept=True)
    nbs2 = _Gm.shape[1]       #  in case used knts2

    Gm = _N.zeros((N, nbs2))
    Gm[0:TM] = _Gm
    Gm = Gm.T

    pctl  = _N.loadtxt("isis0pctl.dat")

######  fitPSTH routine    
def fitPSTH(aS=None, phiS=None):   #  ebf  __exec_base_fn__
    global aSi, phiSi, M, pctl, sts, itvs
    aSi   = aS
    phiSi = phiS
    sts   = []   #  will include one dummy spike
    itvs  = []
    rpsth = []

    for tr in xrange(M):
        itvs.append([])
        lst = _N.where(dat[:, dCols*tr + 2] == 1)[0].tolist()
        lst.insert(0, int(-ranFromPercentile(pctl, lst[0])))    #  one dummy spike
        sts.append(_N.array(lst))
        rpsth.extend(lst)
        Lm  = len(lst) - 1    #  number of spikes this trial

        #  itvs is 1 bin shifted
        for i in xrange(Lm):
            itvs[tr].append([lst[i]+1, lst[i+1]+1]) 
        itvs[tr].append([lst[Lm]+1, N-1+1])

    if aS == None:
        #####  Initialize
        bnsz   = 50
        h, bs = _N.histogram(rpsth, bins=_N.linspace(0, N, (N/bnsz)+1))
        
        fs     = (h / (M * bnsz * dt))
        apsth = _N.repeat(fs, bnsz)    #    piecewise boxy approximate PSTH
        fsbnsz = _N.mean(fs) * _N.ones(N)

        aSi    = _N.linalg.solve(_N.dot(B, B.T), _N.dot(B, _N.log(apsth)))
    if phiS == None:
        l2    = 0.5+_N.random.randn(TM)*0.001
        phiSi = _N.linalg.solve(_N.dot(Gm[:, 0:TM], Gm[:, 0:TM].T), _N.dot(Gm[:, 0:TM], _N.log(l2)))

    x     = _N.array(aSi.tolist() + phiSi.tolist())

    #  If we estimate the the Jacobian, then even if h_dL 
    L0  = h_L_func(aSi, phiSi, M, B, Gm, sts, itvs, TM, dt, mL=mL)
    #sol = _sco.root(h_dL, x, jac=h_d2L, args=(nbs1, nbs2, M, B, Gm, sts, itvs, doAl, doPh, TM, dt))
    bds = mkBounds(x, nbs1, nbs2)
    #sol = _sco.minimize(h_L, x, jac=h_dL, hess=h_d2L, args=(nbs1, nbs2, M, B, Gm, sts, itvs, doAl, doPh, TM, dt, mL), bounds=bds, method="L-BFGS-B")
    sol = _sco.minimize(h_L, x, args=(nbs1, nbs2, M, B, Gm, sts, itvs, doAl, doPh, TM, dt, mL), bounds=bds, method="L-BFGS-B")
    #sol = _sco.minimize(h_L, x, jac=h_dL, hess=h_d2L, args=(nbs1, nbs2, M, B, Gm, sts, itvs, doAl, doPh, TM, dt, mL),)

    print sol.message
    L1  = h_L_func(sol.x[0:nbs1], sol.x[nbs1:nbs1+nbs2], M, B, Gm, sts, itvs, TM, dt, mL=mL)

    return sol, L0, L1

