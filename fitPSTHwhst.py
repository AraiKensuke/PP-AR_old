#  Fit PSTH with spike history
import patsy
import scipy.optimize as _sco
import os
import numpy as _N
import matplotlib.pyplot as _plt
from utildirs import setFN
from shutil import copyfile
from kstat import ranFromPercentile

from fitPSTHhistlib import h_L, h_L_func, mkBounds

nbs1  = None;  knts1 = None
nbs2  = None;  knts2 = None
nbs2v = None;  nbs2c = None
M0    = None;  M1    = None
Gm    = None;  B     = None    #  spline basis
doPh  = True;  doAl  = True
_dat  = None;  dat   = None
dt    = 0.001
dCols = 3
N     = None;  M     = None    #  number of data points per trial, # trials
aSi   = None;  phiSi = None
mL    = -1;
#pctl  = None;
sts   = None;  itvs  = None
gt01  = None;  allISIs= None
minmth= "L-BFGS-B"   # TNC, SLSQP

######  INITIALIZE
def init(ebf):
    copyfile("%s.py" % ebf, "%(s)s/%(s)s.py" % {"s" : ebf, "to" : setFN("%s.py" % ebf, dir=ebf, create=True)})
    global nbs1, knts1, nbs2, nbs2v, nbs2c, knts2, TM, Gm, B, N, M, _dat, dat, aSi, phiSi, M0, M1, pctl
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

    Gm   = patsy.bs(_N.linspace(0, dt*(N - 1), N), knots=knts2, include_intercept=True)
    nbs2 = Gm.shape[1]       #  in case used knts2
    if (nbs2c == None) and (nbs2v == None):
        nbs2v = 4
        nbs2c = nbs2 - nbs2v
    elif (nbs2c == None):
        nbs2c = nbs2 - nbs2v
    else:
        nbs2v = nbs2 - nbs2c

    Gm = Gm.T

######  fitPSTH routine    
def fitPSTH(aS=None, phiS=None, bnsz=50):   #  ebf  __exec_base_fn__
    global aSi, phiSi, M, sts, itvs, minmth, allISIs, gt01
    aSi   = aS
    phiSi = phiS
    sts   = []   #  will include one dummy spike
    itvs  = []
    allISIs = []
    rpsth = []
    gt01  = []

    for tr in xrange(M):
        itvs.append([])
        lst = _N.where(dat[:, dCols*tr + 2] == 1)[0].tolist()
        #lst.insert(0, int(-ranFromPercentile(pctl, lst[0])))    #  one dummy spike
        lst.insert(0, int(-30*_N.random.rand() - 1))    #  one dummy spike
        sts.append(_N.array(lst))
        allISIs.append(sts[tr][1:] - sts[tr][0:-1])  #  do this after random inserted
        rpsth.extend(lst)
        Lm  = len(lst) - 1    #  number of spikes this trial

        #  itvs is 1 bin shifted
        for i in xrange(Lm):
            itvs[tr].append([lst[i]+1, lst[i+1]+1]) 
        itvs[tr].append([lst[Lm]+1, N-1+1])

        gt01.append([])
        for it in xrange(len(itvs[tr])):    #  
            i0, i1 = itvs[tr][it]    # spktime + 1

            gt0= 0      #  if first is a real spike
            gt1= i1-i0
            if i0 < 1:  #  reference to fake spike
                gt0 = 0-i0  #  > 0
                i0 = 0  #  if there was a spike, i0==1
            gt01[tr].append([gt0, gt1])  # gt01[tr][0][0], gt01[tr][0][1]

    if aS == None:
        #####  Initialize
        h, bs = _N.histogram(rpsth, bins=_N.linspace(0, N, (N/bnsz)+1))
        
        fs     = (h / (M * bnsz * dt))
        apsth = _N.repeat(fs, bnsz)    #    piecewise boxy approximate PSTH

        aSi    = _N.linalg.solve(_N.dot(B, B.T), _N.dot(B, _N.log(apsth)))
    if phiS == None:
        l2    = 1. + _N.random.randn(N)*0.001
        phiSi = _N.linalg.solve(_N.dot(Gm, Gm.T), _N.dot(Gm, _N.log(l2)))

    x     = _N.array(aSi.tolist() + phiSi.tolist())

    #  If we estimate the the Jacobian, then even if h_dL 
    L0  = h_L_func(aSi, phiSi, M, B, Gm, sts, itvs, allISIs, gt01, dt, mL=mL)
    bds = mkBounds(x, nbs1, nbs2, nbs2v)

    sol = _sco.minimize(h_L, x, args=(nbs1, nbs2, M, B, Gm, sts, itvs, allISIs, gt01, doAl, doPh, dt, mL), bounds=bds, method=minmth)

    print sol.message
    L1  = h_L_func(sol.x[0:nbs1], sol.x[nbs1:nbs1+nbs2], M, B, Gm, sts, itvs, allISIs, gt01, dt, mL=mL)

    return sol, L0, L1

