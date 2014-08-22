#  Fit PSTH with spike history
import patsy
import scipy.optimize as _sco
import os
import numpy as _N
import matplotlib.pyplot as _plt
from utildirs import setFN
from shutil import copyfile

from fitPSTHhistlib import h_dL, h_d2L, h_L

nbs1  = None;  knts1 = None
nbs2  = None;  knts2 = None
TM    = None
Gm    = None;  B     = None    #  spline basis
doPh  = True;  doAl  = True
dat   = None
dt    = 0.001
dCols = 3
N     = None;  M     = None    #  number of data points per trial, # trials
aSi   = None;  phiSi = None

######  INITIALIZE
def init(ebf):
    copyfile("%s.py" % ebf, "%(s)s/%(s)s.py" % {"s" : ebf, "to" : setFN("%s.py" % ebf, dir=ebf, create=True)})
    global nbs1, knts1, nbs2, knts2, TM, Gm, B, N, M, dat, aSi, phiSi
    dat   = _N.loadtxt("xprbsdN.dat")

    N     = dat.shape[0]            #  how many bins per trial
    M     = dat.shape[1] / dCols    #  TRIALS

    B     = patsy.bs(_N.linspace(0., dt*(N-1), N), df=nbs1, knots=knts1, include_intercept=True)
    nbs1  = B.shape[1]
    B     = B.T

    _Gm   = patsy.bs(_N.linspace(0, dt*(TM-1), TM), knots=knts2, df=nbs2, include_intercept=True)
    nbs2 = _Gm.shape[1]       #  in case used knts2

    Gm = _N.zeros((N, nbs2))
    Gm[0:TM] = _Gm
    Gm = Gm.T

######  fitPSTH routine    
def fitPSTH(aS=None, phiS=None):   #  ebf  __exec_base_fn__
    global aSi, phiSi
    aSi   = aS
    phiSi = phiS
    sts   = []   #  will include one dummy spike
    itvs  = []
    rpsth = []

    for tr in xrange(M):
        itvs.append([])
        lst = _N.where(dat[:, dCols*tr + 2] == 1)[0].tolist()
        if lst[0] != 0:   #  if not spike at time 0, add a dummy spike
            lst.insert(0, int(-1 - 30*_N.random.rand()))    #  one dummy spike
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
    print "^^^^^^^^"
    print x

    print nbs1
    print nbs2
    print M
    print B
    print Gm
    print sts[0]
    print itvs[0]
    print doAl
    print doPh
    print TM
    print dt

    #  If we estimate the the Jacobian, then even if h_dL 
    sol = _sco.root(h_dL, x, jac=h_d2L, args=(nbs1, nbs2, M, B, Gm, sts, itvs, doAl, doPh, TM, dt))

    return sol
