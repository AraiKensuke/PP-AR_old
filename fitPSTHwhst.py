#  Fit PSTH with spike history
import patsy
import scipy.optimize as _sco
import os
import numpy as _N
import matplotlib.pyplot as _plt
from utildirs import setFN
from shutil import copyfile

from fitPSTHhistlib import h_dL, h_d2L, h_L

nbs1  = None   #  one or the other
knts1 = None
nbs2  = None   #  one or the other
knts2 = None
TM    = None
Gm    = None
B     = None
setname=None
doPh  = True
doAl  = True
frstSpk=0

def fitPSTH(ebf):   #  ebf  __exec_base_fn__
    copyfile("%s.py" % ebf, "%(s)s/%(s)s.py" % {"s" : ebf, "to" : setFN("%s.py" % ebf, dir=ebf, create=True)})
    global setname, nbs1, knts1, nbs2, knts2, TM, Gm, B
    setname = os.getcwd().split("/")[-1]

    dCols = 3
    dat   = _N.loadtxt("xprbsdN.dat")
    l2gen   = _N.loadtxt("generate-l2.dat")

    N     = dat.shape[0]            #  how many bins per trial
    M     = dat.shape[1] / dCols    #  TRIALS

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

        for i in xrange(Lm):
            itvs[tr].append([lst[i]+1, lst[i+1]+1]) 
        itvs[tr].append([lst[Lm]+1, N-1+1])

        #  [-20, 16, 66]       spks
        #  [0, 17), [17, 67)   intvs.  first 
        #  [-20, 999]       spks
        #  [0, 1000), [1000, 1000)   intvs.  first    a[1000:1000] empty


    dt= 0.001
    x = _N.linspace(0., dt*(N-1), N)

    B = patsy.bs(x, df=nbs1, knots=knts1, include_intercept=True)
    nbs1 = B.shape[1]
    B  = B.T

    _Gm = patsy.bs(_N.linspace(0, dt*(TM-1), TM), knots=knts2, df=nbs2, include_intercept=True)
    nbs2 = _Gm.shape[1]       #  in case used knts2

    Gm = _N.zeros((N, nbs2))
    Gm[0:TM] = _Gm

    Gm = Gm.T

    aST   = _N.linalg.solve(_N.dot(B, B.T), _N.dot(B, _N.log(dat[:, 1]/dt)))
    phiST = _N.linalg.solve(_N.dot(Gm[:, 0:TM], Gm[:, 0:TM].T), _N.dot(Gm[:, 0:TM], _N.log(l2gen[0:TM])))

    #####  Initialize
    bnsz   = 50
    h, bs = _N.histogram(rpsth, bins=_N.linspace(0, N, (N/bnsz)+1))

    fs     = (h / (M * bnsz * dt))
    apsth = _N.repeat(fs, bnsz)    #    piecewise boxy approximate PSTH
    fsbnsz = _N.mean(fs) * _N.ones(N)

    #aS     = _N.linalg.solve(_N.dot(B, B.T), _N.dot(B, _N.log(apsth)))
    aS     = _N.linalg.solve(_N.dot(B, B.T), _N.dot(B, _N.log(fsbnsz)))

    l2    = 0.5+_N.random.randn(TM)*0.001

    phiS     = _N.linalg.solve(_N.dot(Gm[:, 0:TM], Gm[:, 0:TM].T), _N.dot(Gm[:, 0:TM], _N.log(l2)))
    phiSi = None
    aSi   = None
    if doPh and (not doAl):
        aSi   = aST
        phiSi = phiS
    elif (not doPh) and doAl:
        aSi   = aS
        phiSi = phiST
    else:
        aSi   = aS
        phiSi = phiS
    x     = _N.array(aSi.tolist() + phiSi.tolist())

    #  If we estimate the the Jacobian, then even if h_dL 
    sol = _sco.root(h_dL, x, jac=h_d2L, args=(nbs1, nbs2, M, B, Gm, sts, itvs, doAl, doPh, TM, dt, frstSpk))

    # LTR= h_L(aST, phiST, M, B, Gm, sts, itvs, TM, dt, frstSpk)   #  TRUE
    # L0 = h_L(aS, phiS, M, B, Gm, sts, itvs, TM, dt, frstSpk)
    # L1 = h_L(sol.x[0:nbs1], sol.x[nbs1:nbs1+nbs2], M, B, Gm, sts, itvs, TM, dt, frstSpk)
    # print "%(TR).1f   %(0).1f   %(1).1f" % {"TR" : LTR, "0" : L0, "1" : L1}

    fig = _plt.figure(figsize=(5, 2*4))
    #####
    fig.add_subplot(2, 1, 1)
    #  PSTH used to generate data
    _plt.plot(_N.exp(_N.dot(B.T, aST)), lw=2, color="red")
    #  Calculated fit
    _plt.plot(_N.exp(_N.dot(B.T, sol.x[0:nbs1])), lw=2, color="black")
    #  naive PSTH
    _plt.plot(apsth, lw=2, ls="--", color="grey")   #  rought blocked psth
    #  fit to initial naive PSTH guess
    _plt.plot(_N.exp(_N.dot(B.T, aSi)), lw=2, color="blue")
    _plt.grid()
    _plt.ylim(0, 1.1*max(_N.exp(_N.dot(B.T, sol.x[0:nbs1]))))
    #_plt.ylim(0, 1.1*max(apsth))
    #####

    fig.add_subplot(2, 1, 2)

    # l2 used to generate data    
    _plt.plot(range(1, TM+1), l2gen[0:TM], lw="2", color="red")
    # calculated fit
    _plt.plot(range(1, TM+1), _N.exp(_N.dot(Gm.T[0:TM], sol.x[nbs1:])), color="black", lw=2)
    # initial guess
    _plt.plot(range(1, TM+1), _N.exp(_N.dot(Gm.T[0:TM], phiSi)), lw="2", color="blue")

    _plt.grid()
    _plt.xlim(0, TM)
    _plt.ylim(0, 1.2)
#    fig.suptitle("init L %(i).1f   final L %(f).1f" % {"i" : L0, "f" : L1})
    _plt.savefig(setFN("FIT.png", dir=ebf, create=True), background="transparent")
    _plt.close()

    return sol
    #  let's pickle the results
    
    #  create fitted firing rate
    #  now create
