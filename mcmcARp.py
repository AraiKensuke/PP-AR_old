from kflib import createDataAR
import numpy as _N
from mcmcARpFuncs import loadDat, initBernoulli
import patsy

import scipy.stats as _ss
from kassdirs import resFN, datFN

from   mcmcARpPlot import plotFigs, plotARcomps, plotQ2
from mcmcARpFuncs import loadL2, runNotes
import kfardat as _kfardat
import time as _tm

import utilities as _U

import numpy.polynomial.polynomial as _Npp
import time as _tm
import ARlib as _arl
import LogitWrapper as lw
from   gibbsMP import gibbsSampH, build_lrnLambda2

import logerfc as _lfc
import commdefs as _cd

from ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF

import os

#os.system("taskset -p 0xff %d" % os.getpid())

#  Sampled 
Bsmpx         = None
smp_u         = None
smp_aS        = None
smp_q2        = None
smp_x00       = None
#  store samples of
kntsPSTH      = None
dfPSTH        = None
allalfas      = None
uts           = None
wts           = None
ranks         = None
pgs           = None
fs            = None
amps          = None
dt            = None
#  
histFN        = None
setname       = None
model         = None
ARord         = _cd.__NF__
rs            = -1
burn          = None
NMC           = None
model         = None
_t0           = None
_t1           = None
ID_q2         = True
bpsth         = False
use_prior     = _cd.__COMP_REF__

k             = None
Cn            = 6
Cs            = None
C             = None
x             = None   #  True

_d            = None

fSigMax       = 500.
freq_lims     = [[1 / .85, fSigMax]]
ifs           = [(20. / fSigMax) * _N.pi]    #  initial values

#  u   --  Gaussian prior
u_u          = 0;             s2_u         = 5
#  q2  --  Inverse Gamma prior
a_q2         = 1e-1;          B_q2         = 1e-6
#  initial states
u_x00        = None;          s2_x00       = None

def run(runDir=None, useTrials=None):
    #_lfc.init()
    global setname, _t0, _t1, _d, Bsmpx, uts, wts  #  only these that we are setting inside
    global allalfas, smp_aS, smp_q2, smp_u, B, aS, dfPSTH
    global x
    setname = os.getcwd().split("/")[-1]

    Cs          = len(freq_lims)
    C           = Cn + Cs
    R           = 1
    k           = 2*C + R
    #  x0  --  Gaussian prior
    u_x00        = _N.zeros(k)
    s2_x00       = _arl.dcyCovMat(k, _N.ones(k), 0.4)

    rs=-1
    if runDir == None:
        runDir="%(sn)s/AR%(k)d_[%(t0)d-%(t1)d]" % {"sn" : setname, "ar" : k, "t0" : _t0, "t1" : _t1}

    if rs >= 0:
        unpickle(runDir, rs)
    else:   #  First run
        restarts = 0

    bGetFP = False
    TR, rn, _x, _y, N, kp, _u, rmTrl, kpTrl = loadDat(setname, model, t0=_t0, t1=_t1, filtered=bGetFP, phase=bGetFP)  # u is set initialized

    print setname
    l2 = loadL2(setname, fn=histFN)
    if (l2 is not None) and (len(l2.shape) == 0):
        l2 = _N.array([l2])

    #  if a trial requested in useTrials is not in kpTrl, warn user
    if useTrials is None:
        useTrials = range(TR)
    useTrialsFltrd = []
    for utrl in useTrials:
        try:
            ki = kpTrl.index(utrl)
            useTrialsFltrd.append(ki)
        except ValueError:
            print "a trial requested to use will be removed %d" % utrl
    y     = _N.array(_y[useTrialsFltrd])
    x     = _N.array(_x[useTrialsFltrd])
    if bGetFP:
        fx    = _N.array(_fx[useTrialsFltrd])
        px    = _N.array(_px[useTrialsFltrd])
    u     = _N.array(_u[useTrialsFltrd])
    TR    = len(useTrialsFltrd)

    B    = None
    aS   = None
    if bpsth:
        B = patsy.bs(_N.linspace(0, (_t1 - _t0)*dt, (_t1-_t0)), df=dfPSTH, knots=kntsPSTH, include_intercept=True)    #  spline basis
        if dfPSTH is None:
            dfPSTH = B.shape[1] 
        B = B.T    #  My convention for beta
        aS = _N.linalg.solve(_N.dot(B, B.T), _N.dot(B, _N.ones(_t1 - _t0)*_N.mean(u)))

    ###########  PRIORs
    priors = {"u_u" : u_u, "s2_u" : s2_u, "a_q2" : a_q2, "B_q2" : B_q2,
              "u_x00" : u_x00, "s2_x00" : s2_x00}

    # #generate initial values of parameters
    _d = _kfardat.KFARGauObsDat(TR, N, k)
    _d.copyData(y)

    sPR="cmpref"
    if use_prior==_cd.__FREQ_REF__:
        sPR="frqref"
    elif use_prior==_cd.__ONOF_REF__:
        sPR="onfref"
    sAO="sf"
    if ARord==_cd.__SF__:
        sAO="sf"
    elif ARord==_cd.__NF__:
        sAO="nf"

    ts        = "[%(1)d-%(2)d]" % {"1" : _t0, "2" : _t1}
    baseFN    = "rs=%(rs)d" % {"pr" : sPR, "rs" : restarts}
    setdir="%(sd)s/AR%(k)d_%(ts)s_%(pr)s_%(ao)s" % {"sd" : setname, "k" : k, "ts" : ts, "pr" : sPR, "ao" : sAO}

    #  baseFN_inter   baseFN_comps   baseFN_comps

    ###############

    Bsmpx        = _N.zeros((TR, NMC+burn, (N+1) + 2))
    smp_u        = _N.zeros((TR, burn + NMC))
    smp_q2       = _N.zeros((TR, burn + NMC))
    smp_x00      = _N.empty((TR, burn + NMC-1, k))
    #  store samples of
    allalfas     = _N.empty((burn + NMC, k), dtype=_N.complex)
    uts          = _N.empty((TR, burn + NMC, R, N+2))
    wts          = _N.empty((TR, burn + NMC, C, N+3))
    ranks        = _N.empty((burn + NMC, C), dtype=_N.int)
    pgs          = _N.empty((TR, burn + NMC, N+1))
    fs           = _N.empty((burn + NMC, C))
    amps         = _N.empty((burn + NMC, C))
    if bpsth:
        smp_aS        = _N.zeros((burn + NMC, dfPSTH))

    radians      = buildLims(Cn, freq_lims, nzLimL=1.)
    AR2lims      = 2*_N.cos(radians)

    if (rs < 0):
        smpx        = _N.zeros((TR, (_d.N + 1) + 2, k))   #  start at 0 + u
        ws          = _N.empty((_d.TR, _d.N+1), dtype=_N.float)

        F_alfa_rep  = initF(R, Cs, Cn, ifs=ifs)   #  init F_alfa_rep

        print "begin---"
        print ampAngRep(F_alfa_rep)
        print "begin^^^"
        q20         = 1e-3
        q2          = _N.ones(TR)*q20

        F0          = (-1*_Npp.polyfromroots(F_alfa_rep)[::-1].real)[1:]
        ########  Limit the amplitude to something reasonable
        xE, nul = createDataAR(N, F0, q20, 0.1)
        mlt  = _N.std(xE) / 0.5    #  we want amplitude around 0.5
        q2          /= mlt*mlt
        xE, nul = createDataAR(N, F0, q2[0], 0.1)

        initBernoulli(model, k, F0, TR, _d.N, y, fSigMax, smpx, Bsmpx)
        #smpx[0, 2:, 0] = x[0]    ##########  DEBUG

        ####  initialize ws if starting for first time
        if TR == 1:
            ws   = ws.reshape(1, _d.N+1)
        for m in xrange(_d.TR):
            lw.rpg_devroye(rn, smpx[m, 2:, 0] + u[m], num=(N + 1), out=ws[m, :])

    ARo   = _N.empty((TR, _d.N+1))
    smp_u[:, 0] = u
    smp_q2[:, 0]= q2

    t1    = _tm.time()

    # if model == "bernoulli":
    F_alfa_rep = gibbsSampH(burn, NMC, AR2lims, F_alfa_rep, R, Cs, Cn, TR, rn, _d, u, B, aS, q2, uts, wts, kp, ws, smpx, Bsmpx, smp_u, smp_q2, smp_aS, allalfas, fs, amps, ranks, priors, ARo, l2, prior=use_prior, aro=ARord)

    t2    = _tm.time()
    print (t2-t1)
    
    """
    _plt.ioff()
    for m in xrange(TR):
        plotFigs(setdir, N, k, burn, NMC, x, y, Bsmpx, smp_u, smp_q2, _t0, _t1, Cs, Cn, C, baseFN, TR, m, ID_q2)

    plotARcomps(setdir, N, k, burn, NMC, fs, amps, _t0, _t1, Cs, Cn, C, baseFN, TR, m)


    """
