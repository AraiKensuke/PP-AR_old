"""
timing
1  0.00005
2  0.00733
2a 0.08150
2b 0.03315
3  0.11470
FFBS  1.02443
5  0.03322
"""
import pickle
from kflib import createDataAR
import numpy as _N
import patsy
import re as _re
from filter import bpFilt, lpFilt, gauKer

import scipy.sparse.linalg as _ssl
import scipy.stats as _ss
from kassdirs import resFN, datFN

from   mcmcARpPlot import plotFigs, plotARcomps, plotQ2
from mcmcARpFuncs import loadL2, runNotes
import kfardat as _kfardat

import utilities as _U

import numpy.polynomial.polynomial as _Npp
import time as _tm
import ARlib as _arl
import kfARlibMPmv as _kfar
#import kfARlibMP as _kfar
import LogitWrapper as lw
#import pyPG as lw
from ARcfSmpl import ARcfSmpl, FilteredTimeseries

import logerfc as _lfc
import commdefs as _cd

from ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF
from multiprocessing import Pool

import os

#os.system("taskset -p 0xff %d" % os.getpid())

class mcmcARpBM:
    #  Simulation params
    processes     = 1
    setname       = None
    rs            = -1
    bFixF         = False
    burn          = None;    NMC           = None
    t0            = None;    t1            = None
    useTrials     = None;    restarts      = 0

    #  Description of model
    model         = None
    rn            = None    #  used for count data
    k             = None
    Cn            = None;    Cs            = None;    C             = None
    kntsPSTH      = None;    dfPSTH        = None
    ID_q2         = True
    use_prior     = _cd.__COMP_REF__
    AR2lims       = None
    F_alfa_rep    = None
    #######  TREND 

    #  Sampled 
    Bsmpx         = None
    smp_u         = None;    smp_aS        = None;
    smp_q2        = None
    smp_x00       = None
    smp_zs        = None;    smp_ms        = None;
    allalfas      = None
    uts           = None;    wts           = None
    ranks         = None
    pgs           = None
    fs            = None
    amps          = None
    dt            = None

    ###########  TEMP
    ss0           = None; ss1 = None;
    ARo01         = None

    #  LFC
    lfc           = None

    ####  TEMPORARY
    Bi            = None

    #  input data
    histFN        = None
    y             = None
    kp            = None
    l2            = None
    lrn           = None

    #  Gibbs
    ARord         = _cd.__NF__
    x             = None   #  true latent state
    fx            = None   #  filtered latent state
    px            = None   #  phase of latent state
    
    #  binomial states
    nStates       = 2
    s             = None   #  coupling M x 2
    z             = None   #  state index M x 2  [(1, 0), (0, 1), (0, 1), ...]
    m             = None   #  dim 2
    
    bpsth         = False
    q2            = None
    B             = None;    aS            = None; us             = None;    
    smpx          = None
    ws            = None
    x00           = None
    V00           = None

    #  
    _d            = None

    fSigMax       = 500.    #  fixed parameters
    freq_lims     = [[1 / .85, fSigMax]]

    #  u   --  Gaussian prior
    u_u          = 0;             s2_u         = 1.5
    #  q2  --  Inverse Gamma prior
    a_q2         = 1e-1;          B_q2         = 1e-6
    #  initial states
    u_x00        = None;          s2_x00       = None
    # psth spline coefficient priors
    u_a          = 0;             s2_a         = 0.5
    #  Dirichlet priors
    alp          = None

    def __init__(self):
        self.lfc         = _lfc.logerfc()

    def loadDat(self, trials): #################  loadDat
        oo = self
        bGetFP = False
        if oo.model== "bernoulli":
            x_st_cnts = _N.loadtxt(resFN("xprbsdN.dat", dir=oo.setname))
            y_ch      = 2   #  spike channel
            p = _re.compile("^\d{6}")   # starts like "exptDate-....."
            m = p.match(oo.setname)

            bRealDat = True
            dch = 4    #  # of data columns per trial

            if m == None:   #  not real data
                bRealDat = False
                dch = 3
            else:
                flt_ch      = 1    # Filtered LFP
                ph_ch       = 3    # Hilbert Trans
                bGetFP      = True
        else:
            x_st_cnts = _N.loadtxt(resFN("cnt_data.dat", dir=oo.setname))
            y_ch        = 1    # spks

            dch  = 2
        TR = x_st_cnts.shape[1] / dch    #  number of trials will get filtered

        #  If I only want to use a small portion of the data
        n0     = oo.t0
        oo.N   = x_st_cnts.shape[0] - 1
        if oo.t1 == None:
            oo.t1 = oo.N + 1
        #  meaning of N changes here
        N   = oo.t1 - 1 - oo.t0

        if TR == 1:
            x   = x_st_cnts[oo.t0:oo.t1, 0]
            y   = x_st_cnts[oo.t0:oo.t1, y_ch]
            fx  = x_st_cnts[oo.t0:oo.t1, 0]
            px  = x_st_cnts[oo.t0:oo.t1, y_ch]
            x   = x.reshape(1, oo.t1 - oo.t0)
            y   = y.reshape(1, oo.t1 - oo.t0)
            fx  = x.reshape(1, oo.t1 - oo.t0)
            px  = y.reshape(1, oo.t1 - oo.t0)
        else:
            x   = x_st_cnts[oo.t0:oo.t1, ::dch].T
            y   = x_st_cnts[oo.t0:oo.t1, y_ch::dch].T
            if bRealDat:
                fx  = x_st_cnts[oo.t0:oo.t1, flt_ch::dch].T
                px  = x_st_cnts[oo.t0:oo.t1, ph_ch::dch].T

        ####  Now keep only trials that have spikes
        kpTrl = range(TR)
        if trials is None:
            trials = range(oo.TR)
        oo.useTrials = []
        for utrl in trials:
            try:
                ki = kpTrl.index(utrl)
                if _N.sum(y[utrl, :]) > 0:
                    oo.useTrials.append(ki)
            except ValueError:
                print "a trial requested to use will be removed %d" % utrl
        ######  oo.y are for trials that have at least 1 spike
        oo.y     = _N.array(y[oo.useTrials])
        oo.x     = _N.array(x[oo.useTrials])
        if bRealDat:
            oo.fx    = _N.array(fx[oo.useTrials])
            oo.px    = _N.array(px[oo.useTrials])

        #  INITIAL samples
        if TR > 1:
            mnCt= _N.mean(oo.y, axis=1)
        else:
            mnCt= _N.array([_N.mean(oo.y)])

        #  remove trials where data has no information
        rmTrl = []

        if oo.model == "binomial":
            oo.kp  = oo.y - oo.rn*0.5
            p0   = mnCt / float(oo.rn)       #  matches 1 - p of genearted
            u  = _N.log(p0 / (1 - p0))    #  -1*u generated
        elif oo.model == "negative binomial":
            oo.kp  = (oo.y - oo.rn) *0.5
            p0   = mnCt / (mnCt + oo.rn)       #  matches 1 - p of genearted
            u  = _N.log(p0 / (1 - p0))    #  -1*u generated
        else:
            oo.kp  = oo.y - 0.5
            oo.rn  = 1
            #oo.dt  = 0.001
            logdt = _N.log(oo.dt)
            if TR > 1:
                ysm = _N.sum(oo.y, axis=1)
                u   = _N.log(ysm / ((N+1 - ysm)*oo.dt)) + logdt
            else:   #  u is a vector here
                u   = _N.array([_N.log(_N.sum(oo.y) / ((N+1 - _N.sum(oo.y))*oo.dt)) + logdt])

        oo.TR    = len(oo.useTrials)
        oo.TRm   = 0.5*(oo.TR - 1)
        oo.N     = N
        oo.sd   = _N.zeros((oo.TR, oo.TR))
        oo.us    = _N.array(u)

        ####  
        oo.l2 = loadL2(oo.setname, fn=oo.histFN)

        """
        if (l2 is not None) and (len(l2.shape) == 0):
            print "set up l2"
            oo.l2 = _N.array([l2])
        """

    def run(self, runDir=None, trials=None): ###########  RUN
        oo     = self    #  call self oo.  takes up less room on line
        oo.setname = os.getcwd().split("/")[-1]

        oo.Cs          = len(oo.freq_lims)
        oo.C           = oo.Cn + oo.Cs
        oo.R           = 1
        oo.k           = 2*oo.C + oo.R
        #  x0  --  Gaussian prior
        oo.u_x00        = _N.zeros(oo.k)
        oo.s2_x00       = _arl.dcyCovMat(oo.k, _N.ones(oo.k), 0.4)

        oo.rs=-1
        if runDir == None:
            runDir="%(sn)s/AR%(k)d_[%(t0)d-%(t1)d]" % \
                {"sn" : oo.setname, "ar" : oo.k, "t0" : oo.t0, "t1" : oo.t1}

        if oo.rs >= 0:
            unpickle(runDir, oo.rs)
        else:   #  First run
            oo.restarts = 0

        oo.loadDat(trials)
        oo.initGibbs()
        t1    = _tm.time()
        oo.gibbsSamp()
        t2    = _tm.time()
        print (t2-t1)

    def initGibbs(self):   ################################ INITGIBBS
        oo   = self

        if oo.bpsth:
            oo.B = patsy.bs(_N.linspace(0, (oo.t1 - oo.t0)*oo.dt, (oo.t1-oo.t0)), df=oo.dfPSTH, knots=oo.kntsPSTH, include_intercept=True)    #  spline basis

            if oo.dfPSTH is None:
                oo.dfPSTH = oo.B.shape[1] 
            oo.B = oo.B.T    #  My convention for beta
            oo.aS = _N.linalg.solve(_N.dot(oo.B, oo.B.T), _N.dot(oo.B, _N.ones(oo.t1 - oo.t0)*0.01))   #  small amplitude psth at first
        else:
            oo.B = patsy.bs(_N.linspace(0, (oo.t1 - oo.t0)*oo.dt, (oo.t1-oo.t0)), df=4, include_intercept=True)    #  spline basis

            oo.B = oo.B.T    #  My convention for beta
            oo.aS = _N.zeros(4)

        # #generate initial values of parameters
        oo._d = _kfardat.KFARGauObsDat(oo.TR, oo.N, oo.k)
        oo._d.copyData(oo.y)

        sPR="cmpref"
        if oo.use_prior==_cd.__FREQ_REF__:
            sPR="frqref"
        elif oo.use_prior==_cd.__ONOF_REF__:
            sPR="onfref"
        sAO= "sf" if (oo.ARord==_cd.__SF__) else "nf"

        ts        = "[%(1)d-%(2)d]" % {"1" : oo.t0, "2" : oo.t1}
        baseFN    = "rs=%(rs)d" % {"pr" : sPR, "rs" : oo.restarts}
        setdir="%(sd)s/AR%(k)d_%(ts)s_%(pr)s_%(ao)s" % {"sd" : oo.setname, "k" : oo.k, "ts" : ts, "pr" : sPR, "ao" : sAO}

        #  baseFN_inter   baseFN_comps   baseFN_comps

        ###############

        oo.Bsmpx        = _N.zeros((oo.TR, oo.NMC+oo.burn, (oo.N+1) + 2))
        oo.smp_u        = _N.zeros((oo.TR, oo.burn + oo.NMC))
        oo.smp_ss       = _N.zeros(oo.burn + oo.NMC)
        oo.smp_q2       = _N.zeros((oo.TR, oo.burn + oo.NMC))
        oo.smp_x00      = _N.empty((oo.TR, oo.burn + oo.NMC-1, oo.k))
        #  store samples of
        oo.allalfas     = _N.empty((oo.burn + oo.NMC, oo.k), dtype=_N.complex)
        oo.uts          = _N.empty((oo.TR, oo.burn + oo.NMC, oo.R, oo.N+2))
        oo.wts          = _N.empty((oo.TR, oo.burn + oo.NMC, oo.C, oo.N+3))
        oo.ranks        = _N.empty((oo.burn + oo.NMC, oo.C), dtype=_N.int)
        oo.pgs          = _N.empty((oo.TR, oo.burn + oo.NMC, oo.N+1))
        oo.fs           = _N.empty((oo.burn + oo.NMC, oo.C))
        oo.amps         = _N.empty((oo.burn + oo.NMC, oo.C))
        if oo.bpsth:
            oo.smp_aS        = _N.zeros((oo.burn + oo.NMC, oo.dfPSTH))

        radians      = buildLims(oo.Cn, oo.freq_lims, nzLimL=1.)
        oo.AR2lims      = 2*_N.cos(radians)

        if (oo.rs < 0):
            oo.smpx        = _N.zeros((oo.TR, (oo.N + 1) + 2, oo.k))   #  start at 0 + u
            oo.ws          = _N.empty((oo.TR, oo._d.N+1), dtype=_N.float)

            if oo.F_alfa_rep is None:
                oo.F_alfa_rep  = initF(oo.R, oo.Cs, oo.Cn, ifs=oo.ifs).tolist()   #  init F_alfa_rep

            print "begin---"
            print ampAngRep(oo.F_alfa_rep)
            print "begin^^^"
            q20         = 1e-3
            #oo.q2          = _N.ones(oo.TR)*q20
            oo.q2          = _N.ones(oo.TR)*0.00077

            oo.F0          = (-1*_Npp.polyfromroots(oo.F_alfa_rep)[::-1].real)[1:]
            ########  Limit the amplitude to something reasonable
            xE, nul = createDataAR(oo.N, oo.F0, q20, 0.1)
            mlt  = _N.std(xE) / 0.5    #  we want amplitude around 0.5
            oo.q2          /= mlt*mlt
            xE, nul = createDataAR(oo.N, oo.F0, oo.q2[0], 0.1)

            if oo.model == "Bernoulli":
                oo.initBernoulli()
            #smpx[0, 2:, 0] = x[0]    ##########  DEBUG

            ####  initialize ws if starting for first time
            if oo.TR == 1:
                oo.ws   = oo.ws.reshape(1, oo._d.N+1)
            for m in xrange(oo._d.TR):
                lw.rpg_devroye(oo.rn, oo.smpx[m, 2:, 0] + oo.us[m], num=(oo.N + 1), out=oo.ws[m, :])

        oo.smp_u[:, 0] = oo.us
        oo.smp_q2[:, 0]= oo.q2

        if oo.bpsth:
            oo.u_a            = _N.ones(oo.dfPSTH)*_N.mean(oo.us)

        oo.alp = _N.ones(oo.nStates)
        oo.z = _N.zeros((oo.TR, oo.nStates), dtype=_N.int)
        for tr in xrange(oo.TR):
            oo.z[tr, 0] = 0;                 oo.z[tr, 1] = 1
            if _N.random.rand() < 0.5:
                oo.z[tr, 0] = 1;                 oo.z[tr, 1] = 0

        oo.m = _N.ones(oo.nStates) / float(oo.nStates)
        oo.smp_ms = _N.empty((oo.burn + oo.NMC, oo.nStates))
        oo.smp_zs = _N.empty((oo.TR, oo.burn + oo.NMC, oo.nStates))
        oo.s = _N.zeros((oo.TR, oo.nStates))
        oo.s[:, 0] = 0.01
        oo.s[:, 1] = 1

        """
        _plt.ioff()
        for m in xrange(oo.TR):
            plotFigs(setdir, N, k, oo.burn, oo.NMC, x, y, Bsmpx, smp_u, smp_q2, t0, t1, Cs, Cn, C, baseFN, oo.TR, m, ID_q2)

        plotARcomps(setdir, N, k, oo.burn, oo.NMC, fs, amps, t0, t1, Cs, Cn, C, baseFN, oo.TR, m)


        """

    def initBernoulli(self):  ###########################  INITBERNOULLI
        oo = self


        if oo.model == "bernoulli":
            w  =  5
            wf =  gauKer(w)
            gk = _N.empty((oo.TR, oo.N+1))
            fgk= _N.empty((oo.TR, oo.N+1))
            for m in xrange(oo.TR):
                gk[m] =  _N.convolve(oo.y[m], wf, mode="same")
                gk[m] =  gk[m] - _N.mean(gk[m])
                gk[m] /= 5*_N.std(gk[m])
                fgk[m] = bpFilt(15, 100, 1, 135, 500, gk[m])   #  we want
                fgk[m, :] /= 2*_N.std(fgk[m, :])

                #oo.smpx[m, 2:, 0] = fgk[m, :]
                oo.smpx[m, 2:, 0] = 0
                for n in xrange(2+oo.k-1, oo.N+1+2):  # CREATE square smpx
                    oo.smpx[m, n, 1:] = oo.smpx[m, n-oo.k+1:n, 0][::-1]
                for n in xrange(2+oo.k-2, -1, -1):  # CREATE square smpx
                    oo.smpx[m, n, 0:oo.k-1] = oo.smpx[m, n+1, 1:oo.k]
                    oo.smpx[m, n, oo.k-1] = _N.dot(oo.F0, oo.smpx[m, n:n+oo.k, oo.k-2]) # no noise

                oo.Bsmpx[m, 0, :] = oo.smpx[m, :, 0]

    def gibbsSamp(self):  ###########################  GIBBSSAMP
        oo          = self
        ooTR        = oo.TR
        ook         = oo.k
        ooNMC       = oo.NMC
        ooN         = oo.N
        oo.x00         = _N.array(oo.smpx[:, 2])
        oo.V00         = _N.zeros((ooTR, ook, ook))

        ARo   = _N.empty((ooTR, oo._d.N+1))
        
        kpOws = _N.empty((ooTR, ooN+1))
        lv_f     = _N.zeros((ooN+1, ooN+1))
        lv_u     = _N.zeros((ooTR, ooTR))
        alpR   = oo.F_alfa_rep[0:oo.R]
        alpC   = oo.F_alfa_rep[oo.R:]
        Bii    = _N.zeros((ooN+1, ooN+1))
        
        #alpC.reverse()
        #  F_alfa_rep = alpR + alpC  already in right order, no?

        Wims         = _N.empty((ooTR, ooN+1, ooN+1))
        Oms          = _N.empty((ooTR, ooN+1))
        smWimOm      = _N.zeros(ooN + 1)
        smWinOn      = _N.zeros(ooTR)
        bConstPSTH = False
        D_f          = _N.diag(_N.ones(oo.B.shape[0])*oo.s2_a)   #  spline
        iD_f = _N.linalg.inv(D_f)
        D_u  = _N.diag(_N.ones(oo.TR)*oo.s2_u)   #  This should 
        iD_u = _N.linalg.inv(D_u)
        iD_u_u_u = _N.dot(iD_u, _N.ones(oo.TR)*oo.u_u)
        BDB  = _N.dot(oo.B.T, _N.dot(D_f, oo.B))
        DB   = _N.dot(D_f, oo.B)
        BTua = _N.dot(oo.B.T, oo.u_a)

        it    = 0

        oo.lrn   = _N.empty((ooTR, ooN+1))
        if oo.l2 is None:
            oo.lrn[:] = 1
        else:
            for tr in xrange(ooTR):
                oo.lrn[tr] = oo.build_lrnLambda2(tr)

        ###############################  MCMC LOOP  ########################
        ###  need pointer to oo.us, but reshaped for broadcasting to work
        ###############################  MCMC LOOP  ########################
        oous_rs = oo.us.reshape((ooTR, 1))   #  done for broadcasting rules
        lrnBadLoc = _N.empty(oo.N+1, dtype=_N.bool)

        lows = [3, ]
        for tr in xrange(oo.TR):
            oo.z[tr, 0] = 0;                oo.z[tr, 1] = 1
            try:
                lows.index(tr)
                oo.z[tr, 0] = 1;                oo.z[tr, 1] = 0
            except ValueError:
                pass

        for tr in xrange(oo.TR):
            oo.sd[tr, tr] = oo.s[0, 1]
            if oo.z[tr, 0] == 1:
                oo.sd[tr, tr] = oo.s[0, 0]

        sd01   = _N.zeros((oo.nStates, oo.TR, oo.TR))
        isd01  = _N.zeros((oo.nStates, oo.TR, oo.TR))

        smpx01 = _N.zeros((oo.nStates, oo.TR, oo.N+1))
        Esmpx01= _N.empty((oo.nStates, oo.TR, oo.N+3, oo.k))
        ws01   = _N.empty((oo.nStates, oo.TR, oo.N+1))
        kpOws01= _N.empty((oo.nStates, oo.TR, oo.N+1))
        oo.ARo01  = _N.empty((oo.nStates, oo.TR, oo.N+1))

        for tryZ in xrange(oo.nStates):
            _N.fill_diagonal(sd01[tryZ], oo.s[0, tryZ])
            isd01[tryZ] = _N.linalg.inv(sd01[tryZ])

        while (it < ooNMC + oo.burn - 1):
            t1 = _tm.time()
            it += 1

            print it
            if (it % 10) == 0:
                print it

            BaS = _N.dot(oo.B.T, oo.aS)
            oo._d.copyParams(oo.F0, oo.q2)

            for tryZ in xrange(oo.nStates):
                _N.dot(sd01[tryZ], oo.smpx[..., 2:, 0], out=smpx01[tryZ])

                #  generate latent AR state
                oo._d.f_x[:, 0, :, 0]     = oo.x00
                if it == 1:
                    for m in xrange(ooTR):
                        oo._d.f_V[m, 0]     = oo.s2_x00
                else:
                    oo._d.f_V[:, 0]     = oo._d.f_V[:, 1]

                ###  PG latent variable sample
                for m in xrange(ooTR):
                    _N.log(oo.lrn[m] / (1 + (1 - oo.lrn[m])*_N.exp(smpx01[tryZ, m] + BaS + oo.us[m])), out=oo.ARo01[tryZ, m])   #  history Offset   ####TRD change
                    nani = _N.isnan(oo.ARo01[tryZ, m], out=lrnBadLoc)
                    locs = _N.where(lrnBadLoc == True)
                    if locs[0].shape[0] > 0:
                        L = locs[0].shape[0]
                        print "ARo locations bad tr %(m)d  %(L) d" % {"m" : m, "L" : L}
                        for l in xrange(L):  #  fill with reasonable value
                            oo.ARo01[tryZ, m, locs[0][l]] = oo.ARo01[tryZ, m, locs[0][l] - 1]

                    lw.rpg_devroye(oo.rn, smpx01[tryZ, m] + oo.us[m] + BaS + oo.ARo01[tryZ, m], out=ws01[tryZ, m])  ######  devryoe  ####TRD change

                _N.divide(oo.kp, ws01[tryZ], out=kpOws01[tryZ])

                #  Now that we have PG variables, construct Gaussian timeseries
                #  ws(it+1)    using u(it), F0(it), smpx(it)

                oo._d.y = _N.dot(isd01[tryZ], kpOws01[tryZ] - BaS - oo.ARo01[tryZ] - oous_rs)

                #  (MxM)  (MxN) = (MxN)  (Rv is MxN)
                _N.dot(_N.dot(isd01[tryZ], isd01[tryZ]), 1 / ws01[tryZ], out=oo._d.Rv)

                #######  DATA AUGMENTATION.  If we update 's' before, we need to update _d.y right after, _d.y depends on 's'
                #  _d.F, _d.N, _d.ks, 
                tpl_args = zip(oo._d.y, oo._d.Rv, oo._d.Fs, oo.q2, oo._d.Ns, oo._d.ks, oo._d.f_x[:, 0], oo._d.f_V[:, 0])

                for m in xrange(ooTR):
                    Esmpx01[tryZ, m, 2:], oo._d.f_x[m], oo._d.f_V[m] = _kfar.armdl_FFBS_1itrMP(tpl_args[m])
                    Esmpx01[tryZ, m, 1, 0:ook-1]   = Esmpx01[tryZ, m, 2, 1:]
                    Esmpx01[tryZ, m, 0, 0:ook-2]   = Esmpx01[tryZ, m, 2, 2:]

                _N.dot(sd01[tryZ], Esmpx01[tryZ, :, 2:, 0], out=smpx01[tryZ])

            args0 = -0.5*_N.sum(ws01[0]*((oous_rs + BaS + smpx01[0, :] + oo.ARo01[0] - kpOws01[0])**2 - kpOws01[0]**2), axis=1)
            args1 = -0.5*_N.sum(ws01[1]*((oous_rs + BaS + smpx01[1, :] + oo.ARo01[1] - kpOws01[1])**2 - kpOws01[1]**2), axis=1)

            arg1m0 = args1 - args0
            thrSt0 = 1 / (1 + (oo.m[1]/oo.m[0])*_N.exp(arg1m0))
            print thrSt0

            btrSt = -1            

            for tr in xrange(oo.TR):
                btrSt = 0 if (_N.random.rand() < thrSt0[tr]) else 1

                oo.z[tr, btrSt] = 1;   oo.z[tr, 1-btrSt] = 0
                oo.smpx[tr] = Esmpx01[btrSt, tr]   #  oo.smpx
                ARo[tr]     = oo.ARo01[btrSt, tr]
                kpOws[tr]   = kpOws01[btrSt, tr]
                oo.ws[tr]      = ws01[btrSt, tr]

                oo.smp_zs[tr, it] = oo.z[tr]

                oo.Bsmpx[tr, it, 2:]    = Esmpx01[btrSt, tr, 2:, 0]

            dirArgs = _N.empty(oo.nStates)  #  dirichlet distribution args
            for i in xrange(oo.nStates):
                dirArgs[i] = oo.alp[i] + _N.sum(oo.z[:, i])

            oo.m[:] = _N.random.dirichlet(dirArgs)
            if (oo.m[0] < 0) or (oo.m[1] < 0):
                print "m is negative"
                print dirArgs
            print oo.m
            oo.smp_ms[it] = oo.m


            ########     per trial offset sample
            Ons  = kpOws - oo.smpx[..., 2:, 0] - ARo - BaS
            _N.einsum("mn,mn->m", oo.ws, Ons, out=smWinOn)  #  sum over trials
            ilv_u  = _N.diag(_N.sum(oo.ws, axis=1))  #  var  of LL
            #  diag(_N.linalg.inv(Bi)) == diag(1./Bi).  Bii = inv(Bi)
            _N.fill_diagonal(lv_u, 1./_N.diagonal(ilv_u))
            lm_u  = _N.dot(lv_u, smWinOn)  #  nondiag of 1./Bi are inf, mean LL
            #  now sample
            iVAR = ilv_u + iD_u
            VAR  = _N.linalg.inv(iVAR)  #
            Mn    = _N.dot(VAR, _N.dot(ilv_u, lm_u) + iD_u_u_u)
            oo.us[:]  = _N.random.multivariate_normal(Mn, VAR, size=1)[0, :]
            oo.smp_u[:, it] = oo.us

            ARcfSmpl(oo.lfc, ooN+1, ook, oo.AR2lims, oo.smpx[:, 1:, 0:ook], oo.smpx[:, :, 0:ook-1], oo.q2, oo.R, oo.Cs, oo.Cn, alpR, alpC, oo._d, prior=oo.use_prior, accepts=30, aro=oo.ARord)  
            oo.F_alfa_rep = alpR + alpC   #  new constructed
            prt, rank, f, amp = ampAngRep(oo.F_alfa_rep, f_order=True)
            print prt
            ut, wt = FilteredTimeseries(ooN+1, ook, oo.smpx[:, 1:, 0:ook], oo.smpx[:, :, 0:ook-1], oo.q2, oo.R, oo.Cs, oo.Cn, alpR, alpC, oo._d)
            #ranks[it]    = rank
            oo.allalfas[it] = oo.F_alfa_rep

            for m in xrange(ooTR):
                oo.wts[m, it, :, :]   = wt[m, :, :, 0]
                oo.uts[m, it, :, :]   = ut[m, :, :, 0]
                if not oo.bFixF:
                    oo.amps[it, :]  = amp
                    oo.fs[it, :]    = f

            oo.F0          = (-1*_Npp.polyfromroots(oo.F_alfa_rep)[::-1].real)[1:]

            #  sample u     WE USED TO Do this after smpx
            #  u(it+1)    using ws(it+1), F0(it), smpx(it+1), ws(it+1)

            oo.a2 = oo.a_q2 + 0.5*(ooTR*ooN + 2)  #  N + 1 - 1
            BB2 = oo.B_q2
            for m in xrange(ooTR):
                #   set x00 
                oo.x00[m]      = oo.smpx[m, 2]*0.1

                #####################    sample q2
                rsd_stp = oo.smpx[m, 3:,0] - _N.dot(oo.smpx[m, 2:-1], oo.F0).T
                BB2 += 0.5 * _N.dot(rsd_stp, rsd_stp.T)
            oo.q2[:] = _ss.invgamma.rvs(oo.a2, scale=BB2)

            oo.smp_q2[:, it]= oo.q2

            ########     PSTH sample  Do PSTH after we generate zs
            if oo.bpsth:
                Oms  = kpOws - oo.smpx[..., 2:, 0] - ARo - oous_rs
                _N.einsum("mn,mn->n", oo.ws, Oms, out=smWimOm)   #  sum over 
                ilv_f  = _N.diag(_N.sum(oo.ws, axis=0))
                _N.fill_diagonal(lv_f, 1./_N.diagonal(ilv_f))
                lm_f  = _N.dot(lv_f, smWimOm)  #  nondiag of 1./Bi are inf
                #  now sample
                iVAR = _N.dot(oo.B, _N.dot(ilv_f, oo.B.T)) + iD_f
                VAR  = _N.linalg.inv(iVAR)  #  knots x knots
                iBDBW = _N.linalg.inv(BDB + lv_f)   # BDB not diag
                Mn    = oo.u_a + _N.dot(DB, _N.dot(iBDBW, lm_f - BTua))
                oo.aS   = _N.random.multivariate_normal(Mn, VAR, size=1)[0, :]
                oo.smp_aS[it, :] = oo.aS
            else:
                oo.aS[:]   = 0

        t2 = _tm.time()
        print "gibbs iter %.3f" % (t2-t1)

    def build_lrnLambda2(self, tr):
        oo = self
        #  lmbda2 is short snippet of after-spike depression behavior
        lrn = _N.ones(oo.N + 1)
        lh    = len(oo.l2)

        hst  = []    #  spikes whose history is still felt

        for i in xrange(oo.N + 1):
            L  = len(hst)
            lmbd = 1

            for j in xrange(L - 1, -1, -1):
                th = hst[j]
                #  if i == 10, th == 9, lh == 1
                #  10 - 9 -1 == 0  < 1.   Still efective
                #  11 - 9 -1 == 1         No longer effective
                if i - th - 1 < lh:
                    lmbd *= oo.l2[i - th - 1]
                else:
                    hst.pop(j)

            if oo.y[tr, i] == 1:
                hst.append(i)

            lrn[i] *= lmbd
        return lrn


    def getComponents(self):
        oo    = self
        TR    = oo.TR
        NMC   = oo.NMC
        burn  = oo.burn
        R     = oo.R
        C     = oo.C
        ddN   = oo.N

        oo.rts = _N.empty((TR, burn+NMC, ddN+2, R))    #  real components   N = ddN
        oo.zts = _N.empty((TR, burn+NMC, ddN+2, C))    #  imag components 

        for tr in xrange(TR):
            for it in xrange(1, burn + NMC):
                b, c = dcmpcff(alfa=oo.allalfas[it])

                for r in xrange(R):
                    oo.rts[tr, it, :, r] = b[r] * oo.uts[tr, it, r, :]

                for z in xrange(C):
                    #print "z   %d" % z
                    cf1 = 2*c[2*z].real
                    gam = oo.allalfas[it, R+2*z]
                    cf2 = 2*(c[2*z].real*gam.real + c[2*z].imag*gam.imag)
                    oo.zts[tr, it, 0:ddN+2, z] = cf1*oo.wts[tr, it, z, 1:ddN+3] - cf2*oo.wts[tr, it, z, 0:ddN+2]

        oo.zts0 = _N.array(oo.zts[:, :, 1:, 0])

    def dump(self):
        oo    = self
        pcklme = [oo]
        oo.Bsmpx = None
        oo.smpx  = None
        oo.wts   = None
        oo.uts   = None
        oo._d    = None
        oo.lfc   = None
        oo.rts   = None
        oo.zts   = None

        with open("mARp.dump", "wb") as f:
            pickle.dump(pcklme, f)

            # import pickle
            # with open("mARp.dump", "rb") as f:
            #lm = pickle.load(f)


