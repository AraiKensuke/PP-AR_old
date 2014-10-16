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
from ARcfSmpl import ARcfSmpl, FilteredTimeseries

import logerfc as _lfc
import commdefs as _cd

from ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF
from multiprocessing import Pool

import os

#os.system("taskset -p 0xff %d" % os.getpid())

class mcmcARp:
    #  Simulation params
    processes     = 1
    setname       = None
    rs            = -1
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

    #  Sampled 
    Bsmpx         = None
    smp_u         = None;    smp_aS        = None
    smp_q2        = None
    smp_x00       = None
    allalfas      = None
    uts           = None;    wts           = None
    ranks         = None
    pgs           = None
    fs            = None
    amps          = None
    dt            = None

    #  LFC
    lfc           = None

    ####  TEMPORARY
    Bi            = None
    psthOffset    = None

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
    
    #  Current values of params and state
    bpsth         = False
    q2            = None
    B             = None;    aS            = None; u             = None;    
    smpx          = None
    ws            = None
    x00           = None
    V00           = None

    #  
    _d            = None

    fSigMax       = 500.    #  fixed parameters
    freq_lims     = [[1 / .85, fSigMax]]

    #  u   --  Gaussian prior
    u_u          = 0;             s2_u         = 5
    #  q2  --  Inverse Gamma prior
    a_q2         = 1e-1;          B_q2         = 1e-6
    #  initial states
    u_x00        = None;          s2_x00       = None
    #  initial states
    u_a          = 0;             s2_a         = None

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
            oo.dt  = 0.001
            logdt = _N.log(oo.dt)
            if TR > 1:
                ysm = _N.sum(oo.y, axis=1)
                u   = _N.log(ysm / ((N+1 - ysm)*oo.dt)) + logdt
            else:   #  u is a vector here
                u   = _N.array([_N.log(_N.sum(oo.y) / ((N+1 - _N.sum(oo.y))*oo.dt)) + logdt])
        oo.u     = _N.array(u[oo.useTrials])
        oo.TR    = len(oo.useTrials)

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
                oo.dfPSTH = B.shape[1] 
            oo.B = oo.B.T    #  My convention for beta
            oo.aS = _N.linalg.solve(_N.dot(oo.B, oo.B.T), _N.dot(oo.B, _N.ones(oo.t1 - oo.t0)*_N.mean(oo.u)))

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

            oo.F_alfa_rep  = initF(oo.R, oo.Cs, oo.Cn, ifs=oo.ifs).tolist()   #  init F_alfa_rep

            print "begin---"
            print ampAngRep(oo.F_alfa_rep)
            print "begin^^^"
            q20         = 1e-3
            oo.q2          = _N.ones(oo.TR)*q20

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
                lw.rpg_devroye(oo.rn, oo.smpx[m, 2:, 0] + oo.u[m], num=(oo.N + 1), out=oo.ws[m, :])

        oo.smp_u[:, 0] = oo.u
        oo.smp_q2[:, 0]= oo.q2

        if oo.bpsth:
            oo.u_a            = _N.ones(oo.dfPSTH)*_N.mean(oo.u)

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

                oo.smpx[m, 2:, 0] = fgk[m, :]
                for n in xrange(2+oo.k-1, oo.N+1+2):  # CREATE square smpx
                    oo.smpx[m, n, 1:] = oo.smpx[m, n-oo.k+1:n, 0][::-1]
                for n in xrange(2+oo.k-2, -1, -1):  # CREATE square smpx
                    oo.smpx[m, n, 0:oo.k-1] = oo.smpx[m, n+1, 1:oo.k]
                    oo.smpx[m, n, oo.k-1] = _N.dot(oo.F0, oo.smpx[m, n:n+oo.k, oo.k-2]) # no noise

                oo.Bsmpx[m, 0, :] = oo.smpx[m, :, 0]

    def gibbsSamp(self):  ###########################  GIBBSSAMPH
        oo          = self
        ooTR        = oo.TR
        ook         = oo.k
        ooNMC       = oo.NMC
        ooN         = oo.N
        oo.x00         = _N.array(oo.smpx[:, 2])
        oo.V00         = _N.zeros((ooTR, ook, ook))

        ARo   = _N.empty((ooTR, oo._d.N+1))
        
        kpOws = _N.empty((ooTR, ooN+1))
        alpR   = oo.F_alfa_rep[0:oo.R]
        alpC   = oo.F_alfa_rep[oo.R:]
        Bii    = _N.zeros((ooN+1, ooN+1))
        
        #alpC.reverse()
        #  F_alfa_rep = alpR + alpC  already in right order, no?

        if oo.B is not None:
            psthOffset = _N.empty((ooTR, ooN+1))
            Wims         = _N.empty((ooTR, ooN+1, ooN+1))
            Oms          = _N.empty((ooTR, ooN+1))
            smWimOm      = _N.zeros(ooN + 1)
            bConstPSTH = False
        else:
            bConstPSTH  = True
            psthOffset = _N.empty((ooTR, ooN+1))

        it    = 0

        oo.lrn   = _N.empty((ooTR, ooN+1))
        if oo.l2 is None:
            oo.lrn[:] = 1
        else:
            for tr in xrange(ooTR):
                oo.lrn[tr] = oo.build_lrnLambda2(tr)

        pool = Pool(processes=oo.processes)
        while (it < ooNMC + oo.burn - 1):
            t1 = _tm.time()
            it += 1
            print it
            if (it % 10) == 0:
                print it
            #  generate latent AR state
            oo._d.f_x[:, 0, :, 0]     = oo.x00
            if it == 1:
                for m in xrange(ooTR):
                    oo._d.f_V[m, 0]     = oo.s2_x00
            else:
                oo._d.f_V[:, 0]     = oo._d.f_V[:, 1]

            if bConstPSTH:
                for m in xrange(ooTR):
                    psthOffset[m, :] = oo.u[m]
            else:
                BaS = _N.dot(oo.B.T, oo.aS)
                for m in xrange(ooTR):
                    psthOffset[m] = BaS
            ###  PG latent variable sample

            #tPG1 = _tm.time()
            for m in xrange(ooTR):
                _N.log(oo.lrn[m] / (1 + (1 - oo.lrn[m])*_N.exp(oo.smpx[m, 2:, 0] + psthOffset[m])), out=ARo[m])   #  history Offset

                lw.rpg_devroye(oo.rn, oo.smpx[m, 2:, 0] + psthOffset[m] + ARo[m], out=oo.ws[m])  ######  devryoe
                """
                lst = _N.where(_N.isnan(oo.ws[m]) == True)
                if len(lst[0]) > 0:
                    oo.psthOffset = psthOffset
                    print "!!!!  rpg_devroye giving me a Nan"
                    print lst[0]
                    ind = lst[0]
                    print "lrn has nan?"
                    print oo.lrn[m, ind]
                    print "smpx has nan?"
                    print oo.smpx[m, (2+ind), 0]
                    print "ARo has nan?"
                    print ARo[m, ind]
                    print "psthOffset has nan?"
                    print psthOffset[m, ind]
                    print "oo.ws has nan?"
                    print oo.ws[m, ind]
                """
                    
            if ooTR == 1:
                oo.ws   = oo.ws.reshape(1, ooN+1)
            kpOws = oo.kp / oo.ws

            #  Now that we have PG variables, construct Gaussian timeseries
            #  ws(it+1)    using u(it), F0(it), smpx(it)
            #for m in xrange(ooTR):
                #oo._d.y[m]             = oo.kp[m]/oo.ws[m] - psthOffset[m] - ARo[m]
            #    oo._d.y[m]             = kpOws[m] - psthOffset[m] - ARo[m]

            oo._d.y = kpOws - psthOffset - ARo
            oo._d.copyParams(oo.F0, oo.q2)
            oo._d.Rv[:, :] =1 / oo.ws[:, :]   #  time dependent noise

            #tPG2 = _tm.time()
            if bConstPSTH:
                for m in xrange(ooTR):
                    A    = 0.5*(1./oo.s2_u + _N.sum(oo.ws[m]))
                    B    = oo.u_u/oo.s2_u + _N.sum(oo.kp[m] - oo.ws[m]*(oo.smpx[m, 2:, 0] + ARo[m]))
                    oo.u[m] = B/(2*A) + _N.sqrt(1/(2*A))*_N.random.randn()
                    oo.smp_u[m, it] = oo.u[m]
            else:
                #tPSTH1 = _tm.time()
                smWimOm[:] = 0
                #  cov matrix, prior of aS 
                iD = _N.diag(_N.ones(oo.B.shape[0])*5.)
                for m in xrange(ooTR):
                    Wims[m] = _N.diag(oo.ws[m])
                    Oms[m]  = kpOws[m] - oo.smpx[m, 2:, 0] - ARo[m]
                    smWimOm += _N.dot(Wims[m], Oms[m])
                #tPSTH2 = _tm.time()
                Bi = _N.sum(Wims, axis=0)   #  Bi is diagonal
                #  diag(_N.linalg.inv(Bi)) == diag(1./Bi).  Bii = inv(Bi)
                _N.fill_diagonal(Bii, 1./_N.diagonal(Bi))
                A  = _N.dot(Bii, smWimOm)  #  nondiag of 1./Bi are inf
                #A  = _N.dot(_N.linalg.inv(Bi), smWimOm)  #  nondiag of 1./Bi are inf

                #  now sample
                iVAR = _N.dot(oo.B, _N.dot(Bi, oo.B.T)) + iD
                VAR  = _N.linalg.inv(iVAR)  #  Diagnoal, #knots x #knots
                Mn   = _N.dot(VAR, _N.dot(oo.B, _N.dot(Bi, A.T)) + _N.dot(iD, oo.u_a))# + 0)
                #Mn   = _N.dot(VAR, _N.dot(oo.B, _N.dot(Bi, A)))# + 0)
                #  multivar_normal returns a row vector
                aS   = _N.random.multivariate_normal(Mn, VAR, size=1)[0, :]
                """
                try:
                except ValueError:
                    print "ValueError"
                    print iVAR
                    print "------------"
                    oo.Bi = Bi
                    print _N.dot(Bi, oo.B.T)   #  I found a nan is _N.dot(Bi, oo.B.T)
                    print "------------"
                    print _N.dot(oo.B, _N.dot(Bi, oo.B.T))
                    print "------------"
                    print Wims
                    print "------------"
                    print Mn
                    print VAR
                    raise
                """

                oo.smp_aS[it, :] = aS
                #tPSTH3 = _tm.time()

            #  _d.F, _d.N, _d.ks, 
            tpl_args = zip(oo._d.y, oo._d.Rv, oo._d.Fs, oo.q2, oo._d.Ns, oo._d.ks, oo._d.f_x[:, 0], oo._d.f_V[:, 0])

            #tkf1  = _tm.time()
            sxv = pool.map(_kfar.armdl_FFBS_1itrMP, tpl_args)
            #tkf2  = _tm.time()

            for m in xrange(ooTR):
                oo.smpx[m, 2:] = sxv[m][0]
                oo._d.f_x[m] = sxv[m][1]
                oo._d.f_V[m] = sxv[m][2]
                oo.smpx[m, 1, 0:ook-1]   = oo.smpx[m, 2, 1:]
                oo.smpx[m, 0, 0:ook-2]   = oo.smpx[m, 2, 2:]
                oo.Bsmpx[m, it, 2:]    = oo.smpx[m, 2:, 0]

            # sample F0
            # for mh in xrange(50):
            #  F0(it+1)    using ws(it+1), u(it+1), smpx(it+1), ws(it+1)

            #  wt.shape = (TR, C, _d.N+1+2, 1)
            #  wts.shape = (TR, burn+NMC, C, _d.N+1+2, 1)
            #  ut.shape = (TR, C, _d.N+1+1, 1)

            ARcfSmpl(oo.lfc, ooN+1, ook, oo.AR2lims, oo.smpx[:, 1:, 0:ook], oo.smpx[:, :, 0:ook-1], oo.q2, oo.R, oo.Cs, oo.Cn, alpR, alpC, oo._d, prior=oo.use_prior, accepts=30, aro=oo.ARord)  
            oo.F_alfa_rep = alpR + alpC   #  new constructed
            prt, rank, f, amp = ampAngRep(oo.F_alfa_rep, f_order=True)
            ut, wt = FilteredTimeseries(ooN+1, ook, oo.smpx[:, 1:, 0:ook], oo.smpx[:, :, 0:ook-1], oo.q2, oo.R, oo.Cs, oo.Cn, alpR, alpC, oo._d)
            #ranks[it]    = rank
            oo.allalfas[it] = oo.F_alfa_rep

            for m in xrange(ooTR):
                oo.wts[m, it, :, :]   = wt[m, :, :, 0]
                oo.uts[m, it, :, :]   = ut[m, :, :, 0]
                oo.amps[it, :]  = amp
                oo.fs[it, :]    = f

            oo.F0          = (-1*_Npp.polyfromroots(oo.F_alfa_rep)[::-1].real)[1:]

            print prt
            #  sample u     WE USED TO Do this after smpx
            #  u(it+1)    using ws(it+1), F0(it), smpx(it+1), ws(it+1)

            if oo.ID_q2:
                for m in xrange(ooTR):
                    #####################    sample q2
                    a = oo.a_q2 + 0.5*(ooN+1)  #  N + 1 - 1
                    rsd_stp = oo.smpx[m, 3:,0] - _N.dot(oo.smpx[m, 2:-1], oo.F0).T
                    BB = oo.B_q2 + 0.5 * _N.dot(rsd_stp, rsd_stp.T)
                    oo.q2[m] = _ss.invgamma.rvs(a, scale=BB)
                    oo.x00[m]      = oo.smpx[m, 2]*0.1
                    oo.smp_q2[m, it]= oo.q2[m]
            else:
                oo.a2 = oo.a_q2 + 0.5*(ooTR*ooN + 2)  #  N + 1 - 1
                BB2 = oo.B_q2
                for m in xrange(ooTR):
                    #   set x00 
                    oo.x00[m]      = oo.smpx[m, 2]*0.1

                    #####################    sample q2
                    rsd_stp = oo.smpx[m, 3:,0] - _N.dot(oo.smpx[m, 2:-1], oo.F0).T
                    BB2 += 0.5 * _N.dot(rsd_stp, rsd_stp.T)
                oo.q2[:] = _ss.invgamma.rvs(oo.a2, scale=BB2)

            oo.smp_u[:, it] = oo.u
            oo.smp_q2[:, it]= oo.q2
            t2 = _tm.time()
            # print "1  %.5f" % (tPG1 - t1)
            # print "2  %.5f" % (tPG2 - tPG1)
            # print "2a %.5f" % (tPSTH2 - tPSTH1)
            # print "2b %.5f" % (tPSTH3 - tPSTH2)
            # print "3  %.5f" % (tkf1 - tPG2)
            # print "4  %.5f" % (tkf2 - tkf1)
            # print "5  %.5f" % (t2 - tkf2)
            print "----------------    %.5f" % (t2-t1)

        pool.close()

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


