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
import matplotlib.pyplot as _plt

import scipy.stats as _ss
from kassdirs import resFN, datFN

import utilities as _U

import numpy.polynomial.polynomial as _Npp
import time as _tm
import ARlib as _arl
import kfARlibMPmv as _kfar
import pyPG as lw
from ARcfSmpl import ARcfSmpl, FilteredTimeseries

import commdefs as _cd

from ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF
import os

import mcmcARspk as mcmcARspk

from multiprocessing import Pool

class mcmcARp(mcmcARspk.mcmcARspk):
    #  Description of model
    rn            = None    #  used for count data
    k             = None
    Cn            = None;    Cs            = None;    C             = None
    kntsPSTH      = None;    dfPSTH        = None
    ID_q2         = True
    use_prior     = _cd.__COMP_REF__
    AR2lims       = None
    F_alfa_rep    = None

    #  Sampled 
    ranks         = None
    pgs           = None
    fs            = None
    amps          = None
    dt            = None
    mcmcRunDir    = None

    ####  TEMPORARY
    Bi            = None
    rsds          = None

    #  Gibbs
    ARord         = _cd.__NF__
    x             = None   #  true latent state
    fx            = None   #  filtered latent state
    px            = None   #  phase of latent state
    
    #  Current values of params and state
    B             = None;    aS            = None; us             = None;    

    #  coefficient sampling
    fSigMax       = 500.    #  fixed parameters
    freq_lims     = [[1 / .85, fSigMax]]
    sig_ph0L      = -1
    sig_ph0H      = 0

    #  1 offset for all trials
    bIndOffset    = True
    peek          = 400

    loghist       = None

    VIS           = None
    def gibbsSamp(self, burns=None):  ###########################  GIBBSSAMPH
        oo          = self
        ooTR        = oo.TR
        ook         = oo.k
        ooNMC       = oo.NMC
        ooN         = oo.N
        oo.x00         = _N.array(oo.smpx[:, 2])
        oo.V00         = _N.zeros((ooTR, ook, ook))
        oo.loghist = _N.zeros(oo.N+1)

        print "oo.mcmcRunDir    %s" % oo.mcmcRunDir
        if oo.mcmcRunDir is None:
            print "here!!!!!!!!!!!!!!"
            oo.mcmcRunDir = ""
        elif (len(oo.mcmcRunDir) > 0) and (oo.mcmcRunDir[-1] != "/"):
            oo.mcmcRunDir += "/"

        ARo   = _N.zeros((ooTR, oo._d.N+1))
        
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
        bConstPSTH   = False
        D_f          = _N.diag(_N.ones(oo.B.shape[0])*oo.s2_a)   #  spline
        iD_f = _N.linalg.inv(D_f)
        D_u  = _N.diag(_N.ones(oo.TR)*oo.s2_u)   #  This should 
        iD_u = _N.linalg.inv(D_u)
        iD_u_u_u = _N.dot(iD_u, _N.ones(oo.TR)*oo.u_u)
        BDB  = _N.dot(oo.B.T, _N.dot(D_f, oo.B))
        DB   = _N.dot(D_f, oo.B)
        BTua = _N.dot(oo.B.T, oo.u_a)

        it    = -1

        oous_rs = oo.us.reshape((ooTR, 1))
        lrnBadLoc = _N.empty((oo.TR, oo.N+1), dtype=_N.bool)
        runTO = ooNMC + oo.burn - 1 if (burns is None) else (burns - 1)
        oo.allocateSmp(runTO+1)

        BaS = _N.zeros(oo.N+1)#_N.empty(oo.N+1)

        #  H shape    100 x 9
        Hbf = patsy.bs(_N.linspace(0, 1.2, 1200), knots=_N.array([0, 0.003, 0.01, 0.03, 0.04, 1.2]), include_intercept=True)    #  spline basis
        #Hbf = patsy.bs(_N.linspace(0, 1.2, 1200), knots=_N.array([0, 0.1, 0.2, 0.3, 1.2]), include_intercept=True)    #  spline basis

        cInds = _N.array([0, 1, 6, 7, 8, 9])
        #cInds = _N.array([0, 1, 6, 7, 8, 9])
        vInds = _N.array([2, 3, 4, 5])

        if oo.processes > 1:
            pool = Pool(processes=oo.processes)

        Msts = []
        for m in xrange(ooTR):
            Msts.append(_N.where(oo.y[m] == 1)[0])
        HcM  = _N.empty((4, 4))
        RHS = _N.empty((10, 1))
        RHS[cInds, 0] = 0
        RHS[0, 0] = -5
        RHS[1, 0] = -5

        while (it < runTO):
            t1 = _tm.time()
            it += 1
            print it
            if (it % 10) == 0:
                print it
            t2 = _tm.time()

            for m in xrange(ooTR):
                lw.rpg_devroye(oo.rn, oo.us[m] + ARo[m], out=oo.ws[m])  ######  devryoe
            t3 = _tm.time()

            #  find the mean.  For the HISTORY TERM
            for i in vInds:
                for j in vInds:
                    HcM[i-2, j-2] = 0
                    for m in xrange(oo.TR):
                        sts = Msts[m]
                        for iss in xrange(len(sts)-1):
                            t0  = sts[iss]
                            t1  = sts[iss+1]
                            #t1  = oo.N-1 if t1 > oo.N else t1
                            isi = t1 - t0
                            #mlt = 1 if i == j else 0.5
                            HcM[i-2, j-2] += _N.sum(oo.ws[m, t0+1:t1+1]*Hbf[0:t1-t0, i]*Hbf[0:t1-t0, j])
                            # if i == j:
                            #     HcM[i-2, j-2] += 1./1.**2

                RHS[i, 0] = 0
                for m in xrange(oo.TR):
                    O = oo.kp[m]/oo.ws[m] - oo.us[m]
                    sts = Msts[m]

                    for iss in xrange(len(sts)-1):
                        t0  = sts[iss]
                        t1  = sts[iss+1]
                        RHS[i, 0] += _N.sum(oo.ws[m, t0+1:t1+1]*Hbf[0:t1-t0, i]*O[t0+1:t1+1])
                        RHS[i, 0] += 0 / 1.**2
                        for cj in cInds:
                            RHS[i, 0] -= _N.sum(oo.ws[m, t0+1:t1+1]*Hbf[0:t1-t0, i]*Hbf[0:t1-t0, cj])*RHS[cj, 0]
            print RHS

            vm = _N.linalg.solve(HcM, RHS[2:6])
            Cov = _N.linalg.inv(HcM)
            print vm

            cfs = _N.random.multivariate_normal(vm[:, 0], Cov)

            RHS[2:6,0] = cfs
            #RHS[2:6, 0] = vm[:, 0]

            #print HcM
            #vv = _N.dot(Hbf, RHS)
            #print vv.shape
            #print oo.loghist.shape
            _N.dot(Hbf, RHS[:, 0], out=oo.loghist)
            oo.stitch_Hist(ARo, oo.loghist, Msts)

            if ooTR == 1:
                oo.ws   = oo.ws.reshape(1, ooN+1)
            _N.divide(oo.kp, oo.ws, out=kpOws)

            #  Now that we have PG variables, construct Gaussian timeseries
            #  ws(it+1)    using u(it), F0(it), smpx(it)

            #  cov matrix, prior of aS 

            ########     per trial offset sample  burns==None, only psth fit
            if burns is None: 
                Ons  = kpOws - ARo
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
                if not oo.bIndOffset:
                    oo.us[:] = _N.mean(oo.us)
                oo.smp_u[:, it] = oo.us
        oo.VIS = ARo
            


    def dump(self, dir=None):
        oo    = self
        pcklme = [oo]
        #oo.Bsmpx = None
        oo.smpx  = None
        oo.wts   = None
        oo.uts   = None
        oo._d    = None
        oo.lfc   = None
        oo.rts   = None
        oo.zts   = None

        if dir is None:
            dmp = open("oo.dump", "wb")
        else:
            dmp = open("%s/oo.dump" % dir, "wb")
        pickle.dump(pcklme, dmp, -1)
        dmp.close()

        # import pickle
        # with open("mARp.dump", "rb") as f:
        # lm = pickle.load(f)

    def run(self, runDir=None, trials=None, minSpkCnt=0): ###########  RUN
        oo     = self    #  call self oo.  takes up less room on line
        if oo.processes > 1:
            os.system("taskset -p 0xffffffff %d" % os.getpid())
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
        oo.setParams()
        t1    = _tm.time()
        tmpNOAR = oo.noAR
        oo.noAR = True
        if self.__class__.__name__ == "mcmcARp":  #  
            oo.gibbsSamp(burns=oo.psthBurns)
        else:
            oo.__class__.__bases__[0].gibbsSamp(self, burns=oo.psthBurns)

        oo.noAR = tmpNOAR
        oo.gibbsSamp()
        t2    = _tm.time()
        print (t2-t1)

    def runLatent(self, pckl, trials=None, useMeanOffset=False): ###########  RUN
        """
        """
        oo     = self    #  call self oo.  takes up less room on line
        oo.setname = os.getcwd().split("/")[-1]

        oo.Cs          = len(oo.freq_lims)
        oo.C           = oo.Cn + oo.Cs
        oo.R           = 1
        oo.k           = 2*oo.C + oo.R
        #  x0  --  Gaussian prior
        oo.u_x00        = _N.zeros(oo.k)
        oo.s2_x00       = _arl.dcyCovMat(oo.k, _N.ones(oo.k), 0.4)

        oo.loadDat(trials)
        oo.setParams()
        oo.us = pckl[0]
        oo.q2 = _N.ones(oo.TR)*pckl[1]
        oo.F0 = _N.zeros(oo.k)
        print len(pckl[2])
        for l in xrange(len(pckl[2])):
            oo.F0 += (-1*_Npp.polyfromroots(pckl[2][l])[::-1].real)[1:]
        oo.F0 /= len(pckl[2])
        oo.aS = pckl[3]
        
        oo.latentState(useMeanOffset=useMeanOffset)

