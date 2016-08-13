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
import kfARlib1 as _kfar1
import pyPG as lw
from ARcfSmpl import ARcfSmpl, FilteredTimeseries

import commdefs as _cd

from ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF
import os

from multiprocessing import Pool
import mcmcARspk1_h

#  add slow AR1 component
#  smpx1, Bsmpx1    q2_1, smp_q2_1
#  F1, V_00, x_00, 

class mcmcARp1(mcmcARspk1_h.mcmcARspk1):
    #  Description of model
    rn            = None    #  used for count data
    k             = None
    Cn            = None;    Cs            = None;    C             = None
    kntsPSTH      = None;    dfPSTH        = None
    ID_q2         = True
    use_prior     = _cd.__COMP_REF__
    AR2lims       = None

    #  Sampled 
    ranks         = None
    pgs           = None
    fs            = None
    amps          = None
    dt            = None
    mnStds        = None

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

    #  lmits for AR1
    #a_F1      = -1;    b_F1      =  1
    a_F0      = 0.99;    b_F0      =  1

    #  coefficient sampling
    fSigMax       = 500.    #  fixed parameters
    freq_lims     = [[1 / .85, fSigMax]]
    sig_ph0L      = -1
    sig_ph0H      = 0

    _d1           = None
    
    def gibbsSamp(self, burns=None):  ###########################  GIBBSSAMPH
        oo          = self
        ooTR        = oo.TR
        ook         = oo.k
        ooNMC       = oo.NMC
        ooN         = oo.N
        oo.x00         = _N.array(oo.smpx[:, 0])
        oo.V00         = _N.zeros((ooTR, ook, ook))
        oo.loghist  = _N.zeros(oo.N+1)

        ARo   = _N.empty((ooTR, oo._d.N+1))
        
        kpOws = _N.empty((ooTR, ooN+1))
        lv_f     = _N.zeros((ooN+1, ooN+1))
        lv_u     = _N.zeros((ooTR, ooTR))
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

        it    = -1

        oous_rs = oo.us.reshape((ooTR, 1))
        lrnBadLoc = _N.empty((oo.TR, oo.N+1), dtype=_N.bool)
        runTO = ooNMC + oo.burn - 1 if (burns is None) else (burns - 1)
        oo.allocateSmp(runTO+1)

        BaS = _N.empty(oo.N+1)

        #  H shape    100 x 9
        Hbf = oo.Hbf

        RHS = _N.empty((oo.histknots, 1))

        if oo.h0_1 > 0:   #  first few are 0s
            #cInds = _N.array([0, 1, 5, 6, 7, 8, 9, 10])
            cInds = _N.array([0, 4, 5, 6, 7, 8, 9])
            #vInds = _N.array([2, 3, 4])
            vInds = _N.array([1, 2, 3,])
            RHS[cInds, 0] = 0
            RHS[0, 0] = -5
        else:
            #cInds = _N.array([5, 6, 7, 8, 9, 10])
            cInds = _N.array([4, 5, 6, 7, 8, 9,])
            vInds = _N.array([0, 1, 2, 3, ])
            #vInds = _N.array([0, 1, 2, 3, 4])
            RHS[cInds, 0] = 0

        Msts = []
        for m in xrange(ooTR):
            Msts.append(_N.where(oo.y[m] == 1)[0])
        HcM  = _N.empty((len(vInds), len(vInds)))

        HbfExpd = _N.empty((oo.histknots, ooTR, oo.N+1))
        #  HbfExpd is 11 x M x 1200
        #  find the mean.  For the HISTORY TERM
        for i in xrange(oo.histknots):
            for m in xrange(oo.TR):
                sts = Msts[m]
                HbfExpd[i, m, 0:sts[0]] = 0
                for iss in xrange(len(sts)-1):
                    t0  = sts[iss]
                    t1  = sts[iss+1]
                    HbfExpd[i, m, t0+1:t1+1] = Hbf[0:t1-t0, i]
                HbfExpd[i, m, sts[-1]+1:] = 0

        while (it < runTO):
            t1 = _tm.time()
            it += 1
            print it
            if (it % 10) == 0:
                print it
            #  generate latent AR state
            oo._d.f_x[:, 0,0,0]     = oo.x00
            if it == 0:
                for m in xrange(ooTR):
                    oo._d.f_V[m, 0]     = oo.s2_x00
            else:
                oo._d.f_V[:, 0]     = _N.mean(oo._d.f_V[:, 1:], axis=1)

            _N.dot(oo.B.T, oo.aS, out=BaS)
            ###  PG latent variable sample

            t2 = _tm.time()

            #oo.build_addHistory(ARo, oo.smpx, BaS, oo.us)

            for m in xrange(ooTR):
                lw.rpg_devroye(oo.rn, oo.smpx[m] + oo.us[m] + BaS + ARo[m], out=oo.ws[m])  ######  devryoe
            t3 = _tm.time()
            if ooTR == 1:
                oo.ws   = oo.ws.reshape(1, ooN+1)
            _N.divide(oo.kp, oo.ws, out=kpOws)

            O = kpOws - oo.smpx - oous_rs - BaS

            iOf = vInds[0]   #  offset HcM index with RHS index.
            for i in vInds:
                for j in vInds:
                    HcM[i-iOf, j-iOf] = _N.sum(oo.ws*HbfExpd[i]*HbfExpd[j])

                RHS[i, 0] = _N.sum(oo.ws*HbfExpd[i]*O)
                for cj in cInds:
                    RHS[i, 0] -= _N.sum(oo.ws*HbfExpd[i]*HbfExpd[cj])*RHS[cj, 0]

            vm = _N.linalg.solve(HcM, RHS[vInds])
            Cov = _N.linalg.inv(HcM)
            print vm
            cfs = _N.random.multivariate_normal(vm[:, 0], Cov, size=1)

            RHS[vInds,0] = cfs[0]
            oo.smp_hS[:, it] = RHS[:, 0]

            #RHS[2:6, 0] = vm[:, 0]
            #print HcM
            #vv = _N.dot(Hbf, RHS)
            #print vv.shape
            #print oo.loghist.shape
            _N.dot(Hbf, RHS[:, 0], out=oo.loghist)
            oo.smp_hist[:, it] = oo.loghist
            oo.stitch_Hist(ARo, oo.loghist, Msts)

            #  Now that we have PG variables, construct Gaussian timeseries
            #  ws(it+1)    using u(it), F0(it), smpx(it)

            oo._d.y = kpOws - BaS - ARo - oous_rs
            oo._d.copyParams(oo.F0, oo.q2)
            oo._d.Rv[:, :] =1 / oo.ws[:, :]   #  time dependent noise

            #  cov matrix, prior of aS 

            ########     PSTH sample
            if oo.bpsth:
                Oms  = kpOws - oo.smpx - ARo - oous_rs
                _N.einsum("mn,mn->n", oo.ws, Oms, out=smWimOm)   #  sum over
                ilv_f  = _N.diag(_N.sum(oo.ws, axis=0))
                #  diag(_N.linalg.inv(Bi)) == diag(1./Bi).  Bii = inv(Bi)
                _N.fill_diagonal(lv_f, 1./_N.diagonal(ilv_f))
                lm_f  = _N.dot(lv_f, smWimOm)  #  nondiag of 1./Bi are inf
                #  now sample
                iVAR = _N.dot(oo.B, _N.dot(ilv_f, oo.B.T)) + iD_f
                t4a = _tm.time()
                VAR  = _N.linalg.inv(iVAR)  #  knots x knots
                t4b = _tm.time()
                #iBDBW = _N.linalg.inv(BDB + lv_f)   # BDB not diag
                #Mn    = oo.u_a + _N.dot(DB, _N.dot(iBDBW, lm_f - BTua))

                Mn = oo.u_a + _N.dot(DB, _N.linalg.solve(BDB + lv_f, lm_f - BTua
))

                """
                Oms  = kpOws - oo.smpx - ARo - oous_rs
                _N.einsum("mn,mn->n", oo.ws, Oms, out=smWimOm)   #  sum over 
                ilv_f  = _N.diag(_N.sum(oo.ws, axis=0))
                #  diag(_N.linalg.inv(Bi)) == diag(1./Bi).  Bii = inv(Bi)
                _N.fill_diagonal(lv_f, 1./_N.diagonal(ilv_f))
                lm_f  = _N.dot(lv_f, smWimOm)  #  nondiag of 1./Bi are inf
                #  now sample
                iVAR = _N.dot(oo.B, _N.dot(ilv_f, oo.B.T)) + iD_f
                VAR  = _N.linalg.inv(iVAR)  #  knots x knots
                iBDBW = _N.linalg.inv(BDB + lv_f)   # BDB not diag
                Mn    = oo.u_a + _N.dot(DB, _N.dot(iBDBW, lm_f - BTua))
                """

                oo.aS   = _N.random.multivariate_normal(Mn, VAR, size=1)[0, :]
                oo.smp_aS[it, :] = oo.aS
            else:
                oo.aS   = _N.zeros(4)

            ########     per trial offset sample  burns==None, only psth fit

            Ons  = kpOws - oo.smpx - ARo - BaS

            #  solve for the mean of the distribution
            H    = _N.ones((oo.TR-1, oo.TR-1)) * _N.sum(oo.ws[0])
            uRHS = _N.empty(oo.TR-1)
            for dd in xrange(1, oo.TR):
                H[dd-1, dd-1] += _N.sum(oo.ws[dd])
                uRHS[dd-1] = _N.sum(oo.ws[dd]*Ons[dd] - oo.ws[0]*Ons[0])

            MM  = _N.linalg.solve(H, uRHS)
            Cov = _N.linalg.inv(H)

            oo.us[1:] = _N.random.multivariate_normal(MM, Cov, size=1)
            oo.us[0]  = -_N.sum(oo.us[1:])
            oo.smp_u[:, it] = oo.us



                # Ons  = kpOws - oo.smpx - ARo - BaS
                # _N.einsum("mn,mn->m", oo.ws, Ons, out=smWinOn)  #  sum over trials
                # ilv_u  = _N.diag(_N.sum(oo.ws, axis=1))  #  var  of LL
                # #  diag(_N.linalg.inv(Bi)) == diag(1./Bi).  Bii = inv(Bi)
                # _N.fill_diagonal(lv_u, 1./_N.diagonal(ilv_u))
                # lm_u  = _N.dot(lv_u, smWinOn)  #  nondiag of 1./Bi are inf, mean LL
                # #  now sample
                # iVAR = ilv_u + iD_u
                # VAR  = _N.linalg.inv(iVAR)  #
                # Mn    = _N.dot(VAR, _N.dot(ilv_u, lm_u) + iD_u_u_u)
                # oo.us[:]  = _N.random.multivariate_normal(Mn, VAR, size=1)[0, :]
                # oo.smp_u[:, it] = oo.us

            t4 = _tm.time()
            t5 = t4
            if not oo.noAR:
            #  _d.F, _d.N, _d.ks, 









                oo.smpx = _kfar1.armdl_FFBS_1itr(oo._d, multitrial=True)
                oo.Bsmpx[:, it]    = oo.smpx
                stds = _N.std(oo.Bsmpx[:, it], axis=1)
                oo.mnStds[it] = _N.mean(stds, axis=0)
                print "mnStd  %.3f" % oo.mnStds[it]
                t5 = _tm.time()

                ##  sample AR coefficient here
                F0AA = _N.sum(_N.sum(oo.smpx[:, 0:-1]*oo.smpx[:, 0:-1], axis=1))
                F0BB = _N.sum(_N.sum(oo.smpx[:, 0:-1] * oo.smpx[:, 1:], axis=1))

                F0std= _N.sqrt(oo.q2[0]/F0AA)  #  
                F0a, F0b  = (oo.a_F0 - F0BB/F0AA) / F0std, (oo.b_F0 - F0BB/F0AA) / F0std

                #oo.F0=F0BB/F0AA#+F0std*_ss.truncnorm.rvs(F0a, F0b)
                oo.F0=_N.array([F0BB/F0AA])#+F0std*_ss.truncnorm.rvs(F0a, F0b)

                print "F0 is %.3f      " % (F0BB/F0AA)
                print "F0std is %.3f      " % F0std
                print "F0a is %.3f      " % F0a
                print "F0b is %.3f      " % F0b
                print "oo.F0 is %.3f"     % oo.F0

                #  sample u     WE USED TO Do this after smpx
                #  u(it+1)    using ws(it+1), F0(it), smpx(it+1), ws(it+1)

                #   sample q2
                a = oo.a_q2 + 0.5*(oo.N*oo.TR+1)  #  N + 1 - 1
                #rsd_stp = oo.smpx1[0, 1:] - oo.F1*oo.smpx1[0, 0:-1]
                rsd_stp = oo.smpx[:, 1:] - oo.F0*oo.smpx[:, 0:-1]
                #BB = oo.B_q2_1 + 0.5 * _N.dot(rsd_stp, rsd_stp)
                BB = oo.B_q2 + 0.5 * _N.sum(_N.sum(rsd_stp*rsd_stp, axis=1))
                #print rsd_stp
                #print "!!!!  %(a).3e   %(BB).3e" % {"a" : a, "BB" : BB}

                oo.q2 = _N.ones(oo.TR)*_ss.invgamma.rvs(a, scale=BB)
                oo.smp_q2[:, it]= oo.q2


            t6 = _tm.time()
            print "t2-t1 %.3f" % (t2-t1)
            print "t3-t2 %.3f" % (t3-t2)
            print "t4-t3 %.3f" % (t4-t3)
            print "t5-t4 %.3f" % (t5-t4)
            print "t6-t5 %.3f" % (t6-t5)
            print "gibbs iter %.3f" % (t6-t1)

    def latentState(self, burns=None):  ###########################  GIBBSSAMPH
        oo          = self
        ooTR        = oo.TR
        ook         = oo.k
        ooNMC       = oo.NMC
        ooN         = oo.N
        oo.x00         = _N.array(oo.smpx[:, 2])
        oo.V00         = _N.zeros((ooTR, ook, ook))

        ARo   = _N.empty((ooTR, oo._d.N+1))
        kpOws = _N.empty((ooTR, ooN+1))
        
        it    = -1

        oous_rs = oo.us.reshape((ooTR, 1))
        runTO = ooNMC + oo.burn - 1 if (burns is None) else (burns - 1)
        oo.allocateSmp(runTO+1)

        BaS = _N.empty(oo.N+1)
        _N.dot(oo.B.T, oo.aS, out=BaS)

        while (it < runTO):
            t1 = _tm.time()
            it += 1
            print it
            if (it % 10) == 0:
                print it
            #  generate latent AR state
            oo._d.f_x[:, 0, :, 0]     = oo.x00
            if it == 0:
                for m in xrange(ooTR):
                    oo._d.f_V[m, 0]     = oo.s2_x00
            else:
                oo._d.f_V[:, 0]     = _N.mean(oo._d.f_V[:, 1:], axis=1)

            ###  PG latent variable sample

            oo.build_addHistory(ARo, oo.smpx[:, 2:, 0], BaS, oo.us)
            for m in xrange(ooTR):
                lw.rpg_devroye(oo.rn, oo.smpx[m, 2:, 0] + oo.us[m] + BaS + ARo[m], out=oo.ws[m])  ######  devryoe
            t3 = _tm.time()
            if ooTR == 1:
                oo.ws   = oo.ws.reshape(1, ooN+1)
            _N.divide(oo.kp, oo.ws, out=kpOws)

            #  Now that we have PG variables, construct Gaussian timeseries
            #  ws(it+1)    using u(it), F0(it), smpx(it)

            oo._d.y = kpOws - BaS - ARo - oous_rs
            oo._d.copyParams(oo.F0, oo.q2)
            oo._d.Rv[:, :] =1 / oo.ws[:, :]   #  time dependent noise

            #  cov matrix, prior of aS 
            tpl_args = zip(oo._d.y, oo._d.Rv, oo._d.Fs, oo.q2, oo._d.Ns, oo._d.ks, oo._d.f_x[:, 0], oo._d.f_V[:, 0])

            for m in xrange(ooTR):
                oo.smpx[m, 2:], oo._d.f_x[m], oo._d.f_V[m] = _kfar.armdl_FFBS_1itrMP(tpl_args[m])
                oo.smpx[m, 1, 0:ook-1]   = oo.smpx[m, 2, 1:]
                oo.smpx[m, 0, 0:ook-2]   = oo.smpx[m, 2, 2:]
                oo.Bsmpx[m, it, 2:]    = oo.smpx[m, 2:, 0]
            stds = _N.std(oo.Bsmpx[:, it, 2:], axis=1)
            oo.mnStds[it] = _N.mean(stds, axis=0)
            print "mnStd  %.3f" % oo.mnStds[it]
            t6 = _tm.time()

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

        dmp = open("mARp.dump", "wb")
        pickle.dump(pcklme, dmp)
        dmp.close()

        # import pickle
        # with open("mARp.dump", "rb") as f:
        # lm = pickle.load(f)

    def init1(self):
        #  AR1 part init
        oo._d1 = _kfardat.KFARGauObsDat(1, oo.N, 1, onetrial=True)
        oo._d1.copyData(_N.empty(oo.N+1), _N.empty(oo.N+1), onetrial=True)   #  dummy data copied

    def run(self, runDir=None, trials=None, minSpkCnt=0): ###########  RUN
        oo     = self    #  call self oo.  takes up less room on line
        oo.setname = os.getcwd().split("/")[-1]

        oo.k           = 1
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
        if self.__class__.__name__ == "mcmcARp1":  #  
            oo.gibbsSamp(burns=oo.psthBurns)
        else:
            oo.__class__.__bases__[0].gibbsSamp(self, burns=oo.psthBurns)

        oo.noAR = tmpNOAR
        oo.gibbsSamp()
        t2    = _tm.time()
        print (t2-t1)

    def runLatent(self, pckl, trials=None): ###########  RUN
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
        oo.q2 = _N.ones(len(trials))*pckl[1]
        oo.F0 = (-1*_Npp.polyfromroots(pckl[2])[::-1].real)[1:]
        
        oo.latentState()
        print (t2-t1)
