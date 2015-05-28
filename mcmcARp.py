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

import scipy.stats as _ss
from kassdirs import resFN, datFN

import utilities as _U

import numpy.polynomial.polynomial as _Npp
import time as _tm
import ARlib as _arl
import kfARlibMPmv as _kfar
import LogitWrapper as lw
from ARcfSmpl import ARcfSmpl, FilteredTimeseries

import commdefs as _cd

from ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF

import os

import mcmcARspk as mcmcARspk

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
    
    def gibbsSamp(self, burns=None):  ###########################  GIBBSSAMPH
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

        it    = -1

        oous_rs = oo.us.reshape((ooTR, 1))
        lrnBadLoc = _N.empty((oo.TR, oo.N+1), dtype=_N.bool)
        runTO = ooNMC + oo.burn - 1 if (burns is None) else (burns - 1)
        oo.allocateSmp(runTO+1)

        BaS = _N.empty(oo.N+1)

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

            _N.dot(oo.B.T, oo.aS, out=BaS)
            ###  PG latent variable sample

            #tPG1 = _tm.time()
            oo.build_addHistory(ARo, oo.smpx[:, 2:, 0], BaS, oo.us)

            for m in xrange(ooTR):
                lw.rpg_devroye(oo.rn, oo.smpx[m, 2:, 0] + oo.us[m] + BaS + ARo[m], out=oo.ws[m])  ######  devryoe
                    
            if ooTR == 1:
                oo.ws   = oo.ws.reshape(1, ooN+1)
            _N.divide(oo.kp, oo.ws, out=kpOws)

            #  Now that we have PG variables, construct Gaussian timeseries
            #  ws(it+1)    using u(it), F0(it), smpx(it)

            oo._d.y = kpOws - BaS - ARo - oous_rs
            oo._d.copyParams(oo.F0, oo.q2)
            oo._d.Rv[:, :] =1 / oo.ws[:, :]   #  time dependent noise

            #  cov matrix, prior of aS 

            ########     PSTH sample
            if oo.bpsth:
                Oms  = kpOws - oo.smpx[..., 2:, 0] - ARo - oous_rs
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

                oo.aS   = _N.random.multivariate_normal(Mn, VAR, size=1)[0, :]
                oo.smp_aS[it, :] = oo.aS
            else:
                oo.aS   = _N.zeros(4)

            ########     per trial offset sample  burns==None, only psth fit
            if burns is None: 
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

            if not oo.noAR:
            #  _d.F, _d.N, _d.ks, 
                tpl_args = zip(oo._d.y, oo._d.Rv, oo._d.Fs, oo.q2, oo._d.Ns, oo._d.ks, oo._d.f_x[:, 0], oo._d.f_V[:, 0])

                for m in xrange(ooTR):
                    oo.smpx[m, 2:], oo._d.f_x[m], oo._d.f_V[m] = _kfar.armdl_FFBS_1itrMP(tpl_args[m])
                    oo.smpx[m, 1, 0:ook-1]   = oo.smpx[m, 2, 1:]
                    oo.smpx[m, 0, 0:ook-2]   = oo.smpx[m, 2, 2:]
                    oo.Bsmpx[m, it, 2:]    = oo.smpx[m, 2:, 0]
                stds = _N.std(oo.Bsmpx[:, it, 2:], axis=1)
                mnStd = _N.mean(stds, axis=0)
                print "mnStd  %.3f" % mnStd

                if not oo.bFixF:   
                    ARcfSmpl(oo.lfc, ooN+1, ook, oo.AR2lims, oo.smpx[:, 1:, 0:ook], oo.smpx[:, :, 0:ook-1], oo.q2, oo.R, oo.Cs, oo.Cn, alpR, alpC, oo.TR, prior=oo.use_prior, accepts=30, aro=oo.ARord, sig_ph0L=oo.sig_ph0L, sig_ph0H=oo.sig_ph0H)  
                    oo.F_alfa_rep = alpR + alpC   #  new constructed
                    prt, rank, f, amp = ampAngRep(oo.F_alfa_rep, f_order=True)
                    print prt
                ut, wt = FilteredTimeseries(ooN+1, ook, oo.smpx[:, 1:, 0:ook], oo.smpx[:, :, 0:ook-1], oo.q2, oo.R, oo.Cs, oo.Cn, alpR, alpC, oo.TR)
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
                        #oo.rsds[it, m] = _N.dot(rsd_stp, rsd_stp.T)
                        BB2 += 0.5 * _N.dot(rsd_stp, rsd_stp.T)
                    oo.q2[:] = _ss.invgamma.rvs(oo.a2, scale=BB2)

                oo.smp_q2[:, it]= oo.q2


            t2 = _tm.time()
            print "gibbs iter %.3f" % (t2-t1)

    def build_lrnLambda2(self, tr):
        oo = self
        #  lmbda2 is short snippet of after-spike depression behavior
        lrn = _N.ones(oo.N + 1)
        lh    = len(oo.l2)

        spkts = _N.where(oo.y[tr] == 1)[0]

        #  P(isi | t - t0 = t').  This prob is zero for isi < t-t0.  What is the shape of the distribution for longer isis?
        for t in spkts:
            maxL = lh if t + lh <= oo.N else oo.N - t
            lrn[t+1:t+1 + maxL] = oo.l2[0:maxL]

        ###  It stands to reason that at t=0, the actual state of the
        #    spiking history lambda is not always == 1, ie a spike
        #    occurred just slightly prior to t=0.  Let's just assume
        #    some virtual observation, and set
        bDone = False

        while not bDone:
            ivrtISI = int(oo.mean_isi_1st2spks*_N.random.exponential())
            if (ivrtISI > 2) and (ivrtISI > spkts[0]):
                bDone = True
        #  if vrtISI == oo.y[tr, 0] + 2, put virtual 2 bins back in time

        bckwds = ivrtISI - spkts[0]

        if bckwds < lh:
            lrn[0:lh-bckwds] = oo.l2[bckwds:]

        return lrn

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

    def run(self, runDir=None, trials=None, minSpkCnt=0): ###########  RUN
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
