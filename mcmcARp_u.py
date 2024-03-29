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

    def gibbsSamp(self, burns=None):  ###########################  GIBBSSAMPH
        oo          = self
        ooTR        = oo.TR
        ook         = oo.k
        ooNMC       = oo.NMC
        ooN         = oo.N
        oo.x00         = _N.array(oo.smpx[:, 2])
        oo.V00         = _N.zeros((ooTR, ook, ook))

        print "oo.mcmcRunDir    %s" % oo.mcmcRunDir
        if oo.mcmcRunDir is None:
            print "here!!!!!!!!!!!!!!"
            oo.mcmcRunDir = ""
        elif (len(oo.mcmcRunDir) > 0) and (oo.mcmcRunDir[-1] != "/"):
            oo.mcmcRunDir += "/"

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

        oo.us[:] = 0
        oous_rs = oo.us.reshape((ooTR, 1))
        lrnBadLoc = _N.empty((oo.TR, oo.N+1), dtype=_N.bool)
        runTO = ooNMC + oo.burn - 1 if (burns is None) else (burns - 1)
        oo.allocateSmp(runTO+1)

        BaS = _N.empty(oo.N+1)

        if oo.processes > 1:
            pool = Pool(processes=oo.processes)

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

            t2 = _tm.time()

            oo.build_addHistory(ARo, oo.smpx[:, 2:, 0], BaS, oo.us, oo.knownSig)

            for m in xrange(ooTR):
                lw.rpg_devroye(oo.rn, oo.smpx[m, 2:, 0] + oo.us[m] + BaS + ARo[m] + oo.knownSig[m], out=oo.ws[m])  ######  devryoe
            t3 = _tm.time()
            if ooTR == 1:
                oo.ws   = oo.ws.reshape(1, ooN+1)
            _N.divide(oo.kp, oo.ws, out=kpOws)

            #  Now that we have PG variables, construct Gaussian timeseries
            #  ws(it+1)    using u(it), F0(it), smpx(it)

            oo._d.y = kpOws - BaS - ARo - oous_rs - oo.knownSig
            oo._d.copyParams(oo.F0, oo.q2)
            oo._d.Rv[:, :] =1 / oo.ws[:, :]   #  time dependent noise

            #  cov matrix, prior of aS 

            ########     PSTH sample
            if oo.bpsth:
                Oms  = kpOws - oo.smpx[..., 2:, 0] - ARo - oous_rs - oo.knownSig
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

                Mn = oo.u_a + _N.dot(DB, _N.linalg.solve(BDB + lv_f, lm_f - BTua))

                t4c = _tm.time()

                oo.aS   = _N.random.multivariate_normal(Mn, VAR, size=1)[0, :]
                oo.smp_aS[it, :] = oo.aS
            else:
                oo.aS   = _N.zeros(4)

            ########     per trial offset sample  burns==None, only psth fit
            if burns is None: 
                Ons  = kpOws - oo.smpx[..., 2:, 0] - ARo - BaS - oo.knownSig

                #  solve for the mean of the distribution
                H    = _N.ones((oo.TR-1, oo.TR-1)) * _N.sum(oo.ws[0])
                RHS = _N.empty(oo.TR-1)
                for dd in xrange(1, oo.TR):
                    H[dd-1, dd-1] += _N.sum(oo.ws[dd])
                    RHS[dd-1] = _N.sum(oo.ws[dd]*Ons[dd] - oo.ws[0]*Ons[0])

                MM  = _N.linalg.solve(H, RHS)
                Cov = _N.linalg.inv(H)
                
                oo.us[1:] = _N.random.multivariate_normal(MM, Cov)
                oo.us[0]  = -_N.sum(oo.us[1:])
                if not oo.bIndOffset:
                    oo.us[:] = _N.mean(oo.us)
                oo.smp_u[:, it] = oo.us

            t4 = _tm.time()
            t5 = t4
            if not oo.noAR:
            #  _d.F, _d.N, _d.ks, 
                tpl_args = zip(oo._d.y, oo._d.Rv, oo._d.Fs, oo.q2, oo._d.Ns, oo._d.ks, oo._d.f_x[:, 0], oo._d.f_V[:, 0])

                if oo.processes == 1:
                    for m in xrange(ooTR):
                        oo.smpx[m, 2:], oo._d.f_x[m], oo._d.f_V[m] = _kfar.armdl_FFBS_1itrMP(tpl_args[m])
                        oo.smpx[m, 1, 0:ook-1]   = oo.smpx[m, 2, 1:]
                        oo.smpx[m, 0, 0:ook-2]   = oo.smpx[m, 2, 2:]
                        #oo.Bsmpx[m, it, 2:]    = oo.smpx[m, 2:, 0]
                else:
                    sxv = pool.map(_kfar.armdl_FFBS_1itrMP, tpl_args)
                    for m in xrange(ooTR):
                        oo.smpx[m, 2:] = sxv[m][0]
                        oo._d.f_x[m] = sxv[m][1]
                        oo._d.f_V[m] = sxv[m][2]
                        oo.smpx[m, 1, 0:ook-1]   = oo.smpx[m, 2, 1:]
                        oo.smpx[m, 0, 0:ook-2]   = oo.smpx[m, 2, 2:]
                        #oo.Bsmpx[m, it, 2:]    = oo.smpx[m, 2:, 0]
                        
                stds = _N.std(oo.smpx[:, 2:, 0], axis=1)
                oo.mnStds[it] = _N.mean(stds, axis=0)
                print "mnStd  %.3f" % oo.mnStds[it]
                t5 = _tm.time()
                if not oo.bFixF:   
                    ARcfSmpl(oo.lfc, ooN+1, ook, oo.AR2lims, oo.smpx[:, 1:, 0:ook], oo.smpx[:, :, 0:ook-1], oo.q2, oo.R, oo.Cs, oo.Cn, alpR, alpC, oo.TR, prior=oo.use_prior, accepts=30, aro=oo.ARord, sig_ph0L=oo.sig_ph0L, sig_ph0H=oo.sig_ph0H)  
                    oo.F_alfa_rep = alpR + alpC   #  new constructed
                    prt, rank, f, amp = ampAngRep(oo.F_alfa_rep, f_order=True)
                    print prt
                #ut, wt = FilteredTimeseries(ooN+1, ook, oo.smpx[:, 1:, 0:ook], oo.smpx[:, :, 0:ook-1], oo.q2, oo.R, oo.Cs, oo.Cn, alpR, alpC, oo.TR)
                #ranks[it]    = rank
                oo.allalfas[it] = oo.F_alfa_rep

                for m in xrange(ooTR):
                    #oo.wts[m, it, :, :]   = wt[m, :, :, 0]
                    #oo.uts[m, it, :, :]   = ut[m, :, :, 0]
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

            t6 = _tm.time()

            if (it > 1) and (it % oo.peek == 0):
                fig = _plt.figure(figsize=(8, 8))
                fig.add_subplot(3, 1, 1)
                _plt.plot(oo.amps[1:it, 0])
                fig.add_subplot(3, 1, 2)
                _plt.plot(oo.fs[1:it, 0])
                fig.add_subplot(3, 1, 3)
                _plt.plot(oo.mnStds[1:it])

                _plt.savefig("%(dir)stmp-fsamps%(it)d" % {"dir" : oo.mcmcRunDir, "it" : it})
                _plt.close()

                fig = _plt.figure(figsize=(8, 8))
                fig.add_subplot(2, 1, 1)
                _plt.plot(oo.smpx[0, 2:, 0])
                _plt.plot(oo.y[0])
                fig.add_subplot(2, 1, 2)
                _plt.plot(oo.smpx[1, 2:, 0])
                _plt.plot(oo.y[1])
                _plt.savefig("%(dir)smpxs%(it)d" % {"dir" : oo.mcmcRunDir, "it" : it})                

                W   = 60
                #  do an sta of inferred state
                stwfs = []
                for tr in xrange(oo.TR):
                    sts = _N.where(oo.y[tr] == 1)[0]
    
                for t in sts:
                    if (t > oo.t0+W) and (t < oo.t1-W):
                        stwfs.append(oo.smpx[tr, t-W:t+W, 0])


                rstwfs = []
                inds   = _N.arange(oo.TR)
                _N.random.shuffle(inds)
                for tr in xrange(oo.TR):
                    rtr = inds[tr]
                    sts = _N.where(oo.y[rtr] == 1)[0]
    
                    for t in sts:
                        if (t > oo.t0+W) and (t < oo.t1-W):
                            rstwfs.append(oo.smpx[tr, t-W:t+W, 0])

                astwfs  = _N.array(stwfs)
                arstwfs = _N.array(rstwfs)

                _N.savetxt("%(dir)sstlwfs%(it)d.dat" % {"dir" : oo.mcmcRunDir, "it" : it}, astwfs, fmt=("%.3f " * 2*W))
                _N.savetxt("%(dir)srstlwfs%(it)d.dat" % {"dir" : oo.mcmcRunDir, "it" : it}, arstwfs, fmt=("%.3f " * 2*W))
                _N.savetxt("%(dir)ssmpx%(it)d.dat" % {"dir" : oo.mcmcRunDir, "it" : it}, oo.smpx[:, 2:, 0].T, fmt=("%.3f " * oo.TR))

    def latentState(self, burns=None, useMeanOffset=False):  ###########################  GIBBSSAMPH
        oo          = self

        ooTR        = oo.TR
        ook         = oo.k
        ooNMC       = oo.NMC
        ooN         = oo.N
        oo.x00         = _N.array(oo.smpx[:, 2])
        oo.V00         = _N.zeros((ooTR, ook, ook))
        alpR   = oo.F_alfa_rep[0:oo.R]
        alpC   = oo.F_alfa_rep[oo.R:]

        ARo   = _N.empty((ooTR, oo._d.N+1))
        kpOws = _N.empty((ooTR, ooN+1))
        oo.uts          = _N.empty((oo.TR, oo.burn+oo.NMC, oo.R, oo.N+2))
        oo.wts          = _N.empty((oo.TR, oo.burn+oo.NMC, oo.C, oo.N+3))

        it    = -1

        if useMeanOffset:
            oo.us[:] = _N.mean(oo.us)
        oous_rs = oo.us.reshape((ooTR, 1))
        
        runTO = ooNMC + oo.burn - 1 if (burns is None) else (burns - 1)
        oo.allocateSmp(runTO+1, Bsmpx=True)

        BaS = _N.empty(oo.N+1)
        _N.dot(oo.B.T, oo.aS, out=BaS)

        if oo.processes > 1:
            pool = Pool(processes=oo.processes)

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

            oo.build_addHistory(ARo, oo.smpx[:, 2:, 0], BaS, oo.us, oo.knownSig)
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

            if oo.processes == 1:
                for m in xrange(ooTR):
                    oo.smpx[m, 2:], oo._d.f_x[m], oo._d.f_V[m] = _kfar.armdl_FFBS_1itrMP(tpl_args[m])
                    oo.smpx[m, 1, 0:ook-1]   = oo.smpx[m, 2, 1:]
                    oo.smpx[m, 0, 0:ook-2]   = oo.smpx[m, 2, 2:]
                    oo.Bsmpx[m, it, 2:]    = oo.smpx[m, 2:, 0]
            else:
                sxv = pool.map(_kfar.armdl_FFBS_1itrMP, tpl_args)
                for m in xrange(ooTR):
                    oo.smpx[m, 2:] = sxv[m][0]
                    oo._d.f_x[m] = sxv[m][1]
                    oo._d.f_V[m] = sxv[m][2]
                    oo.smpx[m, 1, 0:ook-1]   = oo.smpx[m, 2, 1:]
                    oo.smpx[m, 0, 0:ook-2]   = oo.smpx[m, 2, 2:]
                    oo.Bsmpx[m, it, 2:]    = oo.smpx[m, 2:, 0]

            ut, wt = FilteredTimeseries(ooN+1, ook, oo.smpx[:, 1:, 0:ook], oo.smpx[:, :, 0:ook-1], oo.q2, oo.R, oo.Cs, oo.Cn, alpR, alpC, oo.TR)
            oo.allalfas[it] = oo.F_alfa_rep

            for m in xrange(ooTR):
                oo.wts[m, it, :, :]   = wt[m, :, :, 0]
                oo.uts[m, it, :, :]   = ut[m, :, :, 0]

            stds = _N.std(oo.smpx[:, 2:, 0], axis=1)
            oo.mnStds[it] = _N.mean(stds, axis=0)
            print "mnStd  %.3f" % oo.mnStds[it]
            t6 = _tm.time()


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

