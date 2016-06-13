from filter import bpFilt, lpFilt, gauKer
import mcmcAR as mAR
import ARlib as _arl
import LogitWrapper as lw
import kfardat as _kfardat
import logerfc as _lfc
import commdefs as _cd
import os
import numpy as _N
import kfARlib1 as _kfar
from kassdirs import resFN, datFN
import re as _re
import matplotlib.pyplot as _plt
import scipy.stats as _ss
from cntUtil import Llklhds, cntmdlMCMCOnly, startingValues, startingValuesMw, CtDistLogNorm2
import time as _tm

class mcmcARcntMW(mAR.mcmcAR):
    ###  prior on F0 truncated normal
    alp       = None
    F0        = None;
    a_F0      = -1;    b_F0      =  1

    ###
    mrns      = None;     mus    = None;    mdty    = None
    smp_rnM   = None;     smp_u  = None;    smp_dty = None; smp_m = None
    smp_zs    = None

    m             = None;     zs = None
    smpxOff = False

    LN        = None

    W      = 1

    ###################################   RUN   #########
    def loadDat(self, usewin=None):
        oo     = self    #  call self oo.  takes up less room on line
        #  READ parameters of generative model from file
        #  contains N, k, singleFreqAR, u, beta, dt, stNz
        x_st_cnts = _N.loadtxt(resFN(oo.datafn, dir=oo.setname, env_dirname=oo.env_dirname))

        oo.N   = x_st_cnts.shape[0] - 1
        if oo.t1 == None:
            oo.t1 = oo.N + 1
        oo.N   = oo.t1 - oo.t0 - 1
        
        cols= x_st_cnts.shape[1]
        nWinsInData = cols - 2   #  latent x, true state, win 1, win 2, ...

        if usewin is not None:
            oo.W = len(usewin)
            for l in xrange(len(usewin)):
                usewin[l] = usewin[l]+2
            x_st_cnts[:, range(2, 2+oo.W)] = x_st_cnts[:, usewin]
        else:
            oo.W = nWinsInData

        oo.xT  = x_st_cnts[oo.t0:oo.t1, 0]
        #######  Initialize z
        mH  = x_st_cnts[oo.t0:oo.t1, 1]
        stCol = 1   #  column containing counts
        ctCol = 2   #  column containing counts

        oo.y   = x_st_cnts[oo.t0:oo.t1, ctCol:ctCol+oo.W]
        oo.st  = x_st_cnts[oo.t0:oo.t1, stCol]
        oo.x   = x_st_cnts[oo.t0:oo.t1, 0]

    #  when we don't have xn
    def initGibbs(self):
        oo     = self    #  call self oo.  takes up less room on line
        #  INITIAL samples

        oo.smpx = _N.array(oo.x)#_N.zeros(oo.N+1)
        oo.us, oo.rn, oo.model = startingValuesMw(oo.y, 1, _N.ones((oo.N+1, 1), dtype=_N.int), fillsmpx=oo.smpx, indLH=True)
        # oo.us = _N.array([[-1.734, -1.734],
        #                   [-1.734, -1.734]])
        # oo.rn = _N.array([[130, 250],
        #                   [130, 250]])
        # oo.model = _N.array([[_cd.__BNML__, _cd.__BNML__],
        #                      [_cd.__BNML__, _cd.__BNML__]])

        print "smpx mean  %.3f" % _N.mean(oo.smpx)

        print "^^^^STARTING VALUES"
        print "datafn=%s" % oo.datafn
        print "rn=%s" % str(oo.rn)
        print "us=%s" % str(oo.us)
        print "md=%s" % str(oo.model)
        print "****"

        #_plt.plot(oo.smpx)
        #######  PRIOR parameters
        #  F0  --  flat prior

        # ################# #generate initial values of parameters, time series
        oo._d = _kfardat.KFARGauObsDat(1, oo.N, 1, onetrial=True)
        oo._d.copyData(_N.empty(oo.N+1), _N.empty(oo.N+1), onetrial=True)   #  dummy data copied

        oo.x00 = 0
        oo.V00 = 0.5#oo.B_V00*_ss.invgamma.rvs(oo.a_V00)

        oo.smp_F        = _N.zeros(oo.NMC + oo.burn)
        oo.smp_zs       = _N.zeros((oo.NMC + oo.burn, oo.N+1), dtype=_N.int)
        oo.smp_rn       = _N.zeros((oo.NMC + oo.burn, oo.W), dtype=_N.int)
        oo.smp_u        = _N.zeros((oo.NMC + oo.burn, oo.W))
        oo.smp_dty      = _N.zeros((oo.NMC + oo.burn, oo.J), dtype=_N.int)
        oo.smp_q2       = _N.zeros(oo.NMC + oo.burn)

        oo.ws   = _N.ones((oo.N + 1, oo.W))*0.1   #  start at 0 + u
        oo.Bsmpx= _N.zeros((oo.burn+oo.NMC, oo.N + 1))


        # sample F0
        F0AA = _N.dot(oo.smpx[0:-1], oo.smpx[0:-1])
        F0BB = _N.dot(oo.smpx[0:-1], oo.smpx[1:])
        oo.q2= 0.01
        F0std= _N.sqrt(oo.q2/F0AA)
        F0a, F0b  = (oo.a_F0 - F0BB/F0AA) / F0std, (oo.b_F0 - F0BB/F0AA) / F0std
        oo.F0=F0BB/F0AA+F0std*_ss.truncnorm.rvs(F0a, F0b)

        #####################    sample q2
        a = oo.a_q2 + 0.5*(oo.N+1)  #  N + 1 - 1
        rsd_stp = oo.smpx[1:] - oo.F0*oo.smpx[0:-1]
        BB = oo.B_q2 + 0.5 * _N.dot(rsd_stp, rsd_stp)
        oo.q2 = _ss.invgamma.rvs(a, scale=BB)

    def gibbsSamp(self):  #########  GIBBS SAMPLER  ############
        #####  MCMC start
        oo   = self
        #  F, q2, u, rn, model

        rns = _N.empty((oo.N+1, oo.W), dtype=_N.int)  #  an rn for each trial
        oo.LN  = _N.empty((oo.N+1, oo.W, 1))  #  log of normlz constant
        oo.kp = _N.empty((oo.N+1, oo.W))

        cts     = _N.array(oo.y).reshape(oo.N+1, oo.W, 1)
        rn     = _N.array(oo.rn).reshape(1, oo.W, oo.J)

        cntMCMCiters = 80
        oo.mrns = _N.empty((oo.burn+oo.NMC, cntMCMCiters, oo.W), dtype=_N.int)
        oo.mus  = _N.empty((oo.burn+oo.NMC, cntMCMCiters, oo.W))
        oo.mdty = _N.empty((oo.burn+oo.NMC, cntMCMCiters, oo.W), dtype=_N.int)

        p   = _N.empty((oo.N+1, oo.W))
        lp1p= _N.empty((oo.N+1, oo.W))
        dlp1p= _N.empty((oo.N+1, oo.W))
        ppd = _N.empty((oo.N+1, oo.W))
        rnsy= _N.empty((oo.N+1, oo.W), dtype=_N.int)
        rnsyC= _N.empty(oo.N+1, dtype=_N.int)  #  for giving to devryoe
        usJ = _N.empty((oo.N+1, oo.W))

        if oo.smpxOff:
            print "setting smpx"
            oo.smpx[:] = oo.x
        susJC = _N.empty(oo.N+1)   #  for giving to devroye, can't handle slices
        wsTST = _N.empty(oo.N+1)   #  for giving to devroye, can't handle slices

        trms = _N.empty((oo.N+1, oo.J))

        wAw = _N.empty((oo.N+1, oo.W))
        wA  = _N.empty(oo.N+1)

        for it in xrange(oo.burn+oo.NMC):
            print "---   iter %d" % it

            ######  update distribution parameters
            smpxOffset = _N.mean(oo.smpx)
            #oo.smpx -= smpxOffset * (1+0.05*_N.random.randn())
            #print "smpxOffset  %f" % smpxOffset

            for w in xrange(oo.W):
                # if it > 150:
                #     if w == 0:
                #         oo.us[0] = -1.1
                #         oo.rn[0] = 70
                #         oo.model[0] = 1
                #     else:
                #         oo.us[w], oo.rn[w], oo.model[w] = cntmdlMCMCOnly(cntMCMCiters, oo.us[w], oo.rn[w], oo.model[w], oo.y[:, w], oo.mrns[it, :, w], oo.mus[it, :, w], oo.mdty[it, :, w], oo.smpx)
                # else:
                oo.us[w], oo.rn[w], oo.model[w] = cntmdlMCMCOnly(it, cntMCMCiters, oo.us[w], oo.rn[w], oo.model[w], oo.y[:, w], oo.mrns[it, :, w], oo.mus[it, :, w], oo.mdty[it, :, w], oo.smpx)

                if oo.model[w] == _cd.__NBML__:
                    oo.kp[:, w]   = (oo.y[:, w] - oo.rn[w]) *0.5
                    rnsy[:, w] = oo.rn[w] + oo.y[:, w]
                else:
                    oo.kp[:, w]   = oo.y[:, w] - oo.rn[w]*0.5
                    rnsy[:, w] = oo.rn[w]
                usJ[:, w]  = oo.us[w]
                wsTST[:]    = oo.ws[:, w]  # bug devryoe, can't be handed 2D arr
                susJC[:]    = oo.smpx + usJ[:, w]
                rnsyC[:]    = rnsy[:, w]

                lw.rpg_devroye(rnsyC, susJC, num=(oo.N + 1), out=wsTST)
                oo.ws[:, w] = wsTST

            if oo.W == 1:
                oo.smp_rn[it] = oo.rn
            else:
                oo.smp_rn[it] = oo.rn[:, 0]
            print oo.rn[:, 0]
            ###  now, adjust
            # print smpxOffset
            # print "----"
            #usJ        += smpxOffset

            #  wA  (M x 1)
            _N.product(1./oo.ws, axis=1, out=wA)
            wAr  = wA.reshape(oo.N+1, 1)

            wAw[:] = wAr

            # _N.mean(oo.ws, axis=0)    - not much diff between W=1,2
            for iw in xrange(oo.W):
                wAw[:, iw] /= 1./oo.ws[:, iw]
            #print _N.mean(wAw[:, 0])
            #print _N.mean(wAw[:, 1])
            #print "^^^^^^^^^^^"

            #  mean(oo._d.y[:]) is 30% of time positive for W=1
            #     ALL the time negative for W=2
            #  oo.kp/oo.ws  same for both wins
            #  sum(axis=1) is summing over the windows.  leaving vector N+1
            oo._d.y[:]             = (_N.sum((oo.kp/oo.ws - usJ)*wAw, axis=1) / _N.sum(wAw, axis=1))
            #off   = (0.9+0.1*_N.random.rand())*_N.mean(oo._d.y)
            #oo._d.y[:]             -= off
            #print _N.std(oo._d.y)
            #print _N.sum(wAw, axis=1)
            #print _N.mean(oo._d.y)

            #  sum(wAw, axis=1)/wA = 1/oo.ws  vector for 1 win
            #  
            ###  As it stands now, 2 identical windows increases the obs. nise
            #oo._d.Rv[:] = _N.sum(wAw, axis=1)/wA  # Rv is inverse variance
            oo._d.Rv[:] = wA/_N.sum(wAw, axis=1)  # Rv is inverse variance
            #oo._d.Rv /= (1-(2*off)/oo._d.Rv)
            #print _N.mean(oo._d.Rv)

            if not oo.smpxOff:
                #  p3 --  samp u here

                # sample F0

                F0AA = _N.dot(oo.smpx[0:-1], oo.smpx[0:-1])
                F0BB = _N.dot(oo.smpx[0:-1], oo.smpx[1:])
                # F0AA = _N.dot(smpx0[0:-1], smpx0[0:-1])
                # F0BB = _N.dot(smpx0[0:-1], smpx0[1:])

                F0std= _N.sqrt(oo.q2/F0AA)
                F0a, F0b  = (oo.a_F0 - F0BB/F0AA) / F0std, (oo.b_F0 - F0BB/F0AA) / F0std
                oo.F0=F0BB/F0AA+F0std*_ss.truncnorm.rvs(F0a, F0b)

                #   sample q2
                a = oo.a_q2 + 0.5*(oo.N+1)  #  N + 1 - 1
                rsd_stp = oo.smpx[1:] - oo.F0*oo.smpx[0:-1]
                BB = oo.B_q2 + 0.5 * _N.dot(rsd_stp, rsd_stp)
                oo.q2 = _ss.invgamma.rvs(a, scale=BB)

                #_N.savetxt("rnsy%d" % w, rnsy[:, w], fmt="%d")
                oo._d.copyParams(_N.array([oo.F0]), oo.q2, onetrial=True)

                #  generate latent AR state
                oo._d.f_x[0, 0, 0]     = 0#-0.1+0.1*_N.random.randn()
                oo._d.f_V[0, 0, 0]     = oo.V00

                oo.smpx = _kfar.armdl_FFBS_1itr(oo._d)
                #oo.smpx = oo.smpx - _N.mean(oo.smpx)
                print "smpx mean  %(1).3f  std %(2).3f    std (x) %(3).3f" % {"1" : _N.mean(oo.smpx), "2" : _N.std(oo.smpx), "3" : _N.std(oo.x)}
                oo.Bsmpx[it, :] = oo.smpx

                oo.smp_F[it]       = oo.F0
                oo.smp_q2[it]      = oo.q2
                if oo.W == 1:
                    oo.smp_u[it]      = oo.us
                else:
                    oo.smp_u[it]      = oo.us[:, 0]

    def run(self, env_dirname=None, datafn="cnt_data.dat", batch=False, usewin=None): ###########  RUN    
        """
        many datafiles in each directory
        """
        oo     = self    #  call self oo.  takes up less room on line
        oo.setname = None if batch else os.getcwd().split("/")[-1]

        oo.env_dirname=env_dirname
        oo.u_x00        = _N.zeros(1)
        oo.s2_x00       = 0.3
        oo.datafn   = datafn

        oo.loadDat(usewin=usewin)
        oo.initGibbs()
        t1    = _tm.time()
        oo.gibbsSamp()
        t2    = _tm.time()
        print (t2-t1)


    def getZs(self, lowst=0):
        oo = self
        occ = _N.mean(oo.smp_zs[oo.burn:, :, lowst], axis=0)
        ms  = _N.mean(oo.smp_m[oo.burn:], axis=0)
        li  = _N.where(occ < ms[lowst])

        zFt = _N.zeros(oo.N+1, dtype=_N.int)
        zFt[li] = 1

        zTr = _N.zeros(oo.N+1, dtype=_N.int)
        li  = _N.where(oo.st == 1)[0]
        zTr[li] = 1

        mtch = _N.where(zFt == zTr)[0]
        return zTr, zFt, (float(len(mtch))/(oo.N+1))


