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
import cntUtil as _cU
import cntUtil_pyx as _cUpyx
import time as _tm
import pickle

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
    outSmplFN = "smpls.dump"

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
            oo.y = _N.empty((oo.t1-oo.t0, oo.W), dtype=_N.int)

            print oo.W
            for il in xrange(oo.W):
                print type(usewin[il])
                if type(usewin[il]) == _N.ndarray:
                    oo.y[:, il] = _N.sum(x_st_cnts[oo.t0:oo.t1, usewin[il]+2], axis=1)
                else:
                    oo.y[:, il] = x_st_cnts[oo.t0:oo.t1, usewin[il]+2]

        else:
            oo.W = nWinsInData
            oo.y   = _N.array(x_st_cnts[oo.t0:oo.t1, ctCol:ctCol+oo.W], dtype=_N.int)

        oo.xT  = x_st_cnts[oo.t0:oo.t1, 0]
        #######  Initialize z
        mH  = x_st_cnts[oo.t0:oo.t1, 1]
        stCol = 1   #  column containing counts
        ctCol = 2   #  column containing counts


        oo.st  = _N.array(x_st_cnts[oo.t0:oo.t1, stCol], dtype=_N.int)
        oo.x   = x_st_cnts[oo.t0:oo.t1, 0]
        oo.zs  = _N.zeros((oo.N+1, oo.J), dtype=_N.int)
        oo.m   = _N.empty(oo.J)

    #  when we don't have xn
    def initGibbs(self, logfact):
        oo     = self    #  call self oo.  takes up less room on line
        _cUpyx._init(logfact)
        #  INITIAL samples

        oo.smpx = _N.zeros(oo.N+1)
        oo.us, oo.rn, oo.model = _cU.startingValuesMw(oo.y, oo.J, oo.zs, fillsmpx=oo.smpx, indLH=True)
        # oo.us = _N.array([[-1.734, -1.734],
        #                   [-1.734, -1.734]])
        # oo.rn = _N.array([[130, 250],
        #                   [130, 250]])
        # oo.model = _N.array([[_cd.__BNML__, _cd.__BNML__],
        #                      [_cd.__BNML__, _cd.__BNML__]])

        print "smpx mean  %.3f" % _N.mean(oo.smpx)
        _N.mean(oo.zs, axis=0, out=oo.m)
        print "initial  m  %s" % str(oo.m)

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
        oo.smp_zs       = _N.zeros((oo.NMC + oo.burn, oo.N+1, oo.J), dtype=_N.bool)
        oo.smp_rn       = _N.zeros((oo.NMC + oo.burn, oo.W, oo.J), dtype=_N.int16)
        oo.smp_u        = _N.zeros((oo.NMC + oo.burn, oo.W, oo.J))
        oo.smp_dty      = _N.zeros((oo.NMC + oo.burn, oo.W, oo.J), dtype=_N.int16)
        oo.smp_q2       = _N.zeros(oo.NMC + oo.burn)
        oo.smp_m        = _N.zeros((oo.NMC + oo.burn, oo.J))

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

        oo.alp       = _N.array([1, 1])

    def gibbsSamp(self, logfact, cntMCMCiters=50):  #########  GIBBS SAMPLER  ############
        #####  MCMC start
        oo   = self
        #  F, q2, u, rn, model

        rns = _N.empty((oo.N+1, oo.W), dtype=_N.int)  #  an rn for each trial
        oo.LN  = _N.empty((oo.N+1, oo.W, oo.J))  #  log of normlz constant
        oo.LN_old  = _N.empty((oo.N+1, oo.W, oo.J))  #  log of normlz constant
        oo.kp = _N.empty((oo.N+1, oo.W))

        cts     = _N.array(oo.y).reshape(oo.N+1, oo.W, 1)
        rn     = _N.array(oo.rn).reshape(1, oo.W, oo.J)

        soox   = _N.std(oo.x)
        oo.mrns = _N.empty((oo.burn+oo.NMC, cntMCMCiters, oo.W, oo.J), dtype=_N.int)
        oo.mus  = _N.empty((oo.burn+oo.NMC, cntMCMCiters, oo.W))
        oo.mdty = _N.empty((oo.burn+oo.NMC, cntMCMCiters, oo.W), dtype=_N.int)

        p   = _N.empty((oo.N+1, oo.W, oo.J))
        lp1p= _N.empty((oo.N+1, oo.W, oo.J))
        dlp1p= _N.empty((oo.N+1, oo.W))
        ppd = _N.empty((oo.N+1, oo.W))
        rnsy= _N.empty((oo.N+1, oo.W), dtype=_N.int)
        rnsyC= _N.empty(oo.N+1, dtype=_N.int)  #  for giving to devryoe
        usJ = _N.empty((oo.N+1, oo.W))

        if oo.smpxOff:
            oo.smpx[:] = 0
        susJC = _N.empty(oo.N+1)   #  for giving to devroye, can't handle slices
        wsTST = _N.empty(oo.N+1)   #  for giving to devroye, can't handle slices

        trms = _N.empty((oo.N+1, oo.J))
        rats  = _N.empty((oo.N+1, oo.J))
        crats= _N.zeros((oo.N+1, oo.J+1))  # cumulative rat
        dirArgs = _N.empty(oo.J, dtype=_N.int)

        wAw = _N.empty((oo.N+1, oo.W))
        wA  = _N.empty(oo.N+1)

        usr   = oo.us.reshape((1, oo.W, oo.J))
        rsmpx = oo.smpx.reshape((oo.N+1, 1, 1))
        bads  = _N.empty(oo.N+1, dtype=_N.int)

        for it in xrange(oo.burn+oo.NMC):
            #dbtt1 = _tm.time()
            if (it % 10) == 0:
                print it

            ########  Allocate into Binary L,H states

            _N.divide(1, 1 + _N.exp(-(usr + rsmpx)), out=p)

            # for w in xrange(oo.W):
            #     for j in xrange(oo.J):
            #         p[:, w, j] = 1 / (1 + _N.exp(-(oo.us[w, j] + oo.smpx)))
            rands= _N.random.rand(oo.N+1)

            z = _N.zeros(oo.J, dtype=_N.int)
            zrs = _N.where(oo.m == 0)[0]
            #
            #  win1  (st1 st2)   win2      if 
            #
            #dbtt2 = _tm.time()
            ######  sTochastic allocation
            if len(zrs) == 0:
                bads[:] = -1   # for all trials n
                for w in xrange(oo.W):
                    #CtDistLogNorm2(oo.model[w], oo.J, oo.y[:, w], oo.rn[w], oo.LN_old[:, w])

                    #  if we have 20 cts in a trial, and state we want to 
                    #  switch to has a binomial with n = 19, 20 can't be generated.  On a window-by
                    #  from such a disribution
                    for j in xrange(oo.J):  #  for ratio of this state
                        if oo.model[w, j] == _cd.__BNML__:
                            oo.LN[:, w, j] = logfact[oo.rn[w,j]] - logfact[oo.y[:,w]] - logfact[oo.rn[w,j]-oo.y[:,w]]
                            bads[_N.where(oo.LN[:, w, j] < 0)[0]] = j

                            lp1p[:, w, j]  = oo.y[:, w]*_N.log(p[:, w, j]) + (oo.rn[w, j] - oo.y[:, w]) * _N.log(1 - p[:, w, j])
                        else:
                            oo.LN[:, w, j] = logfact[oo.y[:,w]+oo.rn[w,j]-1] - logfact[oo.y[:,w]] - logfact[oo.rn[w,j]-1]
                            lp1p[:, w, j] = oo.y[:, w]*_N.log(p[:, w, j]) + oo.rn[w, j] * _N.log(1 - p[:, w, j])
                
                canBgenrtd  = _N.where(bads < 0)[0]  #  ct can b generated
                cantBgenrtd = _N.where(bads >= 0)[0]  #  ct can't b generated
                #print len(cantBgenrtd)
                #print len(cangenrtd)
                            
                for j in xrange(oo.J):  #  for ratio of this state
                    for jo in xrange(oo.J):
                        dlp1p[canBgenrtd] = _N.subtract(lp1p[canBgenrtd, :, jo], lp1p[canBgenrtd, :, j])
                        #_N.subtract(lp1p[:, :, jo], lp1p[:, :, j], out=dlp1p)
                        ppd[canBgenrtd] = _N.exp(dlp1p[canBgenrtd])  # p1p*p1p may be small
                        trms[canBgenrtd, jo] = _N.exp(_N.sum(oo.LN[canBgenrtd, :, jo], axis=1) - _N.sum(oo.LN[canBgenrtd, :, j], axis=1)) * ((oo.m[jo]/oo.m[j]) * _N.product(ppd[canBgenrtd], axis=1))

                    rats[canBgenrtd, j] = 1 / _N.sum(trms[canBgenrtd], axis=1)

                    for j in xrange(1, oo.J):
                        rats[canBgenrtd, j] += rats[canBgenrtd, j-1]

                crats[canBgenrtd, 1:] = rats[canBgenrtd]    #  [0.3, 0.7]  --> [0, 0.3, 1] windows
                if oo.J > 1:
                    #  do regular assigning to zs
                    rs = _N.random.rand(len(canBgenrtd)).reshape(len(canBgenrtd), 1)
                    #  we need to do something about p1p.  Log it.
                    x, y = _N.where((crats[canBgenrtd, 1:] >= rs) & (crats[canBgenrtd, 0:-1] <= rs))
                    #  x is [0, N+1].  
                    oo.zs[canBgenrtd[x], y] = 1;     oo.zs[canBgenrtd[x], 1-y] = 0
                    if len(cantBgenrtd) > 0:
                        oo.zs[cantBgenrtd, bads[cantBgenrtd]] = 0;     
                        oo.zs[cantBgenrtd, 1-bads[cantBgenrtd]] = 1                      
                else:
                    oo.zs[:, 0] = 1;     

                #  in cases where BNML, cts must at most be rn-1.  
                #  Find those trials that can't be from proposed binomial
                for w in xrange(oo.W):
                    for j in xrange(oo.J):
                        if (oo.model[w, j] == _cd.__BNML__):
                            notBNML =  _N.where(oo.rn[w, j] <= oo.y[:, w])[0]
                            if len(notBNML) > 0:
                                oo.zs[notBNML, j] = 0
                                oo.zs[notBNML, 1-j] = 1
            else:  #  binary state weights are not [1, 0] or [0, 1]
                oo.zs[:, 1 - zrs[0]] = 1
                oo.zs[:, zrs[0]] = 0
                print "hit zero"
            oo.smp_zs[it] = oo.zs

            #dbtt3 = _tm.time()
            ########  Now sample the other parameters
            if oo.J > 1:   #  lht[j]   the trials with lo-hi state == j
                lht = [_N.where(oo.zs[:, 0] == 1)[0], _N.where(oo.zs[:, 1] == 1)[0]]
            else:
                lht = [_N.where(oo.zs[:, 0] == 1)[0]]

            for w in xrange(oo.W):
                for j in xrange(oo.J):
                    oo.us[w, j], oo.rn[w, j], oo.model[w, j] = _cUpyx.cntmdlMCMCOnly(it, cntMCMCiters, oo.us[w, j], oo.rn[w, j], oo.model[w, j], oo.y[lht[j], w], oo.mrns[it, :, w], oo.mus[it, :, w], oo.mdty[it, :, w], oo.smpx[lht[j]])

            _N.add(oo.alp, _N.sum(oo.zs, axis=0), out=dirArgs)
            oo.m[:] = _N.random.dirichlet(dirArgs)
            oo.smp_m[it] = oo.m
            oo.smp_rn[it] = oo.rn
            oo.smp_dty[it] = oo.model

            #dbtt4 = _tm.time()
            ### offset
            for j in xrange(oo.J):
                trls = lht[j]
                for w in xrange(oo.W):
                    #rns[trls, w] = oo.rn[w, j]    
                    oo.kp[trls, w]   = (oo.y[trls, w] - oo.rn[w, j]) *0.5 if oo.model[w, j]==_cd.__NBML__ else oo.y[trls, w] - oo.rn[w, j]*0.5
                
                    if oo.model[w, j] == _cd.__NBML__:
                        rnsy[trls, w] = oo.rn[w, j] + oo.y[trls, w]
                    else:
                        rnsy[trls, w] = oo.rn[w, j]
                    usJ[trls, w]  = oo.us[w, j]

            #dbtt5 = _tm.time()
            ### offset
            ###  PG variables
            for w in xrange(oo.W):
                wsTST[:]    = oo.ws[:, w]
                susJC[:]    = oo.smpx + usJ[:, w]
                rnsyC[:]    = rnsy[:, w]

                lw.rpg_devroye(rnsyC, susJC, num=(oo.N + 1), out=wsTST)
                oo.ws[:, w] = wsTST

            #  wA  (M x 1)
            _N.product(1./oo.ws, axis=1, out=wA)
            wAr  = wA.reshape(oo.N+1, 1)

            wAw[:] = wAr

            for iw in xrange(oo.W):
                #wAw[:, iw] /= 1./oo.ws[:, iw]
                wAw[:, iw] *= oo.ws[:, iw]
            
            oo._d.y[:]             = (_N.sum((oo.kp/oo.ws - usJ)*wAw, axis=1) / _N.sum(wAw, axis=1))

            ###  As it stands now, 2 identical windows increases the obs. nise
            oo._d.Rv[:] = wA/_N.sum(wAw, axis=1)  # Rv is inverse variance
            #oo._d.Rv[:] = _N.sum(wAw, axis=1)/wA  # Rv is inverse variance

            #dbtt6 = _tm.time()
            ### offset
            if not oo.smpxOff:
                #  p3 --  samp u here

                # sample F0

                F0AA = _N.dot(oo.smpx[0:-1], oo.smpx[0:-1])
                F0BB = _N.dot(oo.smpx[0:-1], oo.smpx[1:])

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
                oo._d.f_x[0, 0, 0]     = oo.x00
                oo._d.f_V[0, 0, 0]     = oo.V00

                oo.smpx = _kfar.armdl_FFBS_1itr(oo._d)
                oo.Bsmpx[it, :] = oo.smpx

                oo.smp_F[it]       = oo.F0
                oo.smp_q2[it]      = oo.q2
                oo.smp_u[it]      = oo.us
            #dbtt7 = _tm.time()
            # print "#timing start"
            # print "nt+= 1"
            # print "t2t1+=%.4e" % (#dbtt2-#dbtt1)
            # print "t3t2+=%.4e" % (#dbtt3-#dbtt2)
            # print "t4t3+=%.4e" % (#dbtt4-#dbtt3)
            # print "t5t4+=%.4e" % (#dbtt5-#dbtt4)
            # print "t6t5+=%.4e" % (#dbtt6-#dbtt5)
            # print "t7t6+=%.4e" % (#dbtt7-#dbtt6)
            # print "#timing end"

            ### offset


    def run(self, logfact, env_dirname=None, datafn="cnt_data.dat", batch=False, usewin=None, cntMCMCiters=50): ###########  RUN    
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
        oo.initGibbs(logfact)
        t1    = _tm.time()
        oo.gibbsSamp(logfact, cntMCMCiters=cntMCMCiters)
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

    def dump_smps(self, dir=None):
        oo    = self
        pcklme = {}

        pcklme["F"]    = oo.smp_F
        pcklme["q2"]   = oo.smp_q2
        pcklme["u"]    = oo.smp_u
        pcklme["m"]    = oo.smp_m
        pcklme["dty"]  = oo.smp_dty

        pcklme["Bsmpx"]    = oo.Bsmpx
        pcklme["zs"]   = oo.smp_zs

        if dir is None:
            dmp = open(oo.outSmplFN, "wb")
        else:
            dmp = open("%(d)s/%(sfn)s" % {"d" : dir, "sfn" : oo.outSmplFN}, "wb")
        pickle.dump(pcklme, dmp, -1)
        dmp.close()

        # import pickle
        # with open("smpls.dump", "rb") as f:
        # lm = pickle.load(f)
