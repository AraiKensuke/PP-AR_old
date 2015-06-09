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
from cntUtil import Llklhds, cntmdlMCMCOnlyM, startingValues, startingValuesM, CtDistLogNorm
import time as _tm

class mcmcARcntM(mAR.mcmcAR):
    ###  prior on F0 truncated normal
    alp       = None
    F0        = None;
    a_F0      = -1;    b_F0      =  1

    ###
    mrns      = None;     mus    = None;    mdty    = None
    smp_rn    = None;     smp_u  = None;    smp_dty = None

    m             = None;     zs = None

    ###################################   RUN   #########
    def loadDat(self):
        oo     = self    #  call self oo.  takes up less room on line
        #  READ parameters of generative model from file
        #  contains N, k, singleFreqAR, u, beta, dt, stNz
        x_st_cnts = _N.loadtxt(resFN("cnt_data.dat", dir=oo.setname))

        oo.N   = x_st_cnts.shape[0] - 1
        if oo.t1 == None:
            oo.t1 = oo.N + 1
        oo.N   = oo.t1 - oo.t0 - 1
        
        cols= x_st_cnts.shape[1]
        nWinsInData = cols - 2   #  latent x, true state, win 1, win 2, ...
        oo.xT  = x_st_cnts[oo.t0:oo.t1, 0]
        #######  Initialize z
        mH  = x_st_cnts[oo.t0:oo.t1, 1]
        stCol = 1   #  column containing counts
        ctCol = 2   #  column containing counts

        oo.y   = x_st_cnts[oo.t0:oo.t1, ctCol]
        oo.x   = x_st_cnts[oo.t0:oo.t1, 0]
        oo.zs  = _N.zeros((oo.N+1, oo.J), dtype=_N.int)
        oo.m   = _N.empty(oo.J)

        if nWinsInData == 1:
            oo.nWins = 1
            oo.y   = x_st_cnts[oo.t0:oo.t1, ctCol]

    #  when we don't have xn
    def initGibbs(self):
        oo     = self    #  call self oo.  takes up less room on line
        #  INITIAL samples

        oo.smpx = _N.empty(oo.N+1)
        oo.us, oo.rnM, oo.model = startingValuesM(oo.y, oo.J, oo.zs, fillsmpx=oo.smpx)
        _N.mean(oo.zs, axis=0, out=oo.m)

        print "^^^^STARTING VALUES"
        print "rn=%s" % str(oo.rnM)
        print "us=%.3f" % oo.us
        print "md=%d" % oo.model
        print "****"

        #_plt.plot(oo.smpx)
        #######  PRIOR parameters
        #  F0  --  flat prior

        # ################# #generate initial values of parameters, time series
        oo._d = _kfardat.KFARGauObsDat(1, oo.N, 1, onetrial=True)
        oo._d.copyData(_N.empty(oo.N+1), _N.empty(oo.N+1), onetrial=True)   #  dummy data copied

        oo.x00 = oo.smpx[0]
        oo.V00 = 0.5#oo.B_V00*_ss.invgamma.rvs(oo.a_V00)

        oo.smp_F        = _N.zeros(oo.NMC + oo.burn)
        oo.smp_rn       = _N.zeros(oo.NMC + oo.burn, dtype=_N.int)
        oo.smp_u        = _N.zeros(oo.NMC + oo.burn)
        oo.smp_dty      = _N.zeros(oo.NMC + oo.burn, dtype=_N.int)
        oo.smp_q2       = _N.zeros(oo.NMC + oo.burn)

        oo.ws   = _N.ones(oo.N + 1)*0.1   #  start at 0 + u
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

    def gibbsSamp(self):  #########  GIBBS SAMPLER  ############
        #####  MCMC start
        oo   = self
        #  F, q2, u, rn, model

        mL      = 20000

        print oo.model
        print oo.us

        rns = _N.empty(oo.N+1, dtype=_N.int)  #  an rn for each trial
        LN  = _N.empty((oo.N+1, oo.J))  #  an rn for each trial
        for j in xrange(oo.J):
            trls = _N.where(oo.zs[:, j] == 1)[0]
            rns[trls] = oo.rnM[j]
        oo.kp   = (oo.y - rns) *0.5 if oo.model==_cd.__NBML__ else oo.y - rns*0.5

        print rns
        cts     = _N.array(oo.y).reshape(oo.N+1, 1)
        rnM     = _N.array(oo.rnM).reshape(1, oo.J)


        cntMCMCiters = 80
        oo.mrns = _N.empty((cntMCMCiters, oo.J), dtype=_N.int)
        oo.mus  = _N.empty(cntMCMCiters)
        oo.mdty = _N.empty(cntMCMCiters, dtype=_N.int)

        for it in xrange(oo.burn+oo.NMC):
            print "iter %d" % it

            oo.us, oo.rnM, oo.model = cntmdlMCMCOnlyM(cntMCMCiters, oo.J, oo.zs, oo.us, oo.rnM, oo.model, oo.y, oo.mrns, oo.mus, oo.mdty, oo.smpx)

            CtDistLogNorm(_cd.__NBML__, cts, rnM, LN)

            p = 1 / (1 + _N.exp(-(oo.us + oo.smpx)))
            m1p = 1 - p
            trms = _N.empty(oo.J)
            rands= _N.random.rand(oo.N+1)
            crats= _N.zeros(oo.J+1)
            rats  = _N.empty(oo.J)

            z = _N.zeros(oo.J, dtype=_N.int)
            zrs = _N.where(oo.m == 0)[0]

            if len(zrs) == 0:
                for m in xrange(oo.N+1):
                    #  I want the fraction for the mth 

                    for j in xrange(oo.J):  #  for ratio of this state
                        for jo in xrange(oo.J):
                            trms[jo] = _N.exp(LN[m, jo] - LN[m, j]) * ((oo.m[jo] * (1 - p[m])**oo.rnM[jo])/(oo.m[j]* (1 - p[m])**oo.rnM[j]))
                        rats[j] = 1 / _N.sum(trms)
                    rats /= _N.sum(rats)

                    for j in xrange(1, oo.J):
                        rats[j] += rats[j-1]
                    crats[1:] = rats    #  [0.3, 0.7]  --> [0, 0.3, 1]  windows

                    st = _N.where((rands[m] >= crats[0:-1]) &\
                                  (rands[m] <= crats[1:]))[0]
                    z[:]  = 0
                    z[st] = 1
                    oo.zs[m] = z
                dirArgs = _N.empty(oo.J, dtype=_N.int)
            else:
                oo.zs[:, 1 - zrs[0]] = 1
                oo.zs[:, zrs[0]] = 0
                print "hit zero"

            _N.add(oo.alp, _N.sum(oo.zs, axis=0), out=dirArgs)
            oo.m[:] = _N.random.dirichlet(dirArgs)

            oo.smp_F[it]       = oo.F0
            oo.smp_q2[it]      = oo.q2
            oo.Bsmpx[it, :] = oo.smpx

            oo.smp_u[it]       = oo.us
            #oo.smp_rnM[it]      = oo.rnM
            oo.smp_dty[it]     = oo.model
            print _N.mean(oo.zs, axis=0)

            for j in xrange(oo.J):
                trls = _N.where(oo.zs[:, j] == 1)[0]
                rns[trls] = oo.rnM[j]
            oo.kp   = (oo.y - rns) *0.5 if oo.model==_cd.__NBML__ else oo.y - rns*0.5

            if oo.model == _cd.__NBML__:
                lw.rpg_devroye(rns+oo.y, oo.smpx + oo.us, num=(oo.N + 1), out=oo.ws)
            else:
                lw.rpg_devroye(rns, oo.smpx + oo.us, num=(oo.N + 1), out=oo.ws)

            oo._d.copyParams(_N.array([oo.F0]), oo.q2, onetrial=True)

            #  generate latent AR state
            oo._d.f_x[0, 0, 0]     = oo.x00
            oo._d.f_V[0, 0, 0]     = oo.V00
            oo._d.y[:]             = oo.kp/oo.ws - oo.us
            oo._d.Rv[:] = 1 / oo.ws   #  time dependent noise

            #_plt.plot(oo.smpx)
            oo.smpx = _kfar.armdl_FFBS_1itr(oo._d)
            #_plt.plot(oo.smpx)

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
            # # #####################    sample x00
            # mn  = (oo.u_x00*oo.V00 + oo.s2_x00*oo.x00) / (oo.V00 + oo.s2_x00)
            # vr = (oo.V00*oo.s2_x00) / (oo.V00 + oo.s2_x00)
            # oo.x00 = mn + _N.sqrt(vr)*_N.random.randn()
            # #####################    sample V00
            # aa = oo.a_V00 + 0.5
            # BB = oo.B_V00 + 0.5*(oo.smpx[0] - oo.x00)*(oo.smpx[0] - oo.x00)
            # oo.V00 = _ss.invgamma.rvs(aa, scale=BB)

            #  Now do 
            #  p(z) = (m_1 x Lklhd(state 1))^z_1    x    (m_2 x Lklhd(state_2))^z_2  x ...
            #  lp(z) = log(m_1) + Llklhd(state_1) 
            #  total prob is lp(z_1) + lp(z_2) + lp(z_3)
            #  p(state 1) = exp(-(lp(1) - ltotalProb))
            #  p(state 2) = exp(-(lp(2) - ltotalProb))


    
    def run(self): ###########  RUN
        oo     = self    #  call self oo.  takes up less room on line
        oo.setname = os.getcwd().split("/")[-1]
        oo.u_x00        = _N.zeros(1)
        oo.s2_x00       = 0.3

        oo.loadDat()
        oo.initGibbs()
        # t1    = _tm.time()
        # tmpNOAR = oo.noAR
        # oo.noAR = True
        oo.gibbsSamp()
        # oo.noAR = tmpNOAR
        #oo.gibbsSamp()
        # t2    = _tm.time()
        # print (t2-t1)

    #  MCMC 
