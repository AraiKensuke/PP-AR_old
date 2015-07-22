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
from cntUtil import Llklhds, cntmdlMCMCOnly, startingValues, startingValuesM, CtDistLogNorm2
import time as _tm

class mcmcARcntM(mAR.mcmcAR):
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

    ###################################   RUN   #########
    def loadDat(self):
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
        oo.xT  = x_st_cnts[oo.t0:oo.t1, 0]
        #######  Initialize z
        mH  = x_st_cnts[oo.t0:oo.t1, 1]
        stCol = 1   #  column containing counts
        ctCol = 2   #  column containing counts

        oo.y   = x_st_cnts[oo.t0:oo.t1, ctCol]
        oo.st  = x_st_cnts[oo.t0:oo.t1, stCol]
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
        oo.us, oo.rn, oo.model = startingValuesM(oo.y, oo.J, oo.zs, fillsmpx=oo.smpx, indLH=True)
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

        oo.x00 = oo.smpx[0]
        oo.V00 = 0.5#oo.B_V00*_ss.invgamma.rvs(oo.a_V00)

        oo.smp_F        = _N.zeros(oo.NMC + oo.burn)
        oo.smp_zs       = _N.zeros((oo.NMC + oo.burn, oo.N+1, oo.J), dtype=_N.int)
        oo.smp_rnM       = _N.zeros((oo.NMC + oo.burn, oo.J), dtype=_N.int)
        oo.smp_u        = _N.zeros((oo.NMC + oo.burn, oo.J))
        oo.smp_dty      = _N.zeros((oo.NMC + oo.burn, oo.J), dtype=_N.int)
        oo.smp_q2       = _N.zeros(oo.NMC + oo.burn)
        oo.smp_m        = _N.zeros((oo.NMC + oo.burn, oo.J))

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

        rns = _N.empty(oo.N+1, dtype=_N.int)  #  an rn for each trial
        oo.LN  = _N.empty((oo.N+1, oo.J))  #  an rn for each trial
        lht = [_N.where(oo.zs[:, 0] == 1)[0], _N.where(oo.zs[:, 1] == 1)[0]]
        oo.kp = _N.empty(oo.N+1)
        for j in xrange(oo.J):
            trls = lht[j]
            rns[trls] = oo.rn[j]
            oo.kp[trls]   = (oo.y[trls] - oo.rn[j]) *0.5 if oo.model[j]==_cd.__NBML__ else oo.y[trls] - oo.rn[j]*0.5

        cts     = _N.array(oo.y).reshape(oo.N+1, 1)
        rnM     = _N.array(oo.rn).reshape(1, oo.J)

        cntMCMCiters = 20
        oo.mrns = _N.empty((oo.burn+oo.NMC, cntMCMCiters, oo.J), dtype=_N.int)
        oo.mus  = _N.empty((oo.burn+oo.NMC, cntMCMCiters))
        oo.mdty = _N.empty((oo.burn+oo.NMC, cntMCMCiters), dtype=_N.int)

        los = _N.where(oo.st == 0)[0]
        oo.zs[los, 0] = 1;         oo.zs[los, 1] = 0; 
        his = _N.where(oo.st == 1)[0]
        oo.zs[his, 0] = 0;         oo.zs[his, 1] = 1; 

        p   = _N.empty((oo.N+1, oo.J))
        p1p = _N.empty((oo.N+1, oo.J))
        rnsy= _N.empty(oo.N+1, dtype=_N.int)
        usJ = _N.empty(oo.N+1)
        if oo.smpxOff:
            oo.smpx[:] = 0

        trms = _N.empty((oo.N+1, oo.J))
        rats  = _N.empty((oo.N+1, oo.J))
        crats= _N.zeros((oo.N+1, oo.J+1))  # cumulative rat

        for it in xrange(oo.burn+oo.NMC):
            print "iter %d" % it

            for j in xrange(oo.J):
                if len(lht[j]) > 0:  #  
                    oo.us[j], oo.rn[j], oo.model[j] = cntmdlMCMCOnly(cntMCMCiters, oo.us[j], oo.rn[j], oo.model[j], oo.y[lht[j]], oo.mrns[it], oo.mus[it], oo.mdty[it], oo.smpx[lht[j]])

                p[:, j] = 1 / (1 + _N.exp(-(oo.us[j] + oo.smpx)))        

            oo.smp_dty[it] = oo.model
            oo.smp_rnM[it] = oo.rn
            oo.smp_u[it]   = oo.us
            rands= _N.random.rand(oo.N+1)

            z = _N.zeros(oo.J, dtype=_N.int)
            zrs = _N.where(oo.m == 0)[0]
            
            if len(zrs) == 0:
                CtDistLogNorm2(oo.model, oo.J, oo.y, oo.rn, oo.LN)
                for j in xrange(oo.J):  #  for ratio of this state
                    if oo.model[j] == _cd.__BNML__:
                        p1p[:, j] = p[:, j]**oo.y * (1 - p[:, j])**(oo.rn[j] - oo.y)
                    else:
                        p1p[:, j] = p[:, j]**oo.y * (1 - p[:, j])**oo.rn[j]

                for j in xrange(oo.J):  #  for ratio of this state
                    for jo in xrange(oo.J):
                        trms[:, jo] = _N.exp(oo.LN[:, jo] - oo.LN[:, j]) * ((oo.m[jo] * p1p[:, jo])/(oo.m[j]* p1p[:, j]))

                    rats[:, j] = 1 / _N.sum(trms, axis=1)

                    for j in xrange(1, oo.J):
                        rats[:, j] += rats[:, j-1]

                crats[:, 1:] = rats    #  [0.3, 0.7]  --> [0, 0.3, 1] windows

                #  do regular assigning to zs
                rs = _N.random.rand(oo.N+1).reshape(oo.N+1, 1)
                x, y = _N.where((crats[:, 1:] >= rs) & (crats[:, 0:-1] <= rs))
                #  x is [0, N+1].  
                oo.zs[x, y] = 1;     oo.zs[x, 1-y] = 0

                #  in cases where BNML, cts must at most be rn-1.  
                #  Find those trials that can't be from proposed binomial
                for j in xrange(oo.J):
                    if (oo.model[j] == _cd.__BNML__):
                        notBNML =  _N.where(oo.rn[j] <= oo.y)[0]
                        if len(notBNML) > 0:
                            oo.zs[notBNML, j] = 0
                            oo.zs[notBNML, 1-j] = 1
                dirArgs = _N.empty(oo.J, dtype=_N.int)
            else:
                oo.zs[:, 1 - zrs[0]] = 1
                oo.zs[:, zrs[0]] = 0
                print "hit zero"
            oo.smp_zs[it] = oo.zs

            """
            for j in xrange(oo.J):
                trls = _N.where(oo.zs[:, j] == 1)[0]
                if len(trls) > 0:
                    u    = _N.mean(oo.y[trls])
                    s    = _N.std(oo.y[trls])
                    print "mean counts for st %(j)d   %(ct).1f   %(tr)d    %(cv).3f" % {"j" : j, "ct" : u, "tr" : len(trls), "cv" : ((s*s)/u)}
            """

            _N.add(oo.alp, _N.sum(oo.zs, axis=0), out=dirArgs)

            oo.m[:] = _N.random.dirichlet(dirArgs)
            oo.smp_m[it] = oo.m

            oo.Bsmpx[it, :] = oo.smpx

            print _N.mean(oo.zs, axis=0)

            lht = [_N.where(oo.zs[:, 0] == 1)[0], _N.where(oo.zs[:, 1] == 1)[0]]
            for j in xrange(oo.J):
                trls = lht[j]
                rns[trls] = oo.rn[j]
                oo.kp[trls]   = (oo.y[trls] - oo.rn[j]) *0.5 if oo.model[j]==_cd.__NBML__ else oo.y[trls] - oo.rn[j]*0.5
                
                if oo.model[j] == _cd.__NBML__:
                    rnsy[trls] = oo.rn[j] + oo.y[trls]
                else:
                    rnsy[trls] = oo.rn[j]
                usJ[trls]  = oo.us[j]

            lw.rpg_devroye(rnsy, oo.smpx + usJ, num=(oo.N + 1), out=oo.ws)
            oo._d.copyParams(_N.array([oo.F0]), oo.q2, onetrial=True)

            #  generate latent AR state
            oo._d.f_x[0, 0, 0]     = oo.x00
            oo._d.f_V[0, 0, 0]     = oo.V00
            oo._d.y[:]             = oo.kp/oo.ws - usJ
            oo._d.Rv[:] = 1 / oo.ws   #  time dependent noise

            if not oo.smpxOff:
                oo.smpx = _kfar.armdl_FFBS_1itr(oo._d)

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
                oo.smp_F[it]       = oo.F0
                oo.smp_q2[it]      = oo.q2


    def run(self, env_dirname=None, datafn="cnt_data.dat", batch=False): ###########  RUN    
        """
        many datafiles in each directory
        """
        oo     = self    #  call self oo.  takes up less room on line
        oo.setname = None if batch else os.getcwd().split("/")[-1]

        oo.env_dirname=env_dirname
        oo.u_x00        = _N.zeros(1)
        oo.s2_x00       = 0.3
        oo.datafn   = datafn

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


