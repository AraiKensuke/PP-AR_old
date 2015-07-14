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
from cntUtil import Llklhds, cntmdlMCMCOnly, startingValues

class mcmcARcnt(mAR.mcmcAR):
    ###  prior on F0 truncated normal
    F0        = None;
    a_F0      = -1;    b_F0      =  1

    ###
    mrns      = None;     mus    = None;    mdty    = None

    smp_rn    = None;     smp_u  = None;    smp_dty = None

    ###################################   RUN   #########
    def loadDat(self):
        oo     = self    #  call self oo.  takes up less room on line
        #  READ parameters of generative model from file
        #  contains N, k, singleFreqAR, u, beta, dt, stNz
        x_st_cnts = _N.loadtxt(resFN("cnt_data.dat", dir=oo.setname, env_dirname=oo.env_dirname))

        oo.N   = x_st_cnts.shape[0] - 1
        if oo.t1 == None:
            oo.t1 = oo.N + 1
        oo.N   = oo.t1 - oo.t0 - 1
        
        cols= x_st_cnts.shape[1]
        oo.xT  = x_st_cnts[oo.t0:oo.t1, 0]
        #######  Initialize z
        mH  = x_st_cnts[oo.t0:oo.t1, 1]
        ctCol = 2   #  column containing counts

        oo.y   = x_st_cnts[oo.t0:oo.t1, ctCol]
        oo.x   = x_st_cnts[oo.t0:oo.t1, 0]

    #  when we don't have xn
    def initGibbs(self):
        oo     = self    #  call self oo.  takes up less room on line
        #  INITIAL samples

        #oo.smpx = _N.empty(oo.N+1)
        oo.smpx = _N.array(oo.x)
        oo.us, oo.rn, oo.model = startingValues(oo.y, fillsmpx=oo.smpx)
        print "^^^^STARTING VALUES"
        print "rn=%d" % oo.rn
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


    def gibbsSamp(self):  #########  GIBBS SAMPLER  ############
        #####  MCMC start
        oo   = self
        #  F, q2, u, rn, model

        print oo.model
        print oo.rn
        print oo.us
        oo.kp   = (oo.y - oo.rn) *0.5 if oo.model==_cd.__NBML__ else oo.y - oo.rn*0.5

        cntMCMCiters = 30
        oo.mrns = _N.empty(cntMCMCiters, dtype=_N.int)
        oo.mus  = _N.empty(cntMCMCiters)
        oo.mdty = _N.empty(cntMCMCiters, dtype=_N.int)

        for it in xrange(oo.burn+oo.NMC):
            print "iter %d" % it

            oo.us, oo.rn, oo.model = cntmdlMCMCOnly(cntMCMCiters, oo.us, oo.rn, oo.model, oo.y, oo.mrns, oo.mus, oo.mdty, oo.smpx)
            ##  IMPORTANT.  update oo.kp right after getting new BN/NB params
            oo.kp   = (oo.y - oo.rn) *0.5 if oo.model==_cd.__NBML__ else oo.y - oo.rn*0.5

            if oo.model == _cd.__NBML__:
                lw.rpg_devroye(oo.rn+oo.y, oo.smpx + oo.us, num=(oo.N + 1), out=oo.ws)
            else:
                lw.rpg_devroye(oo.rn, oo.smpx + oo.us, num=(oo.N + 1), out=oo.ws)

            oo._d.copyParams(_N.array([oo.F0]), oo.q2, onetrial=True)

            #  generate latent AR state
            oo._d.f_x[0, 0, 0]     = (0.5*(1+_N.random.randn()))*oo.smpx[0]
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

            oo.smp_F[it]       = oo.F0
            oo.smp_q2[it]      = oo.q2
            oo.Bsmpx[it, :] = oo.smpx

            oo.smp_u[it]       = oo.us
            oo.smp_rn[it]      = oo.rn
            oo.smp_dty[it]     = oo.model

    
    def run(self, env_dirname=None): ###########  RUN
        oo     = self    #  call self oo.  takes up less room on line
        oo.setname = os.getcwd().split("/")[-1]
        oo.env_dirname=env_dirname
        oo.u_x00        = _N.zeros(1)
        oo.s2_x00       = 0.3

        oo.loadDat()
        oo.initGibbs()
        # t1    = _tm.time()
        # tmpNOAR = oo.noAR
        # oo.noAR = True
        # oo.gibbsSamp()
        # oo.noAR = tmpNOAR
        oo.gibbsSamp()
        # t2    = _tm.time()
        # print (t2-t1)

    #  MCMC 
