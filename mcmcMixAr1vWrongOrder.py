import numpy as _N
import LogitWrapper as lw
import kflib
import scipy.stats as _ss
import scipy.misc as _sm

import kfardat as _kfardat
from   kassdirs import resFN, datFN
import kstat as _ks
import kfARlibPY as _kfar
import os
import warnings

from   utildirs import setFN, rename_self_vars
import pickle as _pkl

mvnrml    = _N.random.multivariate_normal

# m, z   (samples kept)
# u
class mcmcMixAr1v:
    #  Simulation params
    setname       = None
    rs            = -1
    burn          = None
    NMC           = None
    t0           = None
    t1           = None
    #  Description of model
    model         = None
    nWins         = None   #  set automatically
    oWin          = None   #  if data > 1 window, which win to use?  or sum
    nStates       = None;    states     = None    # we set nStates
    rn            = None   #  Also length (nStates)
    k             = None
    #  Sampled parameters
    Bsmpx         = None
    smp_u         = None
    smp_q2        = None
    smp_x00       = None
    #  Current values of parameters
    m             = None;     z = None
    smpx          = None
    u             = None; us_w1 = None; us_w2 = None
    ws            = None; ws_w1 = None; ws_w2 = None
    F0            = 0.9; q2    = 0.015
    #  store samples of
    allalfas      = None
    pgs           = None
    fs            = None
    amps          = None
    ##  

    _d            = None

    ####  These priors need to be set after nWins specified
    #  u   --  Gaussian prior
    u_u          = None;             s2_u         = None
    #  Dirichlet priors
    alp          = None

    #  q2  --  Inverse Gamma prior
    pr_mn_q2     = 0.05;    a_q2         = 2;  B_q2         = (a_q2 + 1) * pr_mn_q2
    #  initial state
    u_x00        = 0;             s2_x00       = 0.4
    pr_mn_V00    = 1;   a_V00        = 2;    B_V00        = (a_V00 + 1)*pr_mn_V00

    xT          = None;   zT     = None; #  Hidden trend, states
    y           = None   #  spike count observation 
    kp          = None;  kp_w1       = None; kp_w2     = None

    ###################################   RUN   #########
    def loadDat(self):
        oo     = self    #  call self oo.  takes up less room on line
        #  READ parameters of generative model from file
        #  contains N, k, singleFreqAR, u, beta, dt, stNz
        x_st_cnts = _N.loadtxt(resFN("cnt_data.dat", dir=oo.setname))

        oo.N   = x_st_cnts.shape[0] - 1
        if oo.t1 == None:
            oo.t1 = oo.N + 1
        cols= x_st_cnts.shape[1]
        nWinsInData = cols - 2   #  latent x, true state, win 1, win 2, ...
        oo.xT  = x_st_cnts[oo.t0:oo.t1, 0]
        #######  Initialize z
        mH  = x_st_cnts[oo.t0:oo.t1, 1]
        stCol= 1   #  column containing state number
        ctCol = 2   #  column containing counts
        oo.zT  = x_st_cnts[oo.t0:oo.t1, stCol]

        if nWinsInData == 1:
            oo.nWins = 1
            oo.y   = x_st_cnts[oo.t0:oo.t1, ctCol]   
        else:
            if len(oo.rn) == 2:  #  use both windows
                oo.y   = x_st_cnts[oo.t0:oo.t1, ctCol:] 
                oo.nWins = 2
            else:   #  use one of the windows, or a sum of the counts
                if oo.oWin == "sum":
                    oo.y   = _N.sum(x_st_cnts[oo.t0:oo.t1, ctCol:], axis=1)
                else:
                    oo.y   = x_st_cnts[oo.t0:oo.t1, oo.oWin + ctCol]
                oo.nWins = 1
        oo.xT   = x_st_cnts[oo.t0:oo.t1, 0]

    ###################################   RUN   #########
    def run(self, runDir=None):
        """
        RUN
        """
        oo     = self    #  call self oo.  takes up less room on line
        oo.setname = os.getcwd().split("/")[-1]

        oo.nWins    = len(oo.rn)   #  we don't set nWins manually

        if runDir == None:
            runDir="%(sn)s/AR%(k)d_[%(t0)d-%(t1)d]" % {"sn" : oo.setname, "ar" : k, "t0" : oo.t0, "t1" : oo.t1}

        if oo.rs >= 0:
            unpickle(runDir, oo.rs)
        else:   #  First run
            restarts = 0

        oo.loadDat()  # u is set initialized
        oo.initGibbs()
        oo.gibbsSamp()

    def initGibbs(self):
        oo     = self    #  call self oo.  takes up less room on line
        #  INITIAL samples
        if oo.nStates == 2:
            oo.states = _N.array([[1, 0], [0, 1]])
        else:
            oo.states = _N.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        if oo.nWins == 1:
            mnCt= _N.mean(oo.y)
        else:
            mnCt_w1= _N.mean(oo.y[oo.t0:oo.t1, 0])
            mnCt_w2= _N.mean(oo.y[oo.t0:oo.t1, 1])

        if oo.model=="negative binomial":
            if oo.nWins == 1:
                oo.kp   = (oo.y - oo.rn) *0.5
                p0   = mnCt / (mnCt + oo.rn)       #  matches 1 - p of genearted
            else:
                oo.kp_w1   = (y[oo.t0:oo.t1, 0] - oo.rn[0]) *0.5
                oo.kp_w2   = (y[oo.t0:oo.t1, 1] - oo.rn[1]) *0.5
                p0_w1   = mnCt_w1 / (mnCt_w1 + oo.rn[0])
                p0_w2   = mnCt_w2 / (mnCt_w2 + oo.rn[0])
        else:
            if oo.nWins == 1:
                oo.kp  = oo.y - oo.rn*0.5
                p0   = mnCt / float(oo.rn)       #  matches 1 - p of genearted
            else:
                oo.kp_w1  = oo.y[oo.t0:oo.t1, 0] - oo.rn[0]*0.5
                oo.kp_w2  = oo.y[oo.t0:oo.t1, 1] - oo.rn[1]*0.5
                p0_w1  = mnCt_w1 / float(oo.rn[0])       #  matches 1 - p of genearted
                p0_w2  = mnCt_w2 / float(oo.rn[1])       #  matches 1 - p of genearted
        #  gnerate approximate offset
        if oo.nWins == 1:
            u0  = _N.log(p0 / (1 - p0))    #  -1*u generated
        else:
            u0_w1  = _N.log(p0_w1 / (1 - p0_w1))    #  -1*u generated
            u0_w2  = _N.log(p0_w2 / (1 - p0_w2))    #  -1*u generated

        #######  PRIOR parameters
        #  F0  --  flat prior
        #a_F0         = -1
        #  I think a prior assumption of relatively narrow and high F0 range
        #  is warranted.  Small F0 is close to white noise, and as such can
        #  be confused with the independent trial-to-trial count noise.Force
        #  it to search for longer timescale correlations by setting F0 to be
        #  fairly large.
        oo.a_F0         = -1             #  prior assumption: slow fluctuation
        oo.b_F0         =  1
        #  u   --  Gaussian prior
        oo.u_u        = _N.empty(oo.nStates * oo.nWins)
        oo.s2_u       = _N.zeros((oo.nStates * oo.nWins, oo.nStates * oo.nWins))
        # (win1 s1)    (win1 s2)    (win2 s1)    (win2 s2)
        if oo.nWins == 1:
            if oo.nStates == 2:
                oo.u_u[:]     = (u0*1.2, u0*0.8)  #  mean of prior for u
                _N.fill_diagonal(oo.s2_u, [0.5, 0.5])
            else:
                oo.u_u[:] = u0 + _N.random.randn(oo.nStates)
                _N.fill_diagonal(oo.s2_u, _N.ones(oo.nStates)*0.5)
        else:
            if oo.nStates == 2:
                oo.u_u[:]     = (u0_w1*1.2, u0_w1*0.8, u0_w2*1.2, u0_w2*0.8)
                _N.fill_diagonal(oo.s2_u, [0.5, 0.5, 0.5, 0.5])
            else:
                oo.u_u[:]     = 0.5*(u0_w1 + u0_w1) + _N.random.randn(2*oo.nStates)
                _N.fill_diagonal(oo.s2_u, _N.ones(oo.nStates*2)*0.5)

        #  m1, m2 --  Dirichlet prior   ##  PRIORS
        oo.alp          = _N.ones(oo.nStates)
        oo.priors = {"u_u" : oo.u_u, "s2_u" : oo.s2_u, "alp" : oo.alp,
                     "u_x00" : oo.u_x00, "s2_x00" : oo.s2_x00}

        ################# #generate initial values of parameters, time series
        oo._d = _kfardat.KFARGauObsDat(1, oo.N, 1, onetrial=True)
        oo._d.copyData(_N.empty(oo.N+1), _N.empty(oo.N+1), onetrial=True)   #  dummy data copied

        oo.x00 = oo.u_x00 + _N.sqrt(oo.s2_x00)*_N.random.rand()
        oo.V00 = oo.B_V00*_ss.invgamma.rvs(oo.a_V00)

        oo.smp_F        = _N.zeros(oo.NMC + oo.burn)
        oo.smp_q2       = _N.zeros(oo.NMC + oo.burn)
        if oo.nWins == 1:
            oo.smp_u        = _N.zeros((oo.NMC + oo.burn, oo.nStates))   
        else:
            #  uL_w1, uH_w1, uL_w2, uH_w2, ....
            oo.smp_u        = _N.zeros((oo.NMC + oo.burn, oo.nWins, oo.nStates))   
        oo.smp_m        = _N.zeros((oo.NMC + oo.burn, oo.nStates))
        oo.smpx = _N.zeros(oo.N + 1)   #  start at 0 + u
        oo.Bsmpx= _N.zeros((oo.NMC, oo.N + 1))
        oo.smpld_params = _N.empty((oo.NMC + oo.burn, 4 + 2*oo.nStates))  #m1, m2, u1, u2
        oo.z   = _N.empty((oo.NMC+oo.burn, oo.N+1, oo.nStates), dtype=_N.int16)   #  augmented data
        #  _N.where(oo.z[tr, :, 0] == 1)   all trials in state 0

        if oo.nWins == 1:
            meanO = _N.mean(oo.y)
        else:
            meanO = _N.mean(oo.y[:, 0] + oo.y[:, 1])

        oo.m     = _N.empty(oo.nStates)
        #   m[0] is low state, m[1] is high state, z = (1, 0) indicates low state
        if oo.nStates == 2:
            for n in xrange(oo.N+1):
                #oo.z[0, n, :] = int(oo.nStates * _N.random.rand())
                oo.z[0, n, :] = oo.states[1]  #  low state
                if ((oo.nWins == 1) and (oo.y[n] < meanO)) or \
                   ((oo.nWins == 2) and (oo.y[n, 0] + oo.y[n, 1] < meanO)):
                    oo.z[0, n, :] = oo.states[0]  #  low state
        else:
            for n in xrange(oo.N+1):
                oo.z[0, n, :] = oo.states[int(_N.random.rand()*oo.nStates)]  #  

        for ns in xrange(oo.nStates):
            oo.m[ns]   = _N.sum(oo.z[0, :, ns]) / float(oo.N+1)

        if oo.nWins == 1:
            oo.u   = mvnrml(oo.u_u, oo.s2_u)
            trm  = _N.empty(oo.nStates)

            oo.us   = _N.dot(oo.z[0, :, :], oo.u)
            oo.ws = lw.rpg_devroye(oo.rn[0], oo.smpx + oo.us, num=(oo.N + 1))
        else:
            oo.u_w1   = mvnrml(oo.u_u[0:oo.nStates], oo.s2_u[0:oo.nStates, 0:oo.nStates])
            oo.u_w2   = mvnrml(oo.u_u[oo.nStates:2*oo.nStates], 
                            oo.s2_u[oo.nStates:oo.nStates*2, oo.nStates:oo.nStates*2])

            #  generate PG latents.  Depends on Xs and us, zs.  us1 us2 
            oo.us_w1 = _N.dot(oo.z[0, :, :], oo.u_w1)  #  us now correct level for given state
            oo.us_w2 = _N.dot(oo.z[0, :, :], oo.u_w2)
            oo.ws_w1 = lw.rpg_devroye(oo.rn[0], oo.smpx + oo.us_w1, num=(oo.N + 1))
            oo.ws_w2 = lw.rpg_devroye(oo.rn[1], oo.smpx + oo.us_w2, num=(oo.N + 1))

            trm_w1  = _N.empty(oo.nStates)
            trm_w2  = _N.empty(oo.nStates)

    def gibbsSamp(self):  #########  GIBBS SAMPLER  ############
        #####  MCMC start
        oo   = self
        trms = _N.empty(oo.nStates)
        thr  = _N.empty(oo.nStates)


        for it in xrange(1, oo.NMC+oo.burn):
            if (it % 10) == 0:
                print it

            if oo.nWins == 1:
                kw  = oo.kp / oo.ws            # convenience variables
            else:
                kw_w1  = oo.kp_w1 / oo.ws_w1
                kw_w2  = oo.kp_w2 / oo.ws_w2

            rnds =_N.random.rand(oo.N+1)

            #  generate latent zs.  Depends on Xs and PG latents
            for n in xrange(oo.N+1):
                thr[:] = 0
                if oo.nWins == 1:
                    #  for oo.nStates, there are oo.nStates - 1 thresholds
                    for i in xrange(oo.nStates):
                        trms[i] = -0.5*oo.ws[n]*((oo.u[i] + oo.smpx[n]) - kw[n]) * ((oo.u[i] + oo.smpx[n]) - kw[n])
                else:
                    for i in xrange(oo.nStates):
                        #  rsd_w1 is 2-component vector (if oo.nStates == 2)
                        trms[i] = -0.5*oo.ws_w1[n] * ((oo.u_w1[i] + oo.smpx[n] - kw_w1[n]) * (oo.u_w1[i] + oo.smpx[n] - kw_w1[n]) - (kw_w1[n]*kw_w1[n])) \
                                  -0.5*oo.ws_w2[n] * ((oo.u_w2[i] + oo.smpx[n] - kw_w2[n]) * (oo.u_w2[i] + oo.smpx[n] - kw_w2[n]) - (kw_w2[n]*kw_w2[n]))
                        # trm    = trm_w1 * trm_w2  #  trm is 2-component vector
                for tp in xrange(oo.nStates):
                    for bt in xrange(oo.nStates):
                        #  we are calculating the denominators here
                        #  if denominator -> inf, the thr for this term is 0
                        #  practically no difference limiting denom to exp(700)
                        expArg  = 700 if (trms[bt] - trms[tp] > 700) else (trms[bt] - trms[tp])
                        thr[tp] += (oo.m[bt]/oo.m[tp])*_N.exp(expArg)
                thr = 1 / thr

                oo.z[it, n, :] = oo.states[oo.nStates - 1]   #
                thrC = 0
                s = 0
                while s < oo.nStates - 1:
                    thrC += thr[s]
                    if rnds[n] < thrC:
                        oo.z[it, n, :] = oo.states[s]
                        break
                    s += 1

            if oo.nWins == 1:
                oo.us   = _N.dot(oo.z[it, :, :], oo.u)
                oo.ws = lw.rpg_devroye(oo.rn[0], oo.smpx + oo.us, num=(oo.N + 1))
            else:
                #  generate PG latents.  Depends on Xs and us, zs.  us1 us2 
                oo.us_w1 = _N.dot(oo.z[it, :, :], oo.u_w1)   #  either low or high u
                oo.us_w2 = _N.dot(oo.z[it, :, :], oo.u_w2)
                oo.ws_w1 = lw.rpg_devroye(oo.rn[0], oo.smpx + oo.us_w1, num=(oo.N + 1))
                oo.ws_w2 = lw.rpg_devroye(oo.rn[1], oo.smpx + oo.us_w2, num=(oo.N + 1))


            oo._d.copyParams(_N.array([oo.F0]), oo.q2, onetrial=True)
            #  generate latent AR state
            oo._d.f_x[0, 0, 0]     = oo.x00
            oo._d.f_V[0, 0, 0]     = oo.V00
            if oo.nWins == 1:
                oo._d.y[:]             = oo.kp/oo.ws - oo.us
                oo._d.Rv[:] =1 / oo.ws   #  time dependent noise
            else:
                btm      = 1 / oo.ws_w1 + 1 / oo.ws_w2   #  size N
                top = (oo.kp_w1/oo.ws_w1 - oo.us_w1) / oo.ws_w2 + (oo.kp_w2/oo.ws_w2 - oo.us_w2) / oo.ws_w1
                oo._d.y[:] = top/btm
                oo._d.Rv[:] =1 / (oo.ws_w1 + oo.ws_w2)   #  time dependent noise

            oo.smpx = _kfar.armdl_FFBS_1itr(oo._d)

            #  p3 --  samp u here

            dirArgs = _N.empty(oo.nStates)

            for i in xrange(oo.nStates):
                dirArgs[i] = oo.alp[i] + _N.sum(oo.z[it, :, i])
            oo.m[:] = _N.random.dirichlet(dirArgs)

            ###  Sample u, rn
            if oo.nWins == 1:
                xim = _N.empty(oo.nStates)
                yim = _N.empty(oo.nStates)
                ui  = _N.empty(oo.nStates)# mean of proposal density
                u1  = _N.empty(oo.nStates)
                occi= _N.empty(oo.nStates)  #  occupancy state i

                lFlB = _N.empty(2)
                pu   = _N.empty(oo.nStates)
                p1   = _N.empty(oo.nStates)  #  p param generated by proposal sum(p1) != 1
                p0   = _N.empty(oo.nStates)  #  
                dp   = _N.empty((2, oo.nStates))
                du   = _N.empty((2, oo.nStates))  # new, old 
                n1   = _N.empty(oo.nWins)
                n0   = oo.rn
                n1n0 = _N.empty(2)
                indsL= []  #  indices of trials belong in state i

                Mk   = _N.empty(oo.nStates)
                iMk100 = _N.empty(oo.nStates)
                stdu  = 0.01
                stdu2  = stdu*stdu
                nmin= max(oo.y)

                for i in xrange(oo.nStates):
                    indsL.append(_N.where(oo.z[it, :, i] == 1)[0])

                    Mk[i]   = _N.mean(oo.y[indsL[i]])
                    iMk100[i] = int(Mk[i]*100)

                    occi[i] = float(indsL[i].shape[0]) / (oo.N + 1) # ratio in state i
                    
                    xim[i] = _N.mean(oo.smpx[indsL[i]])  #  mean latent state in state i
                    yim[i] = _N.mean(oo.y[indsL[i]])  #  mean cts for state i
                    ui[i]  = xim[i] - _N.log(oo.rn[0] / yim[i] - 1) # mean of proposal density
                    #pu[i]   = Mk[i] / oo.rn  # don't need this, but used in derivation
                #### MCMC
                for ii in xrange(100):
                    #  sample new ui
                    for i in xrange(oo.nStates):
                        u1[i] = ui[i] + stdu * _N.random.randn() # sample ui
                        p1[i] = 1 / (1 + _N.exp(-u1[i] - xim[i]))
                        p0[i] = 1 / (1 + _N.exp(-oo.u[i] - xim[i]))
                    lam = _N.sum(occi*(Mk/p1))
                    rv = _ss.poisson(lam)

                    n1[0] = oo.trPoi(lam, a=nmin, b=max(iMk100))   #  mean is p0/Mk
                    oo.Llklhds(indsL, oo.y, n1, p1, oo.rn, p0, lFlB)
                    #dp[0]   = (p1 - pu)**2;  dp[1]   = (p0 - pu)**2  # p1, pu multi-dim
                    du[0]   = (u1 - ui)**2;  du[1]   = (oo.u - ui)**2  # p1, pu multi-dim
                    n1n0[0] = n1;            n1n0[1] = oo.rn

                    #  du is (fb(2) x states)
                    qFqB = rv.pmf(n1n0) * _N.exp(-0.5*du[0]/stdu2) * _N.exp(-0.5*du[1]/stdu2)

                    posRat = 1e+50 if (lFlB[0]-lFlB[1] > 100) else _N.exp(lFlB[0]-lFlB[1])
                    if (qFqB[1] == 0) and (qFqB[0] == 0):
                        prRat  = 1
                    else:
                        prRat = 1e+10 if (qFqB[0] == 0) else qFqB[1]/qFqB[0]
                    aln  = min(1, posRat*prRat)

                    if _N.random.rand() < aln:
                        oo.u[:] = u1[:]
                        oo.rn[:] = n1[:]
                #  

            print oo.rn

            if oo.model=="negative binomial":
                if oo.nWins == 1:
                    oo.kp   = (oo.y - oo.rn) *0.5
                else:
                    oo.kp_w1   = (y[oo.t0:oo.t1, 0] - oo.rn[0]) *0.5
                    oo.kp_w2   = (y[oo.t0:oo.t1, 1] - oo.rn[1]) *0.5
            else:
                if oo.nWins == 1:
                    oo.kp  = oo.y - oo.rn*0.5
                else:
                    oo.kp_w1  = oo.y[oo.t0:oo.t1, 0] - oo.rn[0]*0.5
                    oo.kp_w2  = oo.y[oo.t0:oo.t1, 1] - oo.rn[1]*0.5

            # sample F0
            F0AA = _N.dot(oo.smpx[0:-1], oo.smpx[0:-1])
            F0BB = _N.dot(oo.smpx[0:-1], oo.smpx[1:])

            F0std= _N.sqrt(oo.q2/F0AA)
            F0a, F0b  = (oo.a_F0 - F0BB/F0AA) / F0std, (oo.b_F0 - F0BB/F0AA) / F0std
            oo.F0=F0BB/F0AA+F0std*_ss.truncnorm.rvs(F0a, F0b)

            #####################    sample q2
            a = oo.a_q2 + 0.5*(oo.N+1)  #  N + 1 - 1
            rsd_stp = oo.smpx[1:] - oo.F0*oo.smpx[0:-1]
            BB = oo.B_q2 + 0.5 * _N.dot(rsd_stp, rsd_stp)
            oo.q2 = _ss.invgamma.rvs(a, scale=BB)
            # #####################    sample x00
            mn  = (oo.u_x00*oo.V00 + oo.s2_x00*oo.x00) / (oo.V00 + oo.s2_x00)
            vr = (oo.V00*oo.s2_x00) / (oo.V00 + oo.s2_x00)
            oo.x00 = mn + _N.sqrt(vr)*_N.random.randn()
            #####################    sample V00
            aa = oo.a_V00 + 0.5
            BB = oo.B_V00 + 0.5*(oo.smpx[0] - oo.x00)*(oo.smpx[0] - oo.x00)
            oo.V00 = _ss.invgamma.rvs(aa, scale=BB)

            oo.smp_F[it]       = oo.F0
            oo.smp_q2[it]      = oo.q2
            if oo.nWins == 1:
                oo.smp_u[it, :] = oo.u
            else:
                oo.smp_u[it, 0, :] = oo.u_w1
                oo.smp_u[it, 1, :] = oo.u_w2
            oo.smp_m[it, :]    = oo.m

            if it >= oo.burn:
                oo.Bsmpx[it-oo.burn, :] = oo.smpx


    def Llklhds(self, indsL, ks, n1, p1, n2, p2, out):
        oo = self
        #  new configuration
        out[0] = 0;        out[1] = 0    #  proposal and original
        for st in xrange(oo.nStates):
            ksS = ks[indsL[st]]
            out[0] += _N.sum(_N.log(_sm.comb(n1, ksS)) + ksS*_N.log(p1[st]) + (n1-ksS)*_N.log(1 - p1[st]))
            #  old configuration
            out[1] += _N.sum(_N.log(_sm.comb(n2, ksS)) + ksS*_N.log(p2[st]) + (n2-ksS)*_N.log(1 - p2[st]))
        
    def trPoi(self, lmd, a, b):
        """
        a, b inclusive
        """
        oo = self
        ct = a - 1  # init value
        while (ct < a) or (ct > b):
            ct = _ss.poisson.rvs(lmd)
        return ct
