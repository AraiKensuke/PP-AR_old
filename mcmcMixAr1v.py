import kflib
import scipy.stats as _ss
import kfardat as _kfardat
from kassdirs import resFN, datFN
import kstat as _ks
import os
import numpy as _N

import pickle as _pkl
import LogitWrapper as lw

from mcmcMixAR1vFuncs import loadDat

mvnrml    = _N.random.multivariate_normal

#  Simulation params
setname       = None
rs            = -1
burn          = None
NMC           = None
_t0           = None
_t1           = None
#  Description of model
model         = None
nWins         = None
rn            = None
nStates       = 2
k             = None
#  Sampled parameters
Bsmpx         = None
smp_u         = None
smp_q2        = None
smp_x00       = None
#  store samples of
allalfas      = None
pgs           = None
fs            = None
amps          = None

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

x           = None   #  Hidden latent trend 
y           = None   #  spike count observation 
kp          = None;  kp_w1       = None; kp_w2     = None

def run(runDir=None):
    global setname, _t0, _t1, _d, Bsmpx, uts, wts  #  only these that we are setting inside
    global allalfas, smp_B, smp_aS, smp_q2
    global x, y, kp, kp_w1, kp_w2
    global nWins

    setname = os.getcwd().split("/")[-1]

    nWins    = len(rn)   #  we don't set nWins manually

    if runDir == None:
        runDir="%(sn)s/AR%(k)d_[%(t0)d-%(t1)d]" % {"sn" : setname, "ar" : k, "t0" : _t0, "t1" : _t1}

    if rs >= 0:
        unpickle(runDir, rs)
    else:   #  First run
        restarts = 0

    if nWins == 1:
        N, x, y, kp, u0 = loadDat(setname, model, nStates, nWins, rn, t0=_t0, t1=_t1)  # u is set initialized
    else:
        N, x, y, kp_w1, kp_w2, u0_w1, u0_w2 = loadDat(setname, model, nStates, nWins, rn, t0=_t0, t1=_t1)  # u is set initialized

    #######  PRIOR parameters
    #  F0  --  flat prior
    #a_F0         = -1
    #  I think a prior assumption of relatively narrow and high F0 range
    #  is warranted.  Small F0 is close to white noise, and as such can
    #  be confused with the independent trial-to-trial count noise.Force
    #  it to search for longer timescale correlations by setting F0 to be
    #  fairly large.
    a_F0         = -0.3             #  prior assumption: slow fluctuation
    b_F0         =  1
    #  u   --  Gaussian prior
    u_u        = _N.empty(nStates * nWins)
    s2_u       = _N.zeros((nStates * nWins, nStates * nWins))
    # (win1 s1)    (win1 s2)    (win2 s1)    (win2 s2)
    if nWins == 1:
        if nStates == 2:
            u_u[:]     = (u0*1.2, u0*0.8)
            _N.fill_diagonal(s2_u, [0.5, 0.5])
        else:
            u_u[:] = u0 + _N.random.randn(nStates)
            _N.fill_diagonal(s2_u, _N.ones(nStates)*0.5)
    else:
        if nStates == 2:
            u_u[:]     = (u0_w1*1.2, u0_w1*0.8, u0_w2*1.2, u0_w2*0.8)
            _N.fill_diagonal(s2_u, [0.5, 0.5, 0.5, 0.5])
        else:
            u_u[:]     = 0.5*(u0_w1 + u0_w1) + _N.random.randn(2*nStates)
            _N.fill_diagonal(s2_u, _N.ones(nStates*2)*0.5)

    #  m1, m2 --  Dirichlet prior   ##  PRIORS
    alp          = _N.ones(nStates)
    priors = {"u_u" : u_u, "s2_u" : s2_u, "alp" : alp,
              "u_x00" : u_x00, "s2_x00" : s2_x00}

    ################# #generate initial values of parameters, time series
    _d = _kfardat.KFARGauObsDat(1, N, 1)
    _d.copyData(_N.empty(N+1), _N.empty(N+1))   #  dummy data copied

    F0  = 0.9
    q2  = 0.015
    x00 = u_x00 + _N.sqrt(s2_x00)*_N.random.rand()
    V00 = B_V00*_ss.invgamma.rvs(a_V00)

    smp_F        = _N.zeros(NMC + burn)
    smp_q2       = _N.zeros(NMC + burn)
    if nWins == 1:
        smp_u        = _N.zeros((NMC + burn, nStates))   
    else:
        #  uL_w1, uH_w1, uL_w2, uH_w2, ....
        smp_u        = _N.zeros((NMC + burn, nWins, nStates))   
    smp_m        = _N.zeros((NMC + burn, nStates))

    smpx = _N.zeros(N + 1)   #  start at 0 + u
    Bsmpx= _N.zeros((NMC, N + 1))
    smpld_params = _N.empty((NMC + burn, 4 + 2*nStates))  #m1, m2, u1, u2
    z   = _N.empty((NMC+burn, N+1, nStates), dtype=_N.int16)   #  augmented data


    #######  Initialize z
    if nStates == 2:
        states = _N.array([[1, 0], [0, 1]])
    elif nStates == 3:
        states = _N.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    if nWins == 1:
        meanO = _N.mean(y)
    else:
        meanO = _N.mean(y[:, 0] + y[:, 1])

    m     = _N.empty(nStates)
    #   m[0] is low state, m[1] is high state, z = (1, 0) indicates low state
    if nStates == 2:
        for n in xrange(N+1):
            z[0, n, :] = states[1]  #  low state
            if ((nWins == 1) and (y[n] < meanO)) or \
               ((nWins == 2) and (y[n, 0] + y[n, 1] < meanO)):
                z[0, n, :] = states[0]  #  low state
    else:
        for n in xrange(N+1):
            z[0, n, :] = states[int(_N.random.rand()*nStates)]  #  

    for ns in xrange(nStates):
        m[ns]   = _N.sum(z[0, :, ns]) / float(N+1)
    
    if nWins == 1:
        u   = mvnrml(u_u, s2_u)

        trm  = _N.empty(nStates)

        us   = _N.dot(z[0, :, :], u)
        ws = lw.rpg_devroye(rn, smpx + us, num=(N + 1))
    else:
        u_w1   = mvnrml(u_u[0:nStates], s2_u[0:nStates, 0:nStates])
        u_w2   = mvnrml(u_u[nStates:2*nStates], 
                        s2_u[nStates:nStates*2, nStates:nStates*2])

        #  generate PG latents.  Depends on Xs and us, zs.  us1 us2 
        us_w1 = _N.dot(z[0, :, :], u_w1)   #  either low or high u
        us_w2 = _N.dot(z[0, :, :], u_w2)
        ws_w1 = lw.rpg_devroye(rn[0], smpx + us_w1, num=(N + 1))
        ws_w2 = lw.rpg_devroye(rn[1], smpx + us_w2, num=(N + 1))

        trm_w1  = _N.empty(nStates)
        trm_w2  = _N.empty(nStates)



    
#    Bsmpx, smp_F, smp_q2, smp_u, smp_m, z  = _ml.mcmcMixAR1(burn, NMC, y, nStates=nStates, nWins=nWins, n=rn, r=rn, model=model)

