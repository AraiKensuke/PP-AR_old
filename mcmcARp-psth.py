"""
#   lbd2 = _N.array([0.001, 0.08, 0.35, 1.2, 1.5, 1.1, 1.01, 1.])     #  use this for 080402,0,121
exf("kflib.py")
exf("restartPickle.py")

from mcmcARpFuncs import loadDat, initBernoulli
import patsy

from gibbsMP import build_lrn, build_lrnLambda2
import scipy.stats as _ss
from kassdirs import resFN, datFN

from mcmcARpPlot import plotFigs, plotARcomps, plotQ2
import kfardat as _kfardat
import time as _tm

import utilities as _U

import numpy.polynomial.polynomial as _Npp
import time as _tm
import ARlib as _arl
import LogitWrapper as lw
#from gibbsAbsRef import gibbsSamp
#from gibbs import gibbsSampH
from gibbsMP import gibbsSampH

import logerfc as _lfc
import commdefs as _cd

from ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF
os.system("taskset -p 0xff %d" % os.getpid())
_lfc.init()

#setname="oscCts-45"   #  params_XXX.py   Results/XXX/params.py
#model  ="binomial"
#setname="spksSloHm-1"   #  params_XXX.py   Results/XXX/params.py
#setname="ArbRef-LF-HM-2"   #  params_XXX.py   Results/XXX/params.py
setname="080402-0-121-cL"   #  params_XXX.py   Results/XXX/params.py
model  ="bernoulli"

burn  = 800
NMC   = 600
dt    = 0.001
ARord =_cd.__NF__    #  AR coefficient sampling order, signal first

rs=-1
rsDir="%(sn)s/AR15_[0-1000]_cmpref_nf_MW" % {"sn" : setname}

if rs >= 0:
    unpickle(rsDir, rs)
else:   #  First run
    restarts = 0
    use_prior=_cd.__COMP_REF__
    bMW      = True

    #  restrict AR class
    fSigMax      = 500.
    freq_lims   = [[0.5, fSigMax]]#, [0.5, fSigMax]]
    spkHz       = 30.
    ifs         = [(spkHz / fSigMax) * _N.pi]
    Cn          = 6   #  # of noize components
    Cs          = len(freq_lims)
    C     = Cn + Cs
    R     = 1
    k     = 2*C + R

    #  only use a portion
    _t0    = 0
    _t1    = 450   #  1700

TR, rn, _x, _y, _fx, _px, N, kp, _u, rmTrl, kpTrl = loadDat(setname, model, t0=_t0, t1=_t1, filtered=True, phase=True)  # u is set initialized

l2 = loadL2(setname)
if (l2 != None) and (len(l2.shape) == 0):
    l2 = _N.array([l2])

ID_q2 = True
ARfixed=False
TR = 87
TR0 = 0
TR1 = TR0 + TR
runNotes(setname, ID_q2, TR0, TR1)

y     = _N.array(_y[kpTrl][TR0:TR1])
x     = _N.array(_x[kpTrl][TR0:TR1])
fx    = _N.array(_fx[kpTrl][TR0:TR1])
px    = _N.array(_px[kpTrl][TR0:TR1])
u     = _N.array(_u[kpTrl][TR0:TR1])

nbs = 4
B = patsy.bs(_N.linspace(0, (_t1 - _t0)*dt, (_t1-_t0)), df=nbs, include_intercept=True)    #  spline basis
B = B.T    #  My convention for beta
aS = _N.linalg.solve(_N.dot(B, B.T), _N.dot(B, _N.ones(_t1 - _t0)*_N.mean(u)))

###########  PRIORs
#  u   --  Gaussian prior
u_u          = 0;             s2_u         = 5
#  q2  --  Inverse Gamma prior
a_q2         = 1e-1;          B_q2         = 1e-6
#  x0  --  Gaussian prior
u_x00        = _N.zeros(k);   s2_x00       = _arl.dcyCovMat(k, _N.ones(k), 0.4)
priors = {"u_u" : u_u, "s2_u" : s2_u, "a_q2" : a_q2, "B_q2" : B_q2,
          "u_x00" : u_x00, "s2_x00" : s2_x00}

# #generate initial values of parameters
_d = _kfardat.KFARGauObsDat(TR, N, k)
_d.copyData(y)

sPR="cmpref"
if use_prior==_cd.__FREQ_REF__:
    sPR="frqref"
elif use_prior==_cd.__ONOF_REF__:
    sPR="onfref"
sAO="sf"
if ARord==_cd.__SF__:
    sAO="sf"
elif ARord==_cd.__NF__:
    sAO="nf"

ts        = "[%(1)d-%(2)d]" % {"1" : _t0, "2" : _t1}
baseFN    = "rs=%(rs)d" % {"pr" : sPR, "rs" : restarts}
if bMW:
    setdir="%(sd)s/AR%(k)d_%(ts)s_%(pr)s_%(ao)s_MW" % {"sd" : setname, "k" : k, "ts" : ts, "pr" : sPR, "ao" : sAO}
else:
    setdir="%(sd)s/AR%(k)d_%(ts)s_%(pr)s_%(ao)s" % {"sd" : setname, "k" : k, "ts" : ts, "pr" : sPR, "ao" : sAO}

#  baseFN_inter   baseFN_comps   baseFN_comps

###############

Bsmpx   = _N.zeros((TR, NMC+burn, (N+1) + 2))
smp_aS  = _N.zeros((burn + NMC, nbs))
smp_q2  = _N.zeros((TR, burn + NMC))
smp_x00 = _N.empty((TR, burn + NMC-1, k))
#  store samples of
allalfas=     _N.empty((burn + NMC, k), dtype=_N.complex)
uts     = _N.empty((TR, burn + NMC, R, N+2))
wts     = _N.empty((TR, burn + NMC, C, N+3))
ranks   = _N.empty((burn + NMC, C), dtype=_N.int)
pgs     = _N.empty((TR, burn + NMC, N+1))
fs          =     _N.empty((burn + NMC, C))
amps        =     _N.empty((burn + NMC, C))

radians     = buildLims(Cn, freq_lims, nzLimL=1.)
AR2lims     = 2*_N.cos(radians)

if (rs < 0):
    smpx        = _N.zeros((TR, (_d.N + 1) + 2, k))   #  start at 0 + u
    ws          = _N.empty((_d.TR, _d.N+1), dtype=_N.float)

    F_alfa_rep  = initF(R, Cs, Cn, ifs=ifs)   #  init F_alfa_rep

    print "begin---"
    print ampAngRep(F_alfa_rep)
    print "begin^^^"
    q20         = 1e-3
    q2          = _N.ones(TR)*q20

    F0          = (-1*_Npp.polyfromroots(F_alfa_rep)[::-1].real)[1:]
    ########  Limit the amplitude to something reasonable
    xE, nul = createDataAR(N, F0, q20, 0.1)
    mlt  = _N.std(xE) / 0.5    #  we want amplitude around 0.5
    q2          /= mlt*mlt
    xE, nul = createDataAR(N, F0, q2[0], 0.1)

    initBernoulli(model, k, F0, TR, _d.N, y, fSigMax, smpx, Bsmpx)
    #smpx[0, 2:, 0] = x[0]    ##########  DEBUG

    ####  initialize ws if starting for first time
    if TR == 1:
        ws   = ws.reshape(1, _d.N+1)
    for m in xrange(_d.TR):
        lw.rpg_devroye(rn, smpx[m, 2:, 0] + _N.dot(B.T, aS), num=(N + 1), out=ws[m, :])

ARo   = _N.empty((TR, _d.N+1))
#smp_u[:, 0] = u
smp_q2[:, 0]= q2

t1    = _tm.time()

# if model == "bernoulli":
F_alfa_rep = gibbsSampH(burn, NMC, AR2lims, F_alfa_rep, R, Cs, Cn, TR, rn, _d, B, aS, q2, uts, wts, kp, ws, smpx, Bsmpx, smp_aS, smp_q2, allalfas, fs, amps, ranks, priors, ARo, l2, bMW=bMW, prior=use_prior, aro=ARord, ID_q2=ID_q2, ARfixed=ARfixed)

t2    = _tm.time()
print (t2-t1)

plotARcomps(setdir, N, k, burn, NMC, fs, amps, _t0, _t1, Cs, Cn, C, baseFN, TR, m)

for m in xrange(TR):
    plotFigs(setdir, N, k, burn, NMC, fx, y, Bsmpx, smp_q2, _t0, _t1, Cs, Cn, C, baseFN, TR, m, ID_q2=ID_q2, bRealDat=True)



if ID_q2:
    plotQ2(setdir, baseFN, burn, NMC, TR0, TR1, smp_q2, hilite=[0, 18, 21])


pickleForLater()


pcs  = []
pcsf = []
T   = 400
allWFs = _N.empty(TR*T)
allXs  = _N.empty(TR*T)
for tr in xrange(TR):
    fwf   = _N.mean(zt[tr, 700:750, 1:, 0], axis=0)
    allWFs[tr*T:(tr+1)*T] = fwf
    allXs[tr*T:(tr+1)*T] = fx[tr]
    #fwf  = bpFilt(8, 20, 1, 30, 500, wf)
    pcf, pvf = _ss.pearsonr(fwf, fx[tr])
    pc, pv = _ss.pearsonr(wf, fx[tr])
    fig = _plt.figure(figsize=(12, 4*2))
    fig.add_subplot(2, 1, 1)
    #_plt.plot(wf, lw=2, color="red")
    #_plt.plot((fx[tr] * _N.std(wf) / _N.std(fx[tr])), lw=2, color="black")
    fig.add_subplot(2, 1, 2)
    _plt.plot(fwf, lw=2, color="red")
    _plt.plot((fx[tr] * _N.std(fwf) / _N.std(fx[tr])), lw=2, color="black")
    _plt.suptitle("pc=%(pc).3f     pcf=%(pcf).3f" % {"pc" : pc, "pcf" : pcf})
    _plt.savefig(resFN("smpx,theta,%d" % tr, dir=setdir)) 
    _plt.close()

    plotWFandSpks(N, y[tr], [fx[tr]*_N.std(wf)/_N.std(fx[tr]), wf], sFilename=resFN("fx_smpx_spks%d" % tr, dir=setdir))   

    pcs.append(pc)
    pcsf.append(pcf)

fig = _plt.figure(figsize=(4, 2*3))
fig.add_subplot(2, 1, 1)
_plt.hist(pcs, bins=_N.linspace(-0.8, 0.8, 26), color="black")
_plt.xlim(-0.7, 0.7)
_plt.axvline(x=_N.mean(pcs), color="red", lw=2, ls="--")
_plt.grid()
fig.add_subplot(2, 1, 2)
_plt.hist(pcsf, bins=_N.linspace(-0.8, 0.8, 26), color="black")
_plt.xlim(-0.7, 0.7)
_plt.axvline(x=_N.mean(pcsf), color="red", lw=2, ls="--")
_plt.grid()
_plt.savefig(resFN("pc_hist", dir=setdir))
_plt.close()


fig = _plt.figure()
_plt.plot(_N.dot(B.T, _N.mean(smp_aS[700:750], axis=0)))
_plt.savefig(resFN("psth", dir=setdir))
_plt.close()


"""

_ss.pearsonr(pcs, _N.std(fx, axis=1))

