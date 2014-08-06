#   lbd2 = _N.array([0.001, 0.08, 0.35, 1.2, 1.5, 1.1, 1.01, 1.])     #  use this for 080402,0,121
exf("kflib.py")
exf("restartPickle.py")

from mcmcARpFuncs import loadDat, initBernoulli

from gibbs import build_lrn, build_lrnLambda2
import scipy.stats as _ss
from kassdirs import resFN, datFN

from mcmcARpPlot import plotFigs, plotARcomps
import kfardat as _kfardat
import time as _tm

import utilities as _U

import numpy.polynomial.polynomial as _Npp
import time as _tm
import ARlib as _arl
import LogitWrapper as lw
#from gibbsAbsRef import gibbsSamp
from gibbs import gibbsSamp, gibbsSampH

import logerfc as _lfc
import commdefs as _cd

from ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF
_lfc.init()

#setname="oscCts-45"   #  params_XXX.py   Results/XXX/params.py
#model  ="binomial"
#setname="spksSloHm-1"   #  params_XXX.py   Results/XXX/params.py
#setname="LIF-080402-0-121-HF-HML"   #  params_XXX.py   Results/XXX/params.py
setname="080402-0-121L"   #  params_XXX.py   Results/XXX/params.py
model  ="bernoulli"

burn  = 500
NMC   = 1000
ARord =_cd.__NF__    #  AR coefficient sampling order, signal first

rs=-1
rsDir="%(sn)s/AR16_[1000-6000]_cmpref_nf_MW" % {"sn" : setname}

if rs >= 0:
    unpickle(rsDir, rs)
else:   #  First run
    restarts = 0
    use_prior=_cd.__COMP_REF__
    bMW      = True

    #  restrict AR class
    fSigMax      = 500
    freq_lims   = [[1, fSigMax],]# [1, fSigMax]]
    #spkHz       = (1000*_N.sum(y)) / N
    spkHz       = 30.
    ifs         = [(spkHz / fSigMax) * _N.pi]#, (spkHz / fSigMax) * _N.pi, ]
    Cn          = 8   #  # of noize components
    Cs          = len(freq_lims)
    C     = Cn + Cs
    R     = 2
    k     = 2*C + R

    #  only use a portion
    _t0    = 0
    _t1    = 400

TR, rn, _x, _y, N, kp, _u = loadDat(setname, model, t0=_t0, t1=_t1)  # u is set initialized
