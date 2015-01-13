import scipy.optimize as _sco
from shutil import copyfile
from utildirs import setFN
from mcmcARpPlot import plotWFandSpks
import matplotlib.pyplot as _plt

from kflib import createDataPPl2Simp, savesetMT
from kassdirs import resFN, datFN
import numpy as _N
import pickle as _pk
import warnings
import numpy.polynomial.polynomial as _Npp
import utilities as _U
from kstat import percentile

TR         = None;     N        = None;      dt       = 0.001
trim       = 50;
nzs        = None;     nRhythms = None;
rs         = None;     ths      = None;      alfa     = None;
lambda2    = None;     psth     = None
lowQpc     = 0;        lowQs    = None
isis       = None;     rpsth    = None
us         = None

def create(setname):
    # _plt.ioff()
    copyfile("%s.py" % setname, "%(s)s/%(s)s.py" % {"s" : setname, "to" : setFN("%s.py" % setname, dir=setname, create=True)})
    global dt, lambda2, rpsth, isis, us
    ARcoeff = _N.empty((nRhythms, 2))
    for n in xrange(nRhythms):
        ARcoeff[n]          = (-1*_Npp.polyfromroots(alfa[n])[::-1][1:]).real

    #  x, prbs, spks    3 columns
    nColumns = 3
    alldat  = _N.empty((N, TR*nColumns))
    spksPT  = _N.empty(TR)
    stNzs   = _N.empty((TR, nRhythms))

    lowQs   = []
    for tr in xrange(TR):
        if _N.random.rand() < lowQpc:
            lowQs.append(tr)
            stNzs[tr] = nzs[:, 0]
        else:
            stNzs[tr] = nzs[:, 1]    #  high
    isis   = []
    rpsth  = []

    if us is None:
        us = _N.zeros(TR)
    elif (type(us) is float) or (type(us) is int):
        us = _N.zeros(TR) * us
    for tr in xrange(TR):
        x, dN, prbs, fs = createDataPPl2Simp(TR, N, dt, ARcoeff, psth + us[tr], stNzs[tr], lambda2=lambda2, p=1, nRhythms=nRhythms)

        spksPT[tr] = _N.sum(dN)
        rpsth.extend(_N.where(dN == 1)[0])
        alldat[:, nColumns*tr] = _N.sum(x, axis=0).T
        alldat[:, nColumns*tr+1] = prbs
        alldat[:, nColumns*tr+2] = dN
        isis.extend(_U.toISI([_N.where(dN == 1)[0].tolist()])[0])

    savesetMT(TR, alldat, model, setname)

    arfs = ""
    xlst = []
    for nr in xrange(nRhythms):
        arfs += "%.1fHz " % (500*ths[nr]/_N.pi)
        xlst.append(x[nr])
    sTitle = "AR2 freq %(fs)s    spk Hz %(spkf).1fHz   TR=%(tr)d   N=%(N)d" % {"spkf" : (_N.sum(spksPT) / (N*TR*0.001)), "tr" : TR, "N" : N, "fs" : arfs}

    plotWFandSpks(N-1, dN, xlst, sTitle=sTitle, sFilename=resFN("generative", dir=setname))

    fig = _plt.figure(figsize=(8, 4))
    _plt.hist(isis, bins=range(100), color="black")
    _plt.grid()
    _plt.savefig(resFN("ISIhist", dir=setname))
    _plt.close()

    fig = _plt.figure(figsize=(13, 4))
    _plt.plot(spksPT, marker=".", color="black", ms=8)
    _plt.ylim(0, max(spksPT)*1.1)
    _plt.grid()
    _plt.suptitle("avg. Hz %.1f" % (_N.mean(spksPT) / (N*0.001)))
    _plt.savefig(resFN("spksPT", dir=setname))
    _plt.close()

    if (lambda2 is None) and (absrefr > 0):
        lambda2 = _N.array([0.0001] * absrefr)
    if lambda2 is not None:
        _N.savetxt(resFN("lambda2.dat", dir=setname), lambda2, fmt="%.7f")

    #  if we want to double bin size
    lambda2db = 0.5*(lambda2[1::2] + lambda2[::2])
    _N.savetxt(resFN("lambda2db.dat", dir=setname), lambda2db, fmt="%.7f")
    #_plt.ion()

    if lowQpc > 0:
        _N.savetxt(resFN("lowQtrials", dir=setname), lowQs, fmt="%d")
