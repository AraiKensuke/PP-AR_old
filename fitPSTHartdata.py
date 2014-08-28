##  DISPLAY fit results for artificial data
import matplotlib.pyplot as _plt
import numpy as _N
from utildirs import setFN

def display(ebf, solx, nbs1, nbs2, nbs2v, B, Gm, aSi, phiSi, psthgen, l2gen, fitpsthgen, fitl2gen, L0=None, L1=None, mPSTH=1.3, mHist=1.3, T0=None):
    #LTR= h_L(aST, phiST, M, B, Gm, sts, itvs, TM, dt, frstSpk)   #  TRUE
    # print "%(TR).1f   %(0).1f   %(1).1f" % {"TR" : LTR, "0" : L0, "1" : L1}

    fig = _plt.figure(figsize=(5, 2*4))
    #####
    fig.add_subplot(2, 1, 1)
    if (L0 != None) and (L1 != None):
        _plt.suptitle("L0: %(l0).1f      L1: %(l1).1f" % {"l1" : L1, "l0" : L0})
    #  PSTH used to generate data
    _plt.plot(psthgen, lw=2, color="red")
    _plt.plot(fitpsthgen, lw=2, color="red", ls="--")
    #  Calculated fit
    _plt.plot(_N.exp(_N.dot(B.T, solx[0:nbs1])), lw=2, color="black")
    #  fit to initial naive PSTH guess
    _plt.plot(_N.exp(_N.dot(B.T, aSi)), lw=2, color="blue")
    _plt.grid()
    _plt.ylim(0, mPSTH*max(_N.exp(_N.dot(B.T, solx[0:nbs1]))))
    #_plt.ylim(0, 1.1*max(apsth))
    #####

    fig.add_subplot(2, 1, 2)

    # l2 used to generate data    
    _plt.plot(l2gen, lw="2", color="red")
    _plt.plot(fitl2gen, lw="2", color="red", ls="--")
    # calculated fit
    _plt.plot(_N.exp(_N.dot(Gm.T, solx[nbs1:])), color="black", lw=2)
    # initial guess
    _plt.plot(_N.exp(_N.dot(Gm.T, phiSi)), lw="2", color="blue")

    _plt.grid()
    if T0 == None:
        _plt.xlim(0, 200)
    else:
        _plt.xlim(0, T0)
    _plt.ylim(0, mHist)
    if (L0 != None) and (L1 != None):
        fig.suptitle("init L %(i).1f   final L %(f).1f" % {"i" : L0, "f" : L1})
    _plt.savefig(setFN("FIT.png", dir=ebf, create=True), background="transparent")
    _plt.close()
