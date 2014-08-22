##  DISPLAY fit results for artificial data
import matplotlib.pyplot as _plt
import numpy as _N
from utildirs import setFN

def display(ebf, sol, nbs1, nbs2, TM, B, Gm, aSi, phiSi, psthgen, l2gen, fitpsthgen, fitl2gen):
    # LTR= h_L(aST, phiST, M, B, Gm, sts, itvs, TM, dt, frstSpk)   #  TRUE
    # L0 = h_L(aS, phiS, M, B, Gm, sts, itvs, TM, dt, frstSpk)
    # L1 = h_L(sol.x[0:nbs1], sol.x[nbs1:nbs1+nbs2], M, B, Gm, sts, itvs, TM, dt, frstSpk)
    # print "%(TR).1f   %(0).1f   %(1).1f" % {"TR" : LTR, "0" : L0, "1" : L1}

    fig = _plt.figure(figsize=(5, 2*4))
    #####
    fig.add_subplot(2, 1, 1)
    #  PSTH used to generate data
    _plt.plot(psthgen, lw=2, color="red")
    _plt.plot(fitpsthgen, lw=2, color="red", ls="--")
    #  Calculated fit
    _plt.plot(_N.exp(_N.dot(B.T, sol.x[0:nbs1])), lw=2, color="black")
    #  fit to initial naive PSTH guess
    _plt.plot(_N.exp(_N.dot(B.T, aSi)), lw=2, color="blue")
    _plt.grid()
    _plt.ylim(0, 1.1*max(_N.exp(_N.dot(B.T, sol.x[0:nbs1]))))
    #_plt.ylim(0, 1.1*max(apsth))
    #####

    fig.add_subplot(2, 1, 2)

    # l2 used to generate data    
    _plt.plot(range(1, TM+1), l2gen[0:TM], lw="2", color="red")
    _plt.plot(range(1, TM+1), fitl2gen[0:TM], lw="2", color="red", ls="--")
    # calculated fit
    _plt.plot(range(1, TM+1), _N.exp(_N.dot(Gm.T[0:TM], sol.x[nbs1:])), color="black", lw=2)
    # initial guess
    _plt.plot(range(1, TM+1), _N.exp(_N.dot(Gm.T[0:TM], phiSi)), lw="2", color="blue")

    _plt.grid()
    _plt.xlim(0, TM)
    _plt.ylim(0, 1.2)
#    fig.suptitle("init L %(i).1f   final L %(f).1f" % {"i" : L0, "f" : L1})
    _plt.savefig(setFN("FIT.png", dir=ebf, create=True), background="transparent")
    _plt.close()

    return sol
    #  let's pickle the results
    
    #  create fitted firing rate
    #  now create
