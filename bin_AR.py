import kflib as _kfl
from shutil import copyfile
from utildirs import setFN
import numpy as _N
import kstat as _ks
from kassdirs import resFN, datFN
import numpy.polynomial.polynomial as _Npp
import matplotlib.pyplot as _plt

TRIALS     = None;
trim       = 50;
rs         = None;     ths      = None;      alfa     = None;
nWins      = 1
model      = None
mH         = 0.3     #  prob of low state as a function of trial. Const or array
rn         = None;   pL       = None;   pH       = None;
alfa       = None; 
upx        = None;   #  a deterministic trend I will enter by hand.  
ctsL       = None;   ctsH     = None;   data     = None;

def ccW12(y, N, blks):
    pcs = []
    for i in xrange(0, N - blks, blks):
        pc, pv = _ss.pearsonr(y[i:i + blks, 0], y[i:i + blks, 1])
        pcs.append(pc)

    return pcs

##  # of cols is 1 + (nStates - 1)                       + 1 + (nWins)
##  data is    x     mH (0.3, 0.6 - ex. of 3 states)   state  counts in win
def create(setname):
    global mH, upx, ctsL, ctsH, data
    copyfile("%s.py" % setname, "%(s)s/%(s)s.py" % {"s" : setname, "to" : setFN("%s.py" % setname, dir=setname, create=True)})
    #  mean rate is n x p.  p determines cv
    #  mean rate is r x p / (1 - p)    .  p prob success
    #  mean is (1-pL)*r/pL for low spike counts
    uL = []
    uH = []

    if type(mH) != _N.ndarray:
        mH = _N.ones(TRIALS) * mH

    for nw in xrange(nWins):
        uL.append(_N.log(pL[nw] / (1 - pL[nw])))
        uH.append(_N.log(pH[nw] / (1 - pH[nw])))

    cts = _N.empty((TRIALS, nWins))
    stcts = [[], []]

    phi          = (-1*_Npp.polyfromroots(alfa)[::-1][1:]).real

    data= _N.empty((TRIALS, 2 + nWins))   #  x, state, spkct x 

    x,y   = _kfl.createDataAR(TRIALS + trim, phi, e, e, trim=trim)
    if upx == None:
        upx   = _N.zeros(TRIALS)

    for tr in xrange(TRIALS):
        st = 0
        uLH= uL
        if _N.random.rand() < mH[tr]:
            st = 1
            uLH= uH

        for nw in xrange(nWins):
            ex = _N.exp(uLH[nw] + (x[tr] + upx[tr]))
            if model == "binomial":
                cts[tr, nw] = _N.random.binomial(rn[nw], ex / (1 + ex))
            else:
                cts[tr, nw] = _N.random.negative_binomial(rn[nw], ex / (1 + ex))
        stcts[st].append(cts[tr, :])
        data[tr, 0:2] = (x[tr] + upx[tr], st)
        data[tr, 2:]  = cts[tr, :]


    ctsL = _N.array(stcts[0])
    ctsH = _N.array(stcts[1])

    fig = _plt.figure(figsize=(4*2, nWins*3.4))
    for nw in xrange(nWins):
        fig.add_subplot(2, nWins, 2*nw+1)
        _plt.hist(cts[:, nw], bins=range(int(max(cts[:, nw])) + 2))

        fig.add_subplot(2, nWins, 2*nw+2)
        pctlL = _ks.percentile(ctsL[:, nw])
        pctlH = _ks.percentile(ctsH[:, nw])
        _plt.plot(pctlL[:, 0], pctlL[:, 1], color="black", lw=2)
        _plt.plot(pctlH[:, 0], pctlH[:, 1], color="red", lw=2)
        _plt.grid()

        cvH = _N.std(ctsH[:, nw])**2/_N.mean(ctsH[:, nw])
        cvL = _N.std(ctsL[:, nw])**2/_N.mean(ctsL[:, nw])
        cv  = _N.std(cts[:, nw])**2 /_N.mean(cts[:, nw])

        print "for win %d" % nw
        print "cvL  %s" % cvL
        print "cvH  %s" % cvH
        print "cv   %s" % cv
    _plt.savefig(resFN("true_hists", dir=setname, create=True))
    _plt.close()

    _N.savetxt(resFN("cnt_data.dat", dir=setname, create=True), data, fmt="%.5f %d %d %d")
    _N.savetxt(resFN("mH.dat", dir=setname, create=True), data, fmt="%.3f")
