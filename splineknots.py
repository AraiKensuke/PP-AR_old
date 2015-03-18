import patsy
import statsmodels.api as _sma
import scipy.io as _sio
import utilities as _U
import numpy as _N
import matplotlib.pyplot as _plt
from utildirs import setFN

v = 5
c = 5
def genKnts(tscl, xMax):
    """
    generate a set of knots for history term.
    """
    knts = _N.empty(6)
    ck   = 0   #  current knot
    TSCL = int(1.5*tscl)
    knts[0:3] = TSCL *_N.random.rand(3)
    knts[3:]  = TSCL + (xMax - TSCL)*_N.random.rand(3)
    return _N.sort(knts)

def hazzard(dt, TR, N, bindat, tsclPct=0.85):
    ####  Suggest knots for history term
    isis    = _U.fromBinDat(bindat, ISIs=True)
    ecdf    = _sma.distributions.ECDF(isis)
    xs      = _N.arange(0, max(isis))        #  in units of ms.
    cdf     = ecdf(_N.arange(0, max(isis)))  # value of cdf from [0, 1)
    tscl    = _N.where(cdf > tsclPct)[0][0]
    dt      = 0.001

    S  = 1 - cdf

    haz = -(_N.diff(S) / (0.5*(S[0:-1]+S[1:])))/dt   #  Units of Hz
    #  defined on xs[0:-1]
    haz[_N.where(haz == 0)[0]] = 0.0001    #  need this because log
    #  rough frequency
    #nSpks = len(isis) + TR
    #Hz = float(nSpks) / (TR*N*dt)
    nhaz = haz / _N.mean(haz[int(1.5*tscl):int(2.5*tscl)])  #  

    #nhaz = haz / Hz
    return xs, nhaz, tscl


def suggestHistKnots(dt, TR, N, bindat, tsclPct=0.85, outfn="fittedL2.dat"):
    global v, c
    xs, nhaz, tscl = hazzard(dt, TR, N, bindat, tsclPct=tsclPct)

    ITERS   = 1000
    allKnts = _N.empty((ITERS, 6))
    r2s    = _N.empty(ITERS)

    ac = _N.zeros(c)
    for tr in xrange(ITERS):
        bGood = False
        while not bGood:
            knts = genKnts(tscl, xs[-1]*0.9)

            B  = patsy.bs(xs[0:-1], knots=knts, include_intercept=True)

            Bc = B[:, v:];   Bv = B[:, 0:v]

            try:
                iBvTBv = _N.linalg.inv(_N.dot(Bv.T, Bv))
                bGood = True
            except _N.linalg.linalg.LinAlgError:
                print "Linalg Error"

        av = _N.dot(iBvTBv, _N.dot(Bv.T, _N.log(nhaz) - _N.dot(Bc, ac)))
        a = _N.array(av.tolist() + ac.tolist())

        #  Now fit where the last few nots are fixed
        splFt          = _N.exp(_N.dot(B, a))
        df             = nhaz - splFt
        r2s[tr]        = _N.dot(df[0:int(tscl)], df[0:int(tscl)])
        allKnts[tr, :] = knts

    bstKnts = allKnts[_N.where(r2s == r2s.min())[0][0], :]

    B  = patsy.bs(xs[0:-1], knots=bstKnts, include_intercept=True)
    Bc = B[:, v:];   Bv = B[:, 0:v]
    iBvTBv = _N.linalg.inv(_N.dot(Bv.T, Bv))
    av = _N.dot(iBvTBv, _N.dot(Bv.T, _N.log(nhaz) - _N.dot(Bc, ac)))
    a = _N.array(av.tolist() + ac.tolist())
    #  Now fit where the last few nots are fixed
    lmd2          = _N.exp(_N.dot(B, a))

    return bstKnts, lmd2, nhaz, tscl

def suggestPSTHKnots(dt, TR, N, bindat, bnsz=50, iknts=2):
    """
    bnsz   binsize used to calculate approximate PSTH
    """
    spkts  = _U.fromBinDat(bindat, SpkTs=True)

    h, bs = _N.histogram(spkts, bins=_N.linspace(0, N, (N/bnsz)+1))
    
    fs     = (h / (TR * bnsz * dt))
    apsth = _N.repeat(fs, bnsz)    #    piecewise boxy approximate PSTH

    ITERS = 1000
    x     = _N.linspace(0., N-1, N, endpoint=False)  # in units of ms.
    r2s   = _N.empty(ITERS)
    allKnts = _N.empty((ITERS, iknts))

    for it in xrange(ITERS):
        bGood = False
        while not bGood:
            try:
                knts  = _N.sort((0.05 + 0.9*_N.random.rand(iknts))*N)
                B     = patsy.bs(x, knots=knts, include_intercept=True)
                iBTB   = _N.linalg.inv(_N.dot(B.T, B))
                bGood  = True
            except _N.linalg.linalg.LinAlgError, ValueError:
                print "Linalg Error or Value Error"

        a     = _N.dot(iBTB, _N.dot(B.T, _N.log(apsth)))
        ft    = _N.exp(_N.dot(B, a))
        r2s[it] = _N.dot(ft - apsth, ft - apsth)
        allKnts[it, :] = knts

    mnIt = _N.where(r2s == r2s.min())[0][0]
    knts = allKnts[mnIt]

    return knts, apsth

def display(N, dt, tscl, nhaz, apsth, lambda2, psth, histknts, psthknts, dir=None):
    """
    N        length of trial, also time in ms
    tscl
    nhaz     normalized hazzard function.  calculated under assumption of stationarity of psth
    apsth    approximate stepwise psth
    lambda2  ground truth   lambda2 term
    psth     ground truth   lambda1 term
    """
    global v, c

    x  = _N.linspace(0., N-1, N, endpoint=False)  # in units of ms.

    theknts = [histknts, psthknts]
    for f in xrange(1, 3):
        knts = theknts[f-1]
        
        if f == 1:
            fig = _plt.figure()
            B  = patsy.bs(x[0:len(nhaz)], knots=knts, include_intercept=True)
            Bc = B[:, v:];    Bv = B[:, 0:v]
            ac = _N.zeros(c)
            iBvTBv = _N.linalg.inv(_N.dot(Bv.T, Bv))

            av = _N.dot(iBvTBv, _N.dot(Bv.T, _N.log(nhaz) - _N.dot(Bc, ac)))

            a = _N.array(av.tolist() + ac.tolist())
            _plt.plot(x[0:len(nhaz)], nhaz, color="grey", lw=2)   #  empirical
            ymax = -1
            if lambda2 is not None:
                _plt.plot(lambda2, color="red", lw=2)               #  ground truth
                ymax = max(lambda2)
            _plt.ylim(0, max(ymax, max(nhaz[0:tscl]))*1.1)
            _plt.xlim(0, 3*tscl)
            splFt = _N.exp(_N.dot(B, a))
            _plt.plot(splFt)
            _plt.savefig(setFN("hist.eps", dir=dir))
            _plt.xlim(0, tscl)
            _plt.grid()
            _plt.savefig(setFN("histZ.eps", dir=dir))
            _plt.close()
        else:
            fig = _plt.figure()
            B  = patsy.bs(x, knots=knts, include_intercept=True)
            iBTB   = _N.linalg.inv(_N.dot(B.T, B))
            a     = _N.dot(iBTB, _N.dot(B.T, _N.log(apsth)))
            _plt.plot(x, apsth, color="grey", lw=2)   #  empirical
            if psth is not None:
                fHz   = ((_N.exp(psth)*dt) / (1 + dt*_N.exp(psth))) / dt
                _plt.plot(fHz, color="red", lw=2)               #  ground truth

            splFt = _N.exp(_N.dot(B, a))
            _plt.plot(splFt)
            _plt.savefig(setFN("psth.eps", dir=dir))
            _plt.close()

"""
def reasonableHist(lmd, maxH=1.2):
    L = lmd.shape[0]
    cmpLmd = _N.array(lmd)  #  compressed lambda

    maxAmp = _N.max(lmd-1)
    bAbv    = False
    bFellOnce=False
    bForce1 = False   #  force it to be 1 hereafter
    
    for i in xrange(L):
        if (not bAbv) and lmd[i] > 1:
            bAbv = True
            
        if (lmd[i] > 1):
            cmpLmd[i] = 1 + ((lmd[i] - 1) / maxAmp) * (maxH - 1)
        elif (lmd[i] < 1) and bAbv:
            cmpLmd[i] = 1

        if bAbv and (not bFellOnce):
            if lmd[i] < lmd[i-1]:
                bFellOnce = True

        if (not bForce1) and bAbv and (lmd[i] > lmd[i-1]) and bFellOnce:
            bForce1 = True
            cmpLmd[i] = 1
        if bForce1:
            cmpLmd[i] = 1

            
    return cmpLmd
"""

def reasonableHistory(lmd, maxH=1.2, cutoff=100):
    """
    search for max between 0 and cutoff
    """
    L = lmd.shape[0]
    cmpLmd = _N.array(lmd)  #  compressed lambda

    maxAmp = _N.max(lmd-1)
    bAbv    = False
    bFellOnce=False
    bForce1 = False   #  force it to be 1 hereafter
    
    for i in xrange(L):
        if (not bAbv) and lmd[i] > 1:
            bAbv = True
            
        if (lmd[i] > 1):
            cmpLmd[i] = 1 + ((lmd[i] - 1) / maxAmp) * (maxH - 1)
        elif (lmd[i] < 1) and bAbv:
            cmpLmd[i] = 1

        if bAbv and (not bFellOnce):
            if lmd[i] < lmd[i-1]:
                bFellOnce = True

        if (not bForce1) and bAbv and (lmd[i] > lmd[i-1]) and bFellOnce:
            bForce1 = True
            cmpLmd[i] = 1
        if bForce1:
            cmpLmd[i] = 1

            
    return cmpLmd

def reasonableHistory2(lmd, maxH=1.2, strT=1, cutoff=100, dcyTS=60):
    """
    search for max between 0 and cutoff
    stretchT
    """
    hiest = max(lmd[0:cutoff])
    L = lmd.shape[0]
    cmpLmd = _N.empty(dcyTS)  #  compressed lambda

    ihiest= _N.where(lmd == hiest)[0][0]

    ###  
    x    = _N.linspace(0, ihiest, ihiest+1)

    for i in xrange(ihiest + 1):
        cmpLmd[i] = lmd[i] * (maxH / hiest)
    
    print ihiest
    if strT > 1:

        nIDP   = int((ihiest+1)*strT)   #  number of interpolated data points
        xI     = _N.linspace(0, ihiest, nIDP)
        cI     = _N.interp(xI, x, cmpLmd[0:ihiest+1])

        cmpLmd[0:nIDP] = cI
        ihiest = int(ihiest*strT)

    dy  = (maxH - 1) / float(dcyTS - ihiest)
    for i in xrange(ihiest + 1, dcyTS):
        cmpLmd[i] = maxH -  (i - ihiest) * dy

    return cmpLmd
