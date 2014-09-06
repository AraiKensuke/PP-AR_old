import patsy
import statsmodels.api as _sma
import scipy.io as _sio
import utilities as _U
import numpy as _N
import matplotlib.pyplot as _plt

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
    return bstKnts, nhaz, tscl

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

def display(N, dt, tscl, nhaz, apsth, lambda2, psth, histknts, psthknts, outfn=None):
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

    fig = _plt.figure(figsize=(6, 4*2))

    theknts = [histknts, psthknts]
    for f in xrange(1, 3):
        fig.add_subplot(2, 1, f)
        knts = theknts[f-1]
        
        if f == 1:
            B  = patsy.bs(x[0:len(nhaz)], knots=knts, include_intercept=True)
            Bc = B[:, v:];    Bv = B[:, 0:v]
            ac = _N.zeros(c)
            iBvTBv = _N.linalg.inv(_N.dot(Bv.T, Bv))

            av = _N.dot(iBvTBv, _N.dot(Bv.T, _N.log(nhaz) - _N.dot(Bc, ac)))

            a = _N.array(av.tolist() + ac.tolist())
            _plt.plot(x[0:len(nhaz)], nhaz, color="grey", lw=2)   #  empirical
            _plt.plot(lambda2, color="red", lw=2)               #  ground truth
            _plt.ylim(0, 2.5)
            _plt.xlim(0, 3*tscl)
        else:
            B  = patsy.bs(x, knots=knts, include_intercept=True)
            iBTB   = _N.linalg.inv(_N.dot(B.T, B))
            a     = _N.dot(iBTB, _N.dot(B.T, _N.log(apsth)))
            _plt.plot(x, apsth, color="grey", lw=2)   #  empirical
            fHz   = ((_N.exp(psth)*dt) / (1 + dt*_N.exp(psth))) / dt
            _plt.plot(fHz, color="red", lw=2)               #  ground truth

        splFt = _N.exp(_N.dot(B, a))

        _plt.plot(splFt)

    if outfn != None:
        _plt.savefig(outfn)
        _plt.close()

