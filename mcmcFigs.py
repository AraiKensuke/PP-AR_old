import numpy as _N
import scipy.signal as _ssig
import scipy.stats as _ss
from ARcfSmplFuncs import dcmpcff
import matplotlib.pyplot as _plt
from filter import gauKer
import scipy.signal as _ssig

def showFsHist(mARp, low, hi, nbins=100):
    t0 = mARp.burn
    t1 = mARp.burn + mARp.NMC
    _plt.hist(mARp.fs[t0:t1, 0], bins=_N.linspace(low, hi, nbins))
    _plt.grid()

def filterByFsAmps(mARp, flow, fhi, alow=0.9, ahigh=1):
    t0 = mARp.burn
    t1 = mARp.burn + mARp.NMC
    inds = _N.where((mARp.fs[t0:t1, 0] > flow) & (mARp.fs[t0:t1, 0] < fhi) &
                    (mARp.amps[t0:t1, 0] > alow) & (mARp.amps[t0:t1, 0] < ahigh))
    return inds

def histPhase0_phaseInfrd(mARp, _mdn, t0=None, t1=None, bRealDat=False, trials=None):
    #  what is the inferred phase when ground truth phase is 0
    fig = _plt.figure()
    pInfrdAt0 = []
    _fx  = mARp.x
    if bRealDat:
        _fx = mARp.fx
    gk  = gauKer(1) 
    gk /= _N.sum(gk)

    if trials is None:
        trials = range(mARp.TR)
        TR     = mARp.TR
    else:
        TR     = len(trials)
    nPh0s = _N.zeros(TR)
    t1    = t1-t0   #  mdn already size t1-t0
    t0    = 0

    mdn = _mdn
    fx  = _fx
    if _mdn.shape[0] != t1 - t0:
        mdn = _mdn[:, t0:t1]
    if _fx.shape[0] != t1 - t0:
        fx = _fx[:, t0:t1]

    itr   = 0

    for tr in trials:
        itr += 1
        cv = _N.convolve(mdn[tr, t0:t1] - _N.mean(mdn[tr, t0:t1]), gk, mode="same")
#        #cv = _N.convolve(mdn[tr, t0:t1] - _N.mean(mdn[tr, t0:t1]), gk, mode="same")

        #ht_mdn  = _ssig.hilbert(mdn[tr, t0:t1] - _N.mean(mdn[tr, t0:t1]))
        ht_mdn  = _ssig.hilbert(cv)
        ht_fx   = _ssig.hilbert(fx[tr, t0:t1] - _N.mean(fx[tr, t0:t1]))
        ph_mdn  = _N.arctan2(ht_mdn.imag, ht_mdn.real) / _N.pi
        ph_fx   = _N.arctan2(ht_fx.imag, ht_fx.real)   / _N.pi
        #  phase = 0 is somewhere in middle
        for i in xrange(t0-t0, t1-t0-1):
            if (ph_mdn[i] < 1) and (ph_mdn[i] > 0.5) and (ph_mdn[i+1] < -0.5):
                nPh0s[itr-1] += 1
                pInfrdAt0.append(ph_fx[i])
                pInfrdAt0.append(ph_fx[i]+2)
    pInfrdAt0A = _N.array(pInfrdAt0[::2])
    Npts = len(pInfrdAt0A)
    R2   = (1./(Npts*Npts)) * (_N.sum(_N.cos(_N.pi*pInfrdAt0A))**2 + _N.sum(_N.sin(_N.pi*pInfrdAt0A))**2)
    _plt.hist(pInfrdAt0, bins=_N.linspace(-1, 3, 41), color="black")
    _plt.suptitle("R  %.3f" % _N.sqrt(R2))
    _plt.savefig("fxPhase1_phaseInfrd.eps")
    _plt.close()
    return pInfrdAt0, _N.sqrt(R2), nPh0s

def getComponents(mARp):
    TR    = mARp.TR
    NMC   = mARp.NMC
    burn  = mARp.burn
    R     = mARp.R
    C     = mARp.C
    ddN   = mARp._d.N

    rt = _N.empty((TR, NMC, ddN+2, R))    #  real components   N = ddN
    zt = _N.empty((TR, NMC, ddN+2, C))    #  imag components 

    for tr in xrange(TR):
        for it in xrange(burn, burn + NMC):
            i    = it - burn
            b, c = dcmpcff(alfa=mARp.allalfas[it])

            for r in xrange(R):
                rt[tr, i, :, r] = b[r] * mARp.uts[tr, it, r, :]

            for z in xrange(C):
                #print "z   %d" % z
                cf1 = 2*c[2*z].real
                gam = mARp.allalfas[it, R+2*z]
                #cf2 = 2*(c[2*z].real*gam.real + c[2*z].imag*gam.imag)
                cf2 = 2*(c[2*z].real*gam.real + c[2*z].imag*gam.imag)
                #print "%(1).3f    %(2).3f" % {"1": cf1, "2" : cf2}
                zt[tr, i, 0:ddN+2, z] = cf1*mARp.wts[tr, it, z, 1:ddN+3] - cf2*mARp.wts[tr, it, z, 0:ddN+2]
                #for n in xrange(1, ddN+3):
                #    zt[tr, i, n-1, z] = cf1*mARp.wts[tr, it, z, n] - cf2*mARp.wts[tr, it, z, n-1]
    return rt, zt


###############################################
def plotWFandSpks(mARp, zts0, sFilename="latnt,GrdTr,Spks", tMult=1, intv=None, sTitle=None, tr=None, cls2EvryO=None, bRealDat=False, norm=False):
    """
    tMult    multiply by time 
    """

    mdn = _N.empty((mARp.TR, mARp.N + 1))

    it0 = mARp.burn
    it1 = mARp.burn+mARp.NMC
    cmp = 0   #  AR component to show

    ITER  = it1-it0
    avgDs  = _N.empty(ITER)
    avgD = _N.empty(ITER)
    df = _N.empty((ITER, mARp.N+1))

    xLFPGT =mARp.x
    if bRealDat:
        xLFPGT =mARp.fx

    for tr in xrange(mARp.TR):
        """
        for it1 in xrange(ITER):
            _N.subtract(zts0[tr, it1], zts0[tr], out=df)  # dim(df) is ITER x N
            _N.sqrt(_N.sum(df*df, axis=1), out=avgD)  #  one term in sum is 0

            avgDs[it1] = _N.sum(avgD) / (ITER-1)

        cls2EvryO = [i[0] for i in sorted(enumerate(avgDs), key=lambda x:x[1])]
        """

        #mdn[tr] = zts0[tr, int(ITER*_N.random.rand())]
        #mdn[tr] = _N.mean(zts0[tr, cls2EvryO[0:30]], axis=0)
        mdn[tr] = _N.mean(zts0[tr, cls2EvryO], axis=0)
        MINx = _N.min(zts0[tr])
        MAXx = _N.max(zts0[tr])

        fig = _plt.figure(figsize=(12, 4), frameon=False)

        AMP  = MAXx - MINx
        ht   = 0.08*AMP
        ys1  = MINx - 0.5*ht
        ys2  = MINx - 3*ht

        for it in xrange(1, len(cls2EvryO), 5):
            _plt.plot(zts0[tr, cls2EvryO[it]], lw=1.5, color="#cccccc")  
        _plt.plot(mdn[tr], lw=2, color="black")
        AMP = 1
        if norm:
            AMP = _N.std(mdn[tr]) / _N.std(xLFPGT[tr])
        AMP = _N.std(mdn[tr]) / _N.std(xLFPGT[tr])
        _plt.plot(xLFPGT[tr]*AMP, color="red", lw=2)

        for n in xrange(mARp.N+1):
            if mARp.y[tr, n] == 1:
                _plt.plot([n, n], [ys1, ys2], lw=2.5, color="black")
        _plt.ylim(ys2 - 0.05*AMP, MAXx + 0.05*AMP)
        _plt.yticks(fontsize=20)

        tcks = _plt.xticks()
        ot   = _N.array(tcks[0], dtype=_N.int)
        mt   = _N.array(tcks[0] * tMult, dtype=_N.int)
        _plt.xticks(ot, mt, fontsize=20)

        #_plt.xticks(_N.linspace(0, mARp.N+1, 7, dtype=_N.int), _N.linspace(0, mARp.N+1, 7, dtype=_N.int)*tMult, fontsize=20)
        _plt.xlim(0, (mARp.N+1)+1)
        _plt.xlabel("millisecond", fontsize=22)
        _plt.axhline(y=0, color="grey", lw=2, ls="--")

        if intv is not None:
            _plt.xlim(intv[0], intv[1])
        if sTitle is not None:
            _plt.title(sTitle)
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.85)
        if sFilename != None:
            _plt.savefig("%(fn)s,tr=%(tr)d.eps" % {"fn" : sFilename, "tr" : tr})
            _plt.close()

    return mdn

def plotPSTH(mARp):
    fig = _plt.figure(figsize=(6, 4))
    meanPSTH = _N.mean(_N.dot(mARp.B.T, mARp.smp_aS[mARp.burn:mARp.burn+mARp.NMC].T), axis=1)
    _plt.plot(meanPSTH, lw=2)
    _plt.savefig("psth-splines")
    _plt.close()
    return meanPSTH

def plotFsAmp(mARp):
    fig = _plt.figure(figsize=(5, 6))
    fig.add_subplot(2, 1, 1)
    _plt.plot(mARp.fs[1:, 0])
    fig.add_subplot(2, 1, 2)
    _plt.plot(mARp.amps[1:, 0])
    _plt.savefig("fs_amps")
    _plt.close()

def corrcoeffs(mARp, mdn, bRealDat=False):
    xLFPGT =mARp.x
    if bRealDat:
        xLFPGT =mARp.fx
    fig = _plt.figure()
    pcs = _N.empty(mARp.TR)
    for tr in xrange(mARp.TR):
        pc, pv = _ss.pearsonr(mdn[tr], xLFPGT[tr])
        pcs[tr] = pc
    _plt.hist(pcs, bins=_N.linspace(-0.5, 0.5, 41), color="black")
    _plt.axvline(x=_N.mean(pcs), ls="--", color="red", lw=1.5)
    _plt.xlim(-0.6, 0.6)
    _plt.suptitle("pc mean %.2f" % _N.mean(pcs))
    _plt.grid()
    _plt.savefig("pcs")
    _plt.close()

def acorrs(mARp, mdn, realDat=False):
    fx = mARp.fx
    if not realDat:
        fx = mARp.x
    _plt.acorr(fx.flatten(), usevlines=False, maxlags=300, color="red", linestyle="-", marker=".", ms=0, lw=2)
    _plt.acorr(mdn.flatten(), usevlines=False, maxlags=300, color="black", linestyle="-", marker=".", ms=0, lw=2)

    _plt.savefig("acorrs")
    _plt.close()

