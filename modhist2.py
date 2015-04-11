from kassdirs import resFN
import scipy.signal as _ssig
import re as _re
from filter import lpFilt, bpFilt, base_q4atan
import numpy as _N
import matplotlib.pyplot as _plt
import random as _ran
import myColors as mC
import itertools as itls

#  [3.3, 11, 1, 15]
def modhistAll(setname, shftPhase=0, haveFiltered=False, fltPrms=[3.3, 11, 1, 15], t0=None, t1=None, tr0=0, tr1=None, trials=None, fn=None, maxY=None, yticks=None, normed=False, surrogates=1):
    """
    shftPhase from 0 to 1.  
    yticks should look like [[0.5, 1, 1.5], ["0.5", "1", "1.5"]]
    """

    _dat     = _N.loadtxt(resFN("xprbsdN.dat", dir=setname))
    #  modulation histogram.  phase @ spike
    Na, cols = _dat.shape

    t0 = 0 if (t0 is None) else t0
    t1 = Na if (t1 is None) else t1

    dat = _dat[t0:t1, :]
    N   = t1-t0

    p = _re.compile("^\d{6}")   # starts like "exptDate-....."
    m = p.match(setname)

    bRealDat, COLS, sub, phC = True, 4, 2, 3
    
    if m == None:
        bRealDat, COLS, sub = False, 3, 1

    TR   = cols / COLS
    tr1 = TR if (tr1 is None) else tr1

    trials = _N.arange(tr0, tr1) if (trials is None) else trials
    if type(trials) == list:
        trials = _N.array(trials)

    phs  = []
    cSpkPhs= []
    sSpkPhs= []

    for tr in trials:
        if fltPrms is not None:
            x   = dat[:, tr*COLS]
            if len(fltPrms) == 2:
                fx = lpFilt(fltPrms[0], fltPrms[1], 500, x)
            elif len(fltPrms) == 4: 
                # 20, 40, 10, 55 #(fpL, fpH, fsL, fsH, nyqf, y):
                fx = bpFilt(fltPrms[0], fltPrms[1], fltPrms[2], fltPrms[3], 500, x)
            ht_x  = _ssig.hilbert(fx)
            ph_x  = (_N.arctan2(ht_x.imag, ht_x.real) + _N.pi) / (2*_N.pi)
            ph_x  = _N.mod(ph_x + shftPhase, 1)
        else:
            ph_x  = dat[:, tr*COLS + phC]
            if tr == 0:
                print ph_x

        ispks  = _N.where(dat[:, tr*COLS+(COLS-sub)] == 1)[0]
        cSpkPhs.append(_N.cos(2*_N.pi*ph_x[ispks]))
        sSpkPhs.append(_N.sin(2*_N.pi*ph_x[ispks]))
        phs.append(ph_x[ispks])

    return figCircularDistribution(phs, cSpkPhs, sSpkPhs, trials, surrogates=surrogates, normed=normed, fn=fn, maxY=maxY, yticks=yticks, setname=setname)

def figCircularDistribution(phs, cSpkPhs, sSpkPhs, trials, setname=None, surrogates=1, normed=False, fn=None, maxY=None, yticks=None):  #  phase histogram
    ltr = len(trials)
    inorderTrials = _N.arange(ltr)   #  original trial IDs no lnger necessary
    R2s = _N.empty(surrogates)
    for srgt in xrange(surrogates):
        if srgt == 0:
            trls = inorderTrials
        else:
            trls = inorderTrials[_N.sort(_N.asarray(_N.random.rand(ltr)*ltr, _N.int))]

        cS = []
        sS = []
        for tr in trls:
            cS.extend(cSpkPhs[tr])
            sS.extend(sSpkPhs[tr])

        Nspks = len(cS)
        vcSpkPhs = _N.array(cS)
        vsSpkPhs = _N.array(sS)

        R2s[srgt]  = _N.sqrt((1./(Nspks*Nspks)) * (_N.sum(vcSpkPhs)**2 + _N.sum(vsSpkPhs)**2))


    vPhs  = _N.fromiter(itls.chain.from_iterable(phs), _N.float)
    bgFnt = 22
    smFnt = 20
    fig, ax = _plt.subplots(figsize=(6, 4.2))
    #_plt.hist(phs + (vPhs + 1).tolist(), bins=_N.linspace(0, 2, 51), color=mC.hist1, edgecolor=mC.hist1, normed=normed)
    _plt.hist(vPhs.tolist() + (vPhs + 1).tolist(), bins=_N.linspace(0, 2, 51), color=mC.hist1, edgecolor=mC.hist1, normed=normed)

    if maxY is not None:
        _plt.ylim(0, maxY)
    elif normed:
        _plt.ylim(0, 1)
    #_plt.title("R = %.3f" % _N.sqrt(R2), fontsize=smFnt)
    _plt.xlabel("phase", fontsize=bgFnt)
    _plt.ylabel("probability", fontsize=bgFnt)
    _plt.xticks(fontsize=smFnt)
    _plt.yticks(fontsize=smFnt)
    if yticks is not None:
        #  plotting 2 periods, probability is halved, so boost it by 2
        _plt.yticks(yticks[0], yticks[1])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False

    fig.subplots_adjust(left=0.17, bottom=0.17, right=0.95, top=0.9)

    if fn is None:
        fn = "modulationHistogram,R=%.3f.eps" % R2s[0]
    else:
        fn = "%(1)s,R=%(2).3f.eps" % {"1" : fn, "2" : R2s[0]}
        
    if setname is not None:
        _plt.savefig(resFN(fn, dir=setname), transparent=True)
    else:
        _plt.savefig(fn, transparent=True)
    _plt.close()
    return R2s

def oscPer(setname, fltPrms=[3.3, 11, 1, 15], t0=None, t1=None, tr0=0, tr1=None, trials=None, fn=None, showHist=True):
    """
    find period of oscillation
    """
    _dat     = _N.loadtxt(resFN("xprbsdN.dat", dir=setname))
    #  modulation histogram.  phase @ spike

    Na, cols = _dat.shape

    if t0 is None:
        t0 = 0
    if t1 is None:
        t1 = Na
    dat = _dat[t0:t1, :]
    N   = t1-t0

    p = _re.compile("^\d{6}")   # starts like "exptDate-....."
    m = p.match(setname)

    bRealDat = True
    COLS = 4
    sub  = 1

    if m == None:
        bRealDat = False
        COLS = 3
        sub  = 0

    TR   = cols / COLS
    if trials is None:
        if tr1 is None:
            tr1 = TR
        trials = _N.arange(tr0, tr1)

    Ts = []
    for tr in trials:
        x   = dat[:, tr*COLS+sub]

        if fltPrms is None:
            fx = x
        elif len(fltPrms) == 2:
            fx = lpFilt(fltPrms[0], fltPrms[1], 500, x)
        else: # 20, 40, 10, 55    #(fpL, fpH, fsL, fsH, nyqf, y):
            fx = bpFilt(fltPrms[0], fltPrms[1], fltPrms[2], fltPrms[3], 500, x)

        intvs = _N.where((fx[1:] < 0) & (fx[0:-1] >= 0))

        Ts.extend(_N.diff(intvs[0]))

    if showHist:
        fig = _plt.figure()
        _plt.hist(Ts, bins=range(min(Ts) - 1, max(Ts)+1))
    mn = _N.mean(Ts)
    std= _N.std(Ts)
    print "mean Hz %(f).3f    cv: %(cv).3f" % {"cv" : (std/mn), "f" : (1000/mn)}
