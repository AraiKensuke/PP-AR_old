from kassdirs import resFN
import scipy.signal as _ssig
import re as _re
from filter import lpFilt, bpFilt, base_q4atan
import numpy as _N
import matplotlib.pyplot as _plt
import random as _ran
import myColors as mC

#  [3.3, 11, 1, 15]
def modhist(setname, shftPhase=0, fltPrms=[3.3, 11, 1, 15], t0=None, t1=None, tr0=0, tr1=None, trials=None, fn=None, maxY=None, yticks=None):
    """
    shftPhase from 0 to 1.  
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
    sub  = 2

    if m == None:
        bRealDat = False
        COLS = 3
        sub  = 1

    TR   = cols / COLS
    if trials is None:
        if tr1 is None:
            tr1 = TR
        trials = _N.arange(tr0, tr1)

    wfs  = []
    phs  = []

    for tr in trials:
        x   = dat[:, tr*COLS]
        if len(fltPrms) == 2:
            fx = lpFilt(fltPrms[0], fltPrms[1], 500, x)
        else: # 20, 40, 10, 55    #(fpL, fpH, fsL, fsH, nyqf, y):
            fx = bpFilt(fltPrms[0], fltPrms[1], fltPrms[2], fltPrms[3], 500, x)

        ht_x  = _ssig.hilbert(fx)
        #ph_x  = _N.empty(N)
        ph_x  = (_N.arctan2(ht_x.imag, ht_x.real) + _N.pi) / (2*_N.pi)
        ph_x  = _N.mod(ph_x + shftPhase, 1)
        #for n in xrange(N):
        #ph_x[n] = base_q4atan(ht_x[n].real, ht_x[n].imag) / (2*_N.pi)
        #        ph_x[n] = base_q4atan(ht_x[n].real, ht_x[n].imag) / (2*_N.pi)

        for i in xrange(50, N - 50):
            if dat[i, tr*COLS+(COLS-sub)] == 1:
                wfs.append(ph_x[i-50:i+50])
                phs.append(ph_x[i])

    vphs = _N.array(phs)
    vphs *= 2*_N.pi
    Nspks = len(vphs)
    R2  = 1./(Nspks*Nspks) * (_N.sum(_N.cos(vphs))**2 + _N.sum(_N.sin(vphs))**2)

    wfsv= _N.array(wfs)
    sta = _N.mean(wfsv, axis=0)

    bgFnt = 22
    smFnt = 20
    fig, ax = _plt.subplots(figsize=(6, 4.2))
    _plt.hist(phs + (_N.array(phs) + 1).tolist(), bins=_N.linspace(0, 2, 51), color=mC.hist1, edgecolor=mC.hist1)
    if maxY is not None:
        _plt.ylim(0, maxY)
    #_plt.title("R = %.3f" % _N.sqrt(R2), fontsize=smFnt)
    _plt.xlabel("phase", fontsize=bgFnt)
    _plt.ylabel("frequency", fontsize=bgFnt)
    _plt.xticks(fontsize=smFnt)
    _plt.yticks(fontsize=smFnt)
    if yticks is not None:
        _plt.yticks(yticks)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False


    fig.subplots_adjust(left=0.17, bottom=0.17, right=0.95, top=0.9)

    if fn is None:
        fn = "modulationHistogram,R=%.3f.eps" % _N.sqrt(R2)
    else:
        fn = "%(1)s,R=%(2).3f.eps" % {"1" : fn, "2" : _N.sqrt(R2)}
    
    _plt.savefig(resFN(fn, dir=setname), transparent=True)
    _plt.close()


def oscPer(setname, fltPrms=[3.3, 11, 1, 15], t0=None, t1=None, tr0=0, tr1=None, trials=None, fn=None):
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
    sub  = 2

    if m == None:
        bRealDat = False
        COLS = 3
        sub  = 1

    TR   = cols / COLS
    if trials is None:
        if tr1 is None:
            tr1 = TR
        trials = _N.arange(tr0, tr1)

    Ts = []
    for tr in trials:
        x   = dat[:, tr*COLS]

        if fltPrms is None:
            fx = x
        elif len(fltPrms) == 2:
            fx = lpFilt(fltPrms[0], fltPrms[1], 500, x)
        else: # 20, 40, 10, 55    #(fpL, fpH, fsL, fsH, nyqf, y):
            fx = bpFilt(fltPrms[0], fltPrms[1], fltPrms[2], fltPrms[3], 500, x)

        intvs = _N.where((fx[1:] < 0) & (fx[0:-1] >= 0))

        Ts.extend(_N.diff(intvs[0]))

    _plt.hist(Ts, bins=range(min(Ts) - 1, max(Ts)+1))
    print _N.mean(Ts)
    print _N.std(Ts)


def modhistFltrd(setname, shftPhase=0, t0=None, t1=None, tr0=0, tr1=None, trials=None, surrogates=1):
    """
    shftPhase from 0 to 1.  
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

    COLS = 4
    sub  = 2
    phC  = 3

    TR   = cols / COLS
    if trials is None:
        if tr1 is None:
            tr1 = TR
        trials = _N.arange(tr0, tr1)

    R2s  = _N.empty(surrogates)

    ctrials = trials
    shf_phs = []
    for sur in xrange(surrogates):
        phs  = []
        if sur == 0:
            ctrials = trials
        if sur > 1:
            inds = _N.array(_N.floor(_N.random.rand(len(trials))*len(trials)), dtype=_N.int)
            ctrials = trials[inds]
        for tr in ctrials:
            ispks  = _N.where(dat[:, tr*COLS+(COLS-sub)] == 1)
            phs.extend(dat[ispks[0], tr*COLS + phC])
            
            #for i in xrange(N):
            #if dat[i, tr*COLS+(COLS-sub)] == 1:
            #        phs.append(dat[i, tr*COLS + phC])

        vphs = _N.array(phs)
        vphs *= 2*_N.pi
        Nspks = len(vphs)
        R2s[sur]  = 1./(Nspks*Nspks) * (_N.sum(_N.cos(vphs))**2 + _N.sum(_N.sin(vphs))**2)
        shf_phs.append(phs)

    phs = shf_phs[0]
    fig, ax = _plt.subplots(figsize=(6, 4.2))
    _plt.hist(phs + (_N.array(phs) + 1).tolist(), bins=_N.linspace(0, 2, 51), color="black")
    #_plt.title("R = %.3f" % _N.sqrt(R2s[0]))
    _plt.savefig(resFN("modFltrdHistogram,R=%.3f.eps" % _N.sqrt(R2s[0]), dir=setname))
    _plt.close()
    return R2s


def modhistFltrdSearch(setname, t0=None, t1=None, tr0=0, tr1=None, minGrpSz=None, surrogates=1):
    """
    shftPhase from 0 to 1.  
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

    COLS = 4
    sub  = 2
    phC  = 3

    TR   = cols / COLS

    R2s  = _N.empty((surrogates, 2))
    grps  = []

    allTrials = range(tr0-tr0, tr1-tr0)
    nTRs      = tr1 - tr0
    rands = _N.random.rand(surrogates)

    spkPhs = []
    cSpkPhs= []
    sSpkPhs= []
    for tr in allTrials:
        ispks  = _N.where(dat[:, (tr+tr0)*COLS+(COLS-sub)] == 1)
        cSpkPhs.append(_N.cos(2*_N.pi*dat[ispks[0], (tr+tr0)*COLS + phC]))
        sSpkPhs.append(_N.sin(2*_N.pi*dat[ispks[0], (tr+tr0)*COLS + phC]))

    for sur in xrange(surrogates):
        k    = minGrpSz + int((nTRs - 2*minGrpSz)*rands[sur])
        grp1 = _N.array(_ran.sample(allTrials, k))       #grp1 is a list
        grp2 = _N.setdiff1d(allTrials, grp1)

        grps.append([grp1, grp2])
        g   = 0
        for grp in [grp1, grp2]:
            cphs  = []
            sphs  = []

            for tr in grp:
                cphs.extend(cSpkPhs[tr])
                sphs.extend(sSpkPhs[tr])
            vcphs = _N.array(cphs)
            vsphs = _N.array(sphs)

            Nspks = len(vcphs)
            R2   = 1./(Nspks*Nspks) * (_N.sum(vcphs)**2 + _N.sum(vsphs)**2)
            R2s[sur, g] = R2
            g += 1

    return grps, _N.sqrt(R2s)

def modhistFltrdShuffle(setname, _grp1, _grp2, t0=None, t1=None, tr0=0, tr1=None, surrogates=1, maxSwitch=10, maxMigr=10):
    """
    maxSwitch    symmetric move of elements between two groups
    maxMigr      one-way migration.  One of groups becomes smaller, one becomes larger
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

    COLS = 4
    sub  = 2
    phC  = 3

    TR   = cols / COLS

    R2s  = _N.empty((surrogates, 2))
    grps  = []

    allTrials = range(tr0-tr0, tr1-tr0)
    nTRs      = tr1 - tr0
    rands = _N.random.rand(surrogates)

    spkPhs = []
    cSpkPhs= []
    sSpkPhs= []
    for tr in allTrials:
        ispks  = _N.where(dat[:, (tr+tr0)*COLS+(COLS-sub)] == 1)
        cSpkPhs.append(_N.cos(2*_N.pi*dat[ispks[0], (tr+tr0)*COLS + phC]))
        sSpkPhs.append(_N.sin(2*_N.pi*dat[ispks[0], (tr+tr0)*COLS + phC]))

    for sur in xrange(surrogates):
        grp1 = list(_grp1)
        grp2 = list(_grp2)
        _N.random.shuffle(grp1)
        _N.random.shuffle(grp2)

        sw   = int(_N.random.rand()*maxSwitch)
        mr12 = int(_N.random.rand()*maxMigr)
        mr21 = int(_N.random.rand()*maxMigr)
        
        l1   = len(grp1)
        l2   = len(grp2)
        for s in xrange(sw):
            #  generate sw numbers
            tmp = grp1[s]
            grp1[s] = grp2[s]
            grp2[s] = tmp

        for s in xrange(mr12):
            #  lengthen 2
            grp2.append(grp1.pop())
        grp2.reverse()
        for s in xrange(mr21):
            #  lengthen 2
            grp1.append(grp2.pop())

        grps.append([grp1, grp2])
        g   = 0
        for grp in [grp1, grp2]:
            cphs  = []
            sphs  = []

            for tr in grp:
                cphs.extend(cSpkPhs[tr])
                sphs.extend(sSpkPhs[tr])
            vcphs = _N.array(cphs)
            vsphs = _N.array(sphs)

            Nspks = len(vcphs)
            R2   = 1./(Nspks*Nspks) * (_N.sum(vcphs)**2 + _N.sum(vsphs)**2)
            R2s[sur, g] = R2
            g += 1

    return grps, _N.sqrt(R2s)
