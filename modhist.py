from kassdirs import resFN
import scipy.signal as _ssig
import re as _re
from filter import lpFilt, bpFilt, base_q4atan
import numpy as _N
import matplotlib.pyplot as _plt

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
    fig = _plt.figure(figsize=(6, 4.2))
    _plt.hist(phs + (_N.array(phs) + 1).tolist(), bins=_N.linspace(0, 2, 51), color="black")
    if maxY is not None:
        _plt.ylim(0, maxY)
    _plt.title("R = %.3f" % _N.sqrt(R2), fontsize=smFnt)
    _plt.xlabel("phase", fontsize=bgFnt)
    _plt.ylabel("frequency", fontsize=bgFnt)
    _plt.xticks(fontsize=smFnt)
    _plt.yticks(fontsize=smFnt)
    if yticks is not None:
        _plt.yticks(yticks)

    fig.subplots_adjust(left=0.17, bottom=0.17, right=0.95, top=0.9)

    if fn is None:
        fn = "modulationHistogram.eps"
    
    _plt.savefig(resFN(fn, dir=setname), transparent=True)
    _plt.close()



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
    _plt.figure()
    _plt.hist(phs + (_N.array(phs) + 1).tolist(), bins=_N.linspace(0, 2, 51), color="black")
    _plt.title("R = %.3f" % _N.sqrt(R2s[0]))
    _plt.savefig(resFN("modFltrdHistogram.eps", dir=setname))
    _plt.close()
    return R2s

