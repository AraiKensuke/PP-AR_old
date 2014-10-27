import numpy as _N
from kassdirs import resFN

def timeRescaleTest(fr, spkts, dt):
    """
    t in units of 1.
    """
    if len(fr.shape) == 1:
        T  = fr.shape[0]
        M = 1
        fr.reshape(T, 1)
    else:
        M = fr.shape[0]
        T  = fr.shape[1]

    zs = []
    Nm1 = 0
    for tr in xrange(M):
        N  = len(spkts[tr])
        rt = _N.empty(N)    #  rescaled time
        for i in xrange(N):
            rt[i] = _N.trapz(fr[tr, 0:spkts[tr][i]]+1)*dt

        #  this means that small ISIs are underrepresented
        taus = _N.diff(rt)
        zs.extend((1 - _N.exp(-taus)).tolist())
        Nm1 += N - 1

    zss  = _N.sort(_N.array(zs))
    ks  = _N.arange(1, Nm1 + 1)
    bs  = (ks - 0.5) / Nm1         #  needs
    bsp = bs + 1.36 / _N.sqrt(Nm1)
    bsn = bs - 1.36 / _N.sqrt(Nm1)
    x   = _N.linspace(1./Nm1, 1, Nm1)
    return x, zss, bs, bsp, bsn

def zoom(fr, spkts, m):
    """
    fr = [  29,  30,  0]
    sp = [   0,   1,  0]   spike occurs at last point fr is high
    m   multiply time by
    """
    if len(fr.shape) == 1:
        T  = fr.shape[0]
        M = 1
        fr.reshape(T, 1)
    else:
        M = fr.shape[0]
        T  = fr.shape[1]

    frm = _N.empty((M, T*m))
    x   = _N.linspace(0, 1, T*m, endpoint=False)
    Lmspkts = []

    for tr in xrange(M):
        Lmspkts.append(_N.empty(len(spkts[tr]), dtype=_N.int))

        lt  = -1
        for i in xrange(len(spkts[tr])):
            sti = spkts[tr][i]
            frm[tr, (lt+1)*m:(sti+1)*m] = _N.interp(x[(lt+1)*m:(sti+1)*m],
                                                    x[(lt+1)*m:(sti+1)*m:m],
                                                    fr[tr, lt+1:sti+1])
            #  somewhere in [spkts[i]*m:(spkts[i]+1)*m]
            Lmspkts[tr][i] = sti*m + int(_N.random.rand()*m)
            frm[tr, Lmspkts[tr][i]+1:(sti+1)*m] = 0.000001
            lt = sti
        
    return frm, Lmspkts
