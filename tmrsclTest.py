import numpy as _N
from kassdirs import resFN

def timeRescaleTest(fr, spkts, dt):
    """
    t in units of 1.
    """
    N  = len(spkts)
    rt = _N.empty(N)

    for i in xrange(N):
#        rt[i] = _N.trapz(fr[0:spkts[i]]+1)*dt
        rt[i] = _N.trapz(fr[0:spkts[i]]+1)*dt

    #  this means that small ISIs are underrepresented
    taus = _N.diff(rt)
    zs   = 1 - _N.exp(-taus)

    zss  = _N.sort(zs)
    Nm1  = N - 1        #  same as length - 1
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
    frm = _N.empty(len(fr)*m)
    mspkts = _N.empty(len(spkts))

    lt  = -1
    
    x   = _N.linspace(0, 1, len(fr)*m, endpoint=False)

    for i in xrange(len(spkts)):
        frm[(lt+1)*m:(spkts[i]+1)*m] = _N.interp(x[(lt+1)*m:(spkts[i]+1)*m],
                                                 x[(lt+1)*m:(spkts[i]+1)*m:m],
                                                 fr[lt+1:spkts[i]+1])
        #  somewhere in [spkts[i]*m:(spkts[i]+1)*m]
        mspkts[i] = spkts[i]*m + int(_N.random.rand()*m)
        frm[mspkts[i]+1:(spkts[i]+1)*m] = 0.000001
        lt = spkts[i]
        
    return frm, mspkts
