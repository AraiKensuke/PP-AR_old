exf("kflib.py")
from kassdirs import resFN, datFN

import pickle as _pk
import warnings
import numpy.polynomial.polynomial as _Npp
import utilities as _U

TR = 10
dt = 0.001
setname="simpPSTH1"   #  params_XXX.py   Results/XXX/params.py

N     = 1000

#  x, prbs, spks    3 columns
alldat= _N.empty((N, TR*3))
us    = _N.empty(TR)
dNs   = _N.empty(TR)

ps    = _N.empty(N)  
ps[0:N-600] = 5*dt
ps[N-600:N] = 5*dt + _N.linspace(0, 5, 600)*dt
dN    = _N.empty(N)

for tr in xrange(TR):
    dN[:] = 0
    for t in xrange(N):
        if _N.random.rand() < ps[t]:
            dN[t] = 1
    dNs[tr] = _N.sum(dN)
    alldat[:, 3*tr] = 0
    alldat[:, 3*tr+1] = ps
    alldat[:, 3*tr+2] = dN

fmt = "%.1e %.3f %d " * TR

_N.savetxt(resFN("xprbsdN.dat", dir=setname, create=True), alldat, fmt=fmt)
