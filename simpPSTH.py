from kassdirs import resFN, datFN

import pickle as _pk
import warnings
import numpy.polynomial.polynomial as _Npp
import utilities as _U

TR = 200
dt = 0.001
setname="simpPSTH2"   #  params_XXX.py   Results/XXX/params.py

N     = 1000

#  x, prbs, spks    3 columns
alldat= _N.empty((N, TR*3))
us    = _N.empty(TR)
dNs   = _N.empty(TR)

#ps    = _N.empty(N)  
ps    = (30 + 22*_N.sin(2*_N.pi*_N.linspace(0, 1, N)))*dt
#ps[0:N-600] = 15*dt
#ps[N-600:N] = 15*dt + _N.linspace(0, 10, 600)*dt
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
