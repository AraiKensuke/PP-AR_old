exf("kflib.py")

from kassdirs import resFN, datFN

import pickle as _pk
import warnings
import numpy.polynomial.polynomial as _Npp
import utilities as _U

TR = 500
dt = 0.001
setname="histPSTH7"   #  params_XXX.py   Results/XXX/params.py

N     = 1000

#  x, prbs, spks    3 columns
alldat= _N.empty((N, TR*3))
us    = _N.empty(TR)
dNs   = _N.empty(TR)

x     = _N.linspace(0, 1*2, 2*N)
#  ABRUPT shoulder
#l2    = _N.ones(N + 50)
#l2[0:20] = _N.linspace(0., 1, 20)**2
#  SMOOTH shoulder
a = 30.
b = 8.
ms = _N.arange(0, 2*N+50)
l2 = _N.exp((ms-a)/b) / (1 + _N.exp((ms-a)/b))
ps    = (70 + 35*_N.sin(4*_N.pi*x))*dt# +  5*_N.sin(2*3.1*_N.pi*x - 1))*dt
#ps    = 50*_N.ones(N)*dt# + 15*_N.sin(2*_N.pi*x) +  5*_N.sin(2*3.1*_N.pi*x - 1))*dt
#ps    = _N.ones(N)*50*dt# + 15*_N.sin(2*_N.pi*x))*dt
dN    = _N.empty(2*N)

ls    = -30   #  time of last spike

isis  = []

for tr in xrange(TR):
    dN[:] = 0
    for t in xrange(2*N):
        if _N.random.rand() < ps[t]*l2[t-ls-1]:
            dN[t] = 1
            ls = t
    dNs[tr] = _N.sum(dN)
    isis.extend(_U.toISI([_N.where(dN == 1)[0].tolist()])[0])
    alldat[:, 3*tr] = 0
    alldat[:, 3*tr+1] = ps[N:2*N]
    alldat[:, 3*tr+2] = dN[N:2*N]

fmt = "%.1e %.5f %d " * TR

_N.savetxt(resFN("xprbsdN.dat", dir=setname, create=True), alldat, fmt=fmt)

relspks = quickPSTH(alldat, TR, 3)

fig = _plt.figure(figsize=(6, 2*5))
fig.add_subplot(2, 1, 1)
h, bs = _N.histogram(relspks, bins=_N.linspace(0, 1000, 101))
bnsz   = 10
fpsth     = h / (TR * bnsz * dt)
_plt.plot(bs[0:-1], fpsth, color="black")
_plt.ylim(0, max(fpsth)*1.1)
#_plt.hist(relspks, bins=_N.linspace(0, 1000, 101), normed=True, color="black")
fig.add_subplot(2, 1, 2)
_plt.hist(isis, bins=_N.linspace(0, 100, 101), color="black")
_plt.savefig(resFN("psth_isi.png", dir=setname))
_plt.close()

fig = _plt.figure(figsize=(5, 2*4))
fig.add_subplot(2, 1, 1)
_plt.plot(ps/dt)
_plt.grid()
_plt.ylim(0, max(ps/dt)*1.1)
fig.add_subplot(2, 1, 2)
_plt.plot(range(1, 2*N+50+1), l2)
_plt.grid()
_plt.ylim(0, 1.2)
_plt.xlim(0, 70)
_plt.savefig(resFN("PSTHandHIST.png", dir=setname))
_plt.close()

_N.savetxt(resFN("generate-l2.dat", dir=setname, create=True), l2, fmt="%.3e")
