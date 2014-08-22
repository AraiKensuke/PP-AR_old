exf("kflib.py")

from kassdirs import resFN, datFN

import pickle as _pk
import warnings
import numpy.polynomial.polynomial as _Npp
import utilities as _U
import patsy

TR = 100
dt = 0.001
setname="nohistPSTH2"   #  params_XXX.py   Results/XXX/params.py

N     = 1000

#  x, prbs, spks    3 columns
alldat= _N.empty((N, TR*3))
us    = _N.empty(TR)
dNs   = _N.empty(TR)

x     = _N.linspace(0, 1*2, 2*N)
#  ABRUPT shoulder
l2    = _N.ones(2*N + 50)
#l2[0:20] = _N.linspace(0., 1, 20)**2
#  SMOOTH shoulder
a = 20.
b = 4.5
ms = _N.arange(0, 2*N+50)
#l2 = _N.exp((ms-a)/b) / (1 + _N.exp((ms-a)/b))
ps    = (45 + 25*_N.sin(2*_N.pi*x))*dt# +  5*_N.sin(2*3.1*_N.pi*x - 1))*dt
#ps    = 50*_N.ones(N)*dt# + 15*_N.sin(2*_N.pi*x) +  5*_N.sin(2*3.1*_N.pi*x - 1))*dt
#ps    = _N.ones(N)*50*dt# + 15*_N.sin(2*_N.pi*x))*dt
dN    = _N.empty(2*N)

ls    = -150   #  time of last spike

isis  = []
rpsth = []
for tr in xrange(TR):
    dN[:] = 0
    for t in xrange(2*N):
        if _N.random.rand() < ps[t]*l2[t-ls-1]:
            dN[t] = 1
            if t >= N:
                rpsth.append(t-N)
            ls = t
    dNs[tr] = _N.sum(dN)
    isis.extend(_U.toISI([_N.where(dN == 1)[0].tolist()])[0])
    alldat[:, 3*tr] = 0
    alldat[:, 3*tr+1] = ps[N:2*N]
    alldat[:, 3*tr+2] = dN[N:2*N]

fmt = "%.1e %.5f %d " * TR

_N.savetxt(resFN("xprbsdN.dat", dir=setname, create=True), alldat, fmt=fmt)

relspks = quickPSTH(alldat, TR, 3)


#  For fit onto naive PSTH
x = _N.linspace(0., dt*(N-1), N)
nbs1 = 8
B = patsy.bs(x, df=nbs1, include_intercept=True)
B  = B.T


#####  Fit of Poisson PSTH
bnsz   = 50
approxPSTH_x = _N.linspace(0, N, (N/bnsz)+1)
h, bs = _N.histogram(rpsth, bins=approxPSTH_x)
fs     = (h / (TR * bnsz * dt))
apsth = _N.repeat(fs, bnsz)    #    piecewise boxy approximate PSTH
aS     = _N.linalg.solve(_N.dot(B, B.T), _N.dot(B, _N.log(apsth)))
#####  

bnsz = 10
fig = _plt.figure(figsize=(6, 2*5))
fig.add_subplot(2, 1, 1)
_plt.plot(apsth, color="black")
_plt.plot(_N.exp(_N.dot(B.T, aS)), lw=2, color="blue")
_plt.ylim(0, max(apsth)*1.1)
_plt.grid()
#_plt.hist(relspks, bins=_N.linspace(0, 1000, 101), normed=True, color="black")
fig.add_subplot(2, 1, 2)
_plt.hist(isis, bins=_N.linspace(0, N/bnsz, N/bnsz+1), color="black")
_plt.savefig(resFN("psth_isi.png", dir=setname))
_plt.close()

fig = _plt.figure(figsize=(5, 3*4))
fig.add_subplot(3, 1, 1)
bnsz   = 50
_plt.hist(rpsth, bins=_N.linspace(0, N, (N/bnsz)+1))
fig.add_subplot(3, 1, 2)
_plt.plot(ps[N:2*N]/dt)
_plt.grid()
_plt.ylim(0, max(ps/dt)*1.1)
fig.add_subplot(3, 1, 3)
_plt.plot(range(1, 2*N+50+1), l2)
_plt.grid()
_plt.ylim(0, 1.2)
_plt.xlim(0, 70)
_plt.savefig(resFN("PSTHandHIST.png", dir=setname))
_plt.close()

_N.savetxt(resFN("generate-l2.dat", dir=setname, create=True), l2, fmt="%.3e")



    
