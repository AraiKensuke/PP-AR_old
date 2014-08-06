import scipy.optimize as _sco
from mcmcARpPlot import plotWFandSpks

exf("kflib.py")
from kassdirs import resFN, datFN

import pickle as _pk
import warnings
import numpy.polynomial.polynomial as _Npp
import utilities as _U

TR = 40
dt = 0.001
setname="mdOsc-1"   #  params_XXX.py   Results/XXX/params.py

_plt.ioff()
#  now using the same data, I might want to run it with different EM settings

#  start with
nzs   = _N.array([[1e-8, 7e-3]])   #  rhthm0  low high
#                  [1e-6, 8e-3]])
nRhythms = nzs.shape[0]
#  data length N + 1.  n=0...N
#  state vector dim k.  For AR(p).  k=p

# observations.  dim-l vector  (l=1 here)

rs = [0.95, ]
ths= [_N.pi*0.05]#, _N.pi*0.1]

lambda2 = _N.array([0.001, 0.05, 0.2, 0.5, 0.8, 0.9, 0.96, 0.99])
#lambda2 = _N.array([0.001, 0.1, 0.4, 0.95])


alfa  = _N.array([[rs[0]*(_N.cos(ths[0]) + 1j*_N.sin(ths[0])), 
                   rs[0]*(_N.cos(ths[0]) - 1j*_N.sin(ths[0]))]])
#                  [rs[1]*(_N.cos(ths[1]) + 1j*_N.sin(ths[1])), 
#                   rs[1]*(_N.cos(ths[1]) - 1j*_N.sin(ths[1]))]])

ARcoeff = _N.empty((nRhythms, 2))
ARcoeff[0]          = (-1*_Npp.polyfromroots(alfa[0])[::-1][1:]).real
#ARcoeff[1]          = (-1*_Npp.polyfromroots(alfa[1])[::-1][1:]).real
u0     = 3.9


#  AR weights.  In Kitagawa, a[0] is weight for most recent state
#  when used like dot(a, x), a needs to be stored in reverse direction
k      = 2

N     = 2000
trim  = 1000

#  x, prbs, spks    3 columns
nColumns = 3
alldat= _N.empty((N, TR*nColumns))
us    = _N.empty(TR)
spksPT = _N.empty(TR)
stNzs = _N.empty((TR, nRhythms))

lowQpc = 0.
lowQs  = []
for tr in xrange(TR):
    if _N.random.rand() < lowQpc:
        lowQs.append(tr)
        stNzs[tr] = nzs[:, 0]
    else:
        stNzs[tr] = nzs[:, 1]    #  high

isis  = []

for tr in xrange(TR):
    us[tr]    = u0

    if lambda2 == None:
        x, dN, prbs, fs = createDataPP(N+trim, ARcoeff, us[tr], stNzs[tr], p=1, trim=trim, absrefr=absrefr, nRhythms=nRhythms)
    else:
        x, dN, prbs, fs = createDataPPl2(N+trim, ARcoeff, us[tr], stNzs[tr], lambda2=lambda2, p=1, trim=trim, nRhythms=nRhythms)

    #print "%(sd).3f  %(dN)d" % {"sd" : _N.std(prbs), "dN" : int(_N.sum(dN))}
    spksPT[tr] = _N.sum(dN)
    alldat[:, nColumns*tr] = _N.sum(x, axis=0).T
    alldat[:, nColumns*tr+1] = prbs
    alldat[:, nColumns*tr+2] = dN
    isis.extend(_U.toISI([_N.where(dN == 1)[0].tolist()])[0])

savesetMT("bernoulli", setname, nRhythms)

arfs = ""
xlst = []
for nr in xrange(nRhythms):
    arfs += "%.1fHz " % (500*ths[nr]/_N.pi)
    xlst.append(x[nr])
sTitle = "AR2 freq %(fs)s    spk Hz %(spkf).1fHz   TR=%(tr)d   N=%(N)d" % {"spkf" : (_N.sum(spksPT) / (N*TR*0.001)), "tr" : TR, "N" : N, "fs" : arfs}

plotWFandSpks(N-1, dN, xlst, sTitle=sTitle, sFilename=resFN("generative", dir=setname))

fig = _plt.figure(figsize=(8, 4))
_plt.hist(isis, bins=range(100), color="black")
_plt.grid()
_plt.savefig(resFN("ISIhist", dir=setname))
_plt.close()

fig = _plt.figure(figsize=(13, 4))
_plt.plot(spksPT, marker=".", color="black", ms=8)
_plt.ylim(0, max(spksPT)*1.1)
_plt.grid()
_plt.suptitle("avg. Hz %.1f" % (_N.mean(spksPT) / (N*0.001)))
_plt.savefig(resFN("spksPT", dir=setname))
_plt.close()

if (lambda2 == None) and (absrefr > 0):
    lambda2 = _N.array([0.0001] * absrefr)
if lambda2 != None:
    _N.savetxt(resFN("lambda2.dat", dir=setname), lambda2, fmt="%.7f")

#  if we want to double bin size
lambda2db = 0.5*(lambda2[1::2] + lambda2[::2])
_N.savetxt(resFN("lambda2db.dat", dir=setname), lambda2db, fmt="%.7f")


